
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#include "SpGEMM_util.h"
#include "SpGEMM_cuda.h"

#define RED     0
#define YELLOW  1
#define GREEN   2
#define AQUA    3
#define BLUE    4
#define PURPLE  5
#define BLACK   6
#define GRAY    7
#define WHITE   8

#ifdef PROFILE
#include <nvtx3/nvToolsExtCuda.h>
const uint32_t colors[] = { 0xffff0000, 0xffffff00, 0xff00ff00, 0xff00ffff, 0xff0000ff, 0xffff00ff, 0xff000000, 0xff808080, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define POP_RANGE() nvtxRangePop();
#define NAME_THREAD(name) nvtxNameOsThread(pthread_self(), name);
#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#else
#define POP_RANGE()
#define NAME_THREAD(name)
#define PUSH_RANGE(name,cid)
#endif

namespace cuda {

__global__ void dacc_spgemm_preprocess(int* maxRow, int* rowPtr, int len) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int val = id < len-1 ? rowPtr[id+1] - rowPtr[id] : 0;
    val = blockReduceMax(val, 0);
    if (threadIdx.x == 0)
        atomicMax(maxRow, val);
}

// Sparse General Matrix Multiplication with Dense Accumulator
__global__ void spgemm_dacc(int partOffset, int rowsA, int colsA, int* rowPtrA, int* colIdxA, double* valsA,
                        int rowsB, int colsB, int* rowPtrB, int* colIdxB, double* valsB,
                        double* valsC) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    if (tx < colsB) {
        for (int i = rowPtrA[partOffset+bx]; i < rowPtrA[partOffset+bx+1]; i++) {
            double valA = valsA[i]; int c = colIdxA[i];

            int bs = rowPtrB[c]; int be = rowPtrB[c+1];
            while (bs < be) {
                if (bs + tx < be) { // Prevent over reading of B
                    atomicAdd(&valsC[bx * colsB + colIdxB[bs + tx]], valA * valsB[bs + tx]);
                }
                bs += blockDim.x;
            }
        }
    }
}

int check(int rows, int* rowPtr, int partSize) {
    int prev = 0; int n = 0;
    for (int i = 1; i < rows+1; i++) {
        if (rowPtr[i] > partSize + prev) {
            prev = rowPtr[i-1];
            n++;
        }
    }
    return n+1;
}
void partitionRows(int rows, int* rowPtr, int* partitionInd, int N) {
    if (N < 2) {
        partitionInd[0] = rows;
        return;
    }
    int s = 0; int e = rowPtr[rows]; int best = 0;
    while (s < e) {
        int m = (s+e)/2;
        int part = check(rows, rowPtr, m);
        if (part == N) {
            best = m;
            e = m;
        }
        else if (part < N)
            e = m;
        else
            s = m+1;
    }

    int prev = 0; int n = 0;
    for (int i = 1; i < rows+1; i++) {
        if (rowPtr[i] > best + prev) {
            prev = rowPtr[i-1];
            partitionInd[n++] = i-1;
        }
    }
    partitionInd[n] = rows;
}

CSRMatrix<double>* dacc_spgemm(CSRMatrix<double>* A, CSRMatrix<double>* B) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("%d GPU%s\n", deviceCount, deviceCount > 1 ? "s": "");
    if (deviceCount == 0)
        return nullptr;
//#ifdef PROFILE
    NAME_THREAD("MAIN_THREAD");
    PUSH_RANGE("CUDA_spgemm", BLACK);
    PUSH_RANGE("CUDA_spgemm_cudamalloc", RED);
//#endif
    int *d_rowPtrA[MAX_GPU], *d_colIdxA[MAX_GPU], *d_rowPtrB[MAX_GPU], *d_colIdxB[MAX_GPU];
    double *d_valsA[MAX_GPU], *d_valsB[MAX_GPU], *d_valsC[MAX_GPU], *h_valsC;
    int* d_maxRow; int maxRow;

    int partition[MAX_GPU+1]; partition[0] = 0;
    partitionRows(A->rows, A->rowPtr, partition+1, deviceCount);

    size_t
    rowPtrA_size  = (A->rows+1) * sizeof(int),
    colIdxA_size = (A->nnz) * sizeof(int),
    valsA_size = (A->nnz) * sizeof(double),
    rowPtrB_size  = (B->rows+1) * sizeof(int),
    colIdxB_size = (B->nnz) * sizeof(int),
    valsB_size = (B->nnz) * sizeof(double);
    size_t valsC_sizes[MAX_GPU];

    cudaSetDevice(0);
    cudaMalloc(&d_maxRow, sizeof(int));
    for (int d = 0; d < deviceCount; d++) {
        cudaSetDevice(d);
        cudaMalloc(&d_rowPtrA[d], rowPtrA_size);
        cudaMalloc(&d_colIdxA[d], colIdxA_size);
        cudaMalloc(&d_valsA[d], valsA_size);
        cudaMalloc(&d_rowPtrB[d], rowPtrB_size);
        cudaMalloc(&d_colIdxB[d], colIdxB_size);
        cudaMalloc(&d_valsB[d], valsB_size);

        valsC_sizes[d] = ((partition[d+1]-partition[d]) * B->cols) * sizeof(double);
        cudaMalloc(&d_valsC[d], valsC_sizes[d]);
    }

//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CUDA_spgemm_HtoD", AQUA);
//#endif
    cudaSetDevice(0);
    cudaMemsetAsync(d_maxRow, 0, sizeof(int));
    for (int d = 0; d < deviceCount; d++) {
        cudaSetDevice(d);
        cudaMemcpyAsync(d_rowPtrA[d], A->rowPtr , rowPtrA_size , cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_colIdxA[d], A->dataCol, colIdxA_size, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_valsA[d], A->dataVal, valsA_size, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_rowPtrB[d], B->rowPtr , rowPtrB_size , cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_colIdxB[d], B->dataCol, colIdxB_size, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_valsB[d], B->dataVal, valsB_size, cudaMemcpyHostToDevice);
        cudaMemsetAsync(d_valsC[d], 0, valsC_sizes[d]);
    }
    syncDevice(deviceCount);

//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CUDA_spgemm_preprocess", YELLOW);
//#endif

    cudaSetDevice(0);
    dim3 threadsPerBlock(std::min(32*((B->rows+31)/32),1024));
    dim3 numBlocks((B->rows+threadsPerBlock.x-1)/threadsPerBlock.x);
    dacc_spgemm_preprocess<<<numBlocks,threadsPerBlock>>>(d_maxRow, d_rowPtrB[0], B->rows+1);
    cudaMemcpyAsync(&maxRow, d_maxRow, sizeof(int), cudaMemcpyDeviceToHost);
    syncDevice(deviceCount);

//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CUDA_spgemm_compute", GREEN);
//#endif
    threadsPerBlock = dim3(std::min(32*((maxRow+31)/32),1024));
    for (int d = 0; d < deviceCount; d++) {
        cudaSetDevice(d);
        numBlocks = dim3(partition[d+1] - partition[d]);
        spgemm_dacc<<<numBlocks,threadsPerBlock>>>(partition[d],
                                                     A->rows, A->cols, d_rowPtrA[d], d_colIdxA[d], d_valsA[d],
                                                     B->rows, B->cols, d_rowPtrB[d], d_colIdxB[d], d_valsB[d],
                                                     d_valsC[d]);
    }
    syncDevice(deviceCount);
//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CUDA_spgemm_malloc", RED);
//#endif
    h_valsC = (double*) malloc((A->rows * B->cols) * sizeof(double));
//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CUDA_spgemm_DtoH", AQUA);
//#endif
    int offset = 0;
    for (int d = 0; d < deviceCount; d++) {
        cudaSetDevice(d);
        cudaMemcpyAsync(h_valsC+offset, d_valsC[d], valsC_sizes[d], cudaMemcpyDeviceToHost);
        offset += valsC_sizes[d]/sizeof(double);
    }
    syncDevice(deviceCount);
//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CUDA_spgemm_reconstruct", GRAY);
//#endif
    CSRMatrix<double>* ret = new CSRMatrix<double>(A->rows, B->cols, h_valsC);
//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CUDA_spgemm_cudafree", BLUE);
//#endif

    cudaSetDevice(0);
    cudaFree(d_maxRow);
    for (int d = 0; d < deviceCount; d++) {
        cudaSetDevice(d);
        cudaFree(d_rowPtrA[d]);
        cudaFree(d_colIdxA[d]);
        cudaFree(d_valsA[d]);
        cudaFree(d_rowPtrB[d]);
        cudaFree(d_colIdxB[d]);
        cudaFree(d_valsB[d]);
        cudaFree(d_valsC[d]);
    }
    syncDevice(deviceCount);
//#ifdef PROFILE
    POP_RANGE();
    POP_RANGE();
//#endif
    cudaSetDevice(0);

    return ret;
}

__global__ void sacc_spgemm_preprocess(int rows, int cols, int* rowPtr, int* colIdx, uint64_t* mask, int* maxRow) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    // Find max number of column between every row
    int val = id < rows ? rowPtr[id+1] - rowPtr[id] : 0;
    val = blockReduceMax(val, 0);
    if (threadIdx.x == 0)
        atomicMax(maxRow, val);

    // Generate matrix mask
    int maskCols = (cols+63)/64;
    if (id < rows) {
        int colAligned = 0; uint64_t m = 0;
        for (int i = rowPtr[id]; i < rowPtr[id+1]; i++) {
            int col = colIdx[i];
            if (col >= colAligned+64) {
                mask[id * maskCols + colAligned/64] = m;
                colAligned = 64*(col / 64);
                m = 0;
            }
            m |= 1ull << (63 - (col - colAligned));
        }
        mask[id * maskCols + colAligned/64] = m;
    }
}
__global__ void sacc_spgemm_structureC(int* rowPtrA, int* colIdxA,
                                       int colsB, uint64_t* maskB,
                                       uint64_t* maskC, int* rowPtrC) {
    int tx = threadIdx.x; int bx = blockIdx.x;
    int maskCols = (colsB+63)/64;
    int countC = 0;
    for (int j = 0; j < maskCols; j += blockDim.x) {
        if (j + tx < maskCols) {
            uint64_t mask = 0;
            for (int i = rowPtrA[bx]; i < rowPtrA[bx+1]; i++) {
                mask |= maskB[colIdxA[i] * maskCols + j + tx];
            }
            countC += __popcll(mask);
            maskC[bx * maskCols + j + tx] = mask;
        }
    }
    countC = blockReduceAdd(countC, 0);
    if (tx == 0)
        rowPtrC[bx+1] = countC;
}
__global__ void spgemm_sacc_col(int rowsA, int colsA, int* rowPtrA, int* colIdxA, double* valsA,
                        int rowsB, int colsB, int* rowPtrB, int* colIdxB, double* valsB,
                        int* rowPtrC, int* colIdxC, double* valsC, uint64_t* maskC) {
	__shared__ int sBlock[CUMSUM_BLOCK_SIZE+1];
    int tx = threadIdx.x; int bx = blockIdx.x;
    if (tx == 0) sBlock[0] = 0;
    int maskCols = (colsB+63)/64;

    int runningSum = 0;
    int start = rowPtrC[bx];
    for (int n = 0; n < (maskCols+blockDim.x-1)/blockDim.x; n++) {
        int i = n * blockDim.x + tx;
        uint64_t mask = maskC[bx * maskCols + i];
        sBlock[tx+1] = i < maskCols ? __popcll(mask) : 0; __syncthreads();
        for (int stride = 1; stride <= CUMSUM_BLOCK_SIZE; stride *= 2) {
            int index = (tx + 1) * stride * 2 - 1;
            if (index < CUMSUM_BLOCK_SIZE)
                sBlock[index] += sBlock[index - stride];
            __syncthreads();
        }
        for (int stride = CUMSUM_BLOCK_SIZE / 2; stride > 0; stride /= 2) {
            int index = (tx + 1) * stride * 2 - 1;
            if (index + stride < CUMSUM_BLOCK_SIZE)
                sBlock[index + stride] += sBlock[index];
            __syncthreads();
        }
        int ind = 0;
        if (i < maskCols) {
#pragma unroll
            for (int k = 0; k < 64; k++) {
                if ((mask & (1ull << (63-k))) != 0ull) {
                    colIdxC[start + runningSum + sBlock[tx] + (ind++)] = i * 64 + k;
                }
            }
        }
        runningSum += sBlock[blockDim.x-1];
    }
}
__global__ void spgemm_sacc_val(int partOffset, int rowsA, int colsA, int* rowPtrA, int* colIdxA, double* valsA,
                        int rowsB, int colsB, int* rowPtrB, int* colIdxB, double* valsB,
                        int* rowPtrC, int* colIdxC, double* valsC, uint64_t* maskC) {
    int tx = threadIdx.x; int bx = blockIdx.x;
    
    int baseC = rowPtrA[partOffset];
    if (tx < colsB) {
        for (int i = rowPtrA[partOffset+bx]; i < rowPtrA[partOffset+bx+1]; i++) {
            double valA = valsA[i]; int c = colIdxA[i];

            int bs = rowPtrB[c]; int be = rowPtrB[c+1];
            int cs = rowPtrC[partOffset+bx]; int ce = rowPtrC[partOffset+bx+1];
            while (bs < be) {
                if (bs + tx < be) { // Prevent over reading of B
                    int colB = colIdxB[bs + tx];
                    int ss = cs; int se = ce;
                    while (ss < se) {
                        int cm = (ss+se)/2;
                        if (colIdxC[cm] == colB) {
                            atomicAdd(valsC+cm-baseC, valA * valsB[bs + tx]);
                            break;
                        }
                        else if (colIdxC[cm] < colB)
                            ss = cm+1;
                        else
                            se = cm;
                    }
                }
                bs += blockDim.x;
            }
            __syncthreads();
        }
    }
}

CSRMatrix<double>* sacc_spgemm(CSRMatrix<double>* A, CSRMatrix<double>* B) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("%d GPU%s\n", deviceCount, deviceCount > 1 ? "s": "");
    if (deviceCount == 0)
        return nullptr;
//#ifdef PROFILE
    NAME_THREAD("MAIN_THREAD");
    PUSH_RANGE("CUDA_spgemm", BLACK);
    PUSH_RANGE("CUDA_spgemm_cudamalloc", RED);
//#endif
    int *d_rowPtrA[MAX_GPU], *d_colIdxA[MAX_GPU], *d_rowPtrB[MAX_GPU], *d_colIdxB[MAX_GPU], *d_rowPtrC[MAX_GPU];
    double *d_valsA[MAX_GPU], *d_valsB[MAX_GPU];
    uint64_t *d_maskB[MAX_GPU], *d_maskC[MAX_GPU];
    int* d_maxRow; int maxRow;
    
    int partition[MAX_GPU+1]; partition[0] = 0;
    partitionRows(A->rows, A->rowPtr, partition+1, deviceCount);

    size_t
    rowPtrA_size  = (A->rows+1) * sizeof(int),
    colIdxA_size = (A->nnz) * sizeof(int),
    valsA_size = (A->nnz) * sizeof(double),
    rowPtrB_size  = (B->rows+1) * sizeof(int),
    colIdxB_size = (B->nnz) * sizeof(int),
    valsB_size = (B->nnz) * sizeof(double),
    rowPtrC_size  = (A->rows+1) * sizeof(int);
    
    size_t
    maskB_size = (B->rows) * ((B->cols+63)/64) * sizeof(uint64_t),
    maskC_size = (A->rows) * ((B->cols+63)/64) * sizeof(uint64_t);

    cudaSetDevice(0);
    cudaMalloc(&d_maxRow, sizeof(int));
    for (int d = 0; d < deviceCount; d++) {
        cudaSetDevice(d);
        cudaMalloc(&d_rowPtrA[d], rowPtrA_size);
        cudaMalloc(&d_colIdxA[d], colIdxA_size);
        cudaMalloc(&d_valsA[d], valsA_size);
        cudaMalloc(&d_rowPtrB[d], rowPtrB_size);
        cudaMalloc(&d_colIdxB[d], colIdxB_size);
        cudaMalloc(&d_valsB[d], valsB_size);
        cudaMalloc(&d_rowPtrC[d], rowPtrC_size);
        cudaMalloc(&d_maskB[d], maskB_size);
        cudaMalloc(&d_maskC[d], maskC_size);
    }

//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CUDA_spgemm_malloc", RED);
//#endif
    int *h_rowPtrC;
    h_rowPtrC = (int*) malloc(rowPtrC_size);

//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CUDA_spgemm_HtoD", AQUA);
//#endif
    cudaSetDevice(0);
    cudaMemsetAsync(d_maxRow, 0, sizeof(int));
    for (int d = 0; d < deviceCount; d++) {
        cudaSetDevice(d);
        cudaMemcpyAsync(d_rowPtrA[d], A->rowPtr , rowPtrA_size , cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_colIdxA[d], A->dataCol, colIdxA_size, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_valsA[d], A->dataVal, valsA_size, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_rowPtrB[d], B->rowPtr , rowPtrB_size , cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_colIdxB[d], B->dataCol, colIdxB_size, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_valsB[d], B->dataVal, valsB_size, cudaMemcpyHostToDevice);
        cudaMemsetAsync(d_maskB[d], 0, maskB_size);
        cudaMemsetAsync(d_rowPtrC[d], 0, sizeof(int)); // Set first int to 0
    }
    syncDevice(deviceCount);
    
//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CUDA_spgemm_preprocess", YELLOW);
//#endif

    dim3 threadsPerBlock(std::min(32*((B->rows+31)/32),1024));
    dim3 numBlocks((B->rows+threadsPerBlock.x-1)/threadsPerBlock.x);
    for (int d = 0; d < deviceCount; d++) {
        cudaSetDevice(d);
        sacc_spgemm_preprocess<<<numBlocks,threadsPerBlock>>>(
            B->rows, B->cols, d_rowPtrB[d], d_colIdxB[d], d_maskB[d], d==0?d_maxRow:nullptr);
    }
    cudaSetDevice(0);
    cudaMemcpyAsync(&maxRow, d_maxRow, sizeof(int), cudaMemcpyDeviceToHost);
    syncDevice(deviceCount);

    threadsPerBlock = dim3(std::min(32*((((B->cols+63)/64)+31)/32),1024)); // equlvalent to ceil( ceil(colsB/64) / 32)
    numBlocks = dim3(A->rows);
    for (int d = 0; d < deviceCount; d++) {
        cudaSetDevice(d);
        sacc_spgemm_structureC<<<numBlocks,threadsPerBlock>>>(
            d_rowPtrA[d], d_colIdxA[d], B->cols, d_maskB[d], d_maskC[d], d_rowPtrC[d]);
        cumsumI(d_rowPtrC[d], d_rowPtrC[d], A->rows+1);
    }

    cudaSetDevice(0);
    cudaMemcpyAsync(h_rowPtrC, d_rowPtrC[0], rowPtrC_size, cudaMemcpyDeviceToHost);
    int nnzC = h_rowPtrC[A->rows];
    syncDevice(deviceCount);
    //cudaMemcpyAsync(&nnzC, d_rowPtrC+A->rows, sizeof(int), cudaMemcpyDeviceToHost);
//#ifdef PROFILE
    PUSH_RANGE("CUDA_spgemm_preprocess_cudamalloc", RED);
//#endif
    size_t
    colIdxC_size[MAX_GPU],// = nnzC * sizeof(int),
    valsC_size[MAX_GPU];// = nnzC * sizeof(double);
    int* d_colIdxC[MAX_GPU];
    double* d_valsC[MAX_GPU];
    for (int d = 0; d < deviceCount; d++) {
        cudaSetDevice(d);
        int pnnz = (h_rowPtrC[partition[d+1]] - h_rowPtrC[partition[d]]);
        colIdxC_size[d] = pnnz * sizeof(int);
        valsC_size[d] = pnnz * sizeof(double);
        cudaMalloc(&d_colIdxC[d], colIdxC_size[d]);
        cudaMalloc(&d_valsC[d], valsC_size[d]);
    }
//#ifdef PROFILE
    POP_RANGE();
//#endif

//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CUDA_spgemm_compute", GREEN);
//#endif
    threadsPerBlock = dim3(std::min(32*((maxRow+31)/32),1024));
    for (int d = 0; d < deviceCount; d++) {
        cudaSetDevice(d);
        /*spgemm_sacc<<<numBlocks,threadsPerBlock>>>(partition[d],
                                                   A->rows, A->cols, d_rowPtrA, d_colIdxA, d_valsA,
                                                   B->rows, B->cols, d_rowPtrB, d_colIdxB, d_valsB,
                                                   d_rowPtrC, d_colIdxC, d_valsC, d_maskC);*/
        numBlocks = dim3(A->rows);
        spgemm_sacc_col<<<numBlocks,threadsPerBlock>>>(A->rows, A->cols, d_rowPtrA[d], d_colIdxA[d], d_valsA[d],
                                                       B->rows, B->cols, d_rowPtrB[d], d_colIdxB[d], d_valsB[d],
                                                       d_rowPtrC[d], d_colIdxC[d], d_valsC[d], d_maskC[d]);
        numBlocks = dim3(partition[d+1] - partition[d]);
        spgemm_sacc_val<<<numBlocks,threadsPerBlock>>>(partition[d],
                                                       A->rows, A->cols, d_rowPtrA[d], d_colIdxA[d], d_valsA[d],
                                                       B->rows, B->cols, d_rowPtrB[d], d_colIdxB[d], d_valsB[d],
                                                       d_rowPtrC[d], d_colIdxC[d], d_valsC[d], d_maskC[d]);
    }
    syncDevice(deviceCount);
//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CUDA_spgemm_malloc", RED);
//#endif
    int *h_colIdxC;
    double* h_valsC;
    h_colIdxC = (int*)malloc(nnzC * sizeof(int));
    h_valsC = (double*)malloc(nnzC * sizeof(double));
//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CUDA_spgemm_DtoH", AQUA);
//#endif
    int offset = 0;
    for (int d = 0; d < deviceCount; d++) {
        cudaSetDevice(d);
        cudaMemcpyAsync(h_colIdxC+offset, d_colIdxC[d], colIdxC_size[d], cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(h_valsC+offset, d_valsC[d], valsC_size[d], cudaMemcpyDeviceToHost);
        offset += colIdxC_size[d]/sizeof(int);
    }
    syncDevice(deviceCount);
//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CUDA_spgemm_reconstruct", GRAY);
//#endif
    CSRMatrix<double>* ret = new CSRMatrix<double>(A->rows, B->cols, h_rowPtrC, h_colIdxC, h_valsC, nnzC);
//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CUDA_spgemm_cudafree", BLUE);
//#endif

    cudaSetDevice(0);
    cudaFree(d_maxRow);
    for (int d = 0; d < deviceCount; d++) {
        cudaSetDevice(d);
        cudaFree(d_rowPtrA[d]);
        cudaFree(d_colIdxA[d]);
        cudaFree(d_valsA[d]);
        cudaFree(d_rowPtrB[d]);
        cudaFree(d_colIdxB[d]);
        cudaFree(d_valsB[d]);
        cudaFree(d_valsC[d]);
        cudaFree(d_maskB[d]);
        cudaFree(d_maskC[d]);
        cudaFree(d_rowPtrC[d]);
        cudaFree(d_colIdxC[d]);
        cudaFree(d_valsC[d]);
    }
    syncDevice(deviceCount);
//#ifdef PROFILE
    POP_RANGE();
    POP_RANGE();
//#endif
    cudaSetDevice(0);

    return ret;
}

}
