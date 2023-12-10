
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#include "SpGEMM_util.h"
#include "SpGEMM_cuda.h"

#ifdef PROFILE
#include <nvtx3/nvToolsExtCuda.h>
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
#ifdef PROFILE
    nvtxNameOsThread(pthread_self(), "MAIN_THREAD");
    nvtxRangePushA("CUDA_spgemm");
    nvtxRangePushA("CUDA_spgemm_cudamalloc");
#endif
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

#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_HtoD");
#endif
    cudaSetDevice(0);
    cudaMemset(d_maxRow, 0, sizeof(int));
    for (int d = 0; d < deviceCount; d++) {
        cudaSetDevice(d);
        cudaMemcpy(d_rowPtrA[d], A->rowPtr , rowPtrA_size , cudaMemcpyHostToDevice);
        cudaMemcpy(d_colIdxA[d], A->dataCol, colIdxA_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_valsA[d], A->dataVal, valsA_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_rowPtrB[d], B->rowPtr , rowPtrB_size , cudaMemcpyHostToDevice);
        cudaMemcpy(d_colIdxB[d], B->dataCol, colIdxB_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_valsB[d], B->dataVal, valsB_size, cudaMemcpyHostToDevice);
        cudaMemset(d_valsC[d], 0, valsC_sizes[d]);
    }
    
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_preprocess");
#endif

    cudaSetDevice(0);
    dim3 threadsPerBlock(std::min(32*((B->rows+31)/32),1024));
    dim3 numBlocks((B->rows+threadsPerBlock.x-1)/threadsPerBlock.x);
    dacc_spgemm_preprocess<<<numBlocks,threadsPerBlock>>>(d_maxRow, d_rowPtrB[0], B->rows+1);
    cudaMemcpy(&maxRow, d_maxRow, sizeof(int), cudaMemcpyDeviceToHost);

#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_compute");
#endif
    threadsPerBlock = dim3(std::min(32*((maxRow+31)/32),1024));
    for (int d = 0; d < deviceCount; d++) {
        cudaSetDevice(d);
        numBlocks = dim3(partition[d+1] - partition[d]);
        spgemm_dacc<<<numBlocks,threadsPerBlock>>>(partition[d],
                                                     A->rows, A->cols, d_rowPtrA[d], d_colIdxA[d], d_valsA[d],
                                                     B->rows, B->cols, d_rowPtrB[d], d_colIdxB[d], d_valsB[d],
                                                     d_valsC[d]);
    }
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_malloc");
#endif
    h_valsC = (double*) malloc((A->rows * B->cols) * sizeof(double));
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_DtoH");
#endif
    int offset = 0;
    for (int d = 0; d < deviceCount; d++) {
        cudaSetDevice(d);
        cudaMemcpy(h_valsC+offset, d_valsC[d], valsC_sizes[d], cudaMemcpyDeviceToHost);
        offset += valsC_sizes[d]/sizeof(double);
    }
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_reconstruct");
#endif
    CSRMatrix<double>* ret = new CSRMatrix<double>(A->rows, B->cols, h_valsC);
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_cudafree");
#endif

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
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePop();
#endif

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
__global__ void spgemm_sacc_val(int rowsA, int colsA, int* rowPtrA, int* colIdxA, double* valsA,
                        int rowsB, int colsB, int* rowPtrB, int* colIdxB, double* valsB,
                        int* rowPtrC, int* colIdxC, double* valsC, uint64_t* maskC) {
    int tx = threadIdx.x; int bx = blockIdx.x;
    
    if (tx < colsB) {
        for (int i = rowPtrA[bx]; i < rowPtrA[bx+1]; i++) {
            double valA = valsA[i]; int c = colIdxA[i];

            int bs = rowPtrB[c]; int be = rowPtrB[c+1];
            int cs = rowPtrC[bx]; int ce = rowPtrC[bx+1];
            while (bs < be) {
                if (bs + tx < be) { // Prevent over reading of B
                    int colB = colIdxB[bs + tx];
                    int ss = cs; int se = ce;
                    while (ss < se) {
                        int cm = (ss+se)/2;
                        if (colIdxC[cm] == colB) {
                            atomicAdd(valsC+cm, valA * valsB[bs + tx]);
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
    if (deviceCount > 1) deviceCount = 1;
    printf("%d GPU%s\n", deviceCount, deviceCount > 1 ? "s": "");
    if (deviceCount == 0)
        return nullptr;
#ifdef PROFILE
    nvtxNameOsThread(pthread_self(), "MAIN_THREAD");
    nvtxRangePushA("CUDA_spgemm");
    nvtxRangePushA("CUDA_spgemm_cudamalloc");
#endif
    int *d_rowPtrA, *d_colIdxA, *d_rowPtrB, *d_colIdxB, *d_rowPtrC;
    double *d_valsA, *d_valsB;
    int* d_maxRow; int maxRow;
    int nnzC;

    size_t
    rowPtrA_size  = (A->rows+1) * sizeof(int),
    colIdxA_size = (A->nnz) * sizeof(int),
    valsA_size = (A->nnz) * sizeof(double),
    rowPtrB_size  = (B->rows+1) * sizeof(int),
    colIdxB_size = (B->nnz) * sizeof(int),
    valsB_size = (B->nnz) * sizeof(double),
    rowPtrC_size  = (A->rows+1) * sizeof(int);

    cudaMalloc(&d_rowPtrA , rowPtrA_size );
    cudaMalloc(&d_colIdxA, colIdxA_size);
    cudaMalloc(&d_valsA, valsA_size);
    cudaMalloc(&d_rowPtrB , rowPtrB_size );
    cudaMalloc(&d_colIdxB, colIdxB_size);
    cudaMalloc(&d_valsB, valsB_size);
    cudaMalloc(&d_rowPtrC , rowPtrC_size );
    cudaMalloc(&d_maxRow, sizeof(int));

    size_t
    maskB_size = (B->rows) * ((B->cols+63)/64) * sizeof(uint64_t),
    maskC_size = (A->rows) * ((B->cols+63)/64) * sizeof(uint64_t);
    uint64_t *d_maskB, *d_maskC;
    cudaMalloc(&d_maskB, maskB_size);
    cudaMalloc(&d_maskC, maskC_size);

#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_HtoD");
#endif
    cudaMemcpy(d_rowPtrA , A->rowPtr , rowPtrA_size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdxA, A->dataCol, colIdxA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_valsA, A->dataVal, valsA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtrB , B->rowPtr , rowPtrB_size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdxB, B->dataCol, colIdxB_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_valsB, B->dataVal, valsB_size, cudaMemcpyHostToDevice);
    cudaMemset(d_maxRow, 0, sizeof(int));
    cudaMemset(d_maskB, 0, maskB_size);
    cudaMemset(d_rowPtrC, 0, sizeof(int)); // Set first int to 0
    
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_preprocess");
#endif

    dim3 threadsPerBlock(std::min(32*((B->rows+31)/32),1024));
    dim3 numBlocks((B->rows+threadsPerBlock.x-1)/threadsPerBlock.x);
    sacc_spgemm_preprocess<<<numBlocks,threadsPerBlock>>>(B->rows, B->cols, d_rowPtrB, d_colIdxB, d_maskB, d_maxRow);
    cudaMemcpy(&maxRow, d_maxRow, sizeof(int), cudaMemcpyDeviceToHost);

    threadsPerBlock = dim3(std::min(32*((((B->cols+63)/64)+31)/32),1024)); // equlvalent to ceil( ceil(colsB/64) / 32)
    numBlocks = dim3(A->rows);
    sacc_spgemm_structureC<<<numBlocks,threadsPerBlock>>>(d_rowPtrA, d_colIdxA, B->cols, d_maskB, d_maskC, d_rowPtrC);

    cumsumI(d_rowPtrC, d_rowPtrC, A->rows+1);
    cudaMemcpy(&nnzC, d_rowPtrC+A->rows, sizeof(int), cudaMemcpyDeviceToHost);
#ifdef PROFILE
    nvtxRangePushA("CUDA_spgemm_preprocess_cudamalloc");
#endif
    size_t
    colIdxC_size = nnzC * sizeof(int),
    valsC_size = nnzC * sizeof(double);
    int* d_colIdxC;
    double* d_valsC;
    cudaMalloc(&d_colIdxC, colIdxC_size);
    cudaMalloc(&d_valsC, valsC_size);
#ifdef PROFILE
    nvtxRangePop();
#endif

#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_compute");
#endif
    threadsPerBlock = dim3(std::min(32*((maxRow+31)/32),1024));
    numBlocks = dim3(A->rows);
    spgemm_sacc_col<<<numBlocks,threadsPerBlock>>>(A->rows, A->cols, d_rowPtrA, d_colIdxA, d_valsA,
                                               B->rows, B->cols, d_rowPtrB, d_colIdxB, d_valsB,
                                               d_rowPtrC, d_colIdxC, d_valsC, d_maskC);
    spgemm_sacc_val<<<numBlocks,threadsPerBlock>>>(A->rows, A->cols, d_rowPtrA, d_colIdxA, d_valsA,
                                               B->rows, B->cols, d_rowPtrB, d_colIdxB, d_valsB,
                                               d_rowPtrC, d_colIdxC, d_valsC, d_maskC);
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_malloc");
#endif
    int *h_rowPtrC, *h_colIdxC;
    double* h_valsC;
    h_rowPtrC = (int*) malloc(rowPtrC_size);
    h_colIdxC = (int*) malloc(colIdxC_size);
    h_valsC = (double*) malloc(valsC_size);
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_DtoH");
#endif
    cudaMemcpy(h_rowPtrC, d_rowPtrC, rowPtrC_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_colIdxC, d_colIdxC, colIdxC_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_valsC, d_valsC, valsC_size, cudaMemcpyDeviceToHost);
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_reconstruct");
#endif
    CSRMatrix<double>* ret = new CSRMatrix<double>(A->rows, B->cols, h_rowPtrC, h_colIdxC, h_valsC, nnzC);
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_cudafree");
#endif

    cudaFree(d_rowPtrA );
    cudaFree(d_colIdxA);
    cudaFree(d_valsA);
    cudaFree(d_rowPtrB );
    cudaFree(d_colIdxB);
    cudaFree(d_valsB);
    cudaFree(d_valsC);
    cudaFree(d_maxRow);
    cudaFree(d_maskB);
    cudaFree(d_maskC);
    cudaFree(d_colIdxC);
    cudaFree(d_valsC);
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePop();
#endif

    return ret;
}

}
