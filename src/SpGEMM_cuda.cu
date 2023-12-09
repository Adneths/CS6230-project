
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#include "SpGEMM_util.h"
#include "SpGEMM_cuda.h"

#ifdef PROFILE
#include <nvtx3/nvToolsExtCuda.h>
#endif




#include <inttypes.h>

namespace cuda {

__global__ void dacc_spgemm_preprocess(int* maxRow, int* rowPtr, int len) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int val = id < len-1 ? rowPtr[id+1] - rowPtr[id] : 0;
    val = blockReduceMax(val, 0);
    if (threadIdx.x == 0)
        atomicMax(maxRow, val);
}

// Sparse General Matrix Multiplication with Dense Accumulator
__global__ void spgemm_dacc(int rowsA, int colsA, int* rowPtrA, int* colIdxA, double* valsA,
                        int rowsB, int colsB, int* rowPtrB, int* colIdxB, double* valsB,
                        double* valsC) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    if (tx < colsB) {
        int as = rowPtrA[bx]; int ae = rowPtrA[bx+1];
        for (int i = as; i < ae; i++) {
            double valA = valsA[i]; int c = colIdxA[i];

            int bs = rowPtrB[c]; int be = rowPtrB[c+1];
            while (bs < be) {
                if (bs + tx < be) { // Prevent over reading of B
                    valsC[bx * colsB + colIdxB[bs + tx]] += valA * valsB[bs + tx];
                }
                bs += blockDim.x;
            }
            __syncthreads();
        }
    }
}

CSRMatrix<double>* dacc_spgemm(CSRMatrix<double>* A, CSRMatrix<double>* B) {
#ifdef PROFILE
    nvtxNameOsThread(pthread_self(), "MAIN_THREAD");
    nvtxRangePushA("CUDA_spgemm");
    nvtxRangePushA("CUDA_spgemm_cudamalloc");
#endif
    int *d_rowPtrA, *d_colIdxA, *d_rowPtrB, *d_colIdxB;
    double *d_valsA, *d_valsB, *d_valsC, *h_valsC;
    int* d_maxRow; int maxRow;

    size_t
    rowPtrA_size  = (A->rows+1) * sizeof(int),
    colIdxA_size = (A->nnz) * sizeof(int),
    valsA_size = (A->nnz) * sizeof(double),
    rowPtrB_size  = (B->rows+1) * sizeof(int),
    colIdxB_size = (B->nnz) * sizeof(int),
    valsB_size = (B->nnz) * sizeof(double),
    valsC_size = (A->rows * B->cols) * sizeof(double);

    cudaMalloc(&d_rowPtrA , rowPtrA_size );
    cudaMalloc(&d_colIdxA, colIdxA_size);
    cudaMalloc(&d_valsA, valsA_size);
    cudaMalloc(&d_rowPtrB , rowPtrB_size );
    cudaMalloc(&d_colIdxB, colIdxB_size);
    cudaMalloc(&d_valsB, valsB_size);
    cudaMalloc(&d_valsC, valsC_size);

    cudaMalloc(&d_maxRow, sizeof(int));

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
    cudaMemset(d_valsC, 0, valsC_size);
    cudaMemset(d_maxRow, 0, sizeof(int));
    
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_preprocess");
#endif

    dim3 threadsPerBlock(std::min(32*((B->rows+31)/32),1024));
    dim3 numBlocks((B->rows+threadsPerBlock.x-1)/threadsPerBlock.x);
    dacc_spgemm_preprocess<<<numBlocks,threadsPerBlock>>>(d_maxRow, d_rowPtrB, B->rows+1);
    cudaMemcpy(&maxRow, d_maxRow, sizeof(int), cudaMemcpyDeviceToHost);
    //maxRow = B->cols;
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_compute");
#endif
    threadsPerBlock = dim3(std::min(32*((maxRow+31)/32),1024));
    numBlocks = dim3(A->rows);
    spgemm_dacc<<<numBlocks,threadsPerBlock>>>(A->rows, A->cols, d_rowPtrA, d_colIdxA, d_valsA,
                                               B->rows, B->cols, d_rowPtrB, d_colIdxB, d_valsB,
                                               d_valsC);
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_malloc");
#endif
    h_valsC = (double*) malloc(valsC_size);
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_DtoH");
#endif
    cudaMemcpy(h_valsC, d_valsC, valsC_size, cudaMemcpyDeviceToHost);
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_reconstruct");
#endif
    CSRMatrix<double>* ret = new CSRMatrix<double>(A->rows, B->cols, h_valsC);
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
__global__ void spgemm_sacc(int rowsA, int colsA, int* rowPtrA, int* colIdxA, double* valsA,
                        int rowsB, int colsB, int* rowPtrB, int* colIdxB, double* valsB,
                        int* rowPtrC, int* colIdxC, double* valsC, uint64_t* maskC) {
	__shared__ int sBlock[CUMSUM_BLOCK_SIZE+1];
    int tx = threadIdx.x;
    int bx = blockIdx.x;
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
        //for (int j = 0; j < 0; j++) { // TODO
        int ind = 0;
        if (i < maskCols) {
#pragma unroll
            for (int k = 0; k < 64; k++) {
                if (mask & (1 << (63-k)) != 0) {
                    colIdxC[start + runningSum + sBlock[tx] + (ind++)] = i * 64 + k;
                    printf("%d:%d\n", bx, start + runningSum + sBlock[tx]);
                }
            }
        }
        //}
        //if (tx == 0) printf("%d:%d %d\n", bx, uint32_t(mask >> 32), uint32_t(mask & 0xffffffff));
        runningSum += sBlock[blockDim.x-1];
    }
    /*
    if (tx < colsB) {
        int as = rowPtrA[bx]; int ae = rowPtrA[bx+1];
        for (int i = as; i < ae; i++) {
            double valA = valsA[i]; int c = colIdxA[i];

            int bs = rowPtrB[c]; int be = rowPtrB[c+1];
            while (bs < be) {
                if (bs + tx < be) { // Prevent over reading of B
                    //valsC[bx * colsB + colIdxB[bs + tx]] += valA * valsB[bs + tx];
                    int colB = colIdxB[bs + tx];
                    int cs = rowPtrC[c]; int ce = rowPtrC[c+1]; int cm = (cs+ce)/2;
                    while (cs < ce) {
                        if (colIdxC[cm] == colB) {
                            valsC[cm] += valA * valsB[bs + tx];
                        }
                        else if (colIdxC[cm] < colB)
                            cs = cm+1;
                        else
                            ce = cm;
                    }
                }
                bs += blockDim.x;
            }
            __syncthreads();
        }
    }*/
}

CSRMatrix<double>* sacc_spgemm(CSRMatrix<double>* A, CSRMatrix<double>* B) {
#ifdef PROFILE
    nvtxNameOsThread(pthread_self(), "MAIN_THREAD");
    nvtxRangePushA("CUDA_spgemm");
    nvtxRangePushA("CUDA_spgemm_cudamalloc");
#endif
    int *d_rowPtrA, *d_colIdxA, *d_rowPtrB, *d_colIdxB, *d_rowPtrC;
    double *d_valsA, *d_valsB;
    int* d_maxRow; int maxRow;

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

    threadsPerBlock = dim3(std::min(32*((((B->cols+63)/64)+31)/32),1024)); // ceil( ceil(colsB/64) / 32)
    numBlocks = dim3(A->rows);
    //printf("(%d,%d,%d)\n", numBlocks.x, numBlocks.y, numBlocks.z);
    //printf("(%d,%d,%d)\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
    sacc_spgemm_structureC<<<numBlocks,threadsPerBlock>>>(d_rowPtrA, d_colIdxA, B->cols, d_maskB, d_maskC, d_rowPtrC);

    /*uint64_t* temp = (uint64_t*)malloc(maskC_size);
    cudaMemcpy(temp, d_maskB, maskC_size, cudaMemcpyDeviceToHost);
    printf("maskB: ");
    for (int i = 0; i < (A->rows) * ((B->cols+63)/64); i++)
        std::cout << temp[i] << ", ";
    printf("\n");
    cudaMemcpy(temp, d_maskC, maskC_size, cudaMemcpyDeviceToHost);
    printf("maskC: ");
    for (int i = 0; i < (A->rows) * ((B->cols+63)/64); i++)
        std::cout << temp[i] << ", ";
    printf("\n");
    int* temp1 = (int*)malloc(rowPtrC_size);
    cudaMemcpy(temp1, d_rowPtrC, rowPtrC_size, cudaMemcpyDeviceToHost);
    printf("rowSizeC: ");
    for (int i = 0; i < A->rows+1; i++)
        printf("%d, ", temp1[i]);
    printf("\n");
    free(temp);
    free(temp1);*/

    cumsumI(d_rowPtrC, d_rowPtrC, A->rows+1);
    int nnzC;
    cudaMemcpy(&nnzC, d_rowPtrC+A->rows, sizeof(int), cudaMemcpyDeviceToHost);
    //printf("NNZ=%d\n", nnzC);
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
    spgemm_sacc<<<numBlocks,threadsPerBlock>>>(A->rows, A->cols, d_rowPtrA, d_colIdxA, d_valsA,
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
