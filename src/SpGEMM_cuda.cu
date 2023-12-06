
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#include "SpGEMM_cuda.h"

#ifdef PROFILE
#include <nvtx3/nvToolsExtCuda.h>
#endif

namespace cuda {

// Sparse General Matrix Multiplication with Dense Accumulator
__global__ void spgemm_dacc(int rowsA, int colsA, int* rowPtrA, int* dataColA, double* dataValA,
                        int rowsB, int colsB, int* rowPtrB, int* dataColB, double* dataValB,
                        double* dataValC) {
    int tx = threadIdx.x; int bx = blockIdx.x;
    if (tx < colsB) {
        int as = rowPtrA[bx]; int ae = rowPtrA[bx+1];
        for (int i = as; i < ae; i++) {
            double valA = dataValA[i]; int c = dataColA[i];

            int bs = rowPtrB[c]; int be = rowPtrB[c+1];
            if (bs + tx < be) { // Prevent over reading of B
                dataValC[bx * colsB + dataColB[bs + tx]] += valA * dataValB[bs + tx];
            }
        }
    }
}

CSRMatrix<double>* spgemm(CSRMatrix<double>* A, CSRMatrix<double>* B) {
#ifdef PROFILE
    nvtxNameOsThread(pthread_self(), "MAIN_THREAD");
    nvtxRangePushA("CUDA_spgemm");
    nvtxRangePushA("CUDA_spgemm_setup");
#endif
    if (A->rows > 1024 || B->cols > 1024) {
        printf("Error: Does not support resultant matrices of size greater than 1024x1024\n");
        return nullptr;
    }
    int *d_rowPtrA, *d_dataColA, *d_rowPtrB, *d_dataColB;
    double *d_dataValA, *d_dataValB, *d_dataValC, *h_dataValC;

    size_t
    rowPtrA_size  = (A->rows+1) * sizeof(int),
    dataColA_size = (A->nnz) * sizeof(int),
    dataValA_size = (A->nnz) * sizeof(double),
    rowPtrB_size  = (B->rows+1) * sizeof(int),
    dataColB_size = (B->nnz) * sizeof(int),
    dataValB_size = (B->nnz) * sizeof(double),
    dataValC_size = (A->rows * B->cols) * sizeof(double);

    cudaMalloc(&d_rowPtrA , rowPtrA_size );
    cudaMalloc(&d_dataColA, dataColA_size);
    cudaMalloc(&d_dataValA, dataValA_size);
    cudaMalloc(&d_rowPtrB , rowPtrB_size );
    cudaMalloc(&d_dataColB, dataColB_size);
    cudaMalloc(&d_dataValB, dataValB_size);
    cudaMalloc(&d_dataValC, dataValC_size);

    cudaMemcpy(d_rowPtrA , A->rowPtr , rowPtrA_size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataColA, A->dataCol, dataColA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataValA, A->dataVal, dataValA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtrB , B->rowPtr , rowPtrB_size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataColB, B->dataCol, dataColB_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataValB, B->dataVal, dataValB_size, cudaMemcpyHostToDevice);
    cudaMemset(d_dataValC, 0, dataValC_size);

    
    dim3 threadsPerBlock(32*((B->cols+31)/32));
    dim3 numBlocks(A->rows);
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_compute");
#endif

    spgemm_dacc<<<numBlocks,threadsPerBlock>>>(A->rows, A->cols, d_rowPtrA, d_dataColA, d_dataValA,
                                               B->rows, B->cols, d_rowPtrB, d_dataColB, d_dataValB,
                                               d_dataValC);
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_deardown");
#endif

    h_dataValC = (double*) malloc(dataValC_size);
    cudaMemcpy(h_dataValC, d_dataValC, dataValC_size, cudaMemcpyDeviceToHost);
    CSRMatrix<double>* ret = new CSRMatrix<double>(A->rows, B->cols, h_dataValC);

    cudaFree(d_rowPtrA );
    cudaFree(d_dataColA);
    cudaFree(d_dataValA);
    cudaFree(d_rowPtrB );
    cudaFree(d_dataColB);
    cudaFree(d_dataValB);
    cudaFree(d_dataValC);
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePop();
#endif

    return ret;
}

}
