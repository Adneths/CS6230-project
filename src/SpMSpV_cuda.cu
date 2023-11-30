
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#include "SpMSpV_cuda.h"

#ifdef PROFILE
#include "timer.h"
#endif

namespace cuda {

__global__ void spmspv_naive_matdriven_dacc(int rowsA, int cols A, int rowPtrA, int* dataColA, doulbe*dataValA, int lenB, int nnzB, int* indB, double* dataB, double* dataC) {
    int tx = threadIdx.x; int bx = blockIdx.x;
    int rowA_x = bx * blockDim.x + threadIdx;
    if (rowA_x < rowsA) {
        int as = rowPtrA[rowA_x]; int ae = rowPtr[rowA_x+1];
        for (int i = as; i < ae; i++) {
            double valA = dataValA[i]; int c = dataColA[i];

            for (int j = 0; j < nnzB; j++)
                if (indB[j] == c)
                    dataC[rowA_x] += valA * dataB[j];
        }
    }  
}


SpVector<double>* spmspv_naive_matdriven(CSRMatrix<double>* A, SpVector<double>* B){
    if (A->cols != B->len) {
        printf("#Cols of the Matrix dosen't match Len of the Vector!\n");
        return nullptr;
    }
#ifdef PROFILE
    Timer timer;
    auto time = timer.tick();
#endif
    int *d_rowPtrA, *d_dataColA, *d_indB;
    double *d_dataValA, *d_dataValB, *d_dataValC, *h_dataValC;

    size_t
    rowPtrA_size  = (A->rows+1) * sizeof(int),
    dataColA_size = (A->nnz) * sizeof(int),
    dataValA_size = (A->nnz) * sizeof(double),
    indB_size = (B->nnz) * sizeof(int),
    datavalB_size = (B->nnz) * sizeof(double),
    datavalC_size = (B->len) * sizeof(double);

    cudaMalloc(&d_rowPtrA , rowPtrA_size );
    cudaMalloc(&d_dataColA, dataColA_size);
    cudaMalloc(&d_dataValA, dataValA_size);
    cudaMalloc(&d_indB, indB_size);
    cudaMalloc(&d_dataValB, dataValB_size);
    cudaMalloc(&d_dataValC, dataValC_size);

    cudaMemcpy(d_rowPtrA , A->rowPtr , rowPtrA_size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataColA, A->dataCol, dataColA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataValA, A->dataVal, dataValA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indB, B->ind, indB_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataValB, B->data, dataValB_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataValB, B->dataVal, dataValB_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32*((A->rows+31)/32));
    dim3 numBlocks((A->rows + threadsPerBlock - 1)/threadsPerBlock);
#ifdef PROFILE
    time = timer.tick();
    std::cout << "Cuda Setup: " << time << std::endl;
    timer.tick();
#endif

    spmspv_naive_matdriven_dacc<<<numBlocks, threadsPerBlock>>>(A->rows, A->cols, d_rowPtrA, d_dataColA, d_dataValA, B->len, B->nnz, d_indB, d_dataValB, d_dataValC);

#ifdef PROFILE
    time = timer.tick();
    std::cout << "Cuda Compute: " << time << std::endl;
    timer.tick();
#endif

    h_dataValC = (double*) malloc(dataValC_size);
    cudaMemcpy(h_dataValC, d_dataValC, dataValC_size, cudaMemcpyDeviceToHost);
    SpVector<double>* ret = new SpVector<double>(B->len, h_dataValC);
    
    cudaFree(d_rowPtrA );
    cudaFree(d_dataColA);
    cudaFree(d_dataValA);
    cudaFree(d_indB);
    cudaFree(d_dataValB);
    cudaFree(d_dataValC);
#ifdef PROFILE
    time = timer.tick();
    std::cout << "Cuda Teardown: " << time << std::endl;
#endif

    return ret;
}

LF_SpVector<double>* spmspv_naive_vecdriven(CSCMatrix<double>* A, LF_SpVector<double>* B){

}

}