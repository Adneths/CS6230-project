
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <random>
#include <cfloat>

#include "device_launch_parameters.h"

#include "SpMSpV_cuda.h"

#ifdef PROFILE
#include "timer.h"
#endif

namespace cuda {

SpVector<double>* gen_rand_spvec(int len, double sparsity) {
    if (len <= 0 || sparsity <= 0 || sparsity >= 1) {
        printf("invalid sparsity: %d", sparsity);
        return nullptr;
    }
    int* ind = (int*)malloc(sizeof(int));
    double* data = (double*)malloc(sizeof(double));
    std::random_device rd;  
    std::mt19937 gen(rd()); 

    // Define the distribution to be uniform between 0 and 1
    // Note that std::uniform_real_distribution is half-open [a, b), so we use 0 and nextafter(1, DBL_MAX) to mimic (0, 1)
    std::uniform_real_distribution<> dis(std::nextafter(0, DBL_MAX), std::nextafter(1, DBL_MAX)); 
    int nnz = 0;
    for (int i = 0; i < len; i++)
        if (dis(gen) < sparsity) {
            ind[nnz] = i;
            data[nnz++] = 1.0;
        }
    return new SpVector(len, nnz, ind, data);
    
}




__global__ void spmspv_naive_matdriven_dacc(int rowsA, int colsA, int* rowPtrA, int* dataColA, double* dataValA, int lenB, int nnzB, int* indB, double* dataB, double* dataC) {
    int tx = threadIdx.x; int bx = blockIdx.x; int bm = blockDim.x;
    int rowA_x = bx * bm + tx;
    if (rowA_x < rowsA) {
        int as = rowPtrA[rowA_x]; int ae = rowPtrA[rowA_x+1];
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
    dataValB_size = (B->nnz) * sizeof(double),
    dataValC_size = (B->len) * sizeof(double);

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
    cudaMemcpy(d_dataValB, B->data, dataValB_size, cudaMemcpyHostToDevice);
    cudaMemset(d_dataValC, 0, dataValC_size);

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

__global__ void spmspv_naive_vecdriven_dacc(int rowsA, int colsA, int colPtrA, int* dataRowA, doulbe*dataValA, int lenB, int nnzB, listformat_element<double>* elementsB, double* dataC, double* vecC) {
    int tx = threadIdx.x; int bx = blockIdx.x;
    if (bx * blockDim.x + threadIdx < lenB) {
        int indB_x = elementsB[bx * blockDim.x + threadIdx]->idx;
        double valB = elementsB[bx * blockDim.x + threadIdx]->data;
        int as = colPtrA[indB_x]; int ae = colPtr[indB_x+1];
        for (int i = as; i < ae; i++) {
            dataC[dataRowA[i]][indB_x] = dataValA[i] * valB;
        }
    }
    // synchronize()
    __syncthreads(); // does this work?

    for (int i = 0; i < colsA; i++) {
        vecC[indB_x] += data[i][indB_x];
    }

}

LF_SpVector<double>* spmspv_naive_vecdriven(CSCMatrix<double>* A, LF_SpVector<double>* B){
    if (A->cols != B->len) {
        printf("#Cols of the Matrix dosen't match Len of the Vector!\n");
        return nullptr;
    }
#ifdef PROFILE
    Timer timer;
    auto time = timer.tick();
#endif
    int *d_colPtrA, *d_dataRowA;
    double *d_dataValA, *d_dataValC, *d_dataVecC, *h_dataValC;
    listformat_element<double>* d_elements_B;

    size_t
    colPtrA_size  = (A->cols+1) * sizeof(int),
    dataRowA_size = (A->nnz) * sizeof(int),
    dataValA_size = (A->nnz) * sizeof(double),
    elementsB_size = (B->nnz) * sizeof(listformat_element<double>),
    dataValC_size = (A->cols * B->len) * sizeof(double),
    dataVecC_size = (B->len) * sizeof(double);

    cudaMalloc(&d_colPtrA , colPtrA_size );
    cudaMalloc(&d_dataRowA, dataRowA_size);
    cudaMalloc(&d_dataValA, dataValA_size);
    cudaMalloc(&d_elements_B, elementsB_size);
    cudaMalloc(&d_dataValC, dataValC_size);

    cudaMemcpy(d_colPtrA , A->colPtr , colPtrA_size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataRowA, A->dataRow, dataRowA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataValA, A->dataVal, dataValA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_elements_B, B->elements, elementsB_size, cudaMemcpyHostToDevice);
    cudaMemset(d_dataValC, 0, dataValC_size);
    cudaMemset(d_dataVecC, 0, dataVecC_size);

    dim3 threadsPerBlock(32*((B->len+31)/32));
    dim3 numBlocks((B->len + threadsPerBlock - 1)/threadsPerBlock);
#ifdef PROFILE
    time = timer.tick();
    std::cout << "Cuda Setup: " << time << std::endl;
    timer.tick();
#endif

    spmspv_naive_vecdriven_dacc<<<numBlocks, threadsPerBlock>>>(A->rows, A->cols, d_colPtrA, d_dataRowA, d_dataValA, B->len, B->nnz, d_elements_B, d_dataValC, d_dataVecC);

#ifdef PROFILE
    time = timer.tick();
    std::cout << "Cuda Compute: " << time << std::endl;
    timer.tick();
#endif

    h_dataValC = (double*) malloc(dataValC_size);
    cudaMemcpy(h_dataValC, d_datVecC, dataVecC_size, cudaMemcpyDeviceToHost);
    LF_SpVector<double>* ret = new LF_SpVector<double>(B->len, h_dataValC);
    
    cudaFree(d_colPtrA );
    cudaFree(d_dataRowA);
    cudaFree(d_dataValA);
    cudaFree(d_elements_B);
    cudaFree(d_dataValC);
#ifdef PROFILE
    time = timer.tick();
    std::cout << "Cuda Teardown: " << time << std::endl;
#endif

    return ret;
}

}
