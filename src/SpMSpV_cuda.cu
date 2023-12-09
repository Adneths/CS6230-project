
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

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return nullptr;                                                        \
    }                                                                          \
}

namespace cuda {

SpVector<double>* gen_rand_spvec(int len, double sparsity, int* ind, double* data) {
    if (len <= 0 || sparsity <= 0 || sparsity >= 1) {
        printf("invalid sparsity: %d", sparsity);
        return nullptr;
    }
    if (ind == nullptr || data == nullptr) {
        printf("ind or data still null\n");
        return nullptr;
    }
    // int* ind = (int*)malloc(sizeof(int));
    // double* data = (double*)malloc(sizeof(double));
    std::random_device rd;  
    std::mt19937 gen(rd()); 

    // Define the distribution to be uniform between 0 and 1
    // Note that std::uniform_real_distribution is half-open [a, b), so we use 0 and nextafter(1, DBL_MAX) to mimic (0, 1)
    std::uniform_real_distribution<> dis(std::nextafter(0, DBL_MAX), std::nextafter(1, DBL_MAX)); 
    int nnz = 0;
    for (int i = 0; i < len; i++)
        if (dis(gen) < sparsity) {
        // if (i == 8|| i ==9|| i ==11|| i ==12|| i ==20|| i ==27|| i ==29|| i ==36|| i ==39|| i ==41|| i ==45|| i ==46) {
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
        printf("A->cols %d\n",A->cols);
        printf("B->len %d\n", B->len);
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
    dataValC_size = (A->rows) * sizeof(double);

    CHECK_CUDA(cudaMalloc(&d_rowPtrA , rowPtrA_size ));
    CHECK_CUDA(cudaMalloc(&d_dataColA, dataColA_size));
    CHECK_CUDA(cudaMalloc(&d_dataValA, dataValA_size));
    CHECK_CUDA(cudaMalloc(&d_indB, indB_size));
    CHECK_CUDA(cudaMalloc(&d_dataValB, dataValB_size));
    CHECK_CUDA(cudaMalloc(&d_dataValC, dataValC_size));

    cudaMemcpy(d_rowPtrA , A->rowPtr , rowPtrA_size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataColA, A->dataCol, dataColA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataValA, A->dataVal, dataValA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indB, B->ind, indB_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataValB, B->data, dataValB_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataValB, B->data, dataValB_size, cudaMemcpyHostToDevice);
    cudaMemset(d_dataValC, 0, dataValC_size);


    int threadsPerBlock;
    if (32 * ((A->rows + 31) / 32) < 1025)
        threadsPerBlock = (32 * ((A->rows + 31) / 32));
    else
        threadsPerBlock = (1024);


    std::cout << "threadsPerBlock: " << threadsPerBlock << std::endl;
    // Calculate the number of blocks as an integer first
    int numBlocks = (A->rows + threadsPerBlock - 1) / threadsPerBlock.x;
    std::cout << "numBlocks: " << numBlocks << std::endl;
    
    // Then use this integer to create a dim3 object
    // dim3 numBlocks(numBlocksInt);
    
#ifdef PROFILE
    time = timer.tick();
    std::cout << "SPMSPV_Naive_Matrix_Driven\n";
    std::cout << "Cuda Setup: " << time << std::endl;
    timer.tick();
#endif

    spmspv_naive_matdriven_dacc<<<numBlocks, threadsPerBlock>>>(A->rows, A->cols, d_rowPtrA, d_dataColA, d_dataValA, B->len, B->nnz, d_indB, d_dataValB, d_dataValC);
    cudaDeviceSynchronize();

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

// __global__ void spmspv_naive_vecdriven_dacc(int rowsA, int colsA, int* colPtrA, int* dataRowA, double* dataValA, int lenB, int nnzB, listformat_element<double>* elementsB, double* dataC, double* vecC) {
//     int tx = threadIdx.x; int bx = blockIdx.x;
//     int indB_x = elementsB[bx * blockDim.x + threadIdx.x].idx;
//     double valB = elementsB[bx * blockDim.x + threadIdx.x].data;
//     if (bx * blockDim.x + threadIdx.x < lenB) {

//         int as = colPtrA[indB_x]; int ae = colPtrA[indB_x+1];
//         for (int i = as; i < ae; i++) {
//             dataC[dataRowA[i]*colsA+indB_x] = dataValA[i] * valB;
//         }
//     }
//     // synchronize()
//     __syncthreads(); // need synchronize in wider range

//     for (int i = 0; i < colsA; i++) {
//         vecC[indB_x] += dataC[indB_x * colsA + i];
//     }

// }

__global__ void spmspv_naive_vecdriven_mul(int rowsA, int colsA, int* colPtrA, int* dataRowA, double* dataValA, int lenB, int nnzB, listformat_element<double>* elementsB, double* dataC) {
    int tx = threadIdx.x; int bx = blockIdx.x;
    int stride = blockDim.x;
    int idx = elementsB[bx].idx;
    double valB = elementsB[bx].data;
    int as = colPtrA[idx];
    int ae = colPtrA[idx+1];
    for (int i = as + tx; i < ae; i += stride) {
        dataC[idx * colsA + dataRowA[i]] = dataValA[i] * valB;
    }
    // __syncthreads();
}
__global__ void spmspv_naive_vecdriven_dacc(int lenB, double* dataC, double* vecC) {
    int tx = threadIdx.x; int bx = blockIdx.x;
    int idx = bx * blockDim.x + tx;
    for (int i = 0; i < lenB; i++)
        vecC[idx] += dataC[i * lenB + idx]; 
}

LF_SpVector<double>* spmspv_naive_vecdriven(CSCMatrix<double>* A, LF_SpVector<double>* B){
    if (A->cols != B->len) {
        printf("In spmspv_naive_vecdriven:\n");
        printf("A addr:%d\n", A);
        printf("#Cols of the Matrix dosen't match Len of the Vector!\n");
        printf("A->cols %d\n",A->cols);
        printf("B->len %d\n", B->len);
        return nullptr;
    }
#ifdef PROFILE
    Timer timer;
    auto time = timer.tick();
#endif
    int *d_colPtrA, *d_dataRowA;
    double *d_dataValA, *d_dataValC, *d_dataVecC, *h_dataVecC;
    listformat_element<double>* d_elements_B;

    size_t
    colPtrA_size  = (A->cols+1) * sizeof(int),
    dataRowA_size = (A->nnz) * sizeof(int),
    dataValA_size = (A->nnz) * sizeof(double),
    elementsB_size = (B->nnz) * sizeof(listformat_element<double>),
    dataValC_size = (A->rows * B->len) * sizeof(double),
    dataVecC_size = (A->rows) * sizeof(double);

    cudaMalloc(&d_colPtrA , colPtrA_size );
    cudaMalloc(&d_dataRowA, dataRowA_size);
    cudaMalloc(&d_dataValA, dataValA_size);
    cudaMalloc(&d_elements_B, elementsB_size);
    cudaMalloc(&d_dataValC, dataValC_size);
    cudaMalloc(&d_dataVecC, dataVecC_size);

    cudaMemcpy(d_colPtrA , A->colPtr , colPtrA_size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataRowA, A->dataRow, dataRowA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataValA, A->dataVal, dataValA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_elements_B, B->elements, elementsB_size, cudaMemcpyHostToDevice);
    cudaMemset(d_dataValC, 0, dataValC_size);
    cudaMemset(d_dataVecC, 0, dataVecC_size);

    dim3 threadsPerBlock(32 * ((A->rows + 31) / 32));

    // Calculate the number of blocks as an integer first
    // int numBlocksInt = (B->nnz + threadsPerBlock.x - 1) / threadsPerBlock.x;
    
    // Then use this integer to create a dim3 object
    dim3 numBlocks(B->nnz);
    
#ifdef PROFILE
    time = timer.tick();
    std::cout << "SPMSPV_naive_vector_driven\n";
    std::cout << "Cuda Setup: " << time << std::endl;
    timer.tick();
#endif

    spmspv_naive_vecdriven_mul<<<numBlocks, threadsPerBlock>>>(A->rows, A->cols, d_colPtrA, d_dataRowA, d_dataValA, B->len, B->nnz, d_elements_B, d_dataValC);
    dim3 threadsPerBlock_1(32 * ((B->len + 31) / 32));

    // Calculate the number of blocks as an integer first
    int numBlocksInt = (B->len + threadsPerBlock_1.x - 1) / threadsPerBlock.x;
    
    // Then use this integer to create a dim3 object
    dim3 numBlocks_1(numBlocksInt);
    cudaDeviceSynchronize();
    spmspv_naive_vecdriven_dacc<<<numBlocks_1, threadsPerBlock_1>>>(B->len, d_dataValC, d_dataVecC);
    cudaDeviceSynchronize();

#ifdef PROFILE
    time = timer.tick();
    std::cout << "Cuda Compute: " << time << std::endl;
    timer.tick();
#endif

    h_dataVecC = (double*) malloc(dataVecC_size);
    cudaMemcpy(h_dataVecC, d_dataVecC, dataVecC_size, cudaMemcpyDeviceToHost);
    LF_SpVector<double>* ret = new LF_SpVector<double>(B->len, h_dataVecC);
    
    cudaFree(d_colPtrA );
    cudaFree(d_dataRowA);
    cudaFree(d_dataValA);
    cudaFree(d_elements_B);
    cudaFree(d_dataValC);
    cudaFree(d_dataVecC);
#ifdef PROFILE
    time = timer.tick();
    std::cout << "Cuda Teardown: " << time << std::endl;
#endif

    return ret;
}

}
