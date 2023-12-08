

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

__global__ void spmspv_bucket_prepare(int rowsA, int colsA, int* colPtrA, int* dataRowA, double* dataValA, int lenB, int nnzB, listformat_element<double>* elementsB, int* d_Boffset, int nbucket) {
    int tx = threadIdx.x;
    int stride = blockDim.x;
    for (int i = tx; i < nnzB; i += stride) {
        int c = elementsB[i].idx;
        int as = colPtrA[c]; int ae = colPtrA[c+1];
        for (int j = as; j < ae; j++) {
            int b = dataRowA[j] * nbucket / rowsA;
            d_Boffset[tx * nbucket + b] += 1;
        }
    }

    __syncthreads();
    for (int i = tx; i < nbucket; i += stride) {
        for (int j = 1; j < stride; j++) 
            d_Boffset[j * nbucket + i] += d_Boffset[(j-1) * nbucket + i];
    }

    __syncthreads();
    if (tx == 0) {
        d_Boffset[stride * nbucket] = d_Boffset[(stride-1) * nbucket];
        for (int i = 1; i < nbucket; i++)
            d_Boffset[stride * nbucket + i] = d_Boffset[stride * nbucket + i-1] + d_Boffset[(stride-1) * nbucket + i];
    }
    
}
__global__ void spmspv_bucket_insert(int rowsA, int colsA, int* colPtrA, int* dataRowA, double* dataValA, int lenB, int nnzB, listformat_element<double>* elementsB, int* d_Boffset, int nbucket, struct listformat_element<double> *d_bucket, double* d_SPA) {
    int tx = threadIdx.x;
    int stride = blockDim.x;
    int inserted_cnt[64*4] = {0};
    // int inserted_cnt[nbucket] = {0};

    for (int i = tx; i < nnzB; i += stride) {
        int c = elementsB[i].idx;
        int as = colPtrA[c]; int ae = colPtrA[c+1];
        for (int j = as; j < ae; j++) {
            int k = dataRowA[j] * nbucket / rowsA;
            double val = elementsB[i].data * dataValA[j];
            int idx = 0;
            if (k > 0)
                idx += d_Boffset[stride * nbucket + k-1];
            if (tx > 0)
                idx += d_Boffset[(tx - 1) * nbucket + k];
            idx += inserted_cnt[k];
            d_bucket[idx].idx = dataRowA[j];
            d_bucket[idx].data = val;
            inserted_cnt[k]++;
        }
    }

    __syncthreads();
    for (int i = tx; i < nbucket; i+= stride) {
        int bs;
        if (i > 0)
            bs = d_Boffset[stride * nbucket + i-1];
        else
            bs = 0;
        int be = d_Boffset[stride * nbucket + i];
        for (int j = bs; j < be; j++) {
            int ind = d_bucket[j].idx;
            d_SPA[ind] += d_bucket[j].data;
        }
    }
}

LF_SpVector<double>* spmspv_bucket(CSCMatrix<double>* A, LF_SpVector<double>* B) {
// int* spmspv_bucket(CSCMatrix<double>* A, LF_SpVector<double>* B) {
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
    const int nthread = 64;
    const int nbucket = nthread * 4;
    int *d_Boffset, *d_colPtrA, *d_dataRowA;
    struct listformat_element<double> *d_bucket;
    double *d_SPA, *h_SPA, *d_dataValA;
    listformat_element<double>* d_elements_B;

    size_t
    Boffset_zize = ((nthread+1) * nbucket) * sizeof(int),
    bucket_size = (A->nnz) * sizeof(struct listformat_element<double>),
    spa_size = (A->rows) * sizeof(double),
    colPtrA_size  = (A->cols+1) * sizeof(int),
    dataRowA_size = (A->nnz) * sizeof(int),
    dataValA_size = (A->nnz) * sizeof(double),
    elementsB_size = (B->nnz) * sizeof(listformat_element<double>);

    cudaMalloc(&d_colPtrA , colPtrA_size );
    cudaMalloc(&d_dataRowA, dataRowA_size);
    cudaMalloc(&d_dataValA, dataValA_size);
    cudaMalloc(&d_elements_B, elementsB_size);

    cudaMalloc(&d_Boffset, Boffset_zize);
    cudaMalloc(&d_bucket, bucket_size);
    cudaMalloc(&d_SPA, spa_size);

    cudaMemcpy(d_colPtrA , A->colPtr , colPtrA_size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataRowA, A->dataRow, dataRowA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataValA, A->dataVal, dataValA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_elements_B, B->elements, elementsB_size, cudaMemcpyHostToDevice);
    cudaMemset(d_SPA, 0, spa_size);
    cudaMemset(d_Boffset, 0, Boffset_zize);


    dim3 threadsPerBlock(nthread);
    dim3 numBlocks(1);
    
#ifdef PROFILE
    time = timer.tick();
    std::cout << "Cuda Setup: " << time << std::endl;
    timer.tick();
#endif

    spmspv_bucket_prepare<<<numBlocks, threadsPerBlock>>>(A->rows, A->cols, d_colPtrA, d_dataRowA, d_dataValA, B->len, B->nnz, d_elements_B, d_Boffset, nbucket);
    cudaDeviceSynchronize();

    // Boffset debugging
    // int* h_Boffset;
    // h_Boffset = (int*)malloc(Boffset_zize);
    // cudaMemcpy(h_Boffset, d_Boffset, Boffset_zize, cudaMemcpyDeviceToHost);

    //     std::cout << "Boffset:\n[";
    //     for (int i = 64; i < 65; i++) {
    //         // std::cout << i << ": ";
    //         std::cout << "[";
    //         for (int j = 0; j < 64*4; j++) {
    //             std::cout << *(h_Boffset+ i*64*4+j);
    //             if (j != 64*4-1)
    //                 std::cout << ", ";
    //         }
    //         std::cout << "]";
    //         if (i != 64)
    //         std::cout << ",";
    //     }
    //     std::cout << "]" << std::endl;

    spmspv_bucket_insert<<<numBlocks, threadsPerBlock>>>(A->rows, A->cols, d_colPtrA, d_dataRowA, d_dataValA, B->len, B->nnz, d_elements_B, d_Boffset, nbucket, d_bucket, d_SPA);
    h_SPA = (double*)malloc(spa_size);
    cudaDeviceSynchronize();

#ifdef PROFILE
    time = timer.tick();
    std::cout << "Cuda Compute: " << time << std::endl;
    timer.tick();
#endif

    cudaMemcpy(h_SPA, d_SPA, spa_size, cudaMemcpyDeviceToHost);

    // Bucket debugging
    // struct listformat_element<double>* h_bucket;
    // h_bucket = (struct listformat_element<double>*)malloc(bucket_size);
    // cudaMemcpy(h_bucket, d_bucket, bucket_size, cudaMemcpyDeviceToHost);
    // std::cout << "h_bucket:\n";
    // for (int i = 0; i < A->nnz; i++) {
    //     std::cout << h_bucket[i] << ",";
    // }
    // std::cout << std::endl;

    // std::cout <<"h_SPA:\n";
    // for(int i = 0; i < A->rows; i++) {
    //     std::cout << h_SPA[i] <<", ";
    // }
    // std::cout << std::endl;



    LF_SpVector<double>* ret = new LF_SpVector<double>(A->rows, h_SPA);
    
    cudaFree(d_colPtrA );
    cudaFree(d_dataRowA);
    cudaFree(d_dataValA);
    cudaFree(d_elements_B);
    cudaFree(d_Boffset);
    cudaFree(d_bucket);
    cudaFree(d_SPA);
#ifdef PROFILE
    time = timer.tick();
    std::cout << "Cuda Teardown: " << time << std::endl;
#endif

    return ret;
}

}