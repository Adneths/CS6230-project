
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#include "TileSpGEMM_cuda.h"

#ifdef PROFILE
#include <nvtx3/nvToolsExtCuda.h>
#endif

namespace cuda {

CSRMatrix<double>* tile_spgemm(CSRMatrix<double>* A, CSRMatrix<double>* B) {
    TileSpMatrix<double>* tileA = new TileSpMatrix<double>(A);
    TileSpMatrix<double>* tileB = new TileSpMatrix<double>(B);
    std::cout << tileA << std::endl;
    return nullptr;
}

}

namespace cusparse {
    CSRMatrix<double>* spgemm_symbolic(CSRMatrix<double>* A, CSRMatrix<double>* B) {
#ifdef PROFILE
    nvtxNameOsThread(pthread_self(), "MAIN_THREAD");
    nvtxRangePushA("CuSparse_spgemm");
    nvtxRangePushA("CuSparse_spgemm_setup");
#endif
    double              alpha       = 1.0f;
    double              beta        = 0.0f;
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_64F;

    // Allocate
    int *d_rowPtrA, *d_dataColA, *d_rowPtrB, *d_dataColB, *d_rowPtrC, *d_dataColC;
    double *d_dataValA, *d_dataValB, *d_dataValC;
    size_t
    rowPtrA_size  = (A->rows+1) * sizeof(int),
    dataColA_size = (A->nnz) * sizeof(int),
    dataValA_size = (A->nnz) * sizeof(double),
    rowPtrB_size  = (B->rows+1) * sizeof(int),
    dataColB_size = (B->nnz) * sizeof(int),
    dataValB_size = (B->nnz) * sizeof(double);
    
    cudaMalloc(&d_rowPtrA , rowPtrA_size );
    cudaMalloc(&d_dataColA, dataColA_size);
    cudaMalloc(&d_dataValA, dataValA_size);
    cudaMalloc(&d_rowPtrB , rowPtrB_size );
    cudaMalloc(&d_dataColB, dataColB_size);
    cudaMalloc(&d_dataValB, dataValB_size);
    cudaMalloc(&d_rowPtrC , rowPtrA_size );

    // Copy
    cudaMemcpy(d_rowPtrA , A->rowPtr , rowPtrA_size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataColA, A->dataCol, dataColA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataValA, A->dataVal, dataValA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtrB , B->rowPtr , rowPtrB_size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataColB, B->dataCol, dataColB_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataValB, B->dataVal, dataValB_size, cudaMemcpyHostToDevice);
    //--------------------------------------------------------------------------
    // Cusparse Setup
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A->rows, A->cols, A->nnz,
                                      d_rowPtrA, d_dataColA, d_dataValA,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, B->rows, B->cols, B->nnz,
                                      d_rowPtrB, d_dataColB, d_dataValB,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, A->rows, B->cols, 0,
                                      NULL, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )
    //--------------------------------------------------------------------------

#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CuSparse_spgemm_compute");
#endif
    // SpGEMM Computation
    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, NULL) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, dBuffer1) )

    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, matA, matB, &beta, matC,
                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, NULL) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer2, bufferSize2) )

    // compute the intermediate product of A * B
    CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, opA, opB,
                                           &alpha, matA, matB, &beta, matC,
                                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc, &bufferSize2, dBuffer2) )
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CuSparse_spgemm_deardown");
#endif
    // get matrix C non-zero entries C_nnz
    int64_t C_num_rows1, C_num_cols1, C_nnz;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz) )
    // allocate matrix C
    CHECK_CUDA( cudaMalloc((void**) &d_dataColC, C_nnz * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &d_dataValC,  C_nnz * sizeof(double)) )

    // NOTE: if 'beta' != 0, the values of C must be update after the allocation
    //       of dC_values, and before the call of cusparseSpGEMM_copy
    // update matC with the new pointers
    CHECK_CUSPARSE( cusparseCsrSetPointers(matC, d_rowPtrC, d_dataColC, d_dataValC) )

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values
    // copy the final products to the matrix C
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    
    int *h_rowPtrC, *h_dataColC;
    double *h_dataValC;
    h_rowPtrC = (int*)malloc((A->rows+1) * sizeof(int));
    h_dataColC = (int*)malloc(C_nnz * sizeof(int));
    h_dataValC = (double*)malloc(C_nnz * sizeof(double));

    cudaMemcpy(h_rowPtrC, d_rowPtrC, (A->rows+1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dataColC, d_dataColC, C_nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dataValC, d_dataValC, C_nnz * sizeof(double), cudaMemcpyDeviceToHost);

    CSRMatrix<double>* ret = new CSRMatrix<double>(A->rows, B->cols, h_rowPtrC, h_dataColC, h_dataValC, C_nnz);

    cudaFree(dBuffer1);
    cudaFree(dBuffer2);
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