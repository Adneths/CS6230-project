#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <cusparse.h>

#include "SpMM_cusparse.h"

#ifdef PROFILE
#include "timer.h"
#endif

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return nullptr;                                            \
        }                                                              \
    }

#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            return nullptr;                                                \
        }                                                                  \
    }

namespace cusparse
{

    CSRMatrix<double> *spmm(CSRMatrix<double> *A, dense_mat<double> *B)
    {
#ifdef PROFILE
        Timer timer;
        auto time = timer.tick();
#endif
        double alpha = 1.0f;
        double beta = 0.0f;
        // cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
        // cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
        // cudaDataType computeType = CUDA_R_64F;

        // Allocate
        int *d_rowPtrA, *d_dataColA;
        double *d_dataValA, *d_B, *d_C;
        size_t
            rowPtrA_size = (A->rows + 1) * sizeof(int),
            dataColA_size = (A->nnz) * sizeof(int),
            dataValA_size = (A->nnz) * sizeof(double),
            B_size = (B->total_size) * sizeof(double),
            C_size = (A->rows * B->col_num) * sizeof(double);

        cudaMalloc(&d_rowPtrA, rowPtrA_size);
        cudaMalloc(&d_dataColA, dataColA_size);
        cudaMalloc(&d_dataValA, dataValA_size);
        cudaMalloc(&d_B, B_size);
        cudaMalloc(&d_C, C_size);

        // Copy
        cudaMemcpy(d_rowPtrA, A->rowPtr, rowPtrA_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_dataColA, A->dataCol, dataColA_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_dataValA, A->dataVal, dataValA_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B->matrix, B_size, cudaMemcpyHostToDevice);
        //--------------------------------------------------------------------------
        // Cusparse Setup
        cusparseHandle_t handle = NULL;
        cusparseSpMatDescr_t matA;
        cusparseDnMatDescr_t matB, matC;
        void *dBuffer = NULL;
        size_t bufferSize = 0;
        CHECK_CUSPARSE(cusparseCreate(&handle))
        CHECK_CUSPARSE(cusparseCreateCsr(&matA, A->rows, A->cols, A->nnz,
                                         d_rowPtrA, d_dataColA, d_dataValA,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
        // Create dense matrix B
        CHECK_CUSPARSE(cusparseCreateDnMat(&matB, B->row_num, B->col_num, B->col_num, d_B,
                                           CUDA_R_64F, CUSPARSE_ORDER_ROW))
        // Create dense matrix C
        CHECK_CUSPARSE(cusparseCreateDnMat(&matC, A->rows, B->col_num, A->rows, d_C,
                                           CUDA_R_64F, CUSPARSE_ORDER_ROW))
        //--------------------------------------------------------------------------
#ifdef PROFILE
        time = timer.tick();
        std::cout << "CuSparse Setup: " << time << std::endl;
        timer.tick();
#endif
        // SpMM Computation
        // allocate an external buffer if needed
        CHECK_CUSPARSE(cusparseSpMM_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize))
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

        // execute SpMM
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, matB, &beta, matC, CUDA_R_64F,
                                    CUSPARSE_SPMM_ALG_DEFAULT, dBuffer))
#ifdef PROFILE
        time = timer.tick();
        std::cout << "CuSparse Compute: " << time << std::endl;
        timer.tick();
#endif
        // destroy matrix/vector descriptors
        CHECK_CUSPARSE(cusparseDestroySpMat(matA))
        CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
        CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
        CHECK_CUSPARSE(cusparseDestroy(handle))
        //--------------------------------------------------------------------------

        double *h_C;
        h_C = (double *)malloc(A->rows * B->col_num * sizeof(double));

        cudaMemcpy(h_C, d_C, A->rows * B->col_num * sizeof(double), cudaMemcpyDeviceToHost);

        CSRMatrix<double> *ret = new CSRMatrix<double>(A->rows, B->col_num, h_C);

        cudaFree(dBuffer);
        cudaFree(d_rowPtrA);
        cudaFree(d_dataColA);
        cudaFree(d_dataValA);
        cudaFree(d_B);
        cudaFree(d_C);
#ifdef PROFILE
        time = timer.tick();
        std::cout << "CuSparse Teardown: " << time << std::endl;
#endif

        return ret;
    }
}