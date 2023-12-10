#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <cusparse.h>

#include "SpGEMM_cusparse.h"

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

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return nullptr;                                                        \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return nullptr;                                                        \
    }                                                                          \
}

namespace cusparse {

CSRMatrix<double>* spgemm(CSRMatrix<double>* A, CSRMatrix<double>* B) {
    cudaSetDevice(0);
//#ifdef PROFILE
    NAME_THREAD("MAIN_THREAD");
    PUSH_RANGE("CuSparse_spgemm", BLACK);
    PUSH_RANGE("CuSparse_spgemm_cudamalloc", RED);
//#endif
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

//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CuSparse_spgemm_HtoD", AQUA);
//#endif
    // Copy
    cudaMemcpy(d_rowPtrA , A->rowPtr , rowPtrA_size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataColA, A->dataCol, dataColA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataValA, A->dataVal, dataValA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtrB , B->rowPtr , rowPtrB_size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataColB, B->dataCol, dataColB_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataValB, B->dataVal, dataValB_size, cudaMemcpyHostToDevice);
    //--------------------------------------------------------------------------
    // Cusparse Setup
//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CuSparse_spgemm_setupcusparse", PURPLE);
//#endif
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

//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CuSparse_spgemm_compute", GREEN);
//#endif
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

    // get matrix C non-zero entries C_nnz
    int64_t C_num_rows1, C_num_cols1, C_nnz;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz) )
    // allocate matrix C
//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CuSparse_spgemm_cudamalloc", RED);
//#endif
    CHECK_CUDA( cudaMalloc((void**) &d_dataColC, C_nnz * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &d_dataValC,  C_nnz * sizeof(double)) )

//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CuSparse_spgemm_DtoD", AQUA);
//#endif
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

//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CuSparse_spgemm_destroycusparse", PURPLE);
//#endif
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    
//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CuSparse_spgemm_malloc", RED);
//#endif
    int *h_rowPtrC, *h_dataColC;
    double *h_dataValC;
    h_rowPtrC = (int*)malloc((A->rows+1) * sizeof(int));
    h_dataColC = (int*)malloc(C_nnz * sizeof(int));
    h_dataValC = (double*)malloc(C_nnz * sizeof(double));

//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CuSparse_spgemm_DtoH", AQUA);
//#endif
    cudaMemcpy(h_rowPtrC, d_rowPtrC, (A->rows+1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dataColC, d_dataColC, C_nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dataValC, d_dataValC, C_nnz * sizeof(double), cudaMemcpyDeviceToHost);

//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CuSparse_spgemm_reconstruct", GRAY);
//#endif
    CSRMatrix<double>* ret = new CSRMatrix<double>(A->rows, B->cols, h_rowPtrC, h_dataColC, h_dataValC, C_nnz);

//#ifdef PROFILE
    POP_RANGE();
    PUSH_RANGE("CuSparse_spgemm_cudafree", BLUE);
//#endif
    cudaFree(dBuffer1);
    cudaFree(dBuffer2);
    cudaFree(d_rowPtrA );
    cudaFree(d_dataColA);
    cudaFree(d_dataValA);
    cudaFree(d_rowPtrB );
    cudaFree(d_dataColB);
    cudaFree(d_dataValB);
    cudaFree(d_rowPtrC );
    cudaFree(d_dataColC);
    cudaFree(d_dataValC);
//#ifdef PROFILE
    POP_RANGE();
    POP_RANGE();
//#endif
    
    return ret;
}
}
