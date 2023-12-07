
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <cusparse.h>

#include "TileSpGEMM_cuda.h"

#ifdef PROFILE
#include <nvtx3/nvToolsExtCuda.h>
#endif

void _spgemm_symbolic(int rowsA, int colsA, int nnzA, int rowsB, int colsB, int nnzB,
    int *d_rowPtrA, int *d_colIdxA, int *d_rowPtrB, int *d_colIdxB, int **d_rowPtrC, int **d_colIdxC);

// Could optimize by searching in larger blocks
__global__ void formRowIdx(int rows, int nnz, int* rowPtr, int* rowIdx) {
    int tx = threadIdx.x; int bx = blockIdx.x;
    int id = bx*1024+tx;
    if (id < nnz) {
        int s = 0, e = rows;
        int m = (s+e)/2;
        if (rowPtr[m] <= id && rowPtr[m] > id) {
            rowIdx[id] = m;
        }
    }
}

namespace cuda {

CSRMatrix<double>* tile_spgemm(CSRMatrix<double>* A, CSRMatrix<double>* B) {
    TileSpMatrix<double>* tileA = new TileSpMatrix<double>(A);
    TileSpMatrix<double>* tileB = new TileSpMatrix<double>(B);
    std::cout << tileA << std::endl;
    
    int *dtileRowPtrA, *dtileColIdxA, *dtileNnzsA;
    uint8_t drowPtrA, drowcolIdxA;
    double *dvalsA;
    uint16_t *dmasksA;
    int *dtileRowPtrB, *dtileColIdxB, *dtileNnzsB;
    uint8_t drowPtrB, drowcolIdxB;
    double *dvalsB;
    uint16_t *dmasksB;
    cudaMalloc((void**)&dtileRowPtrA, tileA->tileRowPtr.size() * sizeof(int)   );
    cudaMalloc((void**)&dtileColIdxA, tileA->tileColIdx.size() * sizeof(int)   );
    cudaMalloc((void**)  &dtileNnzsA, tileA->tileNnzs.size() * sizeof(int)     );
    cudaMalloc((void**)    &drowPtrA, tileA->rowPtr.size() * sizeof(uint8_t)   );
    cudaMalloc((void**) &drowcolIdxA, tileA->rowcolIdx.size() * sizeof(uint8_t));
    cudaMalloc((void**)      &dvalsA, tileA->vals.size() * sizeof(double)      );
    cudaMalloc((void**)     &dmasksA, tileA->masks.size() * sizeof(uint16_t)   );
    cudaMalloc((void**)&dtileRowPtrB, tileB->tileRowPtr.size() * sizeof(int)   );
    cudaMalloc((void**)&dtileColIdxB, tileB->tileColIdx.size() * sizeof(int)   );
    cudaMalloc((void**)  &dtileNnzsB, tileB->tileNnzs.size() * sizeof(int)     );
    cudaMalloc((void**)    &drowPtrB, tileB->rowPtr.size() * sizeof(uint8_t)   );
    cudaMalloc((void**) &drowcolIdxB, tileB->rowcolIdx.size() * sizeof(uint8_t));
    cudaMalloc((void**)      &dvalsB, tileB->vals.size() * sizeof(double)      );
    cudaMalloc((void**)     &dmasksB, tileB->masks.size() * sizeof(uint16_t)   );

    cudaMemcpy(&dtileRowPtrA, tileA->tileRowPtr.data(), tileA->tileRowPtr.size() * sizeof(int)   , cudaMemcpyHostToDevice);
    cudaMemcpy(&dtileColIdxA, tileA->tileColIdx.data(), tileA->tileColIdx.size() * sizeof(int)   , cudaMemcpyHostToDevice);
    cudaMemcpy(  &dtileNnzsA, tileA->tileNnzs  .data(), tileA->tileNnzs.size() * sizeof(int)     , cudaMemcpyHostToDevice);
    cudaMemcpy(    &drowPtrA, tileA->rowPtr    .data(), tileA->rowPtr.size() * sizeof(uint8_t)   , cudaMemcpyHostToDevice);
    cudaMemcpy( &drowcolIdxA, tileA->rowcolIdx .data(), tileA->rowcolIdx.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(      &dvalsA, tileA->vals      .data(), tileA->vals.size() * sizeof(double)      , cudaMemcpyHostToDevice);
    cudaMemcpy(     &dmasksA, tileA->masks     .data(), tileA->masks.size() * sizeof(uint16_t)   , cudaMemcpyHostToDevice);
    cudaMemcpy(&dtileRowPtrB, tileB->tileRowPtr.data(), tileB->tileRowPtr.size() * sizeof(int)   , cudaMemcpyHostToDevice);
    cudaMemcpy(&dtileColIdxB, tileB->tileColIdx.data(), tileB->tileColIdx.size() * sizeof(int)   , cudaMemcpyHostToDevice);
    cudaMemcpy(  &dtileNnzsB, tileB->tileNnzs  .data(), tileB->tileNnzs.size() * sizeof(int)     , cudaMemcpyHostToDevice);
    cudaMemcpy(    &drowPtrB, tileB->rowPtr    .data(), tileB->rowPtr.size() * sizeof(uint8_t)   , cudaMemcpyHostToDevice);
    cudaMemcpy( &drowcolIdxB, tileB->rowcolIdx .data(), tileB->rowcolIdx.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(      &dvalsB, tileB->vals      .data(), tileB->vals.size() * sizeof(double)      , cudaMemcpyHostToDevice);
    cudaMemcpy(     &dmasksB, tileB->masks     .data(), tileB->masks.size() * sizeof(uint16_t)   , cudaMemcpyHostToDevice);

    int *dtileRowPtrC, *dtileColIdxC;
    _spgemm_symbolic(tileA->tileRows, tileA->tileCols, tileA->tileNnz,
                     tileB->tileRows, tileB->tileCols, tileB->tileNnz,
                     dtileRowPtrA, dtileColIdxA, dtileRowPtrB, dtileColIdxB,
                     &dtileRowPtrC, &dtileColIdxC);
    
    

    return nullptr;
}

}

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return;                                                                \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return;                                                                \
    }                                                                          \
}

__global__ void _fill(float* arr0, int len0, float* arr1, int len1, float val) {
    int tx = threadIdx.x; int bx = blockIdx.x;
    int id = bx*1024+tx;
    if (id < len0)
        arr0[id] = val;
    if (id < len1)
        arr1[id] = val;
}

void _spgemm_symbolic(int rowsA, int colsA, int nnzA, int rowsB, int colsB, int nnzB,
    int *d_rowPtrA, int *d_colIdxA, int *d_rowPtrB, int *d_colIdxB, int **d_rowPtrC, int **d_colIdxC) {
#ifdef PROFILE
    nvtxNameOsThread(pthread_self(), "MAIN_THREAD");
    nvtxRangePushA("_spgemm_symbolic");
    nvtxRangePushA("_spgemm_symbolic_setup");
#endif
    float              alpha       = 1.0f;
    float              beta        = 0.0f;
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_32F;

    // Allocate
    float *d_dataValA, *d_dataValB, *d_dataValC;
    size_t
    rowPtrA_size  = (rowsA+1) * sizeof(int),
    //dataColA_size = (nnzA) * sizeof(int),
    dataValA_size = (nnzA) * sizeof(float),
    //rowPtrB_size  = (rowsB+1) * sizeof(int),
    //dataColB_size = (nnzB) * sizeof(int),
    dataValB_size = (nnzB) * sizeof(float);
    
    cudaMalloc(d_rowPtrC  , rowPtrA_size );
    cudaMalloc(&d_dataValA, dataValA_size);
    cudaMalloc(&d_dataValB, dataValB_size);

    _fill<<<(std::max(nnzA,nnzB)+1023)/1024,1024>>>(d_dataValA, nnzA, d_dataValB, nnzB, 1.0f);

    //--------------------------------------------------------------------------
    // Cusparse Setup
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, rowsA, colsA, nnzA,
                                      d_rowPtrA, d_colIdxA, d_dataValA,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, rowsB, colsB, nnzB,
                                      d_rowPtrB, d_colIdxB, d_dataValB,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, rowsA, colsB, 0,
                                      NULL, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )
    //--------------------------------------------------------------------------

#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("_spgemm_symbolic_compute");
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
    nvtxRangePushA("_spgemm_symbolic_deardown");
#endif
    // get matrix C non-zero entries C_nnz
    int64_t C_num_rows1, C_num_cols1, C_nnz;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz) )
    // allocate matrix C
    CHECK_CUDA( cudaMalloc((void**) d_colIdxC, C_nnz * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &d_dataValC,  C_nnz * sizeof(float)) )

    // NOTE: if 'beta' != 0, the values of C must be update after the allocation
    //       of dC_values, and before the call of cusparseSpGEMM_copy
    // update matC with the new pointers
    CHECK_CUSPARSE( cusparseCsrSetPointers(matC, *d_rowPtrC, d_colIdxC, d_dataValC) )

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

    cudaFree(dBuffer1);
    cudaFree(dBuffer2);
    cudaFree(d_dataValA);
    cudaFree(d_dataValB);
    cudaFree(d_dataValC);
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePop();
#endif
}