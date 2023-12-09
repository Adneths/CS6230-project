
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
    int *d_rowPtrA, int *d_colIdxA, int *d_rowPtrB, int *d_colIdxB, int **d_rowPtrC, int **d_colIdxC, int64_t *C_nnz);

__device__ inline int binary_search(int* arr, int start, int end, int target) {
    while (start < end) {
        int mid = (start + end)/2;
        if (arr[mid] == target) {
            return mid;
        }
        if (arr[mid] > target)
            end = mid;
        else
            start = mid+1;
    }
    return -1;
}
__device__ inline int binary_search_between(int* arr, int start, int end, int target) {
    while (start < end) {
        int mid = (start + end)/2;
        if (arr[mid] <= target && arr[mid+1] > target) {
            return mid;
        }
        if (arr[mid] > target)
            end = mid;
        else
            start = mid+1;
    }
    return -1;
}
__device__ inline int decode(uint8_t rowcol) {
    return (rowcol>>4)*16 + (rowcol&0x0f);
}

__global__ void tile_spgemm_cuda(int S, int *tileRowPtrC, int *tileColIdxC, int len,
    int* tileRowPtrA, int* tileColIdxA, int* tileNnzsA, uint8_t* rowPtrA, uint8_t* rowcolIdxA, double* valsA, uint16_t* masksA,
    int* tileColPtrB, int* tileRowIdxB, int* tileNnzsB, uint8_t* colPtrB, uint8_t* colrowIdxB, double* valsB, uint16_t* masksB)
{
    __shared__ int matchA[1024];
    __shared__ int matchB[1024];
    __shared__ double denseA[256];
    __shared__ double denseBT[256];
    __shared__ double denseC[256];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;

    // While tile-row is this tile-nnz in
    int trow = binary_search_between(tileRowPtrC, 0, len, bx*S);
    for (int n = 0; n < S; i++) {
        if (tx == 0 && ty == 0) {
            int pos = 0;
            int idC = bx*S+n;
            // If outside of current row rebinary search starting at current tile-row
            if (tileRowPtrC[trow+1] <= idC)
                trow = binary_search_between(tileRowPtrC, trow, len, idC);
            int tcol = tileColIdxC[idC];

            if (tileRowPtrA[trow+1] - tileRowPtrA[trow] < tileColPtrB[tcol+1] - tileColPtrB[tcol]) {
                for (int i = tileRowPtrA[trow]; i < tileRowPtrA[trow+1]; i++) {
                    int match = 
                        binary_search(tileRowIdxB, tileRowIdxB[tileColPtrB[tcol]], tileRowIdxB[tileColPtrB[tcol]+1], tileColIdxA[i]);
                    if (match != -1) {
                        matchA[pos] = i;
                        matchB[pos] = match;
                        atomicAdd(&pos, 1);
                    }
                }
            }
            else {
                for (int i = tileColPtrB[tcol]; i < tileColPtrB[tcol+1]; i++) {
                    int match = 
                        binary_search(tileColIdxA, tileColIdxA[tileRowPtrA[trow]], tileColIdxA[tileRowPtrA[trow]+1], tileRowIdxB[i]);
                    if (match != -1) {
                        matchA[pos] = match;
                        matchB[pos] = i;
                        atomicAdd(&pos, 1);
                    }
                }
            }
        }
        // Mask

        int id = ty*16+tx;
        denseC[id] = 0;
        for (int i = 0; i < pos; i++) {
            denseA[id] = 0; denseBT[id] = 0;
            int s = rowPtrA[matchA[i]]; int e = tileNnzsA[matchA[i]+1];
            if (id+s < e)
                denseA[decode(rowcolIdxA[id+s])] = valsA[id+s];
            s = colPtrB[matchA[i]]; e = tileNnzsB[matchA[i]+1];
            if (id+s < e)
                denseB[decode(colrowIdxB[id+s])] = valsB[id+s];

            for (int j = 0; j < 16; j++) {
                denseC[id] = denseA[ty*16+j] * denseBT[tx*16+j];
            }
        }
    }
}

namespace cuda {

CSRMatrix<double>* tile_spgemm(CSRMatrix<double>* A, CSRMatrix<double>* B) {
    TileSpMatrix<double>* tileA = new TileSpMatrix<double>(A);
    B->transpose();
    TileSpMatrix<double>* tileB = new TileSpMatrix<double>(B);
    //std::cout << tileA << std::endl;
    
    int *dtileRowPtrA, /**dtileRowIdxA,*/ *dtileColIdxA, *dtileNnzsA;
    uint8_t drowPtrA, drowcolIdxA;
    double *dvalsA;
    uint16_t *dmasksA;
    int *dtileRowPtrB, /**dtileRowIdxB,*/ *dtileColIdxB, *dtileNnzsB;
    uint8_t drowPtrB, drowcolIdxB;
    double *dvalsB;
    uint16_t *dmasksB;
    cudaMalloc((void**)&dtileRowPtrA, tileA->tileRowPtr.size() * sizeof(int)   );
    //cudaMalloc((void**)&dtileRowIdxA, tileA->tileRowIdx.size() * sizeof(int)   );
    cudaMalloc((void**)&dtileColIdxA, tileA->tileColIdx.size() * sizeof(int)   );
    cudaMalloc((void**)  &dtileNnzsA, tileA->tileNnzs.size() * sizeof(int)     );
    cudaMalloc((void**)    &drowPtrA, tileA->rowPtr.size() * sizeof(uint8_t)   );
    cudaMalloc((void**) &drowcolIdxA, tileA->rowcolIdx.size() * sizeof(uint8_t));
    cudaMalloc((void**)      &dvalsA, tileA->vals.size() * sizeof(double)      );
    cudaMalloc((void**)     &dmasksA, tileA->masks.size() * sizeof(uint16_t)   );
    cudaMalloc((void**)&dtileRowPtrB, tileB->tileRowPtr.size() * sizeof(int)   );
    //cudaMalloc((void**)&dtileRowIdxB, tileB->tileRowIdx.size() * sizeof(int)   );
    cudaMalloc((void**)&dtileColIdxB, tileB->tileColIdx.size() * sizeof(int)   );
    cudaMalloc((void**)  &dtileNnzsB, tileB->tileNnzs.size() * sizeof(int)     );
    cudaMalloc((void**)    &drowPtrB, tileB->rowPtr.size() * sizeof(uint8_t)   );
    cudaMalloc((void**) &drowcolIdxB, tileB->rowcolIdx.size() * sizeof(uint8_t));
    cudaMalloc((void**)      &dvalsB, tileB->vals.size() * sizeof(double)      );
    cudaMalloc((void**)     &dmasksB, tileB->masks.size() * sizeof(uint16_t)   );

    cudaMemcpy(&dtileRowPtrA, tileA->tileRowPtr.data(), tileA->tileRowPtr.size() * sizeof(int)   , cudaMemcpyHostToDevice);
    //cudaMemcpy(&dtileRowIdxA, tileA->tileRowIdx.data(), tileA->tileColIdx.size() * sizeof(int)   , cudaMemcpyHostToDevice);
    cudaMemcpy(&dtileColIdxA, tileA->tileColIdx.data(), tileA->tileColIdx.size() * sizeof(int)   , cudaMemcpyHostToDevice);
    cudaMemcpy(  &dtileNnzsA, tileA->tileNnzs  .data(), tileA->tileNnzs.size() * sizeof(int)     , cudaMemcpyHostToDevice);
    cudaMemcpy(    &drowPtrA, tileA->rowPtr    .data(), tileA->rowPtr.size() * sizeof(uint8_t)   , cudaMemcpyHostToDevice);
    cudaMemcpy( &drowcolIdxA, tileA->rowcolIdx .data(), tileA->rowcolIdx.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(      &dvalsA, tileA->vals      .data(), tileA->vals.size() * sizeof(double)      , cudaMemcpyHostToDevice);
    cudaMemcpy(     &dmasksA, tileA->masks     .data(), tileA->masks.size() * sizeof(uint16_t)   , cudaMemcpyHostToDevice);

    cudaMemcpy(&dtileRowPtrB, tileB->tileRowPtr.data(), tileB->tileRowPtr.size() * sizeof(int)   , cudaMemcpyHostToDevice);
    //cudaMemcpy(&dtileRowIdxB, tileB->tileRowIdx.data(), tileB->tileColIdx.size() * sizeof(int)   , cudaMemcpyHostToDevice);
    cudaMemcpy(&dtileColIdxB, tileB->tileColIdx.data(), tileB->tileColIdx.size() * sizeof(int)   , cudaMemcpyHostToDevice);
    cudaMemcpy(  &dtileNnzsB, tileB->tileNnzs  .data(), tileB->tileNnzs.size() * sizeof(int)     , cudaMemcpyHostToDevice);
    cudaMemcpy(    &drowPtrB, tileB->rowPtr    .data(), tileB->rowPtr.size() * sizeof(uint8_t)   , cudaMemcpyHostToDevice);
    cudaMemcpy( &drowcolIdxB, tileB->rowcolIdx .data(), tileB->rowcolIdx.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(      &dvalsB, tileB->vals      .data(), tileB->vals.size() * sizeof(double)      , cudaMemcpyHostToDevice);
    cudaMemcpy(     &dmasksB, tileB->masks     .data(), tileB->masks.size() * sizeof(uint16_t)   , cudaMemcpyHostToDevice);

    int *dtileRowPtrC, *dtileColIdxC;
    int64_t C_nnz;
    _spgemm_symbolic(tileA->tileRows, tileA->tileCols, tileA->tileNnz,
                     tileB->tileRows, tileB->tileCols, tileB->tileNnz,
                     dtileRowPtrA, dtileColIdxA, dtileRowPtrB, dtileColIdxB,
                     &dtileRowPtrC, &dtileColIdxC, &C_nnz);
    
    const int S = 4;
    dim3 threadPerBlock(16,16,1);
    dim3 blocks((C_nnz+S-1)/S);
    <<<blocks,threadsPerBlock>>>

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
    int *d_rowPtrA, int *d_colIdxA, int *d_rowPtrB, int *d_colIdxB, int **d_rowPtrC, int **d_colIdxC, int64_t *C_nnz) {
#ifdef PROFILE
    nvtxNameOsThread(pthread_self(), "MAIN_THREAD");
    nvtxRangePushA("_spgemm_symbolic");
    nvtxRangePushA("_spgemm_symbolic_setup");
#endif
    float               alpha       = 1.0f;
    float               beta        = 0.0f;
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
    CHECK_CUSPARSE( cusparseCreateCsc(&matB, rowsB, colsB, nnzB,
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
    int64_t C_num_rows1, C_num_cols1;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, C_nnz) )
    // allocate matrix C
    CHECK_CUDA( cudaMalloc((void**) d_colIdxC, *C_nnz * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &d_dataValC,  *C_nnz * sizeof(float)) )

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