
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#include "SpGEMM_cuda.h"

#ifdef PROFILE
#include <nvtx3/nvToolsExtCuda.h>
#endif

namespace cuda {

__inline__ __device__ int warpAllReduceMax(int val) {
#pragma unroll
	for (int mask = warpSize / 2; mask > 0; mask /= 2)
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask));
	return val;
}
__inline__ __device__ int blockReduceMax(int val, int resultantWarp) {
	static __shared__ int shared[32];
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpAllReduceMax(val);
	if (lane == 0) shared[wid] = val;

	__syncthreads();

	if (wid == resultantWarp) {
		val = (lane < blockDim.x / warpSize) ? shared[lane] : 0;
		val = warpAllReduceMax(val);
	}
	return val;
}

__global__ void spgemm_preprocess(int* maxRow, int* rowPtr, int len) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < len-1) {
        int val = rowPtr[id+1] - rowPtr[id];
        val = blockReduceMax(val, 0);
        if (id == 0)
            *maxRow = val;
    }
}

// Sparse General Matrix Multiplication with Dense Accumulator
__global__ void spgemm_dacc(int rowsA, int colsA, int* rowPtrA, int* dataColA, double* dataValA,
                        int rowsB, int colsB, int* rowPtrB, int* dataColB, double* dataValB,
                        double* dataValC) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    if (tx < colsB) {
        int as = rowPtrA[bx]; int ae = rowPtrA[bx+1];
        for (int i = as; i < ae; i++) {
            double valA = dataValA[i]; int c = dataColA[i];

            int bs = rowPtrB[c]; int be = rowPtrB[c+1];
            while (bs < be) {
                if (bs + tx < be) { // Prevent over reading of B
                    dataValC[bx * colsB + dataColB[bs + tx]] += valA * dataValB[bs + tx];
                }
                bs += blockDim.x;
            }
            __syncthreads();
        }
    }
}

CSRMatrix<double>* spgemm(CSRMatrix<double>* A, CSRMatrix<double>* B) {
#ifdef PROFILE
    nvtxNameOsThread(pthread_self(), "MAIN_THREAD");
    nvtxRangePushA("CUDA_spgemm");
#endif
    /*if (A->rows > 1024 || B->cols > 1024) {
        printf("Error: Does not support resultant matrices of size greater than 1024x1024\n");
        return nullptr;
    }*/
#ifdef PROFILE
    nvtxRangePushA("CUDA_spgemm_cudamalloc");
#endif
    int *d_rowPtrA, *d_dataColA, *d_rowPtrB, *d_dataColB;
    double *d_dataValA, *d_dataValB, *d_dataValC, *h_dataValC;
    int* d_maxRow; int maxRow;

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

    cudaMalloc(&d_maxRow, sizeof(int));

#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_HtoD");
#endif
    cudaMemcpy(d_rowPtrA , A->rowPtr , rowPtrA_size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataColA, A->dataCol, dataColA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataValA, A->dataVal, dataValA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtrB , B->rowPtr , rowPtrB_size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataColB, B->dataCol, dataColB_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataValB, B->dataVal, dataValB_size, cudaMemcpyHostToDevice);
    cudaMemset(d_dataValC, 0, dataValC_size);
    
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_preprocess");
#endif

    dim3 threadsPerBlock(std::max(32*((B->rows+31)/32),1024));
    dim3 numBlocks((B->rows+threadsPerBlock.x-1)/threadsPerBlock.x);
    spgemm_preprocess<<<numBlocks,threadsPerBlock>>>(d_maxRow, d_rowPtrB, B->rows+1);
    cudaMemcpy(&maxRow, d_maxRow, sizeof(int), cudaMemcpyDeviceToHost);
    //maxRow = B->cols;

#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_compute");
#endif
    threadsPerBlock = dim3(32*((maxRow+31)/32));
    numBlocks = dim3(A->rows);
    spgemm_dacc<<<numBlocks,threadsPerBlock>>>(A->rows, A->cols, d_rowPtrA, d_dataColA, d_dataValA,
                                               B->rows, B->cols, d_rowPtrB, d_dataColB, d_dataValB,
                                               d_dataValC);
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_malloc");
#endif
    h_dataValC = (double*) malloc(dataValC_size);
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_DtoH");
#endif
    cudaMemcpy(h_dataValC, d_dataValC, dataValC_size, cudaMemcpyDeviceToHost);
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_reconstruct");
#endif
    CSRMatrix<double>* ret = new CSRMatrix<double>(A->rows, B->cols, h_dataValC);
#ifdef PROFILE
    nvtxRangePop();
    nvtxRangePushA("CUDA_spgemm_cudafree");
#endif

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
