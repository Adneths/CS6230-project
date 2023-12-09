
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#include "SpGEMM_util.h"

namespace cuda {
__global__ void cudaCumsumI(int* in, int* out, int* blocks, int len) {
	__shared__ int sBlock[CUMSUM_BLOCK_SIZE];

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int tx = threadIdx.x;

	if (id < len)
		sBlock[tx] = in[id];
	else
		sBlock[tx] = 0;
	__syncthreads();

	for (int stride = 1; stride <= CUMSUM_BLOCK_SIZE; stride *= 2) {
		int index = (tx + 1) * stride * 2 - 1;
		if (index < CUMSUM_BLOCK_SIZE)
		{
			sBlock[index] += sBlock[index - stride];
		}
		__syncthreads();
	}

	for (int stride = CUMSUM_BLOCK_SIZE / 2; stride > 0; stride /= 2) {
		int index = (tx + 1) * stride * 2 - 1;
		if (index + stride < CUMSUM_BLOCK_SIZE)
		{
			sBlock[index + stride] += sBlock[index];
		}
		__syncthreads();
	}

	if (id < len)
		out[id] = sBlock[tx];
	if (tx == 0)
		blocks[blockIdx.x] = sBlock[CUMSUM_BLOCK_SIZE - 1];
}
__global__ void cudaCumsumI(int* in, int* out, int len) {
	__shared__ int sBlock[CUMSUM_BLOCK_SIZE];

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int tx = threadIdx.x;

	if (id < len)
		sBlock[tx] = in[id];
	else
		sBlock[tx] = 0;
	__syncthreads();

	for (int stride = 1; stride <= CUMSUM_BLOCK_SIZE; stride *= 2) {
		int index = (tx + 1) * stride * 2 - 1;
		if (index < CUMSUM_BLOCK_SIZE)
		{
            sBlock[index] += sBlock[index - stride];
		}
		__syncthreads();
	}

	for (int stride = CUMSUM_BLOCK_SIZE / 2; stride > 0; stride /= 2) {
		int index = (tx + 1) * stride * 2 - 1;
		if (index + stride < CUMSUM_BLOCK_SIZE)
		{
			sBlock[index + stride] += sBlock[index];
		}
		__syncthreads();
	}

	if (id < len)
		out[id] = sBlock[tx];
}
__global__ void cudaFillSumI(int* inout, int* blocks, size_t len) {
	__shared__ int sum;
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (threadIdx.x == 0)
		if (blockIdx.x > 0)
			sum = blocks[blockIdx.x-1];
		else
			sum = 0;
	__syncthreads();

	if (id < len)
	{
		inout[id] += sum;
	}
}
void cumsumI(int* in, int* out, size_t len) {
	if (len > CUMSUM_BLOCK_SIZE)
	{
		size_t blockDim = (len + CUMSUM_BLOCK_SIZE - 1) / CUMSUM_BLOCK_SIZE;
		int* blocks;
		cudaMalloc(&blocks, sizeof(int) * blockDim);
		cudaCumsumI<<<blockDim, CUMSUM_BLOCK_SIZE>>>(in, out, blocks, len);
		cumsumI(blocks, blocks, blockDim);
		cudaFillSumI<<<blockDim, CUMSUM_BLOCK_SIZE>>>(out, blocks, len);
		cudaFree(blocks);
	}
	else
		cudaCumsumI<<<1, CUMSUM_BLOCK_SIZE>>>(in, out, len);
}
}