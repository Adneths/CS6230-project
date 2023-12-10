#ifndef SPGEMM_UTIL_H
#define SPGEMM_UTIL_H

namespace cuda {
#define CUMSUM_BLOCK_SIZE 1024

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
__inline__ __device__ int warpAllReduceAdd(int val) {
#pragma unroll
	for (int mask = warpSize / 2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xffffffff, val, mask);
	return val;
}
__inline__ __device__ int blockReduceAdd(int val, int resultantWarp) {
	static __shared__ int shared[32];
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpAllReduceAdd(val);
	if (lane == 0) shared[wid] = val;

	__syncthreads();

	if (wid == resultantWarp) {
		val = (lane < blockDim.x / warpSize) ? shared[lane] : 0;
		val = warpAllReduceAdd(val);
	}
	return val;
}
void cumsumI(int* in, int* out, size_t len);

__inline__ void syncDevice(int N) {
	for (int i = 0; i < N; i++) {
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}
}
}
#endif
