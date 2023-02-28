#include "../headers/kernels.cuh"

template<typename T>
__global__ void ker_0(const T * g_in, T * g_out) {
	// prepare shared data allocated by kernel invocation
	extern __shared__ T sdata[];

	// load data into shared memory
	sdata[threadIdx.x] = g_in[threadIdx.x + blockIdx.x*blockDim.x];
	__syncthreads();

	// do treereduction in interleaved addressing style
	for (unsigned int i = 1; i < blockDim.x; i *= 2) {
		if (threadIdx.x % (2*i) == 0) {
			sdata[threadIdx.x] += sdata[threadIdx.x+i];
		}
		__syncthreads();
	}

	// export result to global memory
	if (threadIdx.x == 0) {
		g_out[blockIdx.x] = sdata[0];
	}
}
// PROBLEMS:
// Divergent branching, threads in warp should follow the same branching
// Solution: Change in addressing
