#include <iostream>
#include <random>
#include <chrono>

__device__ void warpReduce(volatile int * sdata, int tIdx) {
    sdata[tIdx] += sdata[tIdx + 32];
    sdata[tIdx] += sdata[tIdx + 16];
    sdata[tIdx] += sdata[tIdx + 8];
    sdata[tIdx] += sdata[tIdx + 4];
    sdata[tIdx] += sdata[tIdx + 2];
    sdata[tIdx] += sdata[tIdx + 1];
}

__global__ void ker_4(const int * g_in, int * g_out) {
	// prepare shared data allocated by kernel invocation
	extern __shared__ int sdata[];

	// load data into shared memory
	sdata[threadIdx.x] = g_in[threadIdx.x + blockIdx.x*blockDim.x];
	__syncthreads();

	for (unsigned int i = blockDim.x/2; i > 32; i >>= 1) {
		if (threadIdx.x < i) {
			sdata[threadIdx.x] += sdata[threadIdx.x+i];
		}
		__syncthreads();
	}

    if (threadIdx.x < 32) warpReduce(sdata, threadIdx.x);

	// export result to global memory
	if (threadIdx.x == 0) {
		g_out[blockIdx.x] = sdata[0];
	}
}
 

int main () {
	// set the parallelisation parameters
	const int numThreadsPerBlock = 512;
    const int numBlocks = 47;

    // array size
	const int N = numThreadsPerBlock*numBlocks;

	// allocate memory on host for input and output
	int * h_in = (int*)malloc(sizeof(int)*N);
    int * h_out = (int*)malloc(sizeof(int));

	// fill array with 1s
	for (int i = 0; i < N; ++i) {
		h_in[i] = 1;
	}

	// allocate and populate memory on the device
	int * d_in, * d_out;
	cudaMalloc(&d_in, sizeof(int)*N);
	cudaMemcpy(d_in, h_in, sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMalloc(&d_out, sizeof(int)*numBlocks); // does not need to be initialized


    // DEBUG KERNEL 4
    for (unsigned int i = 0; i < 1000; i++) {
        ker_4 <<< numBlocks, numThreadsPerBlock, sizeof(int)*numThreadsPerBlock >>> (d_in, d_out);
        cudaDeviceSynchronize();
        cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_out[0] != numThreadsPerBlock)
            std::cout << h_out[0] << std::endl;
    }
}
