#pragma once

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


// optimisation: avoid divergent branching
template<typename T>
__global__ void ker_1(const T * g_in, T * g_out) {
	// prepare shared data allocated by kernel invocation
	extern __shared__ T sdata[];

	// load data into shared memory
	sdata[threadIdx.x] = g_in[threadIdx.x + blockIdx.x*blockDim.x];
	__syncthreads();

	for (unsigned int i = 1; i < blockDim.x; i *= 2) {
        int idx = 2*i*threadIdx.x;
		if (idx < blockDim.x) {
			sdata[idx] += sdata[idx+i];
		}
		__syncthreads();
	}

	// export result to global memory
	if (threadIdx.x == 0) {
		g_out[blockIdx.x] = sdata[0];
	}
}


// optimisation: shared memory bank conficts
template<typename T>
__global__ void ker_2(const T * g_in, T * g_out) {
	// prepare shared data allocated by kernel invocation
	extern __shared__ T sdata[];

	// load data into shared memory
	sdata[threadIdx.x] = g_in[threadIdx.x + blockIdx.x*blockDim.x];
	__syncthreads();

	for (unsigned int i = blockDim.x/2; i > 0; i >>= 1) {
		if (threadIdx.x < i) {
			sdata[threadIdx.x] += sdata[threadIdx.x+i];
		}
		__syncthreads();
	}

	// export result to global memory
	if (threadIdx.x == 0) {
		g_out[blockIdx.x] = sdata[0];
	}
}


// optimisation: avoid idle threads
// WARNING: Must be called with half amount of blocks
template<typename T>
__global__ void ker_3(const T * g_in, T * g_out) {
	// prepare shared data allocated by kernel invocation
	extern __shared__ T sdata[];

	// load data into shared memory
	sdata[threadIdx.x] = g_in[threadIdx.x + blockIdx.x*blockDim.x*2]
                        + g_in[threadIdx.x + blockIdx.x*blockDim.x*2 + blockDim.x];
	__syncthreads();

	for (unsigned int i = blockDim.x/2; i > 0; i >>= 1) {
		if (threadIdx.x < i) {
			sdata[threadIdx.x] += sdata[threadIdx.x+i];
		}
		__syncthreads();
	}

	// export result to global memory
	if (threadIdx.x == 0) {
		g_out[blockIdx.x] = sdata[0];
	}
}


// optimisation: unroll the warp
template<typename T>
__device__ void warpReduce(volatile T * sdata, int tIdx) {
    sdata[tIdx] += sdata[tIdx + 32];
    sdata[tIdx] += sdata[tIdx + 16];
    sdata[tIdx] += sdata[tIdx + 8];
    sdata[tIdx] += sdata[tIdx + 4];
    sdata[tIdx] += sdata[tIdx + 2];
    sdata[tIdx] += sdata[tIdx + 1];
}

template<typename T>
__global__ void ker_4(const T * g_in, T * g_out) {
	// prepare shared data allocated by kernel invocation
	extern __shared__ T sdata[];

	// load data into shared memory
	sdata[threadIdx.x] = g_in[threadIdx.x + blockIdx.x*blockDim.x*2]
                        + g_in[threadIdx.x + blockIdx.x*blockDim.x*2 + blockDim.x];
	__syncthreads();

	for (unsigned int i = blockDim.x/2; i > 32; i >>= 1) {
		if (threadIdx.x < i) {
			sdata[threadIdx.x] += sdata[threadIdx.x+i];
		}
		__syncthreads();
	}

    warpReduce<T>(sdata, threadIdx.x);

	// export result to global memory
	if (threadIdx.x == 0) {
		g_out[blockIdx.x] = sdata[0];
	}
}


// optimisation: unroll for loop using templates
// WARNING: This function will give false results for blockSize of 2048 or larger
template<typename T, unsigned int blockSize>
__device__ void warpReduce(volatile T * sdata, int tIdx) {
    if (blockSize >= 64) sdata[tIdx] += sdata[tIdx + 32];
    if (blockSize >= 32) sdata[tIdx] += sdata[tIdx + 16];
    if (blockSize >= 16) sdata[tIdx] += sdata[tIdx + 8];
    if (blockSize >= 8) sdata[tIdx] += sdata[tIdx + 4];
    if (blockSize >= 4) sdata[tIdx] += sdata[tIdx + 2];
    if (blockSize >= 2) sdata[tIdx] += sdata[tIdx + 1];
}

template<typename T, unsigned int blockSize>
__global__ void ker_5(const T * g_in, T * g_out) {
	// prepare shared data allocated by kernel invocation
	extern __shared__ T sdata[];

	// load data into shared memory
	sdata[threadIdx.x] = g_in[threadIdx.x + blockIdx.x*blockDim.x*2]
                        + g_in[threadIdx.x + blockIdx.x*blockDim.x*2 + blockDim.x];
	__syncthreads();

    // unrolled for loop
    if (blockSize >= 1024)
        if (threadIdx.x < 512) {sdata[threadIdx.x] += sdata[threadIdx.x + 512]; __syncthreads();}
    if (blockSize >= 512)
        if (threadIdx.x < 256) {sdata[threadIdx.x] += sdata[threadIdx.x + 256]; __syncthreads();}
    if (blockSize >= 256)
        if (threadIdx.x < 128) {sdata[threadIdx.x] += sdata[threadIdx.x + 128]; __syncthreads();}
    if (blockSize >= 128)
        if (threadIdx.x < 64) {sdata[threadIdx.x] += sdata[threadIdx.x + 64]; __syncthreads();}

    warpReduce<T>(sdata, threadIdx.x);

	// export result to global memory
	if (threadIdx.x == 0) {
		g_out[blockIdx.x] = sdata[0];
	}
}

