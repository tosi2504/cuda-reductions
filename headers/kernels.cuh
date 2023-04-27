#pragma once

#define DIV_UP(val, div) ((val + div - 1)/(div))
#define ROUND_UP(val, div) (DIV_UP(val, div)*div)

#include <iostream>

#define CCE(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CCEL() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

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

// WARNING: assuming the bit layout of the one-element of T wrt T + T to be all zeros!
template<typename T>
void wrapperKer_0(const T * d_in, T * d_temp1, T * d_temp2, T ** d_out, const int N, const int numThreadsPerBlock) {
    // all buffers must be allocated before-hand with sizes:
    // d_in : ROUND_UP(N, numThreadsPerBlock)
    // d_out : 1
    // d_temp1 : ROUND_UP(DIV_UP(N, numThreadsPerBlock), numThreadsPerBlock) = ROUND_UP(numBlocks1, numThreadsPerBlock)
    // d_temp2 :  ROUND_UP(DIV_UP(numBlocks1, numThreadsPerBlock), numThreadsPerBlock) = ROUND_UP(numBlocks2, numThreadsPerBlock)

    // Note: the first entry of d_temp1 serves as output

    // first iteration
    int numBlocks = DIV_UP(N, numThreadsPerBlock);
    CCE(cudaMemset(d_temp1, 0, ROUND_UP(numBlocks, numThreadsPerBlock)*sizeof(T)));
    ker_0 < T > <<< numBlocks , numThreadsPerBlock , sizeof(T) * numThreadsPerBlock >>> (d_in, d_temp1);
    CCEL();
    
    // do recursive reduction
    while (numBlocks > 1) {
        // switch buffers
        T * temp = d_temp1;
        d_temp1 = d_temp2;
        d_temp2 = temp;

        // calculate new numBlocks
        numBlocks = DIV_UP(numBlocks, numThreadsPerBlock);

        // perpare the output buffer
        CCE(cudaMemset(d_temp1, 0, ROUND_UP(numBlocks, numThreadsPerBlock)*sizeof(T)));

        // invoke the kernel
        ker_0 < T > <<< numBlocks , numThreadsPerBlock , sizeof(T) * numThreadsPerBlock >>> (d_temp2, d_temp1);
        CCEL();
    }
    *d_out = d_temp1;
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

template<typename T>
void wrapperKer_1(const T * d_in, T * d_temp1, T * d_temp2, T ** d_out, const int N, const int numThreadsPerBlock) {
    // all buffers must be allocated before-hand with sizes:
    // d_in : ROUND_UP(N, numThreadsPerBlock)
    // d_out : 1
    // d_temp1 : ROUND_UP(DIV_UP(N, numThreadsPerBlock), numThreadsPerBlock) = ROUND_UP(numBlocks1, numThreadsPerBlock)
    // d_temp2 :  ROUND_UP(DIV_UP(numBlocks1, numThreadsPerBlock), numThreadsPerBlock) = ROUND_UP(numBlocks2, numThreadsPerBlock)

    // Note: the first entry of d_temp1 serves as output

    // first iteration
    int numBlocks = DIV_UP(N, numThreadsPerBlock);
    CCE(cudaMemset(d_temp1, 0, ROUND_UP(numBlocks, numThreadsPerBlock)*sizeof(T)));
    ker_1 < T > <<< numBlocks , numThreadsPerBlock , sizeof(T) * numThreadsPerBlock >>> (d_in, d_temp1);
    CCEL();
    
    // do recursive reduction
    while (numBlocks > 1) {
        // switch buffers
        T * temp = d_temp1;
        d_temp1 = d_temp2;
        d_temp2 = temp;

        // calculate new numBlocks
        numBlocks = DIV_UP(numBlocks, numThreadsPerBlock);

        // perpare the output buffer
        CCE(cudaMemset(d_temp1, 0, ROUND_UP(numBlocks, numThreadsPerBlock)*sizeof(T)));

        // invoke the kernel
        ker_1 < T > <<< numBlocks , numThreadsPerBlock , sizeof(T) * numThreadsPerBlock >>> (d_temp2, d_temp1);
        CCEL();
    }
    *d_out = d_temp1;
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


template<typename T>
void wrapperKer_2(const T * d_in, T * d_temp1, T * d_temp2, T ** d_out, const int N, const int numThreadsPerBlock) {
    // all buffers must be allocated before-hand with sizes:
    // d_in : ROUND_UP(N, numThreadsPerBlock)
    // d_out : 1
    // d_temp1 : ROUND_UP(DIV_UP(N, numThreadsPerBlock), numThreadsPerBlock) = ROUND_UP(numBlocks1, numThreadsPerBlock)
    // d_temp2 :  ROUND_UP(DIV_UP(numBlocks1, numThreadsPerBlock), numThreadsPerBlock) = ROUND_UP(numBlocks2, numThreadsPerBlock)

    // Note: the first entry of d_temp1 serves as output

    // first iteration
    int numBlocks = DIV_UP(N, numThreadsPerBlock);
    CCE(cudaMemset(d_temp1, 0, ROUND_UP(numBlocks, numThreadsPerBlock)*sizeof(T)));
    ker_2 < T > <<< numBlocks , numThreadsPerBlock , sizeof(T) * numThreadsPerBlock >>> (d_in, d_temp1);
    CCEL();
    
    // do recursive reduction
    while (numBlocks > 1) {
        // switch buffers
        T * temp = d_temp1;
        d_temp1 = d_temp2;
        d_temp2 = temp;

        // calculate new numBlocks
        numBlocks = DIV_UP(numBlocks, numThreadsPerBlock);

        // perpare the output buffer
        CCE(cudaMemset(d_temp1, 0, ROUND_UP(numBlocks, numThreadsPerBlock)*sizeof(T)));

        // invoke the kernel
        ker_2 < T > <<< numBlocks , numThreadsPerBlock , sizeof(T) * numThreadsPerBlock >>> (d_temp2, d_temp1);
        CCEL();
    }
    *d_out = d_temp1;
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

template<typename T>
void wrapperKer_3(const T * d_in, T * d_temp1, T * d_temp2, T ** d_out, const int N, const int numThreadsPerBlock) {
    // all buffers must be allocated before-hand with sizes:
    // d_in : ROUND_UP(N, 2*numThreadsPerBlock) NOTE: we need 2*numThreadsPerBlock for kernel 3!!!
    // d_out : 1
    // d_temp1 : ROUND_UP(DIV_UP(N, 2*numThreadsPerBlock), 2*numThreadsPerBlock) = ROUND_UP(numBlocks1, 2*numThreadsPerBlock)
    // d_temp2 :  ROUND_UP(DIV_UP(numBlocks1, 2*numThreadsPerBlock), 2*numThreadsPerBlock) = ROUND_UP(numBlocks2, 2*numThreadsPerBlock)

    // first iteration
    int numBlocks = DIV_UP(N, 2*numThreadsPerBlock); // we only need half the amount of blocks for kernel 3
    CCE(cudaMemset(d_temp1, 0, ROUND_UP(numBlocks, 2*numThreadsPerBlock)*sizeof(T)));
    ker_3 < T > <<< numBlocks , numThreadsPerBlock , sizeof(T) * numThreadsPerBlock >>> (d_in, d_temp1);
    CCEL();
    
    // do recursive reduction
    while (numBlocks > 1) {
        // switch buffers
        T * temp = d_temp1;
        d_temp1 = d_temp2;
        d_temp2 = temp;

        // calculate new numBlocks
        numBlocks = DIV_UP(numBlocks, 2*numThreadsPerBlock);

        // prepare the output buffer
        CCE(cudaMemset(d_temp1, 0, ROUND_UP(numBlocks, 2*numThreadsPerBlock)*sizeof(T)));

        // invoke the kernel
        ker_3 < T > <<< numBlocks , numThreadsPerBlock , sizeof(T) * numThreadsPerBlock >>> (d_temp2, d_temp1);
        CCEL();
    }
    *d_out = d_temp1;
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

    if (threadIdx.x < 32) warpReduce<T>(sdata, threadIdx.x);

	// export result to global memory
	if (threadIdx.x == 0) {
		g_out[blockIdx.x] = sdata[0];
	}
}

template<typename T>
void wrapperKer_4(const T * d_in, T * d_temp1, T * d_temp2, T ** d_out, const int N, const int numThreadsPerBlock) {
    // all buffers must be allocated before-hand with sizes:
    // d_in : ROUND_UP(N, 2*numThreadsPerBlock) NOTE: we need 2*numThreadsPerBlock for kernel 3!!!
    // d_out : 1
    // d_temp1 : ROUND_UP(DIV_UP(N, 2*numThreadsPerBlock), 2*numThreadsPerBlock) = ROUND_UP(numBlocks1, 2*numThreadsPerBlock)
    // d_temp2 :  ROUND_UP(DIV_UP(numBlocks1, 2*numThreadsPerBlock), 2*numThreadsPerBlock) = ROUND_UP(numBlocks2, 2*numThreadsPerBlock)

    // first iteration
    int numBlocks = DIV_UP(N, 2*numThreadsPerBlock); // we only need half the amount of blocks for kernel 3
    CCE(cudaMemset(d_temp1, 0, ROUND_UP(numBlocks, 2*numThreadsPerBlock)*sizeof(T)));

    ker_4 < T > <<< numBlocks , numThreadsPerBlock , sizeof(T) * numThreadsPerBlock >>> (d_in, d_temp1);
    CCEL();
    
    // do recursive reduction
    while (numBlocks > 1) {
        // switch buffers
        T * temp = d_temp1;
        d_temp1 = d_temp2;
        d_temp2 = temp;

        // calculate new numBlocks
        numBlocks = DIV_UP(numBlocks, 2*numThreadsPerBlock);

        // prepare the output buffer
        CCE(cudaMemset(d_temp1, 0, ROUND_UP(numBlocks, 2*numThreadsPerBlock)*sizeof(T)));

        // invoke the kernel
        ker_4 < T > <<< numBlocks , numThreadsPerBlock , sizeof(T) * numThreadsPerBlock >>> (d_temp2, d_temp1);
        CCEL();
    }
    *d_out = d_temp1;
}



// optimisation: unroll for loop using templates
// WARNING: This function will give false results for blockSize of 2048 or larger
template<typename T, unsigned int blockSize>
__device__ void warpReduce(volatile T * sdata, unsigned int tIdx) {
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

    if (threadIdx.x < 32) warpReduce<T>(sdata, threadIdx.x);

	// export result to global memory
	if (threadIdx.x == 0) {
		g_out[blockIdx.x] = sdata[0];
	}
}


template<typename T, unsigned int blockSize>
void wrapperKer_5(const T * d_in, T * d_temp1, T * d_temp2, T ** d_out, const int N, const int numThreadsPerBlock) {
    // all buffers must be allocated before-hand with sizes:
    // d_in : ROUND_UP(N, 2*numThreadsPerBlock) NOTE: we need 2*numThreadsPerBlock for kernel 3!!!
    // d_out : 1
    // d_temp1 : ROUND_UP(DIV_UP(N, 2*numThreadsPerBlock), 2*numThreadsPerBlock) = ROUND_UP(numBlocks1, 2*numThreadsPerBlock)
    // d_temp2 :  ROUND_UP(DIV_UP(numBlocks1, 2*numThreadsPerBlock), 2*numThreadsPerBlock) = ROUND_UP(numBlocks2, 2*numThreadsPerBlock)

    // first iteration
    int numBlocks = DIV_UP(N, 2*numThreadsPerBlock); // we only need half the amount of blocks for kernel 3
    CCE(cudaMemset(d_temp1, 0, ROUND_UP(numBlocks, 2*numThreadsPerBlock)*sizeof(T)));
    ker_5 < T , blockSize > <<< numBlocks , numThreadsPerBlock , sizeof(T) * numThreadsPerBlock >>> (d_in, d_temp1);
    CCEL();
    
    // do recursive reduction
    while (numBlocks > 1) {
        // switch buffers
        T * temp = d_temp1;
        d_temp1 = d_temp2;
        d_temp2 = temp;

        // calculate new numBlocks
        numBlocks = DIV_UP(numBlocks, 2*numThreadsPerBlock);

        // prepare the output buffer
        CCE(cudaMemset(d_temp1, 0, ROUND_UP(numBlocks, 2*numThreadsPerBlock)*sizeof(T)));

        // invoke the kernel
        ker_5 < T , blockSize > <<< numBlocks , numThreadsPerBlock , sizeof(T) * numThreadsPerBlock >>> (d_temp2, d_temp1);
        CCEL();
    }
    *d_out = d_temp1;
}


template<typename T, unsigned int blockSize>
__global__ void ker_6(const T * g_in, T * g_out, unsigned int N) {
	// prepare shared data allocated by kernel invocation
	extern __shared__ T sdata[];

	// load data into shared memory
    unsigned int i = threadIdx.x + blockIdx.x*blockSize*2;
    unsigned int gridSize = blockSize*2*gridDim.x; 
    sdata[threadIdx.x] = 0;
    while (i < N) {
        sdata[threadIdx.x] += g_in[i] + g_in[i + blockSize];
        i += gridSize;
    }
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

    if (threadIdx.x < 32) warpReduce<T>(sdata, threadIdx.x);

	// export result to global memory
	if (threadIdx.x == 0) {
		g_out[blockIdx.x] = sdata[0];
	}
}


template<typename T, unsigned int blockSize>
void wrapperKer_6(const T * d_in, T * d_temp1, T * d_temp2, T ** d_out, int N, const int numThreadsPerBlock, const int numElementsPerThread) {
    // all buffers must be allocated before-hand with sizes:
    // d_in : ROUND_UP(N, numElementsPerThread*2*numThreadsPerBlock) NOTE: we need 2*numThreadsPerBlock for kernel 3!!!
    // d_out : 1
    // d_temp1 : ROUND_UP(DIV_UP(N, numElementsPerThread*2*numThreadsPerBlock), numElementsPerThread*2*numThreadsPerBlock) 
    //          = ROUND_UP(numBlocks1, numElementsPerThread*2*numThreadsPerBlock)
    // d_temp2 :  ROUND_UP(DIV_UP(numBlocks1, numElementsPerThread*2*numThreadsPerBlock), numElementsPerThread*2*numThreadsPerBlock) 
    //          = ROUND_UP(numBlocks2, numElementsPerThread*2*numThreadsPerBlock)
    // first iteration

    int numBlocks = DIV_UP(N, numElementsPerThread*2*numThreadsPerBlock); 
    CCE(cudaMemset(d_temp1, 0, ROUND_UP(numBlocks, numElementsPerThread*2*numThreadsPerBlock)*sizeof(T)));
    ker_6 < T , blockSize > <<< numBlocks , numThreadsPerBlock , sizeof(T) * numThreadsPerBlock >>> (d_in, d_temp1, N);
    CCEL();

    N = numBlocks;

    // do recursive reduction
    while (numBlocks > 1) {
        // switch buffers
        T * temp = d_temp1;
        d_temp1 = d_temp2;
        d_temp2 = temp;

        // calculate new numBlocks
        numBlocks = DIV_UP(numBlocks, numElementsPerThread*2*numThreadsPerBlock);

        // prepare the output buffer
        CCE(cudaMemset(d_temp1, 0, ROUND_UP(numBlocks, numElementsPerThread*2*numThreadsPerBlock)*sizeof(T)));

        // invoke the kernel
        ker_6 < T , blockSize > <<< numBlocks , numThreadsPerBlock , sizeof(T) * numThreadsPerBlock >>> (d_temp2, d_temp1, N);
        CCEL();

        N = numBlocks;
    }

    *d_out = d_temp1;
}
