#include "../headers/kernels.cuh"
#include <iostream>
#include <random>
#include <chrono>

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
 

int main () {
	int nDev;
	CCE(cudaGetDeviceCount(&nDev));
	std::cout << "Cuda detected " << nDev << " device(s)" << std::endl;
	CCE(cudaSetDevice(0));
	std::cout << "Working on device 0" << std::endl;

	// set the parallelisation parameters
	const int threads_per_block = 1024;
	const int numBlocks = 1024;
	const int N = threads_per_block * numBlocks;

	// allocate memory on host for input and output
	double * h_in = (double*)malloc(sizeof(double)*N);
	double * h_out = (double*)malloc(sizeof(double)*numBlocks);

	// fill array with random numbers
	std::mt19937 rng(0);
	std::uniform_real_distribution<double> dist(0,1); 
	for (int i = 0; i < N; ++i) {
		h_in[i] = dist(rng);
		//h_in[i] = 1;
	}

	// allocate and populate memory on the device
	double * d_in, * d_out;
	CCE(cudaMalloc(&d_in, sizeof(double)*N));
	CCE(cudaMemcpy(d_in, h_in, sizeof(double)*N, cudaMemcpyHostToDevice));
	CCE(cudaMalloc(&d_out, sizeof(double)*numBlocks)); // does not need to be initialized

	// benchmark the kernel
    unsigned int numCycles = 1<<15;

    // kernel 0
    std::cout << "benchmarking kernel 0" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < numCycles; ++i) {
	    ker_0 <double> <<< numBlocks, threads_per_block, threads_per_block*sizeof(double) >>> (d_in, d_out);
    }
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
	std::cout << "kernel 0 took " << duration.count()/(double)numCycles << " us" << std::endl;
	CCEL();

    // kernel 1
    std::cout << "benchmarking kernel 1" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < numCycles; ++i) {
	    ker_1 <double> <<< numBlocks, threads_per_block, threads_per_block*sizeof(double) >>> (d_in, d_out);
    }
    cudaDeviceSynchronize();
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
	std::cout << "kernel 1 took " << duration.count()/(double)numCycles << " us" << std::endl;
	CCEL();

    // kernel 2
    std::cout << "benchmarking kernel 2" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < numCycles; ++i) {
	    ker_2 <double> <<< numBlocks, threads_per_block, threads_per_block*sizeof(double) >>> (d_in, d_out);
    }
    cudaDeviceSynchronize();
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
	std::cout << "kernel 2 took " << duration.count()/(double)numCycles << " us" << std::endl;
	CCEL();

    // kernel 3
    std::cout << "benchmarking kernel 3" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < numCycles; ++i) {
	    ker_3 <double> <<< numBlocks/2, threads_per_block, threads_per_block*sizeof(double) >>> (d_in, d_out);
    }
    cudaDeviceSynchronize();
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
	std::cout << "kernel 3 took " << duration.count()/(double)numCycles << " us" << std::endl;
	CCEL();

    // kernel 4
    std::cout << "benchmarking kernel 4" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < numCycles; ++i) {
	    ker_4 <double> <<< numBlocks/2, threads_per_block, threads_per_block*sizeof(double) >>> (d_in, d_out);
    }
    cudaDeviceSynchronize();
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
	std::cout << "kernel 4 took " << duration.count()/(double)numCycles << " us" << std::endl;
	CCEL();

    // kernel 5
    std::cout << "benchmarking kernel 5" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < numCycles; ++i) {
	    ker_5 < double , threads_per_block > <<< numBlocks/2, threads_per_block, threads_per_block*sizeof(double) >>> (d_in, d_out);
    }
    cudaDeviceSynchronize();
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
	std::cout << "kernel 5 took " << duration.count()/(double)numCycles << " us" << std::endl;
	CCEL();

	// transfer the data back
	CCE(cudaMemcpy(h_out, d_out, sizeof(double)*numBlocks, cudaMemcpyDeviceToHost));

	// check for correctness
	for (int i = 0; i < 5; ++i) {
		std::cout << h_out[i] << std::endl;
	}

}
