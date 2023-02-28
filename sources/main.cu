#include "../headers/kernels.cuh"
#include "../sources/kernels.cu" // has to be included bc of templates ... uagh
#include <iostream>
#include <random>

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
	int threads_per_block = 1024;
	int numBlocks = 1024;
	int N = threads_per_block * numBlocks;

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

	// run the kernel
	ker_0 <double> <<< numBlocks, threads_per_block, threads_per_block*sizeof(double) >>> (d_in, d_out);
	std::cout << "Kernel finished" << std::endl;
	CCEL();

	// transfer the data back
	CCE(cudaMemcpy(h_out, d_out, sizeof(double)*numBlocks, cudaMemcpyDeviceToHost));

	// check for correctness
	for (int i = 0; i < 5; ++i) {
		std::cout << h_out[i] << std::endl;
	}
}
