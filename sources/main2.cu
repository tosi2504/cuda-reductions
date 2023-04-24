#include "../headers/kernels.cuh"
#include <iostream>
#include <random>
#include <chrono>

 

int main () {
	int nDev;
	CCE(cudaGetDeviceCount(&nDev));
	std::cout << "Cuda detected " << nDev << " device(s)" << std::endl;
	CCE(cudaSetDevice(0));
	std::cout << "Working on device 0" << std::endl;
    
    // set the datatype
    typedef int datatype;

	// set the parallelisation parameters
	const int numThreadsPerBlock = 256;
    const int n = 2<<26;
    std::cout << n << std::endl;
	const int N = ROUND_UP(n, 2*numThreadsPerBlock); // NOTE: for kernels 0,1,2 dont need the 2*...
    std::cout << N << std::endl;

	// allocate memory on host for input and output
	datatype * h_in = (datatype*)malloc(sizeof(datatype)*N);
    datatype * h_out = (datatype*)malloc(sizeof(datatype));

	// fill array with random numbers
	std::mt19937 rng(106);
	//std::uniform_real_distribution<double> dist(0,1); 
	std::uniform_int_distribution<> dist(0,10); 
	for (int i = 0; i < n; ++i) {
		h_in[i] = dist(rng);
		//h_in[i] = 1;
	}
	for (int i = n; i < N; ++i) {
		h_in[i] = 0;
	}

	// allocate and populate memory on the device
	datatype * d_in, * d_temp1, * d_temp2, * d_out;
	CCE(cudaMalloc(&d_in, sizeof(datatype)*N));
	CCE(cudaMemcpy(d_in, h_in, sizeof(datatype)*N, cudaMemcpyHostToDevice));
    int numBlocks1 = DIV_UP(N, numThreadsPerBlock);
	CCE(cudaMalloc(&d_temp1, sizeof(datatype)*ROUND_UP(numBlocks1, 2*numThreadsPerBlock))); // does not need to be initialized
    // NOTE: 2* is not required for kernels 0,1,2
    int numBlocks2 = DIV_UP(numBlocks1, numThreadsPerBlock);
	CCE(cudaMalloc(&d_temp2, sizeof(datatype)*ROUND_UP(numBlocks2, 2*numThreadsPerBlock))); // does not need to be initialized
    // NOTE: 2* is not required for kernels 0,1,2

	// benchmark the kernel
    unsigned int numCycles = 1<<10;

    // kernel 0
    std::cout << "benchmarking kernel 0" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < numCycles; ++i) {
        wrapperKer_0 < datatype > (d_in, d_temp1, d_temp2, &d_out, N, numThreadsPerBlock);
        cudaDeviceSynchronize();
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
	std::cout << "kernel 0 took " << duration.count()/(double)numCycles << " us" << std::endl;
	CCEL();
	CCE(cudaMemcpy(h_out, d_out, sizeof(datatype), cudaMemcpyDeviceToHost));
	std::cout << "Result for total reduction: " << h_out[0] << std::endl;

    // kernel 1
    std::cout << "benchmarking kernel 1" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < numCycles; ++i) {
        wrapperKer_1 < datatype > (d_in, d_temp1, d_temp2, &d_out, N, numThreadsPerBlock);
        cudaDeviceSynchronize();
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
	std::cout << "kernel 1 took " << duration.count()/(double)numCycles << " us" << std::endl;
	CCEL();
	CCE(cudaMemcpy(h_out, d_out, sizeof(datatype), cudaMemcpyDeviceToHost));
	std::cout << "Result for total reduction: " << h_out[0] << std::endl;

    // kernel 2
    std::cout << "benchmarking kernel 2" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < numCycles; ++i) {
        wrapperKer_2 < datatype > (d_in, d_temp1, d_temp2, &d_out, N, numThreadsPerBlock);
        cudaDeviceSynchronize();
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
	std::cout << "kernel 2 took " << duration.count()/(double)numCycles << " us" << std::endl;
	CCEL();
	CCE(cudaMemcpy(h_out, d_out, sizeof(datatype), cudaMemcpyDeviceToHost));
	std::cout << "Result for total reduction: " << h_out[0] << std::endl;

    // kernel 3
    std::cout << "benchmarking kernel 3" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < numCycles; ++i) {
        wrapperKer_3 < datatype > (d_in, d_temp1, d_temp2, &d_out, N, numThreadsPerBlock);
        cudaDeviceSynchronize();
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
	std::cout << "kernel 3 took " << duration.count()/(double)numCycles << " us" << std::endl;
	CCEL();
	CCE(cudaMemcpy(h_out, d_out, sizeof(datatype), cudaMemcpyDeviceToHost));
	std::cout << "Result for total reduction: " << h_out[0] << std::endl;

    // kernel 4
    std::cout << "benchmarking kernel 4" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < numCycles; ++i) {
        wrapperKer_4 < datatype > (d_in, d_temp1, d_temp2, &d_out, N, numThreadsPerBlock);
        cudaDeviceSynchronize();
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
	std::cout << "kernel 4 took " << duration.count()/(double)numCycles << " us" << std::endl;
	CCEL();
	CCE(cudaMemcpy(h_out, d_out, sizeof(datatype), cudaMemcpyDeviceToHost));
	std::cout << "Result for total reduction: " << h_out[0] << std::endl;

    // kernel 5
    std::cout << "benchmarking kernel 5" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < numCycles; ++i) {
        wrapperKer_5 < datatype , numThreadsPerBlock > (d_in, d_temp1, d_temp2, &d_out, N, numThreadsPerBlock);
        cudaDeviceSynchronize();
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
	std::cout << "kernel 5 took " << duration.count()/(double)numCycles << " us" << std::endl;
	CCEL();
	CCE(cudaMemcpy(h_out, d_out, sizeof(datatype), cudaMemcpyDeviceToHost));
	std::cout << "Result for total reduction: " << h_out[0] << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_temp1);
    cudaFree(d_temp2);
    free(h_in);
    free(h_out);
}
