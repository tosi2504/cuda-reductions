#include "../headers/kernels.cuh"
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <stdint.h>

double timeKernel4(const int n) {
    // set the datatype
    typedef int datatype;

	// set the parallelisation parameters
    const int numThreadsPerBlock = 256;
	const int N = ROUND_UP(n, 2*numThreadsPerBlock); // NOTE: for kernels 0,1,2 dont need the 2*...

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
    // zeropadding
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
    unsigned int numCycles = 1<<8;

    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < numCycles; ++i) {
        wrapperKer_4 < datatype > (d_in, d_temp1, d_temp2, &d_out, N, numThreadsPerBlock);
        cudaDeviceSynchronize();
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
	CCE(cudaMemcpy(h_out, d_out, sizeof(datatype), cudaMemcpyDeviceToHost));
    CCEL();

    cudaFree(d_in);
    CCEL();
    cudaFree(d_out);
    CCEL();
    //cudaFree(d_temp1);
    CCEL();
    //cudaFree(d_temp2);
    CCEL();
    free(h_in);
    free(h_out);

    return duration.count()/(double)numCycles;
}


int main () {
    std::ofstream file("data.txt");
    // time the kernel
    int N = 1024;
    for (int i = 0; i < 21; i++) {
        std::cout << N << " " << timeKernel4(N) << std::endl;
        file << N << " " << timeKernel4(N) << "\n";
        N *= 2;
    }
    file.close();
}

