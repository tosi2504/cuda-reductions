all:
	nvcc sources/main2.cu -O3 -o out

numThreads:
	nvcc sources/number_of_threads_per_block.cu -O3 -o out

scaling:
	nvcc sources/scaling.cu -O3 -o out

datatype:
	nvcc sources/datatype.cu -O3 -o out
