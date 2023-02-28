all:
	nvcc sources/main.cu sources/kernels.cu -O3 -o out
