compile:
	mpicc -c distributed_memory_cpu.c -o main.o
	nvcc -c parallel_shared_memory_gpu.cu -o gpu_code.o
	mpicc main.o gpu_code.o -lcudart -o project -lstdc++
