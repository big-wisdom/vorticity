compile:
	gcc -c serial_vorticity.c -o sv.o
	gcc -c vorticity.c -o v.o
	mpicc -c distributed_memory_cpu.c -o main.o
	nvcc -c parallel_shared_memory_gpu.cu -o gpu_code.o
	mpicc main.o gpu_code.o sv.o v.o -lcudart -o project -lstdc++
