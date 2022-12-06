compile: compile_gpu_code compile_cpu_code
	gcc psm_cpu.o v.o main.o sv.o gpu_code.o -lcudart -lstdc++ -o vorticity
	
compile_cpu_code:
	gcc -c parallel_shared_memory_cpu.c -o psm_cpu.o
	gcc -c vorticity.c -o v.o
	gcc -c main.c -o main.o
	gcc -c serial_vorticity.c -o sv.o

compile_gpu_code:
	nvcc -c parallel_shared_memory_gpu.cu -o gpu_code.o
