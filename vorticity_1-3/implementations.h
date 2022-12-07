#include <stdbool.h>

bool validate(int HEIGHT, int WIDTH, unsigned char * test, unsigned char * valid);
// impl 1
float serial_vorticity(int HEIGHT, int WIDTH, float* input, unsigned char * output);
// impl 2
float parallel_shared_memory_cpu(int HEIGHT, int WIDTH, float* input, unsigned char * output, int thread_count);
// impl 3
float parallel_shared_memory_gpu(int height, int width, float * input, unsigned char * output, int length);
// impl 4

