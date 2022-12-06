#include <stdbool.h>
#include "implementations.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define WIDTH 1300
#define HEIGHT 600
#define CHANNELS 2

int main() {
    // initialize data storage
    int length = HEIGHT * WIDTH;
    float* input = malloc(length*CHANNELS*sizeof(float));
    unsigned char* valid = malloc(length*sizeof(unsigned char));
    printf("Initialized data arrays\n");

    // read in data from file
    FILE* pf = fopen("cyl2d_1300x600_float32[2].raw", "rb");
    fread(input, sizeof(float), length*CHANNELS*sizeof(float), pf);
    fclose(pf);
    printf("done reading data from file\n");

    // create valid with serial algorithm
    printf("Running serial vorticity\n");
    serial_vorticity(HEIGHT, WIDTH, input, valid);

    // Parallel shared memory
    printf("Running parallel shared memory cpu\n");
    unsigned char* psm_cpu_output = malloc(length*sizeof(unsigned char));
    parallel_shared_memory_cpu(HEIGHT, WIDTH, input, psm_cpu_output);

    if (validate(HEIGHT, WIDTH, psm_cpu_output, valid))
        printf("Parallel shared CPU valid\n");
    else
        printf("Parallel shared CPU invalid\n");

    // Parallel shared memory gpu
    printf("Running parallel shared memory GPU\n");
    unsigned char* psm_gpu_output = malloc(length*sizeof(unsigned char));
    parallel_shared_memory_gpu(HEIGHT, WIDTH, input, psm_gpu_output, length*CHANNELS*sizeof(float));

    if (validate(HEIGHT, WIDTH, psm_gpu_output, valid))
        printf("Parallel shared GPU valid\n");
    else
        printf("Parallel shared GPU invalid\n");

    // Writing output to file
    FILE* wf = fopen("psm_cpu_outfield.raw", "wb");
    fwrite(psm_cpu_output, sizeof(unsigned char), length, wf);
    fclose(wf);

    wf = fopen("psm_gpu_outfield.raw", "wb");
    fwrite(psm_gpu_output, sizeof(unsigned char), length, wf);
    fclose(wf);

    // free up memory again
    free(input);
    free(psm_cpu_output);
    free(psm_gpu_output);
    free(valid);

    return 0;
}
