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
    unsigned char* output = malloc(length*sizeof(unsigned char));
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
    parallel_shared_memory_cpu(HEIGHT, WIDTH, input, output);

    if (validate(HEIGHT, WIDTH, output, valid))
        printf("Parallel shared memory valid\n");
    else
        printf("Parallel shared memory invalid\n");

    // Writing output to file
    FILE* wf = fopen("outfield.raw", "wb");
    fwrite(output, sizeof(unsigned char), length, wf);
    fclose(wf);

    // free up memory again
    free(input);
    free(output);
    free(valid);

    return 0;
}
