#include <cstring> 
#include <fstream>
#include <iostream>
#include <vector>
#include <cuda.h>
  
#define BLOCK_WIDTH 32
#define TILE_WIDTH 32

//This doesn't work just the beginnings of an idea

void convertTile(int HEIGHT, int WIDTH, unsigned char *output, unsigned char *input) {

  __shared__ float flippedTile[TILE_WIDTH][TILE_WIDTH];

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  for (int i = 0; i < HEIGHT; i++) {
    for (int j = 0; j < WIDTH; j++) {
        float vort = vorticity(j, i, WIDTH, HEIGHT, input);
        flippedTile[threadIdx.x][threadIdx.y] = vort;
        __syncthreads();
    }
  }

  for (int i = 0; i < HEIGHT; i++) {
    for (int j = 0; j < WIDTH; j++) {
        float vort = flippedTile[threadIdx.y][threadIdx.x]
        if (vort < -0.2f) {
          vortChar = 0;
        } else if (vort > 0.2f) {
          vortChar = 127;
        } else {
          vortChar = 255;
        }
      output[i * WIDTH + j] = vortChar
    }
  }
  __syncthreads();
}

int parallel_shared_memory_gpu(int HEIGHT, int WIDTH, float * input, unsigned char * output) {
    //Prepare cuda stuff
    float *inputDevice;
    unsigned char * outputDevice;
    cudaMalloc((void **) &inputDevice, length);
    cudaMalloc((void **) &outputDevice, length);

    cudaMemcpy(inputDevice, input, length, cudaMemcpyHostToDevice);
    cudaMemcpy(outputDevice, output, length, cudaMemcpyHostToDevice);

    const dim3 block_size (BLOCK_WIDTH, BLOCK_WIDTH);
    const dim3 grid_size (BLOCK_WIDTH, BLOCK_WIDTH);

    convertTile<<<block_size, grid_size>>>(outputDevice, inputDevice);

    //Return image to device and free memory
    cudaMemcpy(outputGPUShared, outputDevice, length, cudaMemcpyDeviceToHost);
    cudaFree(inputDevice);
    cudaFree(outputDevice);

}


