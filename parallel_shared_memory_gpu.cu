#include <cstring> 
#include <fstream>
#include <iostream>
#include <vector>
#include "vorticity.hpp"


  
#define BLOCK_WIDTH 20
#define TILE_WIDTH 20
#define TILE_HEIGHT 20
#define HALO 2
#define CHANNELS 2

//This doesn't work just the beginnings of an idea

// __global__
// void convertTile(int HEIGHT, int WIDTH, unsigned char *output, unsigned char *input) {

//   __shared__ float vortTile[TILE_WIDTH + HALO][TILE_HEIGHT + HALO][CHANNELS];

//   int x = threadIdx.x + blockIdx.x * blockDim.x;
//   int y = threadIdx.y + blockIdx.y * blockDim.y;

//   // Copy over the vector information to the tile
//   // Copy over the vector information to the tile
//   for (int i = 0; i < TILE_HEIGHT + HALO; i++) {
//     for (int j = 0; j < TILE_WIDTH + HALO; j++) {
//       if (y - 1 == -1) { y = 1; }
//       if (x - 1 == -1) { x = 1; }
//       if (y - 1 == HEIGHT) { y = WIDTH; }

//       vortTile[threadIdx.x][threadIdx.y][0] = input[CHANNELS * ((y - 1) * WIDTH + (x - 1))];
//       vortTile[threadIdx.x][threadIdx.y][1] = input[CHANNELS * ((y - 1) * WIDTH + (x - 1)) + 1];
//       __syncthreads();
//     }
//   }

//   for (int i = 1; i < TILE_HEIGHT; i++) {
//     for (int j = 1; j < TILE_WIDTH; j++) {
//         float vort = vorticity(j, i, WIDTH, HEIGHT, vortTile);
//         if (vort < -0.2f) {
//           vortChar = 0;
//         } else if (vort > 0.2f) {
//           vortChar = 127;
//         } else {
//           vortChar = 255;
//         }
//         output[x * WIDTH + y] = vortChar
//         __syncthreads();
//     }
//   }
// }

__global__
void Convert(int HEIGHT, int WIDTH, unsigned char *output, float *input) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  float vort = vorticity(x, y, WIDTH, HEIGHT, input);
  if (vort < -0.2f) {
    vortChar = 0;
  } else if (vort > 0.2f) {
    vortChar = 127;
  } else {
    vortChar = 255;
  }
  output[x * WIDTH + y] = vortChar
}

void parallel_shared_memory_gpu(int HEIGHT, int WIDTH, float * input, unsigned char * output) {
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


