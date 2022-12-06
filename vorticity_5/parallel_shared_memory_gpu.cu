/*
  This code is meant to be run on a cuda device. 
  You must first use the command module load cuda 
  to ensure that you can compile correctly with 
  nvcc parallel_shared_memory_gpu.cu -o object_name.
  Once the code is compiled you can run then run the
  code with ./object_name assuming you have prepared 
  the gpu correctly. 
*/
#include <cstring> 
#include <iostream>
#include <string>
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}



#define WIDTH 1300
#define HEIGHT 600
#define GRID_WIDTH 65
#define GRID_HEIGHT 20
#define BLOCK_WIDTH 20
#define BLOCK_HEIGHT 30
// #define WIDTH 260
// #define HEIGHT 120
// #define GRID_WIDTH 13
// #define GRID_HEIGHT 4
// #define BLOCK_WIDTH 20
// #define BLOCK_HEIGHT 30
#define HALO 2
#define CHANNELS 2

__global__
void convertTile(int height, int width, unsigned char *output, float *input, int my_rank, int core_count) {

  __shared__ float vortTile[BLOCK_WIDTH + HALO][BLOCK_HEIGHT + HALO][CHANNELS];

  // these are coordinates in the already tiled block that the node recieved
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  // adjust those coordinates for the data block this node is responsible for
  if (my_rank != 0) y+=1;

  // Copy over the vector information to the tile
  if (threadIdx.x == 0 && x != 0) { //get Left Halo
    vortTile[threadIdx.x][threadIdx.y + 1][0] = input[CHANNELS * ((y * WIDTH) + (x - 1))];
    vortTile[threadIdx.x][threadIdx.y + 1][1] = input[(CHANNELS * ((y * WIDTH) + (x - 1))) + 1];
  }
  
  if (threadIdx.x == (BLOCK_WIDTH - 1) && x != width - 1) { //get Right Halo
    vortTile[threadIdx.x + 2][threadIdx.y + 1][0] = input[CHANNELS * ((y * WIDTH) + (x + 1))];
    vortTile[threadIdx.x + 2][threadIdx.y + 1][1] = input[(CHANNELS * ((y * WIDTH) + (x + 1))) + 1];
  }
 
  if (threadIdx.y == 0 && y != 0) { //get Upper Halo
    if (threadIdx.x == (BLOCK_WIDTH - 1) && x != width - 1) { //get Upper Right Corner 
      vortTile[threadIdx.x + 2][threadIdx.y][0] = input[CHANNELS * (((y - 1) * WIDTH) + (x + 1))];
      vortTile[threadIdx.x + 2][threadIdx.y][1] = input[CHANNELS * (((y - 1) * WIDTH) + (x + 1)) + 1];
    }
    vortTile[threadIdx.x + 1][threadIdx.y][0] = input[CHANNELS * (((y - 1) * WIDTH) + x)];
    vortTile[threadIdx.x + 1][threadIdx.y][1] = input[(CHANNELS * (((y - 1) * WIDTH) + x)) + 1];
  }
  
  if (threadIdx.y == (BLOCK_HEIGHT - 1) && y != height - 1) { // Get Lower Halo
    if (threadIdx.x == 0 && x != 0) { //get Lower Left Corner
      vortTile[threadIdx.x][threadIdx.y + 2][0] = input[CHANNELS * (((y + 1) * WIDTH) + (x - 1))];  
      vortTile[threadIdx.x][threadIdx.y + 2][1] = input[(CHANNELS * (((y + 1) * WIDTH) + (x - 1))) + 1];
    }
    vortTile[threadIdx.x + 1][threadIdx.y + 2][0] = input[CHANNELS * (((y + 1) * WIDTH) + x)];  
    vortTile[threadIdx.x + 1][threadIdx.y + 2][1] = input[(CHANNELS * (((y + 1) * WIDTH) + x)) + 1];
  }
  vortTile[threadIdx.x + 1][threadIdx.y + 1][0] = input[CHANNELS * ((y * WIDTH) + x)];
  vortTile[threadIdx.x + 1][threadIdx.y + 1][1] = input[(CHANNELS * ((y * WIDTH) + x)) + 1];
  __syncthreads();

  //I am not sure if cuda can call a function that is in another file so I just put this here. 
  //The vorticity funciton
  float dx = 0.01;
  float dy = 0.01;

  int start_x = (x == 0) ? 1 : threadIdx.x;
  int end_x = (x == width - 1) ? threadIdx.x + 1: threadIdx.x + 2;

  int start_y = (y == 0) ? 1 : threadIdx.y;
  int end_y = (y == height - 1) ? threadIdx.y + 1: threadIdx.y + 2;

  double fdu[2] = {vortTile[end_x][start_y][0], vortTile[end_x][start_y][1]};
  double fdv[2] = {vortTile[start_x][end_y][0], vortTile[start_x][end_y][1]};
  double vec0[2] = {vortTile[threadIdx.x + 1][threadIdx.y + 1][0], vortTile[threadIdx.x + 1][threadIdx.y + 1][1]};
  float duy = (fdu[1] - vec0[1]) / (dx * (end_x - start_x));
  float dvx = (fdv[0] - vec0[0]) / (dy * (end_y - start_y));

  float vort = duy - dvx;
  //End of vorticity function

  unsigned char vortChar;
  if (vort < -0.2f) {
    vortChar = 0;
  } else if (vort > 0.2f) {
    vortChar = 127;
  } else {
    vortChar = 255;
  }
  output[y * width + x] = vortChar;
  __syncthreads();

}


extern "C" void parallel_shared_memory_gpu(int height, int width, float* input, unsigned char* output, int length, int my_rank, int core_count) {
    //Prepare cuda stuff
    float *inputDevice;
    unsigned char * outputDevice;
    cudaMalloc((void **) &inputDevice, length);
    cudaMalloc((void **) &outputDevice, length / 8);

    cudaMemcpy(inputDevice, input, length, cudaMemcpyHostToDevice);
    cudaMemcpy(outputDevice, output, length / 8, cudaMemcpyHostToDevice);

    // calculate grid and block size
    int data_height = height - 2; // data size is the height minus the top and bottom
    if (my_rank == 0 || my_rank == core_count - 1) data_height += 1; // unless this is the fist or last rank
    int grid_height = data_height / BLOCK_HEIGHT;
    if (grid_height * BLOCK_HEIGHT < data_height) {
      grid_height += 1;
    }
    printf("Height: %d, data_height: %d grid_height: %d\n", height, data_height, grid_height);
    const dim3 block_size (BLOCK_WIDTH, BLOCK_HEIGHT);
    const dim3 grid_size (GRID_WIDTH, grid_height);

    convertTile<<<grid_size, block_size>>>(height, width, outputDevice, inputDevice, my_rank, core_count);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // printf("Error: %d", cudaDeviceSynchronize());

    //Return image to device and free memory
    cudaMemcpy(output, outputDevice, length / 8, cudaMemcpyDeviceToHost);
    cudaFree(inputDevice);
    cudaFree(outputDevice);
}

// int main()
// {
//     int height = 301;
//     int width = 1300;
//     int channels = 2;
// 
//     // read in input
//     float* input = (float*)malloc(height*width*channels*sizeof(float));
//     unsigned char* output = (unsigned char*)malloc(height*width*sizeof(unsigned char));
//     FILE* pf = fopen("cyl2d_1300x600_float32[2].raw", "rb");
//     fread(input, sizeof(float), height*width*channels, pf);
//     fclose(pf);
// 
//     parallel_shared_memory_gpu(height, width, input, output, height*width*channels*sizeof(float), 0, 2);
// }
