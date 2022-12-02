#include <cstring> 
#include <fstream>
#include <iostream>
#include <vector>
#include "vorticity.hpp"


#define WIDTH 1300
#define HEIGHT 600
#define GRID_WIDTH 20
#define GRID_HEIGHT 30
#define BLOCK_WIDTH 65
#define BLOCK_HEIGHT 20
#define HALO 2
#define CHANNELS 2

//This doesn't work just the beginnings of an idea

__global__
void convertTile(int height, int width, unsigned char *output, float *input) {

  __shared__ float vortTile[BLOCK_WIDTH + HALO][BLOCK_HEIGHT + HALO][CHANNELS];

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  // Copy over the vector information to the tile
  
  if (threadIdx.x == 0 && x != 0) {
    vortTile[threadIdx.x][threadIdx.y + 1][0] = input[CHANNELS * (y * WIDTH + (x - 1))];
    vortTile[threadIdx.x][threadIdx.y + 1][1] = input[CHANNELS * (y * WIDTH + (x - 1)) + 1];
  }
  
  if (threadIdx.x == (BLOCK_WIDTH - 1) && x != width - 1) {
    vortTile[threadIdx.x + 2][threadIdx.y + 1][0] = input[CHANNELS * (y * WIDTH + (x + 1))];
    vortTile[threadIdx.x + 2][threadIdx.y + 1][1] = input[CHANNELS * (y * WIDTH + (x + 1)) + 1];
  }
 
  if (threadIdx.y == 0 && y != 0) {
    vortTile[threadIdx.x + 1][threadIdx.y][0] = input[CHANNELS * ((y - 1) * WIDTH + x)];
    vortTile[threadIdx.x + 1][threadIdx.y][1] = input[CHANNELS * ((y - 1) * WIDTH + x) + 1];
  }
  
  if (threadIdx.y == (BLOCK_HEIGHT - 1) && y != height - 1) {
    vortTile[threadIdx.x + 1][threadIdx.y + 2][0] = input[CHANNELS * ((y + 1) * WIDTH + x)];
    vortTile[threadIdx.x + 1][threadIdx.y + 2][1] = input[CHANNELS * ((y + 1) * WIDTH + x) + 1];
  }
  vortTile[threadIdx.x + 1][threadIdx.y + 1][0] = input[CHANNELS * (y * WIDTH + x)];
  vortTile[threadIdx.x + 1][threadIdx.y + 1][1] = input[CHANNELS * (y * WIDTH + x) + 1];
  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0 && x == 0 && y == 0)
  {
    for(int i = 1; i < BLOCK_WIDTH + 1; i++) {
      for(int j = 1; j < BLOCK_HEIGHT + 1; j++) {
        int newX = (i - 1) + blockIdx.x * blockDim.x;
        int newY = (j - 1) + blockIdx.y * blockDim.y;
        printf("x: %d y: %d Tile x: %f Tile y: %f input x: %f input y: %f\n",newX, newY,vortTile[i][j][0], vortTile[i][j][1], input[CHANNELS * (newY * width + (newX))], input[CHANNELS * (newY * width + (newX)) + 1]);
      }
    }
  }

  //The vorticity funciton
  float dx = 0.01;
  float dy = 0.01;

  //uint32_t idx = y * width + x;

  int start_x = (x == 0) ? 0 : threadIdx.x;
  int end_x = (x == width - 1) ? threadIdx.x + 1: threadIdx.x + 2;

  int start_y = (y == 0) ? 0 : threadIdx.y;
  int end_y = (y == height - 1) ? threadIdx.y + 1: threadIdx.y + 2;

  // duidx = (start_y * width + end_x) * 2;
  //uint32_t dvidx = (end_y * width + start_x) * 2;

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

__global__
void convert(int height, int width, unsigned char *output, float *input) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  //The vorticity funciton
  float dx = 0.01;
  float dy = 0.01;

  uint32_t idx = y * width + x;

  int start_x = (x == 0) ? 0 : x - 1;
  int end_x = (x == width - 1) ? x : x + 1;

  int start_y = (y == 0) ? 0 : y - 1;
  int end_y = (y == height - 1) ? y : y + 1;

  uint32_t duidx = (start_y * width + end_x) * 2;
  uint32_t dvidx = (end_y * width + start_x) * 2;

  double fdu[2] = {input[duidx], input[duidx + 1]};
  double fdv[2] = {input[dvidx], input[dvidx + 1]};
  double vec0[2] = {input[idx * 2], input[idx * 2 + 1]};

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
}

void parallel_shared_memory_gpu(int height, int width, float * input, unsigned char * output, int length) {
    //Prepare cuda stuff
    float *inputDevice;
    unsigned char * outputDevice;
    cudaMalloc((void **) &inputDevice, length);
    cudaMalloc((void **) &outputDevice, length / 8);

    cudaMemcpy(inputDevice, input, length, cudaMemcpyHostToDevice);
    cudaMemcpy(outputDevice, output, length / 8, cudaMemcpyHostToDevice);

    const dim3 block_size (BLOCK_WIDTH, BLOCK_HEIGHT);
    const dim3 grid_size (GRID_WIDTH, GRID_HEIGHT);

    convertTile<<<block_size, grid_size>>>(height, width, outputDevice, inputDevice);
    printf("Error: %d", cudaDeviceSynchronize());

    //Return image to device and free memory
    cudaMemcpy(output, outputDevice, length / 8, cudaMemcpyDeviceToHost);
    cudaFree(inputDevice);
    cudaFree(outputDevice);
}


int main() {
  std::ifstream vectorField("cyl2d_1300x600_float32[2].raw", std::ios::binary);
  if (vectorField.is_open()) {
    // Get the length of the image should be 3145728
    std::cout << "opened" << std::endl;
    vectorField.seekg(0, std::ios_base::end);
    auto length = vectorField.tellg();
    printf("%d", length);
    vectorField.seekg(0, std::ios::beg);

    auto fl_size = sizeof(float);

    // Initialize arrays
    float *input = new float[length / fl_size];
    unsigned char *output = new unsigned char[length / fl_size / CHANNELS];

    // Get rgb values from image into input array
    vectorField.read((char *)input, length);
    vectorField.close();

    // serial_vorticity(HEIGHT, WIDTH, input, output);
    //parallel_shared_memory_cpu(HEIGHT, WIDTH, input, output);
    parallel_shared_memory_gpu(HEIGHT, WIDTH, input, output, length);
    // Writing output to file
    std::fstream outField("outfield.raw", std::ios::out | std::ios::binary);
    outField.write(reinterpret_cast<char *>(output), length / CHANNELS);
    outField.close();
  } else {
    std::cout << "Didn't open" << std::endl;
  }
  return 0;
}
