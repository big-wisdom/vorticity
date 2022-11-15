#include <cstring>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>
#include "vorticity.cpp"

#define WIDTH 1300
#define HEIGHT 600

int main() {
  std::ifstream vectorField("cyl2d_1300x600_float[2].raw", std::ios::in);
  if (vectorField.is_open()) {
    // Get the length of the image should be 3145728
    vectorField.seekg(0, std::ios_base::end);
    auto length = vectorField.tellg();
    vectorField.seekg(0, std::ios::beg);

    // Initialize arrays
    float *input = new float[length];
    uint32_t *output = new uint32_t[length / 2];

    // Get rgb values from image into input array
    vectorField.read((float *)input, length);
    vectorField.close();

  }

  // CPU serial implementation
  for (int i = 0; i < WIDTH; i++) {
    for (int j = 0; j < HEIGHT; j++) {
      output[i*WIDTH + j] = vorticity(i, j, WIDTH, HEIGHT, input);
    }
  }

  // Writing output to file
  std::fstream outField ("outfield.raw", std::ios::out|std::ios::binary);
  outField.write(reinterpret_cast<char *> (outputGPUGlobal), length);
  outField.close();


  return 0;
}