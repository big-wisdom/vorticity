#include "vorticity.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>

#define WIDTH 1300
#define HEIGHT 600

int main() {
  std::ifstream vectorField("cyl2d_1300x600_float32[2].raw", std::ios::binary);
  if (vectorField.is_open()) {
    // Get the length of the image should be 3145728
    std::cout << "opened" << std::endl;
    vectorField.seekg(0, std::ios_base::end);
    auto length = vectorField.tellg();
    vectorField.seekg(0, std::ios::beg);

    // Initialize arrays
    float *input = new float[length];
    uint32_t *output = new uint32_t[length / 2];

    // Get rgb values from image into input array
    vectorField.read((char *)input, length);
    vectorField.close();

    // CPU serial implementation
    for (int i = 0; i < WIDTH; i++) {
      for (int j = 0; j < HEIGHT; j++) {
        output[i * WIDTH + j] = vorticity(i, j, WIDTH, HEIGHT, input);
      }
    }

    // Writing output to file
    std::fstream outField("outfield.raw", std::ios::out | std::ios::binary);
    outField.write(reinterpret_cast<char *>(output), length);
    outField.close();
  } else {
    std::cout << "Didn't open" << std::endl;
  }
  return 0;
}