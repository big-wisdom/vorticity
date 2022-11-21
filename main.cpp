#include "vorticity.hpp"
#include "implementations.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>

#define WIDTH 1300
#define HEIGHT 600
#define CHANNELS 2

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
    unsigned char *output = new unsigned char[length / CHANNELS];

    // Get rgb values from image into input array
    vectorField.read((char *)input, length);
    vectorField.close();

    // serial_vorticity(HEIGHT, WIDTH, input, output);
    parallel_shared_memory_cpu(HEIGHT, WIDTH, input, output);

    // Writing output to file
    std::fstream outField("outfield.raw", std::ios::out | std::ios::binary);
    outField.write(reinterpret_cast<char *>(output), length / CHANNELS);
    outField.close();
  } else {
    std::cout << "Didn't open" << std::endl;
  }
  return 0;
}
