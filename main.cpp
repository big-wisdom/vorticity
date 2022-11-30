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

    // Get rgb values from image into input array
    float *input = new float[length];
    vectorField.read((char *)input, length);
    vectorField.close();

    // create output and valid arrays
    unsigned char *output = new unsigned char[length / CHANNELS]; // we probably need to clean this up because we use the new keyword
    unsigned char *valid = new unsigned char[length / CHANNELS];

    // create valid with serial algorithm
    serial_vorticity(HEIGHT, WIDTH, input, valid);

    // Parallel shared memory
    parallel_shared_memory_cpu(HEIGHT, WIDTH, input, output);

    if (validate(HEIGHT, WIDTH, output, valid))
        printf("Parallel shared memory valid\n");
    else
        printf("Parallel shared memory invalid\n");

    // Writing output to file
    std::fstream outField("outfield.raw", std::ios::out | std::ios::binary);
    outField.write(reinterpret_cast<char *>(output), length / CHANNELS);
    outField.close();

  } else {
    std::cout << "Didn't open" << std::endl;
  }
  return 0;
}
