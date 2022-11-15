#include <cstring>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>

int main() {
  std::ifstream image("gc_1024x1024.raw", std::ios::binary);
  if (image.is_open()) {
    // Get the length of the image should be 3145728
    image.seekg(0, std::ios_base::end);
    auto length = image.tellg();
    image.seekg(0, std::ios::beg);

    // Initialize arrays
    unsigned char *input = new unsigned char[length];
    unsigned char *outputGPUGlobal = new unsigned char[length];
    unsigned char *outputGPUShared = new unsigned char[length];
    unsigned char *outputCPU = new unsigned char[length];

    // Get rgb values from image into input array
    image.read((char *)input, length);
    image.close();

    return 0;
  }
}