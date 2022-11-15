#include <cstring>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>
#include <vorticity.cpp>

int main() {
  std::ifstream vectorField("cyl2d_1300x600_float[2].raw", std::ios::binary);
  if (vectorField.is_open()) {
    // Get the length of the image should be 3145728
    vectorField.seekg(0, std::ios_base::end);
    auto length = vectorField.tellg();
    vectorField.seekg(0, std::ios::beg);

    // Initialize arrays
    float *input = new float[length];

    // Get rgb values from image into input array
    vectorField.read((char *)input, length);
    vectorField.close();

    return 0;
  }
}