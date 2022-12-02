#include "vorticity.hpp"
#include <float.h>

void mapFloatToChar(float * f_matrix, unsigned char * output, int HEIGHT, int WIDTH) {
    float min = FLT_MAX;
    float max = FLT_MIN;
    for (int y=0; y<HEIGHT; y++) {
        for (int x=0; x<WIDTH; x++) {
            float float_value = f_matrix[(y * WIDTH) + x];
            if (float_value < min) min = float_value;
            if (float_value > max) max = float_value;
        }
    }
    float range = max - min;
    for (int y=0; y<HEIGHT; y++) {
        for (int x=0; x<WIDTH; x++) {
            output[(WIDTH*y) + x] = (unsigned char)(((f_matrix[(y*WIDTH) + x] - min) / range) * 255);
        }
    }
}

void serial_vorticity(int HEIGHT, int WIDTH, float* input, unsigned char * output) {
    float vort[HEIGHT * WIDTH];
    // CPU serial implementation
    for (int i = 0; i < HEIGHT; i++) {
      for (int j = 0; j < WIDTH; j++) {
        vort[(WIDTH * i) + j] = vorticity(j, i, WIDTH, HEIGHT, input);
      }
    }
    mapFloatToChar(vort, output, HEIGHT, WIDTH);
}

bool validate(int HEIGHT, int WIDTH, unsigned char * test, unsigned char * valid) {
    for (int i=0; i < HEIGHT; i++) {
        for (int j=0; j<WIDTH; j++) {
            if (test[(i * WIDTH) + j] != valid[(i * WIDTH) + j]) {
                return false;
            }
        }
    }
    return true;
}
