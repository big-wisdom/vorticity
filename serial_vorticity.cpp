#include "vorticity.hpp"

void serial_vorticity(int HEIGHT, int WIDTH, float* input, unsigned char * output) {
    // CPU serial implementation
    for (int i = 0; i < HEIGHT; i++) {
      for (int j = 0; j < WIDTH; j++) {
        float vort = vorticity(j, i, WIDTH, HEIGHT, input);
        unsigned char vortChar;
        if (vort < -0.2f) {
          vortChar = 0;
        } else if (vort > 0.2f) {
          vortChar = 127;
        } else {
          vortChar = 255;
        }
        output[i * WIDTH + j] = vortChar;
      }
    }
}
