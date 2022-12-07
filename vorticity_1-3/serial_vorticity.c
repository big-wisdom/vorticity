/*
  The approach we took to run the serial code is to just travel along the entire image 
  row by row and call the vorticity function at each location and send that to the output. 
*/


#include "vorticity.h"
#include <stdbool.h>
#include <time.h>

float serial_vorticity(int HEIGHT, int WIDTH, float* input, unsigned char * output) {
    // CPU serial implementation
    float time;
    clock_t start, end;
    start = clock();
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
    end = clock();
    time = ((double) (end - start)) / CLOCKS_PER_SEC;
    return time;
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
