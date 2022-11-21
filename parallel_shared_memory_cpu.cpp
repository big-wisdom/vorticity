// NOTE FROM ELI: right now this is just the same as the serial statement, I'm about to start working on the omp statement
// that I just copied in

#include "vorticity.hpp"
#include <omp.h>

int thread_count = 4;

void parallel_shared_memory_cpu(int HEIGHT, int WIDTH, float * input, unsigned char * output) {
// #pragma omp parallel num_threads(thread_count) default(none) shared(temp, a, n) private(i, j, count)
    {
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
}
