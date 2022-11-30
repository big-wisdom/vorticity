// NOTE FROM ELI: right now this is just the same as the serial statement, I'm about to start working on the omp statement
// that I just copied in

#include "vorticity.hpp"
#include <omp.h>

int thread_count = 4;

void parallel_shared_memory_cpu(int HEIGHT, int WIDTH, float * input, unsigned char * output) {
    int i, j;
    #pragma omp parallel num_threads(thread_count) default(none) shared(output, input, HEIGHT, WIDTH) private(i, j)
    {
        #pragma omp for collapse(2)
        for (i = 0; i < HEIGHT; i++) {
          for (j = 0; j < WIDTH; j++) {
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
