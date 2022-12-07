#include "vorticity.h"
#include <omp.h>
#include <time.h>

float parallel_shared_memory_cpu(int HEIGHT, int WIDTH, float * input, unsigned char * output, int thread_count) {
  int i, j;
  float time;
  clock_t start, end;
  start = clock();
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
  end = clock();
  time = ((double) (end - start)) / CLOCKS_PER_SEC;
  return time;
}
