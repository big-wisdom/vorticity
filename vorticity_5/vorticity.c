#include "vorticity.h"

/*
        Compute vorticity feature at cell coordinate (x, y) of the field f
*/
float vorticity(int x, int y, int width, int height, float *f) {
  float dx = 0.01;
  float dy = 0.01;

  int idx = y * width + x;

  int start_x = (x == 0) ? 0 : x - 1;
  int end_x = (x == width - 1) ? x : x + 1;

  int start_y = (y == 0) ? 0 : y - 1;
  int end_y = (y == height - 1) ? y : y + 1;

  int duidx = (start_y * width + end_x) * 2;
  int dvidx = (end_y * width + start_x) * 2;

  // Vec2d fdu(f[duidx], f[duidx + 1]);
  // Vec2d fdv(f[dvidx], f[dvidx + 1]);
  // Vec2d vec0(f[idx * 2], f[idx * 2 + 1]);

  double fdu[2] = {f[duidx], f[duidx + 1]};
  double fdv[2] = {f[dvidx], f[dvidx + 1]};
  double vec0[2] = {f[idx * 2], f[idx * 2 + 1]};

  float duy = (fdu[1] - vec0[1]) / (dx * (end_x - start_x));
  float dvx = (fdv[0] - vec0[0]) / (dy * (end_y - start_y));

  return duy - dvx;
}
