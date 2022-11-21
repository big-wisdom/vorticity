#ifndef VORTICITY
#define VORTICITY
#include <cstdint>
#include <utility>


typedef std::pair<double, double> Vec2d;

/*
        Compute vorticity feature at cell coordinate (x, y) of the field f
*/
float vorticity(int x, int y, int width, int height, float *f);

#endif
