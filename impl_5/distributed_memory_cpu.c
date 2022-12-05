/*  
Author: Jerry Zhou
A_number: a02377965

Load Modules: 

module load openmpi cuda

Compile using: 

mpicc -c distributed_memory.c -o main.o
nvcc -c parallel_shared_memory_gpu.cu -o gpu_code.o
mpicc main.o paralle_shared_memory_gpu.cu -o project -lstdc++

Run using: 
    mpiexec -n <core_count> ./project
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <mpi.h>
#include "gpu.h"

/* global variables */
int     tileh; // tile width and height
int     tileid;
int     tilesize; //base tile size with no channel consideration
int     my_rank;
int     core_count;
int     width;
int     height;
int     channels;
int*    sendcounts;
int*    displs;
float*  input;
float*  tempin;
unsigned char*  tempout;
unsigned char*  output;

/* functions */
float vorticity(int x, int y, int width, int height, float *f);

/*--------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
  int i; // ,j,k; //multipurpose ints
  // int counter;
  // int start,end;
  // unsigned char vortChar;
  FILE* pf;
  FILE* wf;

  /* Initiate MPI*/
  MPI_Init(NULL, NULL);

  /* Get rank */
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  /* Find total number of processes */
  MPI_Comm_size(MPI_COMM_WORLD, &core_count);
  //printf("core count: %d\n", core_count);

  printf("MPI initiated\n");
  /* Initiate important variables and arrays for all */
  width = 1300;
  height = 600;
  channels = 2;
  if (height/core_count*core_count == height) {tileh = height/core_count;}  ///////// This seems weird. Does this just check if core_count == 0?
  else {tileh = height/core_count + 1;} ///////// integer division then add one great idea
  sendcounts = malloc(core_count*sizeof(int));
  displs = malloc(core_count*sizeof(int));
  tempin = malloc((tileh+2)*width*channels*sizeof(float)); //// tempin is the tiled data array
  tempout = malloc(tileh*width*sizeof(unsigned char)); //// No need for a data halo on the output. Good call.
  //printf("tileh = %d\n", tileh);
  printf("initial values created memory allocated\n");

  /* Rank 0 initiation work*/
  if (my_rank == 0) {
    // getting data from file
    input = malloc(height*width*channels*sizeof(float));
    output = malloc(height*width*sizeof(float));
    pf = fopen("cyl2d_1300x600_float32[2].raw", "rb");
    fread(input, sizeof(float), height*width*channels, pf);
    fclose(pf);
  } 

  printf("File read, getting values for sendcounts and displs\n");

  /* get values for sendcounts and displs arrays for scatterv */
  for (i = 0; i < core_count; i++) {
    if (core_count == 1) {
      sendcounts[i] = tileh*width*channels; /////////// if only using one core set the single value in sendcounts
    } else if (i == core_count-1) { // at the end
        ///////// is it possible that this could be negative from the adding 1 to tileh
      sendcounts[i] = (height-(core_count-1)*tileh+1)*width*channels; //////// set the last one to take whatever is left
    } else if (i == 0) {  // at the beginning
      sendcounts[i] = (tileh+1)*width*channels; /////// no need to grab a line above if you're at the top
    } else {
      sendcounts[i] = (tileh+2)*width*channels; 
    }
    if (i == 0) {displs[i] = 0;} // displs
    else {displs[i] = (tileh*i-1)*width*channels;}
    //printf("i-%d, sendcount-%d, displs-%d\n",i,sendcounts[i],displs[i]);
  }
  
  printf("Scattering data \n");

  // send data out to all cores
  MPI_Scatterv(input, sendcounts, displs, MPI_FLOAT, tempin, sendcounts[my_rank], MPI_FLOAT,0, MPI_COMM_WORLD); 
  
  /*
  for (i = 0; i < 600; i++) {
    for (j = 0; j < 1300; j++) {
      printf("tempin: %4.3f, %4.3f;", tempin[i*width+j], tempin[i*width+j+1]);
    }
  }
  for (i = 0; i < 600; i++) {
    for (j = 0; j < 1300; j++) {
      printf("tempout: %d;", tempout[i*width+j]);
    }
  }
  */

  printf("Calculating vorticity \n");

  /* calculating vorticity */ 
  parallel_shared_memory_gpu(height, width, tempin, tempout, height*width*channels);
  // if (core_count == 1) {// using k as "height" of the tempin
  //   k = tileh;
  //   start = 0;
  //   end = k;
  // } else if (my_rank == core_count-1) {
  //   k = height-(core_count-1)*tileh+1;
  //   start = 1;
  //   end = k;
  // } else if (my_rank == 0) {
  //   k = tileh+1;
  //   start = 0;
  //   end = k-1;
  // } else {
  //   k = tileh+2;
  //   start = 1;
  //   end = k-1;
  // }
  // counter = 0;
  // //printf("k: %d\n",k);
  // //printf("start: %d\n",start);
  // //printf("end: %d\n",end);
  // for (i = start; i < end; i++) { // choose row
  //   for (j = 0; j < width; j++) { // go through elements in row i
  //     //printf("(%d, %d)-%d;", j, i, counter);
  //     // errors at 
  //     float vort = vorticity(j, i, width, height, tempin);
  //     if (vort < -0.2f) {
  //       vortChar = 0;
  //     } else if (vort > 0.2f) {
  //       vortChar = 127;
  //     } else {
  //       vortChar = 255;
  //     }
  //     tempout[counter] = vortChar;
  //     counter++;
  //   }
  // }

  printf("Calculating sendcounts and displs again\n"); //////// are they different now? Would this be because now you don't have channels?

  /* get values for sendcounts and displs arrays for gatherv */
  for (i = 0; i < core_count; i++) {
    if (core_count == 1) {
      sendcounts[i] = tileh*width;
    } else if (i == core_count-1) { // at the end
      sendcounts[i] = (height-(core_count-1)*tileh+1)*width;
    } else {
      sendcounts[i] = tileh*width; 
    }
    if (i == 0) {displs[i] = 0;} // displs
    else {displs[i] = (tileh*i-1)*width;}
    //printf("i-%d, sendcount-%d, displs-%d\n",i,sendcounts[i],displs[i]);
  }

  printf("Gathering data \n");

  /* collecting data from all cores*/
  MPI_Gatherv(tempout, sendcounts[my_rank], MPI_UNSIGNED_CHAR, output, sendcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  printf("Writing outfile\n");

  /* writing outfile */
  if (my_rank == 0) {
    wf = fopen("outfield.raw", "wb");
    fwrite(output, sizeof(unsigned char), height*width, wf);
    fclose(wf);

    free(input);
    free(output);
  }

  printf("Clean up and finalizing\n");

  /* cleanup */
  free(sendcounts);
  free(displs);
  free(tempin);
  free(tempout);
  
  MPI_Finalize();

  return 0;

}


float vorticity(int x, int y, int width, int height, float *f) {
  float dx = 0.01;
  float dy = 0.01;

  uint32_t idx = y * width + x;

  int start_x = (x == 0) ? 0 : x - 1;
  int end_x = (x == width - 1) ? x : x + 1;

  int start_y = (y == 0) ? 0 : y - 1;
  int end_y = (y == height - 1) ? y : y + 1;

  uint32_t duidx = (start_y * width + end_x) * 2;
  uint32_t dvidx = (end_y * width + start_x) * 2;

  float duy = (f[duidx + 1] - f[idx * 2 + 1]) / (dx * (end_x - start_x));
  float dvx = (f[dvidx] - f[idx * 2]) / (dy * (end_y - start_y));

  return duy - dvx;
}

