/*  
Author: Jerry Zhou
A_number: a02377965

Instructions for use: this code was ran successfully on the notchpeak
distribution of chpc. To use this code, load the following modules 
(in order) in the batch file or in an interactive node:
[Load Modules]: module load gcc/8.5.0
              module load intel-mpi
Once the modules are loaded, the code can be compiled using the line below
[Compile using]: mpicc -g -Wall -o project distributed_memory_cpu.cpp
The compiler should output no errors nor warnings. Finally, run the code
using the line below, where the only user input is <core_count>
[Run using]: 
    mpiexec -n <core_count> ./project

My approach with the MPI code was the separate the image purely by the rows,
thereby flattening the 2D image into a 1D line of rows. The main benefit
for this approach is that it is much easier to code and debug since most
of the data manipulation is in 1D (there would be less worry about column
indices). However, this has a major drawback in the amount of data needed 
for haloing the data for the cores. Since everything is in the unit of rows,
the size of the halo is necessarily also in the unit of rows. Compared to a
rectangular or square block approach, whose maximum halo size is 
2*tile_width + 2*tile_height, the maximum halo size for my implementation is
2*total_width. Then, for instances where tile_height + tile_width < total width,
the halo size for my implementation would be larger, which uses more memory
and therefore would be much less feasible for an extremely wide image.

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
//#include <sys/time.h>

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
  int i,j,k; //multipurpose ints
  int counter;
  int start,end;
  unsigned char vortChar;
  FILE* pf;
  FILE* wf;
  //struct timeval start, end;
  //float time_spent;

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
  if (height/core_count*core_count == height) {tileh = height/core_count;}
  else {tileh = height/core_count + 1;}
  sendcounts = malloc(core_count*sizeof(int));
  displs = malloc(core_count*sizeof(int));
  tempin = malloc((tileh+2)*width*channels*sizeof(float));
  tempout = malloc(tileh*width*sizeof(unsigned char));
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
      sendcounts[i] = tileh*width*channels;
    } else if (i == core_count-1) { // at the end
      sendcounts[i] = (height-(core_count-1)*tileh+1)*width*channels;
    } else if (i == 0) {  // at the beginning
      sendcounts[i] = (tileh+1)*width*channels;
    } else {
      sendcounts[i] = (tileh+2)*width*channels; 
    }
    if (i == 0) {displs[i] = 0;} // displs
    else {displs[i] = (tileh*i-1)*width*channels;}
  }
  
  printf("Scattering data \n");

  // send data out to all cores
  MPI_Scatterv(input, sendcounts, displs, MPI_FLOAT, tempin, sendcounts[my_rank], MPI_FLOAT,0, MPI_COMM_WORLD); 
  
  //gettimeofday(&start, NULL); // start timer
  printf("Calculating vorticity \n");

  /* calculating vorticity */ 
  if (core_count == 1) {// using k as "height" of the tempin
    k = tileh;
    start = 0;
    end = k;
  } else if (my_rank == core_count-1) {
    k = height-(core_count-1)*tileh+1;
    start = 1;
    end = k;
  } else if (my_rank == 0) {
    k = tileh+1;
    start = 0;
    end = k-1;
  } else {
    k = tileh+2;
    start = 1;
    end = k-1;
  }
  counter = 0;

  for (i = start; i < end; i++) { // choose row
    for (j = 0; j < width; j++) { // go through elements in row i
      float vort = vorticity(j, i, width, height, tempin);
      if (vort < -0.2f) {
        vortChar = 0;
      } else if (vort > 0.2f) {
        vortChar = 127;
      } else {
        vortChar = 255;
      }
      tempout[counter] = vortChar;
      counter++;
    }
  }

  printf("Calculating sendcounts and displs again\n");

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
  }

  //gettimeofday(&end,NULL); // end timer
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

  //time_spent = (end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec)/1000000.0;
  //printf("Total time spent on core %d: %f sec\n",my_rank, time_spent);
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

