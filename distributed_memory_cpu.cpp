/*  
Author: Jerry Zhou
A_number: a02377965

Load Modules: module load gcc/8.5.0
              module load intel-mpi
Compile using: mpicc -g -Wall -o project distributed_memory_cpu.c
Run using: 
    mpiexec -n <core_count> ./project
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "vorticity.cpp"

#define WIDTH 1300
#define HEIGHT 600
#define CHANNELS 2

/* global variables */
int     tileh; // tile width and height
int     tileid;
int     tilesize; //base tile size with no channel consideration
int*    sendcounts;
int*    displs;
float*  input;
float*  tempin;
float*  tempout;
unsigned char*  output;

/* functions */

/*--------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
    int i,j,k; //multipurpose ints
    MPI_Status status; 
    
    /* Initiate MPI*/
    MPI_Init(NULL, NULL);

    /* Get rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Find total number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &core_count);

    /* Rank 0 initiation work*/
    if (my_rank == 0) {

        // getting data from file
        std::ifstream vectorField("cyl2d_1300x600_float32[2].raw", std::ios::binary);
        if (vectorField.is_open()) {
            // Get the length of the image should be 3145728
            std::cout << "opened" << std::endl;
            vectorField.seekg(0, std::ios_base::end);
            auto length = vectorField.tellg();
            vectorField.seekg(0, std::ios::beg);

            // Initialize arrays
            float *input = new float[length];
            unsigned char *output = new unsigned char[length / CHANNELS];

            // Get rgb values from image into input array
            vectorField.read((char *)input, length);
            vectorField.close();
        }
    } 

    /* broadcast variables and allocate memory to prep for scatterv*/
    tileh = HEIGHT/core_count + 1;
    sendcounts = malloc(core_count*sizeof(int));
    displs = malloc(core_count*sizeof(int));
    tempin = malloc(tileh*WIDTH*CHANNELS*sizeof(float));
    tempout = malloc(tileh*WIDTH*sizeof(unsigned char));

    for (i == 0; i < core_count; i++) {
        if (i == core_count-1) {
            sendcounts[i] = (HEIGHT-(core_count-1)*tileh)*WIDTH*CHANNELS;
        } else {
            sendcounts[i] = tileh*WIDTH*CHANNELS;
        }
        if (i != 0) {displs[i] = displs[i-1] + sendcounts[i];}
    }
    
    // send data out to all cores
    MPI_Scatterv(input, sendcounts, displs, MPI_FLOAT,
                 tempin, sendcounts[my_rank], MPI_FLOAT, MPI_COMM_WORLD); 

    /* serial implementation of vorticity plot */ 
    if (my_rank == core_count-1) {k = HEIGHT-(core_count-1)*tileh;}
    else {k = tileh;}
    // using k as "height"
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < WIDTH; j++) {
        float vort = vorticity(j, i, WIDTH, HEIGHT, tempin);
        unsigned char vortChar;
        if (vort < -0.2f) {
          vortChar = 0;
        } else if (vort > 0.2f) {
          vortChar = 127;
        } else {
          vortChar = 255;
        }
        tempout[i * WIDTH + j] = vortChar;
      }
    }

    /* collecting data from all cores*/
    MPI_Gather(tempout, sendcounts[my_rank]/2, MPI_UNSIGNED_CHAR, 
        output, sendcounts[my_rank]/2, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    /* writing outfile */
    if (my_rank == 0) {
        std::fstream outField("outfield.raw", std::ios::out | std::ios::binary);
        outField.write(reinterpret_cast<char *>(output), length / CHANNELS);
        outField.close();

        free(input);
        free(output);
    }

    /* cleanup */
    free(sendcounts);
    free(displs);
    free(tempin);
    free(tempout);
    
    MPI_Finalize();

    return 0;

}


