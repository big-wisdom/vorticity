# Compilation
## Instructions for implentations 1-3
The folder "vorticity_1-3" has all necessary source code for implementations 1-3 of the vorticity function. 
The make file will compile all of them into an executable called vorticity. Compile and execute with 
the following commands.

Copy the folder into the CHPC environment and import modules, compile with a make command

    module load cuda
    make

Then run the bash script to queue the job

    ./queue_job

## Instructions for implementation 4
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

## Instructions for implementation 5
Load Modules: 

module load gcc/11.2.0
module load openmpi cuda

Compile using: 

mpicc -c distributed_memory.c -o main.o
nvcc -c parallel_shared_memory_gpu.cu -o gpu_code.o
mpicc main.o gpu_code.o -lcudart -o project -lstdc++

Run using: 
    mpiexec -n <core_count> ./project

(NOTE: right now it works for core_count 2. This was another thing we planned to make more robust 
but we couldn't write code or test it because CHPC was down.)

# Approach
## Implementation 1: Serial Vorticity
The approach we took to run the serial code is to just travel along the entire image 
row by row and call the vorticity function at each location and send that to the output. 
## Implementation 2: Parallel shared memory CPU
This implementation was very similar to the serial version. All we really had to do was add an omp statement to divide up the for loop between different threads.
## Implementation 3: Parallel shared memory GPU
The approach we took was to read in the input file 
and then call the kernal. The kernal will take each 
point and place itself into shared memory. If the 
point is on one of the four edges it grabs that point
as well, assuming it is not the edge of the input vector 
field. 

Once the shared memory tile is filled out the code calls
the vorticity for each point using the vortTile. The 
code then places the vorticity output into the final 
image, and ends. 
## Implementation 4: Distributed memory CPU
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
## Implementation 5: Distributed memory GPU
This implementation was really just connecting implementation 3 to the vorticity solving part of
implementation number 4. There were of course some modifications to make so that data would be
located correctly as now the size of the sub plot that the GPU recieves is smaller. Then after
figuring out how to compile them together, it works like a charm.
(NOTE: right now there is a line between nodes because we weren't able to test our code for 
the last day before the due date. That's also why there is no timing or validation function
on it because that's what we planned to do today, but CHPC is down.)
