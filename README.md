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

# Timing Study
## Timing of 1 vs 2 vs 3
The timing of the first implementation vs the second vs the third reveal some expected behavior and 
some unbehaviour. As the data input increased in size the time to run the code also increase with a 
linear scale to the data increase size. This is seen in all 3 implementations. We don't see strong 
scaling in the shared cpu implementation. There is actually no increase in speed from increasing 
the number of threads in the parallel cpu implementation. This is unexpected behaviour, but we 
imagine it is due to some serial part of the code being needed to run on every number of threads that 
takes the same amount of time. The speed up from the multithreading does not increase the time, because
the serial part of the code requires the same amount of time. 

The size of the blocks for implementation 3, or the shared GPU implementation shows a increase in time 
with problem size. It also shows a decrease with an increased number of threads up to a certain point 
when the number of threads is to high and it takes longer to initialize threads than to run the data. 
A decrease by half the size of the block, so therefore a doubling of threads, does not result in a halving
of the time, but there is speedup and that is what is expected from a shared GPU implementation. 

## Timing of 4 vs 5
For the timing of 4 and 5, two cases are studied. The first is the time spent using k number of cores for 
k times the image size, thereby keeping constant the data size processed per core. This first case is done
as a validation study to ensure that the implementation behaves as expected. The second case is the time
spent using k number of cores for some fixed amount of data size. This second case is done to observe any 
possible speedup that results from using MPI and/or MPI+Cuda.

For the first case in the 4vs5-timevssizecore figure, the distributed memory GPU line in orange clearly shows 
that the time spent clearly does not scale correctly with the number of cores and the data size. We expected
the trend showed by the distributed memory CPU line in blue, which is mostly constant. We would have liked to 
see a slightly negative slope for the distributed memory CPU line, however, as can be seen in the 
4-timevssizecore figure the data points are mostly constant with a difference that can be attributed to 
slight hardware and iterative differences.

For the second case in the 4vs5-timevscore figure, the distributed memory GPU line in orange again shows a
positive slope, which is completely not we expected. The distributed memory CPU line in blue shows an
exponential negative slope, which is promising. A better figure, 4-timevscore, more clearly shows that the 
distributed memroy CPU implementation speeds up dramatically initially and eventually becomes less
effective as more cores are added. The diminishing returns are a likely effect of the increase in overhead
as more cores need to communicate. The degree of speedup can be observed more clearly with the figure
4-timevscorelog, which has log scaling on the y-axis. Since the line for the log scale figure still looks
negative exponential, we know that there is some initial superlinear speedup, which then dissipates. 
