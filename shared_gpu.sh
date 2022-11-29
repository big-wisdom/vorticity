#!/bin/bash
#SBATCH --time=00:03:00
#SBATCH --nodes=1
#SBATCH -o slurmjob-%j.out-%N 
#SBATCH -e slurmjob-%j.err-%N 
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak
#SBATCH --gres=gpu

#Run the program with input
./vorticityGpu 