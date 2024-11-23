#!/bin/bash
#SBATCH --job-name="shallow_water_simulation"
#SBATCH --output=shallow_water_simulation_1t_gpu.out
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --gpus=1
#SBATCH --time=15:00
#SBATCH --account=ulghpsc

module load EasyBuild/2023a
module load Clang/18.1.8-GCCcore-12.3.0-CUDA-12.2.0

# Set environment variable to ensure offloading is mandatory
export OMP_TARGET_OFFLOAD=MANDATORY



# Compile the source file
clang -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -o shallow_gpu shallow_gpu.c -lm
for
    ./shallow_gpu param_simple_no_output.txt
# Run the executable with the parameter file
#./shallow_gpu param_simple_no_output.txt


# To submit a job interactively, you can use the following command:
# srun --partition=debug-gpu --account=ulghpsc --time=01:00:00 --gpus=1 --pty $SHELL
# This will allocate resources and open an interactive shell where you can manually run your commands.

