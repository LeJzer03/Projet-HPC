#!/bin/bash
#SBATCH --job-name="mpi_strong_1"
#SBATCH --exclusive
#SBATCH --output=strong_mpi_bigdomain_1n.out
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64  # Set the number of tasks per node

module load OpenMPI

PROJECT_DIR="${HOME}/project_info0939"
EXEC_DIR="${PROJECT_DIR}/bin"
INPUT_DIR="${PROJECT_DIR}/example_inputs/simple"
PARAM_FILE="${INPUT_DIR}/param_simple_no_output.txt"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/simulation_results"
SOURCE_FILE="${PROJECT_DIR}/workdir/MPI/shallow_mpi.c"
EXEC_FILE="${EXEC_DIR}/shallow_mpi"

# Ensure the EXEC_DIR exists
mkdir -p ${EXEC_DIR}

# Compilation step
mpicc -O3 -Wall -o ${EXEC_FILE} ${SOURCE_FILE} -lm

mpirun -np 64 ${EXEC_FILE} ${PARAM_FILE}

