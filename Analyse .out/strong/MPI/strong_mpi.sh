#!/bin/bash
#SBATCH --job-name="mpi_strong"
#SBATCH --exclusive
#SBATCH --partition=hmem
#SBATCH --mem=0
#SBATCH --output=strong_mpi.out
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64  

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

# Loop over the different process counts
echo "Running with 64 processes ..."
for ((i=1; i<=5; i++)); do
    echo "Run ${i} ..."
    mpirun -np 64 ${EXEC_FILE} ${PARAM_FILE}
done
echo "Running with 32 processes ..."
for ((i=1; i<=5; i++)); do
    echo "Run ${i} ..."
    mpirun -np 32 ${EXEC_FILE} ${PARAM_FILE}
done
echo "Running with 16 processes ..."
for ((i=1; i<=5; i++)); do
    echo "Run ${i} ..."
    mpirun -np 16 ${EXEC_FILE} ${PARAM_FILE}
done
echo "Running with 8 processes ..."
for ((i=1; i<=5; i++)); do
    echo "Run ${i} ..."
    mpirun -np 8 ${EXEC_FILE} ${PARAM_FILE}
done
echo "Running with 4 processes ..."
for ((i=1; i<=5; i++)); do
    echo "Run ${i} ..."
    mpirun -np 4 ${EXEC_FILE} ${PARAM_FILE}
done
echo "Running with 2 processes ..."
for ((i=1; i<=5; i++)); do
    echo "Run ${i} ..."
    mpirun -np 2 ${EXEC_FILE} ${PARAM_FILE}
done
echo "Running with 1 processes ..."
for ((i=1; i<=5; i++)); do
    echo "Run ${i} ..."
    mpirun -np 1 ${EXEC_FILE} ${PARAM_FILE}
done