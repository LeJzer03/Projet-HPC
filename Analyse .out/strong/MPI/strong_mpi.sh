#!/bin/bash
#SBATCH --job-name="mpi_strong"
#SBATCH --exclusive
#SBATCH --partition=hmem
#SBATCH --mem=0
#SBATCH --output=strong_mpi.out
#SBATCH --time=48:00:00
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

PROCS=(64 32 16 8 4 2 1)  # Define your process count here

# Loop over the different process counts
for PROCS_NUM in "${PROCS[@]}"; do
    echo "Running with ${PROCS_NUM} processes ..."

    # Inner loop to run the program multiple times for each configuration
    for ((i=1; i<=5; i++)); do
        echo "Run ${i} ..."

        # Run the MPI program with the current number of processes
        mpirun -np ${PROCS_NUM} ${EXEC_FILE} ${PARAM_FILE}
    done
done