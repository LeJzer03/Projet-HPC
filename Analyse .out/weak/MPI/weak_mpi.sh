#!/bin/bash
#SBATCH --job-name="mpi_weak"
#SBATCH --exclusive
#SBATCH --partition=hmem
#SBATCH --mem=0
#SBATCH --output=weak_mpi.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64  # Set the number of tasks per node

module load OpenMPI

PROJECT_DIR="${HOME}/project_info0939"
EXEC_DIR="${PROJECT_DIR}/bin"
INPUT_DIR="${PROJECT_DIR}/example_inputs/simple"
PARAM_FILE="param_simple.txt"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/simulation_results"
PARAM_SOURCE_TEST = "${PROJECT_DIR}/workdir/MPI"
SOURCE_FILE="${PROJECT_DIR}/workdir/MPI/shallow_mpi.c"
EXEC_FILE="${EXEC_DIR}/shallow_mpi"

# Ensure the EXEC_DIR exists
mkdir -p ${EXEC_DIR}

# Compilation step
mpicc -O3 -Wall -o ${EXEC_FILE} ${SOURCE_FILE} -lm

#PROCS=(64 32 16 8 4 2 1)  # Define your process count here
PROCS=(64)
# Loop over the different process counts
for PROCS_NUM in "${PROCS[@]}"; do
    echo "Running with ${PROCS_NUM} processes ..."

    # Calculate the problem size for weak scaling 
    DX=$(echo "scale=6; 5 / sqrt($PROCS_NUM)" | bc -l)

    # Create a specific parameter file for this configuration
    PARAM_FILE="param_${PROCS_NUM}procs.txt"
    cp ${PARAM_SOURCE_TEST}/param_simple.txt ${PARAM_FILE}

    # Update the first value in the param file to adjust the problem size
    sed -i "1s/.*/${DX}/" ${PARAM_FILE}
    sed -i "2s/.*/${DX}/" ${PARAM_FILE}

    # Inner loop to run the program multiple times for each configuration
    for ((i=1; i<=5; i++)); do
        echo "Run ${i} ..."

        # Run the MPI program with the current number of processes
        mpirun -np ${PROCS_NUM} ${EXEC_FILE} ${PARAM_FILE}
    done
done