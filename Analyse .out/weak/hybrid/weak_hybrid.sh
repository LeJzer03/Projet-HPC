#!/bin/bash
#SBATCH --job-name="hybrid_weak_scaling"
#SBATCH --exclusive
#SBATCH --partition=hmem
#SBATCH --mem=0
#SBATCH --output=hybrid_weak_scaling.out
#SBATCH --time=48:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64  # Max tasks per node, adjust as needed

module load OpenMPI

PROJECT_DIR="${HOME}/project_info0939"
EXEC_DIR="${PROJECT_DIR}/bin"
INPUT_DIR="${PROJECT_DIR}/example_inputs/simple"
PARAM_SOURCE="${PROJECT_DIR}/workdir/MPI+open_mp/param_simple.txt"  # Source param file path
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/simulation_results"
SOURCE_FILE="${PROJECT_DIR}/workdir/MPI+open_mp/shallow_mpi+open_mp.c"
EXEC_FILE="${EXEC_DIR}/shallow_hybrid"

# Ensure the EXEC_DIR exists
mkdir -p ${EXEC_DIR}

# Compilation step
mpicc -O3 -Wall -fopenmp -o ${EXEC_FILE} ${SOURCE_FILE} -lm

#THREADS=(2 4 8)  # OpenMP threads
#RANKS=(1 2 4 8 16 32 64)  # MPI ranks
THREADS=(2 4 8)
RANKS=(8 4 2 1)

# Loop over the number of threads
for NTHREADS in "${THREADS[@]}"; do
    export OMP_NUM_THREADS=${NTHREADS}

    # Loop over the number of ranks
    for NRANKS in "${RANKS[@]}"; do
        echo "Running with ${NRANKS} ranks and ${NTHREADS} threads..."

        # Calculate the scaling factor for the spatial dimension
        SCALING_FACTOR=$(echo "scale=6; sqrt(${NRANKS} * ${NTHREADS})" | bc -l)
        NEW_DX=$(echo "scale=6; 5.0 / ${SCALING_FACTOR}" | bc -l)  # Adjust the base spatial step as needed

        # Create a specific parameter file for this configuration
        PARAM_FILE="${OUTPUT_DIR}/param_${NRANKS}ranks_${NTHREADS}threads.txt"
        cp "${PARAM_SOURCE}" "${PARAM_FILE}"
        # Update the spatial step in the parameter file
        sed -i "1s/.*/${NEW_DX}/" "${PARAM_FILE}"  # Update DX on the first line
        sed -i "2s/.*/${NEW_DX}/" "${PARAM_FILE}"  # Update DY on the second line

        # Run the application multiple times for averaging results
        for ((i=1; i<=1; i++)); do
            echo "Run ${i} with ${NRANKS} MPI ranks and ${NTHREADS} threads..."
            mpirun -np ${NRANKS} ${EXEC_FILE} ${PARAM_FILE}
        done
    done
done
