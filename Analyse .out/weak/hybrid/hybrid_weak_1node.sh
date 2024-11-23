#!/bin/bash
#SBATCH --job-name="hybrid_weak_1node"
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --partition=hmem
#SBATCH --output=hybrid_weak_1node.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64  

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
#THREADS=(2 4 8)
#RANKS=(8 4 2 1)

#couple of threads and ranks to execute 
RANKS=(1 2 4 8 16 32 1 2 4 8 16 1 2 4 8)
THREADS=(2 2 2 2 2 2 4 4 4 4 4 8 8 8 8)

# Loop over the indices of the arrays
for i in "${!RANKS[@]}"; do
    NRANKS=${RANKS[$i]}
    NTHREADS=${THREADS[$i]}
    export OMP_NUM_THREADS=${NTHREADS}

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
    for ((j=1; j<=5; j++)); do
        echo "Run ${j} with ${NRANKS} MPI ranks and ${NTHREADS} threads..."
        mpirun -np ${NRANKS} ${EXEC_FILE} ${PARAM_FILE}
    done
done