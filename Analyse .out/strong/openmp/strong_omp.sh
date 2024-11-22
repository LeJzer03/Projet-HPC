#!/bin/bash
#SBATCH --job-name="omp_strong"
#SBATCH --exclusive
#SBATCH --partition=hmem
#SBATCH --mem=0
#SBATCH --output=strong_omp.out
#SBATCH --time=24:00:00
#SBATCH --nodes=2

PROJECT_DIR="${HOME}/project_info0939"
EXEC_DIR="${PROJECT_DIR}/bin"
INPUT_DIR="${PROJECT_DIR}/example_inputs/simple"
PARAM_FILE="param_simple_no_output.txt"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/simulation_results"
SOURCE_FILE="${PROJECT_DIR}/workdir/open_mp/shallow_open_mp.c"
EXEC_FILE="${EXEC_DIR}/shallow_open_mp"

EXE="${EXEC_FILE}"
EXE_ARGS="${INPUT_DIR}/${PARAM_FILE}"




THREADS=(64 32 16 8 4 2 1)

# Loop over the different BIND configurations
for BIND in spread close; do
    # Loop over the number of threads (instead of ranks as in MPI)
    for NTHREADS in "${THREADS[@]}"; do
        echo "Running with ${NTHREADS} threads and BIND=${BIND}..."
        
        # Create a specific parameter file for this configuration
        PARAM_FILE="param_${NTHREADS}threads.txt"
        cp ${EXE_ARGS} ${PARAM_FILE}

        # Inner loop to run the program 10 times for each configuration
        for ((i=1; i<=5; i++)); do
            echo "Run ${i} ..."

            # Set the OpenMP environment variable for the number of threads
            export OMP_NUM_THREADS=${NTHREADS}

            # Run the OpenMP program with the current number of threads
            export OMP_PROC_BIND=${BIND}  # Set thread binding option
            ${EXE} ${PARAM_FILE}
        done
    done
done