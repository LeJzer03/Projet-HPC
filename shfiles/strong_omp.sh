#!/bin/bash
#SBATCH --job-name="omp_strong_2"
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=strong_omp_big_domain.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64

PROJECT_DIR="${HOME}/project_info0939"
EXEC_DIR="${PROJECT_DIR}/bin"
INPUT_DIR="${PROJECT_DIR}/example_inputs/simple"
PARAM_FILE="${INPUT_DIR}/param_simple_no_output.txt"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/simulation_results"
SOURCE_FILE="${PROJECT_DIR}/workdir/open_mp/shallow_open_mp.c"
EXEC_FILE="${EXEC_DIR}/shallow_open_mp"

EXE="${EXEC_FILE}"
EXE_ARGS="${INPUT_DIR}/${PARAM_FILE}"

gcc -march=native -Wall -Wextra -Ofast -fopenmp -o ${EXEC_FILE} ${SOURCE_FILE} -lm


export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread

${EXE} ${PARAM_FILE}
