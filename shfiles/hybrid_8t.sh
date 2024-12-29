#!/bin/bash
#SBATCH --job-name="hybrid_strong_test"
#SBATCH --output=strong_hybrid_bigdomain_8t.out
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --exclusive


PROJECT_DIR="${HOME}/project_info0939"
EXEC_DIR="${PROJECT_DIR}/bin"
INPUT_DIR="${PROJECT_DIR}/example_inputs/simple"
PARAM_FILE="${INPUT_DIR}/param_simple_no_output.txt"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/simulation_results"
SOURCE_FILE="${PROJECT_DIR}/workdir/MPI+open_mp/shallow_mpi+open_mp.c"
EXEC_FILE="${EXEC_DIR}/shallow_mpi+open_mp"

module load OpenMPI

mpicc -O3 -Wall -fopenmp -o ${EXEC_FILE} ${SOURCE_FILE} -lm



export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
mpirun -np 8 ${EXEC_FILE} ${PARAM_FILE}


