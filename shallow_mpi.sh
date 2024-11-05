#!/bin/bash
#SBATCH --job-name="shallow_mpi"
#SBATCH --ntasks=39                # Number of tasks for MPI (adjust as needed)
#SBATCH --cpus-per-task=1         # Number of cores per task
#SBATCH --time=05:00
#SBATCH --output=shallow_mpi_%j.out


# Define some variables
PROJECT_DIR="${HOME}/project_info0939"
EXEC_DIR="${PROJECT_DIR}/bin"
INPUT_DIR="${PROJECT_DIR}/example_inputs/simple"
PARAM_FILE="param_simple.txt"
SOURCE_FILE="${PROJECT_DIR}/shallow_MPI.c"
EXEC_FILE="${EXEC_DIR}/shallow_MPI"

module load OpenMPI

# Compile the C file with mpicc
echo "Compiling ${SOURCE_FILE}..."
mpicc -O3 -Wall  -o ${EXEC_FILE} ${SOURCE_FILE} -lm
if [ $? -ne 0 ]; then
    echo "Compilation failed" >&2
    exit 1
fi
echo "Compilation successful."


# Copy all files from the input directory to the submission directory
cp ${INPUT_DIR}/* ${SLURM_SUBMIT_DIR}



# Change to the submission directory
cd ${SLURM_SUBMIT_DIR}



# Execute the simulation in parallel with MPI
mpirun -np ${SLURM_NTASKS} ${EXEC_FILE} ${PARAM_FILE}


# Check for errors during execution
if [ $? -ne 0 ]; then
    echo "Error during MPI execution" >&2
    exit 1
fi



mv *.vti *.pvd output/



