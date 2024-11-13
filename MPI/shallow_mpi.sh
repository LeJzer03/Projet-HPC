#!/bin/bash
#SBATCH --job-name="shallow_mpi"
#SBATCH --ntasks=16             # Number of tasks for MPI (adjust as needed)
#SBATCH --cpus-per-task=1         # Number of cores per task
#SBATCH --time=05:00
#SBATCH --output=shallow_mpi_%j.out

# Define some variables
PROJECT_DIR="${HOME}/project_info0939"
EXEC_DIR="${PROJECT_DIR}/bin"
INPUT_DIR="${PROJECT_DIR}/example_inputs/simple_perso"
PARAM_FILE="param_simple_perso.txt"
SOURCE_FILE="${PROJECT_DIR}/workdir/MPI/shallow_mpi.c"
EXEC_FILE="${EXEC_DIR}/shallow_mpi"
OUTPUT_DIR="${PROJECT_DIR}/workdir/MPI/simulation_results"

module load OpenMPI

# Compile the C file with mpicc
echo "Compiling ${SOURCE_FILE}..."
mpicc -O3 -Wall -o ${EXEC_FILE} ${SOURCE_FILE} -lm
if [ $? -ne 0 ]; then
    echo "Compilation failed" >&2
    exit 1
fi
echo "Compilation successful."

# Copy all files from the input directory to the submission directory
cp ${INPUT_DIR}/* ${SLURM_SUBMIT_DIR}

# Change to the submission directory
cd ${SLURM_SUBMIT_DIR}

# Create the output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Find the next available output directory with incrementing numbers (output_j)
j=1
while [ -d "${OUTPUT_DIR}/output_${j}" ]; do
    j=$((j+1))
done
DATA_OUTPUT_DIR="${OUTPUT_DIR}/output_${j}"
mkdir -p ${DATA_OUTPUT_DIR}

# Print the DATA_OUTPUT_DIR to verify
echo "DATA_OUTPUT_DIR is ${DATA_OUTPUT_DIR}"

# Execute the simulation in parallel with MPI
mpirun -np ${SLURM_NTASKS} ${EXEC_FILE} ${PARAM_FILE}

# Execute the simulation in parallel with MPI
#valgrind --leak-check=full --track-origins=yes mpirun -np 5 ./shallow_MPI param_simple.txt

# Check for errors during execution
if [ $? -ne 0 ]; then
    echo "Error during MPI execution" >&2
    exit 1
fi

# Ensure we are in the submission directory
cd ${SLURM_SUBMIT_DIR}

# Move the .vti, .pvd, and .out files to the DATA_OUTPUT_DIR
mv ${SLURM_SUBMIT_DIR}/*.vti ${SLURM_SUBMIT_DIR}/*.pvd ${DATA_OUTPUT_DIR}/
mv ${SLURM_SUBMIT_DIR}/shallow_mpi_${SLURM_JOB_ID}.out ${DATA_OUTPUT_DIR}/

# Get the base name of the DATA_OUTPUT_DIR
ZIP_FILENAME=$(basename ${DATA_OUTPUT_DIR})

# Change to the parent directory of DATA_OUTPUT_DIR
cd ${OUTPUT_DIR}

# Zip the entire output folder
zip -r ${ZIP_FILENAME}.zip ${ZIP_FILENAME}

# Remove the original output folder
#rm -rf ${ZIP_FILENAME}


echo "Simulation completed and results saved to ${DATA_OUTPUT_DIR}/."