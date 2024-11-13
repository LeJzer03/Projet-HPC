#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=59:00
#SBATCH --output=shallow_serial_%j.out

module load OpenMPI

# define some variables
PROJECT_DIR="${HOME}/project_info0939"
EXEC_DIR="${PROJECT_DIR}/bin"
INPUT_DIR="${PROJECT_DIR}/example_inputs/simple_perso"
PARAM_FILE="param_simple_perso.txt"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/simulation_results"
SOURCE_FILE="${PROJECT_DIR}/workdir/serial/shallow_serial.c"
EXEC_FILE="${EXEC_DIR}/shallow_serial"

# Compile the C file with gcc
echo "Compiling ${SOURCE_FILE}..."
gcc -O3 -Wall -o ${EXEC_FILE} ${SOURCE_FILE} -lm
if [ $? -ne 0 ]; then
    echo "Compilation failed" >&2
    exit 1
fi
echo "Compilation successful."

# copy all files from the input directory to the directory from which we submitted the job
cp ${INPUT_DIR}/* ${SLURM_SUBMIT_DIR}

# create the output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# find the next available output directory with incrementing numbers (output_j)
j=1
while [ -d "${OUTPUT_DIR}/output_${j}" ]; do
    j=$((j+1))
done
DATA_OUTPUT_DIR="${OUTPUT_DIR}/output_${j}"
mkdir -p ${DATA_OUTPUT_DIR}

# run the simulation and save output to a file in the DATA_OUTPUT_DIR
${EXEC_FILE} ${PARAM_FILE} > ${DATA_OUTPUT_DIR}/output.txt

# move the .vti and .pvd files (assuming shallow generates them in ${SLURM_SUBMIT_DIR})
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