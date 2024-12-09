#!/bin/bash
#SBATCH --job-name=sweepagent
#SBATCH --output=output/slurms/agent-%j.out
#SBATCH --error=output/slurms/agent-%j.err
#SBATCH --partition=thin_course
#SBATCH --time=10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=1
 
# Load necessary modules (adjust based on your environment)
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Debugging: Print the shell script being executed
echo "==============================="
echo $(date)
echo "-------------------------------"
echo $1
echo "==============================="
echo
echo "Job script $0"

# Debugging: Print the current environment before sourcing anything
echo "Current working directory:"
pwd
echo
echo "-------------------------------"


# Print Python version and location
echo "Python executable path:"
which python
echo "Version: $(python --version 2>&1)"

#  TODO: Add the necessary commands to install the required packages
pipenv sync
pipenv shell
# print the current environment
echo "Current pipenv environment:"
echo $(pipenv --venv)

# Export wandb env variable
export WANDB_DIR="~/fusion-plasma-simulation/output"
export WANDB_NOTES=$(git log -n 5 --pretty=format:"%B (%h - %ar) %N")

# Run the Python script
echo "---------------- JOB START ----------------"
srun python run.py run_name=$1

# srun wandb agent deep-learning-course-team/plasma/96sckgvi
# Deactivate the conda environment
# echo "Deactivating conda environment..."
# conda deactivate
echo "--------/-------- JOB END --------/--------"
