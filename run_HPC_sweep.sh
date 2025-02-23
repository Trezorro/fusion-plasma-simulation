#!/bin/bash
#SBATCH --job-name=sweepjob
#SBATCH --output=output/slurms/agent-%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Retrieve output with
# sync -avz TUE_s162507@datamininghpc.win.tue.nl:/home/TUE/s162507/fusion-plasma-simulation/output/slurms/ output/hpc/
echo "==============================="
echo $(date)
echo "-------------------------------"
echo Run agent to $1
echo "==============================="
echo
echo "Job script $0"
echo "Current working directory:"
pwd
echo
echo "-------------------------------"



# Source the conda setup script
echo "Sourcing conda setup..."
source /home/TUE/s162507/miniconda3/etc/profile.d/conda.sh

# Debugging: Check if the conda command is available
if ! command -v conda &> /dev/null
then
    echo "Conda command not found!"
    exit 1
fi

# Activate the conda environment
echo "Activating conda environment 'py11'..."
conda activate py11

# Debugging: Verify the environment is activated
echo "Current conda environment:"
conda info --envs
echo "-------------------------------"

# Print Python version and location
echo "Python executable path:"
which python
echo "Version: $(python --version 2>&1)"


# Run the Python script
echo "Running testcuda.py..."
python src/HPC_setup/testcuda.py

# Export wandb env variable
export WANDB_DIR="/home/TUE/s162507/fusion-plasma-simulation/output"
export WANDB_NOTES=$(git log -n 5 --pretty=format:"%B (%h - %ar) %N")


# Run the Python script
echo "---------------- JOB START ----------------"
srun wandb agent $1

# srun wandb agent deep-learning-course-team/plasma/96sckgvi
# Deactivate the conda environment
# echo "Deactivating conda environment..."
# conda deactivate
echo "--------/-------- JOB END --------/--------"
