#!/bin/bash
#SBATCH --output=output/slurms/slurm-%j.out
#SBATCH --gres=gpu:1
# Run with sbatch -p zirconium testcuda.sh
# Retrieve output with
# sync -avz TUE_s162507@datamininghpc.win.tue.nl:/home/TUE/s162507/fusion-plasma-simulation/output/slurms/ output/hpc/

# Debugging: Print the shell script being executed
echo "==============================="
echo "Starting job script $0 for run: $1"
echo $(date)
echo "-------------------------------"

# Debugging: Print the current environment before sourcing anything
echo "Conda - Initial conda environment:"
conda info --envs

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
echo "Python version:"
python --version

# Print current working directory
echo "Current working directory:"
pwd

# Export wandb env variable
export WANDB_DIR="/home/TUE/s162507/fusion-plasma-simulation/output"

# Run the Python script
echo "Running testcuda.py..."
python src/HPC_setup/testcuda.py
echo "---------------- JOB START ----------------"
python run.py run_name=$0
# Deactivate the conda environment
# echo "Deactivating conda environment..."
# conda deactivate
echo "--------/-------- JOB END --------/--------"
