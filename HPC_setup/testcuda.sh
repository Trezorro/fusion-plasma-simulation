#!/bin/bash
#SBATCH --job-name=test_cuda # Optional: Name the job for easier tracking
#SBATCH --output=slurm-%j.out # Optional: Name the output file based on job ID
# Run with sbatch -p zirconium --gres=gpu:1 testcuda.sh

# Debugging: Print the shell script being executed
echo "Starting job script..."
echo "Script: $0"
echo "-------------------------------"

# Debugging: Print the current environment before sourcing anything
echo "Initial conda environment:"
conda info --envs
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
echo "Python version:"
python --version
echo "-------------------------------"

# Print current working directory
echo "Current working directory:"
pwd
echo "-------------------------------"

# Run the Python script
echo "Running testcuda.py..."
python testcuda.py
echo "-------------------------------"

# Deactivate the conda environment
echo "Deactivating conda environment..."
conda deactivate

# Debugging: Verify the environment is deactivated
echo "Conda environment after deactivation:"
conda info --envs
echo "Job script finished."
