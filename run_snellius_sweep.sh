#!/bin/bash
#SBATCH --job-name=sweepjob
#SBATCH --output=output/slurms/agent-%A_%a.out
#SBATCH --error=output/slurms/agent-%A_%a.out
#SBATCH --partition=gpu_mig
#SBATCH --time=250:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus=1
#SBATCH --array=1-40


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

echo "Job array index: $SLURM_ARRAY_TASK_ID"
echo "Job array ID: $SLURM_ARRAY_JOB_ID"

echo "Loading modules..."
# Load necessary modules (adjust based on your environment)
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a  # has beniget-0.4.1, Bottleneck-1.3.7, deap-1.4.0, gast-0.5.4, mpmath-1.3.0, numexpr-2.8.4, numpy-1.25.1, pandas-2.0.3, ply-3.11, pythran-0.13.1, scipy-1.11.1, tzdata-2023.3, versioneer-0.29
module load matplotlib/3.7.2-gfbf-2023a # vs 3.9.3
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 # vs my installed version 2.2.2
module load plotly.py/5.16.0-GCCcore-12.3.0 # vs 5.24.1

echo "Activating virtual environment..."
source ~/fusion/bin/activate

# Print Python version and location
echo "Python executable path:"
which python
echo "Version: $(python --version 2>&1)"


# Export wandb env variable
export WANDB_DIR="~/fusion-plasma-simulation/output"
export WANDB_NOTES=$(git log -n 5 --pretty=format:"%B (%h - %ar) %N")

# Run the Python script
echo "---------------- JOB START ----------------"
srun wandb agent --count 1 $1

# srun wandb agent deep-learning-course-team/plasma/96sckgvi
# Deactivate the conda environment
# echo "Deactivating conda environment..."
# conda deactivate
echo "--------/-------- JOB END --------/--------"
