PS1='\w$ '

echo "Loading modules..."
# Load necessary modules (adjust based on your environment)
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a  # has beniget-0.4.1, Bottleneck-1.3.7, deap-1.4.0, gast-0.5.4, mpmath-1.3.0, numexpr-2.8.4, numpy-1.25.1, pandas-2.0.3, ply-3.11, pythran-0.13.1, scipy-1.11.1, tzdata-2023.3, versioneer-0.29
module load matplotlib/3.7.2-gfbf-2023a # vs 3.9.3
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 # vs my installed version 2.2.2

echo "Activating virtual environment..."
source ~/fusion/bin/activate

# Print Python version and location
echo "Python executable path:"
which python
echo "Version: $(python --version 2>&1)"

export WANDB_DIR="/scratch-shared/mtresoor/wandb"
export WANDB_CACHE_DIR="/scratch-shared/mtresoor/wandb/cache"
export WANDB_ARTIFACT_DIR="/scratch-shared/mtresoor/artifacts"
# Ensure WANDB_CACHE_DIR and WANDB_ARTIFACT_DIR exist
[ ! -d "$WANDB_DIR" ] && mkdir -p "$WANDB_DIR"
[ ! -d "$WANDB_CACHE_DIR" ] && mkdir -p "$WANDB_CACHE_DIR"
[ ! -d "$WANDB_ARTIFACT_DIR" ] && mkdir -p "$WANDB_ARTIFACT_DIR"
echo "Created wandb dirs in scratch-shared"