#!/bin/bash

# Check if a job name argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <job_name>"
    exit 1
fi

# Variables
JOB_NAME=$1
REMOTE_USER="TUE_s162507"
REMOTE_HOST="datamininghpc.win.tue.nl"
REPO_PATH="~/fusion-plasma-simulation"
JOB_SCRIPT="run_job.sh"
SLURM_PARTITION="zirconium"  # Adjust as needed
GIT_BRANCH="main"  # Branch to pull from

# Check local git status and ensure it's clean and pushed
# cd $REPO_PATH

# # Check if there are uncommitted changes
# if [[ -n $(git status --porcelain) ]]; then
#     echo "Error: Local repository is not clean. Please commit or stash your changes."
#     exit 1
# fi

# Check if there are commits that have not been pushed
LOCAL_COMMITS=$(git rev-list HEAD --not --remotes)
if [[ -n $LOCAL_COMMITS ]]; then
    echo "Error: There are local commits that have not been pushed. Please push your changes."
    exit 1
fi

# Tag the last commit with the job name
git tag -a "$JOB_NAME" -m "Job '$JOB_NAME' [$(date)]"
git push origin "$JOB_NAME"

# SSH into the main node, pull latest code, submit SLURM job, and inspect queue
ssh $REMOTE_USER@$REMOTE_HOST << EOF
    cd $REPO_PATH
    git fetch origin
    git reset --hard origin/$GIT_BRANCH  # Reset local branch to match remote
    git pull origin $GIT_BRANCH
    sbatch --job-name=$JOB_NAME -p $SLURM_PARTITION $JOB_SCRIPT
    echo "Submitted job '$JOB_NAME'. Checking queue status for partition '$SLURM_PARTITION':"
    squeue -p $SLURM_PARTITION
EOF

echo "Code updated, SLURM job '$JOB_NAME' submitted, and Git tag '$JOB_NAME' created."
