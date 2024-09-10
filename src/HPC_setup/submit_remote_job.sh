#!/bin/bash

# Variables
JOB_NAME=$1
REMOTE_USER="TUE_s162507"
REMOTE_HOST="datamininghpc.win.tue.nl"
REPO_PATH="~/fusion-plasma-simulation"
JOB_SCRIPT="run_job.sh"
SLURM_PARTITION="zirconium"  # Adjust as needed
GIT_BRANCH="main"  # Branch to pull from

# Check if a job name argument is provided
if [ "$#" -ne 1 ] || [ -z "$1" ]; then
    echo "To submit a new job, use: $0 <job_name>"
    echo "Will check the queue and sync results from HPC."
    # SSH into the main node, check the queue status, and run rsync
    ssh $REMOTE_USER@$REMOTE_HOST << EOF
        squeue
EOF
    rsync -avzv TUE_s162507@datamininghpc.win.tue.nl:/home/TUE/s162507/fusion-plasma-simulation/output/slurms output/hpc/
    exit 0
fi



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
# Tag the last commit with the job name
if ! git tag -a "$JOB_NAME" -m "Job '$JOB_NAME' [$(date)]"; then
    echo "Error: Failed to tag the last commit."
    exit 1
fi

git push origin "$JOB_NAME"

# SSH into the main node, pull latest code, submit SLURM job, and inspect queue
ssh $REMOTE_USER@$REMOTE_HOST << EOF
    cd $REPO_PATH
    git fetch origin
    git reset --hard origin/$GIT_BRANCH  # Reset local branch to match remote
    git pull origin $GIT_BRANCH
    sbatch --job-name=$JOB_NAME -p $SLURM_PARTITION $JOB_SCRIPT $JOB_NAME
    echo "Submitted job '$JOB_NAME'. Checking queue status for partition '$SLURM_PARTITION':"
    squeue
EOF

echo "Code updated, SLURM job '$JOB_NAME' submitted, and Git tag '$JOB_NAME' created. :D"
sleep 1
echo "Syncing results from HPC in 10 seconds..."
sleep 10
ssh $REMOTE_USER@$REMOTE_HOST << EOF
    squeue
EOF
rsync -avz TUE_s162507@datamininghpc.win.tue.nl:/home/TUE/s162507/fusion-plasma-simulation/output/slurms output/hpc/
echo "Syncing results from HPC in 30 seconds..."
sleep 30
ssh $REMOTE_USER@$REMOTE_HOST << EOF
    squeue
EOF
rsync -avz TUE_s162507@datamininghpc.win.tue.nl:/home/TUE/s162507/fusion-plasma-simulation/output/slurms output/hpc/
echo "Re-sync results with:"
echo "rsync -avz TUE_s162507@datamininghpc.win.tue.nl:/home/TUE/s162507/fusion-plasma-simulation/output/slurms output/hpc/"
exit 0
