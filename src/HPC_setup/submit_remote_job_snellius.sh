#!/bin/bash

# Variables
JOB_NAME=$1
REMOTE_USER="mtresoor"
REMOTE_HOST="snellius.surf.nl"
REPO_PATH="~/fusion-plasma-simulation"
JOB_SCRIPT="run_snellius_job.sh"
SLURM_PARTITION="staging"  # Adjust as needed
GIT_BRANCH="main"  # Branch to pull from
REMOTE_SLURM_DIR="$REMOTE_USER@$REMOTE_HOST:/home/$REMOTE_USER/fusion-plasma-simulation/output/slurms"
LOCAL_HPC_PATH="output/snellius/"

# Check if a job name argument is provided
if [ "$#" -ne 1 ] || [ -z "$1" ]; then
    echo "To submit a new job, use: $0 <job_name>"
    echo "Will check the queue and sync results from snellius."
    # SSH into the main node, check the queue status, and run rsync
    ssh -T -o LogLevel=ERROR $REMOTE_USER@$REMOTE_HOST << EOF
        squeue
EOF
    rsync -avzv $REMOTE_SLURM_DIR $LOCAL_HPC_PATH
    exit 0
fi

# Check if there are commits that have not been pushed
LOCAL_COMMITS=$(git rev-list HEAD --not --remotes)
if [[ -n $LOCAL_COMMITS ]]; then
    echo "Error: There are local commits that have not been pushed. Please push your changes."
    exit 1
fi

# Tag the last commit with the job name
if ! git tag -a "$JOB_NAME" -m "Job '$JOB_NAME' [$(date)]"; then
    echo "Error: Failed to tag the last commit."
    exit 1
fi

git push origin "$JOB_NAME"

# SSH into the main node, pull latest code, submit SLURM job, and inspect queue
ssh -T -o LogLevel=ERROR $REMOTE_USER@$REMOTE_HOST << EOF
    cd $REPO_PATH
    git fetch origin
    git reset --hard origin/$GIT_BRANCH  # Reset local branch to match remote
    git pull origin $GIT_BRANCH
    sbatch --job-name=$JOB_NAME $JOB_SCRIPT $JOB_NAME
    echo "Submitted job '$JOB_NAME'. Checking queue status:"
    squeue --format="%.18i %.50j %.12u %.8T %.10M %.6D %.10P"
EOF

echo "Code updated, SLURM job '$JOB_NAME' submitted, and Git tag '$JOB_NAME' created. :D"

sync_slurms() {
    # Run rsync and capture the output
    RSYNC_OUTPUT=$(rsync -avzv $REMOTE_SLURM_DIR $LOCAL_HPC_PATH)
    # Filter and format the output
    echo "Updated files:"
    echo "$RSYNC_OUTPUT" | grep '\.out' | grep -v 'is uptodate' | awk -v LOCAL_HPC_PATH="$LOCAL_HPC_PATH" '{print LOCAL_HPC_PATH $1}'
}

sleep 1
echo "Syncing results from HPC in 10 seconds..."
sleep 10
ssh -T -o LogLevel=ERROR $REMOTE_USER@$REMOTE_HOST << EOF
    squeue --format="%.18i %.50j %.12u %.8T %.10M %.6D %.10P"
EOF
echo "-----------------------------------------------------------"
sync_slurms
echo "-----------------------------------------------------------"

echo "Syncing results from HPC in 10 seconds..."
sleep 10
ssh -T -o LogLevel=ERROR $REMOTE_USER@$REMOTE_HOST << EOF
    squeue --format="%.18i %.50j %.12u %.8T %.10M %.6D %.10P"
EOF
echo "-----------------------------------------------------------"
sync_slurms
echo "-----------------------------------------------------------"
echo "Syncing results from HPC in 10 seconds..."
sleep 10
ssh -T -o LogLevel=ERROR $REMOTE_USER@$REMOTE_HOST << EOF
    squeue --format="%.18i %.50j %.12u %.8T %.10M %.6D %.10P"
EOF
echo "-----------------------------------------------------------"
sync_slurms
echo "-----------------------------------------------------------"
echo "Syncing results from HPC in 30 seconds..."
sleep 30
ssh -T -o LogLevel=ERROR $REMOTE_USER@$REMOTE_HOST << EOF
    squeue --format="%.18i %.50j %.12u %.8T %.10M %.6D %.10P"
EOF
echo "-----------------------------------------------------------"
sync_slurms
echo "-----------------------------------------------------------"

echo "Syncing results from HPC in 1 minute..."
sleep 60
ssh -T -o LogLevel=ERROR $REMOTE_USER@$REMOTE_HOST << EOF
    squeue --format="%.18i %.50j %.12u %.8T %.10M %.6D %.10P"
EOF
echo "-----------------------------------------------------------"
sync_slurms
echo "-----------------------------------------------------------"


echo "Syncing results from HPC in 2 minutes..."
sleep 120
ssh -T -o LogLevel=ERROR $REMOTE_USER@$REMOTE_HOST << EOF
    squeue --format="%.18i %.50j %.12u %.8T %.10M %.6D %.10P"
EOF
echo "-----------------------------------------------------------"
sync_slurms
echo "-----------------------------------------------------------"


echo "Re-sync results with:"
echo "rsync -avzv $REMOTE_SLURM_DIR $LOCAL_HPC_PATH"
exit 0
