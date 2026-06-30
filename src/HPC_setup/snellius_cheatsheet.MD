# Snellis access

Login with
```bash
# ssh scur2932@snellius.surf.nl
ssh mtresoor@snellius.surf.nl
```
Then provide your password.

You can make login easier by generating ssh keys and adding them to the SURF portal. You can also add a host name to your ssh config file to make the login easier.

> Add to `~/.ssh/config`:
```bash
Host snellius
    HostName snellius.surf.nl
    User mtresoor
```
Then you can login with `ssh snellius`.

Copy your ssh key to the server with
```bash
ssh-copy-id 

# Snellius file transfer

# Installing packages
This is not trivial because we need to use the module system. 
This inherently means that the environment is not the same as on your local machine, and that we need to be lucky that the versions match enough to be compatible.

We also cannot use pipenv, because it cannot lock the packages with constraints that are compatible with the module system.

So, we need to have a base list of packages. This can be based on the local Pipfile. But we need to install one by one, and check if it works. 

On snellius, create and activate a virtual environment. 
```bash
source ~/fusion/bin/activate
```

# Generate Keys manually (Linux, Mac, Windows + WSL)

## Generate key pair

First, open a terminal window on your local machine and issue the following command:
```bash
ssh-keygen -t ed25519
```
This will generate a key-pair consisting of a private and a public key. You will be asked in which file you want to store the file, just press enter to use the default (that is, the private key will be stored in `$HOME/.ssh/id_ed25519` and the public key in `$HOME/.ssh/id_ed25519.pub`). Then, you will be asked to choose a passphrase to keep the private key safe (the public key can safely be shared - in fact, you're meant to share it).
### On key algorithms: (RSA vs. ED25519)
-RSA (ssh-rsa):
    - Uses the older RSA algorithm for key generation.
    - Typically longer keys (e.g., 2048 or 4096 bits) are required to maintain security, which makes RSA keys slower during connection establishment.
    - Vulnerable to quantum computing attacks in the future.
 - ED25519 (ssh-ed25519):
   - Based on elliptic curve cryptography (more modern and secure).
   - Keys are smaller in size but provide strong security.
   - Resistant to certain side-channel attacks that RSA might be vulnerable to.
   - Faster for signing and verification, which is efficient for frequent SSH usage.

## Copy the public key to SURF portal

To get the public key, type on your local machine:
```bash
cat ~/.ssh/id_ed25519.pub
```
The key starts with `ssh-rsa`, then contains a long string of random characters, and ends with some identification of where it was created. Copy the whole key, i.e., starting with `ssh-rsa` and including the identification of where it was created. Now, login to the SURF portal, go to 'Public ssh keys' and click 'add key'. Paste your public key in the SSH key field, and enter the password corresponding to your login in the CUA password field.

## Add private key to local SSH agent

With the key in the SURF portal, the HPC system will authenticate you using the key-pair, rather than your password. However, now, your local PC will ask you for the passphrase every time you want to use the private key to authenticate. That doesn't help much. To have password-less login, you have two options: either you don't enter a passphrase when generating the key-pair (not the safest solution) or you use the local SSH agent to store your passphrase for the duration of the login session on your workstation.

To add the key to the local SSH agent, open a new terminal on your local machine and type:
```bash
ssh-add ~/.ssh/id_ed25519
```
### Troubleshooting
If you get an error “Could not open a connection to your authentication agent”, you may need to start the ssh-agent first (using `eval $(ssh-agent -s)` for a bash shell or `eval $(ssh-agent -c)` for a c-shell).

Verify the key is added:
```bash
ssh-add -l
```

Finally, to make your life even easier, you can add the key to the SSH agent automatically, by adding:
```plaintext
Host *
    AddKeysToAgent yes
```
to the `$HOME/.ssh/config` file on your local machine.

Now, your local PC should only ask you for the passphrase once per session (i.e., until you log out of your local machine).

# Creating the environment

We need to use the modules system to load the correct versions of the packages. 
```bash
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a # vs my installed version 2.2.2
module load plotly.py/5.16.0-GCCcore-12.3.0 # vs 5.24.1
module load SciPy-bundle/2023.07-gfbf-2023a 
```
Then verify that the correct versions are loaded with `python --version` and `pip list`.

Open the following for more information on the [module system](https://servicedesk.surf.nl/wiki/display/WIKI/Environment+Modules).
Most used module commands:

    - `module avail`
        -lists the modules that are available
    - `module load modulename`
        -sets the environment up to use modulename
    - `module add modulename`
        -same as previous
    - `module display modulename`
        -shows what is done when you issue the command `module add modulename`
    - `module unload modulename`
        -removes the environment neccessary for modulename
    - `module rm modulename`
        -same as previous
    - `module list`
        -shows the loaded modules
    - `module help`
        - tells how to use the module command
    - `module spider`
        -The module spider command reports all the modules that can be loaded on a system
    - `module purge`
        -Clear the currently set of loaded modules

# NOTE: maybe venv is not necessary, see [python installation on snellius](https://servicedesk.surf.nl/wiki/display/WIKI/Python#Python-Scipy)
Then create a virtual environment with
```bash
python -m venv fusion
```
Activate the virtual environment with
```bash
source ~/fusion/bin/activate
```
Then install the packages that are not available in the module system via pip.
```bash
pip install --upgrade pip
pip install pyarrow torchinfo omegaconf lightning wandb pytest
```

Login to wandb:
```bash
wandb login
```
Then paste the API key from your [wandb profile page](https://wandb.ai/authorize).

# Running code on Snellius
We need to pull the code from the repository. 

First generate deployment ssh keys on the server and copy the <key>.pub contents to the repo at https://github.com/Trezorro/fusion-plasma-simulation/settings/keys

Then:
```bash
git clone git@github.com:Trezorro/fusion-plasma-simulation.git
git fetch
git pull
```
## Troubleshooting
If this doesn't work because you gave the key non-default names, 
run:
```bash
echo -e "Host github.com\n    HostName github.com\n    User git\n    IdentityFile ~/.ssh/<key_name>" >> ~/.ssh/config
```
Then try again.

## Uploading the data
We have a couple of options to upload the data to the server.
- Use the scp command to copy the data from your local machine to the server.
   ```bash
    scp [file_name]  remoteuser@remotehost:/remote/directory
    ```
- Use the `rsync` command to copy the data from your local machine to the server.
    ```bash
    rsync -avz [file_name] remoteuser@remotehost:/remote/directory
    ```
Local path is `data/2024_05_01-NaNsFiltered.parquet`
So command is:
```bash
rsync -avz data/2024_05_01-NaNsFiltered.parquet snellius:~/fusion-plasma-simulation/data
```
This works if we configured the host name `snellius` in the ssh config file.

Then we need to run the code. 

Make sure:
- The output folder for the sbatch script exists for the `#SBATCH --output=output/slurms/agent-%j.out` line. 
-  `mkdir -p output/slurms` is run before the sbatch script.

Then run the sbatch script with
```bash
sbatch run_snellius_job.sh snelliusexperimenttest

squeue --format="%.15i %.80j %.8T %.5M %.5D %P %R"  # to see the jobs in prettier format

cat output/slurms/* # to see the output

```

# Maintenance
## Handling File quota issues and Cleaning up heavy files

If you run into a file quota issue, you can check the disk usage with `myquota` on snellius, or `du -sh ~/fusion-plasma-simulation/` in general.

You can use more quota tools by running:
```bash
export PATH=$PATH:/gpfs/admin/hpc/usertools/
myquota
home-quota
```

See the largest files sorted by size with: 
```bash
du -ah ~/fusion-plasma-simulation/ | sort -rh | head -n 100
```

Wandb has some cleanup utilities that you can try such as  `wandb artifact cache cleanup 1GB`  See [Wandb Storage Docs](https://docs.wandb.ai/guides/artifacts/storage/)

Otherwise you can manually delete the files with `rm -rf <file>` or `rm -rf <folder>`.
Advanced automated search for all checkpoints folders: (use with caution)
```bash
find ~/fusion-plasma-simulation/output/ -type d -name 'checkpoints' -exec rm -rf {} +
```
This will find all folders named `checkpoints` and delete them.


# Appendix
From https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial1/Lisa_Cluster.html#Job-Arrays

## Interactive sessions
As alternative to running jobs via sbatch, you can request an interactive job session. In this format, you gain access to the node via the terminal/command line, and can interact with it as you would have connected via ssh directly to this compute node. Hence, you can debug your script much easier and have faster respond time. However, please keep in mind the following two disadvantages: 1. If you happen to disconnect from Snellius because of an instable connection or your computer going in stand-by mode, the job will be canceled. 2. Your job is not automatically terminated when your script has finished running, but you have to manually kill the job once you are done. If you forget it, you block the compute node for other students and waste credits that UvA payed for Hence, only use interactive sessions for short jobs/scripts, for example, if you want to debug whether your script starts running and train the model. Do not use interactive session to train a model for a long time.

In order to start an interactive session on Snellius, you can use srun with the same input parameters as for the job file. For example:

```bash
srun --partition=gpu_mig --gpus=1 --ntasks=1 --cpus-per-task=9 --time=00:10:00 --pty bash -i
```

This will start an interactive session with a GPU for 10 minutes. Once resources have been allocated, you will see in your terminal that you are now on one of the compute nodes, for example, gcn28. As a first step, you need to load the modules (module load 2023; module load Anaconda3/2023.07-2) and activate your conda environment. Then, you are ready to run your script.

# Job Arrays
You might come into the situation where you need to run a hyperparameter search over multiple values, and don’t want to write an endless number of job scripts. A much more elegant solution is a job array. Job arrays are created with two files: a job file, and a hyperparameter file. The job file will start multiple sub-jobs that each use a different set of hyperparameters, as specified in the hyperparameter file. In the job file, you need to add the argument #SBATCH --array=.... The argument specifies how many sub-jobs you want to start, how many to run in parallel (at maximum), and which lines to use from the hyperparameter file. For example, if we specify #SBATCH --array=1-16%8, this means that we start 16 jobs using the lines 1 to 16 in the hyperparameter file, and running at maximum 8 jobs in parallel at the same time. Note that the number of parallel jobs is there to limit yourself from blocking the whole cluster. However, with your student accounts, you will not be able to run more than 1 job in parallel anyways. The template job file array_job.job looks slightly different than the one we had before. The slurm output file is specified using %A and %a. %A is being automatically replaced with the job ID, while %a is the index of the job within the array (so 1 to 16 in our example above). Below, we also added a block for creating a checkpoint folder for the job array, and copying the job file including hyperparameters to that folder. This is good practice for ensuring reproducibility. Finally, in the training call, we specify the path checkpoint path (make sure to have implemented this argument in your argparse) with the addition of experiment_${SLURM_ARRAY_TASK_ID} which is a sub-folder in the checkpoint directory with the sub-job ID (1 to 16 in the example). The next line, $(head -$SLURM_ARRAY_TASK_ID $HPARAMS_FILE | tail -1), copies the N-th line of the hyperparameter file to this job file, and hence submits the hyperparameter arguments to the training file.

File `array_job.job`:
```bash
#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=ExampleArrayJob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --array=1-16%8
#SBATCH --output=slurm_array_testing_%A_%a.out

module purge
module load 2023
module load Anaconda3/2023.07-2

# Your job starts in the directory where you call sbatch
cd $HOME/...
# Activate your environment
source activate ...

# Good practice: define your directory where to save the models, and copy the job file to it
JOB_FILE=$HOME/.../array_job.job
HPARAMS_FILE=$HOME/.../array_job_hyperparameters.txt
CHECKPOINTDIR=$HOME/.../checkpoints/array_job_${SLURM_ARRAY_JOB_ID}

mkdir $CHECKPOINTDIR
rsync $HPARAMS_FILE $CHECKPOINTDIR/
rsync $JOB_FILE $CHECKPOINTDIR/

# Run your code
srun python train.py \
            --checkpoint_path $CHECKPOINTDIR/experiment_${SLURM_ARRAY_TASK_ID} \
            $(head -$SLURM_ARRAY_TASK_ID $HPARAMS_FILE | tail -1)

```
The hyperparameter file is nothing else than a text file in which each line denotes one set of hyperparameters for which you want to run an experiment. There is no specific order in which you need to put the lines, and you can extend the lines with as many hyperparameter arguments as you want.

File `array_job_hyperparameters.txt`:
```plaintext
--seed 42 --learning_rate 1e-3
--seed 43 --learning_rate 1e-3
--seed 44 --learning_rate 1e-3
--seed 45 --learning_rate 1e-3
--seed 42 --learning_rate 2e-3
--seed 43 --learning_rate 2e-3
--seed 44 --learning_rate 2e-3
--seed 45 --learning_rate 2e-3
--seed 42 --learning_rate 4e-3
--seed 43 --learning_rate 4e-3
--seed 44 --learning_rate 4e-3
--seed 45 --learning_rate 4e-3
--seed 42 --learning_rate 1e-2
--seed 43 --learning_rate 1e-2
--seed 44 --learning_rate 1e-2
--seed 45 --learning_rate 1e-2
```

# Advanced
Launch temporary interactive session on a gpu node for testing and debugging
```

salloc --partition=gpu_mig --reservation=terv92681 --gpus=1 --time=00:10:00
# then once you're on the compute node:
nvidia-smi. # shows available gpu devices
python -c "from numba import cuda; cuda.detect()"  # tests them
```
