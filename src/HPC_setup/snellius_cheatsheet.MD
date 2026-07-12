# Snellius cheatsheet

This is a tutorial for getting set up on the Snellius HPC cluster, plus a reference section for day to day use. It is written for a new machine, starting from nothing.

**The goal:** be able to run the "Run Snellius Experiment" task from [`.vscode/tasks.json`](../../.vscode/tasks.json) (Ctrl+Shift+P -> "Run Task"). That task pushes a git tag, ssh's into Snellius, pulls the code, and submits a Slurm job. For that to work, four things need to exist on the remote machine first:

1. SSH access from your laptop to Snellius.
2. The repo cloned on Snellius.
3. A Python virtual environment on Snellius, with packages installed.
4. Your data uploaded to the right folder on Snellius.

Do these once, in order, and the VS Code tasks will work from then on.

## Step 1: Set up SSH access

Open a terminal on your own machine.

1. Generate a key pair (press enter to accept the default location, and skip the passphrase):
   ```bash
   ssh-keygen -t ed25519
   ```
   More instructions at [Github Guide for generating ssh keys.](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
2. Print your public key and copy the whole output:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
3. Go to the [SURF portal](https://portal.cua.surf.nl/user/keys), open "Public ssh keys", click "add key", and paste it in.
4. Add a shortcut to `~/.ssh/config` so you can just type `ssh snellius`:
   ```bash
   Host snellius
       HostName snellius.surf.nl
       User mtresoor
       IdentityFile ~/.ssh/id_ed25519
       VerifyHostKeyDNS yes
   ```
5. Test it:
   ```bash
   ssh snellius
   ```
   You may still be asked for your passphrase once per terminal session, that is normal.

6. If you get asked for your SURF password instead, it can help to copy your local key to the server directly:
```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub snellius
```
If something is still wrong, see [Detailed SSH key setup](#detailed-ssh-key-setup) below.

## Step 2: Clone the repo on Snellius

Log in (`ssh snellius`) and, on the remote machine, generate a separate key pair for talking to GitHub (do not reuse your laptop key):
```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_github
cat ~/.ssh/id_ed25519_github.pub
```
Copy that output and add it as a deploy key at `https://github.com/Trezorro/fusion-plasma-simulation/settings/keys`.

Then, still on Snellius, tell SSH to use that key for GitHub:
```bash
echo -e "Host github.com\n    HostName github.com\n    User git\n    IdentityFile ~/.ssh/id_ed25519_github" >> ~/.ssh/config
```
Now clone the repo:
```bash
git clone git@github.com:Trezorro/fusion-plasma-simulation.git
```

## Step 3: Create and populate the virtual environment

Still on Snellius, inside the repo folder. First load the modules (these give you Python, PyTorch, etc. at versions compatible with the cluster):
```bash
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a
```
Now create the venv, named `fusion` (this is the name [`setup.sh`](setup.sh) and the job scripts expect):
```bash
python -m venv ~/fusion
source ~/fusion/bin/activate
```
Install the packages that are not covered by the modules above:
```bash
pip install --upgrade pip
pip install pyarrow torchinfo omegaconf lightning wandb pytest
```
Log in to wandb once, pasting the API key from your [wandb profile page](https://wandb.ai/authorize):
```bash
wandb login
```

From now on, you never repeat this setup manually again. Every job script, and the `. setup.sh` command for manual logins, loads the same modules and activates this same `fusion` venv automatically. If you add a new pip dependency later, just `pip install` it once on Snellius the same way (with the venv activated).

## Step 4: Upload your data

Run this from your own machine (not on Snellius), from the repo root:
```bash
rsync -avz data/2024_05_01-NaNsFiltered.parquet snellius:~/fusion-plasma-simulation/data
```
Adjust the filename to whatever data file the run needs. The `snellius` alias only works if you completed Step 1.

## Step 5: Run a job

You're set up. From VS Code, run the task "Run Snellius Experiment" and give it a run name. Behind the scenes this tags your last commit, pushes it, ssh's in, pulls the code, and submits the Slurm job for you.

If you'd rather do it by hand, see [Submitting jobs manually](#submitting-jobs-manually) below.

---

# Reference

## Everyday commands

Logging in for a quick manual session:
```bash
ssh snellius
. setup.sh          # loads modules + activates the fusion venv
```

Checking on jobs:
```bash
squeue --format="%.15i %.80j %.8T %.5M %.5D %P %R"   # your jobs, readable format
cat output/slurms/*                                   # job output
scancel <job_id>                                       # cancel a job
```

## Submitting jobs manually

The VS Code tasks wrap these, but you can run them directly if you're already logged in:
```bash
mkdir -p output/slurms                    # only needed once, the job writes its output here
sbatch run_snellius_job.sh <run_name>
```

For a hyperparameter sweep, we let wandb manage the parameter search, and a Slurm job array just launches one sweep agent per array task ([`run_snellius_sweep.sh`](../../run_snellius_sweep.sh)):
```bash
sbatch run_snellius_sweep.sh <entity>/<project>/<sweep_id>
```

### Interactive sessions

Useful for debugging (faster feedback than sbatch), but only for short work: the job dies if your connection drops, and it is *not* cancelled automatically when you're done, you have to `exit` or `scancel` it yourself, or you keep burning allocated compute.
```bash
srun --partition=gpu_mig --gpus=1 --ntasks=1 --cpus-per-task=9 --time=00:10:00 --pty bash -i
. setup.sh
```

## Maintenance

**Watch your file quota.** Every job writes checkpoints, wandb logs, and slurm output files. If you run many jobs (especially sweeps with dozens of array tasks) without cleaning up, you *will* hit your file quota and new jobs will fail to write output. Check it regularly, not just when something breaks:
```bash
myquota
```
Extra quota tools:
```bash
export PATH=$PATH:/gpfs/admin/hpc/usertools/
myquota
home-quota
```
Find what's taking up space:
```bash
du -ah ~/fusion-plasma-simulation/ | sort -rh | head -n 100
```
Clean up wandb's local cache:
```bash
wandb artifact cache cleanup 1GB
```
Bulk-delete all checkpoint folders (use with caution, this is not reversible):
```bash
find ~/fusion-plasma-simulation/output/ -type d -name 'checkpoints' -exec rm -rf {} +
```
For which filesystem to put what on (home has a small quota and is backed up, `scratch-shared` is large but temporary), see the [SURF filesystems wiki](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/85295828/Snellius+filesystems). `setup.sh` already points wandb's cache and artifacts at `/scratch-shared/mtresoor/` for this reason.

## Module system reference

- `module avail`, lists modules you can load
- `module load <name>`, sets up the environment for that module
- `module add <name>`, same as `module load`
- `module display <name>`, shows what `module load <name>` actually does
- `module unload <name>`, removes a loaded module
- `module rm <name>`, same as `module unload`
- `module list`, shows what's currently loaded
- `module spider <name>`, searches for a module across all versions
- `module purge`, clears everything loaded

Full docs: [SURF module system wiki](https://servicedesk.surf.nl/wiki/display/WIKI/Environment+Modules).

## Using Job arrays for experiment sweeps

Our sweeps use wandb to manage parameters, but plain Slurm job arrays (e.g. for a manual grid search without wandb) work like this: `#SBATCH --array=1-16%8` starts 16 sub-jobs, at most 8 running at once, and `$SLURM_ARRAY_TASK_ID` inside the script tells you which one you are. See [`run_snellius_sweep.sh`](../../run_snellius_sweep.sh) for a working example of this pattern combined with `wandb agent --count 1`.

---

## Detailed SSH key setup

<details>
<summary>Expand if Step 1 didn't work, or you want to understand what's happening</summary>

### Key type: ED25519 vs RSA

`ssh-keygen -t ed25519` uses a modern elliptic curve algorithm: smaller keys, faster to verify, and considered more secure than RSA at comparable key sizes. There's no good reason to use RSA (`-t rsa`) unless something you connect to specifically requires it.

### "Could not open a connection to your authentication agent"

The ssh-agent isn't running. Start it first:
```bash
eval $(ssh-agent -s)    # bash
eval $(ssh-agent -c)    # csh
```
Then `ssh-add ~/.ssh/id_ed25519` again. Check it worked with `ssh-add -l`.

### Still asks for a password every time

Two options: skip the passphrase when generating the key (less safe, but simplest), or add this to `~/.ssh/config` so your OS keeps the key unlocked for the session:
```
Host *
    AddKeysToAgent yes
```


</details>
For anything not covered here (agent forwarding, known_hosts conflicts, permission denied errors), see the [SURF SSH Troubleshooting Guide](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/154239005/SSH+Troubleshooting+Guide).

## Gaps / TODO

- No troubleshooting notes yet for module load conflicts or job submission failures beyond the git deploy key case.
- Default/recommended `--time` and partition choices aren't documented, currently copied ad hoc between job scripts.
- No data backup/archival strategy documented (`scratch-shared` is not permanent storage, see the filesystems wiki above).
