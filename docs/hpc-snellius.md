# Running PlasmaFlow on Snellius (HPC)

This is the reference for running and debugging PlasmaFlow jobs on the SURF Snellius
cluster. It covers the full workflow: how I submit jobs from VS Code, what the
submission and SBATCH scripts actually do, how the cluster environment is built, and
how to get outputs back.

Everything here is confirmed against the actual scripts in `src/HPC_setup/` and
`run_snellius_job.sh`.

## TL;DR

- Submit a run: VS Code task **Run Snellius Experiment**, or
  `bash src/HPC_setup/submit_remote_job_snellius.sh '<run_name>'` from repo root.
- Every `<run_name>` becomes a git tag and must be unique. Reusing a name aborts the submission.
- Local commits must be pushed before submitting; the script refuses otherwise.
- The cluster venv lives at `~/fusion/` and is built by hand, not from the Pipfile.
- Slurm logs stream back to `output/snellius/slurms/` while the job runs.

## VS Code tasks (the intended entry points)

The tasks in `.vscode/tasks.json` are how I actually drive everything. The important ones:

| Task | What it runs |
|---|---|
| Run Local Experiment | Guards on a clean git tree, then `pipenv run python run.py run_name=<name>` locally |
| Run Snellius Experiment | `bash src/HPC_setup/submit_remote_job_snellius.sh '<run_name>'` |
| Reeval Snellius Experiment | Same script with `--reeval` appended |
| Sync snellius | The submit script with no args (queue check plus rsync only) |
| Sync pdf plots | `rsync` of cluster `output/pdfplots/` down to local `output/pdfplots/` |
| Submit Snellius Sweep Agent | Submits a wandb sweep agent job array |

## `submit_remote_job_snellius.sh` (runs locally)

Script: `src/HPC_setup/submit_remote_job_snellius.sh`

This is the local orchestrator. It tags the code, pushes it, SSHes into Snellius,
resets the cluster checkout to match, submits the SBATCH job, then polls and streams
logs back.

### Hardcoded config

```bash
REMOTE_USER="mtresoor"
REMOTE_HOST="snellius.surf.nl"
REPO_PATH="~/fusion-plasma-simulation"        # cluster home
GIT_BRANCH="paper"
REMOTE_SLURM_DIR="mtresoor@snellius.surf.nl:/home/mtresoor/fusion-plasma-simulation/output/slurms"
LOCAL_HPC_PATH="output/snellius/"             # relative: run from repo root
```

### Sync-only mode (no job name)

This is the **Sync snellius** task. With no run name:

1. SSHes in and runs `squeue`.
2. `sync_slurms()`: rsyncs `output/slurms` down, then greps the updated `.out` files
   for the wandb "View run" URL and prints it.
3. Prints the manual re-sync command and exits.

Use this to check the queue and pull fresh logs without submitting anything.

### Submit mode (`submit_remote_job_snellius.sh <run_name>`)

1. Checks for unpushed local commits with `git rev-list HEAD --not --remotes`. If any
   exist, it aborts and tells you to push.
2. `git tag -a "$JOB_NAME"`, then `git push origin "$JOB_NAME"`. This pins the exact
   code version the cluster will run.
3. SSHes into Snellius and runs, as a heredoc:
   - `cd ~/fusion-plasma-simulation`
   - `git fetch origin`
   - `git checkout paper`
   - `git reset --hard origin/paper`
   - `git pull origin paper`
   - `sbatch --job-name=$JOB_NAME run_snellius_job.sh $JOB_NAME $REEVAL_MODE`
   - `squeue`
4. Polls with increasing sleep intervals (1, 5, 5, 5, 5, 5, 10, 10, 10, 30, 60 seconds).
   Each cycle runs remote `squeue` and `sync_slurms`, streaming the wandb run URL back
   to your local terminal as soon as the job prints it.
5. Prints the manual re-sync command and exits. The job keeps running on the cluster
   after the script ends; only the log streaming stops.

### Gotchas

- **Run names must be unique.** If `<run_name>` was used before, `git tag` fails and the
  script aborts. Pick a fresh name each time.
- **`git reset --hard origin/paper` on the cluster discards uncommitted cluster-side
  changes.** Do not keep local edits in the cluster checkout that you care about.
- **Run from repo root.** `LOCAL_HPC_PATH` is relative, so running from elsewhere puts
  synced logs in the wrong place.
- **`--reeval` sets `REEVAL_MODE=true`**, which is passed as `$2` to the SBATCH script
  and changes the cluster-side git checkout (see below).

## `run_snellius_job.sh` (runs on the cluster under SLURM)

Script: `run_snellius_job.sh` (repo root)

This is the job script SLURM actually executes on the GPU node.

### SBATCH parameters

```
#SBATCH --output=output/slurms/gpujob-%j-%x.out   (stderr goes to the same file)
#SBATCH --partition=gpu_mig
#SBATCH --reservation=terv92681
#SBATCH --time=800:00                              (800 minutes)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
```

### Steps

1. **Load modules** (inline, matching `setup.sh`):

   ```bash
   module load 2023
   module load Python/3.11.3-GCCcore-12.3.0
   module load matplotlib/3.7.2-gfbf-2023a
   module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
   module unload SciPy-bundle/2023.07-gfbf-2023a
   ```

   `SciPy-bundle` is unloaded because it pins numpy, pandas, and scipy versions that
   conflict with the versions in the venv.

2. **Activate the venv:** `source ~/fusion/bin/activate` (the manually built venv).

3. **Git checkout based on the reeval flag:**
   - Normal run: `git checkout tags/$1`, the exact commit tagged by the local script.
   - Reeval (`$2 == "true"`): `git checkout paper`, the latest paper branch HEAD.

4. **Export `WANDB_NOTES`** from the last 5 commit messages.

5. **Set cluster env vars** (each directory is `mkdir -p`'d if missing):

   ```bash
   export WANDB_DIR=/scratch-shared/mtresoor/wandb
   export WANDB_CACHE_DIR=/scratch-shared/mtresoor/wandb/cache
   export WANDB_ARTIFACT_DIR=/scratch-shared/mtresoor/artifacts
   export TEST_CACHE_DIR=/scratch-shared/mtresoor/final_cache
   ```

6. **Run:** `srun python run.py run_name=$1 reeval=$2 "${@:3}"`

   Anything from position 3 onward is forwarded straight to `run.py` as OmegaConf
   override syntax (`key=value`).

Log output lands in `output/slurms/gpujob-{jobid}-{run_name}.out` (combined stdout and
stderr).

### Forwarding config overrides

Because extra args are forwarded, you can override config from the command line. For
example:

```bash
bash src/HPC_setup/submit_remote_job_snellius.sh 'my_run' trainer.max_epochs=50 model.lr=1e-4
```

passes `trainer.max_epochs=50 model.lr=1e-4` through to `run.py`.

## `setup.sh` (manual env helper, not called automatically)

Script: `src/HPC_setup/setup.sh`

This reproduces the same module and venv environment as the SBATCH script, but **nothing
calls it automatically**. It is a standalone helper meant to be sourced by hand in an
interactive shell:

```bash
source src/HPC_setup/setup.sh
```

The SBATCH script declares its env inline instead of sourcing this file. That duplication
is intentional, so a job is fully self-contained and reproducible. `setup.sh` is what I
use when running `salloc` for interactive debugging.

## Cluster venv setup (one-time, manual)

The cluster venv lives at `~/fusion/` and is built by hand. It is **not** installed from
the Pipfile (pipenv is local-only). The commands are documented at the bottom of
`Pipfile`:

```bash
# Load modules first (as in setup.sh)
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module unload SciPy-bundle/2023.07-gfbf-2023a

# Create venv and install
python -m venv ~/fusion
source ~/fusion/bin/activate
pip install pyarrow torchinfo omegaconf lightning wandb pytest pot antropy "plotly<6.0.0" \
            kaleido torchdiffeq pysdtw tables datashader h5py scipy
```

### Known version issue: numba

`numba` must be `>= 0.61` on the cluster. The Pipfile pins an older version for local
`ydata-profiling` compatibility, but on the cluster numba 0.58.x **segfaults** under
driver 595.71.05 / CUDA 13.2 when `pysdtw` launches its CUDA kernel (the SoftDTW kernel
in `src/metrics/metrics.py`). The cluster venv runs numba 0.65.1 (updated June 2026).

## One-time cluster setup (cheatsheet)

Documented in `src/HPC_setup/snellius_cheatsheet.MD`:

1. Set up an ed25519 SSH key via the SURF portal; add it to `~/.ssh/config` as
   `Host snellius`.
2. Clone the repo with the deploy key:
   ```bash
   git clone git@github.com:Trezorro/fusion-plasma-simulation.git ~/fusion-plasma-simulation
   ```
3. Upload data:
   ```bash
   rsync -vz ./data/*.parquet snellius:~/fusion-plasma-simulation/data/
   ```
4. `wandb login`
5. Create the slurm output directory (required before the first submission):
   ```bash
   mkdir -p ~/fusion-plasma-simulation/output/slurms
   ```
6. Build the venv (see the section above).

## Data sync

Data files are **not** synced automatically. Upload them by hand before running:

```bash
rsync -vz ./data/2026_06_29-TCV_shots_V2.parquet snellius:~/fusion-plasma-simulation/data/
```

## Getting outputs back

| Output | Auto-synced | How to get it |
|---|---|---|
| Slurm logs | Yes (polling loop in the submit script) | Appear in `output/snellius/slurms/` |
| PDF plots | Via the **Sync pdf plots** task | `rsync` cluster `output/pdfplots/` to local |
| Model checkpoints | Via wandb artifacts (cloud) | Download from the wandb UI or API |
| HDF5 test cache | No | Manual `rsync` from `/scratch-shared/mtresoor/final_cache/` |
| wandb metrics | Yes (auto-synced during the run) | View at wandb.ai |

## Sweep agents

For hyperparameter sweeps, use the **Submit Snellius Sweep Agent** task, which runs:

```bash
src/HPC_setup/submit_snellius_sweep_agent.sh <sweep_id>
```

This submits a job array (1 to 40 GPU workers) via `run_snellius_sweep.sh`. Create the
wandb sweep first:

```bash
wandb sweep configs/sweep.yaml
```

then pass the resulting sweep ID to the task.

## Interactive debugging

To grab an interactive GPU session and verify the environment:

```bash
salloc --partition=gpu_mig --reservation=terv92681 --gpus=1
source src/HPC_setup/setup.sh                       # load modules and activate venv
nvidia-smi                                           # verify the GPU
python -c "from numba import cuda; cuda.detect()"    # verify numba CUDA
```

If `cuda.detect()` segfaults or errors, check the numba version first (see the numba
note above): that is the usual culprit.
