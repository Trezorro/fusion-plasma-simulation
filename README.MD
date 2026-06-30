# Fusion Plasma Simulation

This repository contains the code for the PlasmaFlow model used for the master thesis: 
_Flow Matching for data-driven simulation of plasma confinement diagnostics with stochastic temporal dynamics_

Work by Milan Tresoor, supervised by Vlado Menkovski, Simon Koop and Yoeri Poels.

# Usage
Slurm jobs can be run using the `.sh` files in root (e.g. `run_snellius_job.sh`). 

Slurm submission is automated using
tasks and the bash files in `src/HPC_setup`. E.g.: `submit_remote_job_snellius.sh`

## Requirements
Requirements are maintained in `Pipfile` using pipenv for local use. The same file describes the pip commands needed for 
setup on a cluster.


## Experiments
*The most important files are listed here:*
 - `run.py`: Main run script, for training and evaluation. Called by slurm job scripts.
 - `src/models/flow.py`: Defines the TorchLightning module that handles training, validation and testing procedures. All
    Flow Matching logic is contained in here as well.
 - `src/models/unet_conditional.py`: Defines the main architecture used for the PlasmaFlow model.
 - Evaluation functionality is spread between `src/evaluation.py`, the files in `srs/metrics/`, and the plotting code in 
   `src/plotters/`.

Training and evaluation config is done using config YAML files in `configs/`, supported by wandb and `src/config.py`.

# Data
Data can be acquired on request and should be stored in `data/`. Use `src/run_preprocessing.py` to prepare parquet files.
Data is loaded using `src/data_loaders.py`.

Prior to running jobs on snellius, upload the preprocessed data package to the cluster. 
```bash
rsync -vz ./data/*.parquet snellius:fusion-plasma-simulation/data/  
```
 - This depends on `snellius` being a defined ssh config on your local machine. 


**Example config file in `~/.ssh/config**:
```
Host *
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_rsa

Host snellius
    AddKeysToAgent yes
    UseKeychain yes
    HostName snellius.surf.nl
    User mtresoor
    IdentityFile ~/.ssh/snellius
```




# Running on Cluster
1. Set up ssh keys for passwordless login, use of github deploy keys is recommended.
2. Run  
   ```git clone  git@github.com:Trezorro/fusion-plasma-simulation.git```
3. Upload the data directory with rsync
    ```rsync -avz data/ TUE_s162507@datamininghpc.win.tue.nl:/home/TUE/s162507/fusion-plasma-simulation/ ```
    *NOTE: You need the full canonicalized path to the data directory on the server, starting from the home directory.*
4. Sync output files back to local machine
    ```rsync -avz TUE_s162507@datamininghpc.win.tue.nl:/home/TUE/s162507/fusion-plasma-simulation/output output/hpc/```
