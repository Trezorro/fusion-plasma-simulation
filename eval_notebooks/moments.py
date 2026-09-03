"""Wandb run pull plus cache-to-thesis-name mapping; renames wandb runs to human names and builds the cache/experiment map.

Scientific output: none directly; produces the run_map and cache_experiment_map (human_name, Model, History Length, Full Covariates, Conditioning, Prior) that the table notebooks reuse. (Despite the filename, this file does NOT compute distributional moment tables; the moment-table caption text lives in peak_analysis.py/peaks_tables.py.)
Inputs:  wandb.Api() runs from "tresoor/flowtoy" tagged final_reeval; output/XR-overview.csv; local output/test_cache/*.h5 glob (listed only). The 16 human names are assigned positionally from a hardcoded block, matching the sorted config-group order.
Outputs: inline only (dataframes); side effect: renames matched wandb runs in place via find_wandb_run(run_name).update(). No files written.
Usage:   run cells top to bottom; the RERUNNING RUNS sbatch cell is commented out; the human-name block order must match the sorted group order or names misalign.
Limits:  fragile; needs wandb auth and network; positional human-name mapping breaks if the group set changes; mutates live wandb run names; frozen half-finished ("...start moments...").
Handy:   get_cache_overview() (config-group to human-name mapping) and the find_wandb_run rename loop duplicate logic in mode_analysis.py/model_cache_overview.py; consolidate into one src/ helper.
History: created Jun 16 2025 ("...start moments and check tiny evaluation"), single commit, never edited (frozen).
"""
#%% Imports
from numpy import isin
import pandas as pd
from pathlib import Path
import h5py
from sympy import Id
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

CACHE_DIR = Path('output/test_cache')  # contains .h5 and .jsons
# List all .h5 files in the CACHE_DIR
CACHED_H5_LIST = list(CACHE_DIR.glob('*.h5'))
CACHED_H5_LIST
#%% show keys available in mode key for a cache




import pandas as pd
import wandb

api = wandb.Api()

# Project is specified by <entity/project-name>
wruns = api.runs("tresoor/flowtoy", filters={'tags': 'final_reeval', })
print(list(run.name for run in wruns),sep='\n')

#%%
summary_list, config_list, name_list = [], [], []
for run in wruns:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)
#%%
runs_df = pd.DataFrame({"summary": summary_list, "config": config_list, "name": name_list})
# Unpack the 'summary' and 'config' columns into separate columns
summary_df = pd.json_normalize(runs_df['summary'])
config_df = pd.json_normalize(runs_df['config'])
combined_df = pd.concat([runs_df.drop(['summary', 'config'], axis=1), summary_df, config_df], axis=1)
del runs_df
combined_df.head()
#%%

# Get all unique keys containing 'test/final' from summary_list
W_GROUP_COLS = [
    'model.Class', 'data.history_length', 'model.params.model_params.c_channels', 'Cond_dim', 'model.params.prior'
]
IMPORTANT_COLS = W_GROUP_COLS + ['name', 'test_cache_name', 'base_run', 'run_name', 'test_cache_mode']
df = combined_df.loc[:, combined_df.columns.str.startswith('test/final') | combined_df.columns.isin(IMPORTANT_COLS)]


# Reorder columns so IMPORTANT_COLS come first, followed by the rest
other_cols = [col for col in df.columns if col not in IMPORTANT_COLS]

df = df[IMPORTANT_COLS + other_cols]
run_restart_info = df[['name', 'test_cache_name', 'base_run', 'run_name', 'test_cache_mode']]
run_restart_info
# name is the wandb ui name. run_name is the old original
# Move the row with base_run == 'seq_brown_smC_BIG' to the top
seq_brown_mask = run_restart_info['base_run'] == 'seq_brown_smC_BIG'
if seq_brown_mask.any():
    seq_brown_row = run_restart_info[seq_brown_mask]
    other_rows = run_restart_info[~seq_brown_mask]
    run_restart_info = pd.concat([seq_brown_row, other_rows], ignore_index=True)
#%% RERUNNING RUNS
# import subprocess

# REMOTE_USER = "mtresoor"
# REMOTE_HOST = "snellius.surf.nl"
# REPO_PATH = "~/fusion-plasma-simulation"
# JOB_SCRIPT = "run_snellius_job.sh"
# SBATCH_JOB_NAME = "myjob"  # or set dynamically if needed
# REEVAL_MODE = "true"  # or "false" as needed

# for idx, row in run_restart_info.iterrows():
#     # Set new run name to the first split part of 'name'
#     new_run_name = row['name'].split()[0]
#     base_run = row['base_run']
#     # if new_run_name == "FM-Sequence-Brownian":
#     #     print("skip")
#     #     continue
#     print("running new: ", new_run_name, "base run:", base_run)
#     ssh_command = (
#         f"ssh -T -o LogLevel=ERROR snellius << EOF\n"
#         f"    cd {REPO_PATH}\n"
#         f"    sbatch --job-name={new_run_name} {JOB_SCRIPT} {new_run_name} {REEVAL_MODE} base_run={base_run}\n"
#         f"EOF"
#     )
#     # print("Running command:\n", ssh_command)
#     # Uncomment the next line to actually run the command
#     subprocess.run(ssh_command, shell=True, check=True)
# # REMOTE_USER="mtresoor"
# # REMOTE_HOST="snellius.surf.nl"
# # REPO_PATH="~/fusion-plasma-simulation"
# # JOB_SCRIPT="run_snellius_job.sh"
# # GIT_BRANCH="main"  # Branch to pull from
# # REMOTE_SLURM_DIR="mtresoor@snellius:/home/$REMOTE_USER/fusion-plasma-simulation/output/slurms"
# # ssh -T -o LogLevel=ERROR snellius << EOF






#%% Map cache name to the pretty thesis name and Conditioning Dim, Prior, C Variation,

CSV = "output/XR-overview.csv"

MODEL_GROUP_COLS = ['Model', 'History Length', 'Full Covariates', 'Conditioning', 'Prior']


def get_cache_overview(csvpath):
    df = pd.read_csv(csvpath)

    # Add checkmark column
    # df['cache_exists'] = df['test_cache_name'].apply(lambda v: (v + '.h5') in caches)
    # Drop columns starting with 'test/'
    df = df.loc[:, ~df.columns.str.startswith('test/')]
    # Reorder columns: group_cols + [cache_col] + rest
    other_cols = [col for col in df.columns if col not in IMPORTANT_COLS]
    df = df[IMPORTANT_COLS + other_cols]
    rename_map = {
        'model.Class': 'Model',
        'data.history_length': 'History Length',
        'model.params.model_params.c_channels': 'Full Covariates',
        'Cond_dim': 'Conditioning',
        'model.params.prior': 'Prior'
    }
    # Rename 'model.Class' values for display
    df['model.Class'] = df['model.Class'].replace({'FlowModule': 'Flow Matching', 'UnFlowModule': 'Base Unet'})
    W_GROUP_COLS = list(rename_map.values())
    df = df.rename(columns=rename_map)
    print(df)
    # Show possible unique group combinations
    possible_groups = df.set_index('test_cache_name')[W_GROUP_COLS +
                                                      ['run_name']].drop_duplicates().sort_values(W_GROUP_COLS)
    print("Possible groups:")
    possible_groups['human_name'] = """
            Unet-Channel-Brownian  
            Unet-Sequence-Brownian  
            Unet-Sequence-AllCov-Brownian  
            FM-Sequence-Tiny-Gaussian  
            FM-Channel-Brownian  
            FM-Channel-CP  
            FM-Channel-Gaussian  
            FM-Channel-Resampled  
            FM-Sequence-Brownian  
            FM-Sequence-Constant  
            FM-Sequence-CP  
            FM-Sequence-Gaussian  
            FM-Sequence-Resampled  
            FM-Channel-AllCov-Brownian  
            FM-Sequence-AllCov-Brownian  
            FM-Sequence-2x-Gaussian  """.split()
    print(possible_groups.to_string(index=True))
    return possible_groups


runs = get_cache_overview(CSV)
run_map = runs.set_index('run_name')['human_name'] + ' (' + runs.index + ')'
run_name_map = run_map.to_dict()





#%%
from src.config import find_wandb_run

# r = find_wandb_run('XR3-UnFlow_seq_brown_smC')
for run_name in run_map.index:
    run = find_wandb_run(run_name)
    if run is not None:
        print(f"Updating run: {run.name} -> {run_map[run_name]}")
        run.name = run_map[run_name]
        run.update()
        # run.sync()
    else:
        print(f"#### Run not found: {run_name} ###")
#%%
cache_experiment_map = runs[MODEL_GROUP_COLS + ['human_name']]
cache_experiment_map.loc['Ground Truth'] = ['-', '-', '-', '-', '-', 'Ground Truth']
cache_experiment_map

#%% map the cache name to the rows in runs
final_df = combined_df.join(cache_experiment_map, on='name')

#%%
MODEL_ORDER = [
    'FM-Sequence-Tiny-Gaussian',
    'FM-Sequence-Gaussian',
    'FM-Sequence-2x-Gaussian',
    'FM-Sequence-Brownian',
    'FM-Sequence-Constant',
    'FM-Sequence-CP',
    'FM-Sequence-Resampled',
    'FM-Channel-Gaussian',
    'FM-Channel-Brownian',
    'FM-Channel-CP',
    'FM-Channel-Resampled',
    'FM-Channel-AllCov-Brownian',
    'FM-Sequence-AllCov-Brownian',
    # 'Unet-Sequence-AllCov-Brownian',
    'Unet-Sequence-Brownian',
    'Unet-Channel-Brownian',
    'Ground Truth'
]
CONCAT_SORT = [
    '-',
    'sequence',
    'channels',
]
published_names_groups = cache_experiment_map.set_index('human_name')
#%% Pivot to Latex format
CONDITION = 'any_Wh'

# Sort by Model, History Length, Full Covariates, WH concat dim, Prior
MODEL_GROUP_COLS = ['Model', 'History Length', 'Full Covariates', 'Conditioning', 'Prior']
