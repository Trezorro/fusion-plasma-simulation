#%%# Peak Metrics Analysis from Archived HDF5 Files

# This notebook loads peak property DataFrames directly from HDF5 archives created by PeakMetric, combines them, and explores their dimensions for pivot table analysis.

import os
import glob
import pandas as pd
from tqdm import tqdm
from pathlib import Path
# Find all HDF5 files in the output or cache directories
h5_pattern = "*.h5"
h5_files = list(Path('output/test_cache/').glob(h5_pattern))
print(f"Found {len(h5_files)} HDF5 files:", h5_files)
EXAMPLE_MODEL = h5_files[0]


#%%
#%%# Helper: List all /peaks/* keys in an HDF5 file
def list_peak_keys(h5_path):
    with pd.HDFStore(h5_path, "r") as store:
        return [k for k in store.keys() if k.startswith("/peaks/")]


DF_KEYS = ['/peaks/D_only_Wh', '/peaks/H_only_Wh', '/peaks/L_only_Wh', '/peaks/any_Wh', '/peaks/mixed']
list_peak_keys(EXAMPLE_MODEL)
#%%# Collect all DataFrames from all HDF5 files and all /peaks/* keys
all_peak_dfs = []
for h5_path in h5_files[:3]:
    try:
        peak_keys = list_peak_keys(h5_path)
        for key in tqdm(peak_keys):
            df = pd.read_hdf(h5_path, key=key)
            df["human_name"] = Path(h5_path).stem
            df["condition"] = key.split('/')[2]
            all_peak_dfs.append(df.query("distribution=='Generated' & channel_name=='PD'"))
    except Exception as e:
        print(f"Error reading {h5_path}: {e}")

if all_peak_dfs:
    combined_df = pd.concat(all_peak_dfs, ignore_index=True)
else:
    combined_df = pd.DataFrame()
print(f"Combined DataFrame shape: {combined_df.shape}")
combined_df.head()
#%%
del all_peak_dfs
df = combined_df.copy()
#%%## DataFrame Structure and Available Dimensions
"""
The combined DataFrame includes:
- `human_name`: Name of model / Ground Truth
- `condition`: Logical condition in {L_only_Wh (b) D_only_Wh (orange) H_only_Wh (r)  mixed (purple) any_Wh (darkgrey) }
- `channel_name`: Signal/channel (e.g., DML, PD, etc.)
- `measure`: Peak property (e.g., height, prominence, count, etc.)
  - relevant: count, prominence, width. For DML: also energy_ratio
- `distribution`: 'Generated' or 'Real'. Real will be taken as a model row.
- `value`: The measured property value of one peak, or per window for 'count'
"""

# for one channel, one measure per column, and the human_name model name, create a box plot, split out in colors per condition.
import matplotlib.pyplot as plt
import seaborn as sns
#%%
df = combined_df.sample(frac=0.011).sort_values(['human_name'])
#%%
# def plot_facet_boxplots(df):
# Define color palette for conditions
condition_order = ["L_only_Wh", "D_only_Wh", "H_only_Wh", "mixed", "any_Wh"]
condition_labels = {"L_only_Wh": "L", "D_only_Wh": "D", "H_only_Wh": "H", "mixed": "mixed", "any_Wh": "any"}
condition_palette = {'L': '#1f77b4', 'D': '#ff7f0e', 'H': '#d62728', 'mixed': 'purple', 'any': '#444444'}
df = df[df['condition'].isin(condition_order)].copy()
df['cond_label'] = df['condition'].map(condition_labels)
cond_label_order = [condition_labels[c] for c in condition_order]

models = df['human_name'].unique()
measures = ['count', 'prominence', 'width']
if 'energy_delta' in df['measure'].unique():
    measures.append('energy_delta')

n_rows = len(models)
n_cols = len(measures)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), sharey='col')

if n_rows == 1:
    axes = [axes]
if n_cols == 1:
    axes = [[ax] for ax in axes]

for i, model in enumerate(models):
    for j, measure in enumerate(measures):
        ax = axes[i][j]
        data = df[(df['human_name'] == model) & (df['measure'] == measure)]
        if not data.empty:

            sns.boxplot(
                data=data,
                x='value',
                y='cond_label',
                hue='cond_label',
                order=cond_label_order,
                palette=condition_palette,
                boxprops=dict(edgecolor='black', linewidth=1),
                # autorange=True,
                notch=True,
                ax=ax,
                showfliers=False,
                orient='h'
            )
        ax.set_title(f"{model} - {measure}", fontsize=10)
        if j == 0:
            ax.set_ylabel(model, rotation=0, labelpad=30)
        else:
            ax.set_ylabel('')
        ax.set_xlabel(model)
        sns.despine(ax=ax)
plt.tight_layout()
plt.show()

#%%## Getting Precalculated distances from config
h5_pattern = "*.json"
h5_files = Path('output/test_cache/').glob(h5_pattern)
print(f"Found {len(h5_files)} HDF5 files:", h5_files)
EXAMPLE_MODEL = h5_files[0]

####
"""
The combined DataFrame includes:
- `condition`: Logical condition (from the key, e.g., L_only_Wh)
- `channel_name`: Signal/channel (e.g., DML, PD, etc.)
- `measure`: Peak property (e.g., height, prominence, count, etc.)
- `distribution`: 'Generated' or 'Real'
- `value`: The measured value of one peak, or per window for 'count'
- `human_name`: Name of model / Ground Truth
- `condition`: HDF5 subkey key (condition)
"""

#%%
#%%# Example: Pivot table for a specific channel (e.g., 'DML')
channel = "DML"
pivot = combined_df[combined_df["channel_name"] == channel].pivot_table(
    index=["condition", "measure", "distribution"], values="value", aggfunc=["mean", "std", "count"]
)
pivot
#%%
#%%# Example: Loop over all channels and display summary tables
channels = combined_df["channel_name"].unique()
for channel in channels:
    print(f"=== Channel: {channel} ===")
    display(
        combined_df[combined_df["channel_name"] == channel].pivot_table(
            index=["condition", "measure", "distribution"], values="value", aggfunc=["mean", "std", "count"]
        )
    )
#%%# %%
