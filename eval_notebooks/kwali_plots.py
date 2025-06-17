#%% Imports
from typing import Type
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import plotly
import seaborn as sns
from src.config import load_config_from_file
import src.hdf_cache
import src.data_loaders
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('src').setLevel(logging.DEBUG)

CACHE_DIR = Path('output/test_cache')  # contains .h5 and .jsons
C = load_config_from_file('fm_toy', as_omega=True)
# List all .h5 files in the CACHE_DIR
CACHED_H5_LIST = list(CACHE_DIR.glob('*.h5'))
CACHED_H5_LIST
#%%
windows = C.window_set
# Load validation dataset

DataSetClass: Type[src.data_loaders.FusionShotDataModule] = getattr(src.data_loaders, C.data.Class)
data_module = DataSetClass(**C.data)

data_module.prepare_data()
data_module.setup()
TEST_DF = data_module.data[data_module.data['ShotNum'].isin(data_module.test_shots)]
TEST_DF
# %%
from importlib import reload
reload(src.hdf_cache)

CHANNEL_NAMES = C.data.cols.x
history_length = C.data.history_length
seq_length = C.data.seq_length
COLOR_SCALE = plotly.colors.qualitative.Plotly
C_COLOR_SCALE = plotly.colors.qualitative.Pastel

#%%
# Organize the samples into a grid with 2 columns and as many rows as needed

def plot_one_window_many_samples(cache_list, shot, t):
    num_samples = len(cache_list)
    print(f"Plotting {num_samples} samples.")
    num_cols = 2
    num_rows = (num_samples // num_cols)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(10, 4 * num_rows),
        sharex=True,
        sharey=True,
        layout="constrained",
        gridspec_kw={
            'hspace': 0.05,
            'wspace': -0.05
        }
    )
    for idx, cache_path in enumerate(cache_list):
        model_name = cache_path.stem
        print(f"Processing window {window} with cache {cache_path.stem} for shot {shot} at time {t}")
        cache = src.hdf_cache.TestStepHDFCache(cache_path.stem)
        sample, l_pred, _l_real = cache.quick_window(shot, t, TEST_DF)
        # Flatten axes for easy indexing if only one row
        if num_rows == 1:
            axes = axes.reshape(1, -1)

        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col]
        for ch in range(sample.shape[0]):
            sns.lineplot(x=range(sample[ch, :].shape[0]), y=sample[ch, :], ax=ax, label=CHANNEL_NAMES[ch])
        ax.set_title(f"{model_name}", fontsize=12)
        fig.legends.clear()
        if idx == num_samples - 1:
            ax.legend(
                loc='upper left',
                bbox_to_anchor=(1.01, 0.4),
                borderaxespad=0.0,
                fontsize='small',
                title=r'$\mathbf{x}_{W_F}$',
                title_fontsize='small',
                frameon=False
            )
        else:
            ax.legend_.remove() if ax.get_legend() else None
        sns.despine(ax=ax)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)

    for idx in range(num_samples, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].axis('off')
    # plt.tight_layout()
    plt.suptitle("5-Channel Time Series", fontsize=16)
    plt.show()


for window in windows:
    samples = []
    for cache_path in CACHED_H5_LIST[:4]:
        print(f"Processing window {window} with cache {cache_path.stem}")
        cache = src.hdf_cache.TestStepHDFCache(cache_path.stem)
        try:
            data, l_pred, l_real = cache.quick_window(*window, TEST_DF)
            samples.append(data)

        except KeyError:
            print(f"Window {window} not found in cache {cache_path.stem}")
            continue

    break

# %%
shot, t = windows[3]
plot_one_window_many_samples(CACHED_H5_LIST[:4], shot, t)

# %%
