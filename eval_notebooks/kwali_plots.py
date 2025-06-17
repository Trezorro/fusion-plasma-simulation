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
C_CHANNEL_NAMES = ["NBI", "ECRH"]

history_length = C.data.history_length
SEQ_LENGTH = C.data.seq_length
COLOR_SCALE = plotly.colors.qualitative.Plotly
C_COLOR_SCALE = plotly.colors.qualitative.Pastel

#%%
# Organize the samples into a grid with 2 columns and as many rows as needed

def plot_one_window_many_samples(models, shot, t):
    num_samples = len(models)
    print(f"Plotting {num_samples} samples.")
    num_cols = 2
    num_rows = (num_samples // num_cols) +2
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
    for idx, model in enumerate(models):
        # model_name = model.stem
        print(f"Processing window {window} with model {model} for shot {shot} at time {t}")
        cache = src.hdf_cache.TestStepHDFCache(model)
        sample, l_pred, _l_real, start_idx = cache.quick_window(shot, t, TEST_DF)
        # Flatten axes for easy indexing if only one row
        if num_rows == 1:
            axes = axes.reshape(1, -1)

        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col]
        for ch in range(sample.shape[0]):
            sns.lineplot(x=range(sample[ch, :].shape[0]), y=sample[ch, :], ax=ax, label=CHANNEL_NAMES[ch])
        ax.set_title(f"{model}", fontsize=12, fontfamily="serif")
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
        ax.set_facecolor("#F6F6F6")
        ax.grid(color="white", linewidth=1.5)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=-1, right=SEQ_LENGTH+1)

    # for idx in range(num_samples, num_rows * num_cols):
    #     row = idx // num_cols
    #     col = idx % num_cols
    #     axes[row, col]
    

    # Get ground truth window from TEST_DF using start_idx
    shot_data = TEST_DF[TEST_DF['ShotNum'] == shot]
    histori_start = start_idx-SEQ_LENGTH
    # Plot history in the left column of the last row, merging axes if possible
    history_end = start_idx
    history_start = max(0, start_idx - history_length)
    history_x = shot_data[CHANNEL_NAMES].iloc[history_start:history_end].values.T

    # If only one row, axes is 2D, else it's already 2D
    ax_hist = axes[-1, 0]
    ax_hist.axis('off')
    
    for ch in range(history_x.shape[0]):
        sns.lineplot(x=range(history_x[ch, :].shape[0]), y=history_x[ch, :], ax=ax_hist, label=CHANNEL_NAMES[ch])
    ax_hist.set_title("History", fontsize=13, fontfamily="serif")
    if ax_hist.get_legend():
        ax_hist.legend_.remove()
    sns.despine(ax=ax_hist)
    ax_hist.set_facecolor("#F6F6F6")
    ax_hist.grid(color="white", linewidth=1.5)
    ax_hist.set_ylim(bottom=0)
    ax_hist.set_xlim(left=0)
    end_i = start_idx + SEQ_LENGTH
    x = shot_data[CHANNEL_NAMES].iloc[start_idx:end_i].values.T
    # Plot ground truth in the last subplot
    ax = axes[-1, -1]
    for ch in range(x.shape[0]):
        sns.lineplot(x=range(x[ch, :].shape[0]), y=x[ch, :], ax=ax, label=CHANNEL_NAMES[ch])
    ax.set_title("Ground Truth", fontsize=13, fontfamily="serif")
    if ax.get_legend():
        ax.legend_.remove()
    sns.despine(ax=ax)
    ax.set_facecolor("#F6F6F6")
    ax.grid(color="white", linewidth=1.5)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    plt.suptitle(rf"Generated $\hat{{\mathbf{{x}}}}_{{W_F}}$ samples for shot {shot} at $t={t}$", fontsize=14, fontfamily="serif")
    plt.show()


# for window in windows:
#     samples = []
#     for cache_path in CACHED_H5_LIST[:4]:
#         print(f"Processing window {window} with cache {cache_path.stem}")
#         cache = src.hdf_cache.TestStepHDFCache(cache_path.stem)
#         try:
#             data, l_pred, l_real = cache.quick_window(*window, TEST_DF)
#             samples.append(data)

#         except KeyError:
#             print(f"Window {window} not found in cache {cache_path.stem}")
#             continue

#     break


# %%
MODELS = [
            'Unet-Channel-Brownian',
            'Unet-Sequence-Brownian',
            'Unet-Sequence-AllCov-Brownian',
            'FM-Sequence-AllCov-Brownian',
            # 'FM-Channel-AllCov-Brownian',
            'FM-Sequence-Constant',
            'FM-Channel-CP',
            'FM-Sequence-CP',
            'FM-Channel-Resampled',
            'FM-Sequence-Resampled',
            'FM-Channel-Brownian',
            'FM-Sequence-Brownian',
            'FM-Sequence-Tiny-Gaussian',
            'FM-Sequence-2x-Gaussian',
            'FM-Sequence-Gaussian',
            'FM-Channel-Gaussian',
            'Ground Truth'  # sourced from 'FM-Sequence-Gaussian' -> distribution == Real
        ]

SEQ_VS_CHANNEL = [
            'FM-Sequence-Gaussian',
            'FM-Channel-Gaussian',
            'FM-Sequence-Brownian',
            'FM-Channel-Brownian',
            # 'Unet-Sequence-Brownian',
            # 'Unet-Channel-Brownian',
            # 'Unet-Sequence-AllCov-Brownian',
            # 'FM-Sequence-AllCov-Brownian',
            # 'FM-Channel-AllCov-Brownian',
            'FM-Sequence-CP',
            'FM-Channel-CP',
            'FM-Sequence-Resampled',
            'FM-Channel-Resampled',
            # 'FM-Sequence-Constant',
            # # 'FM-Sequence-Tiny-Gaussian',
            # # 'FM-Sequence-2x-Gaussian',
            # 'Ground Truth'  # sourced from 'FM-Sequence-Gaussian' -> distribution == Real
        ]
plot_one_window_many_samples(SEQ_VS_CHANNEL, shot, t)

# %%
for shot, t in windows:
    