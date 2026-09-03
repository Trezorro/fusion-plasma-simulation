"""Qualitative multi-model sample comparison panels (ground truth vs N generated samples per model), with history context and control signals overlaid.

Scientific output: thesis qualitative generative-quality figures (the "sequence vs channel" model comparison panels).
Inputs:  config via load_config_from_file('fm_toy', as_omega=True); FusionShotDataModule test dataframe (TEST_DF); HDF5 caches in output/test_cache/ read through src.hdf_cache.TestStepHDFCache(model_name).quick_window(shot, t, TEST_DF); WINDOWS = C.window_set; hardcoded model-name lists (SEQ_VS_CHANNEL pairs, AllCov variants).
Outputs: output/pdfplots/seqVSchannel/multimodel_{WxH}/kwali_{SUBNAME}_{shot}_{t}.pdf across the 10 aspect ratios exported per window; inline display.
Usage:   every referenced model cache must exist locally; iterate the WINDOWS loop; SUBNAME is reassigned between cells ("seq_vs_channel" then "new" then "allC"); edit the model-name lists to match available caches.
Limits:  fragile; hardcoded 'fm_toy' config, cache names, and channel naming; SUBNAME reused across cells so output subfolders differ per cell run.
Handy:   export_pdf() multi-aspect-ratio PDF exporter and plot_one_window_many_samples() grid builder are the reusable pieces; candidates for src/plotters/.
History: created Jun 17 2025 ("Plot multiple models together, one window"), 4 same-day iterations, final polish Oct 11 2025 ("Final table and plot scripts").
"""
#%% Imports
import math
from typing import Type
from pathlib import Path
import os
import matplotlib.pyplot as plt
import plotly
from psutil import WINDOWS
import seaborn as sns
from src.config import load_config_from_file
import src.hdf_cache
import src.data_loaders
import logging
import matplotlib.gridspec as gridspec

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
WINDOWS = C.window_set
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
COLOR_SCALE = [c for c in  plotly.colors.qualitative.Plotly]
C_COLOR_SCALE = [c for c in  plotly.colors.qualitative.Alphabet]
# C_COLOR_SCALE = plotly.colors.qualitative.Pastel

windows = WINDOWS

PDF_DIR = Path("output/pdfplots/seqVSchannel")
PDF_DIR.mkdir(exist_ok=True)
SUBNAME = "seq_vs_channel"

def export_pdf(
    shot,
    t,
    w=5.77,
    h=9.69,
    factor=1,
):
    w, h = w * factor, h * factor
    fig = plt.gcf()  # Get the current figure
    fig.set_size_inches(w, h)  # Swap width and height for vertical
    pdf_path = PDF_DIR / f"multimodel_{w:.1f}x{h:.1f}" / f"kwali_{SUBNAME}_{shot}_{t}.pdf"
    pdf_path.parent.mkdir(parents=False, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches='tight')

#%%
# Organize the samples into a grid with 2 columns and as many rows as needed

def plot_one_window_many_samples(models, shot, t):
    num_samples = len(models)
    print(f"Plotting {num_samples} samples.")
    num_cols = 2
    num_rows = math.ceil(num_samples / num_cols) + 2
    sns.set_theme(style="whitegrid")

    fig = plt.figure(figsize=(12, 4 * num_rows))
    gs = gridspec.GridSpec(num_rows, num_cols, height_ratios=[1] * (num_rows - 2) + [1, 0.5])

    axes = []
    ax_ground_truth = fig.add_subplot(gs[-2, :])  # Span both columns, will be used as the leading axis
    for idx in range(num_samples):
        row = idx // num_cols
        col = idx % num_cols
        ax = fig.add_subplot(gs[row, col], sharey=ax_ground_truth)  # Share y-axis with the ground truth plot
        axes.append(ax)
        print(f"Processing window {shot} with model {models[idx]} at time {t}")
        cache = src.hdf_cache.TestStepHDFCache(models[idx])
        sample, l_pred, _l_real, start_idx = cache.quick_window(shot, t, TEST_DF)

        for ch in range(sample.shape[0]):
            sns.lineplot(x=range(sample[ch, :].shape[0]), y=sample[ch, :], ax=ax, label=CHANNEL_NAMES[ch], color=COLOR_SCALE[ch])
        ax.set_title(f"{models[idx]}", fontsize=12, fontfamily="serif")
        fig.legends.clear()
        if idx == num_samples - 1:
            ax.legend(
                loc='lower left',
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
        ax.set_xlim(left=-1, right=SEQ_LENGTH + 1)
        ax.axvline(x=0, color="darkblue", linewidth=1)  # Annotate vertical line at x=0

    plt.subplots_adjust(top=0.9)  # Reduce whitespace above the subplots

    for idx, ax in enumerate(axes):
        if idx < len(axes) - 2:  # Hide tick labels for all axes except the last two wide plots
            ax.set_xticklabels([])
        # if idx == len(axes) -3:
        #     ax.set_xticks(range(50, 200 + 1, 50))
    shot_data = TEST_DF[TEST_DF['ShotNum'] == shot]
    history_end = start_idx
    history_start = max(0, start_idx - history_length)
    history_x = shot_data[CHANNEL_NAMES].iloc[history_start:history_end].values.T
    end_i = start_idx + SEQ_LENGTH
    x = shot_data[CHANNEL_NAMES].iloc[start_idx:end_i].values.T

    for ch in range(history_x.shape[0]):
        sns.lineplot(
            x=range(-history_x[ch, :].shape[0], 0),
            y=history_x[ch, :],
            ax=ax_ground_truth,
            linestyle="dashed",
            label=f"History {CHANNEL_NAMES[ch]}",
            color=COLOR_SCALE[ch],
            alpha=0.9  # Lower opacity for history traces
        )
        sns.lineplot(x=range(x[ch, :].shape[0]), y=x[ch, :], ax=ax_ground_truth, label=f"Ground Truth {CHANNEL_NAMES[ch]}", color=COLOR_SCALE[ch])
    ax_ground_truth.axvline(x=0, color="darkblue", linewidth=2)  # Annotate vertical line at x=0
    ax_ground_truth.set_xticks(range(-250, SEQ_LENGTH + 1, 50))  # Set ticks every 50 steps
    ax_ground_truth.set_xticklabels([f"$t={t}$" if tick == 0 else str(tick) for tick in range(-250, SEQ_LENGTH + 1, 50)])  # Replace 0 tick with $t$
    # ax_ground_truth.set_title(r"Ground Truth $\mathbf{x}_W$", fontsize=13, fontfamily="serif")
    sns.despine(ax=ax_ground_truth)
    ax_ground_truth.set_facecolor("#F6F6F6")
    ax_ground_truth.grid(color="white", linewidth=1.5)
    ax_ground_truth.set_ylim(bottom=0, top=1)
    ax_ground_truth.text(
        1.01,
        0.5,
        f"Shot {shot}\n\n\nGround Truth\n$x_W$",
        transform=ax_ground_truth.transAxes,
        fontsize=13,
        fontfamily="serif",
        rotation=0,
        va="center",
        ha="left"
    )
    if ax_ground_truth.get_legend():
        ax_ground_truth.get_legend().remove()  # Correctly remove legend

    cwh = shot_data[C_CHANNEL_NAMES].iloc[history_start:history_end].values.T
    cwf = shot_data[C_CHANNEL_NAMES].iloc[start_idx:end_i].values.T

    ax_controls = fig.add_subplot(gs[-1, :])  # Span both columns, no shared y-axis
    for ch in range(len(C_CHANNEL_NAMES)):
        sns.lineplot(x=range(-cwh[ch, :].shape[0], 0), y=cwh[ch, :], alpha=0.6, ax=ax_controls, label=rf"{C_CHANNEL_NAMES[ch]}", color=C_COLOR_SCALE[ch])
        sns.lineplot(x=range(cwf[ch, :].shape[0]), y=cwf[ch, :], ax=ax_controls, color=C_COLOR_SCALE[ch], legend=False)
    ax_controls.axvline(x=0, color="darkblue", linewidth=2)  # Annotate vertical line at x=0
    sns.despine(ax=ax_controls)
    ax_controls.set_facecolor("#F6F6F6")
    ax_controls.grid(color="white", linewidth=1.5)
    ax_controls.set_ylim(bottom=0, top =1)
    # ax_controls.text(1.02, 1.1, "Controls", transform=ax_controls.transAxes, fontsize=12, fontfamily="serif", rotation=0, va="center", ha="left")
    ax_controls.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize='small',
        title=r'$\mathbf{c}_{W}$',
        title_fontsize='small',
        frameon=False
    )
    plt.subplots_adjust(top=0.9)  # Reduce whitespace above the subplots
    plt.tight_layout()
    # plt.suptitle(rf"Generated $\hat{{\mathbf{{x}}}}_{{W_F}}$ samples for shot {shot} at $t={t}$", fontsize=14, fontfamily="serif")
    # plt.show()
    w, h = fig.get_size_inches()
    # Save a more vertical version (portrait orientation)
    export_pdf(shot, t, 15, 10, factor=1)
    export_pdf(shot, t, 18, 10, factor=1)
    export_pdf(shot, t, 15, 8, factor=1)
    export_pdf(shot, t, 5.77, 9.69/2, factor=2)
    export_pdf(shot, t, w, h, factor=1)
    w, h = 5.77 * 1.2, 9.69 * 1.2
    export_pdf(shot, t, w, h, factor=1)
    w, h = 5.77 * 2, 9.69 * 2
    export_pdf(shot, t, w, h, factor=1)
    w, h = 5.77 * 2.5, 9.69 * 2.5
    export_pdf(shot, t, w, h, factor=1)
    w, h = 15, 14
    export_pdf(shot, t, w, h, factor=1)
    w, h = 16, 12
    export_pdf(shot, t, w, h, factor=1)
    # plt.show()
    plt.close()



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
            # 'Unet-Sequence-AllCov-Brownian',
            # 'FM-Sequence-AllCov-Brownian',
            # 'FM-Channel-AllCov-Brownian',
            'FM-Sequence-CP',
            'FM-Channel-CP',
            'FM-Sequence-Resampled',
            'FM-Channel-Resampled',
            'Unet-Sequence-Brownian',
            'Unet-Channel-Brownian',
            # 'FM-Sequence-Constant',
            # # 'FM-Sequence-Tiny-Gaussian',
            # # 'FM-Sequence-2x-Gaussian',
            # 'Ground Truth'  # sourced from 'FM-Sequence-Gaussian' -> distribution == Real
        ]

shot, t = WINDOWS[1]
plot_one_window_many_samples(SEQ_VS_CHANNEL, shot, t)

# %%
SUBNAME = "new"

for shot, t in windows:
    print("plotting", shot, t)
    try:
        plot_one_window_many_samples(SEQ_VS_CHANNEL, shot, t)
    except ValueError as e:
        logger.error(f"Error plotting shot {shot} at time {t}: {e}")
        continue
# %%

SUBNAME = "allC"

SEQ_VS_CHANNEL = [
    # 'FM-Sequence-Gaussian',
    # 'FM-Channel-Gaussian',
    'FM-Sequence-Brownian',
    'FM-Sequence-AllCov-Brownian',
    'FM-Channel-Brownian',
    'FM-Channel-AllCov-Brownian',
    # 'FM-Sequence-CP',
    # 'FM-Channel-CP',
    # 'FM-Sequence-Resampled',
    # 'FM-Channel-Resampled',
    'Unet-Sequence-Brownian',
    'Unet-Sequence-AllCov-Brownian',
    'Unet-Channel-Brownian',
    'FM-Sequence-Constant',
    # # 'FM-Sequence-Tiny-Gaussian',
    # # 'FM-Sequence-2x-Gaussian',
    # 'Ground Truth'  # sourced from 'FM-Sequence-Gaussian' -> distribution == Real
]

C_CHANNEL_NAMES = [
'NBI',
'ECRH',
    'IP',
'gas_fringes',
'a_minor',
'KAPPA',
'DELTA',
]

for shot, t in windows:
    print("plotting", shot, t)
    try:
        plot_one_window_many_samples(SEQ_VS_CHANNEL, shot, t)
    except ValueError as e:
        logger.error(f"Error plotting shot {shot} at time {t}: {e}")
        continue

# %%
