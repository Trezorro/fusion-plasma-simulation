"""Paper single-variate model-comparison figures (PD / DML).

Purpose: paper-quality qualitative comparison of several models' generations over ONE window,
for a single observable (default PD). Two layouts, both exported at multiple PDF aspect ratios:

  * plot_stacked(...) : one subplot row per model. Shared history Wh, vertical line at t=0 (T),
                        the model's single generated future, all against the black ground truth.
                        Bottom row: ground-truth mode-label bar (L/D/H) run-length coloured.
  * plot_overlay(...) : a single trace row with every model's sample overlaid in print-safe,
                        colourblind-distinguishable colours (thin lines, so small noise shows).
                        Same bottom mode-label bar.

Inputs:  MODELS = list of (cache_name, print_name). cache_name is the HDF5 file in output/test_cache/
         (== the run's `test_cache_name`); print_name is the label shown in the figure.
         Deterministic baselines just carry one sample per window in their cache; treated identically.
Data:    src.hdf_cache.TestStepHDFCache(cache_name).quick_window(shot, t, TEST_DF)
         -> (sample[5,seq], labels_gen, labels_real, start_idx), all normalised [0,1].
         Ground truth history/future pulled from TEST_DF positionally at start_idx (as in kwali_plots.py).
Outputs: output/pdfplots/paper_single/{stacked|overlay}_{WxH}/{signal}_{shot}_{t}.pdf
Style:   serif fonts, muted print palette, minimal chrome. Built to mirror eval_notebooks/kwali_plots.py.
"""
# %% Imports
import math
from pathlib import Path
from typing import Type

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from src.config import load_config_from_file
import src.hdf_cache
import src.data_loaders

# --- Print styling (serif, thin, minimal) ---
mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.6,
    "axes.edgecolor": "0.3",
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.0,
    "figure.dpi": 120,
})

# Mode-label colours, index == label value (0=Unknown,1=L,2=D,3=H). Matches add_mode_bars in flow_plots.py.
MODE_COLORS = ["0.7", "lightskyblue", "orange", "red"]
MODE_NAMES = ["Unknown", "L", "D", "H"]

GT_COLOR = "black"          # ground-truth future
HISTORY_COLOR = "0.35"      # shared Wh, consistent across every subplot
ACCENT = "#0072B2"          # single accent for the model sample in the stacked layout

# Okabe-Ito colourblind-safe palette for the overlay layout (black reserved for GT).
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#999999"]

# %% Config + data (edit CONFIG_NAME if your caches were made with a different data config)
CONFIG_NAME = "plasmaflow"
C = load_config_from_file(CONFIG_NAME, as_omega=True)

CHANNEL_NAMES = list(C.data.cols.x)          # e.g. ["FIR_LIDs_core","PD","DML","POHM","Z_axis"]
HISTORY_LENGTH = C.data.history_length
SEQ_LENGTH = C.data.seq_length

DataSetClass: Type[src.data_loaders.FusionShotDataModule] = getattr(src.data_loaders, C.data.Class)
data_module = DataSetClass(**C.data)
data_module.prepare_data()
data_module.setup()
TEST_DF = data_module.data[data_module.data["ShotNum"].isin(data_module.test_shots)]

PDF_DIR = Path("output/pdfplots/paper_single")
PDF_DIR.mkdir(parents=True, exist_ok=True)


# %% Helpers
def _rle(labels):
    """Run-length encode a 1-D label array -> list of (start, end, value)."""
    labels = np.asarray(labels)
    spans, start = [], 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            spans.append((start, i, int(labels[start])))
            start = i
    spans.append((start, len(labels), int(labels[start])))
    return spans


def _gt_window(shot, start_idx, ch):
    """Ground-truth history and future for one channel, from TEST_DF (normalised)."""
    shot_data = TEST_DF[TEST_DF["ShotNum"] == shot]
    hist_start = max(0, start_idx - HISTORY_LENGTH)
    col = CHANNEL_NAMES[ch]
    hist = shot_data[col].iloc[hist_start:start_idx].values
    fut = shot_data[col].iloc[start_idx:start_idx + SEQ_LENGTH].values
    return hist, fut


def _sample(cache_name, shot, t, ch):
    """One model's generated future (single channel) + real mode labels + start_idx."""
    cache = src.hdf_cache.TestStepHDFCache(cache_name, mode="r")
    sample, _labels_gen, labels_real, start_idx = cache.quick_window(shot, t, TEST_DF)
    return np.asarray(sample)[ch], np.asarray(labels_real), start_idx


def _style_axis(ax):
    ax.set_facecolor("#F7F7F7")
    ax.grid(color="white", linewidth=1.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_xlim(-HISTORY_LENGTH - 1, SEQ_LENGTH + 1)
    ax.axvline(0, color="0.25", linewidth=0.8, zorder=1)


def _add_mode_bar(ax, labels_real):
    """Draw the ground-truth mode-label bar aligned so future starts at x=0."""
    for start, end, val in _rle(labels_real):
        ax.axvspan(start - HISTORY_LENGTH, end - HISTORY_LENGTH,
                   color=MODE_COLORS[val], linewidth=0, zorder=0)
    ax.axvline(0, color="0.25", linewidth=0.8)
    ax.set_xlim(-HISTORY_LENGTH - 1, SEQ_LENGTH + 1)
    ax.set_yticks([])
    ax.set_ylabel("mode", rotation=0, ha="right", va="center", labelpad=12)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def _xticks(ax, t):
    ticks = list(range(-HISTORY_LENGTH, SEQ_LENGTH + 1, 64))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"$t={t}$" if k == 0 else str(k) for k in ticks])


def _export(fig, subdir, signal, shot, t, sizes):
    for w, h in sizes:
        fig.set_size_inches(w, h)
        out = PDF_DIR / f"{subdir}_{w:.0f}x{h:.0f}" / f"{signal}_{shot}_{t}.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")


# %% Layout A: one row per model
def plot_stacked(models, shot, t, signal="PD",
                 sizes=((7, 9), (5.5, 8), (9, 11), (6, 4.5))):
    ch = CHANNEL_NAMES.index(signal)
    n = len(models)

    fig = plt.figure()
    gs = gridspec.GridSpec(n + 1, 1, height_ratios=[1] * n + [0.18], hspace=0.15)

    # Shared history/labels come from the first cache's window (target is model-independent).
    _s0, labels_real, start_idx = _sample(models[0][0], shot, t, ch)
    hist, fut = _gt_window(shot, start_idx, ch)
    xf = np.arange(len(fut))
    xh = np.arange(-len(hist), 0)

    axes = []
    for i, (cache_name, name) in enumerate(models):
        ax = fig.add_subplot(gs[i, 0])
        axes.append(ax)
        sample, _lr, s_idx = _sample(cache_name, shot, t, ch)
        ax.plot(xh, hist, color=HISTORY_COLOR, lw=0.9, ls="--", zorder=2)
        ax.plot(xf, fut, color=GT_COLOR, lw=1.1, zorder=3)
        ax.plot(np.arange(len(sample)), sample, color=ACCENT, lw=1.1, zorder=4)
        _style_axis(ax)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", labelpad=14)
        if i < n - 1:
            ax.set_xticklabels([])
        else:
            _xticks(ax, t)

    ax_mode = fig.add_subplot(gs[-1, 0])
    _add_mode_bar(ax_mode, labels_real)
    ax_mode.set_xticks([])

    handles = [Line2D([], [], color=HISTORY_COLOR, ls="--", label=r"history $x_{W_H}$"),
               Line2D([], [], color=GT_COLOR, label=r"ground truth $x_{W_F}$"),
               Line2D([], [], color=ACCENT, label="generated")]
    handles += [Patch(color=MODE_COLORS[v], label=MODE_NAMES[v]) for v in (1, 2, 3)]
    axes[0].legend(handles=handles, loc="lower left", bbox_to_anchor=(1.01, 0.0),
                   frameon=False, title=f"{signal}")
    fig.align_ylabels(axes)
    _export(fig, "stacked", signal, shot, t, sizes)
    plt.close(fig)


# %% Layout B: all models overlaid on one row
def plot_overlay(models, shot, t, signal="PD",
                 sizes=((9, 4), (7, 3.5), (12, 5), (6, 3))):
    ch = CHANNEL_NAMES.index(signal)

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.14], hspace=0.1)
    ax = fig.add_subplot(gs[0, 0])
    ax_mode = fig.add_subplot(gs[1, 0])

    _s0, labels_real, start_idx = _sample(models[0][0], shot, t, ch)
    hist, fut = _gt_window(shot, start_idx, ch)
    ax.plot(np.arange(-len(hist), 0), hist, color=HISTORY_COLOR, lw=0.9, ls="--",
            label=r"history $x_{W_H}$", zorder=2)
    ax.plot(np.arange(len(fut)), fut, color=GT_COLOR, lw=1.6, label=r"ground truth $x_{W_F}$", zorder=3)

    for i, (cache_name, name) in enumerate(models):
        sample, _lr, _s = _sample(cache_name, shot, t, ch)
        ax.plot(np.arange(len(sample)), sample, color=OKABE_ITO[i % len(OKABE_ITO)],
                lw=0.9, alpha=0.9, label=name, zorder=4)

    _style_axis(ax)
    _xticks(ax, t)
    ax.set_ylabel(signal)
    ax.legend(loc="lower left", bbox_to_anchor=(1.01, 0.0), frameon=False)

    _add_mode_bar(ax_mode, labels_real)
    ax_mode.set_xticks([])
    _export(fig, "overlay", signal, shot, t, sizes)
    plt.close(fig)


# %% Self-check (no data needed)
def _selftest():
    assert _rle([1, 1, 2, 2, 2, 3]) == [(0, 2, 1), (2, 5, 2), (5, 6, 3)]
    assert _rle([0]) == [(0, 1, 0)]
    print("selftest ok")


# %% Run: specify your models (cache_name, print_name) and windows, then plot both layouts.
MODELS = [
    ("FM-Sequence-Gaussian", "CFM (ours)"),
    ("FM-Channel-Gaussian", "CFM channel"),
    ("Unet-Sequence-Brownian", "U-Net"),
    ("FM-Sequence-CP", "CFM copy-prior"),
    ("FM-Sequence-Resampled", "CFM resampled"),
    ("FM-Sequence-Brownian", "CFM brownian"),
]

WINDOWS = list(C.window_set)  # [(shot, t), ...]

# %% Fetch any missing caches from Snellius (set AUTO_FETCH=False to only print the command)
AUTO_FETCH = True
SNELLIUS_CACHE = "snellius:/scratch-shared/mtresoor/final_cache"

missing = [n for n, _ in MODELS if not (Path("output/test_cache") / f"{n}.h5").exists()]
if missing:
    cmd = f"rsync -vz {SNELLIUS_CACHE}/{{{','.join(missing)}}}.h5 output/test_cache/"
    print(f"missing caches: {missing}\n{cmd}")
    if AUTO_FETCH:
        import subprocess
        # ponytail: brace expansion needs a shell; rsync's own exit code surfaces failures
        subprocess.run(cmd, shell=True, check=True)
else:
    print("all caches present")

if __name__ == "__main__":
    _selftest()
    for shot, t in WINDOWS:
        for signal in ("PD", "DML"):
            try:
                plot_stacked(MODELS, shot, t, signal=signal)
                plot_overlay(MODELS, shot, t, signal=signal)
            except (KeyError, ValueError) as e:
                print(f"skip {signal} {shot}@{t}: {e}")
