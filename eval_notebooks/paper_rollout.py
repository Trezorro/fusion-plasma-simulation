"""Paper autoregressive-rollout figures: full X + C + surrogate/real mode labels per rollout.

Purpose: printable overview of ONE autoregressive rollout: all five observables with the
generated trace against the real one, the control covariates, and two mode-label bars
(surrogate labels on the generated trace vs surrogate labels on the real trace). The x axis
is actual shot time in seconds; the rollout start T is a vertical line, the original real
history window W_H sits left of it, and thin dotted lines mark the chained window boundaries.

Inputs:  CACHE_NAME = the rollout cache in output/test_cache/ ({test_cache_name}_rollout.h5,
         written by src/rollout.py during the test phase). Real traces are re-derived from the
         parquet via the data module; the cache stores only generated data and labels.
Outputs: output/pdfplots/paper_rollout/{WxH}/{shot}_{frac}.pdf
         (override the directory with the ROLLOUT_PDF_DIR env var, e.g. output/testplots/...).
Style:   mirrors eval_notebooks/paper_single_variate.py (serif, thin, minimal).

Run:     PYTHONPATH=. python eval_notebooks/paper_rollout.py   (from repo root; the shell is
         already inside the pipenv venv). Without PYTHONPATH, `import src` fails.

Mode labels: the cache's surr_labels_* are the FNOLSTM argmax and therefore UNSHIFTED:
         0=L, 1=D, 2=H, never Unknown. MODE_COLORS/MODE_NAMES below are indexed directly by
         label value. This is NOT the +1-shifted LHD_label convention; see
         eval_notebooks/paper_single_variate.py's docstring and docs/evaluation-metrics.md.
"""
# %% Imports
import os
import sys
from pathlib import Path
from typing import Type

# Put the repo root on sys.path so `import src` works under VSCode's "Run Python File" (and any plain
# `python eval_notebooks/paper_rollout.py`), which puts THIS file's dir on sys.path, not the root.
_REPO_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from src.config import load_config_from_file
import src.data_loaders
from src.hdf_cache import RolloutHDFCache
from src.rollout import build_rollout_records, load_results_from_cache

# --- Print styling (serif, thin, minimal), mirroring paper_single_variate.py ---
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

# Unshifted surrogate convention, index == label value (see module docstring).
MODE_COLORS = ["lightskyblue", "orange", "red"]
MODE_NAMES = ["L", "D", "H"]

GT_COLOR = "black"
HISTORY_COLOR = "0.55"
HISTORY_ALPHA = 0.85
GENERATED_COLOR = "#D55E00"  # Okabe-Ito vermillion, same as the interactive browser
# C-row palette matches the interactive browser and avoids the mode-bar colours
# (D is orange, L is light blue, generated is vermillion).
C_COLORS = ["#0072B2", "#009E73", "#CC79A7", "#E69F00"]
MODE_BAR_RATIO = 0.09
C_ROW_RATIO = 0.6

# %% Config + data
CONFIG_NAME = "plasmaflow"
C = load_config_from_file(CONFIG_NAME, as_omega=True)

CHANNEL_NAMES = list(C.data.cols.x)
C_NAMES = list(C.data.cols.c)
HISTORY_LENGTH = C.data.history_length

DataSetClass: Type[src.data_loaders.FusionShotDataModule] = getattr(src.data_loaders, C.data.Class)
data_module = DataSetClass(**C.data)
data_module.prepare_data()
data_module.setup()

PDF_DIR = Path(os.environ.get("ROLLOUT_PDF_DIR", "output/pdfplots/paper_rollout"))
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


def _style_axis(ax, xlim):
    ax.set_facecolor("#F7F7F7")
    ax.grid(color="white", linewidth=1.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(*xlim)


def _add_mode_bar(ax, labels, times, name, xlim, t_start):
    """Mode-label bar over shot time; labels align 1:1 with the times array."""
    for start, end, val in _rle(labels):
        ax.axvspan(times[start], times[min(end, len(times) - 1)],
                   color=MODE_COLORS[val], linewidth=0, zorder=0)
    ax.axvline(t_start, color="0.25", linewidth=0.8)
    ax.set_xlim(*xlim)
    ax.set_yticks([])
    ax.set_ylabel(name, rotation=0, ha="right", va="center", labelpad=12)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def _export(fig, shot, frac, sizes):
    for w, h in sizes:
        fig.set_size_inches(w, h)
        out = PDF_DIR / f"{w:.0f}x{h:.0f}" / f"{shot}_{frac:.2f}.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print("Succesfully exported", out)


# %% The figure: 5 X rows + C row + 2 mode bars, one rollout per figure
def plot_rollout(record, sizes=((9, 11), (7, 9), (12, 14))):
    times = record["times"]
    hl = int(record["history_length"])
    t_start = float(record["t_start"])
    xlim = (float(times[0]), float(times[-1]))
    gen_times = times[hl:]
    step = int(record["step"])
    boundary_times = [times[hl + b] for b in range(step, record["generated_x"].shape[-1], step)]

    n_x = len(CHANNEL_NAMES)
    fig = plt.figure()
    gs = gridspec.GridSpec(
        n_x + 3, 1,
        height_ratios=[1] * n_x + [C_ROW_RATIO] + [MODE_BAR_RATIO] * 2, hspace=0.18
    )

    axes = []
    for ch in range(n_x):
        ax = fig.add_subplot(gs[ch, 0])
        axes.append(ax)
        ax.plot(times[:hl], record["real_x"][ch, :hl], color=HISTORY_COLOR, lw=0.9,
                alpha=HISTORY_ALPHA, zorder=2)
        ax.plot(gen_times, record["real_x"][ch, hl:], color=GT_COLOR, lw=0.8, zorder=3)
        ax.plot(gen_times, record["generated_x"][ch], color=GENERATED_COLOR, lw=0.8,
                alpha=0.9, zorder=4)
        for t_b in boundary_times:
            ax.axvline(t_b, color="0.6", linewidth=0.4, linestyle=":", zorder=1)
        ax.axvline(t_start, color="0.25", linewidth=0.8, zorder=5)
        _style_axis(ax, xlim)
        ax.set_ylabel(CHANNEL_NAMES[ch], rotation=0, ha="right", va="center", labelpad=14)
        ax.set_xticklabels([])

    ax_c = fig.add_subplot(gs[n_x, 0])
    for ci, c_name in enumerate(C_NAMES):
        ax_c.plot(times, record["real_c"][ci], color=C_COLORS[ci % len(C_COLORS)],
                  lw=0.9, label=c_name)
    ax_c.axvline(t_start, color="0.25", linewidth=0.8)
    _style_axis(ax_c, xlim)
    ax_c.set_ylabel("C", rotation=0, ha="right", va="center", labelpad=14)
    ax_c.set_xticklabels([])
    axes.append(ax_c)

    ax_gen = fig.add_subplot(gs[n_x + 1, 0])
    _add_mode_bar(ax_gen, record["surr_labels_gen"], times, "gen", xlim, t_start)
    ax_gen.set_xticklabels([])
    ax_real = fig.add_subplot(gs[n_x + 2, 0])
    _add_mode_bar(ax_real, record["surr_labels_real"], times, "real", xlim, t_start)
    ax_real.set_xlabel("Shot time (s)")

    handles = [
        Line2D([], [], color=HISTORY_COLOR, alpha=HISTORY_ALPHA, label=r"real history $x_{W_H}$"),
        Line2D([], [], color=GT_COLOR, label="real"),
        Line2D([], [], color=GENERATED_COLOR, label="generated (rollout)"),
    ]
    handles += [Line2D([], [], color=C_COLORS[ci % len(C_COLORS)], label=c_name)
                for ci, c_name in enumerate(C_NAMES)]
    handles += [Patch(color=c, label=n) for c, n in zip(MODE_COLORS, MODE_NAMES)]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(
        f"Shot {record['shot_number']}, rollout from {record['start_frac']:.0%} "
        f"(t={t_start:.2f}s, {record['n_windows']} windows)", y=0.91
    )
    fig.align_ylabels(axes)
    _export(fig, record["shot_number"], record["start_frac"], sizes)
    plt.close(fig)


# %% Run: pick the rollout cache and which rollouts to print
CACHE_NAME = os.environ.get("ROLLOUT_CACHE_NAME", "R-NormalMidAttSig03_anim_rollout")
SHOTS = None  # None = all cached rollouts; or a list like [57013, 61237, 64770, 77604]

# %% Fetch the cache from Snellius if missing (set AUTO_FETCH=False to only print the command)
AUTO_FETCH = True
SNELLIUS_CACHE = "snellius:/scratch-shared/mtresoor/final_cache"

if not (Path("output/test_cache") / f"{CACHE_NAME}.h5").exists():
    cmd = f"rsync -vz {SNELLIUS_CACHE}/{CACHE_NAME}.h5 output/test_cache/"
    print(f"missing cache: {CACHE_NAME}\n{cmd}")
    if AUTO_FETCH:
        import subprocess
        subprocess.run(cmd, shell=True, check=True)
else:
    print("cache present")

if __name__ == "__main__":
    cache = RolloutHDFCache(CACHE_NAME, mode="r")
    results = load_results_from_cache(cache)
    step = cache.get_rollout(*cache.list_rollouts()[0])["step"]
    records = build_rollout_records(results, data_module, step=step, shots=SHOTS)
    print(f"{len(records)} rollouts to plot from {CACHE_NAME}")
    for record in records:
        plot_rollout(record)

# %%
