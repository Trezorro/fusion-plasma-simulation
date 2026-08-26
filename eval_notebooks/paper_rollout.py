"""Paper autoregressive-rollout figures: full X + C + surrogate/real mode labels per rollout.

Purpose: printable overview of ONE autoregressive rollout: all five observables with the
generated trace against the real one, the control covariates, and two mode-label bars
(surrogate labels on the generated trace vs surrogate labels on the real trace). The x axis
is actual shot time in seconds; the rollout start T is a vertical line, the original real
history window W_H sits left of it, and thin dotted lines mark the chained window boundaries.

Inputs:  CACHE_NAME = the rollout cache in output/test_cache/ ({test_cache_name}_rollout.h5,
         written by src/rollout.py during the test phase). Real traces are re-derived from the
         parquet via the data module; the cache stores only generated data and labels.
Outputs: output/pdfplots/paper_rollout/{WxH}/{shot}_{frac}_s{sample_idx}.pdf
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

# Physics-symbol labels from docs/variable_reference.md, keyed by the parquet column name.
# `PD` is the renamed `Halpha1` column (see docs/data-pipeline.md column-rename table), so it
# takes the Halpha1 symbol. Falls back to the raw column name if a channel isn't in this table.
VAR_LATEX = {
    "FIR_LIDs_core": r"$n_{e,\text{core}}$",
    "FIR_LIDs_LFS": r"$n_{e,\text{LFS}}$",
    "PD": r"$\text{PD}_{H\alpha}$",
    "Halpha1": r"$\text{PD}_{H\alpha}$",
    "Halpha13": r"$\text{PD}_{CIII}$",
    "Halpha_fft": r"$\text{PD}_{\text{FFT}}$",
    "DML": r"$\text{DML}$",
    "POHM": r"$P_{\mathit{OHM}}$",
    "Z_axis": r"$Z_{\text{axis}}$",
    "R_axis": r"$R_{\text{axis}}$",
    "IP": r"$I_{p,\mathit{ref}}$",
    "IPLA": r"$I_p$",
    "PNBI": r"$P_{\mathit{NBI}}$",
    "PNBI2": r"$P_{\mathit{NBI2}}$",
    "PECRH": r"$P_{\mathit{ECRH}}$",
    "INPWR": r"$P_{\mathit{in}}$",
    "AREA": r"$A_p$",
    "DELTA_BOTTOM": r"$\delta_{\text{bottom}}$",
    "DELTA_TOP": r"$\delta_{\text{top}}$",
    "GAP_in": r"$\Delta_{\text{in}}$",
    "GAP_out": r"$\Delta_{\text{out}}$",
    "KAPPA": r"$\kappa$",
    "MAJRAD": r"$R_0$",
    "MINRAD": r"$a$",
    "VOL": r"$V_p$",
    "BZERO": r"$B_0$",
    "Q95": r"$q_{95}$",
    "GWfr": r"$n_e/n_{\mathit{GW}}$",
    "Ne_rho_max_grad1": r"$\max(n'_{e,\text{edge}})$",
    "Ne_rho_max_grad2": r"$\max(n''_{e,\text{edge}})$",
    "TS_Ne_on_axis": r"$n_{e,0}$",
    "SXRcore": r"$\mathit{SXR}_{\text{core}}$",
    "Te_rho_max_grad1": r"$\max(T'_{e,\text{edge}})$",
    "Te_rho_max_grad2": r"$\max(T''_{e,\text{edge}})$",
    "TS_Te_on_axis": r"$T_{e,0}$",
    "P_LH": r"$P_{\mathit{LH}}$",
    "BETAN": r"$\beta_N$",
    "BETAP": r"$\beta_p$",
    "BETAT": r"$\beta_t$",
    "Wtot": r"$W_{\mathit{tot}}$",
    "H98y2calc": r"$H_{\mathit{98y2}}$",
    "Prad": r"$P_{\mathit{rad}}$",
    "PradBulk": r"$P_{\mathit{rad},\text{bulk}}$",
    "PSOL_RT": r"$P_{\mathit{rad},\text{SOL}}$",
    "LI": r"$l_i$",
    "ZEFF": r"$Z_{\mathit{eff}}$",
    "nu_e_star": r"$\nu_{e,\text{ped}}^{*}$",
    "Vloop": r"$V_{\mathit{loop}}$",
}


def _var_label(col):
    return VAR_LATEX.get(col, col)

GT_COLOR = "black"
GT_ALPHA = 0.9
HISTORY_COLOR = "0.55"
HISTORY_ALPHA = 0.85
GENERATED_COLOR = "#D55E00"  # Okabe-Ito vermillion, same as the interactive browser
# C-row palette matches the interactive browser and avoids the mode-bar colours
# (D is orange, L is light blue, generated is vermillion).
C_COLORS = ["#0072B2", "#009E73", "#CC79A7", "#E69F00"]
MODE_BAR_RATIO = 0.15
C_ROW_RATIO = 0.5
C_GAP_RATIO = 0.06  # spacer row between the C row and the mode bars

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
    ax.set_axisbelow(True)  # grid must sit below the boundary/t_start lines, not cut through them
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(*xlim)
    ax.margins(y=0.03)


def _add_boundaries(ax, boundary_times, t_start, zorder_boundary=1, zorder_start=5):
    """Dotted chained-window boundaries + solid rollout-start line, drawn above the grid."""
    for t_b in boundary_times:
        ax.axvline(t_b, color="0.6", linewidth=0.4, linestyle=":", zorder=zorder_boundary)
    ax.axvline(t_start, color="0.25", linewidth=0.8, zorder=zorder_start)


def _add_mode_bar(ax, labels, times, name, xlim, t_start):
    """Mode-label bar over shot time; labels align 1:1 with the times array."""
    for start, end, val in _rle(labels):
        ax.axvspan(times[start], times[min(end, len(times) - 1)],
                   color=MODE_COLORS[val], linewidth=0, zorder=0)
    ax.axvline(t_start, color="0.25", linewidth=0.8, zorder=5)
    ax.set_xlim(*xlim)
    ax.set_yticks([])
    ax.set_ylabel(name, rotation=0, ha="right", va="center", labelpad=12, fontsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def _place_mode_label(fig, ax_c, ax_mode_a, ax_mode_b, text_artist):
    """(Re)compute the "Mode" label's x position from the just-resized figure and place it.

    Label extents are fixed in points/inches, but the figure width changes across export
    sizes, so the correct x as a *figure fraction* is size-dependent and must be recomputed
    after every fig.set_size_inches call, not cached from the pre-export default-size draw.
    """
    if text_artist is not None:
        text_artist.remove()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_x = ax_c.yaxis.label.get_window_extent(renderer).transformed(fig.transFigure.inverted()).x1
    pos_gen = ax_mode_a.get_position()
    pos_real = ax_mode_b.get_position()
    mode_y = (pos_gen.y1 + pos_real.y0) / 2
    return fig.text(label_x, mode_y, "Mode", rotation=90, ha="right", va="center", fontsize=8)


def _export(fig, shot, frac, sample_idx, sizes, ax_c, ax_mode_a, ax_mode_b):
    mode_label = None
    for w, h in sizes:
        fig.set_size_inches(w, h)
        mode_label = _place_mode_label(fig, ax_c, ax_mode_a, ax_mode_b, mode_label)
        # sample_idx in the filename: with n_samples > 1 several rollouts share
        # (shot, frac) and would otherwise overwrite each other's PDF.
        out = PDF_DIR / f"{w:.0f}x{h:.0f}" / f"{shot}_{frac:.2f}_s{sample_idx}.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        print("Succesfully exported", out)


# %% The figure: 5 X rows + C row + 2 mode bars, one rollout per figure
def plot_rollout(record, sizes=((12,4), (7,5), (12,7), (12,5))):
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
        n_x + 4, 1,
        height_ratios=[1] * n_x + [C_ROW_RATIO, C_GAP_RATIO] + [MODE_BAR_RATIO] * 2, hspace=0.0
    )

    axes = []
    for ch in range(n_x):
        ax = fig.add_subplot(gs[ch, 0])
        axes.append(ax)
        ax.plot(times[:hl], record["real_x"][ch, :hl], color=HISTORY_COLOR, lw=0.9,
                alpha=HISTORY_ALPHA, zorder=2)
        ax.plot(gen_times, record["real_x"][ch, hl:], color=GT_COLOR, lw=0.8, zorder=3, alpha=GT_ALPHA)
        ax.plot(gen_times, record["generated_x"][ch], color=GENERATED_COLOR, lw=0.8,
                alpha=0.9, zorder=4)
        _add_boundaries(ax, boundary_times, t_start)
        _style_axis(ax, xlim)
        ax.set_ylabel(_var_label(CHANNEL_NAMES[ch]), rotation=90, ha="right", va="center", labelpad=14)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

    ax_c = fig.add_subplot(gs[n_x, 0])
    for ci, c_name in enumerate(C_NAMES):
        ax_c.plot(times, record["real_c"][ci], color=C_COLORS[ci % len(C_COLORS)],
                  lw=0.9, label=_var_label(c_name))
    _add_boundaries(ax_c, boundary_times, t_start)
    _style_axis(ax_c, xlim)
    ax_c.set_ylabel("C", rotation=90, ha="right", va="center", labelpad=14)
    ax_c.set_xticklabels([])
    ax_c.tick_params(axis="x", length=0)
    axes.append(ax_c)

    ax_mode_a = fig.add_subplot(gs[n_x + 2, 0])
    _add_mode_bar(ax_mode_a, record["surr_labels_real"], times, r"real", xlim, t_start)
    ax_mode_a.set_xticklabels([])
    ax_mode_a.tick_params(axis="x", length=0)
    ax_mode_b = fig.add_subplot(gs[n_x + 3, 0])
    _add_mode_bar(ax_mode_b, record["surr_labels_gen"], times, r"gen", xlim, t_start)
    ax_mode_b.set_xlabel("Shot time (s)")

    fig.align_ylabels(axes)

    handles = [
        Line2D([], [], color=HISTORY_COLOR, alpha=HISTORY_ALPHA, label=r"real history $x_{W_H}$"),
        Line2D([], [], color=GT_COLOR, label="real"),
        Line2D([], [], color=GENERATED_COLOR, label="generated (rollout)"),
    ]
    handles += [Line2D([], [], color=C_COLORS[ci % len(C_COLORS)], label=_var_label(c_name))
                for ci, c_name in enumerate(C_NAMES)]
    handles += [Patch(color=c, label=n) for c, n in zip(MODE_COLORS, MODE_NAMES)]
    # fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    # fig.suptitle(
    #     f"Shot {record['shot_number']}, rollout from {record['start_frac']:.0%} "
    #     f"(t={t_start:.2f}s, {record['n_windows']} windows, sample {record['sample_idx']})", y=0.90
    # )
    _export(fig, record["shot_number"], record["start_frac"], record["sample_idx"], sizes,
            ax_c, ax_mode_a, ax_mode_b)
    plt.close(fig)


# %% Run: pick the rollout cache and which rollouts to print
CACHE_NAME = os.environ.get("ROLLOUT_CACHE_NAME", "R-CNb-cfm-noleak-normal-s05-noatt-e2_rollout")
SHOTS = None  # None = all cached rollouts; or a list like [57013, 61237, 64770, 77604]
# With n_samples > 1 every (shot, frac) has several sample_idx; None prints all of them
# (44 shots x 5 fractions x n_samples PDFs per size at production scale). Set an int to
# cap how many samples per starting point get printed.
MAX_SAMPLES_PER_START = 10

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
    # Filtered at load time: with n_samples in the hundreds, the full cache can be far
    # bigger than the SHOTS/MAX_SAMPLES_PER_START subset this notebook actually prints.
    results = load_results_from_cache(cache, shots=SHOTS, max_samples=MAX_SAMPLES_PER_START)
    step = cache.get_rollout(*cache.list_rollouts()[0])["step"]
    records = build_rollout_records(results, data_module, step=step, shots=SHOTS, max_samples=10)
    print(f"{len(records)} rollouts to plot from {CACHE_NAME}")
    for record in records:
        plot_rollout(record)

# %%
