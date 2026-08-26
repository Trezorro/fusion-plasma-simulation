"""Paper rollout-comparison figure: PD only, three models overlaid on one ground truth.

Purpose: isolate the timing-vs-capacity argument in a single printable panel. Three
rollout caches (headline PlasmaFlow, U-Net noleak, U-Net single-leak/timing-oracle)
each contribute one stochastic sample at start_frac=0.75, overlaid on the shared black
ground truth for the PD (H-alpha) channel only. No C row, no mode bars: mirrors
eval_notebooks/paper_rollout.py's styling but strips it down to what this comparison
needs (see docs/run_grid.md for the grid-cell/cache naming this script reads).

Inputs:  three rollout caches in output/test_cache/, one per model (see MODEL_CACHES).
         Real ground truth is re-derived from the parquet via the data module and is
         identical across models for a given shot (cols.x does not vary across the
         noleak/single-leak grid cells; only cols.c does, which we never plot here).
Outputs: output/pdfplots/paper_rollout_compare/{WxH}/{shot}_0.75.pdf
         (override the directory with the ROLLOUT_COMPARE_PDF_DIR env var).

Run:     PYTHONPATH=. python eval_notebooks/paper_rollout_compare.py
"""
# %% Imports
import os
import sys
from pathlib import Path
from typing import Type

_REPO_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.config import load_config_from_file
import src.data_loaders
from src.hdf_cache import RolloutHDFCache
from src.rollout import build_rollout_records, load_results_from_cache

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

VAR_LATEX = {"PD": r"$\text{PD}_{H\alpha}$"}

GT_COLOR = "black"
GT_ALPHA = 0.6
GT_LW = 2
HISTORY_COLOR = "black"
HISTORY_ALPHA = GT_ALPHA  # same opacity as GT; only the color distinguishes history from horizon

START_FRAC = 0.75
FRAC_TOL = 1e-6
MAX_WINDOWS = 3  # cap the shown rollout horizon; the full chained rollout is much longer

# Three grid cells from docs/run_grid.md's main-text table: headline CFM (noleak) vs
# the U-Net (deterministic U-Net) isolation, noleak vs single-leak (timing oracle).
MODEL_CACHES = {
    "PlasmaFlow": "R-CNb-cfm-noleak-normal-s05-noatt-e2_rollout",
    "U-Net": "UNb-unet-noleak-noatt_rollout",
    "U-Net + timing leak in C": "USb-unet-ipla-noatt_rollout",
}
# Okabe-Ito-derived, distinct in color AND marker/linestyle so it survives grayscale print.
MODEL_STYLE = {
    "PlasmaFlow": dict(color="#D55E00", marker="o", linestyle="-"),
    "U-Net": dict(color="#0072B2", marker="s", linestyle="-"),
    "U-Net + timing leak in C": dict(color="#009E73", marker="^", linestyle="-"),
}
MARKEVERY = 25

SHOTS = None  # None = every shot cached at START_FRAC across all three models

# %% Config + data (ground truth only; identical cols.x across grid cells)
CONFIG_NAME = "plasmaflow"
C = load_config_from_file(CONFIG_NAME, as_omega=True)
CHANNEL_NAMES = list(C.data.cols.x)
PD_IDX = CHANNEL_NAMES.index("PD")

DataSetClass: Type[src.data_loaders.FusionShotDataModule] = getattr(src.data_loaders, C.data.Class)
data_module = DataSetClass(**C.data)
data_module.prepare_data()
data_module.setup()

PDF_DIR = Path(os.environ.get("ROLLOUT_COMPARE_PDF_DIR", "output/pdfplots/paper_rollout_compare"))
PDF_DIR.mkdir(parents=True, exist_ok=True)


# %% Helpers
def _style_axis(ax, xlim):
    ax.set_facecolor("#F7F7F7")
    ax.grid(color="white", linewidth=1.0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(*xlim)
    ax.margins(y=0.05)


def _load_frac_records(cache_name, shots):
    """Best (lowest sample_idx) record per shot at START_FRAC, from one model's cache."""
    if not (Path("output/test_cache") / f"{cache_name}.h5").exists():
        raise FileNotFoundError(
            f"missing cache: {cache_name}\n"
            f"rsync -vz snellius:/scratch-shared/mtresoor/final_cache/{cache_name}.h5 output/test_cache/"
        )
    cache = RolloutHDFCache(cache_name, mode="r")
    results = load_results_from_cache(cache, shots=shots)
    step = cache.get_rollout(*cache.list_rollouts()[0])["step"]
    records = build_rollout_records(results, data_module, step=step, shots=shots)
    by_shot = {}
    for r in records:
        if abs(r["start_frac"] - START_FRAC) > FRAC_TOL:
            continue
        cur = by_shot.get(r["shot_number"])
        if cur is None or r["sample_idx"] < cur["sample_idx"]:
            by_shot[r["shot_number"]] = r
    return by_shot


def _place_bottom_labels(fig, ax, mid_x, prev_artists):
    """(Re)place the W_H / x-axis-label text just under the tick numbers.

    Offsets are in points converted to axes fraction via the *current* axes height in
    inches, so the gap to the axis stays a small, fixed visual distance across the
    different export sizes instead of the huge, size-dependent gap a single fixed
    axes-fraction offset produces (axes height varies from 3in to 7in across `sizes`).
    """
    for a in prev_artists:
        a.remove()
    ax_height_in = fig.get_size_inches()[1] * ax.get_position().height
    pt_to_axfrac = lambda pts: (pts / 72.0) / ax_height_in
    tick_row_y = -pt_to_axfrac(14)  # clears the tick-number row
    xlabel_row_y = tick_row_y - pt_to_axfrac(12)
    trans = ax.get_xaxis_transform()
    t1 = ax.text(mid_x, tick_row_y, r"$W_H$", transform=trans, ha="center", va="top", fontsize=8)
    t3 = ax.text(0.5, xlabel_row_y, "Shot time (s)", transform=ax.transAxes,
                 ha="center", va="top", fontsize=9)
    return [t1, t3]


def _export(fig, ax, shot, sizes, mid_x):
    bottom_artists = []
    for w, h in sizes:
        fig.set_size_inches(w, h)
        bottom_artists = _place_bottom_labels(fig, ax, mid_x, bottom_artists)
        out = PDF_DIR / f"{w:.0f}x{h:.0f}" / f"{shot}_{START_FRAC:.2f}.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        print("Succesfully exported", out)


# %% The figure: one PD row, ground truth + three overlaid model samples
def plot_comparison(shot, model_records, sizes=((12, 4), (12,3), (9,3), (12, 7), (10,2))):
    """model_records: dict model_name -> rollout record (already filtered to START_FRAC)."""
    ref = next(iter(model_records.values()))
    times = ref["times"]
    hl = int(ref["history_length"])
    t_start = float(ref["t_start"])
    step = int(ref["step"])
    n_show = min(int(ref["generated_x"].shape[-1]), MAX_WINDOWS * step)
    gen_times = times[hl:hl + n_show]
    xlim = (float(times[0]), float(gen_times[-1]))
    boundary_times = [times[hl + b] for b in range(step, n_show, step)]

    fig, ax = plt.subplots()
    ax.plot(times[:hl], ref["real_x"][PD_IDX, :hl], color=HISTORY_COLOR, lw=0.9,
            alpha=HISTORY_ALPHA, zorder=2)
    ax.plot(gen_times, ref["real_x"][PD_IDX, hl:hl + n_show], color=GT_COLOR, lw=GT_LW,
            alpha=GT_ALPHA, zorder=3)
    for t_b in boundary_times:
        ax.axvline(t_b, color="0.6", linewidth=0.4, linestyle=":", zorder=1)
    ax.axvline(t_start, color="0.25", linewidth=0.8, zorder=5)

    for model_name, record in model_records.items():
        style = MODEL_STYLE[model_name]
        ax.plot(gen_times, record["generated_x"][PD_IDX, :n_show], lw=1.0, alpha=0.95, zorder=4,
                markevery=MARKEVERY, markersize=4, markeredgewidth=0, **style)

    _style_axis(ax, xlim)
    # Main ticks land exactly on window starts/ends instead of a generic linspace, so the
    # reader can read window width off the tick spacing directly.
    ax.set_xticks([t_start] + boundary_times + [float(gen_times[-1])])
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.3f}")
    ax.set_ylabel(VAR_LATEX["PD"], rotation=90, ha="right", va="center", labelpad=14)

    handles = [
        Line2D([], [], color=HISTORY_COLOR, alpha=HISTORY_ALPHA, label=r"real history $x_{W_H}$"),
        Line2D([], [], color=GT_COLOR, lw=GT_LW, alpha=GT_ALPHA, label="real"),
    ]
    handles += [Line2D([], [], label=name, **MODEL_STYLE[name]) for name in model_records]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=len(handles), frameon=False)

    _export(fig, ax, shot, sizes, (xlim[0] + t_start) / 2)
    plt.close(fig)


# %% Run
if __name__ == "__main__":
    per_model = {name: _load_frac_records(cache, SHOTS) for name, cache in MODEL_CACHES.items()}
    shared_shots = set.intersection(*(set(d.keys()) for d in per_model.values()))
    print(f"{len(shared_shots)} shots with a start_frac={START_FRAC} rollout in all "
          f"{len(MODEL_CACHES)} caches")
    for shot in sorted(shared_shots):
        plot_comparison(shot, {name: per_model[name][shot] for name in MODEL_CACHES})

# %%
