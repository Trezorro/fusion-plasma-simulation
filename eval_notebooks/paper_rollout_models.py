"""Paper appendix figure: one PD rollout per shot, with one row per main-table model.

Purpose: a compact visual companion to the depth-stratified rollout table. Where
`eval_notebooks/paper_rollout.py` prints ALL observables for ONE model, this prints ONE
observable (PD, the ELM-carrying photodiode) for ALL five main-table models stacked as
rows, so the models can be compared trace against trace on the same shot and the same
time axis. Row labels are the model names rather than the channel symbol.

Rollouts start at the same 50% mark the depth table uses (STRAT_START_FRACTION) and are
truncated to the same 12-window budget (STRAT_BINS), with the early/late split drawn as a
labelled divider so a reader can map a row onto the two blocks of the table.

Inputs:  the rollout caches named in rollout_tables.MODELS, under output/test_cache/.
         Real traces come from the parquet via the data module (the cache stores only
         generated data and labels), exactly as in paper_rollout.py.
Outputs: output/pdfplots/paper_rollout_models/{WxH}/{shot}_s{sample_idx}.pdf
         (override the directory with the ROLLOUT_MODELS_PDF_DIR env var).

Run:     PYTHONPATH=. python eval_notebooks/paper_rollout_models.py

Almost everything (styling, colours, axis helpers, the data module, the LaTeX symbol
table, mode-label conventions) is imported from paper_rollout.py; only the layout is new.
"""
# %% Imports
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from src.hdf_cache import RolloutHDFCache
from src.rollout import build_rollout_records, load_results_from_cache

# paper_rollout applies the print rcParams at import and builds the data module once.
from eval_notebooks.paper_rollout import (  # noqa: E402
    C, C_COLORS, C_NAMES, GENERATED_COLOR, GT_ALPHA, GT_COLOR, HISTORY_ALPHA,
    HISTORY_COLOR, MODE_COLORS, MODE_NAMES, C_GAP_RATIO, C_ROW_RATIO, MODE_BAR_RATIO,
    _add_boundaries, _add_mode_bar, _style_axis, _var_label, data_module,
)
from eval_notebooks.rollout_tables import MODELS, STRAT_BINS, STRAT_START_FRACTION

# %% What to plot
CHANNEL = "PD"
CHANNEL_IDX = list(C.data.cols.x).index(CHANNEL)
SAMPLE_IDX = 0  # which stochastic sample of the flow rollouts to show

HISTORY_COLOR = "0.60"
HISTORY_ALPHA = 0.9
GT_COLOR = "0.60"
GT_ALPHA = 0.9
# Hand-picked test shots spanning the regimes the table averages over. Real ELM-scale peak
# counts are the per-pool (6-window) means over the two depth strata at pi >= 0.05 on PD,
# read off output/paper_tables/rollout_pool_metrics.parquet.
SHOTS = [
    76304,  # near-ELM-free: 4 early / 2 late
    79825,  # ELM onset inside the rollout: 0 early / 45 late
    64365,  # shortest rollout from 50% (12 windows), dense late ELMs: 0 / 94
    77409,  # longest rollout from 50% (49 windows), moderate ELMs: 51 / 40
    64770,  # long and ELM-heavy: 246 / 272
    78069,  # densest ELMs in the test set: 351 / 362
    57013,
    73368,
    61237, 
    77604
]

# Truncate to the table's depth budget so the figure covers exactly the rollout the table
# scores. Set to None to draw the full cached rollout instead.
MAX_WINDOWS = max(k_hi for _, _, k_hi in STRAT_BINS) + 1

SIZES = ((6.5, 5.0),(12,4), (7,5), (12,7), (12,5))

PDF_DIR = Path(os.environ.get("ROLLOUT_MODELS_PDF_DIR", "output/pdfplots/paper_rollout_models"))
PDF_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path("output/test_cache")
SNELLIUS_CACHE = "snellius:/scratch-shared/mtresoor/final_cache"


# %% Loading
def load_records(cache_stem, shots):
    """Records at STRAT_START_FRACTION for `shots`, keyed by shot number."""
    if not (CACHE_DIR / f"{cache_stem}.h5").exists():
        cmd = f"rsync -vz {SNELLIUS_CACHE}/{cache_stem}.h5 {CACHE_DIR}/"
        print(f"missing cache: {cache_stem}\n{cmd}")
        import subprocess
        subprocess.run(cmd, shell=True, check=True)
    cache = RolloutHDFCache(cache_stem, mode="r")
    results = load_results_from_cache(cache, shots=shots, max_samples=SAMPLE_IDX + 1)
    step = cache.get_rollout(*cache.list_rollouts()[0])["step"]
    records = build_rollout_records(results, data_module, step=step, shots=shots)
    out = {}
    for r in records:
        if not np.isclose(r["start_frac"], STRAT_START_FRACTION):
            continue
        if r["sample_idx"] != SAMPLE_IDX:
            continue
        out[r["shot_number"]] = r
    return out


def _truncate(record, max_windows):
    """Cut a record down to the first `max_windows` generations (history kept intact)."""
    if max_windows is None or record["n_windows"] <= max_windows:
        return record
    hl, step = int(record["history_length"]), int(record["step"])
    seq = record["generated_x"].shape[-1] - (record["n_windows"] - 1) * step
    keep = (max_windows - 1) * step + seq
    r = dict(record)
    r["generated_x"] = record["generated_x"][:, :keep]
    r["real_x"] = record["real_x"][:, : hl + keep]
    r["real_c"] = record["real_c"][:, : hl + keep]
    r["times"] = record["times"][: hl + keep]
    r["surr_labels_real"] = record["surr_labels_real"][: hl + keep]
    r["surr_labels_gen"] = record["surr_labels_gen"][: hl + keep]
    r["n_windows"] = max_windows
    return r


# %% The figure: one PD row per model, shared C row and real mode bar
def plot_model_rollouts(shot, records, sizes=SIZES):
    """records: {model display name -> record}. Row order follows MODELS."""
    models = [m for m in MODELS if m in records]
    ref = records[models[0]]
    times = ref["times"]
    hl = int(ref["history_length"])
    step = int(ref["step"])
    t_start = float(ref["t_start"])
    xlim = (float(times[0]), float(times[-1]))
    gen_times = times[hl:]
    boundary_times = [times[hl + b] for b in range(step, ref["generated_x"].shape[-1], step)]
    # Divider between the table's two depth strata (end of the last "early" window).
    split_k = STRAT_BINS[0][2] + 1
    t_split = times[hl + split_k * step] if hl + split_k * step < len(times) else None

    n_rows = len(models)
    fig = plt.figure()
    gs = gridspec.GridSpec(
        n_rows + 3, 1,
        height_ratios=[1] * n_rows + [C_ROW_RATIO, C_GAP_RATIO, MODE_BAR_RATIO], hspace=0.0,
    )

    axes = []
    for i, model in enumerate(models):
        rec = records[model]
        ax = fig.add_subplot(gs[i, 0])
        axes.append(ax)
        ax.plot(times[:hl], rec["real_x"][CHANNEL_IDX, :hl], color=HISTORY_COLOR, lw=0.9,
                alpha=HISTORY_ALPHA, zorder=2)
        ax.plot(gen_times, rec["real_x"][CHANNEL_IDX, hl:], color=GT_COLOR, lw=0.8,
                alpha=GT_ALPHA, zorder=3)
        ax.plot(gen_times, rec["generated_x"][CHANNEL_IDX], color=GENERATED_COLOR, lw=0.8,
                alpha=0.9, zorder=4)
        _add_boundaries(ax, boundary_times, t_start)
        if t_split is not None:
            ax.axvline(t_split, color="0.25", linewidth=0.8, linestyle="--", zorder=5)
        _style_axis(ax, xlim)
        # Normalized [0,1] channel: two ticks are enough, and long model names need the room.
        ax.set_yticks([0.0, 1.0])
        ax.set_ylabel(model, rotation=90, ha="center", va="bottom", labelpad=6, fontsize=8)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

    # Controls and mode labels are properties of the shot, not of a model: drawn once.
    ax_c = fig.add_subplot(gs[n_rows, 0])
    for ci, c_name in enumerate(C_NAMES):
        ax_c.plot(times, ref["real_c"][ci], color=C_COLORS[ci % len(C_COLORS)], lw=0.9,
                  label=_var_label(c_name))
    _add_boundaries(ax_c, boundary_times, t_start)
    if t_split is not None:
        ax_c.axvline(t_split, color="0.25", linewidth=0.8, linestyle="--", zorder=5)
    _style_axis(ax_c, xlim)
    ax_c.set_yticks([0.0, 1.0])
    ax_c.set_ylabel("C", rotation=90, ha="center", va="bottom", labelpad=6, fontsize=8)
    ax_c.set_xticklabels([])
    ax_c.tick_params(axis="x", length=0)
    axes.append(ax_c)

    ax_mode = fig.add_subplot(gs[n_rows + 2, 0])
    _add_mode_bar(ax_mode, ref["surr_labels_real"], times, "real", xlim, t_start)
    ax_mode.set_xlabel("Shot time (s)")
    fig.align_ylabels(axes)

    handles = [
        Line2D([], [], color=HISTORY_COLOR, alpha=HISTORY_ALPHA, label=r"real history $x_{W_H}$"),
        Line2D([], [], color=GT_COLOR, label="real"),
        Line2D([], [], color=GENERATED_COLOR, label="generated (rollout)"),
    ]
    handles += [Line2D([], [], color=C_COLORS[ci % len(C_COLORS)], label=_var_label(c_name))
                for ci, c_name in enumerate(C_NAMES)]
    handles += [Patch(color=c, label=n) for c, n in zip(MODE_COLORS, MODE_NAMES)]

    for w, h in sizes:
        fig.set_size_inches(w, h)
        out = PDF_DIR / f"{w:.0f}x{h:.0f}" / f"{shot}_s{SAMPLE_IDX}.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        print("Succesfully exported", out)
    plt.close(fig)


# %% Run
if __name__ == "__main__":
    per_model = {}
    for model, stem in MODELS.items():
        per_model[model] = load_records(stem, SHOTS)
        print(f"{model}: {sorted(per_model[model])}")

    for shot in SHOTS:
        recs = {m: _truncate(d[shot], MAX_WINDOWS) for m, d in per_model.items() if shot in d}
        missing = [m for m in MODELS if m not in recs]
        if missing:
            print(f"shot {shot}: skipping, no rollout for {missing}")
            continue
        lengths = {len(r["times"]) for r in recs.values()}
        assert len(lengths) == 1, f"shot {shot}: models disagree on rollout length {lengths}"
        plot_model_rollouts(shot, recs)

# %%
