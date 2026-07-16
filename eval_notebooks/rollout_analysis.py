"""Horizon-resolved quantitative analysis of autoregressive rollouts.

The question: how does generation quality degrade as the rollout runs further from its
real starting history? A single-window evaluation cannot see feedback drift (the model
consuming its own output); these curves can.

Unit of analysis: one (rollout r, depth k) pair. A rollout r is one (shot, start
fraction, sample_idx) triple; window k covers the generated samples
[k*step, k*step+seq_length), so k is the autoregressive depth: how many windows of its
own output the model has been fed. Per (r, k), always against the real trace over the
exact same time span (both normalized [0,1]):

  * abs_mean_err/{ch} = |mean(gen_ch) - mean(real_ch)|          per observable channel
  * abs_std_err/{ch}  = |std(gen_ch)  - std(real_ch)|
    The horizon extension of the window moment errors ("right level and spread per
    channel", docs/evaluation-metrics.md section 1): if the feedback loop drifts or
    collapses to the mean, it shows up here first.
  * label_agreement   = fraction of the window's samples where the FNOLSTM surrogate
    label of the generated trace equals that of the real trace (0=L,1=D,2=H, unshifted).
  * elm_peaks_gen / elm_peaks_real = count of scipy.signal.find_peaks on the window's
    normalized PD with prominence = evaluation.peaks.elm_pd_prominence, i.e. the same
    definition as the window PeakMetric's 'PD large peaks' channel. Compared as two
    curves: does the model keep producing ELM-scale bursts at the real rate as k grows?

Aggregation: every figure is FACETED BY START FRACTION, one panel per starting point,
and within a panel the line is the median over the rollouts of that start fraction that
reach depth k (band = IQR, 25th-75th percentile over those rollouts). Pooling across
start fractions would confound depth with shot phase: at the same k, a 10% start sits in
ramp-up while a 75% start sits deep in H-mode, so they are different regimes and must not
be averaged together. The grey step (right axis) is n(k) per panel: deep k is only
reached by long shots, so the population still changes along the x axis within a panel;
a bend at large k can be a population change rather than model behaviour.

Models: MODELS is a list of (rollout_cache_name, print_name) pairs, overlaid as one line
per model within each panel (same pattern as paper_single_variate.py). The real ELM peak
rate is drawn once (black, dashed) since it is model-independent. The per-(rollout, k)
dataframe `df` keeps `model`, `shot`, `start_frac`, and `sample_idx` columns for any
further re-slicing.

Inputs:  rollout caches ({test_cache_name}_rollout.h5) in output/test_cache/, written by
         src/rollout.py. Real traces re-derived from the parquet.
Outputs: output/tables/rollout_horizon.csv (per model/start_frac/k aggregates)
         output/tables/rollout_horizon_{main_model}.tex (compact table, main model)
         output/pdfplots/rollout_analysis/{WxH}/{metric}.pdf at several sizes
         (override dirs with ROLLOUT_TABLE_DIR / ROLLOUT_PDF_DIR env vars for test runs;
         ROLLOUT_CACHE_NAME overrides the first model's cache).

Run:     PYTHONPATH=. python eval_notebooks/rollout_analysis.py

Label convention: cache surrogate labels are UNSHIFTED 0=L, 1=D, 2=H.
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
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.2,
    "figure.dpi": 120,
})

# Okabe-Ito, one color per model; black is reserved for the real reference line.
MODEL_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

# %% Config + data
CONFIG_NAME = "plasmaflow"
C = load_config_from_file(CONFIG_NAME, as_omega=True)
CHANNEL_NAMES = list(C.data.cols.x)
PD_INDEX = CHANNEL_NAMES.index("PD")
ELM_PROMINENCE = float(C.evaluation.peaks.elm_pd_prominence)

DataSetClass: Type[src.data_loaders.FusionShotDataModule] = getattr(src.data_loaders, C.data.Class)
data_module = DataSetClass(**C.data)
data_module.prepare_data()
data_module.setup()

TABLE_DIR = Path(os.environ.get("ROLLOUT_TABLE_DIR", "output/tables"))
PDF_DIR = Path(os.environ.get("ROLLOUT_PDF_DIR", "output/pdfplots/rollout_analysis"))
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

# %% Models: (rollout_cache_name, print_name); one line per model in every panel
MODELS = [
    (os.environ.get("ROLLOUT_CACHE_NAME", "R-NormalMidAttSig03_anim_rollout"), "CFM (ours)"),
    # ("R-BrownianMidAttSig1_anim_rollout", "CFM brownian"),
]

# %% Fetch missing caches from Snellius (set AUTO_FETCH=False to only print the command)
AUTO_FETCH = True
SNELLIUS_CACHE = "snellius:/scratch-shared/mtresoor/final_cache"

for cache_name, _ in MODELS:
    if not (Path("output/test_cache") / f"{cache_name}.h5").exists():
        cmd = f"rsync -vz {SNELLIUS_CACHE}/{cache_name}.h5 output/test_cache/"
        print(f"missing cache: {cache_name}\n{cmd}")
        if AUTO_FETCH:
            import subprocess
            subprocess.run(cmd, shell=True, check=True)


# %% Per-(rollout, window) metric rows
def _window_rows(record, step, seq_length):
    """One row of metrics per autoregressive window index k of one rollout."""
    hl = int(record["history_length"])
    gen = record["generated_x"]
    real = record["real_x"][:, hl:]
    lab_gen = record["surr_labels_gen"][hl:]
    lab_real = record["surr_labels_real"][hl:]
    rows = []
    for k in range(int(record["n_windows"])):
        sl = slice(k * step, k * step + seq_length)
        row = {
            "shot": record["shot_number"],
            "start_frac": record["start_frac"],
            "sample_idx": record["sample_idx"],
            "k": k,
            "label_agreement": float((lab_gen[sl] == lab_real[sl]).mean()),
            "elm_peaks_gen": len(find_peaks(gen[PD_INDEX, sl], prominence=ELM_PROMINENCE)[0]),
            "elm_peaks_real": len(find_peaks(real[PD_INDEX, sl], prominence=ELM_PROMINENCE)[0]),
        }
        for ch, name in enumerate(CHANNEL_NAMES):
            row[f"abs_mean_err/{name}"] = float(abs(gen[ch, sl].mean() - real[ch, sl].mean()))
            row[f"abs_std_err/{name}"] = float(abs(gen[ch, sl].std() - real[ch, sl].std()))
        rows.append(row)
    return rows


# %% Load every model's rollouts and build the long dataframe
frames = []
for cache_name, model_name in MODELS:
    cache = RolloutHDFCache(cache_name, mode="r")
    results = load_results_from_cache(cache)
    first = cache.get_rollout(*cache.list_rollouts()[0])
    step, seq_length = int(first["step"]), int(first["seq_length"])
    records = build_rollout_records(results, data_module, step=step)
    frame = pd.DataFrame([row for record in records for row in _window_rows(record, step, seq_length)])
    frame["model"] = model_name
    frames.append(frame)
    print(f"{model_name}: {len(records)} rollouts from {cache_name} (step={step}, seq_length={seq_length})")
df = pd.concat(frames, ignore_index=True)
df["abs_elm_peaks_err"] = (df["elm_peaks_gen"] - df["elm_peaks_real"]).abs()

# %% Aggregate per (model, start fraction, depth k): median + IQR over that panel's rollouts
FRACS = sorted(df["start_frac"].unique())
metric_cols = [c for c in df.columns if c not in ("model", "shot", "start_frac", "sample_idx", "k")]
groups = df.groupby(["model", "start_frac", "k"])[metric_cols]
med, q25, q75 = groups.median(), groups.quantile(0.25), groups.quantile(0.75)
# Population per (start_frac, k); identical across models since specs are shared
counts = df[df["model"] == MODELS[0][1]].groupby(["start_frac", "k"]).size()
med.join(counts.rename("n_rollouts"), on=["start_frac", "k"]).to_csv(TABLE_DIR / "rollout_horizon.csv")
print("wrote", TABLE_DIR / "rollout_horizon.csv")


# %% Horizon figures: one panel per start fraction, one line per model
# Per-panel sizes (width_per_panel, height) in inches; exported once per size, same 4:3
# family idea as the other paper exports so fonts scale relative to the figure.
PANEL_SIZES = ((2.6, 3.2), (3.6, 4.4), (5.0, 6.0))


def _horizon_grid(col, title, fname, ylabel, real_col=None):
    """Faceted horizon figure for one metric column.

    Panels = start fractions (so different shot phases are never pooled); line = median
    over that panel's rollouts at depth k, band = IQR over the same population, grey
    step (right axis) = n(k), the number of rollouts still running at depth k. With
    multiple MODELS each panel holds one line per model; `real_col` adds the
    model-independent real reference (black, dashed) from the first model's windows.
    """
    for panel_w, height in PANEL_SIZES:
        fig, axes = plt.subplots(
            1, len(FRACS), sharey=True, figsize=(panel_w * len(FRACS) + 1.2, height), squeeze=False
        )
        for ax, frac in zip(axes[0], FRACS):
            for (cache_name, model_name), color in zip(MODELS, MODEL_COLORS):
                try:
                    m = med.loc[(model_name, frac)][col]
                except KeyError:
                    continue  # this model has no rollouts at this start fraction
                ax.plot(m.index, m.values, color=color, label=model_name)
                ax.fill_between(
                    m.index, q25.loc[(model_name, frac)][col], q75.loc[(model_name, frac)][col],
                    color=color, alpha=0.15, linewidth=0,
                )
            if real_col is not None:
                r = med.loc[(MODELS[0][1], frac)][real_col]
                ax.plot(r.index, r.values, color="black", linestyle="--", linewidth=1.0, label="real")
            ax_n = ax.twinx()
            n = counts.loc[frac]
            ax_n.step(n.index, n.values, where="mid", color="0.65", linewidth=0.8)
            ax_n.set_ylim(bottom=0)
            ax_n.spines["top"].set_visible(False)
            if ax is axes[0][-1]:
                ax_n.set_ylabel("n rollouts at depth $k$", color="0.45", fontsize=8)
            ax_n.tick_params(axis="y", colors="0.45", labelsize=7)
            ax.set_title(f"start at {frac:.0%} of shot", fontsize=9)
            ax.set_xlabel("depth $k$")
            ax.set_facecolor("#F7F7F7")
            ax.grid(color="white", linewidth=1.0)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[0][0].set_ylabel(ylabel)
        axes[0][0].legend(frameon=False, fontsize=7)
        fig.suptitle(f"{title}\nper start fraction; line = median over rollouts, band = IQR", fontsize=10)
        fig.tight_layout()
        w, h = fig.get_size_inches()
        out = PDF_DIR / f"{w:.0f}x{h:.0f}" / f"{fname}.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)
        plt.close(fig)


for ch_name in CHANNEL_NAMES:
    _horizon_grid(f"abs_mean_err/{ch_name}", f"{ch_name}: mean error vs rollout depth",
                  f"abs_mean_err_{ch_name}", r"$|\,\mu_{gen} - \mu_{real}\,|$ (norm. units)")
    _horizon_grid(f"abs_std_err/{ch_name}", f"{ch_name}: spread error vs rollout depth",
                  f"abs_std_err_{ch_name}", r"$|\,\sigma_{gen} - \sigma_{real}\,|$ (norm. units)")
_horizon_grid("label_agreement", "Surrogate mode-label agreement vs rollout depth",
              "label_agreement", "fraction of window where\nFNOLSTM(gen) = FNOLSTM(real)")
_horizon_grid("elm_peaks_gen", "ELM-scale PD peak rate vs rollout depth",
              "elm_peaks", f"peaks per window (prominence {ELM_PROMINENCE})",
              real_col="elm_peaks_real")

# %% Compact horizon table for the main model: rows = (start fraction, selected depths)
MAIN_MODEL = MODELS[0][1]
DEPTHS = [0, 1, 2, 4, 8, 16, 32]
table_cols = ["label_agreement", "abs_mean_err/PD", "abs_std_err/PD", "elm_peaks_gen", "elm_peaks_real"]
rows = []
for frac in FRACS:
    block = med.loc[(MAIN_MODEL, frac)]
    for k in (d for d in DEPTHS if d in block.index):
        rows.append({"start_frac": frac, "k": k, "n": counts.loc[(frac, k)], **block.loc[k, table_cols]})
table = pd.DataFrame(rows).set_index(["start_frac", "k"])
tex_path = TABLE_DIR / f"rollout_horizon_{MODELS[0][0]}.tex"
table.to_latex(tex_path, float_format="%.3f")
print("wrote", tex_path)
print(table)

# %%
