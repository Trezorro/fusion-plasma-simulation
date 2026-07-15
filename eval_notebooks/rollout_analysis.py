"""Horizon-resolved quantitative analysis of autoregressive rollouts.

The question: how does generation quality degrade as the rollout runs further from its
real starting history? Everything is aggregated per window index k (window k covers
generated samples [k*step, k*step + seq_length), i.e. k is the autoregressive depth).

Metrics per (rollout, k), aggregated as median + IQR over rollouts:
  * per-channel |mean(gen) - mean(real)| and |std(gen) - std(real)| (normalized space)
  * surrogate label agreement: fraction of samples where FNOLSTM(gen) == FNOLSTM(real)
  * ELM-scale PD peak count on gen vs real (scipy find_peaks, prominence =
    evaluation.peaks.elm_pd_prominence on the [0,1]-normalized signal, matching the
    'PD large peaks' channel of the window PeakMetric; see docs/evaluation-metrics.md)

Inputs:  CACHE_NAME = rollout cache ({test_cache_name}_rollout.h5) in output/test_cache/,
         written by src/rollout.py. Real traces re-derived from the parquet.
Outputs: output/tables/rollout_horizon_{CACHE_NAME}.csv (per-k aggregates)
         output/tables/rollout_horizon_{CACHE_NAME}.tex (compact horizon table)
         output/pdfplots/rollout_analysis/{metric}.pdf curves
         (override dirs with ROLLOUT_TABLE_DIR / ROLLOUT_PDF_DIR env vars for test runs).

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

CACHE_NAME = os.environ.get("ROLLOUT_CACHE_NAME", "R-NormalMidAttSig03_anim_rollout")

# %% Fetch the cache from Snellius if missing (set AUTO_FETCH=False to only print the command)
AUTO_FETCH = True
SNELLIUS_CACHE = "snellius:/scratch-shared/mtresoor/final_cache"

if not (Path("output/test_cache") / f"{CACHE_NAME}.h5").exists():
    cmd = f"rsync -vz {SNELLIUS_CACHE}/{CACHE_NAME}.h5 output/test_cache/"
    print(f"missing cache: {CACHE_NAME}\n{cmd}")
    if AUTO_FETCH:
        import subprocess
        subprocess.run(cmd, shell=True, check=True)

# %% Load rollouts + real context
cache = RolloutHDFCache(CACHE_NAME, mode="r")
results = load_results_from_cache(cache)
first = cache.get_rollout(*cache.list_rollouts()[0])
STEP = int(first["step"])
SEQ_LENGTH = int(first["seq_length"])
records = build_rollout_records(results, data_module, step=STEP)
print(f"{len(records)} rollouts from {CACHE_NAME} (step={STEP}, seq_length={SEQ_LENGTH})")


# %% Per-(rollout, window) metric rows
def _window_rows(record):
    """One row of metrics per autoregressive window index k of one rollout."""
    hl = int(record["history_length"])
    gen = record["generated_x"]
    real = record["real_x"][:, hl:]
    lab_gen = record["surr_labels_gen"][hl:]
    lab_real = record["surr_labels_real"][hl:]
    rows = []
    for k in range(int(record["n_windows"])):
        sl = slice(k * STEP, k * STEP + SEQ_LENGTH)
        row = {
            "shot": record["shot_number"],
            "start_frac": record["start_frac"],
            "k": k,
            "label_agreement": float((lab_gen[sl] == lab_real[sl]).mean()),
            "elm_peaks_gen": len(find_peaks(gen[PD_INDEX, sl], prominence=ELM_PROMINENCE)[0]),
            "elm_peaks_real": len(find_peaks(real[PD_INDEX, sl], prominence=ELM_PROMINENCE)[0]),
        }
        for ch, name in enumerate(CHANNEL_NAMES):
            row[f"mean_gap/{name}"] = float(abs(gen[ch, sl].mean() - real[ch, sl].mean()))
            row[f"std_gap/{name}"] = float(abs(gen[ch, sl].std() - real[ch, sl].std()))
        rows.append(row)
    return rows


df = pd.DataFrame([row for record in records for row in _window_rows(record)])
df["elm_peaks_gap"] = (df["elm_peaks_gen"] - df["elm_peaks_real"]).abs()
print(df.describe())

# %% Aggregate over rollouts per window index k (median + IQR)
metric_cols = [c for c in df.columns if c not in ("shot", "start_frac", "k")]
agg = df.groupby("k")[metric_cols].median()
q25 = df.groupby("k")[metric_cols].quantile(0.25)
q75 = df.groupby("k")[metric_cols].quantile(0.75)
counts = df.groupby("k").size()
agg["n_rollouts"] = counts
agg.to_csv(TABLE_DIR / f"rollout_horizon_{CACHE_NAME}.csv")
print("wrote", TABLE_DIR / f"rollout_horizon_{CACHE_NAME}.csv")


# %% Horizon curves
def _horizon_plot(cols, title, fname, ylabel):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for col, color in zip(cols, ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]):
        ax.plot(agg.index, agg[col], color=color, label=col)
        ax.fill_between(agg.index, q25[col], q75[col], color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("Autoregressive window index $k$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.set_facecolor("#F7F7F7")
    ax.grid(color="white", linewidth=1.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(PDF_DIR / fname, bbox_inches="tight")
    print("wrote", PDF_DIR / fname)
    plt.close(fig)


_horizon_plot([f"mean_gap/{n}" for n in CHANNEL_NAMES],
              "Moment error vs rollout depth (median + IQR)", "mean_gap.pdf", "|mean gap|")
_horizon_plot([f"std_gap/{n}" for n in CHANNEL_NAMES],
              "Std error vs rollout depth (median + IQR)", "std_gap.pdf", "|std gap|")
_horizon_plot(["label_agreement"],
              "Surrogate mode-label agreement vs rollout depth", "label_agreement.pdf", "agreement")
_horizon_plot(["elm_peaks_gen", "elm_peaks_real"],
              "ELM-scale PD peaks per window vs rollout depth", "elm_peaks.pdf", "peaks / window")

# %% Compact horizon table (selected depths) -> LaTeX
DEPTHS = [k for k in (0, 1, 2, 4, 8, 16, 32) if k in agg.index]
table = agg.loc[DEPTHS, ["label_agreement", "mean_gap/PD", "std_gap/PD",
                          "elm_peaks_gen", "elm_peaks_real", "n_rollouts"]]
tex_path = TABLE_DIR / f"rollout_horizon_{CACHE_NAME}.tex"
table.to_latex(tex_path, float_format="%.3f")
print("wrote", tex_path)
print(table)

# %%
