"""Horizon-resolved quantitative analysis of autoregressive rollouts (post-hoc driver).

The same figures and tables are produced in-run by run_rollouts when `rollout.analysis`
is enabled; this notebook re-creates them from one or more rollout caches so they can be
tweaked, re-sliced, or overlaid across models without a GPU rerun. All metric math,
aggregation, and figure code lives in src/plotters/rollout_horizon.py (shared with the
in-run path); definitions and the reading guide are in docs/evaluation-metrics.md
section 7 and docs/plots.md.

The short version: every figure tracks one failure mode vs the autoregressive depth k
(windows of own output consumed), faceted into one panel per start fraction because at
equal k different starting points sit in different shot phases and must not be pooled.
Line = median over that panel's rollouts, band = IQR, grey step = n(k). MODELS overlays
one line per model; the per-(model, rollout, k) dataframe `df` keeps model / shot /
start_frac / sample_idx columns for further re-slicing.

Inputs:  rollout caches ({test_cache_name}_rollout.h5) in output/test_cache/, written by
         src/rollout.py. Real traces re-derived from the parquet.
Outputs: output/tables/rollout_horizon.{csv,tex}
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

import matplotlib as mpl

from src.config import load_config_from_file
import src.data_loaders
from src.hdf_cache import RolloutHDFCache
from src.rollout import build_rollout_records, load_results_from_cache
from src.plotters.rollout_horizon import export_horizon_analysis

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

# %% Models: (rollout_cache_name, print_name); one line per model in every panel
MODELS = [
    (os.environ.get("ROLLOUT_CACHE_NAME", "R-CNb-cfm-noleak-normal-s05-noatt-e2"), "CFM (ours)"),
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

# %% Load every model's rollouts + real context
model_records = []
for cache_name, model_name in MODELS:
    cache = RolloutHDFCache(cache_name, mode="r")
    results = load_results_from_cache(cache)
    step = int(cache.get_rollout(*cache.list_rollouts()[0])["step"])
    records = build_rollout_records(results, data_module, step=step)
    model_records.append((model_name, records))
    print(f"{model_name}: {len(records)} rollouts from {cache_name} (step={step})")

# %% Aggregate, plot, export (shared with the in-run path)
df = export_horizon_analysis(
    model_records, CHANNEL_NAMES, PD_INDEX, ELM_PROMINENCE,
    pdf_dir=PDF_DIR, table_dir=TABLE_DIR,
)
print(df.groupby(["model", "start_frac"]).size().rename("windows"))

# %%
