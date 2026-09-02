"""Rebuild the interactive rollout browser HTML locally from a rollout cache.

The same figure is written during the run (output/htmlplots/{run_name}/rollouts.html);
this notebook re-creates it from the cache so shot selection and plot tweaks do not
need a cluster rerun. See src/plotters/rollout_plots.py for the figure itself.

Inputs:  CACHE_NAME rollout cache in output/test_cache/ (autofetched from Snellius below).
Outputs: output/htmlplots/local/rollouts_{CACHE_NAME}.html
         (override the directory with ROLLOUT_HTML_DIR, e.g. output/testplots/...).

Run:     PYTHONPATH=. python eval_notebooks/rollout_browser.py
"""
# %% Imports
import os
import sys
from pathlib import Path
from typing import Type

_REPO_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import plotly.io as pio

from src.config import load_config_from_file
import src.data_loaders
from src.hdf_cache import RolloutHDFCache
from src.rollout import build_rollout_groups, load_results_from_cache
from src.plotters.rollout_plots import rollout_browser_plotly

# %% Config + data
CONFIG_NAME = "plasmaflow"
C = load_config_from_file(CONFIG_NAME, as_omega=True)
DataSetClass: Type[src.data_loaders.FusionShotDataModule] = getattr(src.data_loaders, C.data.Class)
data_module = DataSetClass(**C.data)
data_module.prepare_data()
data_module.setup()
#%% 
# The cache files are written as "<run_name>_rollout.h5"; the suffix is part of the name,
# both locally and on the cluster.
CACHE_NAME = os.environ.get("ROLLOUT_CACHE_NAME", "R-IN-itransformer-noleak-e2_rollout" ) #  or "R-CNb-cfm-noleak-normal-s05-noatt-e2_rollout")
SHOTS = None # list(C.rollout.html_shots) if "rollout" in C else None  # None = every cached rollout
MAX_SAMPLES = int(C.rollout.plot_samples) if "rollout" in C else 2  # samples overlaid per start point


# Fetch the cache from Snellius if missing (set AUTO_FETCH=False to only print the command)
AUTO_FETCH = True
SNELLIUS_CACHE = "snellius:/scratch-shared/mtresoor/final_cache"

if not (Path("output/test_cache") / f"{CACHE_NAME}.h5").exists():
    cmd = f"rsync -vz {SNELLIUS_CACHE}/{CACHE_NAME}.h5 output/test_cache/"
    print(f"missing cache: {CACHE_NAME}\n{cmd}")
    if AUTO_FETCH:
        import subprocess
        subprocess.run(cmd, shell=True, check=True)

# %% Build + write
cache = RolloutHDFCache(CACHE_NAME, mode="r")
# Filtered at load time: with n_samples in the hundreds, the full cache can be far
# bigger than the SHOTS/MAX_SAMPLES subset this notebook actually renders.
results = load_results_from_cache(cache, shots=SHOTS, max_samples=MAX_SAMPLES)
step = cache.get_rollout(*cache.list_rollouts()[0])["step"]
groups = build_rollout_groups(results, data_module, step=step, shots=SHOTS, max_samples=MAX_SAMPLES)
#%%
# Start fractions to keep in the dropdown; None = every cached start point. Matched against
# the cached start_frac with a small tolerance, since it is stored as the requested (pre-clamp)
# fraction. Fewer fractions means a much smaller HTML file.
START_FRACTIONS = [0.05, ]  # e.g. None, or [0.1, 0.5]
HTML_DIR = Path(os.environ.get("ROLLOUT_HTML_DIR", "output/htmlplots/local"))

# Peak overlay. SHOW_PEAKS draws the detected peaks as markers on top of the signals, real and
# generated as separate legend entries so either can be hidden. The thresholds and the filter
# mirror eval_notebooks/rollout_tables.py, so what is drawn here is what the tables counted;
# keep the two in step when tuning. A channel absent from PEAK_PROMINENCE gets no markers.
# SIGNAL_FILTER is the optional pre-detection smoothing (src/signal_filters.py), applied to
# the real and the generated trace alike: None, "gaussian", or {"kind": "gaussian", "sigma": 3}.
SHOW_PEAKS = os.environ.get("SHOW_PEAKS", "1") != "0"
PEAK_PROMINENCE = {"PD": 0.05, "DML": 0.01}
SIGNAL_FILTER = None # {"kind": "gaussian", "sigma": 2}
# The filtered trace and its peak markers share one legend entry per source, so a source can
# be examined without its raw signal moving. None = draw them whenever a filter is active;
# False keeps the markers on the raw trace only, which roughly halves the written HTML.
SHOW_FILTERED = None


if START_FRACTIONS is not None:
    groups = [g for g in groups
              if any(abs(float(g['start_frac']) - f) < 1e-6 for f in START_FRACTIONS)]
print(f"{len(groups)} starting points in the browser ({MAX_SAMPLES} samples each, max)")
fig = rollout_browser_plotly(
    groups, list(data_module.cols.x), list(data_module.cols.get("c", [])),
    show_peaks=SHOW_PEAKS, peak_prominence=PEAK_PROMINENCE, signal_filter=SIGNAL_FILTER,
    show_filtered=SHOW_FILTERED,
)
HTML_DIR.mkdir(parents=True, exist_ok=True)
out = HTML_DIR / f"rollouts_{CACHE_NAME}.html"
pio.write_html(fig, out)
print("wrote", out)

# %%
