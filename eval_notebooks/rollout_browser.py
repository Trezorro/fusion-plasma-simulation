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
from src.rollout import build_rollout_records, load_results_from_cache
from src.plotters.rollout_plots import rollout_browser_plotly

# %% Config + data
CONFIG_NAME = "plasmaflow"
C = load_config_from_file(CONFIG_NAME, as_omega=True)
DataSetClass: Type[src.data_loaders.FusionShotDataModule] = getattr(src.data_loaders, C.data.Class)
data_module = DataSetClass(**C.data)
data_module.prepare_data()
data_module.setup()

CACHE_NAME = os.environ.get("ROLLOUT_CACHE_NAME", "R-NormalMidAttSig03_anim_rollout")
SHOTS = list(C.rollout.html_shots) if "rollout" in C else None  # None = every cached rollout
HTML_DIR = Path(os.environ.get("ROLLOUT_HTML_DIR", "output/htmlplots/local"))

# %% Fetch the cache from Snellius if missing (set AUTO_FETCH=False to only print the command)
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
results = load_results_from_cache(cache)
step = cache.get_rollout(*cache.list_rollouts()[0])["step"]
records = build_rollout_records(results, data_module, step=step, shots=SHOTS)
print(f"{len(records)} rollouts in the browser")
fig = rollout_browser_plotly(records, list(data_module.cols.x), list(data_module.cols.get("c", [])))
HTML_DIR.mkdir(parents=True, exist_ok=True)
out = HTML_DIR / f"rollouts_{CACHE_NAME}.html"
pio.write_html(fig, out)
print("wrote", out)

# %%
