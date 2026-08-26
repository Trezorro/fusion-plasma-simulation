# %% [markdown]
# # Rollout cache explorer
# 
# Onboarding notebook for analysing the autoregressive rollout caches. Everything the
# model generated during a rollout run lives in one HDF5 file per run; this notebook loads
# one, shows what is inside, and gives you working starting points for plotting and peak
# analysis. All the heavy lifting (reading the cache, re-deriving the real traces, building
# the interactive browser and the horizon figures) is imported from the codebase, so this
# file is mostly glue plus a clear space at the bottom for your own work.
# 
# ## What a rollout is
# 
# The model generates one 256-sample future window from a real history window plus the real
# control covariates. A *rollout* chains that: start at some fraction of a shot, generate a
# window, feed the generated window back as the next history, and repeat to the end of the
# shot. The control covariates and the physical time axis always come from the real shot;
# only the observable history is generated. So a rollout is the model running free on its
# own output, and the question is how fast and in what way it drifts from reality.
# 
# Each rollout is done for several *start fractions* (e.g. 5%, 10%, 25%, 50%, 75% of the
# shot) and, for the stochastic model, several *samples* per start point (the flow model is
# stochastic, so the same start gives different rollouts; the deterministic baselines get one).
# 
# ## What is in the cache
# 
# One file per run, named `{run_name}_rollout.h5`, group layout `{shot}/{start_idx}/{sample_idx}`.
# Each leaf holds three arrays:
# 
# - `generated_x` `(channels, T)`, float32, **normalized [0,1]**. The generated observables.
# - `surr_labels_gen` `(history_length + T,)`, int16. Surrogate mode labels (FNOLSTM classifier)
#   on the generated trace, over history + rollout.
# - `surr_labels_real` same shape, the surrogate labels on the real trace over the same span.
# 
# Plus per-leaf attrs (`start_frac`, `start_i`, `t_start`, `t_end`, `n_windows`, `seq_length`,
# `history_length`, `step`) and root attrs (`start_fractions`, `n_samples`, `cols_x`, `cols_c`,
# `run_name`, `config_json`). `config_json` is the full run config, so which columns and
# settings produced a cache can be recovered here without wandb access.
# 
# Two things are deliberately **not** stored, and are re-derived from the parquet when needed:
# the real observables/controls, and the true (non-surrogate) labels. The cache only holds
# what the model produced, everything else is a positional slice of the shot dataframe.
# 
# **Which columns.** Different models used different control covariates (leak vs noleak), so
# this notebook takes the column lists from the cache itself (`cols_x` / `cols_c`, or
# `config_json`), not from the local `plasmaflow.yaml`, and builds the data module to match.
# Older caches made before config-stamping fall back to wandb, then to the local yaml.
# 
# **Label convention.** The surrogate labels are the classifier argmax and are **unshifted**:
# `0 = L, 1 = D, 2 = H`, never Unknown. This is NOT the `+1`-shifted `LHD_label` convention
# used elsewhere in the code. Index colour/name lists directly by these values.
# 
# **Normalized space.** `generated_x` and the re-derived real observables are both in the
# min-max normalized `[0,1]` space (min/max from the train split). Peak prominences in the
# config are set in this same space. Call `data_module.denormalize(...)` for physical units.
# 
# ## The API you will use
# 
# | Import | What it does |
# |---|---|
# | `RolloutHDFCache(name, mode="r")` | Open a cache. `.get_root_attrs()`, `.list_rollouts()`, `.get_rollout(shot, start, sample)`. |
# | `load_results_from_cache(cache, shots=, max_samples=)` | Read rollouts into `RolloutResult` objects, filtered cheaply before any array read. |
# | `build_rollout_records(results, dm, step, ...)` | Flat: one dict per rollout, with the real traces + timeline attached. For stats and single-rollout figures. |
# | `build_rollout_groups(results, dm, step, ...)` | Grouped by (shot, start point): overlays the stochastic samples. For the interactive browser. |
# | `rollout_browser_plotly(groups, x_names, c_names)` | The interactive dropdown browser figure. |
# | `export_horizon_analysis(model_records, ...)` | Error-vs-depth figures and tables. |
# 
# Sibling notebooks do the production versions of each output: `rollout_browser.py`
# (browser HTML), `paper_rollout.py` (per-rollout PDFs), `rollout_analysis.py` (horizon
# figures/tables). This notebook is for exploring, they are for batch export.

# %% [markdown]
# ## Setup: imports and repo root

# %%
import os
import sys
import json
import subprocess
from pathlib import Path

# Put the repo root on sys.path so `import src` works no matter where the kernel started.
_REPO_ROOT = Path.cwd()
while not (_REPO_ROOT / "src").is_dir() and _REPO_ROOT != _REPO_ROOT.parent:
    _REPO_ROOT = _REPO_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)  # so relative paths (output/, data/) resolve from the repo root
print("repo root:", _REPO_ROOT)

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import plotly.io as pio
from omegaconf import OmegaConf
from scipy.signal import find_peaks

from src.config import load_config_from_file, find_wandb_run
import src.data_loaders
from src.hdf_cache import RolloutHDFCache, get_cache_dir
from src.rollout import (
    load_results_from_cache,
    build_rollout_records,
    build_rollout_groups,
)
from src.plotters.rollout_plots import rollout_browser_plotly
from src.plotters.rollout_horizon import export_horizon_analysis
from src.metrics.metrics import batch_get_peakprops

pio.renderers.default = "notebook"  # inline plotly in the notebook

# %% [markdown]
# ## Constants
# 
# Set these once. `CACHE_RUN` is the run name (its cache is `{CACHE_RUN}_rollout.h5`), a tag,
# or a full cache filename with or without `.h5`. The resolver below adds the `_rollout`
# suffix if missing and checks the file really is a rollout cache before using it.

# %%
# --- which cache -------------------------------------------------------------
CACHE_RUN = "R-CNb-cfm-noleak-normal-s05-noatt-e2_rollout"   # headline CFM reeval (5 samples, quick pass); cache is {run}_rollout.h5
# The full 30-sample headline pass, once it finishes:
# CACHE_RUN = "R-CN-cfm-noleak-normal-s05-e2"
# Offline alternative bundled in the repo (2 shots, 3 samples), if you have no cluster access:
# CACHE_RUN = "rollout_multisample_debug"

# --- what to look at ---------------------------------------------------------
SHOTS = None            # None = every shot in the cache, or a list like [57013, 61237, 64770]
MAX_SAMPLES = 10        # cap stochastic samples per start point (keep plots light)
MAX_SHOTS_IN_BROWSER = 10   # the interactive browser gets heavy past ~10 shots x 10 samples
START_FRACTION = None   # None = all start fractions, or a float like 0.50 to filter

# --- config / output ---------------------------------------------------------
CONFIG_NAME = "plasmaflow"          # only a fallback; columns come from the cache
OUTPUT_DIR = Path("output/coauthor")   # where this notebook writes html/pdf
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("output goes to:", OUTPUT_DIR.resolve())

# Snellius fetch (only used if the cache is missing locally AND you have access)
SNELLIUS_REMOTE = "snellius:/scratch-shared/mtresoor/final_cache"

# %% [markdown]
# ## Selecting a cache, and fetching it from Snellius if you can
# 
# `resolve_cache_name` turns a run name / tag / filename into the cache stem: it appends the
# `_rollout` suffix if you left it off (that suffix is always present on these caches) and
# only accepts a local file that actually has the rollout layout, so pointing it at a window
# (test) cache by mistake will not silently match. `ensure_cache` then makes sure the file is
# on disk, rsyncing from Snellius only when an SSH connection actually works, otherwise
# printing the exact command. You most likely will not have Snellius access, so expect to be
# handed the `.h5` files directly and drop them in `output/test_cache/`.

# %%
def looks_like_rollout_cache(path: Path) -> bool:
    """True if the HDF5 file has the rollout layout (root start_fractions or shot/start/sample)."""
    try:
        with h5py.File(path, "r") as f:
            if "start_fractions" in f.attrs:   # every rollout cache stamps this at the root
                return True
            for shot in f.keys():              # structural fallback: 3 levels deep to generated_x
                for start in f[shot].keys():
                    node = f[shot][start]
                    if not isinstance(node, h5py.Group) or "generated_x" in node:
                        return False            # 2-level {shot}/{start} -> dataset is the window cache
                    for samp in node.keys():
                        return "generated_x" in node[samp]
    except Exception:
        return False
    return False


def resolve_cache_name(x: str) -> str:
    """Run name / tag / filename -> rollout-cache stem (no .h5, _rollout suffix ensured)."""
    stem = x[:-3] if x.endswith(".h5") else x
    cache_dir = get_cache_dir()
    candidates = [stem] + ([] if stem.endswith("_rollout") else [f"{stem}_rollout"])
    for cand in candidates:
        p = cache_dir / f"{cand}.h5"
        if p.exists():
            if looks_like_rollout_cache(p):
                return cand
            print(f"note: {p.name} exists but is not a rollout cache (looks like a window cache); skipping it.")
    return stem if stem.endswith("_rollout") else f"{stem}_rollout"  # for fetching


def _snellius_reachable(timeout=6) -> bool:
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}", "snellius", "true"],
            capture_output=True, timeout=timeout + 4,
        )
        return r.returncode == 0
    except Exception:
        return False


def ensure_cache(cache_stem: str) -> Path:
    """Return the local path to {cache_stem}.h5, fetching from Snellius only if possible."""
    path = get_cache_dir() / f"{cache_stem}.h5"
    if path.exists():
        print("cache present:", path.resolve())
        return path
    cmd = f"rsync -vz {SNELLIUS_REMOTE}/{cache_stem}.h5 {get_cache_dir()}/"
    if _snellius_reachable():
        print("fetching from Snellius:\n ", cmd)
        subprocess.run(cmd, shell=True, check=True)
    else:
        print(
            f"cache '{cache_stem}.h5' not found locally and Snellius is not reachable.\n"
            f"Get the file from someone with access (or run, if you have it):\n  {cmd}\n"
            f"then drop it in {get_cache_dir().resolve()}/"
        )
    return path


CACHE_NAME = resolve_cache_name(CACHE_RUN)
CACHE_PATH = ensure_cache(CACHE_NAME)
print("using cache:", CACHE_NAME)

# %% [markdown]
# ## Which config produced this cache
# 
# The columns and settings come from the cache's own stamped config (`config_json` root attr),
# so this notebook matches the model that produced the cache regardless of your local
# `plasmaflow.yaml`. If the cache predates config-stamping it falls back to wandb (needs
# access), then to the local yaml with a warning. The table shows the important settings; you
# also most likely want to eyeball `cols.c` (the control covariates) since that is what differs
# between the leak and noleak models.

# %%
def _get(d, dotted, default="-"):
    for k in dotted.split("."):
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def load_cache_config(cache, run_name=None):
    """Return (config_dict, source). Prefer the cache's stamped config, else wandb, else local yaml."""
    root = cache.get_root_attrs()
    stamped = root.get("config_json")
    if stamped:
        return json.loads(stamped), "cache metadata (config_json)"
    if run_name:
        try:
            run = find_wandb_run(run_name)
            if run is not None:
                return dict(run.config), "wandb run config"
        except Exception as e:
            print("wandb lookup failed:", e)
    print("WARNING: no config stamped on the cache and no wandb; falling back to configs/plasmaflow.yaml, "
          "which may not match the model that produced this cache (check cols.c below).")
    return OmegaConf.to_container(load_config_from_file(CONFIG_NAME, as_omega=True), resolve=False), "local yaml (may mismatch!)"


cache = RolloutHDFCache(CACHE_NAME, mode="r")
root = cache.get_root_attrs()
RUN_NAME = str(root.get("run_name") or CACHE_RUN)
cache_cfg, CFG_SOURCE = load_cache_config(cache, RUN_NAME)

# Columns: prefer the explicit root attrs, then the stamped config. Cast to plain str.
CHANNEL_NAMES = [str(c) for c in (list(root.get("cols_x", [])) or _get(cache_cfg, "data.cols.x", []))]
C_NAMES = [str(c) for c in (list(root.get("cols_c", [])) or _get(cache_cfg, "data.cols.c", []) or [])]
PD_INDEX = CHANNEL_NAMES.index("PD")
peaks = _get(cache_cfg, "evaluation.peaks", {})
PROMINENCE = peaks.get("prominence", 0.001) if isinstance(peaks, dict) else 0.001
ELM_PD_PROMINENCE = peaks.get("elm_pd_prominence", 0.1) if isinstance(peaks, dict) else 0.1

mp = _get(cache_cfg, "model.params.model_params", {})
config_table = pd.DataFrame([
    ("config source", CFG_SOURCE),
    ("run_name", RUN_NAME),
    ("is_reeval", cache_cfg.get("is_reeval", "-")),
    ("base_run", cache_cfg.get("base_run", "-")),
    ("model.Class", _get(cache_cfg, "model.Class")),
    ("prior", _get(cache_cfg, "model.params.prior")),
    ("prior_sigma", _get(cache_cfg, "model.params.prior_sigma")),
    ("mid_attn", mp.get("mid_attn", "-") if isinstance(mp, dict) else "-"),
    ("cols.x (observables)", CHANNEL_NAMES),
    ("cols.c (controls)", C_NAMES),
    ("seq_length", _get(cache_cfg, "data.seq_length")),
    ("history_length", _get(cache_cfg, "data.history_length")),
    ("data.file", _get(cache_cfg, "data.file")),
    ("eval n_steps", _get(cache_cfg, "evaluation.n_steps")),
    ("flow_rho", _get(cache_cfg, "evaluation.flow_rho")),
    ("peaks", peaks),
    ("start_fractions", list(root.get("start_fractions", []))),
    ("n_samples (cache)", root.get("n_samples", "-")),
], columns=["setting", "value"])
print(f"peak prominence (all channels)={PROMINENCE}, ELM-scale PD prominence={ELM_PD_PROMINENCE}")
config_table

# %% [markdown]
# ## Data module (built to match the cache's config)
# 
# The data module re-derives the real observables/controls and the physical time axis. It is
# built from the cache's own `data` config (columns, seq_length, history_length, the
# train/val/test shot lists, everything), not the local yaml, so real traces line up with
# what the model actually saw and the train-split normalization matches. This matters beyond
# just columns: `build_rollout_records`/`build_rollout_groups` slice the real context using
# `data_module.history_length`, and normalization min/max are derived from `train_shots`, so a
# mismatch there would silently misalign or rescale things rather than crash. `prepare_data` +
# `setup` load the parquet and compute that normalization; this is the slow step, run it once.

# %%
C = load_config_from_file(CONFIG_NAME, as_omega=True)  # base; only fills gaps in the fallback case
cache_data_cfg = cache_cfg.get("data")
if isinstance(cache_data_cfg, dict) and cache_data_cfg:
    # Whole data block from the cache's own config, not just columns: seq_length,
    # history_length, and the shot-split lists all affect how real_x/real_c/times get
    # re-derived and normalized.
    C.data = OmegaConf.merge(C.data, OmegaConf.create(cache_data_cfg))
else:
    # Fallback (no cache/wandb config, see the warning above): at least get the columns right.
    C.data.cols.x = list(CHANNEL_NAMES)
    C.data.cols.c = list(C_NAMES)

local_parquet = Path(C.data.dir) / C.data.file
if not local_parquet.exists():
    print(f"WARNING: the cache's dataset {C.data.file} is not present locally at {local_parquet}. "
          "Real traces are re-derived from the local parquet; get the right file before trusting results.")

DataModuleClass = getattr(src.data_loaders, C.data.Class)
data_module = DataModuleClass(**C.data)
data_module.prepare_data()
data_module.setup()
print("data module columns  x:", list(data_module.cols.x), "  c:", list(data_module.cols.get("c", [])))
print("seq_length:", data_module.seq_length, "| history_length:", data_module.history_length,
      "| data.file:", C.data.file)

# %% [markdown]
# ## Overview of the cache
# 
# `list_rollouts()` returns every `(shot, start_idx, sample_idx)` triple without reading any
# array, so it is cheap even for a big cache. Below: a table of what is available per (shot,
# start point).

# %%
keys = cache.list_rollouts()  # sorted (shot, start_idx, sample_idx)
print(f"{len(keys)} rollouts across {len({k[0] for k in keys})} shots")

rows, seen = [], set()
for shot, start_idx, sample_idx in keys:
    if (shot, start_idx) in seen:
        continue
    seen.add((shot, start_idx))
    a = cache.get_rollout(shot, start_idx, 0)  # attrs come along with the arrays
    n_samp = sum(1 for k in keys if k[0] == shot and k[1] == start_idx)
    rows.append({
        "shot": shot, "start_idx": start_idx, "n_samples": n_samp,
        "start_frac": round(float(a["start_frac"]), 3), "n_windows": int(a["n_windows"]),
        "t_start": round(float(a["t_start"]), 3), "t_end": round(float(a["t_end"]), 3),
        "gen_len_T": a["generated_x"].shape[-1],
    })
overview = pd.DataFrame(rows).sort_values(["shot", "start_frac"]).reset_index(drop=True)
overview

# %% [markdown]
# ## Inspecting one rollout
# 
# `get_rollout` returns the arrays plus the attrs as a plain dict. Note the label arrays are
# `history_length + T` long (they cover the real history window too), while `generated_x` is
# only the generated part `T`.

# %%
# Pick the first rollout in the cache (change these to target a specific one).
SHOT0, START0, SAMPLE0 = keys[0]
r = cache.get_rollout(SHOT0, START0, SAMPLE0)
print(f"shot {SHOT0}, start_idx {START0}, sample {SAMPLE0}")
print("generated_x:    ", r["generated_x"].shape, r["generated_x"].dtype, "(normalized [0,1])")
print("surr_labels_gen:", r["surr_labels_gen"].shape, "unique:", np.unique(r["surr_labels_gen"]))
print("surr_labels_real:", r["surr_labels_real"].shape, "unique:", np.unique(r["surr_labels_real"]))
print("attrs:", {k: v for k, v in r.items() if k not in ("generated_x", "surr_labels_gen", "surr_labels_real")})

# %% [markdown]
# ## Attaching the real context: records
# 
# `load_results_from_cache` reads the rollouts (filtered by `shots`/`max_samples` before any
# array is touched), and `build_rollout_records` attaches, per rollout, the real observables
# (`real_x`), the controls (`real_c`) and the physical timeline (`times`) as positional slices
# of the shot dataframe. The timeline spans history + rollout; `generated_x` aligns with
# `times[history_length:]`.
# 
# `build_rollout_records` gives you a flat list (one dict per rollout), which is what the
# single-rollout figures and the peak analysis below use. `build_rollout_groups` instead
# groups the samples per start point, which is what the interactive browser wants.

# %%
results = load_results_from_cache(cache, shots=SHOTS, max_samples=MAX_SAMPLES)
step = int(cache.get_rollout(*keys[0])["step"])
records = build_rollout_records(results, data_module, step=step, shots=SHOTS)
if START_FRACTION is not None:
    records = [rec for rec in records if abs(rec["start_frac"] - START_FRACTION) < 1e-6]
print(f"{len(records)} rollout records (step={step})")
rec = records[0]
print("record keys:", sorted(rec.keys()))
print("real_x:", rec["real_x"].shape, "| generated_x:", rec["generated_x"].shape,
      "| times:", rec["times"].shape, f"[{rec['times'][0]:.3f}..{rec['times'][-1]:.3f}]s")

# %% [markdown]
# ## Plotting one rollout inline
# 
# A compact matplotlib view of a single rollout: every observable channel with the real
# history (grey), the real future (black) and the generated trace (orange), the controls,
# and the two surrogate mode-label bars (generated vs real). The vertical line is the rollout
# start; dotted lines are the chained window boundaries.
# 
# For the polished paper PDFs use `eval_notebooks/paper_rollout.py`; this is the quick look.

# %%
MODE_COLORS = ["lightskyblue", "orange", "red"]  # unshifted 0=L, 1=D, 2=H
MODE_NAMES = ["L", "D", "H"]


def plot_rollout_inline(record, figsize=(11, 9)):
    times = record["times"]
    hl = int(record["history_length"])
    t_start = float(record["t_start"])
    gen_times = times[hl:]
    step = int(record["step"])
    boundaries = [times[hl + b] for b in range(step, record["generated_x"].shape[-1], step)]
    n_x = len(CHANNEL_NAMES)

    fig, axes = plt.subplots(n_x + 3, 1, figsize=figsize, sharex=True,
                             gridspec_kw={"height_ratios": [1] * n_x + [0.6, 0.25, 0.25]})
    for ch in range(n_x):
        ax = axes[ch]
        ax.plot(times[:hl], record["real_x"][ch, :hl], color="0.55", lw=0.9, label="real history")
        ax.plot(gen_times, record["real_x"][ch, hl:], color="black", lw=0.8, label="real")
        ax.plot(gen_times, record["generated_x"][ch], color="#D55E00", lw=0.8, label="generated")
        for b in boundaries:
            ax.axvline(b, color="0.7", lw=0.4, ls=":")
        ax.axvline(t_start, color="0.25", lw=0.8)
        ax.set_ylabel(CHANNEL_NAMES[ch], rotation=0, ha="right", va="center")
        if ch == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=3)

    ax_c = axes[n_x]
    for ci, name in enumerate(C_NAMES):
        ax_c.plot(times, record["real_c"][ci], lw=0.9, label=name)
    ax_c.axvline(t_start, color="0.25", lw=0.8)
    ax_c.set_ylabel("C", rotation=0, ha="right", va="center")
    if C_NAMES:
        ax_c.legend(loc="upper right", fontsize=6, ncol=len(C_NAMES))

    for ax, labels, name in ((axes[n_x + 1], record["surr_labels_gen"], "gen"),
                             (axes[n_x + 2], record["surr_labels_real"], "real")):
        # labels align 1:1 with times; colour each contiguous mode run
        vals = np.asarray(labels)
        change = np.flatnonzero(np.diff(vals)) + 1
        bounds = [0, *change, len(vals)]
        for s, e in zip(bounds[:-1], bounds[1:]):
            ax.axvspan(times[s], times[min(e, len(times) - 1)], color=MODE_COLORS[int(vals[s])], lw=0)
        ax.axvline(t_start, color="0.25", lw=0.8)
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
    axes[-1].set_xlabel("Shot time (s)")
    fig.suptitle(f"Shot {record['shot_number']}  start {record['start_frac']:.0%}  "
                 f"t0={t_start:.2f}s  {record['n_windows']} windows  sample {record['sample_idx']}")
    fig.align_ylabels(axes)
    plt.show()
    return fig


_ = plot_rollout_inline(records[0])

# %% [markdown]
# ## The interactive browser: many rollouts, one figure
# 
# `rollout_browser_plotly` builds one plotly figure with a dropdown, one entry per (shot,
# start point), overlaying the stochastic samples. Keep it to about 10 shots and 10 samples;
# past that the HTML gets heavy and slow to open. The `START_FRACTION` / `SHOTS` / `MAX_SAMPLES`
# constants at the top control the subset. Written to an HTML file you can open in a browser,
# and also shown inline.

# %%
browser_shots = SHOTS
if browser_shots is None:
    browser_shots = sorted({k[0] for k in keys})[:MAX_SHOTS_IN_BROWSER]  # cap for a responsive figure
elif len(browser_shots) > MAX_SHOTS_IN_BROWSER:
    browser_shots = list(browser_shots)[:MAX_SHOTS_IN_BROWSER]

browser_results = load_results_from_cache(cache, shots=browser_shots, max_samples=MAX_SAMPLES)
groups = build_rollout_groups(browser_results, data_module, step=step,
                              shots=browser_shots, max_samples=MAX_SAMPLES)
if START_FRACTION is not None:
    groups = [g for g in groups if abs(g["start_frac"] - START_FRACTION) < 1e-6]
print(f"{len(groups)} starting points in the browser "
      f"({len(browser_shots)} shots, <= {MAX_SAMPLES} samples each)")

fig = rollout_browser_plotly(groups, CHANNEL_NAMES, C_NAMES)
out_html = OUTPUT_DIR / f"rollouts_{CACHE_NAME}.html"
print("writing browser HTML to:", out_html.resolve())
pio.write_html(fig, out_html)
print("wrote", out_html)
fig  # inline

# %% [markdown]
# ## One shot and start fraction to PDF / inline image
# 
# For a single starting point (optionally a single sample), export the matplotlib figure to
# PDF and show a raster preview inline. This is the per-rollout artifact; the batch version
# over a whole cache is `eval_notebooks/paper_rollout.py`.

# %%
from IPython.display import Image, display

TARGET_SHOT = overview.iloc[0]["shot"]        # change to any shot in the overview table
TARGET_FRAC = overview.iloc[0]["start_frac"]  # change to any start fraction for that shot
TARGET_SAMPLE = 0

match = [
    rec for rec in build_rollout_records(
        load_results_from_cache(cache, shots=[int(TARGET_SHOT)]), data_module, step=step)
    if abs(rec["start_frac"] - TARGET_FRAC) < 1e-6 and rec["sample_idx"] == TARGET_SAMPLE
]
if not match:
    print("no rollout for", TARGET_SHOT, TARGET_FRAC, "sample", TARGET_SAMPLE)
else:
    fig = plot_rollout_inline(match[0])
    pdf_path = OUTPUT_DIR / f"rollout_{TARGET_SHOT}_{TARGET_FRAC:.2f}_s{TARGET_SAMPLE}.pdf"
    jpg_path = pdf_path.with_suffix(".jpg")
    print("writing PDF to:", pdf_path.resolve())
    print("writing JPEG to:", jpg_path.resolve())
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(jpg_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", pdf_path, "and", jpg_path)
    display(Image(filename=str(jpg_path)))

# %% [markdown]
# ## Peak analysis: ELM timestamps and frequency
# 
# The point of the rollouts is whether the model reproduces the timing statistics of ELMs
# (stochastic H-alpha bursts on the PD channel), not their exact positions. `peak_times`
# returns the timestamps (in seconds) of peaks above a prominence on any channel, for both the
# generated and the real trace over the rollout span. Prominence is in the normalized `[0,1]`
# space, so `ELM_PD_PROMINENCE` (the cached model's ELM-scale threshold on PD) is the natural
# choice for ELM peaks. From the timestamps you get counts, inter-peak intervals, and the ELM
# rate to compare against ground truth.
# 
# For the full peak *properties* (height, width, prominence, base, energy) the codebase's
# `batch_get_peakprops` gives the same `PeakProps` objects the thesis metrics use; a hook for
# that is at the very bottom (the "full" peak-marker overlay).

# %%
def peak_times(record, channel=None, prominence=None, which="both"):
    """Timestamps (s) of peaks above `prominence` on one channel, over the rollout span.

    Prominence is in normalized [0,1] units (same space as generated_x). Returns
    (gen_times, real_times) arrays of seconds, or one of them if which is 'gen'/'real'.
    Peaks are found on the generated part only (times[history_length:]).
    """
    channel = PD_INDEX if channel is None else channel
    prominence = ELM_PD_PROMINENCE if prominence is None else prominence
    hl = int(record["history_length"])
    gen_times = record["times"][hl:]
    out = {}
    if which in ("gen", "both"):
        idx, _ = find_peaks(record["generated_x"][channel], prominence=prominence)
        out["gen"] = gen_times[idx]
    if which in ("real", "both"):
        idx, _ = find_peaks(record["real_x"][channel, hl:], prominence=prominence)
        out["real"] = gen_times[idx]
    return (out["gen"], out["real"]) if which == "both" else out[which]


# Example: pool ELM inter-peak intervals over the loaded records, generated vs real.
gen_intervals, real_intervals = [], []
gen_counts, real_counts = [], []
for rec in records:
    gt, rt = peak_times(rec, channel=PD_INDEX, prominence=ELM_PD_PROMINENCE)
    gen_counts.append(len(gt))
    real_counts.append(len(rt))
    gen_intervals.extend(np.diff(gt))
    real_intervals.extend(np.diff(rt))

print(f"PD ELM peaks/rollout: generated median {np.median(gen_counts):.0f}, "
      f"real median {np.median(real_counts):.0f}")

fig, (axc, axi) = plt.subplots(1, 2, figsize=(11, 3.4))
bins = np.arange(0, max(gen_counts + real_counts) + 2) - 0.5
axc.hist(real_counts, bins=bins, alpha=0.6, label="real", color="black")
axc.hist(gen_counts, bins=bins, alpha=0.6, label="generated", color="#D55E00")
axc.set_xlabel("ELM peaks per rollout"); axc.set_ylabel("rollouts"); axc.legend()
if real_intervals and gen_intervals:
    ibins = np.linspace(0, np.percentile(real_intervals + gen_intervals, 95), 40)
    axi.hist(real_intervals, bins=ibins, alpha=0.6, label="real", color="black", density=True)
    axi.hist(gen_intervals, bins=ibins, alpha=0.6, label="generated", color="#D55E00", density=True)
    axi.set_xlabel("inter-ELM interval (s)"); axi.set_ylabel("density"); axi.legend()
fig.suptitle("PD ELM timing: generated vs real")
plt.show()

# %% [markdown]
# ## Plotting a rollout with peak markers
# 
# Two levels. **Simple** just drops a dot at every detected peak on the PD trace, generated
# and real. **Full** reuses the thesis peak machinery (`batch_get_peakprops` + `add_peak_markers`)
# to draw the height/width/base/energy markers on top of the signal. The full version is dense
# and slower, and its x axis is the sample index (not shot time); use it for one channel of one
# rollout when you want to inspect the peak properties themselves, not as a browsing view.

# %%
def plot_peak_markers_simple(record, channel=None, prominence=None):
    """Signal with a dot at each detected peak, generated (orange) vs real (black)."""
    channel = PD_INDEX if channel is None else channel
    prominence = ELM_PD_PROMINENCE if prominence is None else prominence
    hl = int(record["history_length"])
    gen_times = record["times"][hl:]
    gen = record["generated_x"][channel]
    real = record["real_x"][channel, hl:]
    gi, _ = find_peaks(gen, prominence=prominence)
    ri, _ = find_peaks(real, prominence=prominence)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(gen_times, real, color="black", lw=0.7, label="real")
    ax.plot(gen_times, gen, color="#D55E00", lw=0.7, alpha=0.9, label="generated")
    ax.plot(gen_times[ri], real[ri], "o", color="black", ms=4)
    ax.plot(gen_times[gi], gen[gi], "o", color="#D55E00", ms=4)
    ax.set_title(f"{CHANNEL_NAMES[channel]} peaks (prominence {prominence}) "
                 f"- shot {record['shot_number']} @ {record['start_frac']:.0%}: "
                 f"gen {len(gi)}, real {len(ri)}")
    ax.set_xlabel("Shot time (s)"); ax.legend(loc="upper right", fontsize=8)
    plt.show()


plot_peak_markers_simple(records[0])

# %%
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.plotters.flow_plots import add_peak_markers


def plot_peak_markers_full(record, channel=None, prominence=None):
    """Dense peak-property markers via the thesis machinery, one channel, generated trace.

    Slow and busy: x is the sample index, and every peak gets height/width/base/energy
    markers. For close inspection of one rollout's peaks, not for browsing.
    """
    warnings.warn("plot_peak_markers_full is dense and slow; use it on one rollout/channel.")
    channel = PD_INDEX if channel is None else channel
    prominence = ELM_PD_PROMINENCE if prominence is None else prominence
    gen = record["generated_x"]  # (channels, T), normalized
    dml_index = CHANNEL_NAMES.index("DML") if "DML" in CHANNEL_NAMES else None
    peaks_per_channel = batch_get_peakprops(
        gen[None], prominence=PROMINENCE, dml_channel_index=dml_index,
        pd_channel_index=PD_INDEX, elm_pd_prominence=prominence,
    )[0]
    # add_peak_markers adds a trace with secondary_y=False, so the figure must be a
    # make_subplots grid with a secondary y axis, not a plain go.Figure.
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(y=gen[channel], mode="lines",
                             line=dict(color="#D55E00", width=1), name=CHANNEL_NAMES[channel]))
    add_peak_markers(
        fig, peaks_per_channel[channel], group="generated", shot_number=record["shot_number"],
        hover_info_template="x=%{x}, y=%{y:.3f}", channel_color="#0072B2",
        channel_name=CHANNEL_NAMES[channel],
    )
    fig.update_layout(height=400, title=f"{CHANNEL_NAMES[channel]} peak properties "
                      f"(shot {record['shot_number']} @ {record['start_frac']:.0%}, sample index axis)",
                      xaxis_title="sample index")
    return fig


# Uncomment to render the dense view for one rollout:
# plot_peak_markers_full(records[0])

# %% [markdown]
# ## Optional: error vs rollout depth (horizon figures)
# 
# `export_horizon_analysis` produces the error-vs-depth figures and the csv/tex tables (median
# + IQR per start fraction and depth `k`, one line per model). This is the same code the run
# produces automatically; here it is against the loaded cache. Pass several `(name, records)`
# pairs to overlay models.

# %%
horizon_records = build_rollout_records(
    load_results_from_cache(cache, shots=SHOTS), data_module, step=step, shots=SHOTS)
horizon_pdf_dir = OUTPUT_DIR / "horizon"
print("horizon figures + csv/tex will be written under:", horizon_pdf_dir.resolve())
df = export_horizon_analysis(
    [(RUN_NAME, horizon_records)],
    channel_names=CHANNEL_NAMES,
    pd_index=PD_INDEX,
    elm_prominence=float(ELM_PD_PROMINENCE),
    pdf_dir=horizon_pdf_dir,
    table_dir=horizon_pdf_dir,
)
print("horizon dataframe:", df.shape)
df.head()

# %% [markdown]
# ## How to hook in, and what you can analyse
# 
# Short version of the moving parts:
# 
# - The cache is read-only and self-contained: `RolloutHDFCache(name, "r")` plus
#   `list_rollouts()` / `get_rollout(...)`, and `get_root_attrs()["config_json"]` for the run
#   config. Nothing writes to it here.
# - Everything real (observables, controls, time, true labels) is a positional slice of the
#   shot dataframe, keyed by `start_idx`. `build_rollout_records` does that join for you; if you
#   need something it does not attach, `data_module.data[data_module.data['ShotNum'] == shot]`
#   is the raw per-shot dataframe (index = physical time in seconds), and
#   `data_module.denormalize(x)` converts normalized arrays to physical units.
# - `generated_x` aligns with `times[history_length:]`; the label arrays cover the full
#   `times`. Labels are unshifted `0=L, 1=D, 2=H`.
# - Columns and prominence thresholds come from the cache (`CHANNEL_NAMES`, `C_NAMES`,
#   `PROMINENCE`, `ELM_PD_PROMINENCE`), so they match the model that produced the file.
# 
# Directions that are set up and ready:
# 
# - ELM timing: `peak_times` gives per-rollout peak timestamps; pool intervals/counts by start
#   fraction, by shot, or by rollout depth and compare the generated and real distributions.
# - Horizon behaviour: `export_horizon_analysis`, or its per-(rollout, k) dataframe, for how any
#   metric degrades with autoregressive depth.
# - Mode dynamics: the surrogate label arrays give L/D/H sequences for generated and real;
#   transition counts and dwell times are a run-length encoding away.
# - Full peak properties: `batch_get_peakprops` for height/width/prominence/base/energy per peak.
# 
# ## Your analysis space
# 
# Everything above is loaded: `cache`, `records`, `groups`, `data_module`, `cache_cfg`, and the
# helpers `peak_times`, `plot_rollout_inline`, `plot_peak_markers_simple`. Start here.

# %% [markdown]
# ## Slice-wise PD and raw DML peak comparison
# 
# This analysis compares each generated rollout sample with its corresponding real future.
# It uses the repository's rollout-window convention: slice `k` starts at `k * step` and
# has the original generated-window length. Slices overlap when `step < seq_length`.
# 
# Raw PD and raw DML peaks are evaluated separately at both configured prominence
# thresholds. For every matched slice, the analysis retains generated and real peak counts,
# signed and absolute count errors, prominence Wasserstein distance, and width Wasserstein
# distance. DML peaks are not gated by PD.
# 
# The aggregation hierarchy is preserved in separate tables:
# 
# 1. `peak_slice_metrics`: each slice of each generated sample.
# 2. `peak_sample_metrics`: slices averaged within one generated trajectory.
# 3. `peak_trajectory_metrics`: five generated samples averaged for one real trajectory.
# 4. `peak_start_metrics`: real trajectories averaged separately at each starting fraction.
# 5. `peak_overall_metrics`: optional average across starting fractions.
# 
# The starting-point table is the main report. It prevents different rollout start fractions
# from being mixed together.

# %%
from src.metrics.metrics import PeakProps

PEAK_CHANNELS = [name for name in ("PD", "DML") if name in CHANNEL_NAMES]
if set(PEAK_CHANNELS) != {"PD", "DML"}:
    raise ValueError(f"The cache must contain PD and DML; found {CHANNEL_NAMES}")

PEAK_THRESHOLDS = {
    "configured_all_peaks": float(PROMINENCE),
    "configured_elm_scale": float(ELM_PD_PROMINENCE),
}
PEAK_ANALYSIS_SAMPLES = 5
SAMPLE_RATE = float(_get(cache_cfg, "data.sample_rate", C.data.sample_rate))


def select_first_samples(records, n_samples=PEAK_ANALYSIS_SAMPLES):
    """Keep the lowest n sample indices for every (shot, start_idx) trajectory."""
    selected, counts = [], {}
    for record in sorted(records, key=lambda r: (r["shot_number"], r["start_idx"], r["sample_idx"])):
        key = (int(record["shot_number"]), int(record["start_idx"]))
        if counts.get(key, 0) < n_samples:
            selected.append(record)
            counts[key] = counts.get(key, 0) + 1
    return selected


def rollout_slice_length(record):
    """Recover the original generated-window length from the rollout layout."""
    return record["generated_x"].shape[-1] - (int(record["n_windows"]) - 1) * int(record["step"])


def build_peak_slice_metrics(records, channels=PEAK_CHANNELS, thresholds=PEAK_THRESHOLDS):
    """Return one raw PD/DML peak-comparison row per matched rollout slice."""
    rows = []
    for record in records:
        history_length = int(record["history_length"])
        step = int(record["step"])
        seq_length = rollout_slice_length(record)
        generated = record["generated_x"]
        real = record["real_x"][:, history_length:]
        for k in range(int(record["n_windows"])):
            sl = slice(k * step, k * step + seq_length)
            for channel_name in channels:
                channel_index = CHANNEL_NAMES.index(channel_name)
                for threshold_name, threshold in thresholds.items():
                    gen_peaks = PeakProps.from_find_peaks(
                        generated[channel_index, sl], prominence=threshold
                    )
                    real_peaks = PeakProps.from_find_peaks(
                        real[channel_index, sl], prominence=threshold
                    )
                    distances = gen_peaks - real_peaks
                    gen_count = gen_peaks.num_peaks()
                    real_count = real_peaks.num_peaks()
                    rows.append({
                        "shot": int(record["shot_number"]),
                        "start_idx": int(record["start_idx"]),
                        "start_frac": float(record["start_frac"]),
                        "sample_idx": int(record["sample_idx"]),
                        "k": k,
                        "channel": channel_name,
                        "threshold_name": threshold_name,
                        "threshold": threshold,
                        "n_peaks_gen": gen_count,
                        "n_peaks_real": real_count,
                        "peak_count_error": gen_count - real_count,
                        "abs_peak_count_error": abs(gen_count - real_count),
                        "prominence_wasserstein": float(distances.prominence),
                        "width_wasserstein_samples": float(distances.width),
                        "width_wasserstein_ms": float(distances.width) * 1000.0 / SAMPLE_RATE,
                    })
    return pd.DataFrame(rows)


# %%
peak_records = select_first_samples(records)
peak_slice_metrics = build_peak_slice_metrics(peak_records)
print(f"{len(peak_slice_metrics):,} slice/channel/threshold comparisons")
peak_slice_metrics.head()

# %%
TRAJECTORY_KEYS = ["shot", "start_idx", "start_frac"]
CONDITION_KEYS = ["channel", "threshold_name", "threshold"]
METRIC_COLUMNS = [
    "n_peaks_gen", "n_peaks_real", "peak_count_error",
    "abs_peak_count_error", "prominence_wasserstein",
    "width_wasserstein_samples", "width_wasserstein_ms",
]

# Average slices within each generated trajectory.
sample_groups = TRAJECTORY_KEYS + ["sample_idx"] + CONDITION_KEYS
peak_sample_metrics = (
    peak_slice_metrics.groupby(sample_groups, as_index=False)
    .agg(**{metric: (metric, "mean") for metric in METRIC_COLUMNS}, n_slices=("k", "size"))
)

# Give each generated sample equal weight within its corresponding real trajectory.
trajectory_groups = TRAJECTORY_KEYS + CONDITION_KEYS
peak_trajectory_metrics = (
    peak_sample_metrics.groupby(trajectory_groups, as_index=False)
    .agg(
        **{metric: (metric, "mean") for metric in METRIC_COLUMNS},
        n_samples=("sample_idx", "nunique"),
        n_slices_per_sample=("n_slices", "first"),
        prominence_wasserstein_sample_std=("prominence_wasserstein", "std"),
        width_wasserstein_ms_sample_std=("width_wasserstein_ms", "std"),
        abs_peak_count_error_sample_std=("abs_peak_count_error", "std"),
    )
)

# Main report: average real trajectories separately for each rollout starting fraction.
start_groups = ["start_frac"] + CONDITION_KEYS
peak_start_metrics = (
    peak_trajectory_metrics.groupby(start_groups, as_index=False)
    .agg(
        **{metric: (metric, "mean") for metric in METRIC_COLUMNS},
        n_trajectories=("shot", "size"),
        mean_samples_per_trajectory=("n_samples", "mean"),
        prominence_wasserstein_trajectory_std=("prominence_wasserstein", "std"),
        width_wasserstein_ms_trajectory_std=("width_wasserstein_ms", "std"),
        abs_peak_count_error_trajectory_std=("abs_peak_count_error", "std"),
    )
)

# Optional summary across starting fractions. Each starting fraction receives equal weight.
peak_overall_metrics = (
    peak_start_metrics.groupby(CONDITION_KEYS, as_index=False)
    .agg(**{metric: (metric, "mean") for metric in METRIC_COLUMNS}, n_starting_points=("start_frac", "nunique"))
)

print("generated samples per real trajectory:", sorted(peak_trajectory_metrics["n_samples"].unique()))
print("starting fractions:", sorted(peak_start_metrics["start_frac"].unique()))
peak_trajectory_metrics.head()

# %%
# REPORT_COLUMNS = [
#     "start_frac", "channel", "threshold_name", "threshold",
#     "n_peaks_gen", "n_peaks_real", "peak_count_error",
#     "abs_peak_count_error", "prominence_wasserstein",
#     "width_wasserstein_samples", "width_wasserstein_ms",
#     "n_trajectories", "mean_samples_per_trajectory",
# ]
# peak_start_metrics[REPORT_COLUMNS].sort_values(
#     ["start_frac", "channel", "threshold"]
# ).round(5)

# %% [markdown]
# ### Reading the result
# 
# The rows of `peak_start_metrics` remain separate for each `start_frac`, channel, and
# threshold. `n_peaks_gen` and `n_peaks_real` are mean peaks per slice. A positive
# `peak_count_error` means over-generation. `abs_peak_count_error` avoids cancellation.
# Lower values are better for both Wasserstein distances and the absolute count error.
# 
# The standard-deviation columns preserve variation across the five generated samples at
# trajectory level and variation across real shots at starting-point level. Check `n_samples`
# before interpreting a row as a five-sample average because an incomplete cache or a
# deterministic model can contain fewer samples.
# 
# Empty-slice behavior follows `PeakProps.__sub__`: two empty peak sets have distance zero.
# When only one side is empty, the distance is measured against the non-empty side's mean
# property value.

# %% [markdown]
# ## Slice-wise peak comparison across all rollout caches
# 
# This section runs the analysis above for every `*_rollout.h5` file in the resolved cache
# directory. It uses every generated sample in each cache: 30 for the full stochastic CFM
# caches, 5 for the quick cache, and 1 for deterministic caches.
# 
# Caches are processed one shot at a time so the large generated arrays and slice-level rows
# do not all remain in memory. Before processing, each cache is checked against the current
# data module for observable columns, parquet file, training split, sample rate, and both peak
# thresholds. Different control columns are allowed because this analysis only uses real and
# generated observables.
# 
# The main result is `all_cache_start_metrics`: one row per cache, starting fraction, channel,
# and threshold. `all_cache_trajectory_metrics` preserves the per-shot results, and
# `all_cache_audit` reports how many generated samples each cache contributed.

# %%
def cache_config_from_root(root_attrs):
    raw = root_attrs.get("config_json")
    if not raw:
        raise ValueError("Cache has no config_json; cannot verify analysis compatibility")
    if isinstance(raw, bytes):
        raw = raw.decode()
    return json.loads(raw)


def verify_peak_cache_compatibility(cache_name, root_attrs, candidate_cfg):
    """Fail rather than silently compare caches normalized or configured differently."""
    cache_x = [str(x) for x in root_attrs.get("cols_x", [])]
    checks = {
        "observable columns": cache_x == list(CHANNEL_NAMES),
        "data file": _get(candidate_cfg, "data.file") == _get(cache_cfg, "data.file"),
        "training split": list(_get(candidate_cfg, "data.train_shots", [])) == list(_get(cache_cfg, "data.train_shots", [])),
        "sample rate": float(_get(candidate_cfg, "data.sample_rate", SAMPLE_RATE)) == SAMPLE_RATE,
        "all-peaks threshold": np.isclose(
            float(_get(candidate_cfg, "evaluation.peaks.prominence")), PEAK_THRESHOLDS["configured_all_peaks"]
        ),
        "ELM-scale threshold": np.isclose(
            float(_get(candidate_cfg, "evaluation.peaks.elm_pd_prominence")), PEAK_THRESHOLDS["configured_elm_scale"]
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{cache_name} is incompatible for: {failed}")


def aggregate_peak_slices_to_trajectories(slice_metrics):
    """Apply slice -> sample -> real-trajectory averaging to one processing chunk."""
    sample_groups = TRAJECTORY_KEYS + ["sample_idx"] + CONDITION_KEYS
    sample_metrics = (
        slice_metrics.groupby(sample_groups, as_index=False)
        .agg(**{metric: (metric, "mean") for metric in METRIC_COLUMNS}, n_slices=("k", "size"))
    )
    trajectory_groups = TRAJECTORY_KEYS + CONDITION_KEYS
    return (
        sample_metrics.groupby(trajectory_groups, as_index=False)
        .agg(
            **{metric: (metric, "mean") for metric in METRIC_COLUMNS},
            n_samples=("sample_idx", "nunique"),
            n_slices_per_sample=("n_slices", "first"),
            prominence_wasserstein_sample_std=("prominence_wasserstein", "std"),
            width_wasserstein_ms_sample_std=("width_wasserstein_ms", "std"),
            abs_peak_count_error_sample_std=("abs_peak_count_error", "std"),
        )
    )


# %%
all_cache_paths = sorted(get_cache_dir().glob("*_rollout.h5"))
if not all_cache_paths:
    raise FileNotFoundError(f"No rollout caches found in {get_cache_dir().resolve()}")

all_trajectory_frames = []
audit_rows = []
for cache_number, cache_path in enumerate(all_cache_paths, start=1):
    model_cache_name = cache_path.stem
    model_cache = RolloutHDFCache(model_cache_name, mode="r")
    model_root = model_cache.get_root_attrs()
    model_cfg = cache_config_from_root(model_root)
    verify_peak_cache_compatibility(model_cache_name, model_root, model_cfg)

    model_keys = model_cache.list_rollouts()
    model_shots = sorted({key[0] for key in model_keys})
    samples_per_trajectory = {}
    for shot, start_idx, sample_idx in model_keys:
        samples_per_trajectory.setdefault((shot, start_idx), set()).add(sample_idx)
    sample_counts = [len(indices) for indices in samples_per_trajectory.values()]
    audit_rows.append({
        "cache_name": model_cache_name,
        "n_shots": len(model_shots),
        "n_trajectories": len(samples_per_trajectory),
        "n_generated": len(model_keys),
        "min_samples_per_trajectory": min(sample_counts),
        "max_samples_per_trajectory": max(sample_counts),
    })
    print(f"[{cache_number}/{len(all_cache_paths)}] {model_cache_name}: "
          f"{len(model_shots)} shots, {len(model_keys)} generated trajectories")

    cache_trajectory_frames = []
    for shot_number_index, shot_number in enumerate(model_shots, start=1):
        shot_keys = [key for key in model_keys if key[0] == shot_number]
        shot_step = int(model_cache.get_rollout(*shot_keys[0])["step"])
        shot_results = load_results_from_cache(model_cache, shots=[shot_number])
        shot_records = build_rollout_records(
            shot_results, data_module, step=shot_step, shots=[shot_number]
        )
        shot_slice_metrics = build_peak_slice_metrics(shot_records)
        shot_trajectory_metrics = aggregate_peak_slices_to_trajectories(shot_slice_metrics)
        cache_trajectory_frames.append(shot_trajectory_metrics)
        if shot_number_index % 10 == 0 or shot_number_index == len(model_shots):
            print(f"  processed {shot_number_index}/{len(model_shots)} shots")

    cache_trajectory_metrics = pd.concat(cache_trajectory_frames, ignore_index=True)
    cache_trajectory_metrics.insert(0, "cache_name", model_cache_name)
    all_trajectory_frames.append(cache_trajectory_metrics)

all_cache_audit = pd.DataFrame(audit_rows)
all_cache_trajectory_metrics = pd.concat(all_trajectory_frames, ignore_index=True)
print("finished all caches")
all_cache_audit

# %%
# Main comparison: caches and rollout starting fractions remain separate.
all_cache_start_groups = ["cache_name", "start_frac"] + CONDITION_KEYS
all_cache_start_metrics = (
    all_cache_trajectory_metrics.groupby(all_cache_start_groups, as_index=False)
    .agg(
        **{metric: (metric, "mean") for metric in METRIC_COLUMNS},
        n_trajectories=("shot", "size"),
        mean_samples_per_trajectory=("n_samples", "mean"),
        prominence_wasserstein_trajectory_std=("prominence_wasserstein", "std"),
        width_wasserstein_ms_trajectory_std=("width_wasserstein_ms", "std"),
        abs_peak_count_error_trajectory_std=("abs_peak_count_error", "std"),
    )
)

# Optional model-level summary. Starting fractions receive equal weight.
all_cache_overall_groups = ["cache_name"] + CONDITION_KEYS
all_cache_overall_metrics = (
    all_cache_start_metrics.groupby(all_cache_overall_groups, as_index=False)
    .agg(**{metric: (metric, "mean") for metric in METRIC_COLUMNS},
         n_starting_points=("start_frac", "nunique"))
)

ALL_CACHE_REPORT_COLUMNS = [
    "cache_name", "start_frac", "channel", "threshold_name", "threshold",
    "n_peaks_gen", "n_peaks_real", "peak_count_error",
    "abs_peak_count_error", "prominence_wasserstein",
    "width_wasserstein_samples", "width_wasserstein_ms",
    "n_trajectories", "mean_samples_per_trajectory",
]
all_cache_start_metrics[ALL_CACHE_REPORT_COLUMNS].sort_values(
    ["cache_name", "start_frac", "channel", "threshold"]
).round(5)

# %% [markdown]
# ### Save one Excel workbook per cache
# 
# Each workbook contains the starting-point report, per-real-trajectory results, the optional
# overall summary, and the cache audit row. Files are written under
# `OUTPUT_DIR / peak_analysis_excel` and keep the rollout cache name in the filename.

# %%
PEAK_EXCEL_DIR = OUTPUT_DIR / "peak_analysis_excel"
PEAK_EXCEL_DIR.mkdir(parents=True, exist_ok=True)

excel_paths = []
for model_cache_name in sorted(all_cache_start_metrics["cache_name"].unique()):
    workbook_path = PEAK_EXCEL_DIR / f"{model_cache_name}_peak_analysis.xlsx"
    cache_start = (
        all_cache_start_metrics[all_cache_start_metrics["cache_name"] == model_cache_name]
        .sort_values(["start_frac", "channel", "threshold"])
    )
    cache_trajectories = (
        all_cache_trajectory_metrics[all_cache_trajectory_metrics["cache_name"] == model_cache_name]
        .sort_values(["shot", "start_frac", "channel", "threshold"])
    )
    cache_overall = (
        all_cache_overall_metrics[all_cache_overall_metrics["cache_name"] == model_cache_name]
        .sort_values(["channel", "threshold"])
    )
    cache_audit = all_cache_audit[all_cache_audit["cache_name"] == model_cache_name]

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        cache_start.to_excel(writer, sheet_name="starting_points", index=False)
        cache_trajectories.to_excel(writer, sheet_name="trajectories", index=False)
        cache_overall.to_excel(writer, sheet_name="overall", index=False)
        cache_audit.to_excel(writer, sheet_name="cache_audit", index=False)
    excel_paths.append(workbook_path)
    print("wrote", workbook_path.resolve())

excel_exports = pd.DataFrame({"excel_file": [str(path.resolve()) for path in excel_paths]})
excel_exports

# %%
# scratch space for additional analysis
