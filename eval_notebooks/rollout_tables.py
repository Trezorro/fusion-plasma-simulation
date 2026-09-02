"""Regenerate the rollout peak/mode result tables from scratch.

Run it as a script from the repo root:

    PYTHONPATH=. python eval_notebooks/rollout_tables.py
    PYTHONPATH=. python eval_notebooks/rollout_tables.py --force        # ignore the intermediate parquet
    PYTHONPATH=. python eval_notebooks/rollout_tables.py --shots 3      # quick smoke run
    PYTHONPATH=. python eval_notebooks/rollout_tables.py --threshold large_scale  # coarser peaks
    PYTHONPATH=. python eval_notebooks/rollout_tables.py --filter gaussian:1      # smooth before find_peaks

--filter smooths both the real and the generated trace with the given kernel immediately
before find_peaks, and nothing else: the Dice score keeps using the surrogate mode labels
stored in the cache, which the classifier produced on the unfiltered signals. A filtered run
writes its own per-slice parquet and its own suffixed tables (..._gaussiansigma1.tex), so it
never overwrites or invalidates the unfiltered ones, and every caption names the filter.

Given the MODELS mapping below (display name -> rollout cache stem) it reads every cache,
recomputes every per-slice statistic, aggregates them along one explicit ladder, and writes
the CSVs plus one stacked LaTeX table. Nothing is read back from a previous run except the
per-slice parquet, which is a pure function of (cache, data module, thresholds) and is
rewritten whenever --force is passed.

What is computed, per rollout slice (one autoregressive generation, `seq_length` samples):

  |dN|      absolute peak-count error between the generated and the real slice
  W1(pi)    Wasserstein-1 distance between the generated and real peak *prominence*
            distributions, in normalized [0,1] signal units
  W1(w)     the same for peak *widths*, converted from samples to milliseconds. Widths are
            full width at half maximum and capped at one generation window; see
            WIDTH_REL_HEIGHT for why the scipy default is unusable on a whole rollout
  miss      1 if the real slice has peaks and the generated slice has none
  Dice      micro-averaged Dice between the generated and real surrogate mode label
            sequences over the slice (equals mode-sequence accuracy); higher is better

Both Wasserstein columns are reported over the slices where *both* peak sets are non-empty.
This is deliberate and it is the one substantive difference from the exploratory notebook.
`PeakProps.__sub__` substitutes the mean property value of the non-empty side when one side
is empty, which is not a distance and rewards models that emit nothing: on the PD ELM-scale
threshold every deterministic baseline produces zero peaks in every slice and collects that
sentinel, scoring *better* than the flow model, which is the only one that produces ELM peaks
at all. The `miss` column carries that information explicitly instead of smuggling it into
the distance. The sentinel-inclusive values are still written to the CSVs as `*_legacy`
columns so the older numbers remain reproducible.

The second substantive difference is where `find_peaks` runs. The exploratory notebook cuts
the rollout into slices and detects peaks inside each one independently, which truncates the
prominence and width walks at the slice boundaries and so mismeasures every peak near an edge.
This script detects once over the whole rollout, with the real history prepended as left
context, and uses the slice boundaries only to attribute each peak to a generation window.
Both variants are computed and stored (the per-slice one under a `_cut` suffix); PEAK_DETECTION
selects which the table reads, and --compare-detection prints them side by side.

See `eval_notebooks/rollout_evaluation_script.py` for the exploratory version of the same
analysis, plotting, and the interactive browser.
"""

# %%
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
while not (_REPO_ROOT / "src").is_dir() and _REPO_ROOT != _REPO_ROOT.parent:
    _REPO_ROOT = _REPO_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import src.data_loaders
from src.config import load_config_from_file
from src.hdf_cache import RolloutHDFCache, get_cache_dir
from src.metrics.metrics import PeakProps
from src.metrics.ot_peak_error import ot_peak_error
from src.signal_filters import apply_filter, filter_label, resolve_filter
from src.rollout import build_rollout_records, load_results_from_cache

# %% ---------------------------------------------------------------------------------
# Configuration. Everything the tables depend on is in this block.
# ------------------------------------------------------------------------------------

# Display name (as it appears in the table) -> rollout cache stem in the cache directory.
# Order here is the row order in the table.
MODELS: "OrderedDict[str, str]" = OrderedDict([
    ("PlasmaFlow",   "R-CNb-cfm-noleak-normal-s05-noatt-e2_rollout"),
    ("U-Net",        "UNb-unet-noleak-noatt_rollout"),
    ("TiDE",         "R-TN-tide-noleak-e2_rollout"),
    ("iTransformer", "R-IN-itransformer-noleak-e2_rollout"),
    ("DLinear",      "D-dlinear-r2_rollout"),
])

# Extra models that appear only in the appendix table: the prior and attention ablations of
# the flow model, plus the U-Net attention ablation. All no-leak, so they stay comparable to
# the main table's rows.
APPENDIX_EXTRA_MODELS: "OrderedDict[str, str]" = OrderedDict([
    ("PlasmaFlow (attn.)",            "R-CN-cfm-noleak-normal-s05-e2_rollout"),
    ("PlasmaFlow (Brownian)",         "R-CNd-cfm-noleak-brownian-s07-noatt-e2_rollout"),
    ("PlasmaFlow (Brownian, attn.)",  "R-CNc-cfm-noleak-brownian-s07-e2_rollout"),
])

# The cache whose stamped config defines the data module and the columns. Every other cache is
# checked against it and the run aborts on a mismatch.
REFERENCE_MODEL = "PlasmaFlow"

# Start fractions to report, in order. Each becomes one stacked block of the main table.
START_FRACTIONS = [0.05, 0.75]

# Depth stratification. Varying the start fraction confounds two things: how deep the rollout
# has run, and where in the discharge it sits. The stratified table fixes the start point and
# splits by rollout depth instead, so the only difference between the blocks is how long the
# model has been feeding on its own output.
#
# The start point is 0.50 because ELMs are effectively absent before ~30% of the discharge
# (measured: 0.02-0.15 real PD peaks per window below 30%, against 12-18 above 40%), so a
# rollout from 0.05 spends most of its length in the quiet phase and a fixed-depth window
# never reaches the interesting regime on the longer shots. From 0.50 every test shot supports
# 13 windows, so the depth budget below costs no shots and every window sits in the ELM-active
# phase. 0.25 would allow 19 windows but starts pre-ELM, which would confound rollout depth
# with ELM onset and is exactly the confound this table exists to remove.
STRAT_START_FRACTION = 0.50
# Equal depth halves, as (label, first k, last k) inclusive. Capped at the shortest test shot's
# window budget from STRAT_START_FRACTION so every shot contributes every depth.
STRAT_BINS = [
    ("early", 0, 5),
    ("late", 6, 11),
]

# --- pooled optimal-transport peak error -------------------------------------------------
# The depth table scores peaks with an unbalanced optimal-transport cost instead of a
# Wasserstein distance between prominence distributions; see src/metrics/ot_peak_error.py for
# the formulation and why the old column had to go. Two things change here.
#
# First, peaks are no longer sliced into generation windows for this metric. A window is 25.6
# ms and an ELM displaced across a window boundary was counted as a miss in one window and a
# false alarm in the next, which is an artefact of the grid, not a model error. The peaks of a
# whole depth stratum (STRAT_BINS, so six windows = 153.6 ms) are pooled per shot and the cost
# is computed once over that pool, with each peak's position inside the pool entering the
# transport cost. Displacement is then measured continuously and priced by how far it is.
#
# Second, mass. A peak carries its prominence as mass, so a hundred noise-scale peaks weigh
# what their prominences say they weigh and cannot stand in for a few large ELMs. This is the
# whole point: the Wasserstein column compared two *normalized* prominence distributions, so a
# model emitting many small peaks against a real trace of many small peaks plus a few large
# ones paid almost nothing for missing the large ones.
# lam is the price of one unit of unmatched mass, per ms. Matching two peaks costs their time
# separation; leaving both unmatched costs 2*lam, so **the effective maximum transport distance
# is 2*lam**, twice the number written here. Every value in OT_LAMBDAS_MS is computed and
# stored as its own set of rows in the pool parquet, keyed by `lam_ms`, so the sensitivity of
# the ranking to lam can be read off without recomputing anything; OT_LAMBDA_MS selects the
# one the tables report. Solving is cheap next to reading the caches, so the list is nearly
# free. 10 ms (a 20 ms matching horizon) is the reported value: beyond about that, a generated
# peak is not a displaced ELM but a different event.
OT_LAMBDAS_MS = [10.0, 30.0]
OT_LAMBDA_MS = 10.0
OT_PEAK_MASS = "prominence"  # "prominence" or "width"

# Which threshold each metric is computed at. The OT error wants every peak, including the
# noise-scale ones, because pricing spurious mass is half of what it measures and a high
# threshold would hide exactly the peaks it is meant to charge for. The peak *count*, by
# contrast, is meaningless at noise level (hundreds of peaks per window on PD, none of them
# ELMs), so it is reported at the large-scale threshold where a count is a count of events.
METRIC_THRESHOLD = {
    "ot_error": "all_peaks",
    "ot_error_rel": "all_peaks",
    "ot_missed_frac": "all_peaks",
    "ot_false_frac": "all_peaks",
    "pool_count_error": "large_scale",
    "pool_abs_count_error": "large_scale",
    "pool_n_peaks_real": "large_scale",
}

# A depth-table variant in which the transport cost is computed on the same ELM-scale peaks as
# the count instead of on every peak. It answers a narrower question than the default mixed
# table: not "how well is the whole peak structure reproduced", but "how well are the events
# large enough to be ELMs placed". It cannot charge a model for spurious noise-scale mass,
# since it never sees those peaks, so it is the easier table to explain and the weaker test.
# Written alongside the default rather than replacing it.
ELM_SCALE_METRIC_THRESHOLD = {m: "large_scale" for m in METRIC_THRESHOLD}

# Depth-table variants: (filename/label tag, METRIC_THRESHOLD override or None).
STRAT_TABLE_VARIANTS = [("", None), ("_elmscale", ELM_SCALE_METRIC_THRESHOLD)]

# Metrics of the depth table, which is now the pooled table.
# The count is reported signed. The absolute value hides the failure it most needs to show: a
# model that emits no ELM-scale peaks at all scores |dN| = N, a number indistinguishable from
# an ordinary miscount, and on PD it lands close enough to the flow model's error to be ranked
# beside it. Signed, the same cell reads -N and says plainly that nothing was generated.
STRAT_TABLE_METRICS = ["ot_error", "ot_error_rel", "pool_count_error"]

# Peak prominence threshold used everywhere in the reported tables. Resolved from the
# reference cache's config: "all_peaks" -> evaluation.peaks.prominence,
# "elm_scale" -> evaluation.peaks.elm_pd_prominence. Both are still computed and written to
# the parquet and the CSVs so an audit can compare them, but only this one is printed and the
# tables carry no threshold annotation.
PEAK_THRESHOLD = "all_peaks"

# Main table: one column group per channel, in this order.
TABLE_CHANNELS = ["PD", "DML"]

# Metrics shown per column group, in order. Keys index METRIC_SPECS below. "miss_rate" is
# available but off by default: at the all-peaks threshold no model misses a window except
# DLinear on DML, where the other columns already say the same thing.
TABLE_METRICS = ["abs_count_error", "prominence_w1", "width_w1_ms"]

# Metrics shown once per model (not per channel), appended after the column groups.
TABLE_GLOBAL_METRICS = ["dice"]

# --- appendix table ---------------------------------------------------------------------
# One stacked block per diagnostic, model rows inside it, at a single start fraction.
# None = every observable channel in the cache.
APPENDIX_CHANNELS = None
APPENDIX_START_FRACTION = 0.05
APPENDIX_METRICS = [
    "rate_gen_per_ms", "rate_real_per_ms", "count_error", "abs_count_error",
    "prominence_w1", "width_w1_ms",
]
APPENDIX_GLOBAL_METRICS = ["dice"]

# --- depth plots ------------------------------------------------------------------------
# Compound error against autoregressive depth. One PDF per metric, at each size in
# DEPTH_PLOT_SIZES, for a single diagnostic: one colour per model, one line style per start
# fraction, band = +/- 1 sd over shots at that depth.
DEPTH_PLOT_CHANNEL = "PD"
DEPTH_PLOT_METRICS = ["abs_count_error", "prominence_w1", "width_w1_ms", "dice"]
DEPTH_PLOT_START_FRACTIONS = [0.05, 0.75]
# Figure sizes in inches, exported once each so fonts scale relative to the figure, matching
# the other paper exports (src/plotters/rollout_horizon.py).
DEPTH_PLOT_SIZES = ((3.3, 2.1), (4.6, 2.8), (6.5, 3.6))
# Depths where too few shots remain are dropped: shots end at different times, so the sample
# shrinks with depth and the tail would otherwise be one or two rollouts wide.
DEPTH_MIN_SHOTS = 5
# Okabe-Ito, one colour per model, ordered so every adjacent pair clears the CVD separation
# check. Black stays reserved for a real-trace reference.
DEPTH_PLOT_COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
DEPTH_PLOT_STYLES = ["-", "--", ":", "-."]

# Where find_peaks runs. "full_trace" detects once over the whole rollout (with the real
# history prepended as left context) and then assigns each peak to the slice containing its
# position; "per_slice" detects inside each 256-sample slice independently, which is what the
# exploratory notebook does. Prominence and width are both measured by walking outwards until
# a higher sample or the array boundary, so per-slice detection measures boundary peaks in a
# truncated context. Both variants are always computed and stored; this only selects which one
# the LaTeX table reads. Run with --compare-detection to see the difference.
PEAK_DETECTION = "full_trace"

# How a peak's width is measured. `PeakProps.from_find_peaks` defaults to rel_height=1.0,
# which measures the width down at the peak's *base* (its full prominence below the summit).
# scipy walks outwards from the summit until the trace rises above that base level, so the
# walk is bounded only by the next higher sample, not by the peak itself. Inside a 256-sample
# window that walk is capped by the window; on a whole rollout it is not, and every peak in a
# cluster sharing the same two bounding maxima inherits the same enormous width. Measured on
# real PD traces this puts dozens of peaks at 4372 samples (437 ms) and generated ones as high
# as 7795 samples (780 ms) on an 8448-sample rollout, which is the trace length, not a
# property of any ELM. Those few values then dominate the Wasserstein distance.
#
# WIDTH_REL_HEIGHT = 0.5 is the standard full width at half maximum: the walk stops halfway
# down the peak's own prominence, so the number describes the peak's shape. WIDTH_CLIP_SAMPLES
# additionally caps a width at the generation-window length, on the grounds that the metric is
# defined per generated window and a feature wider than one window is not a within-window
# event. Set WIDTH_CLIP_SAMPLES to None to disable the cap. The rel_height=1.0 values are still
# written to the CSVs under a `_base` suffix so the earlier numbers stay reproducible.
WIDTH_REL_HEIGHT = 0.5
WIDTH_CLIP_SAMPLES = "seq_length"  # "seq_length", an int, or None

# Optional smoothing applied to both the real and the generated trace immediately before
# find_peaks; see src/signal_filters.py. At a low prominence threshold most detected peaks are
# sensor noise, and raising the threshold to remove them also removes small real events, so a
# short Gaussian kernel is the other axis to trade off against. None disables it. Either one
# spec for every channel ({"kind": "gaussian", "sigma": 3}) or a per-channel dict keyed by
# channel name with an optional "default" entry, mirroring the per-channel thresholds. The
# filter is part of the parquet signature, so changing it recomputes the slice metrics.
# Override at the command line with --filter gaussian / --filter gaussian:2 / --filter none.
SIGNAL_FILTER = None

# Cap the stochastic samples per (shot, start point). None keeps every sample in the cache.
# Deterministic baselines have one sample regardless; see the audit table for what was used.
MAX_SAMPLES = None

# The surrogate classifier assigns a label using a window that reaches ~15 samples past the
# label position, so the Dice window is extended backwards by that much, matching flow.py.
LABEL_SPILL = 15

# Observable column order, taken from the reference cache at run time; the fallback keeps a
# reused parquet renderable without opening a cache.
CHANNEL_ORDER: list = []

OUTPUT_DIR = Path("output/paper_tables")
SLICE_PARQUET = OUTPUT_DIR / "rollout_slice_metrics.parquet"
# Sidecar recording what the parquet was computed for, so a changed model list, start
# fraction, threshold or detection setting invalidates it instead of silently producing rows
# of "--" for the parts that were never computed.
SLICE_META = SLICE_PARQUET.with_suffix(".meta.json")
# The pooled OT rows live at a coarser granularity than the slice rows (one row per shot,
# sample, channel, threshold and depth stratum), so they get their own file rather than being
# broadcast onto the slice table. Both are written and invalidated together under one
# signature, so they can never describe different settings.
POOL_PARQUET = OUTPUT_DIR / "rollout_pool_metrics.parquet"


# Set by --models to skip the appendix-only ablations. Those are three 30-sample flow caches
# and they cost more than the whole main table does, so a run that only needs the main and
# depth tables can leave them out. It is part of the parquet signature, so a main-only parquet
# is correctly rejected by a later full run instead of producing empty appendix rows.
INCLUDE_APPENDIX_MODELS = True


def all_models() -> "OrderedDict[str, str]":
    """Every cache the run touches: the main rows first, then the appendix-only ablations."""
    merged = OrderedDict(MODELS)
    if INCLUDE_APPENDIX_MODELS:
        merged.update(APPENDIX_EXTRA_MODELS)
    return merged


def slice_signature() -> dict:
    """The inputs that determine the parquet's contents."""
    return {
        "models": list(all_models().items()),
        "start_fractions": [float(f) for f in sorted(
            {*START_FRACTIONS, APPENDIX_START_FRACTION, STRAT_START_FRACTION})],
        "label_spill": LABEL_SPILL,
        "max_samples": MAX_SAMPLES,
        "include_appendix_models": INCLUDE_APPENDIX_MODELS,
        "width_rel_height": WIDTH_REL_HEIGHT,
        "width_clip": WIDTH_CLIP_SAMPLES,
        "large_peak_prominence_default": LARGE_PEAK_PROMINENCE_DEFAULT,
        "large_peak_prominence_overrides": sorted(LARGE_PEAK_PROMINENCE_OVERRIDES.items()),
        "signal_filter": {c: resolve_filter(SIGNAL_FILTER, c) for c in ["__default__", *TABLE_CHANNELS]},
        "ot_lambdas_ms": [float(x) for x in OT_LAMBDAS_MS],
        "ot_peak_mass": OT_PEAK_MASS,
        "strat_start_fraction": STRAT_START_FRACTION,
        "strat_bins": [list(b) for b in STRAT_BINS],
        "schema": 11,  # bump when the per-slice columns or the threshold set change
    }


# %% ---------------------------------------------------------------------------------
# Metric presentation
# ------------------------------------------------------------------------------------

METRIC_SPECS = {
    # key: (LaTeX column header, decimals, lower_is_better, scale)
    "count_error":     (r"$\Delta N$",           2, None, 1.0),   # signed; ranked on |value|
    "abs_count_error": (r"$|\Delta N|$",         2, True,  1.0),
    "rel_count_error": (r"$|\Delta N|/N$",       2, True,  1.0),
    "rate_gen_per_ms": (r"$\hat r$",             2, None, 1.0),   # descriptive, never ranked
    "rate_real_per_ms": (r"$r$",                 2, None, 1.0),
    "prominence_w1":   (r"$W_1^{\pi}$",          4, True,  1.0),
    "width_w1_ms":     (r"$W_1^{w}$ [ms]",       2, True,  1.0),
    "width_w1_ms_base": (r"$W_1^{w,\mathrm{base}}$ [ms]", 2, True, 1.0),  # audit only
    "miss_rate":       (r"miss \%",              1, True,  100.0),
    # Pooled optimal-transport columns. E_OT is absolute, in (prominence x ms); E_OT/E_0 is
    # the same number divided by the cost of predicting no peaks at all, so 1.0 means "no
    # better than an empty trace" and it is comparable across shots and diagnostics.
    "ot_error":        (r"$E_{\mathrm{OT}}$",      2, True,  1.0),
    "ot_error_rel":    (r"$E_{\mathrm{OT}}/E_{0}$", 3, True, 1.0),
    "ot_missed_frac":  (r"$m_{\mathrm{miss}}$",    3, True,  1.0),
    "ot_false_frac":   (r"$m_{\mathrm{false}}$",   3, True,  1.0),
    "pool_count_error":     (r"$\Delta N$",        2, None,  1.0),
    "pool_abs_count_error": (r"$|\Delta N|$",      2, True,  1.0),
    "pool_n_peaks_real":    (r"$N$",                2, None,  1.0),
    "dice":            (r"$\mathcal{D}\uparrow$", 3, False, 1.0),
}

# Metrics ranked on the absolute value rather than the signed one (a signed count error of
# -0.1 beats +5.0), and metrics that describe the data rather than score a model.
RANK_ON_ABS = {"count_error"}
NEVER_RANKED = {"rate_gen_per_ms", "rate_real_per_ms", "pool_n_peaks_real"}
RANK_ON_ABS = RANK_ON_ABS | {"pool_count_error"}
# Computed once on the pooled peaks, never per slice, so they have no per-slice detection
# variant to select between.
POOL_METRICS_SET = {"ot_error", "ot_error_rel", "ot_missed_frac", "ot_false_frac",
                    "ot_loc", "ot_missed_mass", "ot_false_mass", "ot_gen_mass", "ot_real_mass",
                    "pool_count_error", "pool_abs_count_error",
                    "pool_n_peaks_gen", "pool_n_peaks_real"}

# %% ---------------------------------------------------------------------------------
# Cache / data module setup
# ------------------------------------------------------------------------------------


def _get(d, dotted, default=None):
    for key in dotted.split("."):
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d


def cache_config(cache: RolloutHDFCache) -> dict:
    raw = cache.get_root_attrs().get("config_json")
    if not raw:
        raise ValueError(
            f"{cache.name if hasattr(cache, 'name') else cache} has no stamped config_json; "
            "this script refuses to guess the config that produced a cache."
        )
    return json.loads(raw.decode() if isinstance(raw, bytes) else raw)


def build_data_module(reference_cfg):
    """Data module built from the reference cache's own data block, not the local yaml."""
    C = load_config_from_file("plasmaflow", as_omega=True)
    data_block = reference_cfg.get("data")
    if not isinstance(data_block, dict) or not data_block:
        raise ValueError("Reference cache config has no data block.")
    C.data = OmegaConf.merge(C.data, OmegaConf.create(data_block))
    parquet = Path(C.data.dir) / C.data.file
    if not parquet.exists():
        raise FileNotFoundError(
            f"The reference cache was produced against {C.data.file}, which is not present at "
            f"{parquet}. The real traces are re-derived from it, so the tables cannot be built."
        )
    data_module = getattr(src.data_loaders, C.data.Class)(**C.data)
    data_module.prepare_data()
    data_module.setup()
    # build_rollout_records slices the real trace as [start_i - history_length, start_i + T).
    # compute_rollout_specs keeps start_i >= crop_margin, so this is only in bounds while
    # crop_margin >= history_length; if that ever stops holding the real trace is silently
    # truncated and every slice comparison shifts.
    if int(C.data.crop_margin) < int(C.data.history_length):
        raise ValueError(
            f"crop_margin ({C.data.crop_margin}) < history_length ({C.data.history_length}); "
            "the real context slice would run off the start of the shot."
        )
    return C, data_module


# Extra prominence threshold for a coarser "large peaks only" pass. Not read from the config
# because it is not one of flow.py's own evaluation thresholds, just a table variant. PD and
# DML do not sit on the same scale (PD's ELM spikes are far taller, in normalized [0,1] units,
# than DML's slower deflections), so this is per channel rather than one number applied
# everywhere like the other two thresholds. A channel with no entry here (the appendix-only
# diagnostics) falls back to LARGE_PEAK_PROMINENCE_DEFAULT.
LARGE_PEAK_PROMINENCE_DEFAULT = 0.01
LARGE_PEAK_PROMINENCE_OVERRIDES = {"PD": 0.05, "DML": 0.005}


def resolve_thresholds(cfg) -> dict:
    peaks = _get(cfg, "evaluation.peaks", {}) or {}
    return {
        "all_peaks": float(peaks["prominence"]),
        "elm_scale": float(peaks["elm_pd_prominence"]),
        "large_scale": dict(LARGE_PEAK_PROMINENCE_OVERRIDES),
    }


def _resolve_threshold(threshold, channel_name):
    """A scalar threshold, or the per-channel override/fallback for a dict-valued one."""
    if isinstance(threshold, dict):
        return threshold.get(channel_name, LARGE_PEAK_PROMINENCE_DEFAULT)
    return float(threshold)


def verify_compatibility(name, cache_name, cfg, ref_cfg, channel_names, thresholds, sample_rate):
    """Refuse to compare caches that were produced under different data or peak settings."""
    root_x = [str(c) for c in cfg_cols(cfg)]
    cache_thresholds = resolve_thresholds(cfg)
    checks = {
        "observable columns": root_x == list(channel_names),
        "data file": _get(cfg, "data.file") == _get(ref_cfg, "data.file"),
        "training split": list(_get(cfg, "data.train_shots") or []) == list(_get(ref_cfg, "data.train_shots") or []),
        "test split": list(_get(cfg, "data.test_shots") or []) == list(_get(ref_cfg, "data.test_shots") or []),
        # history_length and seq_length decide how the real trace is aligned to the generated
        # one and how long a slice is; a mismatch would shift the comparison without crashing.
        "history_length": int(_get(cfg, "data.history_length")) == int(_get(ref_cfg, "data.history_length")),
        "seq_length": int(_get(cfg, "data.seq_length")) == int(_get(ref_cfg, "data.seq_length")),
        "sample rate": float(_get(cfg, "data.sample_rate")) == sample_rate,
        "all-peaks threshold": np.isclose(cache_thresholds["all_peaks"], thresholds["all_peaks"]),
        "ELM-scale threshold": np.isclose(cache_thresholds["elm_scale"], thresholds["elm_scale"]),
    }
    failed = [check for check, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"'{name}' ({cache_name}) is incompatible with the reference cache for: {failed}")


def cfg_cols(cfg):
    return list(_get(cfg, "data.cols.x") or [])


# %% ---------------------------------------------------------------------------------
# Per-slice statistics
# ------------------------------------------------------------------------------------


def dice_scores(pred: np.ndarray, target: np.ndarray, n_classes: int = 3) -> tuple[float, float]:
    """(micro, macro) Dice between two integer label sequences.

    Micro Dice over a single-label multiclass sequence equals the accuracy; this is the
    variant flow.py logs (torchmetrics DiceScore defaults to average='micro'). Macro is the
    unweighted mean over the classes present in either sequence, which is more sensitive to
    the rare D and H modes.
    """
    pred = np.asarray(pred).astype(int)
    target = np.asarray(target).astype(int)
    micro = float((pred == target).mean()) if pred.size else float("nan")
    per_class = []
    for c in range(n_classes):
        p, t = pred == c, target == c
        denom = p.sum() + t.sum()
        if denom == 0:
            continue  # class absent from both; undefined rather than a free 1.0
        per_class.append(2.0 * float((p & t).sum()) / float(denom))
    macro = float(np.mean(per_class)) if per_class else float("nan")
    return micro, macro


def slice_length(record) -> int:
    """The generated-window length L, from the rollout layout T = (K-1)*step + L."""
    return int(record["generated_x"].shape[-1]) - (int(record["n_windows"]) - 1) * int(record["step"])


def _w1(a, b) -> float:
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(np.asarray(a, dtype=float), np.asarray(b, dtype=float)))


def _subset(peaks: PeakProps, mask) -> PeakProps:
    """The PeakProps restricted to a boolean mask over its peaks."""
    return PeakProps(
        X=peaks.X[mask], Y=peaks.Y[mask], prominences=peaks.prominences[mask],
        bases=peaks.bases[mask], left_ips=peaks.left_ips[mask], right_ips=peaks.right_ips[mask],
    )


def detect_full_trace(trace, context, prominence, offset, step, n_windows, rel_height=1.0) -> list[PeakProps]:
    """Detect peaks once on the whole trace, then bucket them into the K rollout slices.

    `find_peaks` derives a peak's prominence by walking left and right until it meets a higher
    sample, and with rel_height=1.0 it measures the width down at the full-prominence level;
    both walks stop at the array boundary. Detecting inside a 256-sample slice therefore
    measures a boundary peak against whatever the slice happens to contain rather than against
    its true surrounding trough, which can push a genuine peak below the threshold or lift a
    minor one above it, and clips the widths of any peak within one width of an edge. At the
    ELM scale the widths run to ~75 samples against a 256-sample slice, so that is roughly a
    third of the span, not a marginal effect. It is also not self-cancelling across models,
    because a model with a different peak density has a different chance of a peak landing on
    a boundary.

    Detecting on the full trace fixes the measurement while keeping the slice boundaries for
    *attribution* only: each peak is assigned to the slice containing its position, so the
    rollout-depth index k still means "the k-th autoregressive generation". `context` is the
    real history prepended to the left so the first slice has proper context too; peaks inside
    it are discarded after detection. The single remaining edge is the end of the rollout,
    where no further samples exist for either trace.

    Args:
        trace: 1-D signal over the rollout span.
        context: samples immediately preceding it (the real history), used for context only.
        prominence: detection threshold in normalized [0,1] units.
        offset: len(context); positions are re-based by this before bucketing.
        step: slice stride, which must equal the slice length for bucketing to be well defined.
        n_windows: K.

    Returns:
        A list of K PeakProps, one per slice.
    """
    peaks = PeakProps.from_find_peaks(np.concatenate([context, trace]), prominence=prominence, rel_height=rel_height)
    positions = np.asarray(peaks.X) - offset
    return [
        _subset(peaks, (positions >= k * step) & (positions < (k + 1) * step))
        for k in range(n_windows)
    ]


def detect_per_slice(trace, prominence, step, seq_length, n_windows, rel_height=1.0) -> list[PeakProps]:
    """Detect peaks independently inside each slice (the original behaviour, kept for comparison)."""
    return [
        PeakProps.from_find_peaks(trace[k * step: k * step + seq_length], prominence=prominence, rel_height=rel_height)
        for k in range(n_windows)
    ]


def _widths(peaks: PeakProps, clip):
    """Peak widths in samples, optionally capped at `clip`.

    The cap keeps a single trace-spanning base walk from dominating the Wasserstein
    distance; see WIDTH_CLIP_SAMPLES.
    """
    w = np.asarray(peaks.widths, dtype=float)
    return w if clip is None else np.minimum(w, float(clip))


def prepare_traces(generated, real, real_full, channel_names, channel_name, history_length):
    """(generated, real, context) traces for one channel, smoothed if SIGNAL_FILTER is set.

    Smoothing is applied here, once per (channel, trace), rather than inside the detectors:
    history and rollout are filtered as one array so the kernel does not see an artificial edge
    at the seam, and every detector downstream then reads exactly the same samples. Real and
    generated get an identical filter, so this changes what counts as a peak but not who is
    being measured. Everything downstream (positions, prominences, widths) is measured on the
    filtered trace.

    The rollout was seeded with the real history, so the real history is the correct left
    context for the generated trace as well as for the real one.
    """
    ci = channel_names.index(channel_name)
    context = real_full[ci, :history_length]
    gen_trace, real_trace = generated[ci], real[ci]
    if resolve_filter(SIGNAL_FILTER, channel_name) is not None:
        gen_cat = apply_filter(np.concatenate([context, gen_trace]), SIGNAL_FILTER, channel_name)
        real_cat = apply_filter(np.concatenate([context, real_trace]), SIGNAL_FILTER, channel_name)
        gen_trace, real_trace = gen_cat[history_length:], real_cat[history_length:]
        # The two contexts are the same real samples, so either filtered copy will do.
        context = real_cat[:history_length]
    return gen_trace, real_trace, context


def _peak_stats(gen_peaks: PeakProps, real_peaks: PeakProps, sample_rate, slice_ms, suffix="", width_clip=None) -> dict:
    """The per-slice peak comparison for one pair of peak sets.

    `slice_ms` is the slice duration in milliseconds, used for the peak rates. Rates are the
    same information as the counts but in physical units, which is what makes them comparable
    across diagnostics with different characteristic peak densities.
    """
    n_gen, n_real = gen_peaks.num_peaks(), real_peaks.num_peaks()
    both = n_gen > 0 and n_real > 0
    legacy = gen_peaks - real_peaks  # sentinel-substituting, see the module docstring
    return {
        f"n_peaks_gen{suffix}": n_gen,
        f"n_peaks_real{suffix}": n_real,
        f"rate_gen_per_ms{suffix}": n_gen / slice_ms,
        f"rate_real_per_ms{suffix}": n_real / slice_ms,
        f"count_error{suffix}": n_gen - n_real,
        f"abs_count_error{suffix}": abs(n_gen - n_real),
        # Relative to the real count, so a stratum with denser ELMs is not automatically
        # scored as worse. Undefined (and skipped, like the Wasserstein columns) where the
        # real window has no peaks at all to be relative to.
        f"rel_count_error{suffix}": abs(n_gen - n_real) / n_real if n_real > 0 else np.nan,
        f"both_nonempty{suffix}": float(both),
        f"miss_rate{suffix}": float(n_real > 0 and n_gen == 0),
        f"spurious_rate{suffix}": float(n_real == 0 and n_gen > 0),
        f"prominence_w1{suffix}": _w1(gen_peaks.prominences, real_peaks.prominences) if both else np.nan,
        f"width_w1_ms{suffix}": (_w1(_widths(gen_peaks, width_clip), _widths(real_peaks, width_clip))
                                 * 1000.0 / sample_rate) if both else np.nan,
        f"prominence_w1_legacy{suffix}": float(legacy.prominence),
        f"width_w1_ms_legacy{suffix}": float(legacy.width) * 1000.0 / sample_rate,
    }


def record_slice_rows(record, channel_names, channels, thresholds, sample_rate, expected_L):
    """One row per (slice, channel, threshold) for a single rollout.

    Every peak column is emitted twice: unsuffixed for the full-trace detection and with a
    `_cut` suffix for per-slice detection, so the two variants live in the same parquet and
    PEAK_DETECTION only selects which one the table reads.
    """
    history_length = int(record["history_length"])
    step = int(record["step"])
    n_windows = int(record["n_windows"])
    generated = record["generated_x"]
    real_full = record["real_x"]
    real = real_full[:, history_length:]
    labels_gen = np.asarray(record["surr_labels_gen"])
    labels_real = np.asarray(record["surr_labels_real"])

    L = slice_length(record)
    if L != expected_L:
        raise ValueError(f"slice length {L} from the rollout layout != config seq_length {expected_L}")
    if step != L:
        # With overlapping slices a peak belongs to several of them and `position // step`
        # is no longer an assignment; the full-trace variant would need a different rule.
        raise ValueError(
            f"full-trace peak attribution assumes non-overlapping slices, but step={step} != L={L}"
        )
    if real.shape[-1] != generated.shape[-1]:
        raise ValueError(
            f"real trace is {real.shape[-1]} samples but the rollout is {generated.shape[-1]}; "
            "the shot ran out before the rollout did and the slices would be misaligned"
        )
    if labels_gen.shape[-1] != history_length + generated.shape[-1]:
        raise ValueError("surrogate label array does not span history + rollout")

    slice_ms = 1000.0 * L / sample_rate
    width_clip = L if WIDTH_CLIP_SAMPLES == "seq_length" else WIDTH_CLIP_SAMPLES
    base = {
        "slice_length": L,
        "shot": int(record["shot_number"]),
        "start_idx": int(record["start_idx"]),
        "start_frac": float(record["start_frac"]),
        "sample_idx": int(record["sample_idx"]),
    }
    rows = []
    for channel_name in channels:
        gen_trace, real_trace, context = prepare_traces(
            generated, real, real_full, channel_names, channel_name, history_length)
        for threshold_name, threshold in thresholds.items():
            threshold = _resolve_threshold(threshold, channel_name)
            rh = WIDTH_REL_HEIGHT
            gen_full = detect_full_trace(gen_trace, context, threshold, history_length, step, n_windows, rh)
            real_full_pk = detect_full_trace(real_trace, context, threshold, history_length, step, n_windows, rh)
            gen_cut = detect_per_slice(gen_trace, threshold, step, L, n_windows, rh)
            real_cut = detect_per_slice(real_trace, threshold, step, L, n_windows, rh)
            # The rel_height=1.0 base-level widths, uncapped, kept only so the pre-fix width
            # numbers stay reproducible from the same parquet. Peak positions, counts and
            # prominences are identical to the pass above; only the widths differ.
            gen_base = detect_full_trace(gen_trace, context, threshold, history_length, step, n_windows, 1.0)
            real_base = detect_full_trace(real_trace, context, threshold, history_length, step, n_windows, 1.0)
            gen_base_cut = detect_per_slice(gen_trace, threshold, step, L, n_windows, 1.0)
            real_base_cut = detect_per_slice(real_trace, threshold, step, L, n_windows, 1.0)
            for k in range(n_windows):
                rows.append({
                    **base,
                    "k": k,
                    "channel": channel_name,
                    "threshold_name": threshold_name,
                    "threshold": threshold,
                    **_peak_stats(gen_full[k], real_full_pk[k], sample_rate, slice_ms, width_clip=width_clip),
                    **_peak_stats(gen_cut[k], real_cut[k], sample_rate, slice_ms, suffix="_cut",
                                  width_clip=width_clip),
                    "width_w1_ms_base": _peak_stats(
                        gen_base[k], real_base[k], sample_rate, slice_ms)["width_w1_ms"],
                    "width_w1_ms_base_cut": _peak_stats(
                        gen_base_cut[k], real_base_cut[k], sample_rate, slice_ms)["width_w1_ms"],
                })

    # Dice does not depend on the channel or threshold, so it is computed once per slice and
    # broadcast onto the rows.
    dice_by_k = {}
    for k in range(n_windows):
        # The slice plus LABEL_SPILL samples of context before it, matching the crop flow.py
        # applies for the classifier's forward-looking window.
        lo = history_length + k * step - LABEL_SPILL
        hi = history_length + k * step + L
        dice_by_k[k] = dice_scores(labels_gen[lo:hi], labels_real[lo:hi])
    for row in rows:
        row["dice"], row["dice_macro"] = dice_by_k[row["k"]]
    return rows


def pool_peaks(trace, context, prominence, history_length, step, lo, hi, sample_rate, rel_height):
    """(times_ms, masses) of every peak in rollout windows lo..hi inclusive, pooled.

    Detection runs once over history+rollout for the same reason it does per slice (a peak's
    prominence and width are properties of its surrounding trough, not of whatever the window
    happens to contain), but the window grid is used only to select the pool: within it, a
    peak keeps its continuous position. Times are milliseconds from the start of the pool, so
    the generated and the real set share an origin, which is what makes their displacement
    meaningful.
    """
    peaks = PeakProps.from_find_peaks(
        np.concatenate([context, trace]), prominence=prominence, rel_height=rel_height)
    positions = np.asarray(peaks.X, dtype=float) - history_length
    keep = (positions >= lo * step) & (positions < (hi + 1) * step)
    times_ms = (positions[keep] - lo * step) * 1000.0 / sample_rate
    if OT_PEAK_MASS == "prominence":
        mass = np.asarray(peaks.prominences, dtype=float)[keep]
    elif OT_PEAK_MASS == "width":
        mass = np.asarray(peaks.widths, dtype=float)[keep] * 1000.0 / sample_rate
    else:
        raise ValueError(f"OT_PEAK_MASS must be 'prominence' or 'width', got {OT_PEAK_MASS!r}")
    return times_ms, mass


def record_pool_rows(record, channel_names, channels, thresholds, sample_rate, expected_L):
    """One row per (depth stratum, channel, threshold) for a single rollout.

    This is the pooled counterpart of `record_slice_rows`: no per-window slicing, one optimal
    transport solve over all the peaks of a stratum. Only records at STRAT_START_FRACTION
    produce rows, since the strata are only defined there.
    """
    if not np.isclose(record["start_frac"], STRAT_START_FRACTION):
        return []
    history_length = int(record["history_length"])
    step = int(record["step"])
    n_windows = int(record["n_windows"])
    generated = record["generated_x"]
    real_full = record["real_x"]
    real = real_full[:, history_length:]

    L = slice_length(record)
    if L != expected_L or step != L:
        raise ValueError(f"pooled peaks assume non-overlapping windows of the config length; "
                         f"got L={L}, step={step}, seq_length={expected_L}")
    # A stratum that the rollout does not reach would be pooled from fewer windows than the
    # same stratum on another shot, so the shots would not be comparable. STRAT_BINS is capped
    # at the shortest test shot's budget precisely so this cannot happen; if it does, the bins
    # and the caches disagree and silently averaging over unequal spans would hide it.
    max_k = max(hi for _, _, hi in STRAT_BINS)
    if n_windows <= max_k:
        raise ValueError(
            f"shot {record['shot_number']} has {n_windows} rollout windows but STRAT_BINS "
            f"needs {max_k + 1}; lower the depth budget or drop the shot")

    base = {
        "shot": int(record["shot_number"]),
        "start_idx": int(record["start_idx"]),
        "start_frac": float(record["start_frac"]),
        "sample_idx": int(record["sample_idx"]),
        "slice_length": L,
    }
    rows = []
    for channel_name in channels:
        gen_trace, real_trace, context = prepare_traces(
            generated, real, real_full, channel_names, channel_name, history_length)
        for threshold_name, threshold in thresholds.items():
            threshold = _resolve_threshold(threshold, channel_name)
            for stratum, lo, hi in STRAT_BINS:
                args = (context, threshold, history_length, step, lo, hi, sample_rate, WIDTH_REL_HEIGHT)
                gen_t, gen_m = pool_peaks(gen_trace, *args)
                real_t, real_m = pool_peaks(real_trace, *args)
                n_gen, n_real = len(gen_t), len(real_t)
                # The peaks are detected once and then priced at every lam: only the transport
                # solve depends on lam, and it is the cheap half.
                for lam in OT_LAMBDAS_MS:
                    err = ot_peak_error(gen_t, gen_m, real_t, real_m, lam)
                    rows.append({
                        **base,
                        "lam_ms": float(lam),
                        "stratum": stratum, "k_lo": lo, "k_hi": hi,
                        "pool_ms": 1000.0 * (hi - lo + 1) * L / sample_rate,
                        "channel": channel_name,
                        "threshold_name": threshold_name, "threshold": threshold,
                        "ot_error": err.total,
                        "ot_error_rel": err.relative,
                        "ot_loc": err.loc,
                        "ot_missed_mass": err.missed_mass,
                        "ot_false_mass": err.false_mass,
                        "ot_gen_mass": err.gen_mass,
                        "ot_real_mass": err.real_mass,
                        # Which side of the budget the error comes from: how much of the real mass
                        # went unmatched, and how much generated mass was spurious relative to the
                        # real total. Both are on the same denominator so they are comparable, and
                        # together with ot_loc they account for the whole of ot_error.
                        "ot_missed_frac": err.missed_mass / err.real_mass if err.real_mass else np.nan,
                        "ot_false_frac": err.false_mass / err.real_mass if err.real_mass else np.nan,
                        "pool_n_peaks_gen": n_gen,
                        "pool_n_peaks_real": n_real,
                        "pool_count_error": n_gen - n_real,
                        "pool_abs_count_error": abs(n_gen - n_real),
                    })
    return rows


# %% ---------------------------------------------------------------------------------
# Aggregation ladder
# ------------------------------------------------------------------------------------

TRAJECTORY_KEYS = ["model", "shot", "start_idx", "start_frac"]
CONDITION_KEYS = ["channel", "threshold_name"]
# Per-slice quantities averaged at every level. Wasserstein columns skip NaN slices by
# construction (pandas mean is NaN-skipping), which is exactly the both-non-empty restriction.
PEAK_METRICS = [
    "n_peaks_gen", "n_peaks_real", "rate_gen_per_ms", "rate_real_per_ms",
    "count_error", "abs_count_error",
    "miss_rate", "spurious_rate", "both_nonempty",
    "prominence_w1", "width_w1_ms", "prominence_w1_legacy", "width_w1_ms_legacy",
]
# Ratios that must never be averaged, at any level. |dN|/N is dominated by whatever unit holds
# the fewest real peaks: 4 of 43 shots average under one real PD peak per window and 10 do on
# DML, so a mean of per-shot ratios reads 2.18 on DML against a true 0.99, and on PD it is
# outright infinite because one shot has no real peaks at all. These are formed once, at the
# very end, as a ratio of the two aggregate means. The per-slice column stays in the parquet
# for auditing but is never aggregated.
DERIVED_METRICS = ["rel_count_error", "rel_count_error_cut"]
SLICE_METRICS = (PEAK_METRICS + [f"{m}_cut" for m in PEAK_METRICS]
                 + ["width_w1_ms_base", "width_w1_ms_base_cut", "dice", "dice_macro"])


def aggregate(slice_metrics: pd.DataFrame):
    """slice -> sample -> real trajectory -> (model, start fraction).

    Every level is an unweighted mean over the level below, so a shot with many slices does
    not outweigh a short one and a stochastic model's samples are averaged before the shot
    mean is taken. The reported spread is the standard deviation *across real trajectories*
    at the last step.
    """
    sample = (
        slice_metrics
        .groupby(TRAJECTORY_KEYS + ["sample_idx"] + CONDITION_KEYS, as_index=False)
        .agg(**{m: (m, "mean") for m in SLICE_METRICS}, n_slices=("k", "size"))
    )
    trajectory = (
        sample
        .groupby(TRAJECTORY_KEYS + CONDITION_KEYS, as_index=False)
        .agg(**{m: (m, "mean") for m in SLICE_METRICS},
             n_samples=("sample_idx", "nunique"), n_slices=("n_slices", "first"))
    )
    agg = {m: (m, "mean") for m in SLICE_METRICS}
    agg |= {f"{m}_std": (m, "std") for m in SLICE_METRICS}
    start = (
        trajectory
        .groupby(["model", "start_frac"] + CONDITION_KEYS, as_index=False)
        .agg(**agg, n_trajectories=("shot", "size"), n_samples=("n_samples", "mean"))
    )
    return sample, trajectory, _add_derived(start)


def _add_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Pooled ratio metrics, formed after the last aggregation step; see DERIVED_METRICS.

    Both operands are already means over shots, so this is one ratio of two aggregates rather
    than an average of per-unit ratios. It therefore carries no standard deviation, and the
    `_std` columns are set to NaN so the formatter prints the bare value.
    """
    for sfx in ("", "_cut"):
        num = frame[f"abs_count_error{sfx}"]
        den = frame[f"n_peaks_real{sfx}"].replace(0, np.nan)
        frame[f"rel_count_error{sfx}"] = num / den
        frame[f"rel_count_error{sfx}_std"] = np.nan
    return frame


def stratum_of(k):
    """The STRAT_BINS label containing depth k, or None if k falls outside every bin."""
    for label, lo, hi in STRAT_BINS:
        if lo <= k <= hi:
            return label
    return None


def aggregate_by_stratum(slice_metrics: pd.DataFrame):
    """Like `aggregate`, but blocks are depth strata at one fixed start fraction.

    The start fraction is held at STRAT_START_FRACTION and the depth index k is binned by
    STRAT_BINS, so the two blocks differ only in how far the rollout has run. Slices outside
    every bin are dropped, which is what keeps the depth budget identical across shots: the
    cap is the shortest shot's, so no shot is missing a depth that another one contributes.
    """
    d = slice_metrics[np.isclose(slice_metrics["start_frac"], STRAT_START_FRACTION)].copy()
    if d.empty:
        raise ValueError(
            f"no slices at start fraction {STRAT_START_FRACTION}; add it to START_FRACTIONS "
            "(it is already in the signature) and rerun with --force"
        )
    d["stratum"] = d["k"].map(stratum_of)
    d = d[d["stratum"].notna()]

    keys = ["model", "shot", "start_idx", "stratum"] + CONDITION_KEYS
    sample = (d.groupby(keys + ["sample_idx"], as_index=False)
               .agg(**{m: (m, "mean") for m in SLICE_METRICS}, n_slices=("k", "size")))
    trajectory = (sample.groupby(keys, as_index=False)
                  .agg(**{m: (m, "mean") for m in SLICE_METRICS},
                       n_samples=("sample_idx", "nunique"), n_slices=("n_slices", "first")))
    agg = {m: (m, "mean") for m in SLICE_METRICS}
    agg |= {f"{m}_std": (m, "std") for m in SLICE_METRICS}
    stratum = (trajectory.groupby(["model", "stratum"] + CONDITION_KEYS, as_index=False)
               .agg(**agg, n_trajectories=("shot", "size"), n_samples=("n_samples", "mean")))
    return _add_derived(stratum)


POOL_METRICS = [
    "ot_error", "ot_error_rel", "ot_loc", "ot_missed_mass", "ot_false_mass",
    "ot_gen_mass", "ot_real_mass", "ot_missed_frac", "ot_false_frac",
    "pool_n_peaks_gen", "pool_n_peaks_real", "pool_count_error", "pool_abs_count_error",
]


def aggregate_pools(pool_metrics: pd.DataFrame) -> pd.DataFrame:
    """sample -> real trajectory -> (model, stratum), the same ladder `aggregate` uses.

    A stochastic model's samples are averaged into one number per real trajectory before the
    shots are averaged, so a model does not gain from having more samples in the cache, and
    the reported spread is the standard deviation across shots. There is no slice level here:
    the pool *is* the unit.
    """
    keys = ["model", "shot", "start_idx", "stratum"] + CONDITION_KEYS + ["lam_ms"]
    trajectory = (pool_metrics.groupby(keys, as_index=False)
                  .agg(**{m: (m, "mean") for m in POOL_METRICS},
                       n_samples=("sample_idx", "nunique")))
    agg = {m: (m, "mean") for m in POOL_METRICS}
    agg |= {f"{m}_std": (m, "std") for m in POOL_METRICS}
    return (trajectory.groupby(["model", "stratum"] + CONDITION_KEYS + ["lam_ms"], as_index=False)
            .agg(**agg, n_trajectories=("shot", "size"), n_samples=("n_samples", "mean")))


def merge_pool_metrics(stratum_metrics: pd.DataFrame, pool_stratum: pd.DataFrame,
                       lam: float = None) -> pd.DataFrame:
    """Attach the pooled columns to the per-slice stratum frame on their shared keys.

    The two frames answer different questions at the same granularity (model, stratum, channel,
    threshold), so the table renderer can read a pooled and a per-slice column side by side
    without knowing which is which. An outer join: a threshold present in one and not the other
    should surface as a '--' cell, not silently drop the row that does exist.
    """
    keys = ["model", "stratum"] + CONDITION_KEYS
    # The pool frame carries every lam in OT_LAMBDAS_MS; the table reports one. Selecting here
    # rather than in the renderer keeps the merged frame at the stratum frame's granularity, so
    # a lam column can never silently duplicate rows into the per-slice metrics.
    lam = OT_LAMBDA_MS if lam is None else float(lam)
    selected = pool_stratum[np.isclose(pool_stratum["lam_ms"], lam)]
    if selected.empty:
        raise ValueError(
            f"lambda={lam} is not in the parquet, which holds "
            f"{sorted(pool_stratum['lam_ms'].unique())}; add it to OT_LAMBDAS_MS and rerun "
            "with --force")
    shared = (set(stratum_metrics.columns) & set(selected.columns)) - set(keys)
    return stratum_metrics.merge(
        selected.drop(columns=sorted(shared)), on=keys, how="outer", validate="one_to_one")


# %% ---------------------------------------------------------------------------------
# LaTeX rendering
# ------------------------------------------------------------------------------------


def _fmt(mean, std, decimals, scale=1.0):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    body = f"{mean * scale:.{decimals}f}"
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return body
    return rf"{body}_{{\pm{std * scale:.{decimals}f}}}"


def _rank_marks(values, metric, tol=1e-12):
    """Index -> LaTeX wrapper for best (bold) and second best (underline).

    Ties share a mark rather than being broken by row order: several models can land on
    exactly the same value (for instance when none of them produces any peak in a window),
    and picking one of them arbitrarily would read as a real ranking. Signed metrics are
    ranked on their absolute value, and descriptive columns are not ranked at all.
    """
    _, _, lower_is_better, _ = METRIC_SPECS[metric]
    if metric in NEVER_RANKED or lower_is_better is None and metric not in RANK_ON_ABS:
        return {}
    key = abs if metric in RANK_ON_ABS else (lambda v: v)
    lower_is_better = True if metric in RANK_ON_ABS else lower_is_better
    finite = [(key(v), i) for i, v in enumerate(values) if v is not None and not np.isnan(v)]
    if len(finite) < 2:
        return {}
    ordered = sorted(v for v, _ in finite)
    if not lower_is_better:
        ordered = ordered[::-1]
    best = ordered[0]
    second = next((v for v in ordered if abs(v - best) > tol), None)
    marks = {}
    for v, i in finite:
        if abs(v - best) <= tol:
            marks[i] = r"\mathbf{%s}"
        elif second is not None and abs(v - second) <= tol:
            marks[i] = r"\underline{%s}"
    return marks


def _metric_key(metric: str, detection: str) -> str:
    """Column name in the aggregated frame for a metric under a detection variant."""
    if metric in ("dice", "dice_macro") or metric in POOL_METRICS_SET:
        return metric  # detection-independent
    return metric if detection == "full_trace" else f"{metric}_cut"


def _block_mask(frame, block_val, block_key="start_frac"):
    """Row mask selecting one table block, by start fraction (numeric) or stratum (label)."""
    if block_key == "start_frac":
        return np.isclose(frame["start_frac"], block_val)
    return frame[block_key] == block_val


def _cell_values(start_metrics, models, start_frac, channel, metric, detection, threshold_name=None,
                 block_key="start_frac", metric_thresholds=None):
    """(means, stds) over `models` for one column, or NaNs where a model has no rows."""
    # A metric may pin its own threshold (METRIC_THRESHOLD): the OT error is computed over
    # every peak while the counts beside it are only meaningful at the large-scale threshold,
    # so one table row legitimately mixes the two.
    threshold_name = _threshold_of(metric, threshold_name or PEAK_THRESHOLD, metric_thresholds)
    col = _metric_key(metric, detection)
    means, stds = [], []
    for model in models:
        sel = start_metrics[
            (start_metrics["model"] == model)
            & _block_mask(start_metrics, start_frac, block_key)
            & (start_metrics["threshold_name"] == threshold_name)
        ]
        if channel is not None:
            sel = sel[sel["channel"] == channel]
        else:
            # Global metrics do not depend on the channel; take one arbitrary channel rather
            # than averaging duplicates of the same number.
            first = sel[["channel"]].drop_duplicates().head(1)
            if len(first):
                sel = sel.merge(first, on="channel")
        means.append(float(sel[col].iloc[0]) if len(sel) else np.nan)
        stds.append(float(sel[f"{col}_std"].iloc[0]) if len(sel) else np.nan)
    return means, stds


def _check_available(start_metrics, models, start_fracs, channels, threshold_name=None):
    """Fail loudly instead of emitting a table full of '--'."""
    threshold_name = threshold_name or PEAK_THRESHOLD
    missing = []
    for start_frac in start_fracs:
        rows = start_metrics[np.isclose(start_metrics["start_frac"], start_frac)]
        if rows.empty:
            missing.append(f"start fraction {start_frac:g} (no rows at all)")
            continue
        for model in models:
            if rows[rows["model"] == model].empty:
                missing.append(f"{model} at f={start_frac:g}")
    have_channels = set(start_metrics["channel"].unique())
    for channel in channels:
        if channel not in have_channels:
            missing.append(f"channel {channel}")
    if threshold_name not in set(start_metrics["threshold_name"].unique()):
        missing.append(f"threshold {threshold_name}")
    if missing:
        raise ValueError(
            "The aggregated metrics do not cover everything the table asks for: "
            f"{sorted(set(missing))}. The per-slice parquet is probably stale; rerun with --force."
        )


def _detection_sentence(detection):
    return (
        r"Peaks are detected once over the full rollout and then assigned to the window "
        r"containing them, so prominences and widths are measured in their true context."
        if detection == "full_trace" else
        r"Peaks are detected inside each window independently."
    )


def _stacked_table(start_metrics, models, blocks, metrics, global_metrics, detection,
                   *, label, caption, environment="table", block_label, group_label,
                   font=r"\scriptsize", tabcolsep=3, threshold_name=None, block_key="start_frac",
                   group_titles=None, block_notes=None, metric_thresholds=None):
    """Shared builder for the stacked tables.

    `blocks` is a list of (block_key, block_title, start_frac, channels_for_columns): each
    becomes one horizontally repeated set of rows under a shared header. `group_label` names
    what the column groups are (a channel for the main table, a metric-free label otherwise).
    """
    n_metrics = len(metrics)
    _, _, columns_channels = blocks[0][1], blocks[0][2], blocks[0][3]
    n_groups = len(columns_channels)
    col_spec = "l" + "".join(
        ("c" * n_metrics) + ("@{\\hspace{6pt}}" if g < n_groups - 1 else "")
        for g in range(n_groups)
    ) + ("@{\\hspace{6pt}}" + "c" * len(global_metrics) if global_metrics else "")

    lines = [
        rf"\begin{{{environment}}}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        font,
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}pt}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    group_cells, cmidrules, col = [], [], 2
    for title in columns_channels:
        # group_titles lets a caller decorate the channel name, e.g. with its peak threshold.
        shown = (group_titles or {}).get(title, title)
        group_cells.append(rf"\multicolumn{{{n_metrics}}}{{c}}{{{shown}}}")
        cmidrules.append(rf"\cmidrule(lr){{{col}-{col + n_metrics - 1}}}")
        col += n_metrics
    if global_metrics:
        group_cells.append(rf"\multicolumn{{{len(global_metrics)}}}{{c}}{{{group_label}}}")
        cmidrules.append(rf"\cmidrule(lr){{{col}-{col + len(global_metrics) - 1}}}")
    lines.append("& " + " & ".join(group_cells) + r" \\")
    lines.append("".join(cmidrules))

    headers = ["Model"]
    for _ in columns_channels:
        headers += [METRIC_SPECS[m][0] for m in metrics]
    headers += [METRIC_SPECS[m][0] for m in global_metrics]
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")

    n_cols = 1 + n_metrics * n_groups + len(global_metrics)
    for block_i, (_, title, start_frac, channels) in enumerate(blocks):
        if block_i:
            lines.append(r"\addlinespace[2pt]")
            lines.append(r"\midrule")
        # block_notes appends a short annotation to the block title line rather than spending
        # a whole extra row on it (the real peak count is the same for every model).
        note = (block_notes or {}).get(title, "")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\itshape {title}{note}}} \\")
        lines.append(r"\addlinespace[1pt]")

        columns = [(c, m) for c in channels for m in metrics] + [(None, m) for m in global_metrics]
        cells = {}
        for channel, metric in columns:
            means, stds = _cell_values(start_metrics, models, start_frac, channel, metric, detection,
                                       threshold_name, block_key, metric_thresholds)
            _, decimals, _, scale = METRIC_SPECS[metric]
            marks = _rank_marks(means, metric)
            for i, model in enumerate(models):
                body = _fmt(means[i], stds[i], decimals, scale)
                if body != "--" and i in marks:
                    body = marks[i] % body
                cells[(model, channel, metric)] = f"${body}$" if body != "--" else body
        for model in models:
            lines.append(" & ".join([model] + [cells[(model, c, m)] for c, m in columns]) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", rf"\end{{{environment}}}"]
    return "\n".join(lines) + "\n"


def _filter_sentence() -> str:
    """Caption clause naming the pre-detection smoothing, empty when none is applied.

    Only the peak columns are affected. The Dice score is computed from the surrogate mode
    labels stored in the cache, which were produced by the classifier on the unfiltered
    signals, so it is identical to the unfiltered table by construction.
    """
    resolved = resolve_filter(SIGNAL_FILTER)
    per_channel = isinstance(SIGNAL_FILTER, dict) and "kind" not in SIGNAL_FILTER
    if resolved is None and not per_channel:
        return ""
    def pretty(spec, channel=None):
        r = resolve_filter(spec, channel)
        if r is None:
            return "unsmoothed"
        if r["kind"] == "gaussian":
            n = r["sigma"]
            return rf"Gaussian kernel, $\sigma={n:g}$ sample{'' if n == 1 else 's'}"
        return filter_label(spec, channel)

    if per_channel:
        applied = ", ".join(f"{c}: {pretty(SIGNAL_FILTER, c)}" for c in TABLE_CHANNELS)
    else:
        applied = pretty(SIGNAL_FILTER)
    return (
        r" Both the real and the generated trace are smoothed before peak detection "
        rf"({applied} at the 10\,kHz sample rate), so the peak columns count features "
        r"above the sensor noise rather than the noise itself; the smoothing is identical on "
        r"both sides. $\mathcal{D}$ is unaffected: it is computed from the surrogate mode "
        r"labels of the unfiltered signals."
    )


def render_latex(start_metrics: pd.DataFrame, label="tab:rollout_results", detection=None,
                 threshold_name=None) -> str:
    """Main table: one stacked block per start fraction, one column group per diagnostic."""
    detection = detection or PEAK_DETECTION
    threshold_name = threshold_name or PEAK_THRESHOLD
    models = list(MODELS)
    _check_available(start_metrics, models, START_FRACTIONS, TABLE_CHANNELS, threshold_name)
    threshold_sentence = (
        "" if threshold_name == PEAK_THRESHOLD else
        rf" Peaks are detected at the {threshold_name!r} prominence threshold rather than the "
        r"default, so counts are not comparable to the other tables."
    )
    caption = (
        r"Autoregressive rollout results. Each entry is the mean over test shots of the "
        r"per-shot mean over stochastic samples and rollout windows, with the subscript giving "
        r"the standard deviation across shots. $|\Delta N|$ is the absolute peak-count error "
        r"per window, and $W_1^{\pi}$ and $W_1^{w}$ are Wasserstein-1 distances between the "
        r"generated and real peak prominence and width distributions, widths in ms. "
        r"$\mathcal{D}$ is the micro-averaged Dice score between the generated and real "
        r"surrogate mode sequences. Lower is better except for $\mathcal{D}$. Best in bold, "
        r"second best underlined, within each start fraction. " + _detection_sentence(detection)
        + threshold_sentence + _filter_sentence()
    )
    blocks = [
        (f, rf"start fraction $f={f:g}$", f, list(TABLE_CHANNELS))
        for f in START_FRACTIONS
    ]
    return _stacked_table(
        start_metrics, models, blocks, TABLE_METRICS, TABLE_GLOBAL_METRICS, detection,
        label=label, caption=caption, block_label="start fraction", group_label="Modes",
        threshold_name=threshold_name,
    )


def _count_sentence(metrics):
    """Caption clause defining the peak-count column, matching whichever variant is shown."""
    if "pool_count_error" in metrics:
        return (
            r"$\Delta N$ is the signed peak-count error over the pool, so $\Delta N = -N$ "
            r"identifies a model that generated no peaks of that scale at all, and "
            r"$\mathcal{D}$ the micro-averaged Dice score between the generated and real "
            r"surrogate mode sequences; $\Delta N$ is ranked on its absolute value. "
        )
    return (
        r"$|\Delta N|$ is the absolute peak-count error over the pool, and $\mathcal{D}$ the "
        r"micro-averaged Dice score between the generated and real surrogate mode sequences. "
    )


def _threshold_of(metric, default_threshold, overrides=None):
    """The threshold name one column is computed at.

    `overrides` replaces METRIC_THRESHOLD wholesale for a table variant, so the same renderer
    can produce the mixed-threshold table and the single-threshold one without either of them
    reading a mapping the other set.
    """
    mapping = METRIC_THRESHOLD if overrides is None else overrides
    return mapping.get(metric, default_threshold)


def _threshold_order(metrics, default_threshold, overrides=None):
    """The distinct threshold names the block's columns use, in column order."""
    names = []
    for metric in metrics:
        name = _threshold_of(metric, default_threshold, overrides)
        if name not in names:
            names.append(name)
    return names


def _threshold_sentence(metrics, default_threshold, overrides=None):
    """Caption clause explaining the `pi >= a/b` annotation over each column group.

    The two numbers are not two variants of one measurement: they are the detection thresholds
    of two different columns, and a reader who takes the header as a single setting will
    misread the table. Which is which follows the column order, so it is derived here rather
    than written out, and stays correct if STRAT_TABLE_METRICS is reordered.
    """
    names = _threshold_order(metrics, default_threshold, overrides)
    if names == ["large_scale"]:
        # The single-threshold variant: every column sees the same peaks, so the header carries
        # one number per channel and the only thing to explain is what that number selects.
        return (
            r"Every column is computed on the same peaks: those above the per-channel "
            r"	exttt{find\_peaks} prominence given under each column group, on the "
            r"$[0,1]$-normalised signal, chosen to admit \acrshort{ELM}-scale events and reject "
            r"sensor noise. The two channels take different values because their normalised "
            r"amplitudes differ: \acrshort{ELM} bursts on \acrshort{PD} are far taller than the slower "
            r"\acrshort{DML} deflections. "
        )
    if len(names) != 2 or names != ["all_peaks", "large_scale"]:
        # Any other combination would need its own wording; say nothing rather than something
        # false, and let the header stand on its own.
        return ""
    return (
        r"Each column group is headed by the two \texttt{find\_peaks} prominence thresholds its "
        r"columns use, on the $[0,1]$-normalised signal, in column order as "
        r"$\pi{\geq}\pi_{\mathrm{OT}}/\pi_{N}$. The transport cost is computed over every peak "
        r"down to $\pi_{\mathrm{OT}}$, which sits at sensor-noise scale: including the noise "
        r"peaks is what lets a model be charged for the spurious mass it emits instead of "
        r"scoring well for emitting a lot of it. The count uses only the peaks above the "
        r"\acrshort{ELM}-scale $\pi_{N}$, where a count of peaks is a count of events rather than of "
        r"noise excursions. The two channels take different values because their normalised "
        r"amplitudes differ: \acrshort{ELM} bursts on \acrshort{PD} are far taller than the slower \acrshort{DML} "
        r"deflections."
    )


def _metric_threshold_annotations(slice_metrics, channels, metrics, default_threshold,
                                  overrides=None):
    """Channel -> header decorated with every prominence the block's columns actually use.

    A block whose columns mix thresholds (the OT error over all peaks, the count at the
    large-scale threshold) cannot carry a single number in its header, so both are listed in
    the order the METRIC_THRESHOLD names them. Values are read back from the parquet rather
    than from the config constants so the annotation cannot drift from what was computed.
    """
    names = _threshold_order(metrics, default_threshold, overrides)
    out = {}
    for channel in channels:
        parts = []
        for name in names:
            vals = slice_metrics[(slice_metrics["threshold_name"] == name)
                                 & (slice_metrics["channel"] == channel)]["threshold"].unique()
            if len(vals) == 1:
                parts.append(f"{float(vals[0]):g}")
        out[channel] = rf"{channel}\,{{\tiny$\pi{{\geq}}{'/'.join(parts)}$}}" if parts else channel
    return out


def _pool_stratum_notes(pool_stratum, channels, overrides=None):
    """Stratum title suffix carrying the real peak count of the pool, per channel.

    A property of the measured trace, so it is identical for every model and belongs in the
    block header rather than in a column repeated down the rows. It is the reference the count
    errors above it are relative to, and it makes the difficulty gradient between the strata
    visible instead of implicit. Counted at the same threshold the count columns use.
    """
    notes = {}
    for label, lo, hi in STRAT_BINS:
        parts = []
        for channel in channels:
            name = _threshold_of("pool_n_peaks_real", PEAK_THRESHOLD, overrides)
            sel = pool_stratum[(pool_stratum["stratum"] == label)
                               & (pool_stratum["channel"] == channel)
                               & (pool_stratum["threshold_name"] == name)]
            if len(sel):
                parts.append(rf"{channel} ${float(sel['pool_n_peaks_real'].mean()):.1f}$")
        title = _stratum_title(label, lo, hi)
        notes[title] = (r", real peaks/pool: " + ", ".join(parts)) if parts else ""
    return notes


def _stratum_title(label, lo, hi):
    return rf"{label} half, rollout windows ${lo}$--${hi}$"


def render_stratified_latex(stratum_metrics: pd.DataFrame, slice_metrics: pd.DataFrame,
                            label="tab:rollout_depth", detection=None, threshold_name=None,
                            metrics=None, models=None, environment="table",
                            font=r"\scriptsize", tabcolsep=3, lam=None,
                            metric_thresholds=None) -> str:
    """Main depth table: one stacked block per depth stratum at a fixed start fraction.

    `stratum_metrics` must already carry the pooled columns for `lam` (see `merge_pool_metrics`);
    `lam` only names the value in the caption, it does not select anything here.
    """
    detection = detection or PEAK_DETECTION
    threshold_name = threshold_name or PEAK_THRESHOLD
    lam = OT_LAMBDA_MS if lam is None else float(lam)
    metrics = list(metrics or STRAT_TABLE_METRICS)
    models = list(models or MODELS)
    channels = list(TABLE_CHANNELS)
    n_windows = STRAT_BINS[-1][2] + 1
    pool_windows = STRAT_BINS[0][2] - STRAT_BINS[0][1] + 1
    caption = (
        r"Autoregressive rollout results by rollout depth. All rollouts start from the same "
        rf"point in the discharge ($f={STRAT_START_FRACTION:g}$) and are split into equal "
        rf"halves of the first {n_windows} generated windows, so the two blocks differ only in "
        r"how long the model has been feeding on its own output and not in where in the shot "
        r"they sit, which is what varying the start point would confound them with. "
        rf"Peaks are pooled over all {pool_windows} generated windows of a half rather than cut "
        r"at the generation-window boundaries, and each peak enters as its prominence placed at "
        r"its time within the pool. $E_{\mathrm{OT}}$ is the unbalanced optimal-transport cost "
        r"between the generated and the real peaks: each peak enters with its prominence as "
        r"mass; matching costs mass times time separation, and mass left unmatched on either "
        rf"side costs $\lambda={lam:g}$\,ms per unit, so peaks pair only when they are within "
        r"$2\lambda$, and a missed \acrshort{ELM} is charged $\lambda$ times its full prominence. "
        r"$E_{0}$ normalises this by the cost of predicting no peaks at all, so "
        r"$E_{\mathrm{OT}}/E_{0}=1$ is exactly as good as an empty trace and anything above it "
        r"is worse. "
        + _count_sentence(metrics) +
        _threshold_sentence(metrics, threshold_name, metric_thresholds) +
        r" The real peak count per pool is given beside each block title, since the later "
        r"stratum carries denser \acrshort{ELM}s. Lower is better except for $\mathcal{D}$. Best in "
        r"bold, second best underlined, within each block."
        + _filter_sentence()
    )
    blocks = [(lab, _stratum_title(lab, lo, hi), lab, list(channels)) for lab, lo, hi in STRAT_BINS]
    return _stacked_table(
        stratum_metrics, models, blocks, metrics, TABLE_GLOBAL_METRICS, detection,
        label=label, caption=caption, environment=environment, block_label="stratum",
        group_label="Modes", font=font, tabcolsep=tabcolsep, threshold_name=threshold_name,
        block_key="stratum",
        metric_thresholds=metric_thresholds,
        group_titles=_metric_threshold_annotations(slice_metrics, channels, metrics,
                                                   threshold_name, metric_thresholds),
        block_notes=_pool_stratum_notes(stratum_metrics, channels, metric_thresholds),
    )


def render_appendix_latex(start_metrics: pd.DataFrame, channel_names,
                          label="tab:rollout_results_appendix", detection=None,
                          threshold_name=None) -> str:
    """Appendix table: one stacked block per diagnostic, every model, one start fraction.

    Wider than the main table (signed as well as absolute count error, and the generated and
    real peak rates in physical units) and over every observable channel, so the diagnostic
    becomes the outer row level. Wrapped in a sideways environment because it does not fit the
    text width; the document needs the rotating package.
    """
    detection = detection or PEAK_DETECTION
    threshold_name = threshold_name or PEAK_THRESHOLD
    models = list(all_models())
    channels = list(APPENDIX_CHANNELS or channel_names)
    _check_available(start_metrics, models, [APPENDIX_START_FRACTION], channels, threshold_name)
    caption = (
        r"Per-diagnostic autoregressive rollout results at start fraction "
        rf"$f={APPENDIX_START_FRACTION:g}$, including the prior and attention ablations. "
        r"$\hat r$ and $r$ are the generated and real peak rates per millisecond, $\Delta N$ "
        r"and $|\Delta N|$ the signed and absolute peak-count error per window, and "
        r"$W_1^{\pi}$, $W_1^{w}$ the Wasserstein-1 distances between the generated and real "
        r"peak prominence and width distributions, widths in ms. $\mathcal{D}$ is the "
        r"micro-averaged Dice score between the generated and real surrogate mode sequences "
        r"and does not depend on the diagnostic, so it repeats down the blocks. Entries are "
        r"means over test shots with the subscript giving the standard deviation across "
        r"shots; $\hat r$ and $r$ are descriptive and are not ranked, $\Delta N$ is ranked on "
        r"its magnitude. " + _detection_sentence(detection) + _filter_sentence()
    )
    # One column group per block, so the group header carries the diagnostic name and the
    # block header repeats it for readers scanning the row labels.
    blocks = [
        (channel, rf"{channel}", APPENDIX_START_FRACTION, [channel])
        for channel in channels
    ]
    return _stacked_table(
        start_metrics, models, blocks, APPENDIX_METRICS, APPENDIX_GLOBAL_METRICS, detection,
        label=label, caption=caption, environment="sidewaystable",
        block_label="diagnostic", group_label="Modes", font=r"\footnotesize", tabcolsep=4,
        threshold_name=threshold_name,
    )


def compare_detection(start_metrics: pd.DataFrame) -> str:
    """Side-by-side of the two detection variants, for deciding which one the paper uses.

    Detecting inside each slice truncates the prominence and width walks at the slice
    boundaries; detecting on the full rollout does not. The difference is not expected to
    cancel between models, so this is a decision to make on the numbers rather than a silent
    substitution.
    """
    metrics = ["n_peaks_gen", "n_peaks_real", "abs_count_error", "miss_rate",
               "prominence_w1", "width_w1_ms"]
    keep = ["model", "start_frac", "channel", "threshold_name"]
    rows = []
    for _, r in start_metrics.iterrows():
        row = {k: r[k] for k in keep}
        for m in metrics:
            full, cut = r[m], r[f"{m}_cut"]
            row[f"{m}|full"] = full
            row[f"{m}|cut"] = cut
            row[f"{m}|d%"] = (100.0 * (cut - full) / full) if full not in (0, None) and not pd.isna(full) and full != 0 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(keep)
    with pd.option_context("display.width", 260, "display.max_rows", 300, "display.max_columns", 40):
        return out.round(4).to_string(index=False)


# %% ---------------------------------------------------------------------------------
# Depth plots
# ------------------------------------------------------------------------------------


def aggregate_by_depth(slice_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to (model, start fraction, channel, threshold, depth k).

    Depth is aggregated *last*: within one rollout a depth-k window is a single slice, so the
    ladder is stochastic samples first, then shots, with k carried through untouched. Averaging
    over depth (as the tables do) would destroy exactly the compounding this is meant to show.

    The returned spread is the standard deviation over shots at that depth, and `n_shots`
    records how many shots still reach it: shots end at different times, so the sample shrinks
    with depth and the band widens partly for that reason.
    """
    per_trajectory = (
        slice_metrics
        .groupby(TRAJECTORY_KEYS + CONDITION_KEYS + ["k"], as_index=False)
        .agg(**{m: (m, "mean") for m in SLICE_METRICS})
    )
    agg = {m: (m, "mean") for m in SLICE_METRICS}
    agg |= {f"{m}_std": (m, "std") for m in SLICE_METRICS}
    return (
        per_trajectory
        .groupby(["model", "start_frac"] + CONDITION_KEYS + ["k"], as_index=False)
        .agg(**agg, n_shots=("shot", "nunique"))
    )


def _depth_offsets(slice_metrics: pd.DataFrame, seq_length: int) -> dict:
    """Start fraction -> how many windows into the shot its depth 0 sits, on average.

    A rollout from f=0.75 begins three quarters of the way through the shot, so plotting its
    k against the k of an f=0.05 rollout on a bare depth axis would overlay two different
    parts of the discharge. Shifting each start fraction by its mean start index, in windows,
    puts both on a common "window index from the start of the shot" axis. Shots differ in
    length, so this is an average offset, not an exact alignment.
    """
    starts = slice_metrics[["start_frac", "shot", "start_idx"]].drop_duplicates()
    return {
        float(f): int(round(g["start_idx"].mean() / seq_length))
        for f, g in starts.groupby("start_frac")
    }


def plot_depth_curves(depth_metrics: pd.DataFrame, slice_metrics, seq_length, pdf_dir,
                      detection=None, channel=None, min_shots=None, threshold_name=None):
    """One PDF per metric per size: compound error against window index.

    Colour is the model, line style is the start fraction, and the band is +/- 1 sd over the
    shots contributing at that depth.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    detection = detection or PEAK_DETECTION
    threshold_name = threshold_name or PEAK_THRESHOLD
    channel = channel or DEPTH_PLOT_CHANNEL
    min_shots = DEPTH_MIN_SHOTS if min_shots is None else min_shots
    pdf_dir = Path(pdf_dir)
    models = list(MODELS)
    offsets = _depth_offsets(slice_metrics, seq_length)
    fracs = [f for f in DEPTH_PLOT_START_FRACTIONS if f in offsets]
    written = []

    rc = {
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "lines.linewidth": 1.1, "figure.dpi": 120,
        "axes.spines.top": False, "axes.spines.right": False,
    }
    for metric in DEPTH_PLOT_METRICS:
        col = _metric_key(metric, detection)
        ylabel, _, _, _ = METRIC_SPECS[metric]
        # Dice is channel-independent; pick one channel so the rows are not duplicated.
        sel_channel = channel
        for (fig_w, fig_h) in DEPTH_PLOT_SIZES:
            with plt.rc_context(rc):
                fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
                for model, color in zip(models, DEPTH_PLOT_COLORS):
                    for frac, style in zip(fracs, DEPTH_PLOT_STYLES):
                        d = depth_metrics[
                            (depth_metrics["model"] == model)
                            & np.isclose(depth_metrics["start_frac"], frac)
                            & (depth_metrics["channel"] == sel_channel)
                            & (depth_metrics["threshold_name"] == threshold_name)
                            & (depth_metrics["n_shots"] >= min_shots)
                        ].sort_values("k")
                        if d.empty:
                            continue
                        x = d["k"].to_numpy() + offsets[frac]
                        y = d[col].to_numpy()
                        sd = d[f"{col}_std"].to_numpy()
                        ax.plot(x, y, color=color, linestyle=style)
                        ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.12, linewidth=0)
                ax.set_xlabel("window index from shot start")
                ax.set_ylabel(ylabel)
                ax.set_xlim(left=0)
                ax.grid(color="0.9", linewidth=0.6)
                ax.set_axisbelow(True)

                # Two-part legend in one box: colour carries the model, style the start
                # fraction, so identity is never colour alone.
                handles = [Line2D([], [], color=c, lw=1.4, label=m)
                           for m, c in zip(models, DEPTH_PLOT_COLORS)]
                handles += [Line2D([], [], color="0.35", lw=1.2, linestyle=st, label=f"$f={fr:g}$")
                            for fr, st in zip(fracs, DEPTH_PLOT_STYLES)]
                ax.legend(handles=handles, frameon=False, ncol=2, handlelength=1.3,
                          columnspacing=0.9, handletextpad=0.5, labelspacing=0.25,
                          borderaxespad=0.2, loc="best")

                out = pdf_dir / f"{fig_w:g}x{fig_h:g}" / f"depth_{sel_channel}_{metric}.pdf"
                out.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out, bbox_inches="tight")
                plt.close(fig)
                written.append(out)
    return written


# %% ---------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------------


def compute_slice_metrics(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [f"{n} ({s})" for n, s in all_models().items() if not (get_cache_dir() / f"{s}.h5").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing rollout caches in {get_cache_dir().resolve()}: {missing}. "
            "Fetch them with: rsync -vz snellius:/scratch-shared/mtresoor/final_cache/<name>.h5 "
            f"{get_cache_dir()}/"
        )

    ref_cache = RolloutHDFCache(all_models()[REFERENCE_MODEL], mode="r")
    ref_cfg = cache_config(ref_cache)
    C, data_module = build_data_module(ref_cfg)
    channel_names = [str(c) for c in data_module.cols.x]
    thresholds = resolve_thresholds(ref_cfg)
    sample_rate = float(C.data.sample_rate)
    seq_length = int(C.data.seq_length)
    # The appendix table covers every diagnostic, so peaks are computed for all of them and
    # the main table just reads the subset it needs.
    channels = list(APPENDIX_CHANNELS or channel_names)
    for channel in set(channels) | set(TABLE_CHANNELS):
        if channel not in channel_names:
            raise ValueError(f"Channel {channel} requested but the caches only have {channel_names}")
    channels = [c for c in channel_names if c in set(channels) | set(TABLE_CHANNELS)]
    global CHANNEL_ORDER
    CHANNEL_ORDER = list(channel_names)
    print(f"channels={channels}  thresholds={thresholds}  sample_rate={sample_rate}")

    frames, pool_frames, audit_rows = [], [], []
    for name, cache_stem in all_models().items():
        cache = RolloutHDFCache(cache_stem, mode="r")
        cfg = cache_config(cache)
        verify_compatibility(name, cache_stem, cfg, ref_cfg, channel_names, thresholds, sample_rate)

        keys = cache.list_rollouts()
        shots = sorted({k[0] for k in keys})
        if args.shots:
            shots = shots[: args.shots]
        samples_per = {}
        for shot, start_idx, sample_idx in keys:
            samples_per.setdefault((shot, start_idx), set()).add(sample_idx)
        counts = [len(v) for v in samples_per.values()]
        audit_rows.append({
            "model": name, "cache": cache_stem, "n_shots": len(shots),
            "n_trajectories": len(samples_per), "n_generated": len(keys),
            "min_samples": min(counts), "max_samples": max(counts),
        })
        print(f"[{name}] {len(shots)} shots, {len(keys)} generated rollouts, "
              f"{min(counts)}-{max(counts)} samples per trajectory")

        # One shot at a time: the generated arrays are large and this keeps peak memory flat.
        for i, shot in enumerate(shots, start=1):
            shot_keys = [k for k in keys if k[0] == shot]
            step = int(cache.get_rollout(*shot_keys[0])["step"])
            results = load_results_from_cache(cache, shots=[shot], max_samples=MAX_SAMPLES)
            records = build_rollout_records(
                results, data_module, step=step, shots=[shot], max_samples=MAX_SAMPLES
            )
            records = [r for r in records
                       if any(np.isclose(r["start_frac"], f)
                              for f in slice_signature()["start_fractions"])]
            rows, pool_rows = [], []
            for record in records:
                rows.extend(record_slice_rows(
                    record, channel_names, channels, thresholds, sample_rate, seq_length
                ))
                # Pooled rows exist only at STRAT_START_FRACTION; the call is a no-op elsewhere.
                pool_rows.extend(record_pool_rows(
                    record, channel_names, channels, thresholds, sample_rate, seq_length
                ))
            for target, batch in ((frames, rows), (pool_frames, pool_rows)):
                if batch:
                    frame = pd.DataFrame(batch)
                    frame.insert(0, "model", name)
                    target.append(frame)
            if i % 10 == 0 or i == len(shots):
                print(f"  {i}/{len(shots)} shots")

    slice_metrics = pd.concat(frames, ignore_index=True)
    pool_metrics = pd.concat(pool_frames, ignore_index=True)
    return slice_metrics, pool_metrics, pd.DataFrame(audit_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute even if the slice parquet exists")
    parser.add_argument("--shots", type=int, default=None, help="use only the first N shots (smoke test)")
    parser.add_argument("--label", default="tab:rollout_results", help="LaTeX label for the table")
    parser.add_argument("--detection", choices=["full_trace", "per_slice"], default=None,
                        help=f"override PEAK_DETECTION (default {PEAK_DETECTION})")
    parser.add_argument("--min-shots", type=int, default=DEPTH_MIN_SHOTS,
                        help="drop depths reached by fewer shots than this in the depth plots "
                             f"(default {DEPTH_MIN_SHOTS}); lower it for --shots smoke runs")
    parser.add_argument("--compare-detection", action="store_true",
                        help="print the full-trace and per-slice numbers side by side and exit "
                             "without choosing between them")
    parser.add_argument("--threshold", choices=["all_peaks", "elm_scale", "large_scale"], default=None,
                        help=f"override PEAK_THRESHOLD (default {PEAK_THRESHOLD}) for the tables, "
                             "the report and the depth plots; non-default values write to "
                             "suffixed filenames instead of overwriting the main outputs")
    parser.add_argument("--models", choices=["all", "main"], default="all",
                        help="'main' drops the appendix-only ablation caches, which are the "
                             "slowest to read and are only needed by the appendix tables; the "
                             "main and depth tables are unaffected")
    parser.add_argument("--filter", default=None,
                        help="pre-detection smoothing, overriding SIGNAL_FILTER: 'none', "
                             "'gaussian' (sigma=3 samples), 'gaussian:1.5', 'median:5'. Peaks "
                             "are detected on the filtered trace for both the real and the "
                             "generated signal; filtered runs write to their own parquet and "
                             "to suffixed table filenames")
    args = parser.parse_args()

    global SIGNAL_FILTER, SLICE_PARQUET, SLICE_META, POOL_PARQUET, INCLUDE_APPENDIX_MODELS
    INCLUDE_APPENDIX_MODELS = args.models == "all"
    if not INCLUDE_APPENDIX_MODELS:
        print(f"--models main: skipping {list(APPENDIX_EXTRA_MODELS)}")
    if args.filter is not None:
        kind, _, param = args.filter.partition(":")
        SIGNAL_FILTER = {"kind": kind} if kind != "none" else None
        if param:
            # The one tunable parameter each filter has; named so the spec stays canonical.
            SIGNAL_FILTER[{"gaussian": "sigma", "median": "size"}[kind]] = float(param)
    # A filtered run measures different peaks, so it gets its own parquet rather than
    # invalidating the unfiltered one every time the two are compared.
    filter_tag = "" if resolve_filter(SIGNAL_FILTER) is None else "_" + filter_label(SIGNAL_FILTER).replace(
        "(", "").replace(")", "").replace("=", "").replace(",", "-").replace(".", "p")
    if filter_tag:
        SLICE_PARQUET = OUTPUT_DIR / f"rollout_slice_metrics{filter_tag}.parquet"
        SLICE_META = SLICE_PARQUET.with_suffix(".meta.json")
        POOL_PARQUET = OUTPUT_DIR / f"rollout_pool_metrics{filter_tag}.parquet"
        print(f"signal filter: {filter_label(SIGNAL_FILTER)} -> {SLICE_PARQUET.name}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signature = slice_signature()
    audit, reuse = None, False
    if (SLICE_PARQUET.exists() and POOL_PARQUET.exists() and SLICE_META.exists()
            and not args.force and not args.shots):
        cached = json.loads(SLICE_META.read_text())
        # Compare the signature after a JSON round-trip, not as live Python. Tuples in the
        # signature (the model pairs, the threshold overrides) come back as lists, so a direct
        # comparison reports every run as stale and silently recomputes the whole parquet.
        canonical = json.loads(json.dumps(signature))
        stale = [k for k, v in canonical.items() if cached.get(k) != v]
        if stale:
            # Reusing a parquet computed for a different model list, start fraction set or
            # threshold silently produces rows of "--" for whatever was never computed.
            print(f"{SLICE_PARQUET} was computed for different settings ({stale}); recomputing")
        else:
            print(f"reusing {SLICE_PARQUET} (pass --force to recompute)")
            slice_metrics = pd.read_parquet(SLICE_PARQUET)
            pool_metrics = pd.read_parquet(POOL_PARQUET)
            reuse = True
    if not reuse:
        slice_metrics, pool_metrics, audit = compute_slice_metrics(args)
        if not args.shots:
            slice_metrics.to_parquet(SLICE_PARQUET)
            pool_metrics.to_parquet(POOL_PARQUET)
            SLICE_META.write_text(json.dumps(signature, indent=2))

    if audit is not None:
        audit.to_csv(OUTPUT_DIR / "cache_audit.csv", index=False)
        print(audit.to_string(index=False))

    sample, trajectory, start = aggregate(slice_metrics)
    trajectory.to_csv(OUTPUT_DIR / "rollout_trajectory_metrics.csv", index=False)
    start.to_csv(OUTPUT_DIR / "rollout_start_metrics.csv", index=False)

    if args.compare_detection:
        print(compare_detection(start))
        return

    detection = args.detection or PEAK_DETECTION
    threshold_name = args.threshold or PEAK_THRESHOLD
    # Non-default thresholds get their own filenames so a --threshold run never clobbers the
    # main paper outputs; suffix is empty (and label untouched) at the default threshold.
    suffix = ("" if threshold_name == PEAK_THRESHOLD else f"_{threshold_name}") + filter_tag
    label = args.label if not suffix else f"{args.label}{suffix}"

    tex = render_latex(start, label=label, detection=detection, threshold_name=threshold_name)
    tex_path = OUTPUT_DIR / f"rollout_results_table{suffix}.tex"
    tex_path.write_text(tex)

    present = list(start["channel"].unique())
    # Keep the cache's column order rather than pandas' order of appearance. CHANNEL_ORDER is
    # empty when the parquet was reused without opening a cache, in which case fall back to
    # the frame's own order.
    channel_names = [c for c in CHANNEL_ORDER if c in present] or present
    appendix = render_appendix_latex(
        start, channel_names, label=f"{label}_appendix", detection=detection,
        threshold_name=threshold_name,
    )
    appendix_path = OUTPUT_DIR / f"rollout_results_appendix_table{suffix}.tex"
    appendix_path.write_text(appendix)

    # Depth-stratified tables: main (absolute count error) and appendix (relative, plus the
    # ablation rows), both at the fixed STRAT_START_FRACTION.
    stratum_slices = aggregate_by_stratum(slice_metrics)
    pool_stratum = aggregate_pools(pool_metrics)
    pool_stratum.to_csv(OUTPUT_DIR / f"rollout_pool_metrics{suffix}.csv", index=False)

    # One depth table per lambda in the parquet. The reported lambda keeps the unsuffixed
    # filename and label so the paper's \input path never moves; every other lambda is a
    # sensitivity variant written beside it. All of them read the same stored peaks, so this
    # costs nothing beyond the rendering.
    lambdas = sorted(pool_metrics["lam_ms"].unique(), key=lambda x: (not np.isclose(x, OT_LAMBDA_MS), x))
    strat_paths = []
    for lam in lambdas:
        lam_tag = "" if np.isclose(lam, OT_LAMBDA_MS) else f"_lam{lam:g}"
        stratum = merge_pool_metrics(stratum_slices, pool_stratum, lam=lam)
        stratum.to_csv(OUTPUT_DIR / f"rollout_stratum_metrics{suffix}{lam_tag}.csv", index=False)
        for var_tag, overrides in STRAT_TABLE_VARIANTS:
            tag = f"{suffix}{var_tag}{lam_tag}"
            lab = f"{label}_depth{var_tag}{lam_tag}"
            strat_tex = render_stratified_latex(
                stratum, slice_metrics, label=lab, detection=detection,
                threshold_name=threshold_name, lam=lam, metric_thresholds=overrides)
            strat_path = OUTPUT_DIR / f"rollout_depth_table{tag}.tex"
            strat_path.write_text(strat_tex)
            strat_paths.append(strat_path)
            strat_appendix = render_stratified_latex(
                stratum, slice_metrics, label=f"{lab}_appendix", detection=detection,
                threshold_name=threshold_name, lam=lam, metric_thresholds=overrides,
                metrics=["ot_error", "ot_error_rel", "ot_missed_frac", "ot_false_frac",
                         "pool_count_error", "pool_abs_count_error"],
                models=list(all_models()), font=r"\footnotesize", tabcolsep=4)
            strat_appendix_path = OUTPUT_DIR / f"rollout_depth_appendix_table{tag}.tex"
            strat_appendix_path.write_text(strat_appendix)
            strat_paths.append(strat_appendix_path)
    # The reported lambda is first in `lambdas`, so this is the frame the report below prints.
    stratum = merge_pool_metrics(stratum_slices, pool_stratum, lam=OT_LAMBDA_MS)

    depth = aggregate_by_depth(slice_metrics)
    depth.to_csv(OUTPUT_DIR / f"rollout_depth_metrics{suffix}.csv", index=False)
    # Stamped on every row so it survives a parquet reuse without opening a cache.
    seq_length = int(slice_metrics["slice_length"].iloc[0])
    plots = plot_depth_curves(depth, slice_metrics, seq_length, OUTPUT_DIR / f"depth{suffix}",
                              detection=detection, min_shots=args.min_shots,
                              threshold_name=threshold_name)
    print(f"wrote {len(plots)} depth plots under {(OUTPUT_DIR / f'depth{suffix}').resolve()}")

    print(f"\n=== start-fraction report (peak detection: {detection}, threshold: {threshold_name}) ===")
    sfx = "" if detection == "full_trace" else "_cut"
    start = start[start["threshold_name"] == threshold_name]
    show = ["model", "start_frac", "channel"] + [
        f"{m}{sfx}" for m in ("rate_gen_per_ms", "rate_real_per_ms", "count_error",
                              "abs_count_error", "miss_rate", "prominence_w1", "width_w1_ms")
    ] + ["dice", "n_trajectories"]
    with pd.option_context("display.width", 220, "display.max_rows", 400):
        print(start[show].sort_values(["start_frac", "channel", "model"]).round(4).to_string(index=False))
    print(f"\n=== pooled OT report (reported lambda={OT_LAMBDA_MS:g} ms, "
          f"computed {OT_LAMBDAS_MS}, mass={OT_PEAK_MASS}) ===")
    pool_show = pool_stratum[pool_stratum["channel"].isin(TABLE_CHANNELS)]
    cols = ["model", "stratum", "channel", "threshold_name", "lam_ms", "ot_error", "ot_error_rel",
            "ot_loc", "ot_missed_frac", "ot_false_frac", "pool_n_peaks_gen",
            "pool_n_peaks_real", "pool_abs_count_error", "n_trajectories"]
    with pd.option_context("display.width", 240, "display.max_rows", 600):
        print(pool_show[cols].sort_values(["lam_ms", "channel", "threshold_name", "stratum", "model"])
              .round(4).to_string(index=False))

    print(f"\nwrote {tex_path}\nwrote {appendix_path}")
    for path in strat_paths:
        print(f"wrote {path}")
    print((OUTPUT_DIR / f"rollout_depth_table{suffix}.tex").read_text())


if __name__ == "__main__":
    main()
