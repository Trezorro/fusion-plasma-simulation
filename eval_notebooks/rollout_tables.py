"""Regenerate the rollout peak/mode result tables from scratch.

Run it as a script from the repo root:

    PYTHONPATH=. python eval_notebooks/rollout_tables.py
    PYTHONPATH=. python eval_notebooks/rollout_tables.py --force        # ignore the intermediate parquet
    PYTHONPATH=. python eval_notebooks/rollout_tables.py --shots 3      # quick smoke run
    PYTHONPATH=. python eval_notebooks/rollout_tables.py --threshold large_scale  # coarser peaks

Given the MODELS mapping below (display name -> rollout cache stem) it reads every cache,
recomputes every per-window statistic, aggregates them along one explicit ladder, and writes
the CSVs plus the LaTeX tables and the depth plots. Nothing is read back from a previous run
except the intermediate parquets, which are a pure function of (cache, data module,
thresholds) and are rewritten whenever --force is passed or the signature below changes.

This file is the configuration and the paper's captions. The work happens in:

    src/rollout_cache.py                read a cache's stamped config, rebuild the data module
    src/metrics/rollout_peaks.py        per-window and pooled peak statistics
    src/metrics/rollout_aggregate.py    the aggregation ladder
    src/plotters/latex_tables.py        stacked LaTeX table assembly
    src/plotters/rollout_depth.py       depth curves

What is computed, per rollout window (one autoregressive generation, `seq_length` samples):

  |dN|      absolute peak-count error between the generated and the real window
  W1(pi)    Wasserstein-1 distance between the generated and real peak *prominence*
            distributions, in normalized [0,1] signal units
  W1(w)     the same for peak *widths*, converted from samples to milliseconds. Widths are
            full width at half maximum and capped at one generation window
  miss      1 if the real window has peaks and the generated window has none
  Dice      micro-averaged Dice between the generated and real surrogate mode label
            sequences over the window (equals mode-sequence accuracy); higher is better

and, per depth stratum, the unbalanced optimal-transport peak error E_OT over the pooled peaks
of that stratum (see src/metrics/ot_peak_error.py).

Both Wasserstein columns are reported over the windows where *both* peak sets are non-empty,
and peaks are detected once over the whole rollout rather than inside each window; see
`src/metrics/rollout_peaks.py` for why both of those matter and what the alternatives do.

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

from src.hdf_cache import RolloutHDFCache, get_cache_dir
from src.metrics.rollout_aggregate import (aggregate, aggregate_by_depth, aggregate_by_stratum,
                                           aggregate_pools, long_shots, merge_pool_metrics,
                                           pool_bin_set_shots)
from src.metrics.rollout_peaks import PeakSpec, record_pool_rows, record_slice_rows
from src.plotters.latex_tables import (METRIC_SPECS, check_available, stacked_table,
                                       threshold_of, threshold_order)
from src.plotters.rollout_depth import plot_depth_curves
from src.rollout import build_rollout_records, load_results_from_cache
from src.rollout_cache import (LARGE_PEAK_PROMINENCE_DEFAULT, LARGE_PEAK_PROMINENCE_OVERRIDES,
                               build_data_module, cache_config, resolve_thresholds,
                               verify_compatibility)

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
# the flow model. All no-leak, so they stay comparable to the main table's rows.
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

# --- depth stratification ------------------------------------------------------------------
# Varying the start fraction confounds two things: how deep the rollout has run, and where in
# the discharge it sits. The stratified table fixes the start point and splits by rollout depth
# instead, so the only difference between the blocks is how long the model has been feeding on
# its own output.
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
#
# A bin set is only usable on shots whose rollout reaches its deepest k, so a deeper set means
# a smaller shot subset and the two choices are one decision, not two: STRAT_SHOT_SUBSETS
# below pairs each subset with the bin set its window budget affords. The halves stay equal in
# every set, because both E_OT and |dN| grow with the pool span and blocks of unequal span
# would not be comparable.
#
# "": every test shot supports 12 windows from f=0.50, so two halves of 6.
# "long22": the 34 shots that reach 22 windows, so two halves of 11, which is the deepest
#   equal split those shots afford (23 windows would cost two more shots for one more window).
STRAT_BIN_SETS = OrderedDict([
    ("", [("early", 0, 5), ("late", 6, 11)]),
    ("long22", [("early", 0, 10), ("late", 11, 21)]),
])
STRAT_BINS = STRAT_BIN_SETS[""]


def bins_of(bin_set: str):
    return STRAT_BIN_SETS[bin_set]


def bin_set_windows(bin_set: str) -> int:
    """Rollout windows a shot must support to contribute every stratum of `bin_set`."""
    return max(hi for _, _, hi in bins_of(bin_set)) + 1


# --- pooled optimal-transport peak error ---------------------------------------------------
# The depth table scores peaks with an unbalanced optimal-transport cost instead of a
# Wasserstein distance between prominence distributions; see src/metrics/ot_peak_error.py for
# the formulation and src/metrics/rollout_peaks.py for why the peaks are pooled over a stratum
# rather than cut at the generation-window boundaries.
#
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

# Which threshold each metric is computed at. Every reported column now sits at the low
# all-peaks threshold. The OT error needs it: pricing spurious mass is half of what it
# measures, and a high threshold would hide exactly the peaks it is meant to charge for. The
# count follows it so that the two columns of a group describe the same peak population and
# the header carries one number, at the cost of counting noise excursions as well as events.
# What the count would have said at ELM scale is instead carried by the reference peak rates
# in the block title, which are given at both thresholds.
METRIC_THRESHOLD = {
    "ot_error": "all_peaks",
    "ot_error_rel": "all_peaks",
    "ot_missed_frac": "all_peaks",
    "ot_false_frac": "all_peaks",
    "pool_count_error": "all_peaks",
    "pool_abs_count_error": "all_peaks",
    "pool_n_peaks_real": "all_peaks",
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

# Shot-subset variants of the depth table: (filename/label tag, STRAT_BIN_SETS key). Each
# subset keeps the shots whose rollout supports its bin set and drops the rest, so a deeper
# split and a smaller shot set are the same choice made once. The default set keeps all 43
# shots at depth 12; "long22" keeps the 34 that reach depth 22 and spends the extra room on
# deeper pools rather than on a longer table. Rendered from the same parquet.
STRAT_SHOT_SUBSETS = [("", ""), ("_long22", "long22")]

# Metrics of the depth table, which is the pooled table. Two columns per channel: the absolute
# transport cost and the absolute peak-count error. E_OT/E_0 is dropped from the reported
# table: at the all-peaks threshold no model comes near the empty-trace cost, so the
# normalisation only compresses the spread into a third column saying what E_OT already says.
# The signed count error is likewise not needed here: it exists to expose a model that emits
# nothing at all, which cannot happen at noise-scale detection. Both remain in the CSVs.
STRAT_TABLE_METRICS = ["ot_error", "pool_abs_count_error"]

# Peak prominence threshold used everywhere in the reported tables. Resolved from the
# reference cache's config: "all_peaks" -> evaluation.peaks.prominence,
# "elm_scale" -> evaluation.peaks.elm_pd_prominence. Both are still computed and written to
# the parquet and the CSVs so an audit can compare them.
PEAK_THRESHOLD = "all_peaks"

# --- main table ----------------------------------------------------------------------------
# One column group per channel, in this order.
TABLE_CHANNELS = ["PD", "DML"]

# Metrics shown per column group, in order. Keys index METRIC_SPECS. "miss_rate" is available
# but off by default: at the all-peaks threshold no model misses a window except DLinear on
# DML, where the other columns already say the same thing.
TABLE_METRICS = ["abs_count_error", "prominence_w1", "width_w1_ms"]

# Metrics shown once per model (not per channel), appended after the column groups.
TABLE_GLOBAL_METRICS = ["dice"]

# --- appendix table ------------------------------------------------------------------------
# One stacked block per diagnostic, model rows inside it, at a single start fraction.
# None = every observable channel in the cache.
APPENDIX_CHANNELS = None
APPENDIX_START_FRACTION = 0.05
APPENDIX_METRICS = [
    "rate_gen_per_ms", "rate_real_per_ms", "count_error", "abs_count_error",
    "prominence_w1", "width_w1_ms",
]
APPENDIX_GLOBAL_METRICS = ["dice"]

# --- depth plots ---------------------------------------------------------------------------
DEPTH_PLOT_CHANNEL = "PD"
DEPTH_PLOT_METRICS = ["abs_count_error", "prominence_w1", "width_w1_ms", "dice"]
DEPTH_PLOT_START_FRACTIONS = [0.05, 0.75]
# Figure sizes in inches, exported once each so fonts scale relative to the figure, matching
# the other paper exports (src/plotters/rollout_horizon.py).
DEPTH_PLOT_SIZES = ((3.3, 2.1), (4.6, 2.8), (6.5, 3.6))
DEPTH_MIN_SHOTS = 5

# --- measurement ---------------------------------------------------------------------------
# How a peak's width is measured; see PeakSpec.rel_height for what the scipy default does to a
# whole-rollout trace. WIDTH_CLIP_SAMPLES additionally caps a width at the generation-window
# length, on the grounds that the metric is defined per generated window and a feature wider
# than one window is not a within-window event. None disables the cap.
WIDTH_REL_HEIGHT = 0.5
WIDTH_CLIP_SAMPLES = "seq_length"  # "seq_length", an int, or None

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
# fraction or threshold invalidates it instead of silently producing rows of "--" for the
# parts that were never computed.
SLICE_META = SLICE_PARQUET.with_suffix(".meta.json")
# The pooled OT rows live at a coarser granularity than the per-window rows (one row per shot,
# sample, channel, threshold and depth stratum), so they get their own file. Both are written
# and invalidated together under one signature, so they can never describe different settings.
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
    """The inputs that determine the parquets' contents."""
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
        "ot_lambdas_ms": [float(x) for x in OT_LAMBDAS_MS],
        "ot_peak_mass": OT_PEAK_MASS,
        "strat_start_fraction": STRAT_START_FRACTION,
        "strat_bin_sets": {name: [list(b) for b in bins] for name, bins in STRAT_BIN_SETS.items()},
        "schema": 13,  # bump when the per-window columns or the threshold set change
    }


# %% ---------------------------------------------------------------------------------
# Captions
# ------------------------------------------------------------------------------------

DETECTION_SENTENCE = (
    r"Peaks are detected once over the full rollout and then assigned to the window "
    r"containing them, so prominences and widths are measured in their true context."
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


def _threshold_sentence(metrics, default_threshold, overrides=None):
    """Caption clause explaining the `pi >= a/b` annotation over each column group.

    The two numbers are not two variants of one measurement: they are the detection thresholds
    of two different columns, and a reader who takes the header as a single setting will
    misread the table. Which is which follows the column order, so it is derived here rather
    than written out, and stays correct if STRAT_TABLE_METRICS is reordered.
    """
    names = threshold_order(metrics, default_threshold, overrides)
    if names == ["all_peaks"]:
        # The reported table: one threshold for every column, so the header carries one number
        # and the only thing to explain is why it sits at noise scale.
        return (
            r"Both columns of a group are computed on the same peaks: everything above the "
            r"\texttt{find\_peaks} prominence given under the column group, on the "
            r"$[0,1]$-normalised signal. That threshold sits at sensor-noise scale on purpose. "
            r"Detecting the noise peaks is what lets a model be charged for the spurious mass "
            r"it emits, instead of scoring well for emitting a great deal of it, and it means "
            r"$|\Delta N|$ counts every excursion rather than only \acrshort{ELM}-scale events."
        )
    if names == ["large_scale"]:
        # The single-threshold variant: every column sees the same peaks, so the header carries
        # one number per channel and the only thing to explain is what that number selects.
        return (
            r"Every column is computed on the same peaks: those above the per-channel "
            r"\texttt{find\_peaks} prominence given under each column group, on the "
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


def _subset_sentence(bin_set, n_shots, n_total):
    """Caption clause naming the shot subset, empty for the full test set."""
    if not bin_set:
        return ""
    n_windows = bin_set_windows(bin_set)
    return (
        rf" Restricted to the {n_shots} of {n_total} test shots whose rollout from "
        rf"$f={STRAT_START_FRACTION:g}$ runs at least {n_windows} windows, i.e. the longer "
        r"discharges. The extra room is spent on depth rather than on more shots: the rollout "
        rf"is followed to window {n_windows - 1} instead of {bin_set_windows('') - 1}, so each "
        r"pool is nearly twice as long as in the full-test-set table and the late block "
        r"reaches correspondingly deeper. Rankings are comparable with that table, absolute "
        r"magnitudes are not."
    )


# %% ---------------------------------------------------------------------------------
# Table headers and block annotations
# ------------------------------------------------------------------------------------

# Channels for which the block title also carries the real peak count at the ELM-scale
# threshold, in parentheses after the count at the reported threshold. Only PD: it is the
# channel whose ELM bursts are separable by prominence alone, so the pair of numbers says how
# much of its noise-scale count is actually ELMs. Maps channel -> threshold name.
STRATUM_NOTE_EXTRA_THRESHOLD = {"PD": "elm_scale"}


def _stratum_title(label, lo, hi):
    return rf"{label} half, rollout windows ${lo}$--${hi}$"


def _threshold_value(slice_metrics, threshold_name, channel):
    """The numeric prominence a (threshold name, channel) pair resolved to, from the parquet.

    Read back from what was computed rather than from the config constants, so a printed
    number cannot drift from the peaks it describes.
    """
    vals = slice_metrics[(slice_metrics["threshold_name"] == threshold_name)
                         & (slice_metrics["channel"] == channel)]["threshold"].unique()
    return float(vals[0]) if len(vals) == 1 else None


def _metric_threshold_annotations(slice_metrics, channels, metrics, default_threshold,
                                  overrides=None):
    """Channel -> header decorated with every prominence the block's columns actually use.

    A block whose columns mix thresholds (the OT error over all peaks, the count at the
    large-scale threshold) cannot carry a single number in its header, so both are listed in
    the order METRIC_THRESHOLD names them.
    """
    out = {}
    for channel in channels:
        parts = []
        for name in threshold_order(metrics, default_threshold, overrides):
            value = _threshold_value(slice_metrics, name, channel)
            if value is not None:
                parts.append(f"{value:g}")
        out[channel] = rf"{channel}\,{{\tiny$\pi{{\geq}}{'/'.join(parts)}$}}" if parts else channel
    return out


def _pool_real_count(pool_stratum, label, channel, threshold_name):
    """(mean, sd) of the real peaks per pool, or None if not in the frame.

    The sd is across shots, the same spread the table cells carry: it is what makes the mean
    readable, since the strata mix near-ELM-free discharges with ones carrying hundreds of
    bursts per pool.
    """
    sel = pool_stratum[(pool_stratum["stratum"] == label)
                       & (pool_stratum["channel"] == channel)
                       & (pool_stratum["threshold_name"] == threshold_name)]
    if not len(sel):
        return None
    return float(sel["pool_n_peaks_real"].mean()), float(sel["pool_n_peaks_real_std"].mean())


def _pool_stratum_notes(pool_stratum, slice_metrics, channels, overrides=None, bins=None):
    """Stratum title suffix carrying the real peak count of the pool, per channel.

    A property of the measured trace, so it is identical for every model and belongs in the
    block header rather than in a column repeated down the rows. It is the reference the count
    errors above it are relative to, and it makes the difficulty gradient between the strata
    visible instead of implicit. Counted at the same threshold the count column uses, plus, for
    the channels in STRATUM_NOTE_EXTRA_THRESHOLD, at the ELM-scale threshold, each annotated
    with the prominence it was counted at so the two numbers cannot be confused.
    """
    notes = {}
    bins = STRAT_BINS if bins is None else bins
    name = threshold_of("pool_n_peaks_real", PEAK_THRESHOLD, overrides)
    for label, lo, hi in bins:
        parts = []
        for channel in channels:
            stats = _pool_real_count(pool_stratum, label, channel, name)
            if stats is None:
                continue
            mean, sd = stats
            pi = _threshold_value(slice_metrics, name, channel)
            cell = rf"{channel} ${mean:.1f}_{{\pm{sd:.1f}}}$ {{\tiny$(\pi{{\geq}}{pi:g})$}}"
            extra_name = STRATUM_NOTE_EXTRA_THRESHOLD.get(channel)
            extra = (_pool_real_count(pool_stratum, label, channel, extra_name)
                     if extra_name and extra_name != name else None)
            if extra is not None:
                e_mean, e_sd = extra
                e_pi = _threshold_value(slice_metrics, extra_name, channel)
                cell += (rf", ${e_mean:.1f}_{{\pm{e_sd:.1f}}}$ "
                         rf"{{\tiny$(\pi{{\geq}}{e_pi:g})$}}")
            parts.append(cell)
        title = _stratum_title(label, lo, hi)
        notes[title] = (r", real peaks/pool: " + "; ".join(parts)) if parts else ""
    return notes


# %% ---------------------------------------------------------------------------------
# Tables
# ------------------------------------------------------------------------------------


def render_latex(start_metrics: pd.DataFrame, label="tab:rollout_results",
                 threshold_name=None) -> str:
    """Main table: one stacked block per start fraction, one column group per diagnostic."""
    threshold_name = threshold_name or PEAK_THRESHOLD
    models = list(MODELS)
    check_available(start_metrics, models, START_FRACTIONS, TABLE_CHANNELS, threshold_name)
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
        r"second best underlined, within each start fraction. " + DETECTION_SENTENCE
        + threshold_sentence
    )
    blocks = [
        (f, rf"start fraction $f={f:g}$", f, list(TABLE_CHANNELS))
        for f in START_FRACTIONS
    ]
    return stacked_table(
        start_metrics, models, blocks, TABLE_METRICS, TABLE_GLOBAL_METRICS,
        label=label, caption=caption, group_label="Modes", threshold_name=threshold_name,
    )


def render_stratified_latex(stratum_metrics: pd.DataFrame, slice_metrics: pd.DataFrame,
                            label="tab:rollout_depth", threshold_name=None,
                            metrics=None, models=None, environment="table",
                            font=r"\scriptsize", tabcolsep=3, lam=None,
                            metric_thresholds=None, subset_note="", bins=None) -> str:
    """Main depth table: one stacked block per depth stratum at a fixed start fraction.

    `stratum_metrics` must already carry the pooled columns for `lam` (see `merge_pool_metrics`);
    `lam` only names the value in the caption, it does not select anything here.
    """
    threshold_name = threshold_name or PEAK_THRESHOLD
    lam = OT_LAMBDA_MS if lam is None else float(lam)
    metrics = list(metrics or STRAT_TABLE_METRICS)
    models = list(models or MODELS)
    channels = list(TABLE_CHANNELS)
    bins = STRAT_BINS if bins is None else bins
    n_windows = bins[-1][2] + 1
    pool_windows = bins[0][2] - bins[0][1] + 1
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
        + _count_sentence(metrics) +
        _threshold_sentence(metrics, threshold_name, metric_thresholds) +
        r" The mean real peak count per pool, with its standard deviation across shots, is "
        r"given beside each block title with the prominence it was counted at, since the later "
        r"stratum carries denser \acrshort{ELM}s and the shots differ widely in how ELMy they "
        r"are. For \acrshort{PD} a second count follows at the \acrshort{ELM}-scale prominence, "
        r"which is the part of the noise-scale count that is plausibly \acrshort{ELM}s. Lower is "
        r"better except for $\mathcal{D}$. Best in bold, second best underlined, within each "
        r"block."
        + subset_note
    )
    blocks = [(lab, _stratum_title(lab, lo, hi), lab, list(channels)) for lab, lo, hi in bins]
    return stacked_table(
        stratum_metrics, models, blocks, metrics, TABLE_GLOBAL_METRICS,
        label=label, caption=caption, environment=environment,
        group_label="Modes", font=font, tabcolsep=tabcolsep, threshold_name=threshold_name,
        block_key="stratum",
        metric_thresholds=metric_thresholds,
        group_titles=_metric_threshold_annotations(slice_metrics, channels, metrics,
                                                   threshold_name, metric_thresholds),
        block_notes=_pool_stratum_notes(stratum_metrics, slice_metrics, channels,
                                        metric_thresholds, bins),
    )


def render_appendix_latex(start_metrics: pd.DataFrame, channel_names,
                          label="tab:rollout_results_appendix", threshold_name=None) -> str:
    """Appendix table: one stacked block per diagnostic, every model, one start fraction.

    Wider than the main table (signed as well as absolute count error, and the generated and
    real peak rates in physical units) and over every observable channel, so the diagnostic
    becomes the outer row level. Wrapped in a sideways environment because it does not fit the
    text width; the document needs the rotating package.
    """
    threshold_name = threshold_name or PEAK_THRESHOLD
    models = list(all_models())
    channels = list(APPENDIX_CHANNELS or channel_names)
    check_available(start_metrics, models, [APPENDIX_START_FRACTION], channels, threshold_name)
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
        r"its magnitude. " + DETECTION_SENTENCE
    )
    # One column group per block, so the group header carries the diagnostic name and the
    # block header repeats it for readers scanning the row labels.
    blocks = [
        (channel, rf"{channel}", APPENDIX_START_FRACTION, [channel])
        for channel in channels
    ]
    return stacked_table(
        start_metrics, models, blocks, APPENDIX_METRICS, APPENDIX_GLOBAL_METRICS,
        label=label, caption=caption, environment="sidewaystable",
        group_label="Modes", font=r"\footnotesize", tabcolsep=4,
        threshold_name=threshold_name,
    )


# %% ---------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------------


def compute_slice_metrics(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read every cache and measure it: (per-window rows, pooled rows, cache audit)."""
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
    spec = PeakSpec(
        sample_rate=sample_rate,
        seq_length=seq_length,
        rel_height=WIDTH_REL_HEIGHT,
        width_clip=seq_length if WIDTH_CLIP_SAMPLES == "seq_length" else WIDTH_CLIP_SAMPLES,
        label_spill=LABEL_SPILL,
        fallback_prominence=LARGE_PEAK_PROMINENCE_DEFAULT,
    )
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
                rows.extend(record_slice_rows(record, channel_names, channels, thresholds, spec))
                # The strata are only defined at STRAT_START_FRACTION.
                if np.isclose(record["start_frac"], STRAT_START_FRACTION):
                    pool_rows.extend(record_pool_rows(
                        record, channel_names, channels, thresholds, spec,
                        bin_sets=STRAT_BIN_SETS, lambdas_ms=OT_LAMBDAS_MS, peak_mass=OT_PEAK_MASS
                    ))
            for target, batch in ((frames, rows), (pool_frames, pool_rows)):
                if batch:
                    frame = pd.DataFrame(batch)
                    frame.insert(0, "model", name)
                    target.append(frame)
            if i % 10 == 0 or i == len(shots):
                print(f"  {i}/{len(shots)} shots")

    return (pd.concat(frames, ignore_index=True),
            pd.concat(pool_frames, ignore_index=True),
            pd.DataFrame(audit_rows))


def load_or_compute(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """The two parquets, recomputed unless a previous run's signature still matches."""
    signature = slice_signature()
    if (SLICE_PARQUET.exists() and POOL_PARQUET.exists() and SLICE_META.exists()
            and not args.force and not args.shots):
        cached = json.loads(SLICE_META.read_text())
        # Compare the signature after a JSON round-trip, not as live Python. Tuples in the
        # signature (the model pairs, the threshold overrides) come back as lists, so a direct
        # comparison reports every run as stale and silently recomputes the whole parquet.
        stale = [k for k, v in json.loads(json.dumps(signature)).items() if cached.get(k) != v]
        if stale:
            # Reusing a parquet computed for a different model list, start fraction set or
            # threshold silently produces rows of "--" for whatever was never computed.
            print(f"{SLICE_PARQUET} was computed for different settings ({stale}); recomputing")
        else:
            print(f"reusing {SLICE_PARQUET} (pass --force to recompute)")
            return pd.read_parquet(SLICE_PARQUET), pd.read_parquet(POOL_PARQUET), None

    slice_metrics, pool_metrics, audit = compute_slice_metrics(args)
    if not args.shots:
        slice_metrics.to_parquet(SLICE_PARQUET)
        pool_metrics.to_parquet(POOL_PARQUET)
        SLICE_META.write_text(json.dumps(signature, indent=2))
    return slice_metrics, pool_metrics, audit


def write_depth_tables(slice_metrics, pool_metrics, suffix, label, threshold_name):
    """Every depth-table variant: shot subset x lambda x threshold override. Returns the paths.

    The reported (default subset, reported lambda, default threshold) table keeps the
    unsuffixed filename and label so the paper's \\input path never moves; the sensitivity
    variants are written beside it. All of them read the same stored peaks, so they cost
    nothing beyond the rendering.
    """
    n_test_shots = slice_metrics["shot"].nunique()
    lambdas = sorted(pool_metrics["lam_ms"].unique(),
                     key=lambda x: (not np.isclose(x, OT_LAMBDA_MS), x))
    paths, full_set_pool = [], None
    for shot_tag, bin_set in STRAT_SHOT_SUBSETS:
        # Filtering before aggregation, not after: the shots are the unit the means and the
        # standard deviations are taken over, so a subset has to re-run the ladder.
        bins = bins_of(bin_set)
        if not bin_set:
            slices, pools, note = slice_metrics, pool_metrics, ""
        else:
            # The pool pass only recorded a bin set for the shots long enough to fill it, so
            # the parquet already defines the subset; long_shots is the same list read off the
            # per-window side, and disagreement means the two passes saw different caches.
            keep = pool_bin_set_shots(pool_metrics, bin_set)
            expected = long_shots(slice_metrics, bin_set_windows(bin_set), STRAT_START_FRACTION)
            if keep != expected:
                raise ValueError(
                    f"bin set {bin_set!r} was recorded for {keep} but the per-window frame says "
                    f"{expected} reach depth {bin_set_windows(bin_set)}; rerun with --force")
            slices = slice_metrics[slice_metrics["shot"].isin(keep)]
            pools = pool_metrics[pool_metrics["shot"].isin(keep)]
            note = _subset_sentence(bin_set, len(keep), n_test_shots)
            print(f"{shot_tag}: bin set {bin_set!r}, windows {bins}, "
                  f"{len(keep)} of {n_test_shots} shots: {keep}")

        stratum_slices = aggregate_by_stratum(slices, bins, STRAT_START_FRACTION)
        subset_pool_stratum = aggregate_pools(pools)
        subset_pool_stratum.to_csv(
            OUTPUT_DIR / f"rollout_pool_metrics{suffix}{shot_tag}.csv", index=False)
        full_set_pool = full_set_pool if full_set_pool is not None else subset_pool_stratum

        for lam in lambdas:
            lam_tag = "" if np.isclose(lam, OT_LAMBDA_MS) else f"_lam{lam:g}"
            stratum = merge_pool_metrics(stratum_slices, subset_pool_stratum, lam=lam,
                                         bin_set=bin_set)
            stratum.to_csv(
                OUTPUT_DIR / f"rollout_stratum_metrics{suffix}{shot_tag}{lam_tag}.csv", index=False)
            for var_tag, overrides in STRAT_TABLE_VARIANTS:
                # A subset only gets the reported lambda and the default threshold variant: the
                # sensitivity variants exist to check the reported table, not each other.
                if shot_tag and (lam_tag or var_tag):
                    continue
                tag = f"{suffix}{var_tag}{lam_tag}{shot_tag}"
                lab = f"{label}_depth{var_tag}{lam_tag}{shot_tag}"
                for kind, extra in (("", {}),
                                    ("_appendix", dict(
                                        metrics=["ot_error", "ot_error_rel", "ot_missed_frac",
                                                 "ot_false_frac", "pool_count_error",
                                                 "pool_abs_count_error"],
                                        models=list(all_models()), font=r"\footnotesize",
                                        tabcolsep=4))):
                    tex = render_stratified_latex(
                        stratum, slices, label=f"{lab}{kind}", threshold_name=threshold_name,
                        lam=lam, metric_thresholds=overrides, subset_note=note, bins=bins,
                        **extra)
                    path = OUTPUT_DIR / f"rollout_depth{kind}_table{tag}.tex"
                    path.write_text(tex)
                    paths.append(path)
    return paths, full_set_pool


def print_report(start, pool_stratum, threshold_name):
    print(f"\n=== start-fraction report (threshold: {threshold_name}) ===")
    start = start[start["threshold_name"] == threshold_name]
    show = ["model", "start_frac", "channel", "rate_gen_per_ms", "rate_real_per_ms",
            "count_error", "abs_count_error", "miss_rate", "prominence_w1", "width_w1_ms",
            "dice", "n_trajectories"]
    with pd.option_context("display.width", 220, "display.max_rows", 400):
        print(start[show].sort_values(["start_frac", "channel", "model"]).round(4).to_string(index=False))

    print(f"\n=== pooled OT report (reported lambda={OT_LAMBDA_MS:g} ms, "
          f"computed {OT_LAMBDAS_MS}, mass={OT_PEAK_MASS}) ===")
    cols = ["model", "bin_set", "stratum", "channel", "threshold_name", "lam_ms", "ot_error",
            "ot_error_rel", "ot_loc", "ot_missed_frac", "ot_false_frac", "pool_n_peaks_gen",
            "pool_n_peaks_real", "pool_abs_count_error", "n_trajectories"]
    pool_show = pool_stratum[pool_stratum["channel"].isin(TABLE_CHANNELS)]
    with pd.option_context("display.width", 240, "display.max_rows", 600):
        print(pool_show[cols].sort_values(["bin_set", "lam_ms", "channel", "threshold_name",
                                           "stratum", "model"]).round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute even if the parquets exist")
    parser.add_argument("--shots", type=int, default=None, help="use only the first N shots (smoke test)")
    parser.add_argument("--label", default="tab:rollout_results", help="LaTeX label for the table")
    parser.add_argument("--min-shots", type=int, default=DEPTH_MIN_SHOTS,
                        help="drop depths reached by fewer shots than this in the depth plots "
                             f"(default {DEPTH_MIN_SHOTS}); lower it for --shots smoke runs")
    parser.add_argument("--threshold", choices=["all_peaks", "elm_scale", "large_scale"], default=None,
                        help=f"override PEAK_THRESHOLD (default {PEAK_THRESHOLD}) for the tables, "
                             "the report and the depth plots; non-default values write to "
                             "suffixed filenames instead of overwriting the main outputs")
    parser.add_argument("--models", choices=["all", "main"], default="all",
                        help="'main' drops the appendix-only ablation caches, which are the "
                             "slowest to read and are only needed by the appendix tables; the "
                             "main and depth tables are unaffected")
    args = parser.parse_args()

    global INCLUDE_APPENDIX_MODELS
    INCLUDE_APPENDIX_MODELS = args.models == "all"
    if not INCLUDE_APPENDIX_MODELS:
        print(f"--models main: skipping {list(APPENDIX_EXTRA_MODELS)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slice_metrics, pool_metrics, audit = load_or_compute(args)
    if audit is not None:
        audit.to_csv(OUTPUT_DIR / "cache_audit.csv", index=False)
        print(audit.to_string(index=False))

    _, trajectory, start = aggregate(slice_metrics)
    trajectory.to_csv(OUTPUT_DIR / "rollout_trajectory_metrics.csv", index=False)
    start.to_csv(OUTPUT_DIR / "rollout_start_metrics.csv", index=False)

    threshold_name = args.threshold or PEAK_THRESHOLD
    # A non-default threshold gets its own filenames so a --threshold run never clobbers the
    # main paper outputs; the suffix is empty (and the label untouched) at the default.
    suffix = "" if threshold_name == PEAK_THRESHOLD else f"_{threshold_name}"
    label = args.label if not suffix else f"{args.label}{suffix}"

    tex_path = OUTPUT_DIR / f"rollout_results_table{suffix}.tex"
    tex_path.write_text(render_latex(start, label=label, threshold_name=threshold_name))

    present = list(start["channel"].unique())
    # Keep the cache's column order rather than pandas' order of appearance. CHANNEL_ORDER is
    # empty when the parquet was reused without opening a cache, in which case fall back to
    # the frame's own order.
    channel_names = [c for c in CHANNEL_ORDER if c in present] or present
    appendix_path = OUTPUT_DIR / f"rollout_results_appendix_table{suffix}.tex"
    appendix_path.write_text(render_appendix_latex(
        start, channel_names, label=f"{label}_appendix", threshold_name=threshold_name))

    depth_paths, pool_stratum = write_depth_tables(slice_metrics, pool_metrics, suffix, label,
                                                   threshold_name)

    depth = aggregate_by_depth(slice_metrics)
    depth.to_csv(OUTPUT_DIR / f"rollout_depth_metrics{suffix}.csv", index=False)
    # Stamped on every row so it survives a parquet reuse without opening a cache.
    seq_length = int(slice_metrics["slice_length"].iloc[0])
    plots = plot_depth_curves(
        depth, slice_metrics, seq_length, OUTPUT_DIR / f"depth{suffix}",
        models=list(MODELS), metrics=DEPTH_PLOT_METRICS, channel=DEPTH_PLOT_CHANNEL,
        start_fractions=DEPTH_PLOT_START_FRACTIONS, threshold_name=threshold_name,
        sizes=DEPTH_PLOT_SIZES, min_shots=args.min_shots,
    )
    print(f"wrote {len(plots)} depth plots under {(OUTPUT_DIR / f'depth{suffix}').resolve()}")

    print_report(start, pool_stratum, threshold_name)
    print(f"\nwrote {tex_path}\nwrote {appendix_path}")
    for path in depth_paths:
        print(f"wrote {path}")
    print((OUTPUT_DIR / f"rollout_depth_table{suffix}.tex").read_text())


if __name__ == "__main__":
    main()
