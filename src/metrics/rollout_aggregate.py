"""The aggregation ladder for the rollout tables.

Turns the per-window rows of `src/metrics/rollout_peaks.py` into the frames the tables and the
depth plots read. Every level is an unweighted mean over the level below, so a shot with many
windows does not outweigh a short one and a stochastic model's samples are averaged before the
shot mean is taken. The reported spread is always the standard deviation across real
trajectories (shots) at the last step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRAJECTORY_KEYS = ["model", "shot", "start_idx", "start_frac"]
CONDITION_KEYS = ["channel", "threshold_name"]

# Per-window quantities averaged at every level. Wasserstein columns skip NaN windows by
# construction (pandas mean is NaN-skipping), which is exactly the both-non-empty restriction.
PEAK_METRICS = [
    "n_peaks_gen", "n_peaks_real", "rate_gen_per_ms", "rate_real_per_ms",
    "count_error", "abs_count_error",
    "miss_rate", "spurious_rate", "both_nonempty",
    "prominence_w1", "width_w1_ms", "prominence_w1_legacy", "width_w1_ms_legacy",
]
SLICE_METRICS = PEAK_METRICS + ["width_w1_ms_base", "dice", "dice_macro"]

# Ratios that must never be averaged, at any level. |dN|/N is dominated by whatever unit holds
# the fewest real peaks: 4 of 43 shots average under one real PD peak per window and 10 do on
# DML, so a mean of per-shot ratios reads 2.18 on DML against a true 0.99, and on PD it is
# outright infinite because one shot has no real peaks at all. These are formed once, at the
# very end, as a ratio of the two aggregate means. The per-window column stays in the parquet
# for auditing but is never aggregated.
DERIVED_METRICS = ["rel_count_error"]

POOL_METRICS = [
    "ot_error", "ot_error_rel", "ot_loc", "ot_missed_mass", "ot_false_mass",
    "ot_gen_mass", "ot_real_mass", "ot_missed_frac", "ot_false_frac",
    "pool_n_peaks_gen", "pool_n_peaks_real", "pool_count_error", "pool_abs_count_error",
]


def _mean_std_agg(metrics):
    agg = {m: (m, "mean") for m in metrics}
    agg |= {f"{m}_std": (m, "std") for m in metrics}
    return agg


def _add_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Pooled ratio metrics, formed after the last aggregation step; see DERIVED_METRICS.

    Both operands are already means over shots, so this is one ratio of two aggregates rather
    than an average of per-unit ratios. It therefore carries no standard deviation, and the
    `_std` columns are set to NaN so the formatter prints the bare value.
    """
    frame["rel_count_error"] = frame["abs_count_error"] / frame["n_peaks_real"].replace(0, np.nan)
    frame["rel_count_error_std"] = np.nan
    return frame


def aggregate(slice_metrics: pd.DataFrame):
    """window -> sample -> real trajectory -> (model, start fraction)."""
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
    start = (
        trajectory
        .groupby(["model", "start_frac"] + CONDITION_KEYS, as_index=False)
        .agg(**_mean_std_agg(SLICE_METRICS), n_trajectories=("shot", "size"),
             n_samples=("n_samples", "mean"))
    )
    return sample, trajectory, _add_derived(start)


def stratum_of(k, bins):
    """The bin label containing depth k, or None if k falls outside every bin."""
    for label, lo, hi in bins:
        if lo <= k <= hi:
            return label
    return None


def aggregate_by_stratum(slice_metrics: pd.DataFrame, bins, start_frac: float):
    """Like `aggregate`, but blocks are depth strata at one fixed start fraction.

    The start fraction is held fixed and the depth index k is binned, so the two blocks differ
    only in how far the rollout has run. Windows outside every bin are dropped, which is what
    keeps the depth budget identical across shots: the cap is the shortest shot's, so no shot
    is missing a depth that another one contributes.
    """
    d = slice_metrics[np.isclose(slice_metrics["start_frac"], start_frac)].copy()
    if d.empty:
        raise ValueError(
            f"no windows at start fraction {start_frac}; add it to START_FRACTIONS "
            "(it is already in the signature) and rerun with --force"
        )
    d["stratum"] = d["k"].map(lambda k: stratum_of(k, bins))
    d = d[d["stratum"].notna()]

    keys = ["model", "shot", "start_idx", "stratum"] + CONDITION_KEYS
    sample = (d.groupby(keys + ["sample_idx"], as_index=False)
               .agg(**{m: (m, "mean") for m in SLICE_METRICS}, n_slices=("k", "size")))
    trajectory = (sample.groupby(keys, as_index=False)
                  .agg(**{m: (m, "mean") for m in SLICE_METRICS},
                       n_samples=("sample_idx", "nunique"), n_slices=("n_slices", "first")))
    stratum = (trajectory.groupby(["model", "stratum"] + CONDITION_KEYS, as_index=False)
               .agg(**_mean_std_agg(SLICE_METRICS), n_trajectories=("shot", "size"),
                    n_samples=("n_samples", "mean")))
    return _add_derived(stratum)


def aggregate_pools(pool_metrics: pd.DataFrame) -> pd.DataFrame:
    """sample -> real trajectory -> (model, stratum), the same ladder `aggregate` uses.

    There is no window level here: the pool *is* the unit.
    """
    keys = ["model", "shot", "start_idx", "bin_set", "stratum"] + CONDITION_KEYS + ["lam_ms"]
    trajectory = (pool_metrics.groupby(keys, as_index=False)
                  .agg(**{m: (m, "mean") for m in POOL_METRICS},
                       n_samples=("sample_idx", "nunique")))
    return (trajectory.groupby(["model", "bin_set", "stratum"] + CONDITION_KEYS + ["lam_ms"],
                               as_index=False)
            .agg(**_mean_std_agg(POOL_METRICS), n_trajectories=("shot", "size"),
                 n_samples=("n_samples", "mean")))


def merge_pool_metrics(stratum_metrics: pd.DataFrame, pool_stratum: pd.DataFrame,
                       lam: float, bin_set: str = "") -> pd.DataFrame:
    """Attach the pooled columns to the per-window stratum frame on their shared keys.

    The two frames answer different questions at the same granularity (model, stratum, channel,
    threshold), so the table renderer can read a pooled and a per-window column side by side
    without knowing which is which. An outer join: a threshold present in one and not the other
    should surface as a '--' cell, not silently drop the row that does exist.

    The pool frame carries every lambda that was computed; selecting one here rather than in
    the renderer keeps the merged frame at the stratum frame's granularity, so a lam column can
    never silently duplicate rows into the per-window metrics.
    """
    keys = ["model", "stratum"] + CONDITION_KEYS
    selected = pool_stratum[np.isclose(pool_stratum["lam_ms"], float(lam))
                            & (pool_stratum["bin_set"] == bin_set)]
    if selected.empty:
        raise ValueError(
            f"(lambda={lam}, bin set {bin_set!r}) is not in the parquet, which holds "
            f"lambdas {sorted(pool_stratum['lam_ms'].unique())} and bin sets "
            f"{sorted(pool_stratum['bin_set'].unique())}; add it to OT_LAMBDAS_MS or "
            "STRAT_BIN_SETS and rerun with --force")
    shared = (set(stratum_metrics.columns) & set(selected.columns)) - set(keys)
    return stratum_metrics.merge(
        selected.drop(columns=sorted(shared)), on=keys, how="outer", validate="one_to_one")


def pool_bin_set_shots(pool_metrics: pd.DataFrame, bin_set: str) -> list:
    """Shots whose rollout was long enough for `bin_set` to be recorded."""
    return sorted(pool_metrics[pool_metrics["bin_set"] == bin_set]["shot"].unique())


def long_shots(slice_metrics: pd.DataFrame, min_windows: int, start_frac: float) -> list:
    """Test shots whose rollout from `start_frac` runs at least `min_windows` windows.

    Read off the per-window frame rather than the caches, so it works on a reused parquet: the
    depth index k is 0-based and dense, so max(k) + 1 is the window budget of that shot.
    """
    at_start = slice_metrics[np.isclose(slice_metrics["start_frac"], start_frac)]
    budget = at_start.groupby("shot")["k"].max() + 1
    return sorted(budget[budget >= min_windows].index)


def aggregate_by_depth(slice_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to (model, start fraction, channel, threshold, depth k).

    Depth is aggregated *last*: within one rollout a depth-k window is a single window, so the
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
    return (
        per_trajectory
        .groupby(["model", "start_frac"] + CONDITION_KEYS + ["k"], as_index=False)
        .agg(**_mean_std_agg(SLICE_METRICS), n_shots=("shot", "nunique"))
    )
