"""Median/mean time between consecutive large PD ELM peaks, over the real test-shot traces.

A quick side script for tuning a new metric, not part of the paper table pipeline. Uses the
same prominence threshold as the "large_scale" PD column in rollout_tables.py (0.05, in
normalized [0,1] units) and the same reference cache to pick the data config and test shots.
A shot contributes an interval only if it has >= 2 detected peaks; a shot with 0 or 1 peak
gives no interval and is excluded rather than propagating a NaN or an inf.

Run:  PYTHONPATH=. python eval_notebooks/elm_interval_stats.py
"""
import numpy as np
from scipy.signal import find_peaks

from eval_notebooks.rollout_tables import REFERENCE_MODEL, MODELS
from src.hdf_cache import RolloutHDFCache
from src.rollout_cache import build_data_module, cache_config

PD_PROMINENCE = 0.1  # bumped from 0.05 to check sensitivity to ripple-scale peaks

ref_cache = RolloutHDFCache(MODELS[REFERENCE_MODEL], mode="r")
ref_cfg = cache_config(ref_cache)
_, data_module = build_data_module(ref_cfg)

df = data_module.test_dataset.data
shot_numbers = df["ShotNum"].unique()

intervals_ms = []
per_shot = []
for shot in shot_numbers:
    shot_df = df[df["ShotNum"] == shot]
    pd_trace = shot_df["PD"].to_numpy()
    t = shot_df.index.to_numpy()  # physical time in seconds, per-sample
    peak_idx, _ = find_peaks(pd_trace, prominence=PD_PROMINENCE)
    if len(peak_idx) < 2:
        continue
    dt_ms = np.diff(t[peak_idx]) * 1000.0
    intervals_ms.append(dt_ms)
    per_shot.append((int(shot), len(peak_idx), float(np.median(dt_ms))))

all_intervals = np.concatenate(intervals_ms)
n_shots_used = len(intervals_ms)
n_shots_total = len(shot_numbers)

print(f"PD prominence >= {PD_PROMINENCE}")
print(f"shots with >=2 peaks: {n_shots_used} / {n_shots_total}")
print(f"total intervals (pooled across shots): {len(all_intervals)}")
print(f"pooled median inter-peak time: {np.median(all_intervals):.2f} ms")
print(f"pooled mean inter-peak time:   {np.mean(all_intervals):.2f} ms")
print(f"pooled std inter-peak time:    {np.std(all_intervals):.2f} ms")

per_shot_medians = np.array([m for _, _, m in per_shot])
print(f"mean of per-shot medians:      {per_shot_medians.mean():.2f} ms")

print("\nper-shot (shot, n_peaks, median interval ms):")
for shot, n, m in sorted(per_shot, key=lambda r: r[0]):
    print(f"  {shot:>6d}  n={n:>4d}  median={m:>8.2f} ms")
