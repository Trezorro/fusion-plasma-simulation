"""Read side of the rollout HDF caches.

Offline analysis (`eval_notebooks/rollout_tables.py`) compares caches produced by different
runs, possibly weeks apart. Each cache stamps the config it was produced under; this module
reads that stamp, rebuilds the data module from it rather than from the local yaml, and
refuses to compare caches whose data or peak settings disagree. A silently mismatched pair
would still render a table, and the numbers in it would not be comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

import src.data_loaders
from src.config import load_config_from_file
from src.hdf_cache import RolloutHDFCache

# Prominence for a coarser "large peaks only" pass. Not read from the config because it is not
# one of flow.py's own evaluation thresholds, just a table variant. PD and DML do not sit on
# the same scale (PD's ELM spikes are far taller, in normalized [0,1] units, than DML's slower
# deflections), so this is per channel rather than one number applied everywhere like the
# other two thresholds. A channel with no entry here falls back to the default.
LARGE_PEAK_PROMINENCE_DEFAULT = 0.01
LARGE_PEAK_PROMINENCE_OVERRIDES = {"PD": 0.05, "DML": 0.005}


def get(cfg, dotted, default=None):
    """Nested lookup into a stamped config dict, by dotted path."""
    for key in dotted.split("."):
        if isinstance(cfg, dict) and key in cfg:
            cfg = cfg[key]
        else:
            return default
    return cfg


def cache_config(cache: RolloutHDFCache) -> dict:
    """The config a cache was produced under, from its own root attributes."""
    raw = cache.get_root_attrs().get("config_json")
    if not raw:
        raise ValueError(
            f"{getattr(cache, 'name', cache)} has no stamped config_json; this refuses to "
            "guess the config that produced a cache."
        )
    return json.loads(raw.decode() if isinstance(raw, bytes) else raw)


def cfg_cols(cfg) -> list:
    """The observable channel names of a stamped config."""
    return list(get(cfg, "data.cols.x") or [])


def build_data_module(reference_cfg):
    """(config, data module) built from the reference cache's own data block, not the local yaml."""
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
    # truncated and every window comparison shifts.
    if int(C.data.crop_margin) < int(C.data.history_length):
        raise ValueError(
            f"crop_margin ({C.data.crop_margin}) < history_length ({C.data.history_length}); "
            "the real context window would run off the start of the shot."
        )
    return C, data_module


def resolve_thresholds(cfg) -> dict:
    """Threshold name -> prominence (a number, or a per-channel dict), from a stamped config."""
    peaks = get(cfg, "evaluation.peaks", {}) or {}
    return {
        "all_peaks": float(peaks["prominence"]),
        "elm_scale": float(peaks["elm_pd_prominence"]),
        "large_scale": dict(LARGE_PEAK_PROMINENCE_OVERRIDES),
    }


def verify_compatibility(name, cache_name, cfg, ref_cfg, channel_names, thresholds, sample_rate):
    """Refuse to compare caches that were produced under different data or peak settings."""
    cache_thresholds = resolve_thresholds(cfg)
    checks = {
        "observable columns": [str(c) for c in cfg_cols(cfg)] == list(channel_names),
        "data file": get(cfg, "data.file") == get(ref_cfg, "data.file"),
        "training split": list(get(cfg, "data.train_shots") or []) == list(get(ref_cfg, "data.train_shots") or []),
        "test split": list(get(cfg, "data.test_shots") or []) == list(get(ref_cfg, "data.test_shots") or []),
        # history_length and seq_length decide how the real trace is aligned to the generated
        # one and how long a window is; a mismatch would shift the comparison without crashing.
        "history_length": int(get(cfg, "data.history_length")) == int(get(ref_cfg, "data.history_length")),
        "seq_length": int(get(cfg, "data.seq_length")) == int(get(ref_cfg, "data.seq_length")),
        "sample rate": float(get(cfg, "data.sample_rate")) == sample_rate,
        "all-peaks threshold": np.isclose(cache_thresholds["all_peaks"], thresholds["all_peaks"]),
        "ELM-scale threshold": np.isclose(cache_thresholds["elm_scale"], thresholds["elm_scale"]),
    }
    failed = [check for check, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"'{name}' ({cache_name}) is incompatible with the reference cache for: {failed}")
