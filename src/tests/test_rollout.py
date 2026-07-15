"""Tests for rollout planning (compute_rollout_specs) and the rollout HDF5 cache."""
import numpy as np
import pandas as pd
import pytest


class FakeDataset:
    """Just enough of FusionShotDataset for compute_rollout_specs."""
    seq_length = 256
    history_length = 256
    crop_margin = 384

    def __init__(self, shot_lengths: dict[int, int]):
        self.data = pd.DataFrame({
            'ShotNum': np.concatenate([[shot] * length for shot, length in shot_lengths.items()])
        })
        self.shot_numbers = list(shot_lengths)


def test_specs_clamp_and_window_count():
    from src.rollout import compute_rollout_specs
    ds = FakeDataset({1: 5000})
    specs, skipped = compute_rollout_specs(ds, [0.1, 0.5, 0.9])
    by_frac = {s.start_frac: s for s in specs}
    assert by_frac[0.1].start_i == 500
    assert by_frac[0.5].start_i == 2500
    # 90% of 5000 = 4500 > hi = 5000 - 384 - 256 = 4360, so it clamps
    assert by_frac[0.9].start_i == 4360
    # available = 5000 - 384 - 500 = 4116 -> 16 full windows
    assert by_frac[0.1].n_windows == 16
    assert by_frac[0.1].total_length == 16 * 256
    assert by_frac[0.9].n_windows == 1
    assert not skipped


def test_specs_dedupe_on_short_shot():
    from src.rollout import compute_rollout_specs
    # Short shot: lo=384, hi=1200-384-256=560; most fractions clamp onto the same starts
    ds = FakeDataset({2: 1200})
    specs, skipped = compute_rollout_specs(ds, [0.1, 0.25, 0.5, 0.75, 0.9])
    starts = [s.start_i for s in specs]
    assert len(starts) == len(set(starts)), "clamped duplicates must be dropped"
    assert all(entry['reason'] == 'clamped onto an existing start' for entry in skipped)


def test_specs_skip_too_short_shot():
    from src.rollout import compute_rollout_specs
    ds = FakeDataset({3: 900})  # < 2*crop_margin + seq_length = 1024
    specs, skipped = compute_rollout_specs(ds, [0.5])
    assert not specs
    assert skipped and 'too short' in skipped[0]['reason']


def test_specs_n_samples_and_step():
    from src.rollout import compute_rollout_specs
    ds = FakeDataset({1: 5000})
    specs, _ = compute_rollout_specs(ds, [0.1], n_samples=3)
    assert [s.sample_idx for s in specs] == [0, 1, 2]
    # Overlapped chaining: step 128 doubles the window count (minus the last partial)
    specs_128, _ = compute_rollout_specs(ds, [0.1], step=128)
    assert specs_128[0].n_windows == (4116 - 256) // 128 + 1
    assert specs_128[0].total_length == (specs_128[0].n_windows - 1) * 128 + 256


def test_rollout_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_CACHE_DIR", str(tmp_path))
    from src.hdf_cache import RolloutHDFCache
    gx = np.random.rand(5, 768).astype(np.float32)
    lg = np.random.randint(0, 3, 1024).astype(np.int16)
    lr = np.random.randint(0, 3, 1024).astype(np.int16)
    attrs = {'start_frac': 0.5, 'start_i': 5000, 't_start': 0.89, 't_end': 0.966,
             'n_windows': 3, 'seq_length': 256, 'history_length': 256, 'step': 256}

    cache = RolloutHDFCache('test_rollout', mode='w')
    cache.set_rollout(61237, 5000, 0, gx, lg, lr, attrs=attrs)
    assert cache.has(61237, 5000, 0) and not cache.has(61237, 5000, 1)

    reader = RolloutHDFCache('test_rollout', mode='r')
    entry = reader.get_rollout(61237, 5000)
    assert np.allclose(entry['generated_x'], gx)
    assert (entry['surr_labels_gen'] == lg).all() and (entry['surr_labels_real'] == lr).all()
    assert entry['n_windows'] == 3 and entry['start_frac'] == 0.5
    assert reader.list_rollouts() == [(61237, 5000, 0)]
    with pytest.raises(RuntimeError):
        reader.set_rollout(61237, 5000, 0, gx, lg, lr)
    with pytest.raises(KeyError):
        reader.get_rollout(99999, 0, 0)

    # Append mode skips existing groups (resumability); write mode overwrites
    appender = RolloutHDFCache('test_rollout', mode='a')
    appender.set_rollout(61237, 5000, 0, np.zeros_like(gx), lg, lr)
    assert np.allclose(reader.get_rollout(61237, 5000)['generated_x'], gx)
    cache.set_rollout(61237, 5000, 0, np.zeros_like(gx), lg, lr)
    assert (reader.get_rollout(61237, 5000)['generated_x'] == 0).all()
