"""Autoregressive rollout evaluation.

A rollout chains model generations over a whole shot: start at a fractional position
with the REAL history window, generate the future window, feed the generated window
back as the next history, and repeat until the end of the shot (minus crop_margin).
Control covariates and position sequences always come from the real shot data; only
the observable history is generated. The conditioning 'label' entry is carried along
for consistency but is not a model input (metrics only), so no label leakage occurs.

Runs after trainer.test in run.py when a `rollout:` block is present in the config
(absent block = feature off). Also runnable standalone:
    PYTHONPATH=. python src/rollout.py
which uses an untrained model as a cheap smoke test of the chaining mechanics.

Label conventions (important): surrogate labels produced here are UNSHIFTED
0=L, 1=D, 2=H (never Unknown), same as the window test cache. The in-memory
LHD_label / conditioning_input['label'] convention is +1-shifted; do not mix them.
"""
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from src.data_loaders import FusionShotDataModule, FusionShotDataset
from src.hdf_cache import RolloutHDFCache
from src.metrics import evaluate_modes
from src.metrics.evaluate_modes import get_mode_predictions_batched_window

logger = logging.getLogger(__name__)


@dataclass
class RolloutSpec:
    """One planned rollout: where it starts and how far it runs."""
    shot_number: int
    start_frac: float  # requested start fraction of the shot length (pre-clamp)
    start_i: int  # shot-local positional index of the first generated sample
    n_windows: int  # number of generation calls
    total_length: int  # kept generated samples: (n_windows - 1) * step + seq_length
    sample_idx: int = 0


@dataclass
class RolloutResult:
    spec: RolloutSpec
    generated_x: np.ndarray  # (channels, total_length), normalized [0,1]
    t_start: float  # seconds, time of the first generated sample
    t_end: float  # seconds, time of the last generated sample
    surr_labels_gen: np.ndarray | None = None  # (history_length + total_length,), 0=L,1=D,2=H
    surr_labels_real: np.ndarray | None = None


def compute_rollout_specs(
    dataset: FusionShotDataset, start_fractions, n_samples=1, step=None
) -> tuple[list[RolloutSpec], list[dict]]:
    """Plan rollouts for every shot in the dataset at each start fraction.

    Start indices are clamped to [crop_margin, shot_len - crop_margin - seq_length]
    (same viability bounds as precompute_indices). On short shots different fractions
    can clamp onto the same index; duplicates are dropped, keeping the first fraction.

    Args:
        dataset: FusionShotDataset (normally the test set).
        start_fractions: Iterable of floats in (0, 1), e.g. [0.1, 0.25, 0.5, 0.75, 0.9].
        n_samples: Stochastic samples per start point.
        step: Samples to advance per generation. Defaults to seq_length (non-overlapping).

    Returns:
        Tuple of (specs, skipped) where skipped records shot/fraction pairs that could
        not produce a rollout and why.
    """
    step = step or dataset.seq_length
    assert 0 < step <= dataset.seq_length, "step must be in (0, seq_length]"
    specs: list[RolloutSpec] = []
    skipped: list[dict] = []
    for shot_number in sorted(int(s) for s in dataset.shot_numbers):
        shot_len = len(dataset.data[dataset.data['ShotNum'] == shot_number])
        lo = dataset.crop_margin
        hi = shot_len - dataset.crop_margin - dataset.seq_length
        if hi < lo:
            skipped.append({'shot': shot_number, 'reason': f'shot too short ({shot_len} samples)'})
            continue
        seen_starts = set()
        for frac in start_fractions:
            start_i = min(max(int(round(frac * shot_len)), lo), hi)
            if start_i in seen_starts:
                skipped.append({'shot': shot_number, 'start_frac': frac, 'reason': 'clamped onto an existing start'})
                continue
            available = shot_len - dataset.crop_margin - start_i
            n_windows = (available - dataset.seq_length) // step + 1
            if n_windows < 1:
                skipped.append({'shot': shot_number, 'start_frac': frac, 'reason': 'no room for a full window'})
                continue
            seen_starts.add(start_i)
            total_length = (n_windows - 1) * step + dataset.seq_length
            for sample_idx in range(n_samples):
                specs.append(
                    RolloutSpec(shot_number, float(frac), start_i, int(n_windows), int(total_length), sample_idx)
                )
    logger.info(
        "Planned %d rollouts over %d shots (%d skipped shot/fraction combinations).",
        len(specs), len({s.shot_number for s in specs}), len(skipped)
    )
    return specs, skipped


def _generate_window_batch(model, cond_b, target_size, n_steps):
    """Generate one batch of future windows, matching each module's own inference style.

    FlowModule integrates the learned velocity field from the prior; UnFlowModule
    (deterministic baselines) predicts directly from the prior with a constant t=1,
    exactly like its inference().
    """
    from src.models.UnFlow import UnFlowModule
    prior = model.get_prior_samples(cond_b, target_size)
    if isinstance(model, UnFlowModule):
        t_dummy = torch.ones(target_size[0], device=model.device)
        return model.model(prior, t_dummy, conditioning_input=cond_b)
    return model.integrate_path(prior, conditioning_input=cond_b, n_steps=n_steps)


def _assemble_history(cond, kept_chunks, history_length, clamp_history):
    """Build the x_history for the next window from generated (and, early on, real) data.

    The history is the last history_length samples of (real trace before the rollout
    start + kept generated trace). Until enough samples are generated, the leading part
    stays real; from then on the history is fully generated.
    """
    generated = np.concatenate(kept_chunks, axis=-1)
    gen_len = generated.shape[-1]
    if gen_len >= history_length:
        history = generated[:, -history_length:]
    else:
        # cond['x_history'] is the REAL window before the current position; its first
        # (history_length - gen_len) samples predate the rollout start and stay real.
        history = np.concatenate([cond['x_history'][:, :history_length - gen_len], generated], axis=-1)
    if clamp_history:
        history = np.clip(history, 0.0, 1.0)
    return history


@torch.inference_mode()
def _generate_rollouts(
    model, dataset: FusionShotDataset, specs: list[RolloutSpec], n_steps, max_batch=128, step=None,
    clamp_history=False
) -> list[RolloutResult]:
    """Generate all rollouts in lockstep: window k of every unfinished rollout is batched together.

    Mixed shots in one batch are fine since conditioning is assembled per sample via
    get_shot_window. Finished rollouts drop out, so batches shrink toward the end.
    """
    step = step or dataset.seq_length
    model.eval()
    device = model.device
    history_length = dataset.history_length
    buffers: list[list[np.ndarray]] = [[] for _ in specs]
    times: list[dict] = [{} for _ in specs]
    drift_min, drift_max = np.inf, -np.inf
    max_windows = max(s.n_windows for s in specs)
    total_windows = sum(s.n_windows for s in specs)
    progress = tqdm(total=total_windows, desc="Rollout windows")
    for k in range(max_windows):
        active = [i for i, s in enumerate(specs) if k < s.n_windows]
        for chunk_start in range(0, len(active), max_batch):
            chunk = active[chunk_start:chunk_start + max_batch]
            samples = []
            for i in chunk:
                spec = specs[i]
                meta, cond, x = dataset.get_shot_window(spec.shot_number, spec.start_i + k * step)
                if k == 0:
                    times[i]['t_start'] = float(meta['start'])
                if k == spec.n_windows - 1:
                    times[i]['t_end'] = float(meta['end'])
                if buffers[i]:
                    cond = {**cond, 'x_history': _assemble_history(cond, buffers[i], history_length, clamp_history)}
                samples.append((meta, cond, x))
            _, cond_b, x_b = default_collate(samples)
            cond_b = {
                key: value.to(device=device, dtype=torch.float32 if value.is_floating_point() else None)
                for key, value in cond_b.items()
            }
            generated = _generate_window_batch(model, cond_b, x_b.size(), n_steps)
            generated = generated.to('cpu', torch.float32).numpy()
            drift_min = min(drift_min, float(generated.min()))
            drift_max = max(drift_max, float(generated.max()))
            for j, i in enumerate(chunk):
                is_last = k == specs[i].n_windows - 1
                buffers[i].append(generated[j] if is_last else generated[j][:, :step])
            progress.update(len(chunk))
    progress.close()
    logger.info("Generated value range across all rollouts: [%.3f, %.3f] (data space is [0,1]).", drift_min, drift_max)
    results = []
    for i, spec in enumerate(specs):
        generated_x = np.concatenate(buffers[i], axis=-1)
        assert generated_x.shape[-1] == spec.total_length
        results.append(RolloutResult(spec, generated_x, times[i]['t_start'], times[i]['t_end']))
    return results


def label_rollout(data_module: FusionShotDataModule, result: RolloutResult):
    """Run the FNOLSTM surrogate mode classifier on a rollout and its real counterpart.

    Uses the same machinery as the window test path (full real pre-rollout history
    prepended, PD channel only, denormalized to physical units) but with the rollout's
    total length as the future span. The sliding-window classifier handles arbitrary
    lengths, carrying its LSTM state across the whole trace. Rollouts are labeled one
    at a time: padding different-length rollouts into one batch would leak garbage
    labels into the shorter ones.

    Fills result.surr_labels_gen / surr_labels_real, both (history_length + total_length,)
    in the UNSHIFTED 0=L,1=D,2=H convention.
    """
    cols_x = list(data_module.cols.x)
    pd_index = cols_x.index("PD")
    history_length = data_module.history_length
    spec = result.spec
    T = result.generated_x.shape[-1]

    full_history = data_module.get_full_history(int(spec.shot_number), int(spec.start_i))
    full_history = data_module.denormalize(full_history, to_device=evaluate_modes.DEVICE)
    shot_data = data_module.data[data_module.data['ShotNum'] == spec.shot_number]
    real_future = shot_data[cols_x].iloc[spec.start_i:spec.start_i + T].values.T
    real_future = data_module.denormalize(real_future, to_device=evaluate_modes.DEVICE)
    generated = data_module.denormalize(result.generated_x, to_device=evaluate_modes.DEVICE)

    gen_pd = torch.cat((full_history, generated), dim=-1)[pd_index].float()
    real_pd = torch.cat((full_history, real_future), dim=-1)[pd_index].float()
    # 2-D timeline (T, batch=1): index 0 = rollout start, history indices negative
    timeline = np.arange(-spec.start_i, T)[:, None]
    surr_gen, surr_real = get_mode_predictions_batched_window(
        pd_rollout_pred=gen_pd.unsqueeze(0),
        pd_rollout_target=real_pd.unsqueeze(0),
        timeline=timeline,
        history_length=history_length,
        seq_length=T,
    )
    result.surr_labels_gen = surr_gen[0].astype(np.int16)
    result.surr_labels_real = surr_real[0].astype(np.int16)


def label_rollout_group(data_module: FusionShotDataModule, results: list[RolloutResult]):
    """Label a whole group of rollouts sharing (shot_number, start_i) in one FNOLSTM call.

    All samples at the same start point have identical length and real trace (only
    sample_idx differs), so the real trace is computed once and the generated traces
    are batched together, instead of calling label_rollout per rollout. The real trace
    is replicated across the batch so the existing (pred, target) equal-batch call can
    be reused as-is; FNOLSTM is cheap next to the flow model's Euler integration, so
    the redundant real-trace passes are worth it against the per-call Python/tensor
    overhead of labeling one rollout at a time. This is what keeps a high n_samples
    rollout evaluation from making labeling (rather than generation) the bottleneck.

    Fills surr_labels_gen / surr_labels_real (both (history_length + T,), UNSHIFTED
    0=L,1=D,2=H) on every result in the group, in place.
    """
    cols_x = list(data_module.cols.x)
    pd_index = cols_x.index("PD")
    history_length = data_module.history_length
    spec0 = results[0].spec
    T = results[0].generated_x.shape[-1]
    assert all(
        r.spec.shot_number == spec0.shot_number and r.spec.start_i == spec0.start_i
        and r.generated_x.shape[-1] == T for r in results
    ), "label_rollout_group requires every result to share (shot_number, start_i) and length"

    full_history = data_module.get_full_history(int(spec0.shot_number), int(spec0.start_i))
    full_history = data_module.denormalize(full_history, to_device=evaluate_modes.DEVICE)
    shot_data = data_module.data[data_module.data['ShotNum'] == spec0.shot_number]
    real_future = shot_data[cols_x].iloc[spec0.start_i:spec0.start_i + T].values.T
    real_future = data_module.denormalize(real_future, to_device=evaluate_modes.DEVICE)
    real_pd = torch.cat((full_history, real_future), dim=-1)[pd_index].float()

    gen_pd_batch = torch.stack([
        torch.cat(
            (full_history, data_module.denormalize(r.generated_x, to_device=evaluate_modes.DEVICE)), dim=-1
        )[pd_index].float()
        for r in results
    ])
    real_pd_batch = real_pd.unsqueeze(0).repeat(len(results), 1)
    # 2-D timeline (T, batch=1): shared by every sample since they all start at the same index
    timeline = np.arange(-spec0.start_i, T)[:, None]
    surr_gen, surr_real = get_mode_predictions_batched_window(
        pd_rollout_pred=gen_pd_batch,
        pd_rollout_target=real_pd_batch,
        timeline=timeline,
        history_length=history_length,
        seq_length=T,
    )
    for i, r in enumerate(results):
        r.surr_labels_gen = surr_gen[i].astype(np.int16)
        r.surr_labels_real = surr_real[i].astype(np.int16)


def _result_attrs(result: RolloutResult, dataset: FusionShotDataset, step) -> dict:
    spec = result.spec
    return {
        'start_frac': spec.start_frac,
        'start_i': spec.start_i,
        't_start': result.t_start,
        't_end': result.t_end,
        'n_windows': spec.n_windows,
        'seq_length': dataset.seq_length,
        'history_length': dataset.history_length,
        'step': step,
    }


def summarize_rollouts(results: list[RolloutResult], data_module: FusionShotDataModule, skipped=()) -> dict:
    """Small in-run summary over all rollouts.

    Deliberately minimal: moments gaps, surrogate label agreement/dice, drift out of
    the [0,1] data range. The horizon-resolved analysis (errors as a function of how
    far the rollout has run) lives in eval_notebooks/rollout_analysis.py, computed
    from the cache. Existing PeakMetric/ModeTransitionMetric are NOT reused here:
    they assume a real 256-sample history window and bucket by its true mode, which
    a rollout does not have.
    """
    cols_x = list(data_module.cols.x)
    gen_all = np.concatenate([r.generated_x for r in results], axis=-1)
    real_parts = []
    for r in results:
        shot_data = data_module.data[data_module.data['ShotNum'] == r.spec.shot_number]
        T = r.generated_x.shape[-1]
        real_parts.append(shot_data[cols_x].iloc[r.spec.start_i:r.spec.start_i + T].values.T)
    real_all = np.concatenate(real_parts, axis=-1)
    metrics = {
        'n_rollouts': len(results),
        'n_skipped': len(skipped),
        'mean_n_windows': float(np.mean([r.spec.n_windows for r in results])),
        'max_n_windows': int(max(r.spec.n_windows for r in results)),
        'frac_outside_01': float(((gen_all < 0) | (gen_all > 1)).mean()),
    }
    for ch, name in enumerate(cols_x):
        # Absolute moment errors |mu_gen - mu_real| and |sigma_gen - sigma_real| over the
        # pooled generated vs real samples, in normalized [0,1] units. Same idea as the
        # window moment errors, but pooled over whole rollouts.
        metrics[f'abs_mean_err/{name}'] = float(abs(gen_all[ch].mean() - real_all[ch].mean()))
        metrics[f'abs_std_err/{name}'] = float(abs(gen_all[ch].std() - real_all[ch].std()))
    # Surrogate label agreement over the generated span only (labels also cover W_H)
    gen_labels = np.concatenate([r.surr_labels_gen[-r.generated_x.shape[-1]:] for r in results])
    real_labels = np.concatenate([r.surr_labels_real[-r.generated_x.shape[-1]:] for r in results])
    metrics['label_agreement'] = float((gen_labels == real_labels).mean())
    for mode_value, mode_name in enumerate(["L", "D", "H"]):  # unshifted surrogate convention
        intersection = int(((gen_labels == mode_value) & (real_labels == mode_value)).sum())
        denominator = int((gen_labels == mode_value).sum() + (real_labels == mode_value).sum())
        metrics[f'dice/{mode_name}'] = 2 * intersection / denominator if denominator else float('nan')
    return metrics


def build_rollout_records(
    results: list[RolloutResult], data_module: FusionShotDataModule, step, shots=None
) -> list[dict]:
    """Attach the real context (observables, controls, timeline) to rollouts for plotting.

    Args:
        results: Rollouts to convert (must be labeled).
        data_module: Source of the real (normalized) traces.
        step: Samples advanced per generation (for window boundary marks).
        shots: Optional shot filter (e.g. rollout.html_shots).

    Returns:
        List of record dicts as consumed by src.plotters.rollout_plots.
    """
    cols_x = list(data_module.cols.x)
    cols_c = list(data_module.cols.get('c', []))
    history_length = data_module.history_length
    records = []
    for r in results:
        if shots is not None and r.spec.shot_number not in list(shots):
            continue
        shot_data = data_module.data[data_module.data['ShotNum'] == r.spec.shot_number]
        T = r.generated_x.shape[-1]
        i0, i1 = r.spec.start_i - history_length, r.spec.start_i + T
        records.append({
            'shot_number': r.spec.shot_number,
            'start_idx': r.spec.start_i,
            'sample_idx': r.spec.sample_idx,
            'start_frac': r.spec.start_frac,
            't_start': r.t_start,
            't_end': r.t_end,
            'n_windows': r.spec.n_windows,
            'history_length': history_length,
            'step': step,
            'generated_x': r.generated_x,
            'surr_labels_gen': r.surr_labels_gen,
            'surr_labels_real': r.surr_labels_real,
            'real_x': shot_data[cols_x].iloc[i0:i1].values.T,
            'real_c': shot_data[cols_c].iloc[i0:i1].values.T,
            'times': shot_data.index[i0:i1].values,
        })
    return records


def build_rollout_groups(
    results: list[RolloutResult], data_module: FusionShotDataModule, step, shots=None, max_samples=None
) -> list[dict]:
    """Group rollouts by (shot, start point) for the interactive browser.

    With n_samples > 1, several stochastic rollouts share the same shot and start
    index; the browser overlays them in one dropdown entry instead of giving every
    sample_idx its own entry. Real context (observables, controls, timeline) and the
    real surrogate labels are identical across a group's samples, so they are computed
    once. Samples are sorted by sample_idx and truncated to max_samples (None = all)
    to keep the figure and the HTML file size in check when n_samples is large.

    Args:
        results: Rollouts to group (must be labeled).
        data_module: Source of the real (normalized) traces.
        step: Samples advanced per generation (for window boundary marks).
        shots: Optional shot filter (e.g. rollout.html_shots).
        max_samples: Cap on stochastic samples shown per (shot, start point).

    Returns:
        List of dicts: shared fields (shot_number, start_idx, start_frac, t_start,
        t_end, n_windows, history_length, step, real_x, real_c, times,
        surr_labels_real) plus 'samples': [{sample_idx, generated_x, surr_labels_gen}, ...].
    """
    cols_x = list(data_module.cols.x)
    cols_c = list(data_module.cols.get('c', []))
    history_length = data_module.history_length
    filtered = [r for r in results if shots is None or r.spec.shot_number in list(shots)]
    filtered.sort(key=lambda r: (r.spec.shot_number, r.spec.start_i, r.spec.sample_idx))
    groups = []
    for (shot_number, start_i), members in itertools.groupby(
        filtered, key=lambda r: (r.spec.shot_number, r.spec.start_i)
    ):
        members = list(members)
        if max_samples:
            members = members[:max_samples]
        r0 = members[0]
        shot_data = data_module.data[data_module.data['ShotNum'] == shot_number]
        T = r0.generated_x.shape[-1]
        i0, i1 = start_i - history_length, start_i + T
        groups.append({
            'shot_number': shot_number,
            'start_idx': start_i,
            'start_frac': r0.spec.start_frac,
            't_start': r0.t_start,
            't_end': r0.t_end,
            'n_windows': r0.spec.n_windows,
            'history_length': history_length,
            'step': step,
            'real_x': shot_data[cols_x].iloc[i0:i1].values.T,
            'real_c': shot_data[cols_c].iloc[i0:i1].values.T,
            'times': shot_data.index[i0:i1].values,
            'surr_labels_real': r0.surr_labels_real,
            'samples': [
                {'sample_idx': r.spec.sample_idx, 'generated_x': r.generated_x, 'surr_labels_gen': r.surr_labels_gen}
                for r in members
            ],
        })
    return groups


def load_results_from_cache(cache: RolloutHDFCache) -> list[RolloutResult]:
    """Rebuild RolloutResults for every rollout in a cache (notebook entry point)."""
    results = []
    for shot_number, start_idx, sample_idx in cache.list_rollouts():
        entry = cache.get_rollout(shot_number, start_idx, sample_idx)
        spec = RolloutSpec(
            shot_number, float(entry.get('start_frac', np.nan)), start_idx,
            int(entry['n_windows']), entry['generated_x'].shape[-1], sample_idx
        )
        results.append(
            RolloutResult(
                spec, entry['generated_x'], float(entry['t_start']), float(entry['t_end']),
                entry['surr_labels_gen'], entry['surr_labels_real']
            )
        )
    return results


def run_rollouts(model, data_module: FusionShotDataModule, rollout_conf):
    """Orchestrate rollout evaluation; never raises (mirrors animate_window_set).

    cache_mode 'create': generate + label + cache (resumable, existing rollouts skipped).
    cache_mode 'use': read rollouts back from the cache, skipping generation, so metrics
    and plots can be redone without a GPU.
    """
    try:
        return _run_rollouts_inner(model, data_module, rollout_conf)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.exception(e)


def _run_rollouts_inner(model, data_module: FusionShotDataModule, rollout_conf):
    dataset = data_module.test_dataset
    step = rollout_conf.get('step') or dataset.seq_length
    cache_mode = rollout_conf.get('cache_mode', 'create')
    n_samples = rollout_conf.get('n_samples', 1)
    cache_name = rollout_conf.cache_name
    test_cache = getattr(model, 'test_cache', None)
    if test_cache is not None and test_cache.cache_filename == cache_name:
        # Same file would mix the window schema ({shot}/{start}: datasets) with the rollout
        # schema ({shot}/{start}/{sample}: group), breaking readers of both.
        cache_name = cache_name + '_rollout'
        logger.warning(
            "rollout.cache_name equals test_cache_name; using '%s' instead to keep the schemas apart.", cache_name
        )
    specs, skipped = compute_rollout_specs(dataset, rollout_conf.start_fractions, n_samples, step)

    if cache_mode == 'use':
        cache = RolloutHDFCache(cache_name, mode='r')
        results = _load_results_from_cache(cache, specs)
    else:
        cache = RolloutHDFCache(cache_name, mode='a')
        todo = [s for s in specs if not cache.has(s.shot_number, s.start_i, s.sample_idx)]
        cached = [s for s in specs if s not in todo]
        logger.info("Generating %d rollouts (%d already cached).", len(todo), len(cached))
        results = _generate_rollouts(
            model, dataset, todo,
            n_steps=rollout_conf.get('n_steps') or model.flow_steps,
            max_batch=rollout_conf.get('max_batch', 128),
            step=step,
            clamp_history=rollout_conf.get('clamp_history', False),
        ) if todo else []
        # Grouped by (shot, start_i): every sample at a start point shares length and
        # real trace, so labeling batches per group instead of one FNOLSTM call per
        # rollout (label_rollout_group). Group count = n_shots * n_fractions,
        # independent of n_samples, which is what keeps labeling cheap when n_samples
        # is large. `results` is contiguous per group because compute_rollout_specs
        # emits sample_idx as the innermost loop.
        grouped = [
            list(members) for _, members in
            itertools.groupby(results, key=lambda r: (r.spec.shot_number, r.spec.start_i))
        ]
        for group in tqdm(grouped, desc="Labeling rollout groups"):
            label_rollout_group(data_module, group)
            for result in group:
                cache.set_rollout(
                    result.spec.shot_number, result.spec.start_i, result.spec.sample_idx,
                    result.generated_x, result.surr_labels_gen, result.surr_labels_real,
                    attrs=_result_attrs(result, dataset, step),
                )
        cache.set_root_attrs(
            start_fractions=list(rollout_conf.start_fractions),
            n_samples=n_samples,
            cols_x=list(data_module.cols.x),
            run_name=str(wandb.run.name) if wandb.run is not None else '',
        )
        if cached:  # pull previously cached rollouts back in for metrics/plots
            results += _load_results_from_cache(RolloutHDFCache(cache_name, mode='r'), cached)

    if results:
        summary = summarize_rollouts(results, data_module, skipped)
        logger.info("Rollout summary: %s", summary)
        if wandb.run is not None and not wandb.run.disabled:
            wandb.log({f'rollout/final/{k}': v for k, v in summary.items()}, commit=True)
        summary['skipped'] = list(skipped)
        cache.save_json_friend(summary)
        _write_rollout_browser(results, data_module, rollout_conf, step)
        if rollout_conf.get('analysis', True):
            _write_horizon_analysis(results, data_module, step)
    logger.info("Rollout evaluation done: %d rollouts, %d skipped combinations.", len(results), len(skipped))
    return results, skipped


def _write_horizon_analysis(results, data_module, step):
    """Export the horizon figures/tables in-run, so a finished run comes back with them.

    Same code path as eval_notebooks/rollout_analysis.py (which can redo or extend them
    from the cache later); output mirrors the evaluate_window_set convention of
    output/pdfplots/{run_name}/.
    """
    from src.config import get_current_config
    from src.plotters.rollout_horizon import export_horizon_analysis
    C = get_current_config()
    cols_x = list(data_module.cols.x)
    run_name = C.get('run_name', None) or (wandb.run.name if wandb.run is not None else 'standalone')
    records = build_rollout_records(results, data_module, step)
    export_horizon_analysis(
        [(str(run_name), records)],
        channel_names=cols_x,
        pd_index=cols_x.index("PD"),
        elm_prominence=float(C.evaluation.peaks.elm_pd_prominence),
        pdf_dir=Path(f"output/pdfplots/{run_name}/rollout_analysis"),
        wandb_prefix="rollout/horizon",
    )


def _write_rollout_browser(results, data_module, rollout_conf, step):
    """Render the interactive rollout browser for the configured html_shots subset.

    Stochastic samples at the same (shot, start point) are overlaid in one dropdown
    entry (build_rollout_groups), capped at rollout.plot_samples per group so the
    figure and HTML file stay readable when n_samples is large.
    """
    import plotly.io as pio

    from src.plotters.rollout_plots import rollout_browser_plotly
    html_shots = rollout_conf.get('html_shots')
    if not html_shots:
        return
    max_samples = rollout_conf.get('plot_samples', 3)
    groups = build_rollout_groups(results, data_module, step, shots=html_shots, max_samples=max_samples)
    if not groups:
        logger.warning("No rollouts matched html_shots %s; browser not written.", list(html_shots))
        return
    fig = rollout_browser_plotly(groups, list(data_module.cols.x), list(data_module.cols.get('c', [])))
    run_name = wandb.run.name if wandb.run is not None else 'standalone'
    out_dir = Path(rollout_conf.get('html_dir') or f"output/htmlplots/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, out_dir / "rollouts.html")
    logger.info("Wrote rollout browser html to %s", out_dir / "rollouts.html")
    if wandb.run is not None and not wandb.run.disabled:
        wandb.log({"rollout/browser": wandb.Html(pio.to_html(fig))})


def _load_results_from_cache(cache: RolloutHDFCache, specs: list[RolloutSpec]) -> list[RolloutResult]:
    results = []
    for spec in specs:
        try:
            entry = cache.get_rollout(spec.shot_number, spec.start_i, spec.sample_idx)
        except KeyError:
            logger.warning("Rollout %s/%s/%s not in cache, skipping.", spec.shot_number, spec.start_i, spec.sample_idx)
            continue
        results.append(
            RolloutResult(
                spec, entry['generated_x'], float(entry['t_start']), float(entry['t_end']),
                entry['surr_labels_gen'], entry['surr_labels_real']
            )
        )
    return results


if __name__ == "__main__":
    # Smoke test with an untrained model on CPU: exercises chaining, truncation,
    # labeling alignment, and cache writing on two short test shots.
    # Run as: PYTHONPATH=. python src/rollout.py
    import sys

    from omegaconf import OmegaConf

    import src.models
    from src.config import load_config_from_file

    logging.basicConfig(level=logging.INFO)
    # Metric constructors resolve the full config, which needs run_name (normally a CLI
    # arg or wandb); provide one so the standalone smoke run resolves too.
    if not any(arg.startswith('run_name=') for arg in sys.argv[1:]):
        sys.argv.append('run_name=rollout_smoke')
    C = load_config_from_file(as_omega=True)
    data_module = FusionShotDataModule(**C.data)
    data_module.prepare_data()
    data_module.setup()
    model = getattr(src.models, C.model.Class)(**C.model.params)
    # Restrict to two shots so the smoke test finishes quickly on CPU
    data_module.test_dataset.data = data_module.test_dataset.data[
        data_module.test_dataset.data['ShotNum'].isin([73368, 64770])]
    data_module.test_dataset.precompute_indices()
    smoke_conf = OmegaConf.create({
        'start_fractions': [0.5, 0.9],
        'n_samples': 1,
        'n_steps': 3,
        'max_batch': 8,
        'cache_name': 'smoke_rollout',
        'cache_mode': 'create',
        'html_shots': [73368, 64770],
        'html_dir': 'output/testplots/rollout_smoke',  # test plots stay out of the production dirs
    })
    results, skipped = _run_rollouts_inner(model, data_module, smoke_conf)
    for r in results:
        print(r.spec, r.generated_x.shape, r.surr_labels_gen.shape, f"t=[{r.t_start:.3f}, {r.t_end:.3f}]s")
    print(f"{len(results)} rollouts, {len(skipped)} skipped -> cache 'smoke_rollout'")
