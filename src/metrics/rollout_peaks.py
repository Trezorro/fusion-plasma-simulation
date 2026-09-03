"""Per-rollout peak and mode statistics.

The measurement layer behind the rollout tables (`eval_notebooks/rollout_tables.py`): given
one rollout record from the HDF cache, produce the per-window rows and the pooled
optimal-transport rows that everything downstream aggregates. Nothing here reads a config or
a global; the caller passes a `PeakSpec` and the thresholds it wants measured.

Peaks are detected once over the whole rollout, with the real history prepended as left
context, and the generation-window boundaries are used only to attribute each peak to a
window. `find_peaks` derives a peak's prominence by walking left and right until it meets a
higher sample, and it measures the width by walking down from the summit; both walks stop at
the array boundary. Detecting inside a 256-sample window would therefore measure a boundary
peak against whatever that window happens to contain rather than against its true surrounding
trough, which can push a genuine peak below the threshold or lift a minor one above it, and
clips the widths of any peak within one width of an edge. At the ELM scale the widths run to
~75 samples against a 256-sample window, roughly a third of the span. The effect does not
cancel across models either, because a model with a different peak density has a different
chance of a peak landing on a boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.metrics.metrics import PeakProps
from src.metrics.ot_peak_error import ot_peak_error


@dataclass(frozen=True)
class PeakSpec:
    """Everything the measurement needs that is not a threshold or a trace.

    sample_rate: Hz, for converting sample counts to milliseconds.
    seq_length: the generated window length L; the rollout layout is checked against it.
    rel_height: `find_peaks` width level. 0.5 is full width at half maximum, i.e. the walk
        stops halfway down the peak's own prominence, so the number describes the peak's
        shape. The scipy default of 1.0 measures at the peak's base, where the walk is bounded
        only by the next higher sample: on a whole rollout every peak in a cluster sharing two
        bounding maxima inherits the same enormous width (measured: dozens of real PD peaks at
        437 ms on an 845 ms rollout), and those few values then dominate the Wasserstein
        distance.
    width_clip: cap on a width in samples, or None. The metric is defined per generated
        window, so a feature wider than one window is not a within-window event.
    label_spill: the surrogate classifier assigns a label using a window reaching ~15 samples
        past the label position, so the Dice window is extended backwards by that much,
        matching flow.py.
    fallback_prominence: the prominence used for a channel that a per-channel threshold does
        not name; see `resolve_channel_thresholds`.
    """

    sample_rate: float
    seq_length: int
    rel_height: float = 0.5
    width_clip: int | None = None
    label_spill: int = 15
    fallback_prominence: float = 0.01


def resolve_channel_thresholds(thresholds, channel_name, fallback) -> dict:
    """Threshold name -> prominence for one channel.

    A threshold is either one number for every channel or a per-channel dict: PD and DML do
    not sit on the same scale, so a "large peaks only" pass needs a different number on each.
    """
    return {
        name: (float(t.get(channel_name, fallback)) if isinstance(t, dict) else float(t))
        for name, t in thresholds.items()
    }


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


def w1(a, b) -> float:
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(np.asarray(a, dtype=float), np.asarray(b, dtype=float)))


def subset_peaks(peaks: PeakProps, mask) -> PeakProps:
    """The PeakProps restricted to a boolean mask over its peaks."""
    return PeakProps(
        X=peaks.X[mask], Y=peaks.Y[mask], prominences=peaks.prominences[mask],
        bases=peaks.bases[mask], left_ips=peaks.left_ips[mask], right_ips=peaks.right_ips[mask],
    )


def detect_windows(trace, context, prominence, offset, step, n_windows, rel_height=1.0) -> list[PeakProps]:
    """Detect peaks once on the whole trace, then bucket them into the K rollout windows.

    Each peak is assigned to the window containing its position, so the rollout-depth index k
    still means "the k-th autoregressive generation". `context` is the real history prepended
    to the left so the first window has proper context too; peaks inside it are discarded
    after detection. The single remaining edge is the end of the rollout, where no further
    samples exist for either trace.

    Args:
        trace: 1-D signal over the rollout span.
        context: samples immediately preceding it (the real history), used for context only.
        prominence: detection threshold in normalized [0,1] units.
        offset: len(context); positions are re-based by this before bucketing.
        step: window stride, which must equal the window length for bucketing to be well defined.
        n_windows: K.

    Returns:
        A list of K PeakProps, one per window.
    """
    peaks = PeakProps.from_find_peaks(np.concatenate([context, trace]), prominence=prominence, rel_height=rel_height)
    positions = np.asarray(peaks.X) - offset
    return [
        subset_peaks(peaks, (positions >= k * step) & (positions < (k + 1) * step))
        for k in range(n_windows)
    ]


def peak_widths(peaks: PeakProps, clip):
    """Peak widths in samples, optionally capped at `clip`; see PeakSpec.width_clip."""
    w = np.asarray(peaks.widths, dtype=float)
    return w if clip is None else np.minimum(w, float(clip))


def channel_traces(generated, real, real_full, channel_names, channel_name, history_length):
    """(generated, real, context) traces for one channel.

    The rollout was seeded with the real history, so the real history is the correct left
    context for the generated trace as well as for the real one.
    """
    ci = channel_names.index(channel_name)
    return generated[ci], real[ci], real_full[ci, :history_length]


def peak_stats(gen_peaks: PeakProps, real_peaks: PeakProps, sample_rate, window_ms, width_clip=None) -> dict:
    """The per-window peak comparison for one pair of peak sets.

    `window_ms` is the window duration in milliseconds, used for the peak rates. Rates are the
    same information as the counts but in physical units, which is what makes them comparable
    across diagnostics with different characteristic peak densities.

    Both Wasserstein columns are reported only where *both* peak sets are non-empty.
    `PeakProps.__sub__` instead substitutes the mean property value of the non-empty side when
    one side is empty, which is not a distance and rewards models that emit nothing: on the PD
    ELM-scale threshold every deterministic baseline produces zero peaks in every window and
    collects that sentinel, scoring better than the flow model, which is the only one
    producing ELM peaks at all. `miss_rate` carries that information explicitly instead of
    smuggling it into the distance. The sentinel-inclusive values stay available as `*_legacy`
    so the older numbers remain reproducible.
    """
    n_gen, n_real = gen_peaks.num_peaks(), real_peaks.num_peaks()
    both = n_gen > 0 and n_real > 0
    legacy = gen_peaks - real_peaks
    return {
        "n_peaks_gen": n_gen,
        "n_peaks_real": n_real,
        "rate_gen_per_ms": n_gen / window_ms,
        "rate_real_per_ms": n_real / window_ms,
        "count_error": n_gen - n_real,
        "abs_count_error": abs(n_gen - n_real),
        # Relative to the real count, so a stratum with denser ELMs is not automatically
        # scored as worse. Undefined (and skipped, like the Wasserstein columns) where the
        # real window has no peaks at all to be relative to.
        "rel_count_error": abs(n_gen - n_real) / n_real if n_real > 0 else np.nan,
        "both_nonempty": float(both),
        "miss_rate": float(n_real > 0 and n_gen == 0),
        "spurious_rate": float(n_real == 0 and n_gen > 0),
        "prominence_w1": w1(gen_peaks.prominences, real_peaks.prominences) if both else np.nan,
        "width_w1_ms": (w1(peak_widths(gen_peaks, width_clip), peak_widths(real_peaks, width_clip))
                        * 1000.0 / sample_rate) if both else np.nan,
        "prominence_w1_legacy": float(legacy.prominence),
        "width_w1_ms_legacy": float(legacy.width) * 1000.0 / sample_rate,
    }


def _check_layout(record, spec: PeakSpec):
    """The rollout layout assumptions, checked once per record."""
    L = slice_length(record)
    step = int(record["step"])
    if L != spec.seq_length:
        raise ValueError(f"window length {L} from the rollout layout != config seq_length {spec.seq_length}")
    if step != L:
        # With overlapping windows a peak belongs to several of them and `position // step`
        # is no longer an assignment.
        raise ValueError(f"peak attribution assumes non-overlapping windows, but step={step} != L={L}")
    return L, step


def record_slice_rows(record, channel_names, channels, thresholds, spec: PeakSpec) -> list[dict]:
    """One row per (window, channel, threshold) for a single rollout."""
    history_length = int(record["history_length"])
    n_windows = int(record["n_windows"])
    generated = record["generated_x"]
    real_full = record["real_x"]
    real = real_full[:, history_length:]
    labels_gen = np.asarray(record["surr_labels_gen"])
    labels_real = np.asarray(record["surr_labels_real"])

    L, step = _check_layout(record, spec)
    if real.shape[-1] != generated.shape[-1]:
        raise ValueError(
            f"real trace is {real.shape[-1]} samples but the rollout is {generated.shape[-1]}; "
            "the shot ran out before the rollout did and the windows would be misaligned"
        )
    if labels_gen.shape[-1] != history_length + generated.shape[-1]:
        raise ValueError("surrogate label array does not span history + rollout")

    window_ms = 1000.0 * L / spec.sample_rate
    base = {
        "slice_length": L,
        "shot": int(record["shot_number"]),
        "start_idx": int(record["start_idx"]),
        "start_frac": float(record["start_frac"]),
        "sample_idx": int(record["sample_idx"]),
    }
    rows = []
    for channel_name in channels:
        gen_trace, real_trace, context = channel_traces(
            generated, real, real_full, channel_names, channel_name, history_length)
        resolved = resolve_channel_thresholds(thresholds, channel_name, spec.fallback_prominence)
        for threshold_name, threshold in resolved.items():
            args = (context, threshold, history_length, step, n_windows)
            gen_peaks = detect_windows(gen_trace, *args, spec.rel_height)
            real_peaks = detect_windows(real_trace, *args, spec.rel_height)
            # The rel_height=1.0 base-level widths, uncapped, kept only so the pre-fix width
            # numbers stay reproducible from the same parquet. Peak positions, counts and
            # prominences are identical to the pass above; only the widths differ.
            gen_base = detect_windows(gen_trace, *args, 1.0)
            real_base = detect_windows(real_trace, *args, 1.0)
            for k in range(n_windows):
                rows.append({
                    **base,
                    "k": k,
                    "channel": channel_name,
                    "threshold_name": threshold_name,
                    "threshold": threshold,
                    **peak_stats(gen_peaks[k], real_peaks[k], spec.sample_rate, window_ms,
                                 width_clip=spec.width_clip),
                    "width_w1_ms_base": peak_stats(
                        gen_base[k], real_base[k], spec.sample_rate, window_ms)["width_w1_ms"],
                })

    # Dice does not depend on the channel or threshold, so it is computed once per window and
    # broadcast onto the rows.
    dice_by_k = {}
    for k in range(n_windows):
        lo = history_length + k * step - spec.label_spill
        hi = history_length + k * step + L
        dice_by_k[k] = dice_scores(labels_gen[lo:hi], labels_real[lo:hi])
    for row in rows:
        row["dice"], row["dice_macro"] = dice_by_k[row["k"]]
    return rows


def pool_peaks(trace, context, prominence, history_length, step, lo, hi, sample_rate, rel_height,
               peak_mass="prominence"):
    """(times_ms, masses) of every peak in rollout windows lo..hi inclusive, pooled.

    Detection runs once over history+rollout for the same reason it does per window, but the
    window grid is used only to select the pool: within it, a peak keeps its continuous
    position. Times are milliseconds from the start of the pool, so the generated and the real
    set share an origin, which is what makes their displacement meaningful.
    """
    peaks = PeakProps.from_find_peaks(
        np.concatenate([context, trace]), prominence=prominence, rel_height=rel_height)
    positions = np.asarray(peaks.X, dtype=float) - history_length
    keep = (positions >= lo * step) & (positions < (hi + 1) * step)
    times_ms = (positions[keep] - lo * step) * 1000.0 / sample_rate
    if peak_mass == "prominence":
        mass = np.asarray(peaks.prominences, dtype=float)[keep]
    elif peak_mass == "width":
        mass = np.asarray(peaks.widths, dtype=float)[keep] * 1000.0 / sample_rate
    else:
        raise ValueError(f"peak_mass must be 'prominence' or 'width', got {peak_mass!r}")
    return times_ms, mass


def record_pool_rows(record, channel_names, channels, thresholds, spec: PeakSpec, *,
                     bin_sets, lambdas_ms, peak_mass="prominence") -> list[dict]:
    """One row per (depth stratum, channel, threshold, lambda) for a single rollout.

    The pooled counterpart of `record_slice_rows`: no per-window slicing, one unbalanced
    optimal-transport solve over all the peaks of a stratum. A window is 25.6 ms, and an ELM
    displaced across a window boundary counts as a miss in one window and a false alarm in the
    next, which is an artefact of the grid rather than a model error; pooling a whole stratum
    measures the displacement continuously instead.

    `bin_sets` maps a bin-set name to its [(label, first k, last k)] bins. A stratum the
    rollout does not reach would be pooled from fewer windows than the same stratum on another
    shot, so bin sets the rollout is too short for are skipped: the shots that do fit are
    exactly the subset the caller pairs with that set. The default set (the first entry) is
    capped at the shortest test shot's budget, so failing to fit *it* means the bins and the
    caches disagree, and that raises.
    """
    history_length = int(record["history_length"])
    n_windows = int(record["n_windows"])
    generated = record["generated_x"]
    real_full = record["real_x"]
    real = real_full[:, history_length:]

    L, step = _check_layout(record, spec)
    default_set = next(iter(bin_sets))
    needed = max(hi for _, _, hi in bin_sets[default_set]) + 1
    if n_windows < needed:
        raise ValueError(
            f"shot {record['shot_number']} has {n_windows} rollout windows but the default "
            f"bin set needs {needed}; lower the depth budget or drop the shot")
    usable = [name for name, bins in bin_sets.items()
              if n_windows >= max(hi for _, _, hi in bins) + 1]

    base = {
        "shot": int(record["shot_number"]),
        "start_idx": int(record["start_idx"]),
        "start_frac": float(record["start_frac"]),
        "sample_idx": int(record["sample_idx"]),
        "slice_length": L,
    }
    rows = []
    for channel_name in channels:
        gen_trace, real_trace, context = channel_traces(
            generated, real, real_full, channel_names, channel_name, history_length)
        resolved = resolve_channel_thresholds(thresholds, channel_name, spec.fallback_prominence)
        for threshold_name, threshold in resolved.items():
            for bin_set, (stratum, lo, hi) in [(n, b) for n in usable for b in bin_sets[n]]:
                args = (context, threshold, history_length, step, lo, hi, spec.sample_rate,
                        spec.rel_height, peak_mass)
                gen_t, gen_m = pool_peaks(gen_trace, *args)
                real_t, real_m = pool_peaks(real_trace, *args)
                n_gen, n_real = len(gen_t), len(real_t)
                # The peaks are detected once and then priced at every lam: only the transport
                # solve depends on lam, and it is the cheap half.
                for lam in lambdas_ms:
                    err = ot_peak_error(gen_t, gen_m, real_t, real_m, lam)
                    rows.append({
                        **base,
                        "lam_ms": float(lam),
                        "bin_set": bin_set,
                        "stratum": stratum, "k_lo": lo, "k_hi": hi,
                        "pool_ms": 1000.0 * (hi - lo + 1) * L / spec.sample_rate,
                        "channel": channel_name,
                        "threshold_name": threshold_name, "threshold": threshold,
                        "ot_error": err.total,
                        "ot_error_rel": err.relative,
                        "ot_loc": err.loc,
                        "ot_missed_mass": err.missed_mass,
                        "ot_false_mass": err.false_mass,
                        "ot_gen_mass": err.gen_mass,
                        "ot_real_mass": err.real_mass,
                        # Which side of the budget the error comes from: how much of the real
                        # mass went unmatched, and how much generated mass was spurious
                        # relative to the real total. Both are on the same denominator so they
                        # are comparable, and with ot_loc they account for all of ot_error.
                        "ot_missed_frac": err.missed_mass / err.real_mass if err.real_mass else np.nan,
                        "ot_false_frac": err.false_mass / err.real_mass if err.real_mass else np.nan,
                        "pool_n_peaks_gen": n_gen,
                        "pool_n_peaks_real": n_real,
                        "pool_count_error": n_gen - n_real,
                        "pool_abs_count_error": abs(n_gen - n_real),
                    })
    return rows
