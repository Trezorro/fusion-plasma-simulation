"""Optional pre-detection smoothing for peak analysis.

Peak counts at a low prominence threshold are dominated by sensor noise: a 0.001-prominence
`find_peaks` pass on a normalized PD trace returns tens of peaks per 256-sample window, most of
which are not physical events. Smoothing the trace before detection is a way to separate "the
model gets the noise statistics wrong" from "the model gets the ELMs wrong" without pushing the
prominence threshold up, which instead removes small *real* peaks.

The filter is applied identically to the real and the generated trace, so it cannot flatter
either one; it changes what counts as a peak, not who is being measured.

A filter is named by a spec: either None (no filtering), a string naming a filter with its
defaults, or a dict {"kind": ..., <params>}. A per-channel dict of specs is resolved by
`resolve_filter(spec, channel_name)`, mirroring how the per-channel prominence thresholds work.

sigma is in samples. At the 10 kHz sample rate of this dataset, sigma=3 is 0.3 ms, roughly an
order of magnitude below the ~7 ms width of an ELM burst on PD.
"""
import numpy as np

GAUSSIAN_SIGMA_DEFAULT = 3.0
MEDIAN_SIZE_DEFAULT = 5


def _identity(trace, **_):
    return np.asarray(trace, dtype=float)


def _gaussian(trace, sigma=GAUSSIAN_SIGMA_DEFAULT, truncate=4.0, **_):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(np.asarray(trace, dtype=float), sigma=float(sigma),
                             truncate=float(truncate), mode="nearest")


def _median(trace, size=MEDIAN_SIZE_DEFAULT, **_):
    from scipy.ndimage import median_filter
    return median_filter(np.asarray(trace, dtype=float), size=int(size), mode="nearest")


FILTERS = {"none": _identity, "gaussian": _gaussian, "median": _median}


def normalize_spec(spec):
    """A filter spec in canonical dict form, or None for no filtering.

    None / "none" -> None; "gaussian" -> {"kind": "gaussian"}; a dict passes through with its
    kind validated. The canonical form is what goes into a cache signature, so it must be
    JSON-serializable and stable.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = {"kind": spec}
    spec = dict(spec)
    kind = spec.get("kind", "none")
    if kind not in FILTERS:
        raise ValueError(f"unknown signal filter {kind!r}; known: {sorted(FILTERS)}")
    if kind == "none":
        return None
    # Defaults are materialized here rather than left to the filter function's signature, so
    # the canonical spec fully describes what was applied: the cache signature and the plot
    # labels then never hide a parameter.
    defaults = {"gaussian": {"sigma": GAUSSIAN_SIGMA_DEFAULT}, "median": {"size": MEDIAN_SIZE_DEFAULT}}
    return {**defaults.get(kind, {}), **spec}


def resolve_filter(spec, channel_name=None):
    """Canonical spec for one channel.

    `spec` may be a single spec applying to every channel, or a dict keyed by channel name
    (with an optional "default" entry). A channel-keyed dict is told apart from a single spec
    by the absence of a "kind" key.
    """
    if isinstance(spec, dict) and "kind" not in spec:
        spec = spec.get(channel_name, spec.get("default"))
    return normalize_spec(spec)


def apply_filter(trace, spec, channel_name=None):
    """Filter one 1-D trace. Returns it unchanged (as float array) when no filter applies."""
    resolved = resolve_filter(spec, channel_name)
    if resolved is None:
        return np.asarray(trace, dtype=float)
    kind = resolved["kind"]
    params = {k: v for k, v in resolved.items() if k != "kind"}
    return FILTERS[kind](trace, **params)


def filter_label(spec, channel_name=None) -> str:
    """Short human-readable tag for a resolved spec, for plot titles and table annotations."""
    resolved = resolve_filter(spec, channel_name)
    if resolved is None:
        return "raw"
    params = ",".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                      for k, v in sorted(resolved.items()) if k != "kind")
    return f"{resolved['kind']}({params})" if params else resolved["kind"]
