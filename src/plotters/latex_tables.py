"""Stacked LaTeX result tables.

The rendering half of the rollout tables. This module knows how a metric is displayed (header,
decimals, direction, scale) and how a stacked table is assembled; it knows nothing about which
metrics the paper reports or what the captions say. Those stay in
`eval_notebooks/rollout_tables.py`, next to the configuration they belong to.

A stacked table is one horizontally repeated set of column groups (a diagnostic, or a
metric-free label) under a shared header, with the model rows repeated once per block. A block
is a start fraction in the main table and a depth stratum in the depth table; `block_key`
selects which column the block value is matched against.
"""

from __future__ import annotations

import numpy as np

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
RANK_ON_ABS = {"count_error", "pool_count_error"}
NEVER_RANKED = {"rate_gen_per_ms", "rate_real_per_ms", "pool_n_peaks_real"}


def fmt(mean, std, decimals, scale=1.0):
    """`mean_{\\pm std}` in the metric's own units, or `--` where there is no value."""
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    body = f"{mean * scale:.{decimals}f}"
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return body
    return rf"{body}_{{\pm{std * scale:.{decimals}f}}}"


def rank_marks(values, metric, tol=1e-12):
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


def threshold_of(metric, default_threshold, metric_thresholds=None):
    """The threshold name one column is computed at.

    `metric_thresholds` pins individual metrics to their own threshold: the transport cost is
    computed over every peak while the counts beside it may only be meaningful at a coarser
    threshold, so one table row can legitimately mix the two.
    """
    return (metric_thresholds or {}).get(metric, default_threshold)


def threshold_order(metrics, default_threshold, metric_thresholds=None):
    """The distinct threshold names the block's columns use, in column order."""
    names = []
    for metric in metrics:
        name = threshold_of(metric, default_threshold, metric_thresholds)
        if name not in names:
            names.append(name)
    return names


def _block_mask(frame, block_val, block_key):
    """Row mask selecting one table block, by start fraction (numeric) or stratum (label)."""
    if block_key == "start_frac":
        return np.isclose(frame["start_frac"], block_val)
    return frame[block_key] == block_val


def cell_values(metrics_frame, models, block_val, channel, metric, threshold_name,
                block_key="start_frac", metric_thresholds=None):
    """(means, stds) over `models` for one column, or NaNs where a model has no rows."""
    threshold_name = threshold_of(metric, threshold_name, metric_thresholds)
    means, stds = [], []
    for model in models:
        sel = metrics_frame[
            (metrics_frame["model"] == model)
            & _block_mask(metrics_frame, block_val, block_key)
            & (metrics_frame["threshold_name"] == threshold_name)
        ]
        if channel is not None:
            sel = sel[sel["channel"] == channel]
        else:
            # Global metrics do not depend on the channel; take one arbitrary channel rather
            # than averaging duplicates of the same number.
            first = sel[["channel"]].drop_duplicates().head(1)
            if len(first):
                sel = sel.merge(first, on="channel")
        means.append(float(sel[metric].iloc[0]) if len(sel) else np.nan)
        stds.append(float(sel[f"{metric}_std"].iloc[0]) if len(sel) else np.nan)
    return means, stds


def check_available(start_metrics, models, start_fracs, channels, threshold_name):
    """Fail loudly instead of emitting a table full of '--'."""
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
            f"{sorted(set(missing))}. The per-window parquet is probably stale; rerun with --force."
        )


def stacked_table(metrics_frame, models, blocks, metrics, global_metrics,
                  *, label, caption, environment="table", group_label,
                  font=r"\scriptsize", tabcolsep=3, threshold_name, block_key="start_frac",
                  group_titles=None, block_notes=None, metric_thresholds=None):
    """Assemble one stacked table.

    `blocks` is a list of (block value, block title, block value again, channels): each becomes
    one vertically repeated set of rows under the shared header. `group_label` names what the
    trailing column group is (the global metrics, e.g. the mode score).
    """
    n_metrics = len(metrics)
    columns_channels = blocks[0][3]
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
    for block_i, (_, title, block_val, channels) in enumerate(blocks):
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
            means, stds = cell_values(metrics_frame, models, block_val, channel, metric,
                                      threshold_name, block_key, metric_thresholds)
            _, decimals, _, scale = METRIC_SPECS[metric]
            marks = rank_marks(means, metric)
            for i, model in enumerate(models):
                body = fmt(means[i], stds[i], decimals, scale)
                if body != "--" and i in marks:
                    body = marks[i] % body
                cells[(model, channel, metric)] = f"${body}$" if body != "--" else body
        for model in models:
            lines.append(" & ".join([model] + [cells[(model, c, m)] for c, m in columns]) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", rf"\end{{{environment}}}"]
    return "\n".join(lines) + "\n"
