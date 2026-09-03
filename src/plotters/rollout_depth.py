"""Compound-error-against-rollout-depth figures.

One PDF per metric per figure size: the metric against the window index from the start of the
shot, one colour per model and one line style per start fraction, with a band of +/- 1 standard
deviation over the shots contributing at that depth. Reads the depth frame from
`src.metrics.rollout_aggregate.aggregate_by_depth`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.plotters.latex_tables import METRIC_SPECS

# Okabe-Ito, one colour per model, ordered so every adjacent pair clears the CVD separation
# check. Black stays reserved for a real-trace reference.
MODEL_COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
START_FRAC_STYLES = ["-", "--", ":", "-."]

RC = {
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "lines.linewidth": 1.1, "figure.dpi": 120,
    "axes.spines.top": False, "axes.spines.right": False,
}


def depth_offsets(slice_metrics, seq_length: int) -> dict:
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


def plot_depth_curves(depth_metrics, slice_metrics, seq_length, pdf_dir, *, models, metrics,
                      channel, start_fractions, threshold_name, sizes, min_shots):
    """Write one PDF per (metric, size) under `pdf_dir`; returns the paths written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pdf_dir = Path(pdf_dir)
    models = list(models)
    offsets = depth_offsets(slice_metrics, seq_length)
    fracs = [f for f in start_fractions if f in offsets]
    written = []

    for metric in metrics:
        ylabel, _, _, _ = METRIC_SPECS[metric]
        for (fig_w, fig_h) in sizes:
            with plt.rc_context(RC):
                fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
                for model, color in zip(models, MODEL_COLORS):
                    for frac, style in zip(fracs, START_FRAC_STYLES):
                        d = depth_metrics[
                            (depth_metrics["model"] == model)
                            & np.isclose(depth_metrics["start_frac"], frac)
                            # Dice is channel-independent; one channel keeps the rows unique.
                            & (depth_metrics["channel"] == channel)
                            & (depth_metrics["threshold_name"] == threshold_name)
                            # Shots end at different times, so the sample shrinks with depth
                            # and the tail would otherwise be one or two rollouts wide.
                            & (depth_metrics["n_shots"] >= min_shots)
                        ].sort_values("k")
                        if d.empty:
                            continue
                        x = d["k"].to_numpy() + offsets[frac]
                        y = d[metric].to_numpy()
                        sd = d[f"{metric}_std"].to_numpy()
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
                           for m, c in zip(models, MODEL_COLORS)]
                handles += [Line2D([], [], color="0.35", lw=1.2, linestyle=st, label=f"$f={fr:g}$")
                            for fr, st in zip(fracs, START_FRAC_STYLES)]
                ax.legend(handles=handles, frameon=False, ncol=2, handlelength=1.3,
                          columnspacing=0.9, handletextpad=0.5, labelspacing=0.25,
                          borderaxespad=0.2, loc="best")

                out = pdf_dir / f"{fig_w:g}x{fig_h:g}" / f"depth_{channel}_{metric}.pdf"
                out.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out, bbox_inches="tight")
                plt.close(fig)
                written.append(out)
    return written
