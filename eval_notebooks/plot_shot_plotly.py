"""Interactive plotly overview of a single TCV shot: every scoped variable from
column_to_latex.json, grouped by category, over a confinement-state (L/D/H) background.

This is the plotly counterpart of `plot_discharge()` in
giants/TCV-confstate-data/utils/overview.py (matplotlib, 3 signals) and of the legacy
`plot_signal_and_spectrum()` in notebooks_and_reference/generate_shot_plot.py (plotly).

Run cell-by-cell (# %%) in an interactive window, or `python plot_shot_plotly.py`
to write an HTML file next to this script.
"""
# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# The confstate parquet-per-shot files and metadata live in the repo's
# `data/public_data_set/`. Anchor to it by walking up from this file (or the
# cwd, in an interactive session) so the script keeps working wherever it moves.
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_ROOT = Path("data/public_data_set")
DATA_DIR = DATA_ROOT / "data"
META_DIR = DATA_ROOT / "metadata"
OUT_DIR = Path("output/shotplots")  # HTML export lands next to this script
OUT_DIR.mkdir(parents=True, exist_ok=True)
PQ_TEMPLATE = "TCV_confstate_{shot}.parquet"

# Which shot to plot, and whether to normalise each signal (z-score) so that
# variables with very different magnitudes share a row legibly.
SHOT = None
SHOTS = [# sourced from test shots in config/plasmaflow.yaml
        57094, 57732, 60814, 64365, 67112, 72929, 75264, 76702, 77409, 77595, 53623, 57013, 60813, 61028, 61056, 61237, 63306, 63878, 64386, 64393, 64678, 64686, 64770, 64857, 65469, 65481, 68631, 68697, 69514, 71344, 73368, 73631, 73935, 76304, 77089, 77193, 77196, 77598, 77599, 77602, 77604, 78069, 79825, 83049
    ]

NORMALISE = True
LABEL_COLUMN = "label_conf"  # use "label_conf_qce" for the QCE shots [61056, 71344, 78069, 83049]

# %%
# Load the variable -> LaTeX grouping. Every column listed here gets plotted,
# one subplot row per category, in this order.
with open(META_DIR / "column_to_latex.json", "r") as f:
    column_to_latex = json.load(f)


def latex_to_name(tex: str) -> str:
    """Strip the $...$ / LaTeX escapes down to something readable in a plotly legend."""
    s = tex.strip("$")
    for junk in ("\\text", "\\textit", "\\mathrm", "{", "}"):
        s = s.replace(junk, "")
    return s


# Confinement-state colours, matching plot_discharge() in utils/overview.py.
STATE_COLORS = {
    0: "rgba(140, 191, 217, 0.30)",  # L
    1: "rgba(153, 128, 179, 0.30)",  # D
    2: "rgba(230, 217, 128, 0.30)",  # H
    3: "rgba(230, 191, 153, 0.30)",  # QCE-H
}
STATE_NAMES = {0: "L", 1: "D", 2: "H", 3: "QCE-H"}


# %%
def add_state_background(fig, time, labels, n_rows):
    """Shade contiguous runs of the confinement-state label behind every subplot row."""
    labels = np.asarray(labels, dtype=float)
    time = np.asarray(time, dtype=float)

    start = 0
    for i in range(1, len(labels) + 1):
        # close a run when the label changes or we hit the end
        if i == len(labels) or labels[i] != labels[start]:
            state = labels[start]
            if np.isfinite(state) and int(state) in STATE_COLORS:
                x0 = time[start]
                x1 = time[i - 1] if i < len(labels) else time[-1]
                fig.add_vrect(
                    x0=x0, x1=x1,
                    fillcolor=STATE_COLORS[int(state)],
                    line_width=0, layer="below",
                )
            start = i

    # one dummy trace per state so the shading gets a legend entry
    present = sorted({int(l) for l in labels if np.isfinite(l) and int(l) in STATE_COLORS})
    for state in present:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=12, symbol="square", color=STATE_COLORS[state].replace("0.30", "0.8")),
                name=f"state: {STATE_NAMES[state]}",
                legendgroup="states", showlegend=True,
            ),
            row=1, col=1,
        )


def plot_shot(shot, normalise=NORMALISE, label_column=LABEL_COLUMN):
    pq = pd.read_parquet(DATA_DIR / PQ_TEMPLATE.format(shot=shot))
    time = pq["time"].values

    categories = list(column_to_latex.keys())
    fig = make_subplots(
        rows=len(categories), cols=1,
        shared_xaxes=True, vertical_spacing=0.012,
        subplot_titles=[c.replace("_", " ") for c in categories],
    )

    for r, category in enumerate(categories, start=1):
        for col, tex in column_to_latex[category].items():
            if col not in pq.columns:
                continue
            y = pq[col].astype(float)
            if normalise:
                std = y.std()
                y = (y - y.mean()) / std if std and np.isfinite(std) else y - y.mean()
            fig.add_trace(
                go.Scatter(
                    x=time, y=y, mode="lines", name=tex,
                    legendgroup=category,
                    legendgrouptitle_text=category,
                    hovertemplate=f"{col}<br>t=%{{x:.4f}}s<br>y=%{{y:.3g}}<extra></extra>",
                ),
                row=r, col=1,
            )

    add_state_background(fig, time, pq[label_column].values, len(categories))

    ytitle = "z-scored" if normalise else "raw value"
    fig.update_yaxes(title_text=ytitle, title_font_size=9)
    fig.update_xaxes(title_text="time [s]", row=len(categories), col=1)
    fig.update_layout(
        height=260 * len(categories),
        title=f"TCV #{shot}: scoped signals over confinement state ({'normalised' if normalise else 'raw'})",
        legend=dict(
            groupclick="togglegroup",
            font=dict(size=15),  # larger legend labels (LaTeX renders via MathJax)
            grouptitlefont=dict(size=16),
        ),
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


# %%
if SHOT:
    fig = plot_shot(SHOT)
    fig.show()
    out = OUT_DIR / f"shot_{SHOT}_overview.html"
    fig.write_html(out, auto_open=False)
    print("wrote", out)

# %%
# Batch: render an overview HTML for each shot in a list.
# Missing parquet files are reported and skipped rather than aborting the loop.
if SHOTS:
    for shot in SHOTS:
        if not (DATA_DIR / PQ_TEMPLATE.format(shot=shot)).exists():
            print(f"skip #{shot}: no parquet in {DATA_DIR}")
            continue
        out = OUT_DIR / f"shot_{shot}_overview.html"
        plot_shot(shot).write_html(
            out,
            auto_open=False,
            include_plotlyjs="cdn",  # or True
            include_mathjax="cdn",  # enable MathJax
        )
        print("wrote", out)

# %%
