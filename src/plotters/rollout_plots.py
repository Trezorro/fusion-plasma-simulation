"""Interactive browser for autoregressive rollouts.

One figure, one rollout visible at a time, a dropdown to switch between rollouts
(shot x start fraction). Rows: one per observable channel (generated vs real),
one for the control covariates, one for the surrogate mode labels. The x axis is
actual shot time in seconds; the original history window W_H and the rollout start
are marked per rollout through the dropdown's layout updates.

Records are assembled by src.rollout.build_rollout_records; this module only draws.

Label convention: surrogate labels are UNSHIFTED 0=L, 1=D, 2=H.
"""
import logging

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

MODE_NAMES = ["L", "D", "H"]  # unshifted surrogate convention
GENERATED_COLOR = "#D55E00"  # Okabe-Ito vermillion
REAL_COLOR = "black"
C_COLORS = ["#0072B2", "#009E73", "#CC79A7", "#E69F00"]  # Okabe-Ito blues/greens for controls
LABEL_STRIDE = 10  # decimation for the label step-lines; classifier output is 1 per 10 samples anyway


def _rollout_label(record) -> str:
    return (
        f"Shot {record['shot_number']} @ {record['t_start']:.2f}s "
        f"({record['start_frac']:.0%}, {record['n_windows']} windows)"
    )


def _rollout_shapes(record):
    """Per-rollout layout shapes: W_H shading, rollout start line, window boundaries."""
    times = record['times']
    history_length = int(record['history_length'])
    t_hist_start = float(times[0])
    t_start = float(record['t_start'])
    shapes = [
        dict(
            type="rect", x0=t_hist_start, x1=t_start, y0=0, y1=1,
            fillcolor="yellow", opacity=0.25, line_width=0, xref="x", yref="paper",
        ),
        dict(
            type="line", x0=t_start, x1=t_start, y0=0, y1=1,
            line=dict(color="goldenrod", width=2), opacity=0.8, xref="x", yref="paper",
        ),
    ]
    step = int(record['step'])
    boundary_idx = np.arange(step, record['generated_x'].shape[-1], step)
    for b in boundary_idx:
        t_b = float(times[history_length + b])
        shapes.append(
            dict(
                type="line", x0=t_b, x1=t_b, y0=0, y1=1,
                line=dict(color="grey", width=1, dash="dot"), opacity=0.4, xref="x", yref="paper",
            )
        )
    return shapes


def rollout_browser_plotly(records: list[dict], channel_names, c_names, title_base="Rollout browser"):
    """Build the interactive rollout browser figure.

    Args:
        records: List of rollout record dicts from src.rollout.build_rollout_records.
            Each holds generated_x (C, T), real_x / real_c over [W_H start, rollout end],
            times (seconds, same span), surr_labels_gen / surr_labels_real
            (history_length + T,), and the cache attrs (t_start, start_frac, n_windows,
            history_length, step, shot_number).
        channel_names: Observable channel names (rows 1..len).
        c_names: Control covariate names (controls row).
        title_base: Figure title prefix.

    Returns:
        go.Figure with one dropdown entry per rollout. Signal rows are drawn with
        Scattergl for performance; the label row stays SVG so the bottom axis
        rangeslider (the minimap) has visible content.
    """
    n_channels = len(channel_names)
    n_rows = n_channels + 2
    labels_row = n_rows
    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.015,
        row_heights=[1.0] * n_channels + [0.7, 0.5],
    )

    record_trace_indices: list[list[int]] = []
    for record in records:
        indices = []
        visible = len(record_trace_indices) == 0  # default view: first rollout only
        times = record['times']
        history_length = int(record['history_length'])
        # The timeline is uniform (10 kHz), so traces use x0/dx instead of explicit x
        # arrays, and y values are rounded to 4 decimals: plotly 5 serializes numpy as
        # JSON number lists, so short decimals keep the written HTML small.
        dx = float((times[-1] - times[0]) / (len(times) - 1))
        x0 = float(times[0])
        x0_gen = float(times[history_length])

        for ch in range(n_channels):
            fig.add_trace(
                go.Scattergl(
                    x0=x0, dx=dx, y=np.round(record['real_x'][ch], 4), mode='lines',
                    line=dict(color=REAL_COLOR, width=1), opacity=0.8,
                    name=f"{channel_names[ch]} real", visible=visible, showlegend=ch == 0,
                    legendgroup='real',
                ), row=ch + 1, col=1,
            )
            indices.append(len(fig.data) - 1)
            fig.add_trace(
                go.Scattergl(
                    x0=x0_gen, dx=dx, y=np.round(record['generated_x'][ch], 4), mode='lines',
                    line=dict(color=GENERATED_COLOR, width=1),
                    name=f"{channel_names[ch]} generated", visible=visible, showlegend=ch == 0,
                    legendgroup='generated',
                ), row=ch + 1, col=1,
            )
            indices.append(len(fig.data) - 1)

        for ci, c_name in enumerate(c_names):
            fig.add_trace(
                go.Scattergl(
                    x0=x0, dx=dx, y=np.round(record['real_c'][ci], 4), mode='lines',
                    line=dict(color=C_COLORS[ci % len(C_COLORS)], width=1),
                    name=f"C: {c_name}", visible=visible,
                ), row=n_channels + 1, col=1,
            )
            indices.append(len(fig.data) - 1)

        # Label step-lines (SVG on purpose: they feed the rangeslider minimap)
        for labels, name, color in (
            (record['surr_labels_real'], "mode real (surrogate)", REAL_COLOR),
            (record['surr_labels_gen'], "mode generated (surrogate)", GENERATED_COLOR),
        ):
            fig.add_trace(
                go.Scatter(
                    x0=x0, dx=dx * LABEL_STRIDE, y=labels[::LABEL_STRIDE], mode='lines',
                    line=dict(color=color, width=1.5, shape='hv'),
                    name=name, visible=visible,
                ), row=labels_row, col=1,
            )
            indices.append(len(fig.data) - 1)
        record_trace_indices.append(indices)

    n_traces = len(fig.data)

    def _button_for(r_i, record):
        vis_set = set(record_trace_indices[r_i])
        times = record['times']
        return dict(
            label=_rollout_label(record),
            method='update',
            args=[
                {'visible': [i in vis_set for i in range(n_traces)]},
                {
                    'shapes': _rollout_shapes(record),
                    'title.text': f"{title_base}: {_rollout_label(record)}",
                    'xaxis.range': [float(times[0]), float(times[-1])],
                },
            ],
        )

    buttons = [_button_for(r_i, record) for r_i, record in enumerate(records)]

    for ch in range(n_channels):
        fig.update_yaxes(title_text=channel_names[ch], row=ch + 1, col=1)
    fig.update_yaxes(title_text="C", row=n_channels + 1, col=1)
    fig.update_yaxes(
        title_text="mode", tickvals=[0, 1, 2], ticktext=MODE_NAMES, range=[-0.5, 2.5],
        row=labels_row, col=1,
    )
    fig.update_xaxes(title_text="Shot time (s)", row=labels_row, col=1)
    # Minimap: rangeslider on the bottom (shared) axis, previewing the label step-lines
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06), row=labels_row, col=1)

    first = records[0]
    fig.update_layout(
        title=f"{title_base}: {_rollout_label(first)}",
        template='ggplot2',
        height=1000,
        shapes=_rollout_shapes(first),
        xaxis=dict(range=[float(first['times'][0]), float(first['times'][-1])]),
        updatemenus=[
            dict(
                buttons=buttons, showactive=True, direction="down",
                x=1.02, xanchor="left", y=1.0, yanchor="top",
            )
        ],
        # Legend anchored BELOW the plot area growing upward would cover row 1; anchor its
        # bottom just above the first row instead so it grows into the top margin.
        legend=dict(orientation="h", y=1.01, yanchor="bottom", x=0),
        margin=dict(t=130),
    )
    return fig
