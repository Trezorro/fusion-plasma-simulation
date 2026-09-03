"""Interactive browser for autoregressive rollouts.

One figure, one starting point (shot x start fraction) visible at a time, a dropdown
to switch between them. Rows: one per observable channel (real history/ground truth
plus each overlaid stochastic sample), one for the control covariates, one for the
surrogate mode labels. The x axis is actual shot time in seconds; the original history
window W_H and the rollout start are marked per starting point through the dropdown's
layout updates.

Groups are assembled by src.rollout.build_rollout_groups (one dropdown entry per
(shot, start point), holding up to rollout.plot_samples stochastic samples overlaid);
this module only draws.

Legend behaviour: clicking a legend entry toggles that entry's whole group of traces
(all channels + the label row) at once. "Ground truth" is one group; each sample is
its own group so individual rollouts can be isolated. The original real history (left
of the rollout start) is never part of a legend group, so it always renders.

Label convention: surrogate labels are UNSHIFTED 0=L, 1=D, 2=H.
"""
import logging

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.metrics.metrics import PeakProps
from src.signal_filters import apply_filter, filter_label, resolve_filter

logger = logging.getLogger(__name__)

MODE_NAMES = ["L", "D", "H"]  # unshifted surrogate convention
HISTORY_COLOR = "#777777"  # real history before the rollout start; never legend-toggleable
GT_COLOR = "black"  # real future ("ground truth")
# Okabe-Ito, cycled by sample_idx (skips black/vermillion's usual "real" role reversal by
# starting on vermillion, since it reads well as the first/primary sample)
SAMPLE_COLORS = ["#D55E00", "#009E73", "#CC79A7", "#0072B2", "#E69F00", "#56B4E9", "#F0E442"]
C_COLORS = ["#0072B2", "#009E73", "#CC79A7", "#E69F00"]
LABEL_STRIDE = 10  # decimation for the label step-lines; classifier output is 1 per 10 samples anyway
PEAK_SYMBOL_REAL = "triangle-down"
PEAK_SYMBOL_GEN = "x"


def _sample_color(sample_idx: int) -> str:
    return SAMPLE_COLORS[sample_idx % len(SAMPLE_COLORS)]


def _group_label(group) -> str:
    n = len(group['samples'])
    return (
        f"Shot {group['shot_number']} @ {group['t_start']:.2f}s "
        f"({group['start_frac']:.0%}, {group['n_windows']} windows, {n} sample{'s' if n != 1 else ''})"
    )


def _group_shapes(group):
    """Per-group layout shapes: W_H shading, rollout start line, window boundaries."""
    times = group['times']
    history_length = int(group['history_length'])
    t_hist_start = float(times[0])
    t_start = float(group['t_start'])
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
    step = int(group['step'])
    T = group['samples'][0]['generated_x'].shape[-1]
    boundary_idx = np.arange(step, T, step)
    for b in boundary_idx:
        t_b = float(times[history_length + b])
        shapes.append(
            dict(
                type="line", x0=t_b, x1=t_b, y0=0, y1=1,
                line=dict(color="grey", width=1, dash="dot"), opacity=0.4, xref="x", yref="paper",
            )
        )
    return shapes


def _detect_peaks(trace, context, prominence, signal_filter, channel_name, rel_height=0.5):
    """Peak positions (as indices into `trace`) and heights, for the marker overlay.

    The real history is prepended as left context and the filter, if any, is applied to the
    concatenation, so this sees exactly the same samples as the table pipeline
    (eval_notebooks/rollout_tables.py, record_slice_rows) rather than a differently-conditioned
    copy of the trace. Peaks inside the context are dropped.
    """
    offset = len(context)
    full = np.concatenate([np.asarray(context, dtype=float), np.asarray(trace, dtype=float)])
    full = apply_filter(full, signal_filter, channel_name)
    peaks = PeakProps.from_find_peaks(full, prominence=prominence, rel_height=rel_height)
    pos = np.asarray(peaks.X) - offset
    keep = pos >= 0
    return pos[keep].astype(int), np.asarray(peaks.Y)[keep], full


def _peak_prominence(prominences, channel_name):
    """A scalar prominence, or the per-channel entry of a dict (missing channels are skipped)."""
    if isinstance(prominences, dict):
        return prominences.get(channel_name)
    return prominences


def rollout_browser_plotly(groups: list[dict], channel_names, c_names, title_base="Rollout browser",
                           show_peaks=False, peak_prominence=0.01, signal_filter=None,
                           show_filtered=None):
    """Build the interactive rollout browser figure.

    Args:
        groups: List of group dicts from src.rollout.build_rollout_groups. Each holds
            real_x / real_c / times over [W_H start, rollout end], surr_labels_real,
            and the cache attrs (t_start, start_frac, n_windows, history_length, step,
            shot_number), plus 'samples': a list of {sample_idx, generated_x,
            surr_labels_gen} for the stochastic samples overlaid at that start point.
        channel_names: Observable channel names (rows 1..len).
        c_names: Control covariate names (controls row).
        title_base: Figure title prefix.
        show_peaks: overlay detected peak markers. Real-trace and generated-trace peaks are
            separate legend entries, so either can be toggled off on its own.
        peak_prominence: detection threshold in normalized [0,1] units. A dict keyed by
            channel name gives per-channel thresholds, matching the table pipeline; a channel
            absent from that dict gets no markers.
        signal_filter: optional pre-detection smoothing spec (src.signal_filters), applied
            identically to the real and the generated trace before find_peaks.
        show_filtered: draw the filtered trace, carrying the peak markers with it in one
            legend entry per source. Defaults to True whenever a filter is active, so what the
            detector saw is visible next to the raw signal. With no filter the markers get the
            legend entry themselves. The same peaks are additionally marked on the raw
            trace, in the same legend group.

    Returns:
        go.Figure with one dropdown entry per (shot, start point). Signal rows are
        drawn with Scattergl for performance; the label row stays SVG so the bottom
        axis rangeslider (the minimap) has visible content.
    """
    if show_filtered is None:
        show_filtered = resolve_filter(signal_filter) is not None
    n_channels = len(channel_names)
    # The overlay legend entry cannot simply hang off channel 0: peak_prominence is per channel
    # and typically only names PD and DML, so channel 0 (FIR_LIDs_core in this dataset) draws no
    # overlay at all and the entry would never be emitted. Put it on the first channel that
    # actually gets one.
    overlay_channels = [ch for ch in range(n_channels)
                        if show_peaks and _peak_prominence(peak_prominence, channel_names[ch]) is not None]
    legend_ch = overlay_channels[0] if overlay_channels else None
    n_rows = n_channels + 2
    labels_row = n_rows
    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.015,
        row_heights=[1.0] * n_channels + [0.7, 0.5],
    )

    group_trace_indices: list[list[int]] = []
    for group in groups:
        indices = []
        visible = len(group_trace_indices) == 0  # default view: first group only
        times = group['times']
        history_length = int(group['history_length'])
        # The timeline is uniform (10 kHz), so traces use x0/dx instead of explicit x
        # arrays, and y values are rounded to 4 decimals: plotly 5 serializes numpy as
        # JSON number lists, so short decimals keep the written HTML small.
        dx = float((times[-1] - times[0]) / (len(times) - 1))
        x0 = float(times[0])
        x0_gen = float(times[history_length])

        for ch in range(n_channels):
            # Real history before the rollout start: always shown while its group is
            # selected, never tied to a legend entry, so it can't be toggled off.
            fig.add_trace(
                go.Scattergl(
                    x0=x0, dx=dx, y=np.round(group['real_x'][ch, :history_length], 4), mode='lines',
                    line=dict(color=HISTORY_COLOR, width=1), opacity=0.8,
                    name="history", visible=visible, showlegend=False,
                ), row=ch + 1, col=1,
            )
            indices.append(len(fig.data) - 1)
            # Real future ("ground truth"): one legend entry for the whole group,
            # across every channel and the label row.
            fig.add_trace(
                go.Scattergl(
                    x0=x0_gen, dx=dx, y=np.round(group['real_x'][ch, history_length:], 4), mode='lines',
                    line=dict(color=GT_COLOR, width=1), opacity=0.85,
                    name="Ground truth", visible=visible, showlegend=ch == 0,
                    legendgroup='ground_truth',
                ), row=ch + 1, col=1,
            )
            indices.append(len(fig.data) - 1)
            # The filtered trace and the peak markers form one legend group per source
            # ("overlay_real", "overlay_sample_N"), separate from the raw signal's group. The
            # markers are what the detector found *on* the filtered trace, so the two belong
            # together: toggling the entry shows or hides both, and the raw signal underneath
            # stays put. Only the filtered line carries the legend entry; the markers ride
            # along with showlegend=False.
            prom = _peak_prominence(peak_prominence, channel_names[ch]) if show_peaks else None
            context = group['real_x'][ch, :history_length]
            if prom is not None:
                pos, heights, filtered = _detect_peaks(
                    group['real_x'][ch, history_length:], context, prom, signal_filter,
                    channel_names[ch])
                if show_filtered:
                    fig.add_trace(
                        go.Scattergl(
                            x0=x0_gen, dx=dx, y=np.round(filtered[history_length:], 4), mode='lines',
                            line=dict(color=GT_COLOR, width=1, dash='dot'), opacity=0.7,
                            name=f"Filtered + peaks (real, {len(pos)})", visible=visible,
                            showlegend=ch == legend_ch, legendgroup='overlay_real',
                        ), row=ch + 1, col=1,
                    )
                    indices.append(len(fig.data) - 1)
                    # Same peaks marked on the raw trace: positions come from the filtered
                    # signal, heights are read off the unfiltered one, so the detections can
                    # be judged against the signal as it actually is.
                    fig.add_trace(
                        go.Scattergl(
                            x=x0_gen + pos * dx,
                            y=np.round(group['real_x'][ch, history_length:][pos], 4),
                            mode='markers',
                            marker=dict(color=GT_COLOR, symbol=PEAK_SYMBOL_REAL, size=7,
                                        line=dict(width=0)),
                            name=f"Peaks (real, {len(pos)})", visible=visible,
                            showlegend=False, legendgroup='overlay_real',
                        ), row=ch + 1, col=1,
                    )
                    indices.append(len(fig.data) - 1)
                fig.add_trace(
                    go.Scattergl(
                        x=x0_gen + pos * dx, y=np.round(heights, 4), mode='markers',
                        marker=dict(color=GT_COLOR, symbol=PEAK_SYMBOL_REAL, size=7,
                                    line=dict(width=0)),
                        name=f"Peaks (real, {len(pos)})", visible=visible,
                        showlegend=ch == legend_ch and not show_filtered, legendgroup='overlay_real',
                    ), row=ch + 1, col=1,
                )
                indices.append(len(fig.data) - 1)
            for sample in group['samples']:
                color = _sample_color(sample['sample_idx'])
                fig.add_trace(
                    go.Scattergl(
                        x0=x0_gen, dx=dx, y=np.round(sample['generated_x'][ch], 4), mode='lines',
                        line=dict(color=color, width=1),
                        name=f"Sample {sample['sample_idx']}", visible=visible, showlegend=ch == 0,
                        legendgroup=f"sample_{sample['sample_idx']}",
                    ), row=ch + 1, col=1,
                )
                indices.append(len(fig.data) - 1)
                if prom is not None:
                    si = sample['sample_idx']
                    pos, heights, filtered = _detect_peaks(
                        sample['generated_x'][ch], context, prom, signal_filter, channel_names[ch])
                    if show_filtered:
                        fig.add_trace(
                            go.Scattergl(
                                x0=x0_gen, dx=dx, y=np.round(filtered[history_length:], 4),
                                mode='lines', line=dict(color=color, width=1, dash='dot'),
                                opacity=0.7,
                                name=f"Filtered + peaks (sample {si}, {len(pos)})",
                                visible=visible, showlegend=ch == legend_ch,
                                legendgroup=f"overlay_sample_{si}",
                            ), row=ch + 1, col=1,
                        )
                        indices.append(len(fig.data) - 1)
                        fig.add_trace(
                            go.Scattergl(
                                x=x0_gen + pos * dx,
                                y=np.round(sample['generated_x'][ch][pos], 4), mode='markers',
                                marker=dict(color=color, symbol=PEAK_SYMBOL_GEN, size=7,
                                            line=dict(width=0)),
                                name=f"Peaks (sample {si}, {len(pos)})", visible=visible,
                                showlegend=False, legendgroup=f"overlay_sample_{si}",
                            ), row=ch + 1, col=1,
                        )
                        indices.append(len(fig.data) - 1)
                    fig.add_trace(
                        go.Scattergl(
                            x=x0_gen + pos * dx, y=np.round(heights, 4), mode='markers',
                            marker=dict(color=color, symbol=PEAK_SYMBOL_GEN, size=7,
                                        line=dict(width=0)),
                            name=f"Peaks (sample {si}, {len(pos)})",
                            visible=visible, showlegend=ch == legend_ch and not show_filtered,
                            legendgroup=f"overlay_sample_{si}",
                        ), row=ch + 1, col=1,
                    )
                    indices.append(len(fig.data) - 1)

        for ci, c_name in enumerate(c_names):
            fig.add_trace(
                go.Scattergl(
                    x0=x0, dx=dx, y=np.round(group['real_c'][ci], 4), mode='lines',
                    line=dict(color=C_COLORS[ci % len(C_COLORS)], width=1),
                    name=f"C: {c_name}", visible=visible,
                ), row=n_channels + 1, col=1,
            )
            indices.append(len(fig.data) - 1)

        # Label step-lines (SVG on purpose: they feed the rangeslider minimap), part
        # of the same legend groups as their channel traces above.
        fig.add_trace(
            go.Scatter(
                x0=x0, dx=dx * LABEL_STRIDE, y=group['surr_labels_real'][::LABEL_STRIDE], mode='lines',
                line=dict(color=GT_COLOR, width=1.5, shape='hv'),
                name="Ground truth", visible=visible, showlegend=False, legendgroup='ground_truth',
            ), row=labels_row, col=1,
        )
        indices.append(len(fig.data) - 1)
        for sample in group['samples']:
            color = _sample_color(sample['sample_idx'])
            fig.add_trace(
                go.Scatter(
                    x0=x0, dx=dx * LABEL_STRIDE, y=sample['surr_labels_gen'][::LABEL_STRIDE], mode='lines',
                    line=dict(color=color, width=1.5, shape='hv'),
                    name=f"Sample {sample['sample_idx']}", visible=visible, showlegend=False,
                    legendgroup=f"sample_{sample['sample_idx']}",
                ), row=labels_row, col=1,
            )
            indices.append(len(fig.data) - 1)
        group_trace_indices.append(indices)

    if show_peaks:
        # The detection settings belong in the figure, not just in the calling notebook: an
        # exported HTML is otherwise indistinguishable from one made at another threshold.
        prom_txt = (", ".join(f"{k}={v:g}" for k, v in peak_prominence.items())
                    if isinstance(peak_prominence, dict) else f"{peak_prominence:g}")
        title_base = f"{title_base} [peaks pi>={prom_txt}, {filter_label(signal_filter)}]"

    n_traces = len(fig.data)

    def _button_for(g_i, group):
        vis_set = set(group_trace_indices[g_i])
        times = group['times']
        return dict(
            label=_group_label(group),
            method='update',
            args=[
                {'visible': [i in vis_set for i in range(n_traces)]},
                {
                    'shapes': _group_shapes(group),
                    'title.text': f"{title_base}: {_group_label(group)}",
                    'xaxis.range': [float(times[0]), float(times[-1])],
                },
            ],
        )

    buttons = [_button_for(g_i, group) for g_i, group in enumerate(groups)]

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

    first = groups[0]
    fig.update_layout(
        title=f"{title_base}: {_group_label(first)}",
        template='ggplot2',
        height=1000,
        shapes=_group_shapes(first),
        xaxis=dict(range=[float(first['times'][0]), float(first['times'][-1])]),
        # Dropdown (pick the starting point) and legend (toggle ground truth / a given
        # sample, across all its channels + the label row at once) both sit in the
        # right margin, dropdown above the legend.
        updatemenus=[
            dict(
                buttons=buttons, showactive=True, direction="down",
                x=1.02, xanchor="left", y=1.0, yanchor="top",
            )
        ],
        legend=dict(orientation="v", x=1.02, xanchor="left", y=0.9, yanchor="top"),
        margin=dict(t=80, r=220),
    )
    return fig
