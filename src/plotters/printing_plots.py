from typing import Optional
from src.config import get_current_config

import numpy as np
import plotly.graph_objects as go
import torch
import wandb
from plotly import colors as plt_colors
from plotly.subplots import make_subplots as plotly_make_subplots


def single_window_lines_plotly(
    target_samples: torch.Tensor,
    generated_samples: Optional[torch.Tensor] = None,
    conditioning_input: Optional[dict] = None,
    labels: Optional[np.ndarray] = None,
    title: str = "",
    show_c: bool = True,
    legend_loc: str = "top right",
    label_bars: bool = True,
    **kwargs
):
    """
    Print-friendly plot for a single window (no batch), with 3 explicit subplots:
    1. x channels
    2. c channels (if present)
    3. label bar (if present)
    All share the x axis.
    """
    C = get_current_config()
    CHANNEL_NAMES = C.data.cols.x
    history_length = C.data.history_length
    seq_length = C.data.seq_length
    COLOR_SCALE = plt_colors.qualitative.Plotly
    C_COLOR_SCALE = plt_colors.qualitative.Pastel
    if target_samples.dim() == 3:
        B, n_channels, n_timepoints = target_samples.shape
        assert B == 1, "Single window plot expects a batch size of 1 or no batch dim."
    else:
        n_channels, n_timepoints = target_samples.shape

    # Determine subplot rows
    has_c = show_c and conditioning_input is not None and "c" in conditioning_input
    has_labels = label_bars and labels is not None
    nrows = 1 + int(has_c) + int(has_labels)
    row_x = 1
    row_c = 2 if has_c else None
    label_row = 3 if has_c and has_labels else (2 if has_labels else None)

    fig = plotly_make_subplots(
        rows=nrows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_heights=[0.5, 0.4, 0.1][:nrows],
        specs=[[{
            "secondary_y": False
        }] for _ in range(nrows)],
    )
    # Add vertical black solid line at x=0 to all subplots
    for row in range(1, nrows + 1):
        fig.add_shape(
            type="line",
            x0=0,
            x1=0,
            y0=0,
            y1=1,
            line=dict(
                color="black",
                width=1,
                dash="solid",
            ),
            xref="x" + (str(row) if row > 1 else ""),
            yref="paper",
            opacity=.8,
            layer="above"
        )
    # Annotate left and right windows with LaTeX labels
    fig.add_annotation(
        text=r"$W_H$",
        x=-history_length * 0.5,
        y=0.9,
        xref=f"x{row_x}",
        yref="paper",
        showarrow=False,
        font=dict(size=18, color="black"),
        align="center",
        bgcolor="rgba(255,255,255,0.0)",
        borderpad=2,
        row=1,
        col=1,
    )
    fig.add_annotation(
        text=r"$W_F$",
        x=seq_length * 0.5,
        y=.9,
        xref=f"x{row_x}",
        yref="paper",
        showarrow=False,
        font=dict(size=18, color="black"),
        align="center",
        bgcolor="rgba(255,255,255,0.0)",
        borderpad=2,
        row=1,
        col=1,
    )
    # X channels subplot
    for channel_i in range(n_channels):
        channel_color = COLOR_SCALE[channel_i % len(COLOR_SCALE)]
        channel_name = CHANNEL_NAMES[channel_i]
        fig.add_trace(
            go.Scatter(
                x=np.arange(seq_length),
                y=target_samples.squeeze()[channel_i, :],
                mode='lines',
                line=dict(color=channel_color, width=2),
                opacity=0.9,
                name=f'{channel_name}',
                legendgroup=f'x',
                legendgrouptitle_text=r"Observables $\mathbf{x}_W$",
            ),
            row=row_x,
            col=1
        )
        if generated_samples is not None:
            fig.add_trace(
                go.Scatter(
                    x=np.arange(seq_length),
                    y=generated_samples.squeeze()[channel_i, :],
                    mode='lines',
                    line=dict(dash='dot', color=channel_color, width=2),
                    opacity=0.9,
                    name=f'{channel_name} (predicted)',
                    legendgroup=f'{channel_name}',
                ),
                row=row_x,
                col=1
            )
        show_history = conditioning_input is not None and "x_history" in conditioning_input
        if show_history:
            x_history = conditioning_input['x_history'].squeeze()  # type: ignore
            fig.add_trace(
                go.Scatter(
                    x=np.arange(-history_length, 0),
                    y=x_history[channel_i, :],
                    mode='lines',
                    line=dict(color=channel_color, width=2, dash='solid'),
                    opacity=0.8,
                    name=f'{channel_name} (history)',
                    showlegend=False,
                    legendgroup=f'x',
                ),
                row=row_x,
                col=1
            )
    # C channels subplot
    if has_c:
        c_input = conditioning_input["c"].squeeze()  # type: ignore
        c_channels = c_input.shape[0]
        c_axis_values = np.arange(-history_length, seq_length)
        C_CHANNEL_NAMES = C.data.cols.c
        for channel_j in range(c_channels):
            channel_color = C_COLOR_SCALE[channel_j % len(C_COLOR_SCALE)]
            channel_name = C_CHANNEL_NAMES[channel_j]
            fig.add_trace(
                go.Scatter(
                    x=c_axis_values,
                    y=c_input[channel_j, :],
                    mode='lines',
                    line=dict(color=channel_color, width=2),
                    opacity=0.9,
                    name=f'{channel_name}',
                    legendgroup=f'(C)',
                    legendgrouptitle_text=r"Controls $\mathbf{c}_W$",
                ),
                row=row_c,
                col=1
            )
    # Label bar subplot
    if has_labels:
        add_mode_bars(fig, history_length, seq_length, labels.squeeze(), layer=0, group="human")
        # Move the last two bar traces to the label bar row
        # (Plotly doesn't support bar row assignment directly, so we move them after creation)

    # Layout
    fig.update_layout(
        title=title,
        template='ggplot2',
        font=dict(family="serif", size=12),
        hovermode='closest',
        margin=dict(l=10, r=20, t=10, b=20),
        height=500,
        width=950,
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.001,
            yref='paper',
            # xanchor="left",
            # x=1,
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0)",
            # valign="middle",  # Use vertical space
            itemsizing="constant",
            # traceorder="normal",
        ),
        barmode='stack',
        barcornerradius=1,
    )
    print(nrows)
    middle_time = conditioning_input['position_sequence'].squeeze()[-seq_length]

    x_ticks = list(range(-250, 251, 50))
    ticktext = x_ticks.copy()
    ticktext[len(x_ticks)//2] = f"$t={middle_time:0.3f}$"
    fig.update_xaxes(
        range=(-history_length, seq_length), showticklabels=False, showgrid=True,
        ticks="",
        tickvals=x_ticks,
    )
    # bottom X axis
    fig.update_xaxes(
        range=(-history_length - 1, seq_length + 1),
        showticklabels=True,
        title_text="Time steps (0.1ms/step)",
        showgrid=False,
        row=nrows,
        col=1,
        tickvals=x_ticks,
        ticktext=ticktext,
        ticks="inside"
    )
    # Remove y ticks on the last subplot (label bar)
    if label_row is not None:
        fig.update_yaxes(showticklabels=False, row=label_row, col=1)
    fig.update_yaxes(title_text="$\mathbf{x}$", showticklabels=True, range=(-0.05,1.01), row=1, col=1)
    fig.update_yaxes(title_text="$\mathbf{c}$", showticklabels=True, range=(-0.02,1.02), row=2, col=1)
    fig.update_yaxes(title_text="$\mathbf{y}$", showticklabels=False, ticks='', row=3, col=1)
    # DUMMY BARS
    # Add manual legend entries for the bars: H (red), D (orange), L (blue)
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[0],
            marker=dict(color="red", opacity=0.5),
            name="High",
            showlegend=True,
            legendgroup="Modes",
            legendgrouptitle_text="Confinement Modes",
        ),
        row=label_row if label_row is not None else nrows,
        col=1
    )
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            marker=dict(color="orange", opacity=0.5),
            name="Dithering",
            showlegend=True,
            legendgroup="Modes",
        ),
        row=label_row if label_row is not None else nrows, col=1
    )
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            marker=dict(color="lightskyblue", opacity=0.5),
            name="Low",
            showlegend=True,
            legendgroup="Modes",
        ),
        row=label_row if label_row is not None else nrows, col=1
    )
    # if has_c:
    #     fig.update_yaxes(range=(-1.1, 1.1), row=row_c, col=1)
    if wandb.run.disabled:  # type: ignore
        fig.show()
    return fig


def add_mode_bars(
    fig,
    history_length,
    seq_length,
    shot_labels,
    layer = 0,
    shot_number=-1,
    row=3,
    group: str = 'human',
    showlegend=False,
    secondary_y=False
):
    BAR_WIDTH = 10
    MODE_COLORS = ["grey", "lightskyblue", "orange", "red"]
    MODE_NAMES = ["Unknown", "L", "D", "H"]
    bar_y_placement = layer * BAR_WIDTH  # - 0.5 * BAR_WIDTH
    if group == 'real':
        bar_y_placement -= BAR_WIDTH *0.2
    fig.update_yaxes(
        range=(-0.8 * BAR_WIDTH, (BAR_WIDTH) * (layer + 0.5)),
        showticklabels=True,
        ticks='',
        tickvals=[-1.5]+list(range(BAR_WIDTH+2, layer * BAR_WIDTH + 3, BAR_WIDTH)),
        ticktext=['Real', '', '', 'Gen', '', ''],
        secondary_y=False,
        fixedrange=True,
        showgrid=False,
        row=row
    )
    spans = []
    modes = []
    custom_data = []
    colors = []
    current_label = shot_labels[0]
    start_t = -history_length
    for ti in range(0, history_length + seq_length):
        if shot_labels[ti] != current_label:
            next_t = ti - history_length  # Translate to the original time step
            spans.append(next_t - start_t)
            modes.append(MODE_NAMES[int(current_label)])
            colors.append(MODE_COLORS[int(current_label)])
            custom_data.append([shot_number, start_t, next_t, MODE_NAMES[int(current_label)]])
            current_label = shot_labels[ti]
            start_t = next_t
            # Add the last range
    spans.append(seq_length - start_t)
    modes.append(MODE_NAMES[int(current_label)])
    colors.append(MODE_COLORS[int(current_label)])
    custom_data.append([shot_number, start_t, seq_length, MODE_NAMES[int(current_label)]])
    # Add scatter lines for each range
    fig.add_trace(
        go.Bar(
            x=(-history_length, 0),
            y=(bar_y_placement, bar_y_placement),
            orientation='h',
            marker=dict(
                color="black",
                opacity=0,
            ),
            yaxis=f'y{row}',
            showlegend=False,  # Bar chart does not need a separate legend
            # name=f'Shot #{shot_number} - Start',
            # legendgroup=f'Shot {shot_number} - Modes',
            hoverinfo='skip',  # Disable hover for this trace
        ),
        secondary_y=secondary_y,
    )

    fig.add_trace(
        go.Bar(
            x=spans,
            y=(bar_y_placement,) * len(spans),
            width=BAR_WIDTH * (1.3 if group == 'real' else 1),
            orientation='h',
            marker=dict(
                color=colors,
                opacity=0.8 if group == 'real' else 0.5,
            ),
            yaxis=f'y{row}',
            hovertemplate=
            "Mode: %{customdata[3]}<br>Shot #%{customdata[0]}<br>Time steps: %{customdata[1]} - %{customdata[2]}<br>(%{x} steps)",
            customdata=custom_data,
            showlegend=showlegend,  # Bar chart does not need a separate legend
            # name=f'Shot #{shot_number} - {group} Labels',
            # # hoverinfo=['skip'] + ['all'] * (len(spans) - 1),  # Disable hover for this trace
            # legendgroup=f'Shot {shot_number} - Modes',
            # legendgrouptitle_text=f'Shot {shot_number} - Modes',
        ),
        secondary_y=secondary_y,
    )
    for i in [-2, -1]:
        fig.data[i].update(xaxis=f'x{row}', yaxis=f'y{row}')  # type: ignore


def multi_sample_single_window_lines_plotly(
    target_samples: torch.Tensor,
    generated_samples: Optional[torch.Tensor] = None,
    conditioning_input: Optional[dict] = None,
    surr_labels_target: Optional[np.ndarray] = None,
    surr_labels_pred: Optional[np.ndarray] = None,
    title: str = "",
    show_c: bool = True,
    **kwargs
):
    """
    Print-friendly plot for a single window (no batch), with 3 explicit subplots:
    1. x channels
    2. c channels (if present)
    3. label bar (if present)
    All share the x axis.
    """
    C = get_current_config()
    CHANNEL_NAMES = C.data.cols.x
    history_length = C.data.history_length
    seq_length = C.data.seq_length
    COLOR_SCALE = plt_colors.qualitative.Plotly
    C_COLOR_SCALE = plt_colors.qualitative.Pastel
    B, n_channels, n_timepoints = target_samples.shape

    # Determine subplot rows
    has_c = show_c and conditioning_input is not None and "c" in conditioning_input
    xrows = len(CHANNEL_NAMES)
    row_c = xrows + 1
    nrows = xrows + 2
    label_row = nrows

    fig = plotly_make_subplots(
        rows=nrows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_heights=[0.5] * xrows + [0.25, 0.25],
        specs=[[{
            "secondary_y": False
        }] for _ in range(nrows)],
    )
    # Add vertical black solid line at x=0 to all subplots
    for row in range(1, nrows + 1):
        fig.add_shape(
            type="line",
            x0=0,
            x1=0,
            y0=0,
            y1=1,
            line=dict(
                color="black",
                width=1,
                dash="solid",
            ),
            xref="x" + (str(row) if row > 1 else ""),
            yref="paper",
            opacity=.8,
            layer="above"
        )
    # Annotate left and right windows with LaTeX labels
    fig.add_annotation(
        text=r"$W_H$",
        x=-history_length * 0.5,
        y=0.9,
        xref=f"x1",
        yref="paper",
        showarrow=False,
        font=dict(size=17, color="black"),
        align="center",
        bgcolor="rgba(255,255,255,0.0)",
        borderpad=2,
        row=1,
        col=1,
    )
    fig.add_annotation(
        text=r"$W_F$",
        x=seq_length * 0.5,
        y=.9,
        xref=f"x1",
        yref="paper",
        showarrow=False,
        font=dict(size=17, color="black"),
        align="center",
        bgcolor="rgba(255,255,255,0.0)",
        borderpad=2,
        row=1,
        col=1,
    )
    # X channels subplot
    for channel_i in range(n_channels):
        channel_color = COLOR_SCALE[channel_i % len(COLOR_SCALE)]
        darker_ch_color = plt_colors.find_intermediate_color(
            'rgb(0,0,0)', plt_colors.label_rgb(plt_colors.hex_to_rgb(channel_color)), 0.7, 'rgb'
        )
        channel_name = CHANNEL_NAMES[channel_i]
        ####  TARGET TRACE  ####
        fig.add_trace(
            go.Scatter(
                x=np.arange(seq_length),
                y=target_samples[0, channel_i, :],
                mode='lines',
                line=dict(color=darker_ch_color, width=2),
                opacity=1,
                name=f'{channel_name} (Real)',
                legendgroup=f'X',
                legendgrouptitle_text=r"Observables $\mathbf{x}_W$",
            ),
            row=channel_i + 1,
            col=1
        )
        if generated_samples is not None:
            ####  GENERATED TRACES  ####
            for i, sample in enumerate(generated_samples):
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(seq_length),
                        y=sample[channel_i, :],
                        mode='lines',
                        line=dict(dash='solid', color=channel_color, width=1.2),
                        showlegend=i == 0,
                        opacity=0.5,
                        name=f'{channel_name} (Generated)',
                        legendgroup=f'X',
                    ),
                    row=channel_i + 1,
                    col=1
                )
        show_history = conditioning_input is not None and "x_history" in conditioning_input
        if show_history:
            x_history = conditioning_input['x_history'][0]  # type: ignore
            fig.add_trace(
                go.Scatter(
                    x=np.arange(-history_length, 0),
                    y=x_history[channel_i, :],
                    mode='lines',
                    line=dict(color=darker_ch_color, width=2),
                    opacity=1,
                    name=f'{channel_name} (history)',
                    showlegend=False,
                    legendgroup=f'x',
                ),
                row=channel_i + 1,
                col=1
            )
    # C channels subplot
    if has_c:
        c_input = conditioning_input["c"][0]  # type: ignore
        c_channels = c_input.shape[0]
        c_axis_values = np.arange(-history_length, seq_length)
        C_CHANNEL_NAMES = C.data.cols.c
        for channel_j in range(c_channels):
            channel_color = C_COLOR_SCALE[channel_j % len(C_COLOR_SCALE)]
            channel_name = C_CHANNEL_NAMES[channel_j]
            fig.add_trace(
                go.Scatter(
                    x=c_axis_values,
                    y=c_input[channel_j, :],
                    mode='lines',
                    line=dict(color=channel_color, width=2),
                    opacity=0.9,
                    name=f'{channel_name}',
                    legendgroup=f'(C)',
                    legendgrouptitle_text=r"Controls $\mathbf{c}_W$",
                ),
                row=row_c,
                col=1
            )
    # Label bar subplot
    if surr_labels_target is not None:
        add_mode_bars(fig, history_length, seq_length, surr_labels_target[0] + 1, layer=0, group="real", row=label_row)
    if surr_labels_pred is not None:
        for i, label_sequence in enumerate(surr_labels_pred):
            add_mode_bars(fig, history_length, seq_length, label_sequence + 1, layer=i + 1, group="gen", row=label_row)
        # Move the last two bar traces to the label bar row
        # (Plotly doesn't support bar row assignment directly, so we move them after creation)

    # Layout
    fig.update_layout(
        title=dict(
            text=title,
            x=.99,
            y=.99,
            xanchor='right',
            yanchor='top',
            font=dict(family="serif", size=14),
        ),
        template='ggplot2',
        font=dict(family="serif", size=12),
        hovermode='closest',
        margin=dict(l=10, r=20, t=0, b=20),
        height=1500,
        width=1000,
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.025,
            yref='paper',
            # xanchor="left",
            # x=1,
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0)",
            # valign="middle",  # Use vertical space
            itemsizing="constant",
            traceorder='grouped',
        ),
        barmode='stack',
        barcornerradius=1,
    )
    ### AXES AND TICKS ###
    x_ticks = list(range(-500, 501, 50))
    ticktext = x_ticks.copy()
    middle_time = conditioning_input['position_sequence'][0, -seq_length]
    ticktext[len(x_ticks) // 2] = f"$t={middle_time:0.3f}$"
    fig.update_xaxes(
        range=(-history_length, seq_length),
        showticklabels=False,
        showgrid=True,
        ticks="",
        tickvals=x_ticks,
    )
    # bottom X axis
    fig.update_xaxes(
        range=(-history_length - 1, seq_length + 1),
        showticklabels=True,
        title_text="Time steps (0.1ms/step)",
        showgrid=False,
        row=nrows,
        col=1,
        tickvals=x_ticks,
        ticktext=ticktext,
        ticks="inside"
    )
    # Remove y ticks on the last subplot (label bar)
    for i, x_col in enumerate(CHANNEL_NAMES):
        fig.update_yaxes(
            title_text=f"$\\mathbf{{x}}_\\text{{{x_col}}}$",
            showticklabels=True,
            #  range=(-0.05, 1.01),
            row=1 + i,
            col=1
        )
    fig.update_yaxes(title_text="$\mathbf{c}$", showticklabels=True, range=(-0.02, 1.02), row=row_c, col=1)
    fig.update_yaxes(title_text="$\mathbf{y}$", showticklabels=True, row=label_row, col=1)
    # DUMMY BARS
    # Add manual legend entries for the bars: H (red), D (orange), L (blue)
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[0],
            marker=dict(color="red", opacity=0.5),
            name="High",
            showlegend=True,
            legendgroup="Modes",
            legendgrouptitle_text="Confinement Modes",
        ),
        row=label_row if label_row is not None else nrows,
        col=1
    )
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            marker=dict(color="orange", opacity=0.5),
            name="Dithering",
            showlegend=True,
            legendgroup="Modes",
        ),
        row=label_row if label_row is not None else nrows,
        col=1
    )
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            marker=dict(color="lightskyblue", opacity=0.5),
            name="Low",
            showlegend=True,
            legendgroup="Modes",
        ),
        row=label_row if label_row is not None else nrows,
        col=1
    )
    # if has_c:
    #     fig.update_yaxes(range=(-1.1, 1.1), row=row_c, col=1)
    if wandb.run.disabled:  # type: ignore
        fig.show()
    return fig
