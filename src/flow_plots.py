# @title Utility code: styles, functions, generators, visualization
from typing import Optional
import torch
import wandb
import numpy as np
from scipy.signal import find_peaks
from matplotlib import gridspec
from matplotlib import colors

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly import colors as plt_colors
from plotly.subplots import make_subplots as plotly_make_subplots

from src.config import get_current_config
import logging

logger = logging.getLogger(__name__)

# Partial code adapted from https://drscotthawley.github.io/blog/posts/FlowModels.html
# for accessibility: Wong's color pallette: cf. https://davidmathlogic.com/colorblind
#WONG_black = [0/255, 0/255, 0/255]          # #000000
WONG_amber = [230 / 255, 159 / 255, 0 / 255]  # #E69F00
WONG_cyan = [86 / 255, 180 / 255, 233 / 255]  # #56B4E9
WONG_green = [0 / 255, 158 / 255, 115 / 255]  # #009E73
WONG_yellow = [240 / 255, 228 / 255, 66 / 255]  # #F0E442
WONG_navy = [0 / 255, 114 / 255, 178 / 255]  # #0072B2
WONG_red = [213 / 255, 94 / 255, 0 / 255]  # #D55E00
WONG_pink = [204 / 255, 121 / 255, 167 / 255]  # #CC79A7
BRIGHTNESS_FACTOR = 3  # values > 1 brighten, < 1 darken
WONG_cmap = [WONG_amber, WONG_cyan, WONG_green, WONG_yellow, WONG_navy, WONG_red, WONG_pink]
for i in range(len(WONG_cmap)):
    WONG_cmap[i][:] = [x**(1 / BRIGHTNESS_FACTOR) for x in WONG_cmap[i]]

SOURCE_COLOR = WONG_navy
TARGET_COLOR = WONG_red
PRED_COLOR = WONG_green
LINE_COLOR = WONG_yellow
plt.style.use('dark_background')


def plot_distributions(dist1, dist2, title1="Distribution 1", title2="Distribution 2", alpha=0.8, show=True):
    """Plot two distributions side by side

    By https://drscotthawley.github.io/blog/posts/FlowModels.html 
    """
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    ax1.scatter(dist1[:, 0], dist1[:, 1], alpha=alpha, s=10, color=SOURCE_COLOR)
    ax2.scatter(dist2[:, 0], dist2[:, 1], alpha=alpha, s=10, color=TARGET_COLOR)

    ax1.set_title(title1)
    ax2.set_title(title2)

    # Set same scale for both plots
    max_range = max(abs(dist1).max().item(), abs(dist2).max().item())
    for ax in [ax1, ax2]:
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_aspect('equal')

    plt.tight_layout()
    if show:
        plt.show(block=False)
        plt.pause(1.0)
    return wandb.Image(fig)


@torch.inference_mode()
def plot_flow(
    target_samples,
    prior_samples,
    generated_samples,
    trajectories,
    n=30,
    title_base="",
    size=20,  # Size of scatter plot points
    alpha=0.5,  # Transparency of scatter plot points
    **kwargs
):
    """Call the integrator to calculate the motion (probability path) given v field, generate new samples
       and visualize the results.

    Args:
        val_points (torch.Tensor): Initial points, shape: [batch_size, num_features].
        target_samples (torch.Tensor): Target samples, shape: [batch_size, num_features].
        trained_model (torch.nn.Module): Trained model to generate new samples.
        size (int, optional): Size of scatter plot points. Defaults to 20.
        alpha (float, optional): Transparency of scatter plot points. Defaults to 0.5.
        n_steps (int, optional): Number of integration steps. Defaults to 100.
        warp_fn (callable, optional): Optional function to warp time steps. Defaults to None.

    Returns:
        None
    """
    # select the first channel for everything
    prior_samples = prior_samples[:, 0, :]
    generated_samples = generated_samples[:, 0, :]
    target_samples = target_samples[:, 0, :]
    trajectories = trajectories[:, :, 0, :]  # Shape: [n_steps, n_samples, num_timepoints]

    n_viz = min(n, target_samples.size(0))  # Number of trajectories to visualize
    plt.close('all')
    fig, ax = plt.subplots(1, 4, figsize=(13, 4))
    plt.suptitle(title_base, fontsize=16)
    data_list = [prior_samples.cpu(), generated_samples.cpu(), target_samples.cpu()]
    label_list = ['Initial Points', 'Generated Samples', 'Target Data', 'Trajectories']
    color_list = [SOURCE_COLOR, PRED_COLOR, TARGET_COLOR]
    max_abs_value = torch.max(torch.abs(torch.cat(data_list)), 0)[0]
    global_max_2d = max(max_abs_value[0], max_abs_value[1])
    for i in range(len(label_list)):
        ax[i].set_title(label_list[i])
        ax[i].set_xlim([-global_max_2d, global_max_2d])
        ax[i].set_ylim([-global_max_2d, global_max_2d])
        if i < 3:  # non-trajectory plots
            ax[i].scatter(
                data_list[i][:, 0],
                data_list[i][:, 1],
                s=size,
                alpha=alpha,
                label=label_list[i],
                color=color_list[i]
            )
        else:
            # Plot trajectory paths first
            for j in range(n_viz):
                path = trajectories[:, j]  # Shape: [n_steps, num_features]
                ax[3].plot(path[:, 0], path[:, 1], '-', color=LINE_COLOR, alpha=1, linewidth=1)

            # Then plot start and end points for the SAME trajectories
            start_points = trajectories[0, :n_viz]  # Shape: [n_viz, num_features]
            end_points = trajectories[-1, :n_viz]  # Shape: [n_viz, num_features]
            ax[3].scatter(
                start_points[:, 0],
                start_points[:, 1],
                color=SOURCE_COLOR,
                s=size,
                alpha=1,
                label='Source Points'
            )
            ax[3].scatter(
                end_points[:, 0],
                end_points[:, 1],
                color=PRED_COLOR,
                s=size,
                alpha=1,
                label='Current Endpoints'
            )
            ax[3].legend()
    plt.tight_layout()
    if wandb.run.disabled:
        plt.show(block=False)
        plt.pause(1.0)
    return wandb.Image(fig)


def plot_flow_and_lines(
    target_samples,
    prior_samples,
    generated_samples,
    trajectories,
    n=30,
    title_base="",
    size=20,  # Size of scatter plot points
    alpha=0.5,  # Transparency of scatter plot points
    **kwargs
):
    """Call the integrator to calculate the motion (probability path) given v field, generate new samples
       and visualize the results.

    Args:
        val_points (torch.Tensor): Initial points, shape: [batch_size, num_features].
        target_samples (torch.Tensor): Target samples, shape: [batch_size, num_features].
        trained_model (torch.nn.Module): Trained model to generate new samples.
        size (int, optional): Size of scatter plot points. Defaults to 20.
        alpha (float, optional): Transparency of scatter plot points. Defaults to 0.5.
        n_steps (int, optional): Number of integration steps. Defaults to 100.
        warp_fn (callable, optional): Optional function to warp time steps. Defaults to None.

    Returns:
        None
    """
    # select the first channel for everything
    prior_samples = prior_samples[:, 0, :]
    generated_samples = generated_samples[:, 0, :]
    target_samples = target_samples[:, 0, :]
    trajectories = trajectories[:, :, 0, :]  # Shape: [n_steps, n_samples, num_timepoints]
    n_steps, num_samples, num_features = trajectories.size()
    num_samples = min(n, num_samples)  # Number of trajectories to visualize
    data_list = [prior_samples, generated_samples, target_samples]
    FACET_LIST = [
        'Initial Points',
        'Generated Samples',
        'Target Data',
        'Trajectories',
    ]
    color_list = [SOURCE_COLOR, PRED_COLOR, TARGET_COLOR]
    max_abs_value = torch.max(torch.abs(torch.cat(data_list)), 0)[0]
    global_max = max(max_abs_value[0], max_abs_value[1])

    plt.close('all')
    fig = plt.figure(figsize=(13, 8))  # Adjusted figsize to accommodate 2 rows
    gs = gridspec.GridSpec(2, 4, height_ratios=[1, 2])
    plt.suptitle(title_base, fontsize=16)
    for facet_i in range(len(FACET_LIST)):
        ax = fig.add_subplot(gs[0, facet_i])
        ax.set_title(FACET_LIST[facet_i])
        ax.set_xlim([-global_max, global_max])
        ax.set_ylim([-global_max, global_max])
        if facet_i < 3:  # non-trajectory plots
            ax.scatter(
                data_list[facet_i][:, 0],
                data_list[facet_i][:, 1],
                s=size,
                alpha=alpha,
                label=FACET_LIST[facet_i],
                color=color_list[facet_i]
            )
        else:
            # Plot trajectory paths first
            for j in range(num_samples):
                path = trajectories[:, j]  # Shape: [n_steps, num_features] (one sample)
                ax.plot(path[:, 0], path[:, 1], '-', color=LINE_COLOR, alpha=1, linewidth=1)

            # Then plot start and end points for the SAME trajectories
            start_points = trajectories[0, :num_samples]  # Shape: [n_viz, num_features]
            end_points = trajectories[-1, :num_samples]  # Shape: [n_viz, num_features]
            ax.scatter(
                start_points[:, 0],
                start_points[:, 1],
                color=SOURCE_COLOR,
                s=size,
                alpha=1,
                label='Source Points'
            )
            ax.scatter(
                end_points[:, 0],
                end_points[:, 1],
                color=PRED_COLOR,
                s=size,
                alpha=1,
                label='Current Endpoints'
            )
            ax.legend()

    # Plot each sample from generated_samples in a line plot against their corresponding target_samples
    axbig = fig.add_subplot(gs[1, :])
    for sample_i in range(len(generated_samples)):
        axbig.plot(
            range(num_features),
            generated_samples[sample_i, :],
            color=WONG_cmap[sample_i % len(WONG_cmap)],
            alpha=alpha,
            label=f'Generated {sample_i+1}'
        )
        axbig.plot(range(num_features), target_samples[sample_i, :], color=TARGET_COLOR, alpha=alpha * 0.5)

    axbig.set_title('Generated vs Target Samples')
    axbig.set_xlabel('Time Steps')
    axbig.set_ylabel('Value')

    plt.tight_layout()
    if wandb.run.disabled:
        plt.show(block=False)
        plt.pause(1.0)
    return wandb.Image(fig)


@torch.inference_mode()
def plot_flow_and_lines_plotly(
    target_samples: torch.Tensor,
    prior_samples: torch.Tensor,
    generated_samples: torch.Tensor,
    trajectories: torch.Tensor,
    n=30,
    title_base="",
    size=5,  # Size of scatter plot points
    alpha=0.9,  # Transparency of scatter plot points
    **kwargs
):
    """Call the integrator to calculate the motion (probability path) given v field, generate new samples
       and visualize the results using Plotly.

    Args:
        val_points (torch.Tensor): Initial points, shape: [batch_size, num_features].
        target_samples (torch.Tensor): Target samples, shape: [batch_size, num_features].
        trained_model (torch.nn.Module): Trained model to generate new samples.
        size (int, optional): Size of scatter plot points. Defaults to 20.
        alpha (float, optional): Transparency of scatter plot points. Defaults to 0.5.
        n_steps (int, optional): Number of integration steps. Defaults to 100.
        warp_fn (callable, optional): Optional function to warp time steps. Defaults to None.

    Returns:
        None
    """
    # select the first channel for everything
    prior_samples = prior_samples[:, 0, :]
    generated_samples = generated_samples[:, 0, :]
    target_samples = target_samples[:, 0, :]
    trajectories = trajectories[:, :, 0, :]  # Shape: [n_steps, n_samples, num_timepoints]
    n_steps, num_samples, num_features = trajectories.size()
    num_samples = min(n, num_samples)  # Number of trajectories to visualize
    data_list = [prior_samples, generated_samples, target_samples]
    FACET_LIST = ['Initial Points', 'Generated Samples', 'Target Data', 'Paths']
    max_abs_value = torch.max(torch.abs(torch.cat(data_list)), 0)[0]
    global_max = max(max_abs_value[0], max_abs_value[1]) * 1.1  # Add some padding
    color_scale = plt_colors.qualitative.Plotly

    fig = plotly_make_subplots(
        rows=2,
        cols=4,
        subplot_titles=FACET_LIST,
        specs=[[{}, {}, {}, {}], [{
            "colspan": 4
        }, None, None, None]],
        row_heights=[0.4, 0.6],
        vertical_spacing=0.05,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    # Set same x and y axis range for the first 4 subplots
    for i in range(1, 5):
        fig.update_xaxes(range=[-global_max, global_max], row=1, col=i)
        fig.update_yaxes(range=[-global_max, global_max], row=1, col=i)

    for facet_i in range(len(FACET_LIST)):
        if facet_i < 3:  # non-trajectory plots
            fig.add_trace(
                go.Scatter(
                    x=data_list[facet_i][:, 0],
                    y=data_list[facet_i][:, 1],
                    mode='markers',
                    marker=dict(
                        size=size,
                        opacity=alpha,  #color=color_list[facet_i]
                    ),
                    name=FACET_LIST[facet_i],
                    yaxis='y1',
                    xaxis='x1',
                ),
                row=1,
                col=facet_i + 1
            )
        else:
            # Plot trajectory paths first
            for j in range(num_samples):
                path = trajectories[:, j]  # Shape: [n_steps, num_features] (one sample)
                fig.add_trace(
                    go.Scatter(
                        x=path[:, 0],
                        y=path[:, 1],
                        mode='lines',
                        line=dict(color=colors.rgb2hex(LINE_COLOR), width=1),
                        opacity=1,
                        showlegend=False,
                        yaxis='y1',
                        xaxis='x1',
                    ),
                    row=1,
                    col=facet_i + 1
                )

            # Then plot start and end points for the SAME trajectories
            start_points = trajectories[0, :num_samples]  # Shape: [n_viz, num_features]
            end_points = trajectories[-1, :num_samples]  # Shape: [n_viz, num_features]
            fig.add_trace(
                go.Scatter(
                    x=start_points[:, 0],
                    y=start_points[:, 1],
                    mode='markers',
                    # marker=dict(size=size, opacity=1, color=SOURCE_COLOR),
                    name='Source Points',
                    yaxis='y1',
                    xaxis='x1',
                ),
                row=1,
                col=facet_i + 1
            )
        fig.update_xaxes(dtick=0.5, row=1, col=facet_i + 1)
        fig.update_yaxes(dtick=0.5, row=1, col=facet_i + 1)

    # Plot each sample from generated_samples in a line plot against their corresponding target_samples
    for sample_i in range(num_samples):
        # Predicted rollouts:
        fig.add_trace(
            go.Scatter(
                x=list(range(num_features)),
                y=generated_samples[sample_i, :],
                mode='lines',
                line=dict(dash='dot', color=color_scale[sample_i]),
                opacity=alpha,
                name=f'Generated {sample_i+1}',
                yaxis='y2',
                xaxis='x2',
            ),
            row=2,
            col=1
        )

        # Plot current endpoints with the same color as the generated sample
        fig.add_trace(
            go.Scatter(
                x=[end_points[sample_i, 0]],
                y=[end_points[sample_i, 1]],
                mode='markers',
                marker=dict(size=size, opacity=1, color=color_scale[sample_i]),
                name=f'Current Endpoint {sample_i+1}',
                showlegend=False,
                yaxis='y1',
                xaxis='x1',
            ),
            row=1,
            col=4
        )

        # Ground truth rollouts:
        fig.add_trace(
            go.Scatter(
                x=list(range(num_features)),
                y=target_samples[sample_i, :],
                mode='lines',
                line=dict(color=colors.rgb2hex(TARGET_COLOR), width=2),
                opacity=alpha * 0.5,
                name=f'Target {sample_i+1}',
                yaxis='y2',
                xaxis='x2',
            ),
            row=2,
            col=1
        )

    fig.update_layout(
        title=title_base,
        # height=800,
        template='plotly_dark' if BG_THEME in ['black', 'dark'] else 'plotly_white'
    )
    if wandb.run.disabled:  # type: ignore
        fig.show()

    return fig


def multi_channel_lines_plotly(
    meta: dict[str, torch.Tensor],
    target_samples: torch.Tensor,
    generated_samples: torch.Tensor,
    conditioning_input: dict[str, torch.Tensor],
    peak_features: Optional[dict] = None,
    show_c: bool = True,
    n=5,
    title_base="",
    subtitle="",  # Subtitle for the plot
    buttons=False,
    label_bars=True,
    **kwargs  # catch-all for other arguments from evaluate.py
):
    """Create a simple line plot where each channel is a separate color, shots are overlaid, and predictions are show in dotted lines.

    Args:
        meta (dict): Dictionary containing metadata about the samples.
        target_samples (torch.Tensor): Target samples, shape: [num_samples, num_channels, num_timepoints].
        generated_samples (torch.Tensor): Generated samples, shape: [num_samples, num_channels, num_timepoints].
        conditioning_input (dict): Dictionary containing the conditioning input.
        show_c (bool): Whether to show the conditioning channels, if available.
        n (int, optional): Number of traces to visualize. Defaults to 5.
        title_base (str, optional): Base title for the plot. Defaults to "".
        subtitle (str, optional): Subtitle for the plot. Defaults to "".
        buttons (bool, optional): Whether to add buttons to the plot. Defaults to False.

    The legend is grouped by shot.
    Legend format is: 
        Shot 1 - Target:
            Target, Shot 2: Target, Shot 2: Predicted, ..."
    """
    C = get_current_config()
    CHANNEL_NAMES = C.data.cols.x
    history_length = C.data.history_length
    seq_length = C.data.seq_length
    # Generate and visualize new samples
    show_history = "x_history" in conditioning_input
    if show_c and "c" in conditioning_input:
        c_input = conditioning_input["c"] - 1  # Translate everything in c to -1 to 0
        c_channels = c_input.size(1)
        c_axis_values = np.arange(-history_length, seq_length)
        C_CHANNEL_NAMES = C.data.cols.c
    else:
        c_channels = 0  # skips the loop below
        C_CHANNEL_NAMES = []
    position_sequence = conditioning_input["position_sequence"]
    labels = conditioning_input.get('label').numpy()
    shot_numbers = meta["shot_number"]
    start_times = meta["start"]
    end_times = meta["end"]
    num_samples, n_channels, num_timepoints = target_samples.size()
    num_samples = min(n, num_samples)  # Number of traces to visualize
    COLOR_SCALE = plt_colors.qualitative.Plotly
    C_COLOR_SCALE = plt_colors.qualitative.Set2
    mse = ((target_samples[:num_samples] - generated_samples[:num_samples])**2).mean().item()
    subtitle = f"MSE: {mse:.4f}" + (f" | {subtitle}" if subtitle else "")

    # Initialize the figure
    fig = plotly_make_subplots(
        rows=1,
        cols=1,
        specs=[[{
            "secondary_y": True
        }]],
        subplot_titles=[title_base],
        # vertical_spacing=0.05,
    )
    fig.update_yaxes(
        range=(-1.5, None),
        secondary_y=False,
    )
    fig.update_xaxes(
        range=(-history_length, seq_length),
        showticklabels=True,
        title_text="Time steps (0.1ms/step)",
        showgrid=True
    )
    fig.update_layout(
        title=title_base + (f"<br><sub>{subtitle}</sub>" if subtitle else ""),
        template='plotly_dark',
        hovermode='closest',
        barmode='stack',
        barcornerradius=15,
    )
    if show_history:
        # draw a vertical line around x = 0 to separate conditioning from prediction
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=-0.5,
            y0=0,
            y1=1,
            line=dict(
                color="yellow",
                width=3,
                dash="solid",
            ),
            xref="x",
            yref="paper",
            opacity=0.5,
        )
        fig.add_shape(
            type="rect",
            x0=-1,
            x1=0,
            y0=0,
            y1=1,
            fillcolor="yellow",
            opacity=0.3,
            line_width=0,
            xref="x",
            yref="paper",
        )
        x_history = conditioning_input['x_history']
    for shot_i in range(num_samples):
        shot_number = shot_numbers[shot_i]
        start_time = start_times[shot_i]
        end_time = end_times[shot_i]
        hover_info_template = "<b>%{y:.5f}</b><br>t: %{x:,}<br><br>" + f"<em>Shot #{shot_number}</em><br>Time span: {start_time:.4f}s-{end_time:.4f}s"
        shot_i_labels = labels[shot_i]

        if label_bars:
            add_mode_bars(fig, history_length, seq_length, num_samples, labels, shot_i, shot_number)
        # Plot target samples
        for channel_i in range(n_channels):
            channel_color = COLOR_SCALE[((n_channels * shot_i) + channel_i) % len(COLOR_SCALE)]
            channel_name = CHANNEL_NAMES[channel_i]
            target_trace = target_samples[shot_i, channel_i, :].numpy()

            if peak_features:
                pred_peak_features = peak_features['pred_peaks'][channel_i][shot_i]
                target_peak_features = peak_features['target_peaks'][channel_i][shot_i]
                # Find peaks and plot them
                add_peak_markers(
                    fig,
                    target_peak_features,
                    "Target",
                    shot_number,
                    hover_info_template,
                    channel_color,
                    channel_name,
                )
                add_peak_markers(
                    fig,
                    pred_peak_features,
                    "Predicted",
                    shot_number,
                    hover_info_template,
                    channel_color,
                    channel_name,
                )
            # Plot target traces
            fig.add_trace(
                go.Scatter(
                    x=np.arange(seq_length),
                    y=target_trace,
                    mode='lines',
                    line=dict(color=channel_color, width=3),
                    opacity=0.6,
                    customdata=shot_i_labels[history_length:],  # Only show labels for the prediction part
                    name=f'{channel_name} (target)',
                    legendgroup=f'Shot {shot_number} - Target',
                    legendgrouptitle_text=f'Shot {shot_number} - Target',
                    hovertemplate="<b>%{y:.5f}</b><br>t: %{x:,}<br>Label: %{customdata}<br><br>" +
                    f"<em>Shot #{shot_number}</em><br>Time span: {start_time:.4f}s-{end_time:.4f}s"
                )
            )
            # Plot target samples
            if show_history:
                history_start_time = meta['history_start'][shot_i]
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(-history_length, 0),
                        y=x_history[shot_i, channel_i, :],
                        mode='lines',
                        line=dict(color=channel_color, width=3),
                        opacity=0.8,
                        customdata=shot_i_labels[:history_length],  # Only show labels for the history part
                        name=f'{channel_name} (history)',
                        legendgroup=f'Shot {shot_number} - History',
                        legendgrouptitle_text=f'Shot {shot_number} - History',
                        hovertemplate="<b>%{y:.5f}</b><br>t: %{x:,}<br>Label: %{customdata}<br><br>" +
                        f"<em>Shot #{shot_number}</em><br>History time span: {history_start_time:.4f}s-{start_time:.4f}s"
                    )
                )

            # Plot generated traces
            fig.add_trace(
                go.Scatter(
                    x=np.arange(seq_length),
                    y=generated_samples[shot_i, channel_i, :],
                    mode='lines',
                    line=dict(dash='dot', color=channel_color),
                    opacity=0.9,
                    name=f'{channel_name} (predicted)',
                    legendgroup=f'Shot {shot_number} - Predicted',
                    legendgrouptitle_text=f'Shot {shot_number} - Predicted',
                    hovertemplate="<b>%{y:.5f}</b><br>t: %{x:,}<br><br>" +
                    f"<em>Shot #{shot_number}</em><br>Time span: {start_time:.4f}s-{end_time:.4f}s"
                )
            )
        for channel_j in range(c_channels):
            # Plot C (covariate) traces to the bottom of the plot
            channel_color = C_COLOR_SCALE[channel_j]
            channel_name = C_CHANNEL_NAMES[channel_j]
            fig.add_trace(
                go.Scatter(
                    x=c_axis_values,
                    y=c_input[shot_i, channel_j, :],
                    mode='lines',
                    line=dict(color=channel_color, width=4),
                    opacity=0.7,
                    name=f'{channel_name} (C)',
                    legendgroup=f'Shot {shot_number} - C',
                    legendgrouptitle_text=f'Shot {shot_number} - C',
                    hovertemplate="<b>%{y:.5f}</b><br>t: %{x:,}<br><br>" +
                    f"<em>Shot #{shot_number}</em><br>Time span: {start_time:.4f}s-{end_time:.4f}s"
                )
            )
    if buttons:
        # Add a button to focus on every shot individually, and all shots
        not_predicted = [not trace.legendgroup.endswith('Predicted') for trace in fig.data]
        not_target = [not trace.legendgroup.endswith('Target') for trace in fig.data]
        main_button_list = [
            dict(
                label='All Shots',
                method='update',
                args=[{
                    'visible': [True] * len(fig.data)
                }, {
                    'title': 'All'
                }]
            ),
            dict(label='Targets', method='update', args=[{
                'visible': not_predicted
            }]),
            dict(label='Predicted', method='update', args=[{
                'visible': not_target
            }])
        ]
        for shot_i in range(num_samples):
            shot_num = str(shot_numbers[shot_i].item())
            main_button_list.append(
                dict(
                    args=[{
                        "visible": [shot_num in trace.legendgroup for trace in fig.data]
                    }],
                    label=shot_num,
                    method="update"
                )
            )
        channel_buttons = [
            dict(label='All', method='update', args=[{
                'visible': [True] * len(fig.data)
            }]),
        ]
        for channel in (CHANNEL_NAMES + C_CHANNEL_NAMES):
            channel_buttons.append(
                dict(
                    args=[{
                        "visible": [trace.name.startswith(channel) for trace in fig.data]
                    }],
                    label=channel,
                    method="update"
                )
            )

        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=main_button_list,
                    showactive=True,
                    direction="down",
                    x=1.02,
                    xanchor="left",
                    y=1.02,
                    yanchor="bottom"
                ),
                dict(
                    active=0,
                    buttons=channel_buttons,
                    showactive=True,
                    direction="down",
                    x=1.019,
                    xanchor="right",
                    y=1.02,
                    yanchor="bottom"
                )
            ]
        )

    if wandb.run.disabled:  # type: ignore
        fig.show()

    return fig


def add_peak_markers(
    fig,
    peak_features,
    group: str,
    shot_number,
    hover_info_template,
    channel_color,
    channel_name,
):
    peak_x_markers = []
    peak_y_markers = []
    peak_width_y_markers = []
    peak_width_x_markers = []
    for peak in peak_features:
        peak_x_markers.extend([peak.X, peak.X, None])  # None creates a break between lines
        peak_y_markers.extend([peak.bases, peak.Y, None])
        peak_width_y_markers.extend([peak.bases, peak.bases, None])
        peak_width_x_markers.extend([peak.left_ips, peak.right_ips, None])


    fig.add_trace(  # Peak markers traces
                    go.Scatter(
                        x=peak_x_markers,
                        y=peak_y_markers,
                        mode='markers+lines',
                        marker=dict(
                            size=10,
                            color=channel_color,
                            symbol="circle-open-dot",
                            angleref="previous",
                            opacity=0.8,
                        ),
                        opacity=0.9,
                        line=dict(
                            color=channel_color,
                            dash="solid",
                            width=0.8,
                        ),
                        name=f'{channel_name} (peaks)',
                        legendgroup=f'Shot {shot_number} - Peaks {group}',
                        legendgrouptitle_text=f'Shot {shot_number} - Peaks {group}',
                        hovertemplate=f"<b>{group}</b><br>{hover_info_template}"
                    ),
                    secondary_y=False
                )
    fig.add_trace(  # Peak markers traces
                go.Scatter(
                    x=peak_width_x_markers,
                    y=peak_width_y_markers,
                    mode='markers+lines',
                    line=dict(
                        color=channel_color,
                        dash="solid",
                        width=0.8,
                    ),
                    marker=dict(
                        size=5,
                        color=channel_color,
                        symbol="line-ns-open",
                        opacity=0.4,
                    ),
                    opacity=0.9,
                    name=f'{channel_name} (width)',
                    legendgroup=f'Shot {shot_number} - Peaks {group}',
                    legendgrouptitle_text=f'Shot {shot_number} - Peaks {group}',
                        hovertemplate=f"<b>{group}</b><br>{hover_info_template}"

                ),
                )


def add_mode_bars(fig, history_length, seq_length, num_samples, labels, shot_i, shot_number):
    BAR_WIDTH = 25
    MODE_COLORS = ["grey", "lightskyblue", "orange", "red"]
    MODE_NAMES = ["Unknown", "L", "D", "H"]
    fig.update_yaxes(
        range=(-BAR_WIDTH / 2, (BAR_WIDTH + num_samples) * 8),
        showticklabels=False,
        secondary_y=True,
        fixedrange=True,
        showgrid=False
    )
    shot_labels = labels[shot_i]
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
            y=(shot_i, shot_i),
            orientation='h',
            marker=dict(
                color="black",
                opacity=0,
            ),
            showlegend=False,  # Bar chart does not need a separate legend
            name=f'Shot #{shot_number} - Start',
            legendgroup=f'Shot {shot_number} - Modes',
            hoverinfo='skip',  # Disable hover for this trace
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Bar(
            x=spans,
            y=(shot_i,) * len(spans),
            width=BAR_WIDTH,
            orientation='h',
            marker=dict(
                color=colors,
                opacity=0.5,
            ),
            hovertemplate=
            "Mode: %{customdata[3]}<br>Shot #%{customdata[0]}<br>Time steps: %{customdata[1]} - %{customdata[2]}<br>(%{x} steps)",
            customdata=custom_data,
            showlegend=True,  # Bar chart does not need a separate legend
            name=f'Shot #{shot_number} - Modes',
            # hoverinfo=['skip'] + ['all'] * (len(spans) - 1),  # Disable hover for this trace
            legendgroup=f'Shot {shot_number} - Modes',
            # legendgrouptitle_text=f'Shot {shot_number} - Modes',
        ),
        secondary_y=True,
    )


def add_label_traces(fig, labels, seq_length, history_length, shot_numbers):
    """Add label traces to the plotly figure.

    Args:
        fig (go.Figure): The plotly figure to which the label traces will be added.
        labels (list): List of labels for each shot.
        seq_length (int): The length of the sequence.
        history_length (int): The length of the history.
        start_times (list): List of start times for each shot.
        end_times (list): List of end times for each shot.
        shot_numbers (list): List of shot numbers.

    Returns:
        None
    """
    # figure out, per shot, the start and end times of each consecutive mode
    for shot_idx, shot_labels in enumerate(labels):
        label_ranges = []
        current_label = shot_labels[0]
        start_idx = 0
        for t in range(1, len(shot_labels)):
            if shot_labels[t] != current_label:
                label_ranges.append((current_label, -history_length + start_idx, -history_length + t))
                current_label = shot_labels[t]
                start_idx = t
        # Add the last range
        label_ranges.append((current_label, -history_length + start_idx, seq_length - 1))  # TODO
        # Add scatter lines for each range
        for label, start_time, end_time in label_ranges:
            # fig.add_vrect(
            #     x0=start_time,
            #     x1=end_time,
            #     line_width=0,
            #     opacity=0.2,
            #     fillcolor=plt_colors.qualitative.Plotly[int(label)],
            #     annotation_text=f"{label} #{shot_numbers[shot_idx]}",
            #     annotation_position="top left",
            #     annotation_font_size=10,
            #     annotation_font_color="black",
            # )
            fig.add_trace(
                go.Scatter(
                    x=[start_time, end_time],
                    y=[0, 0],
                    mode='lines',
                    line=dict(color=plt_colors.qualitative.Plotly[int(label)], width=10),
                    opacity=0.5,
                    name=f'Shot #{shot_numbers[i]} - Label',
                    legendgroup=f'Shot {shot_numbers[i]} - Label',
                    legendgrouptitle_text=f'Shot {shot_numbers[i]} - Label',
                    hovertemplate=f"Shot #{shot_numbers[i]}<br>Label: {labels[i]}"
                )
            )
