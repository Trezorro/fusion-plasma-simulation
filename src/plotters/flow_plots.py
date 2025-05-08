# @title Utility code: styles, functions, generators, visualization
from typing import Optional
import torch
import wandb
import numpy as np
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


def plot_distributions_mpl(dist1, dist2, title1="Distribution 1", title2="Distribution 2", alpha=0.8, show=True):
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
                data_list[i][:, 0], data_list[i][:, 1], s=size, alpha=alpha, label=label_list[i], color=color_list[i]
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
                start_points[:, 0], start_points[:, 1], color=SOURCE_COLOR, s=size, alpha=1, label='Source Points'
            )
            ax[3].scatter(
                end_points[:, 0], end_points[:, 1], color=PRED_COLOR, s=size, alpha=1, label='Current Endpoints'
            )
            ax[3].legend()
    plt.tight_layout()
    if wandb.run.disabled:
        plt.show(block=False)
        plt.pause(1.0)
    return wandb.Image(fig)


def plot_flow_and_lines_mpl(
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
                start_points[:, 0], start_points[:, 1], color=SOURCE_COLOR, s=size, alpha=1, label='Source Points'
            )
            ax.scatter(end_points[:, 0], end_points[:, 1], color=PRED_COLOR, s=size, alpha=1, label='Current Endpoints')
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
    meta: dict[str, torch.Tensor],
    n=30,
    title_base="",
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
    SIZE = 5  # Size of scatter plot points
    # select the first channel for everything
    prior_samples = prior_samples[:, 0, :]
    generated_samples = generated_samples[:, 0, :]
    target_samples = target_samples[:, 0, :]
    trajectories = trajectories[:, :, 0, :]  # Shape: [n_steps, n_samples, num_timepoints]
    n_steps, num_samples, num_features = trajectories.size()
    num_samples = min(n, num_samples)  # Number of trajectories to visualize
    data_list = [prior_samples, generated_samples, target_samples]
    FACET_LIST = ['Initial Points', 'Generated Samples', 'Target Data', 'Paths']
    shot_numbers = meta["shot_number"]
    max_abs_value = torch.max(torch.abs(torch.cat(data_list)), 0)[0]
    global_max = max(max_abs_value[0], max_abs_value[1]) * 1.1  # Add some padding
    COLOR_SCALE = plt_colors.qualitative.Plotly

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
                        size=SIZE,
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
            # Plot yellow arrow trajectory paths first
            for j in range(num_samples):
                path = trajectories[:, j]  # Shape: [n_steps, num_features] (one sample)
                fig.add_trace(
                    go.Scatter(
                        x=path[:, 0],
                        y=path[:, 1],
                        mode='lines+markers',
                        line=dict(color=colors.rgb2hex(LINE_COLOR), width=1),
                        marker=dict(
                            symbol='triangle-up-dot',
                            size=6,
                            angleref='previous',
                            color=colors.rgb2hex(LINE_COLOR),
                            standoff=3,
                        ),
                        opacity=1,
                        name=f'Trajectory shot {shot_numbers[j]}',
                        legendgroup=f'Shot {shot_numbers[j]}',
                        showlegend=True,
                        yaxis='y1',
                        xaxis='x1',
                    ),
                    row=1,
                    col=facet_i + 1
                )

            # Then plot start and end points for the SAME trajectories
            start_points = trajectories[0, :num_samples]  # Shape: [n_viz, num_features]
            end_points = trajectories[-1, :num_samples]  # Shape: [n_viz, num_features]

        fig.update_xaxes(dtick=0.5, row=1, col=facet_i + 1)
        fig.update_yaxes(dtick=0.5, row=1, col=facet_i + 1)

    # Plot each sample from generated_samples in a line plot against their corresponding target_samples
    for sample_i in range(num_samples):
        # Predicted rollouts:
        group = f'Shot {shot_numbers[sample_i]}'
        color = COLOR_SCALE[sample_i % len(COLOR_SCALE)]

        fig.add_trace(
            go.Scatter(
                x=[start_points[sample_i, 0]],
                y=[start_points[sample_i, 1]],
                mode='markers',
                # marker=dict(size=size, opacity=1, color=SOURCE_COLOR),
                marker=dict(size=SIZE, opacity=1, color=color),
                name='Source Point',
                legendgroup=group,
                legendgrouptitle_text=group,
                yaxis='y1',
                xaxis='x1',
            ),
            row=1,
            col=4
        )

        # Plot current endpoints with the same color as the generated sample
        fig.add_trace(
            go.Scatter(
                x=[end_points[sample_i, 0]],
                y=[end_points[sample_i, 1]],
                mode='markers',
                marker=dict(size=SIZE, opacity=1, color=color),
                name='Endpoint',
                showlegend=True,
                legendgroup=group,
                yaxis='y1',
                xaxis='x1',
            ),
            row=1,
            col=4
        )
        # Traces in bottom row:
        fig.add_trace(
            go.Scatter(
                x=list(range(num_features)),
                y=generated_samples[sample_i, :],
                mode='lines',
                line=dict(dash='dot', color=color, width=0.5),
                opacity=0.7,
                name='Generated',
                legendgroup=group,
                yaxis='y2',
                xaxis='x2',
            ),
            row=2,
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(num_features)),
                y=prior_samples[sample_i, :],
                mode='lines',
                line=dict(dash='40, 2, 20, 2', color=color, width=2),
                opacity=0.85,
                name='Prior',
                legendgroup=group,
                yaxis='y2',
                xaxis='x2',
            ),
            row=2,
            col=1
        )
        # Ground truth rollouts:
        fig.add_trace(
            go.Scatter(
                x=list(range(num_features)),
                y=target_samples[sample_i, :],
                mode='lines',
                line=dict(dash='solid', color=color, width=2),
                opacity=1,
                name=f'Target',
                legendgroup=group,
                yaxis='y2',
                xaxis='x2',
            ),
            row=2,
            col=1
        )
    fig.update_yaxes(showgrid=False, row=2, col=1)
    fig.update_layout(
        title=title_base + "Flow Priors and Targets",
        # height=800,
        template='plotly_dark',
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
    surr_labels_target: Optional[torch.Tensor] = None,
    surr_labels_pred: Optional[torch.Tensor] = None,
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
    )
    fig.update_yaxes(
        range=(-1.1, 1.1),
        secondary_y=False,
    )
    fig.update_xaxes(
        range=(-history_length, seq_length), showticklabels=True, title_text="Time steps (0.1ms/step)", showgrid=True
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
    shot_ids = []  # to match buttons to shot number - time identifiers
    for shot_i in range(num_samples):
        start_time = start_times[shot_i]
        shot_sample_id = f"{shot_numbers[shot_i]}:{start_time:.2f}s"
        shot_ids.append(shot_sample_id)
        end_time = end_times[shot_i]
        hover_info_template = "<b>%{y:.5f}</b><br>t: %{x:,}<br><br>" + f"<em>Shot #{shot_sample_id}</em><br>Time span: {start_time:.4f}s-{end_time:.4f}s"
        shot_i_labels = labels[shot_i]

        if label_bars:
            add_mode_bars(fig, history_length, seq_length, num_samples, shot_i_labels, shot_i, shot_sample_id)
            if surr_labels_pred is not None:
                add_mode_bars(
                    fig,
                    history_length,
                    seq_length,
                    num_samples,
                    surr_labels_pred[shot_i],
                    shot_i,
                    shot_sample_id,
                    group='predicted',
                )
            if surr_labels_target is not None:
                add_mode_bars(
                    fig,
                    history_length,
                    seq_length,
                    num_samples,
                    surr_labels_target[shot_i],
                    shot_i,
                    shot_sample_id,
                    group='target',
                )
        # Plot target samples
        for channel_i in range(n_channels):
            channel_color = COLOR_SCALE[((n_channels * shot_i) + channel_i) % len(COLOR_SCALE)]
            channel_name = CHANNEL_NAMES[channel_i]
            target_trace = target_samples[shot_i, channel_i, :].numpy()

            if peak_features:
                pred_peak_features = peak_features['pred_peaks'][shot_i][channel_i]
                target_peak_features = peak_features['target_peaks'][shot_i][channel_i]
                # Find peaks and plot them
                add_peak_markers(
                    fig,
                    target_peak_features,
                    "Target",
                    shot_sample_id,
                    hover_info_template,
                    channel_color,
                    channel_name,
                )
                add_peak_markers(
                    fig,
                    pred_peak_features,
                    "Predicted",
                    shot_sample_id,
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
                    legendgroup=f'Shot {shot_sample_id} - Target',
                    legendgrouptitle_text=f'Shot {shot_sample_id} - Target',
                    hovertemplate="<b>%{y:.5f}</b><br>t: %{x:,}<br>Label: %{customdata}<br><br>" +
                    f"<em>Shot #{shot_sample_id}</em><br>Time span: {start_time:.4f}s-{end_time:.4f}s"
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
                        legendgroup=f'Shot {shot_sample_id} - History',
                        legendgrouptitle_text=f'Shot {shot_sample_id} - History',
                        hovertemplate="<b>%{y:.5f}</b><br>t: %{x:,}<br>Label: %{customdata}<br><br>" +
                        f"<em>Shot #{shot_sample_id}</em><br>History time span: {history_start_time:.4f}s-{start_time:.4f}s"
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
                    legendgroup=f'Shot {shot_sample_id} - Predicted',
                    legendgrouptitle_text=f'Shot {shot_sample_id} - Predicted',
                    hovertemplate="<b>%{y:.5f}</b><br>t: %{x:,}<br><br>" +
                    f"<em>Shot #{shot_sample_id}</em><br>Time span: {start_time:.4f}s-{end_time:.4f}s"
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
                    legendgroup=f'Shot {shot_sample_id} - C',
                    legendgrouptitle_text=f'Shot {shot_sample_id} - C',
                    hovertemplate="<b>%{y:.5f}</b><br>t: %{x:,}<br><br>" +
                    f"<em>Shot #{shot_sample_id}</em><br>Time span: {start_time:.4f}s-{end_time:.4f}s"
                )
            )
    if buttons:
        # Add a button to focus on every shot individually, and all shots
        not_predicted = [not trace.legendgroup.endswith('Predicted') for trace in fig.data]
        not_target = [not trace.legendgroup.endswith('Target') for trace in fig.data]
        main_button_list = [
            dict(label='All Shots', method='update', args=[{
                'visible': [True] * len(fig.data)
            }]),
            dict(label='Targets', method='update', args=[{
                'visible': not_predicted
            }]),
            dict(label='Predicted', method='update', args=[{
                'visible': not_target
            }])
        ]
        for shot_sample_id in shot_ids:
            main_button_list.append(
                dict(
                    label=shot_sample_id,
                    args=[{
                        "visible": [shot_sample_id in trace.legendgroup for trace in fig.data]
                    }],
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

        # update traces such that only the first shot is visible initially
        first_shot = shot_numbers[0].item()
        fig.update_traces(visible=False)
        for trace in fig.data:
            if trace.legendgroup.startswith(f'Shot {first_shot}'):
                trace.visible = True
        fig.update_layout(
            updatemenus=[
                dict(
                    active=3,
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
    do_plot_energy_delta = peak_features.energy_base_x is not None
    peak_x_markers = []
    peak_y_markers = []
    peak_width_y_markers = []
    peak_width_x_markers = []
    peak_energy_delta_y = []
    peak_energy_delta_x = []
    for peak in peak_features.iter_peaks():
        peak_x_markers.extend([peak.X, peak.X, None])  # None creates a break between lines
        peak_y_markers.extend([peak.bases, peak.Y, None])
        peak_width_y_markers.extend([peak.bases, peak.bases, None])
        peak_width_x_markers.extend([peak.left_ips, peak.right_ips, None])
        if do_plot_energy_delta:
            peak_energy_delta_y.extend([peak.Y, peak.Y - peak.energy_delta, None])
            peak_energy_delta_x.extend([peak.X, peak.energy_base_x, None])

    if do_plot_energy_delta:
        fig.add_trace(  # Peak energy delta markers
            go.Scatter(
                x=peak_energy_delta_x,
                y=peak_energy_delta_y,
                mode='lines+markers',
                line=dict(
                    color=channel_color,
                    dash="solid",
                    width=1,
                ),
                marker=dict(
                                symbol='triangle-up-dot',
                                size=6,
                                angleref='previous',
                    color=channel_color,
                                standoff=3,
                    opacity=0.8,
                            ),
                # marker=dict(
                #     size=8,
                #     symbol="triangle-down",
                # ),
                opacity=0.7,
                name=f'{channel_name} (energy delta)',
                legendgroup=f'Shot {shot_number} - Peaks {group}',
                legendgrouptitle_text=f'Shot {shot_number} - Peaks {group}',
                hovertemplate=f"<b>{group}</b><br>{hover_info_template}"
            )
        )
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


def add_mode_bars(fig, history_length, seq_length, num_samples, shot_labels, shot_i, shot_number, group: str = 'human'):
    BAR_WIDTH = 25
    MODE_COLORS = ["grey", "lightskyblue", "orange", "red"]
    MODE_NAMES = ["Unknown", "L", "D", "H"]
    GROUP_Y = {'human': 0, 'target': (BAR_WIDTH + num_samples) * 1, 'predicted': (BAR_WIDTH + num_samples) * 2}
    bar_y_placement = GROUP_Y[group] + shot_i
    fig.update_yaxes(
        range=(-BAR_WIDTH / 2, (BAR_WIDTH + num_samples) * 8),
        showticklabels=False,
        secondary_y=True,
        fixedrange=True,
        showgrid=False
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
            y=(bar_y_placement,) * len(spans),
            width=BAR_WIDTH,
            orientation='h',
            marker=dict(
                color=colors,
                opacity=0.5 if group == 'human' else 0.4,
            ),
            hovertemplate=
            "Mode: %{customdata[3]}<br>Shot #%{customdata[0]}<br>Time steps: %{customdata[1]} - %{customdata[2]}<br>(%{x} steps)",
            customdata=custom_data,
            showlegend=True,  # Bar chart does not need a separate legend
            name=f'Shot #{shot_number} - {group} Labels',
            # hoverinfo=['skip'] + ['all'] * (len(spans) - 1),  # Disable hover for this trace
            legendgroup=f'Shot {shot_number} - Modes',
            legendgrouptitle_text=f'Shot {shot_number} - Modes',
        ),
        secondary_y=True,
    )
