from typing import Tuple

import numpy as np
import plotly
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

import wandb
from src.config import get_current_config
from src.entropy import VALID_FUNCS, batch_entropy
from src.plotters.helpers import as_wandb_image


def plot_entropy(
    generated_samples: torch.Tensor,
    target_samples: torch.Tensor,
    metrics: dict,
    title_base: str = "Sample Entropy",
    group_names: Tuple[str, str] = ('predicted', 'target'),
    method="app_entropy",  # one of app_entropy, perm_entropy, spectral_entropy
    **kwargs
) -> wandb.Image:
    """Plot the entropy of two groups of time series, with a horizontal subplot (range 0-1) for each channel.

    Args:
        generated_s (torch.Tensor): A batch of time series (predicted). Shape (N, C, T).
        groupB (torch.Tensor): A batch of time series (target). Shape (N, C, T).
        channel_names (list[str]): A list of channel names.
        group_names (tuple, optional): A tuple of group names. Defaults to ('predicted', 'target').

    C must match between groupA and groupB and channel_names.
    N is the number of samples per group.
    T is the number of time steps.    
    """
    C = get_current_config()
    channel_names = C.data.cols.x
    SUBPLOT_HEIGHT = 100

    entropy_pred = batch_entropy(generated_samples, VALID_FUNCS[method])
    entropy_target = batch_entropy(target_samples, VALID_FUNCS[method])
    # target_means = entropy_target.mean(axis=0)
    # target_std = entropy_target.std(axis=0) + 1e-10
    # entropy_target = (entropy_target - target_means) / target_std
    # entropy_pred = (entropy_pred - target_means) / target_std
    distances = np.mean(np.abs(entropy_pred - entropy_target), axis=0)  # mean distance per channel
    batch_size, n_channels, seq_length = generated_samples.shape
    name_a, name_b = group_names
    color_a, color_b = 'orange', 'cyan'
    assert n_channels == len(
        channel_names
    ), f"Number of channels ({n_channels}) must match the length of channel_names ({len(channel_names)})."
    fig = make_subplots(
        rows=n_channels, cols=1, shared_xaxes=True, subplot_titles=[s + ' ' for s in channel_names]
    )
    for i, channel_name in enumerate(channel_names):
        # Add scatter trace, per group, for each channel, with N points per group
        # predicted
        fig.add_trace(
            go.Scatter(
                x=entropy_pred[:, i],
                y=np.arange(batch_size),
                # y=np.linspace(0, 1, batch_size),
                mode='markers',
                xaxis="x",
                name=channel_name,
                marker=dict(color=color_a),  # Set fixed color for group A
                legendgroup=name_a,
                legendgrouptitle_text=name_a,
                # hoverinfo="text",  # Enable custom hover text
                # text=f"Mean Distance: {distances[i]:.4f}",  # Add mean distance to hover info
            ),
            row=i + 1,
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=entropy_target[:, i],
                y=np.arange(batch_size),
                mode='markers',
                xaxis="x",
                marker=dict(color=color_b),  # Set fixed color for group A
                name=channel_name,
                # showlegend=(i == 0),  # Show legend only for the first trace of each group
                legendgroup=name_b,
                legendgrouptitle_text=name_b,  # Set legend group title only once
            ),
            row=i + 1,
            col=1
        )
        # Add tiny error lines between xA and xB for the same sample index
        for j in range(batch_size):
            sample_y_positions = np.arange(batch_size)
            fig.add_trace(
                go.Scatter(
                    x=[entropy_pred[j, i], entropy_target[j, i]],
                    y=[sample_y_positions[j], sample_y_positions[j]],
                    mode='lines',
                    line=dict(color="gray", width=1),
                    opacity=0.5,
                    hoverinfo='skip',  # Disable hover for these lines
                    name="Difference Line" if (i == 0 and j == 0) else None,  # Add legend entry only once
                    showlegend=(i == 0 and j == 0),  # Show legend only for the first line
                    legendgroup="Difference",  # Group all lines under the same legend group
                ),
                row=i + 1,
                col=1
            )

    INTER_MARGIN = 0.1
    for i, channel_name in enumerate(channel_names):
        # i = i + 1
        fig['layout'][f'annotations[{i}]']['x'] = -0.00  # Move subplot titles to the left
        fig['layout'][f'annotations[{i}]'][
            'y'] = 1 - (i + .5) / n_channels  # Move subplot titles to the middle of each subplot
        fig['layout'][f'annotations[{i}]']['yanchor'] = 'middle'  # Move subplot titles to the left
        fig['layout'][f'annotations[{i}]']['xanchor'] = 'right'

        fig['layout'][f'yaxis{i+1}']['domain'] = [
            1 - (i + (1 - INTER_MARGIN)) / n_channels,  # Bottom
            1 - (i + INTER_MARGIN) / n_channels  # Top
        ]
        fig['layout'][f'xaxis{i+1}']['title']['text'] = None

    method_name = method.replace("_", " ").title()
    # Remove repeated x-axis titles for subplots
    title = f"{title_base} - {method_name}<br><sub>Mean Wd: {metrics[f'/error/{method}_wasserstein/mean']:0.5f}</sub>"
    fig.update_layout(
        template='plotly_dark',  # list(pio.templates):
        # = ['plotly', 'ggplot2', 'seaborn', 'simple_white', 'plotly_white', 'plotly_dark']
        title=title,
        # xaxis_title="Entropy",
        height=SUBPLOT_HEIGHT * n_channels,  # Adjust height based on the number of channels
        width=750,
        autosize=False,  # Disable autosizing for wandb
        margin=dict(l=120, r=20, t=50, b=10),  # Increase left margin for titles
        # width=700,
        # title_x=0.5,
        # title_y=0.95,
        title_xanchor='left',
        title_yanchor='top',
        hovermode="y",
        # shapes=midlines,
    )

    # Disable grids for all subplots without a loop
    fig.update_xaxes(
        showgrid=False, zeroline=False, matches='x', range=[-0.05, 1.05]
    )  # Disable x-axis grid for all subplots
    fig.update_yaxes(
        showgrid=True, zeroline=False, matches='y', showticklabels=False, range=[0, batch_size]
    )  # Disable y-axis ticks for all subplots
    wandb_image = as_wandb_image(fig, format="png", show=wandb.run.disabled)
    return wandb_image


def test_plot_multiple_entropies_on_target(
    # generated_samples: torch.Tensor,
    target_samples: torch.Tensor,
    metrics: dict,
    title_base: str = "Sample Entropy",
    group_names: Tuple[str, str] = ('predicted', 'target'),
    **kwargs
) -> go.Figure:
    """Plot the entropy target time series with multiple methods, with a horizontal subplot (range 0-1) for
    each channel. For testing most appropriate entropy method.

    Args:
        target_samples (torch.Tensor): A batch of time series. Shape (N, C, T).
        channel_names (list[str]): A list of channel names.
        group_names (tuple, optional): A tuple of group names. Defaults to ('predicted', 'target').

    C must match between groupA and groupB and channel_names.
    N is the number of samples per group.
    T is the number of time steps.    
    """
    C = get_current_config()
    channel_names = C.data.cols.x
    SUBPLOT_HEIGHT = 150

    # target_means = entropy_target.mean(axis=0)
    # target_std = entropy_target.std(axis=0) + 1e-10
    # entropy_target = (entropy_target - target_means) / target_std
    # entropy_pred = (entropy_pred - target_means) / target_std
    batch_size, n_channels, seq_length = target_samples.shape
    name_a, name_b = group_names
    color_a, color_b = 'orange', 'cyan'
    COLOR_PAL = plotly.colors.qualitative.Plotly
    assert n_channels == len(
        channel_names
    ), f"Number of channels ({n_channels}) must match the length of channel_names ({len(channel_names)})."
    fig = make_subplots(rows=n_channels, cols=1, shared_xaxes=True, subplot_titles=channel_names)
    for i, channel_name in enumerate(channel_names):
        # Add scatter trace, per group, for each channel, with N points per group
        for j, (f_name, f) in enumerate(VALID_FUNCS.items()):
            entropy_target = batch_entropy(target_samples[:, i], func=f)
            num_infs = np.sum(np.isinf(entropy_target))
            num_nan = np.sum(np.isnan(entropy_target))
            fig.add_trace(
                go.Scatter(
                    x=entropy_target,
                    y=np.linspace(0, 1, batch_size),
                    mode='markers',
                    marker=dict(color=COLOR_PAL[j]),  # Set fixed color for group A
                    name=f"{channel_name} {f_name}, infs: {num_infs}, nans: {num_nan}",
                    showlegend=True,  # Show legend only for the first trace of each group
                    legendgroup=f_name,
                    legendgrouptitle_text=f_name,  # Set legend group title only once
                ),
                row=i + 1,
                col=1
            )

    INTER_MARGIN = 0.1
    midlines = []
    for i, channel_name in enumerate(channel_names):
        # i = i + 1
        fig['layout'][f'annotations[{i}]']['x'] = -0.08  # Move subplot titles to the left
        fig['layout'][f'annotations[{i}]']['y'] = 1 - (i + .5) / n_channels  # Move subplot titles to the left
        fig['layout'][f'annotations[{i}]']['yanchor'] = 'middle'  # Move subplot titles to the left
        fig['layout'][f'annotations[{i}]']['xanchor'] = 'right'

        fig['layout'][f'yaxis{i+1}']['domain'] = [
            1 - (i + (1 - INTER_MARGIN)) / n_channels,  # Bottom
            1 - (i + INTER_MARGIN) / n_channels  # Top
        ]
        fig['layout'][f'xaxis{i+1}']['title']['text'] = None

        # Prepare midlines for each subplot
        midlines.append(
            dict(
                type="line",
                x0=0,  # Start of the line on the x-axis
                x1=1,  # End of the line on the x-axis (relative to the plot width)
                y0=0.5,  # y-coordinate of the line
                y1=0.5,  # y-coordinate of the line (same as y0 for a horizontal line)
                xref=f"x{i+1}",  # Reference the x-axis of the subplot
                yref=f"y{i+1}",  # Reference the y-axis of the subplot
                line=dict(color="white", width=1),  # Customize the line style
            )
        )

    # Remove repeated x-axis titles for subplots
    title = title_base + "Sample Entropy" + f"<br><sub>{metrics['/error/sample_entropy_wasserstein']}</sub>"
    fig.update_layout(
        template='plotly_dark',
        title=title,
        # xaxis_title="Entropy",
        height=SUBPLOT_HEIGHT * n_channels,  # Adjust height based on the number of channels
        margin=dict(l=120, r=20, t=30, b=10),  # Increase left margin for titles
        # width=700,
        title_x=0.5,
        title_y=0.95,
        title_xanchor='center',
        title_yanchor='top',
        # shapes=midlines,
    )

    # Disable grids for all subplots without a loop
    fig.update_xaxes(showgrid=False, zeroline=False, matches='x')  # Disable x-axis grid for all subplots
    fig.update_yaxes(
        showgrid=True, zeroline=False, matches='y', showticklabels=False
    )  # Disable y-axis ticks for all subplots
    if wandb.run.disabled:  # type: ignore
        fig.show()
    return fig
