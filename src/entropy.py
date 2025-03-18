#%% Imports
from typing import List, Tuple
import numpy as np
import torch
import antropy as ant
import wandb
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import get_current_config
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#%% Definitions
def get_sample_entropy(batched_time_series: torch.Tensor) -> np.ndarray:
    """Calculate the sample entropy for a batch of time series.
    
    See: https://en.wikipedia.org/wiki/Sample_entropy

    An input of shape (N, C, T) will return an output of shape (N, C).
    """
    batched_time_series_np = batched_time_series.numpy()
    return np.apply_along_axis(ant.sample_entropy, -1, batched_time_series_np)


def get_normalized_entropies(pred_batch: torch.Tensor, target_batch: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    entropy_target = get_sample_entropy(target_batch)
    entropy_pred = get_sample_entropy(pred_batch)
    target_means = entropy_target.mean(axis=0)
    target_std = entropy_target.std(axis=0) + 1e-10
    entropy_target = (entropy_target - target_means) / target_std
    entropy_pred = (entropy_pred - target_means) / target_std
    return entropy_target, entropy_pred


def test_sample_entropy(shape=(10, 5, 100)):
    """Test the sample entropy calculation for a given shape."""
    x = torch.randn(*shape)
    start_time = time.time()
    res = get_sample_entropy(x)
    end_time = time.time()
    print(
        f"Sample entropy calculation took {end_time - start_time:4.5f} seconds for shape {shape} ({shape[0] * shape[1]} samples)."
    )
    print(f"Result shape: {res.shape}")
    print(res)
    pass


# if __name__ == "__main__":
#     test_sample_entropy((64, 5, 256))
#     test_sample_entropy((64, 5, 512))
#     test_sample_entropy((128, 5, 256))
#     test_sample_entropy((128, 5, 512))
#     test_sample_entropy((128, 5, 1024))
#     pass
#%%
def plot_entropy(
    generated_samples: torch.Tensor,
    target_samples: torch.Tensor,
    metrics: dict,
    title_base: str = "Sample Entropy",
    group_names: Tuple[str, str] = ('predicted', 'target'),
    **kwargs
) -> go.Figure:
    """Plot the entropy of two groups of time series, with a horizontal subplot (range 0-1) for each channel.

    Args:
        groupA (torch.Tensor): A batch of time series (predicted). Shape (N, C, T).
        groupB (torch.Tensor): A batch of time series (target). Shape (N, C, T).
        channel_names (list[str]): A list of channel names.
        group_names (tuple, optional): A tuple of group names. Defaults to ('predicted', 'target').

    C must match between groupA and groupB and channel_names.
    N is the number of samples per group.
    T is the number of time steps.    
    """
    C = get_current_config()
    channel_names = C.data.cols.x
    SUBPLOT_HEIGHT = 70

    entropy_pred = get_sample_entropy(generated_samples)  # + 1 to test what happens when predicted is much higher
    entropy_target = get_sample_entropy(target_samples)
    # target_means = entropy_target.mean(axis=0)
    # target_std = entropy_target.std(axis=0) + 1e-10
    # entropy_target = (entropy_target - target_means) / target_std
    # entropy_pred = (entropy_pred - target_means) / target_std
    distance = np.abs(entropy_pred - entropy_target)
    batch_size, n_channels, seq_length = generated_samples.shape
    name_a, name_b = group_names
    color_a, color_b = 'orange', 'cyan'
    assert n_channels == len(
        channel_names
    ), f"Number of channels ({n_channels}) must match the length of channel_names ({len(channel_names)})."
    fig = make_subplots(rows=n_channels, cols=1, shared_xaxes=True, subplot_titles=channel_names)
    for i, channel_name in enumerate(channel_names):
        # Add scatter trace, per group, for each channel, with N points per group
        fig.add_trace(
            go.Scatter(
                x=entropy_pred[:, i],
                y=np.linspace(0, 1, batch_size),
                mode='markers',
                xaxis="x",
                name=channel_name,
                marker=dict(color=color_a),  # Set fixed color for group A
                showlegend=(i == 0),  # Show legend only for the first trace of each group
                legendgroup=name_a,
                legendgrouptitle_text=name_a,
            ),
            row=i + 1,
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=entropy_target[:, i],
                y=np.linspace(0, 1, batch_size),
                mode='markers',
                marker=dict(color=color_b),  # Set fixed color for group A
                name=channel_name,
                showlegend=(i == 0),  # Show legend only for the first trace of each group
                legendgroup=name_b,
                legendgrouptitle_text=name_b if i == 0 else None,  # Set legend group title only once
            ),
            row=i + 1,
            col=1
        )
        # Add tiny error lines between xA and xB for the same sample index
        for j in range(batch_size):
            sample_y_positions = np.linspace(0, 1, batch_size)
            fig.add_trace(
                go.Scatter(
                    x=[entropy_pred[j, i], entropy_target[j, i]],
                    y=[sample_y_positions[j], sample_y_positions[j]],
                    mode='lines',
                    line=dict(color="gray", width=1),
                    opacity=0.5,
                    name="Difference Line" if (i == 0 and j == 0) else None,  # Add legend entry only once
                    showlegend=(i == 0 and j == 0),  # Show legend only for the first line
                    legendgroup="Difference",  # Group all lines under the same legend group
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
    fig.update_layout(
        template='plotly_dark',
        title="Sample Entropy",
        xaxis_title="Entropy",
        height=SUBPLOT_HEIGHT * n_channels,  # Adjust height based on the number of channels
        margin=dict(l=120, r=20, t=30, b=10),  # Increase left margin for titles
        width=700,
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

