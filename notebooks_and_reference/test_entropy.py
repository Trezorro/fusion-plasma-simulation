"""Compares sample entropy distributions between two data batches (KS test, Wasserstein, Jensen-Shannon) via src.data_loaders. Prototype/legacy, unmaintained.

Inputs/Outputs: reads ./data/2024_05_01-NaNsFiltered.parquet through FusionShotDataset; scratch plots only, no writes.
Handy: get_sample_entropy/normalized_entropies/ks_test_sample_entropy is a reusable per-channel entropy comparison worth extracting into src/metrics/.
"""
# %% [markdown]
# # Test entropy
#
# %%
from re import A
import torch
import logging
from src.data_loaders import FusionShotDataset
from src.config import load_config_from_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

C = load_config_from_file('fm_toy', as_omega=True)

ds = FusionShotDataset( #Yodo: rewrite to adapt to Data Module
    './data/',
    '2024_05_01-NaNsFiltered.parquet',
    cols=C.data.cols,
    seq_length=512,
    history_length=512,
    crop_margin=1000,
)

# %%
ds.data

# %%
import plotly.graph_objects as go
from plotly import colors as plt_colors
from plotly.subplots import make_subplots as plotly_make_subplots

COLOR_SCALE = plt_colors.qualitative.Plotly


def plot_shot(df, shot: int):
    shot_ids = df['ShotNum'].unique()
    shot_id = shot_ids[shot]
    shot_df = df[df['ShotNum'] == shot_id]
    fig = go.Figure()
    fig.update_layout(
        title=f"Plot {shot_id}",
        template='plotly_dark',
    )
    for i, col in enumerate(C.data.cols.x):
        fig.add_trace(
            go.Scatter(
                x=shot_df.index, y=shot_df[col], mode='lines', name=col, line_color=COLOR_SCALE[i % 10]
            )
        )
    fig.show()


plot_shot(ds.data, 0)

# %% get 2 batches from the data set
batch_size = 64
train_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
for i, batch in enumerate(train_loader):
    if i == 0:
        meta, conditioning_input, targetA = batch
    elif i == 1:
        meta, conditioning_input, targetB = batch
    else:
        break

targetA.shape, targetB.shape

#%% Imports
import numpy as np
import torch
import antropy as ant
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple


#%% Definitions
def get_sample_entropy(batched_time_series: torch.Tensor):
    # https://en.wikipedia.org/wiki/Sample_entropy
    batched_time_series_np = batched_time_series.numpy()
    return np.apply_along_axis(ant.sample_entropy, -1, batched_time_series_np)


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


def first_difference(batched_time_series: torch.Tensor):
    return batched_time_series[:, :, 1:] - batched_time_series[:, :, :-1]


groupA = first_difference(targetA)
groupB = first_difference(targetB)
channel_names = C.data.cols.x
group_names = ('A_pred', 'B_target')

#%% Kolmogorov-Smirnov test for sample entropy
from scipy.stats import ks_2samp, shapiro, wasserstein_distance
from scipy.spatial.distance import jensenshannon


def ks_test_sample_entropy(pred_batch: torch.Tensor, target_batch: torch.Tensor):
    """Perform the Kolmogorov-Smirnov test for sample entropy per channel."""
    entropy_target, entropy_pred = normalized_entropies(pred_batch, target_batch)
    res = {}
    shapiro_res = {}
    for i, channel_name in enumerate(channel_names):
        res[channel_name] = ks_2samp(entropy_target[:, i], entropy_pred[:, i])
        shapiro_res[channel_name] = (shapiro(entropy_target[:, i]), shapiro(entropy_pred[:, i]))
        # interpret shapiro results
        print(f"{channel_name}: KS test for sample entropy:\n{res[channel_name]}")
        print(
            f"{channel_name}: Shapiro-Wilk test for normality:\nA={shapiro_res[channel_name][0]},\n B={shapiro_res[channel_name][1]}"
        )
        wasser_res = wasserstein_distance(entropy_target[:, i], entropy_pred[:, i])
        prob_hist_target = np.histogram(entropy_target[:, i], bins=100, density=True)[0]
        prob_hist_pred = np.histogram(entropy_pred[:, i], bins=100, density=True)[0]
        jensen_res = jensenshannon(prob_hist_target, prob_hist_pred)
        print(f"{channel_name}: Wasserstein distance: {wasser_res:.5f}")
        print(f"{channel_name}: Jensen-Shannon distance: {jensen_res:.5f}\n")
    return res


def normalized_entropies(pred_batch, target_batch):
    entropy_target = get_sample_entropy(target_batch)
    entropy_pred = get_sample_entropy(pred_batch)
    target_means = entropy_target.mean(axis=0)
    target_std = entropy_target.std(axis=0) + 1e-10
    entropy_target = (entropy_target - target_means) / target_std
    entropy_pred = (entropy_pred - target_means) / target_std
    return entropy_target, entropy_pred


ks_test_sample_entropy(groupA, groupB)

# %%
SUBPLOT_HEIGHT = 70
entropy_pred = get_sample_entropy(targetA)  # + 1 to test what happens when predicted is much higher
entropy_target = get_sample_entropy(targetB)
# target_means = entropy_target.mean(axis=0)
# target_std = entropy_target.std(axis=0) + 1e-10
# entropy_target = (entropy_target - target_means) / target_std
# entropy_pred = (entropy_pred - target_means) / target_std
distance = np.abs(entropy_pred - entropy_target)
batch_size, n_channels, seq_length = groupA.shape
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

# %%

# %%
print(fig.layout.annotations)
# %%

# %%
