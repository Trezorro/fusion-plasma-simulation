"""Histograms of distributions such as the peak prominence, peak width, etc."""

#%% Imports
from typing import Callable, List, Literal, Tuple
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from notebooks_and_reference.test_entropy import COLOR_SCALE
from src.config import get_current_config
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#%%

import plotly.graph_objects as go
COLOR_SCALE = plotly.colors.sequential.Plasma

def plot_peak_prominences_histogram(
    peak_features: dict,
    channel_name: str,
    meta: dict,
    conditioning_input: dict,
    n=64,
    title_base="",
    **kwargs,
) -> go.Figure:
    """
    Plot a histogram of peak prominences for a given channel name.

    Args:
        peak_features (dict): Dictionary containing predicted and target peaks.
        channel_name (str): The channel name to plot.
        n (int): Number of samples to plot.
        title_base (str): Base title for the plot.
        **kwargs: Additional arguments for the plot.
    """
    C = get_current_config()
    channel_idx = C.data.cols.x.index(channel_name)
    history_length = C.data.history_length
    pred_peaks = peak_features["pred_peaks"]
    target_peaks = peak_features["target_peaks"]
    shot_numbers = meta["shot_number"]
    labels = conditioning_input['label'].numpy()[:, history_length:] # shape: (n_samples, time steps)
    shot_label_means = labels.mean(axis=1)
    if len(pred_peaks) > n:
        pred_peaks = pred_peaks[:n]
    if len(target_peaks) > n:
        target_peaks = target_peaks[:n]

    pred_prominences = [sample.prominences for sample in pred_peaks[channel_idx]]
    target_prominences = [sample.prominences for sample in target_peaks[channel_idx]]

    # Prepare data for plotly express
    data = []
    for shot_num, pred_i, target_i_data, label_mean_i in zip(shot_numbers, pred_prominences, target_prominences, shot_label_means):
        for prominence in pred_i:
            data.append({
                "group": "Predicted",
                "label": label_mean_i.item(),
                "prominence_value": prominence,
                "shot_num": shot_num.item()
            })
        for prominence in target_i_data:
            data.append({
                "group": "Target",
                "label": label_mean_i.item(),
                "prominence_value": prominence,
                "shot_num": shot_num.item()
            })

    df = pd.DataFrame(data)
    # plot px.histogram
    fig = px.histogram(
        df,
        x="prominence_value",
        color="shot_num",
        barmode="stack",
        facet_col="group",
        facet_col_wrap=2,
        histnorm="probability density",
        color_discrete_sequence=COLOR_SCALE,
        title=f"{title_base} Histogram of Peak Prominences for Channel: {channel_name}",
        labels={"prominence_value": "Prominence", "group": "Group", "shot_num": "Shot Number"},
        hover_data=["shot_num", "label"],
        marginal="rug",
        # marginal_kws=dict(binsize=0.1),
    )
    fig.show()

    def add_hist_trace(fig, shot_num, data, label_mean_i, group_name: Literal['Target', 'Predicted']):
        fig.add_trace(
            go.Histogram(
                x=data,
                name=f"{group_name} for shot {shot_num}",
                nbinsx=100,
                opacity=0.75,
                yaxis="y1" if group_name == "Target" else "y2",
                histnorm='density',
                # marker_color='red',
                marker=dict(
                    colorscale="Plasma" if group_name == "Target" else "Turbo",
                    color=(label_mean_i.item(),) * 50,
                    cmin=1.0,
                    cmid=2.0,
                    cmax=3.0,
                    # colorbar=dict(title="Label Mean"),
                    # coloraxis="coloraxis",
                ),
                text=shot_num.item(),
                customdata=(label_mean_i.item(),) * 50,
                legendgroup=f'{group_name}',
                legendgrouptitle_text=f"{group_name}",
                hovertemplate=group_name + " prominence: %{x}<br>Count: %{y}<br>Label: %{customdata}<extra></extra>",
                hoverinfo="x+y",
                showlegend=True,
                bingroup=1
            )
        )
    fig = go.Figure()
    # Add predicted prominences
    for shot_num, pred_i, target_i_data, label_mean_i in zip(shot_numbers, pred_prominences, target_prominences, shot_label_means):
        add_hist_trace(fig, shot_num, pred_i, label_mean_i, "Predicted")
        add_hist_trace(fig, shot_num, target_i_data, label_mean_i, "Target")

    fig.update_layout(
        title=f"{title_base} Histogram of Peak Prominences for Channel: {channel_name}",
        xaxis_title="Prominence",
        yaxis_title="Probability",
        yaxis=dict(title="Target Probability"),
        yaxis2=dict(title="Predicted Probability", overlaying="y", side="right"),
        barmode="stack",
        legend_title="Distributions",
    )
    fig.show()


