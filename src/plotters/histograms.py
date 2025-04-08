"""Histograms of distributions such as the peak prominence, peak width, etc."""

#%% Imports
from typing import Callable, List, Literal, Tuple
import wandb
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
# COLOR_SCALE = plotly.colors.sequential.Plasma
COLOR_SCALE = ['#636EFA', '#00CC96', '#EF553B']


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
    pred_peaks = peak_features["pred_peaks"][channel_idx]
    target_peaks = peak_features["target_peaks"][channel_idx]
    shot_numbers = meta["shot_number"]
    labels = conditioning_input['label'].numpy()[:, history_length:]  # shape: (n_samples, time steps)
    shot_label_means = labels.mean(axis=1)
    if len(pred_peaks) > n:
        pred_peaks = pred_peaks[:n]
    if len(target_peaks) > n:
        target_peaks = target_peaks[:n]

    # Prepare data for plotly express
    data = []
    for shot_i in range(len(shot_numbers)):
        shot_num = shot_numbers[shot_i].item()
        for peak in pred_peaks[shot_i]:
            data.append(
                {
                    "group": "Predicted",
                    "mode": int(labels[shot_i][peak.X].item()),
                    "prominence": peak.prominences,
                    "shot_num": shot_num
                }
            )
        for peak in target_peaks[shot_i]:
            data.append(
                {
                    "group": "Target",
                    "mode": int(labels[shot_i][peak.X].item()),
                    "prominence": peak.prominences,
                    "shot_num": shot_num
                }
            )
    df = pd.DataFrame(data)
    # map modes to L, D, H
    df["mode"] = df["mode"].map({1: "L", 2: "D", 3: "H", 0: "?"})
    # plot px.histogram
    fig = px.histogram(
        df,
        x="prominence",
        color="mode",
        barmode="stack",
        facet_col="group",
        # facet_col_wrap=2,
        histnorm="probability density",
        color_discrete_sequence=COLOR_SCALE,
        title=f"{title_base} Histogram of Peak Prominences for Channel: {channel_name}",
        labels={
            "prominence": "Prominence",
            "group": "Group",
            "shot_num": "Shot Number"
        },
        hover_data=["shot_num", "mode"],
        marginal="rug",
        nbins=200,  # Make bins smaller by increasing the number of bins
        category_orders={"mode": ['L', 'D', 'H', '?']},  # Add order to mode
    )
    # fig.update_layout(
    #     hovermode="x unified",
    #     xaxis=dict(matches='x'),  # Link x-axis hovering between facets
    # )
    if wandb.run.disabled:  # type: ignore
        fig.show()
    return fig
