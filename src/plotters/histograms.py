"""Histograms of distributions such as the peak prominence, peak width, etc."""

#%% Imports
from typing import Callable, List, Literal, Tuple
import wandb
import plotly.graph_objects as go
import plotly.express as px
from src.config import get_current_config
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#%%

import plotly.graph_objects as go
# COLOR_SCALE = plotly.colors.sequential.Plasma
COLOR_SCALE = ['#636EFA', '#00CC96', '#EF553B', "#999999"]


def plot_peak_prominences_histogram(
    peak_features: dict,
    meta: dict,
    conditioning_input: dict,
    metrics: dict,
    channel_name: str = "PD",
    measure: Literal["prominence", "width", "base"] = "prominence",
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
                    "distribution": "Predicted",
                    "mode": int(labels[shot_i][peak.X].item()),
                    "value": peak.prominences,
                    "width": peak.widths,
                    "shot_num": shot_num
                }
            )
        for peak in target_peaks[shot_i]:
            data.append(
                {
                    "distribution": "Target",
                    "mode": int(labels[shot_i][peak.X].item()),
                    "value": peak.prominences,
                    "width": peak.widths,
                    "shot_num": shot_num
                }
            )
    df = pd.DataFrame(data)
    df["mode"] = df["mode"].map({1: "L", 2: "D", 3: "H", 0: "?"})

    total_target_peaks = df.query("distribution == 'Target'").shape[0]
    total_pred_peaks = df.query("distribution == 'Predicted'").shape[0]
    visualized_shots = len(pred_peaks)
    logger.debug(f"Visualized shots: {visualized_shots}")
    logger.debug(f"Total target peaks: {total_target_peaks}, total predicted peaks: {total_pred_peaks}")

    ## Subtitle
    marginal_metric = f"/error/peak_{measure}/marginal_wasserstein/{channel_name}"
    pairwise_metric = f"/error/peak_{measure}/pairwise_wasserstein/{channel_name}"
    subtitle = (
        f"<br><sub>{visualized_shots} shots: {total_target_peaks} target peaks, "
        f"{total_pred_peaks} predicted peaks &nbsp;&nbsp;&nbsp;&nbsp;"
        f"Marginal Wd: {metrics[marginal_metric]:.4f}, "
        f"Pairwise Wd: {metrics[pairwise_metric]:.4f}"
        "</sub>"
    )

    fig = px.histogram(
        df,
        x="value",
        color="mode",
        barmode="stack",
        facet_col="distribution",
        histnorm="probability density",
        color_discrete_sequence=COLOR_SCALE,
        title=f"{title_base} Histogram of Peak <b>{measure.capitalize()}s</b> "
        f"for Channel: <b>{channel_name}</b>{subtitle}",
        labels={
            "value": measure,
            "group": "Group",
            "shot_num": "Shot Number"
        },
        hover_data=["shot_num", "mode"],
        marginal="rug",
        nbins=100,
        category_orders={
            "mode": ['L', 'D', 'H', '?'],
            "distribution": ["Target", "Predicted"],
        },
    )

    fig.update_layout(
        hovermode="x unified",
        legend_title_text="Mode",
        margin=dict(l=20, r=20, t=120, b=20),
        #     xaxis=dict(matches='x'),  # Link x-axis hovering between facets
    )
    if wandb.run.disabled:  # type: ignore
        fig.show()
    return fig
