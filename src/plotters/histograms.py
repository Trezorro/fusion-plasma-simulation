"""Histograms of distributions such as the peak prominence, peak width, etc."""

#%% Imports
from typing import Callable, List, Literal, Tuple
import plotly
import wandb
import plotly.graph_objects as go
import plotly.express as px
from src.config import get_current_config
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#%%

# COLOR_SCALE = plotly.colors.sequential.Plasma
COLOR_SCALE = ['#636EFA', '#00CC96', '#EF553B', "#999999"]


def plot_peak_prominences_histogram(
    peak_features: dict,
    meta: dict,
    conditioning_input: dict,
    metrics: dict,
    channel_name: str = "PD",
    measure: Literal["prominence", "width", "base", "count"] = "prominence",
    n=64,
    title_base="",
    **kwargs,
):
    """
    Plot a histogram of peak measures for a given channel name.

    Args:
        peak_features (dict): Dictionary containing peak features: pred_peaks and target_peaks.
        meta (dict): Metadata dictionary.
        conditioning_input (dict): Conditioning input dictionary.
        metrics (dict): Dictionary containing metrics.
        channel_name (str): Name of the channel to plot.
        measure (str): Measure to plot. Options are "prominence", "width", "base", or "count".
        n (int): Number of shots to visualize.
        title_base (str): Base title for the plot.
        **kwargs: Additional arguments for the plot.
    """
    C = get_current_config()
    channel_idx = C.data.cols.x.index(channel_name)
    history_length = C.data.history_length
    pred_peaks = peak_features["pred_peaks"]
    target_peaks = peak_features["target_peaks"]
    shot_numbers = meta["shot_number"]
    labels = conditioning_input['label'].numpy(
    )[:, history_length:]  # match the indexing of peak features (on the future window only)
    mean_label_per_shot = labels.mean(axis=1)
    n = min(n, len(shot_numbers))
    count_hist = measure == "count"

    # Prepare data for plotly express
    data = []
    for shot_i in range(n):
        shot_num = shot_numbers[shot_i].item()
        if count_hist:  # one scalar per shot
            data.append(
                {
                    "distribution": "Predicted",
                    "mode": mean_label_per_shot[shot_i],
                    "value": pred_peaks[shot_i][channel_idx].num_peaks(),
                    "width": 0,
                    "shot_num": shot_num
                }
            )
            data.append(
                {
                    "distribution": "Target",
                    "mode": mean_label_per_shot[shot_i],
                    "value": target_peaks[shot_i][channel_idx].num_peaks(),
                    "width": 0,
                    "shot_num": shot_num
                }
            )
        else:  # prominence, width, base, height, with multiple peak samples per shot
            for peak in pred_peaks[shot_i][channel_idx].iter_peaks():
                data.append(
                    {
                        "distribution": "Predicted",
                        "mode": int(labels[shot_i][peak.X].item()),
                        "value": getattr(peak, measure),
                        "shot_num": shot_num
                    }
                )
            for peak in target_peaks[shot_i][channel_idx].iter_peaks():
                data.append(
                    {
                        "distribution": "Target",
                        "mode": int(labels[shot_i][peak.X].item()),
                        "value": getattr(peak, measure),
                        "shot_num": shot_num
                    }
                )
    df = pd.DataFrame(data)
    ## Subtitle
    if count_hist:  # one scalar per shot
        total_pred_peaks = df.query("distribution == 'Predicted'")['value'].sum()
        total_target_peaks = df.query("distribution == 'Target'")['value'].sum()
        marginal_metric = f"/error/peak_count/marginal_wasserstein/{channel_name}"
        pairwise_metric = f"/error/peak_count/pairwise_mse/{channel_name}"
    else:
        df["mode"] = df["mode"].map({1: "L", 2: "D", 3: "H", 0: "?"})

        total_target_peaks = df.query("distribution == 'Target'").shape[0]
        total_pred_peaks = df.query("distribution == 'Predicted'").shape[0]

        marginal_metric = f"/error/peak_{measure}/marginal_wasserstein/{channel_name}"
        pairwise_metric = f"/error/peak_{measure}/pairwise_wasserstein/{channel_name}"

    subtitle = (
        f"<br><sub>{n} shot samples: {total_target_peaks} target peaks, "
        f"{total_pred_peaks} predicted peaks &nbsp;&nbsp;&nbsp;&nbsp;"
        f"Marginal Wd: {metrics[marginal_metric]:.4f}, "
        f"Pairwise {'MSE' if measure == 'count' else 'Wd'}: {metrics[pairwise_metric]:.4f}"
        "</sub>"
    )
    logger.debug(f"Visualized shots: {n}")
    logger.debug(f"Total target peaks: {total_target_peaks}, total predicted peaks: {total_pred_peaks}")
    heavy_render = total_pred_peaks + total_target_peaks > 1000

    fig = px.histogram(
        df,
        x="value",
        color="shot_num" if count_hist else "mode",
        barmode="stack",
        facet_col="distribution",
        histnorm="probability density",
        color_discrete_sequence=COLOR_SCALE if not count_hist else None,
        title=f"{title_base} Histogram of Peak <b>{measure.capitalize()}s</b> "
        f"for Channel: <b>{channel_name}</b>{subtitle}",
        labels={
            "value": f"<b>{measure.capitalize()}s</b>",
            "group": "Group",
            "shot_num": "Shot Number",
            "mode": "Mode",
            "distribution": "Distribution",
        },
        hover_data=["shot_num", "mode"],
        marginal="violin",
        nbins=100,
        category_orders={
            "mode": ['L', 'D', 'H', '?'],
            "distribution": ["Target", "Predicted"],
        },
    )

    fig.update_layout(
        hovermode="x",
        # legend_title_text="Mode",
        margin=dict(l=20, r=20, t=120, b=20),
        violingap=0,
        violingroupgap=0,
        violinmode='overlay'
        #     xaxis=dict(matches='x'),  # Link x-axis hovering between facets
    )
    if wandb.run.disabled:  # type: ignore
        fig.show()
    return wandb.Html(plotly.io.to_html(fig))
