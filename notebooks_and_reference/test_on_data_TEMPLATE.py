"""Minimal template: load a FusionShotDataModule and plot one shot's channels with plotly. Prototype/legacy, unmaintained.

Inputs/Outputs: reads ./data/2024_05_01-NaNsFiltered.parquet via FusionShotDataModule; scratch plot only, no writes.
Handy: plot_shot (per-channel plotly line plot, dark theme) is a minimal reusable starting point for quick data-loading smoke tests.
"""
# %% [markdown]
# # Test entropy
#
# %%
import torch
import logging
from src.data_loaders import FusionShotDataModule, FusionShotDataset
from src.config import load_config_from_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

C = load_config_from_file('fm_toy', as_omega=True)

ds = FusionShotDataModule( # Todo: rewrite to adapt to Data Module
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
            go.Scatter(x=shot_df.index, y=shot_df[col], mode='lines', name=col, line_color=COLOR_SCALE[i % 10])
        )
    fig.show()


plot_shot(ds.data, 0)

# %%

# %%
