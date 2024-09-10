"""Functions for evaluating and visualizing model performance."""
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from torch.utils import data
import lightning as L
import lightning.pytorch as pl

import wandb
from src.config import get_current_config


def build_output_df(shot_numbers, controls: torch.Tensor, observables: torch.Tensor, outputs: torch.Tensor):
    """Given the model outputs, build a dataframe with the predictions.

    Handles multiple shots (small batch) in one go.

    Args:
        shot_numbers (torch.Tensor): The shot numbers. (scalars)
        controls (torch.Tensor): The control variables. (batch_size, seq_length, n_controls)
        observables (torch.Tensor): The observable variables. (batch_size, seq_length, n_observables)
        outputs (torch.Tensor): The model outputs. (batch_size, seq_length, n_observables)

    """
    C = get_current_config()
    output_cols = [f"^{i}" for i in C.data.cols.x]
    seq_length = outputs.shape[1]
    # DF will have columns:
    # ShotNum, t,
    # ^FIR, ^PD, ^DML, ^POHM, ^Z_axis,
    # FIR, PD, DML, POHM, Z_axis,
    # IP, gas_fringes, NBI, ECRH, a_minor, KAPPA, DELTA
    df = pd.DataFrame(
        index=np.repeat(shot_numbers.numpy().astype(int),
                        seq_length),  # ShotNum, each repeated seq_length times
        columns=["t"] + output_cols + C.data.cols.x  # + C.data.cols.c
    )
    for shot, output, control_seq, observable_seq in zip(
            shot_numbers,
            outputs,
            controls,
            observables,
    ):
        df.loc[int(shot)] = np.concatenate(
            [
                np.arange(seq_length)[:, np.newaxis],  # t
                output.numpy(),
                observable_seq.numpy(),
                # control_seq.numpy()
            ],
            axis=1)
    return df


def log_predictions(model, data_set, n=4, title_postfix=""):  # TODO use n_samples instead of n
    C = get_current_config()
    batch = next(iter(data.DataLoader(data_set, batch_size=n, shuffle=False)))
    loss, val_train_rollout, outputs = model.validation_step(
        batch, 0).values()  # (batch_size, seq_length, target_variables)
    shot_numbers, controls, observables = batch
    pred_out = observables.clone()
    # pred_out = torch.full_like(observables, fill_value=np.nan) # use if you want to start the forecast horizon cleanly.
    pred_out[:, -C.validation_rollout:] = outputs[:, -C.validation_rollout:]
    df = build_output_df(shot_numbers, controls, observables, pred_out)
    fig = plot_sample(df,
                      title=title_postfix + f" (TRO Loss: {val_train_rollout:.5f} / Full Loss: {loss:.5f})",
                      cutoff_t=C.seq_length - C.validation_rollout)
    if wandb.run.disabled:
        fig.show()
    # table = wandb.Table(dataframe=df.reset_index(names='ShotNum'))
    return fig


def plot_sample(df: pd.DataFrame, title="", cutoff_t=50):
    """Plot a shot of the data. Df shot be a single shot."""
    df_stacked = df.set_index('t',
                              append=True).stack(future_stack=True).reset_index(name='value').rename(columns={
                                  'level_0': 'shot',
                                  'level_2': 'variable'
                              })
    df_stacked['is_prediction'] = df_stacked['variable'].str.startswith('^').map({
        True: 'Prediction',
        False: 'Actual'
    })
    df_stacked['variable'] = df_stacked['variable'].str.replace('^', '')
    fig = px.line(df_stacked,
                  x='t',
                  y='value',
                  color='shot',
                  symbol='variable',
                  line_dash='is_prediction',
                  line_shape='linear',
                  category_orders={'is_prediction': ["Actual", "Prediction"]},
                  title="Predictions " + title)
    C = get_current_config()
    fig.add_vrect(
        # type="line",
        x0=0,
        x1=cutoff_t - 0.5,
        opacity=0.2,
        line_width=0,
        layer="below",
        fillcolor="LightSalmon",
    )
    fig.update_xaxes(range=[-0.5, C.seq_length])
    return fig


class PlotPredictionsCallback(L.Callback):

    def __init__(self, num_samples=8, every_n_epochs=5):
        super().__init__()
        self.num_samples = num_samples  # Number of samples to log
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: L.LightningModule):
        # Skip for all other epochs
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Generate images
            fig = log_predictions(
                model=pl_module,
                data_set=trainer.val_dataloaders.dataset,  # type: ignore
                n=self.num_samples,
                title_postfix=f"Epoch {trainer.current_epoch}")
            wandb.log(
                {  #"predictions/val": table, 
                    "val/prediction_plot": fig
                },
                commit=False)
