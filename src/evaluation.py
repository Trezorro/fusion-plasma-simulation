"""Functions for evaluating and visualizing model performance."""
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from torch.utils import data
import lightning as L
import lightning.pytorch as pl

import wandb
from src.fourier import spectogram_plot, FourierMSLE
from src.config import get_current_config


def build_output_df(shot_numbers, controls: torch.Tensor, observables: torch.Tensor, outputs: torch.Tensor):
    """Given the model outputs, build a dataframe with the predictions.

    Handles multiple shots (small batch) in one go.

    Args:
        shot_numbers (torch.Tensor): The shot numbers. (scalars)
        controls (torch.Tensor): The control variables. (batch_size, seq_length, n_controls)
        observables (torch.Tensor): The observable variables. (batch_size, seq_length, n_observables)
        outputs (torch.Tensor): The model outputs. (batch_size, seq_length, n_observables)
    Returns (pd.DataFrame): The dataframe with the predictions.
        Index: ShotNum
        Columns: t,
            Target variables predictions (^FIR, ^PD, ^DML, ^POHM, ^Z_axis),
            ground truth target variables (FIR, PD, DML, POHM, Z_axis)
            Removed: control variables (IP, gas_fringes, NBI, ECRH, a_minor, KAPPA, DELTA)
    """
    C = get_current_config()
    output_cols = [f"^{i}" for i in C.data.cols.x]
    seq_length = outputs.shape[1]

    df = pd.DataFrame(
        index=np.repeat(shot_numbers.numpy().astype(int),
                        seq_length),  # ShotNum, each repeated seq_length times
        columns=["t"] + output_cols + C.data.cols.x,  # + C.data.cols.c,
        dtype=np.float32
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
            axis=1
        )
    return df


def plot_shot_batch(df: pd.DataFrame, title="", cutoff_t=50):
    """Plot a batch of shots into one plot, with predictions on the right in dotted lines."""
    df_stacked = df.set_index('t', append=True).stack(future_stack=True).reset_index(name='value').rename(
        columns={
            'level_0': 'shot',
            'level_2': 'variable'
        }
    )
    df_stacked['is_prediction'] = df_stacked['variable'].str.startswith('^').map(
        {
            True: 'Prediction',
            False: 'Actual'
        }
    )
    df_stacked['variable'] = df_stacked['variable'].str.replace('^', '')
    fig = px.line(
        df_stacked,
        x='t',
        y='value',
        color='shot',
        symbol='variable',
        line_dash='is_prediction',
        line_shape='linear',
        category_orders={'is_prediction': ["Actual", "Prediction"]},
        title=f"Signal Plot: {title}"
    )
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


def get_and_plot_predictions(model, data_set, n=4, title_base=""):
    """Get and plot predictions for a batch of shots. 

    Plots n shots in the signal plot, and one signal from one shot in the spectogram plot.
    """
    C = get_current_config()
    fourier_loss = FourierMSLE().to(model.device)
    model.eval()
    with torch.inference_mode():
        shot_numbers, controls, observables = next(
            iter(data.DataLoader(data_set, batch_size=n, shuffle=False))
        )
        x_out_pred = model.prediction_step(
            (shot_numbers, controls.to(model.device), observables.to(model.device)), 0
        )
        x_out = observables[:, -model.val_rollout:].to(model.device)
        loss = model.loss(x_out_pred, x_out)
        fourier_loss_batch = fourier_loss(x_out_pred, x_out)
        val_train_rollout = model.loss(x_out_pred[:, :model.train_rollout], x_out[:, :model.train_rollout])
        pred_full_line = observables.clone()
        # pred_out = torch.full_like(observables, fill_value=np.nan) # use if you want to start the forecast horizon cleanly.
        pred_full_line[:, -C.validation_rollout:] = x_out_pred[:, -C.validation_rollout:].cpu()
        df = build_output_df(shot_numbers, controls, observables, pred_full_line)
        title = title_base + f" (Fourier Loss: {fourier_loss_batch:.5f} / Full Loss: {loss:.5f})"
        fig_shots = plot_shot_batch(df, title=title, cutoff_t=C.seq_length - C.validation_rollout)
        # Plot spectograms of each prediction pair in each shot
        # start with plotting: 1. the ground truth in the forecast horizon.
        # Get the signal for the first shot, below the cutoff_t, for only the target variable.
        # TODO: Test what happens if we have multiple variables.
        first_shot = df.loc[df.index[0]]
        forecast_only = first_shot.query("t >= @C.seq_length - @C.validation_rollout")
        signal_pred = forecast_only[[f"^{i}" for i in C.data.cols.x]].values
        signal_true = forecast_only[C.data.cols.x].values
        fourier_loss_forecast = FourierMSLE()(torch.tensor(signal_pred), torch.tensor(signal_true))
        full_true_signal = first_shot[C.data.cols.x].values
        fig_spec = spectogram_plot(
            full_true_signal,
            title=title_base + f" shot #{df.index[0]} (Fourier Loss forecast: {fourier_loss_forecast:.6f})",
            hop_length=10,
            win_length=50
        )
        if wandb.run.disabled:
            fig_shots.show()
            fig_spec.show()
    model.train()
    return fig_shots, fig_spec


class PlotPredictionsCallback(L.Callback):

    def __init__(self, num_samples=8, every_n_epochs=5, train_every_n_epochs=20):
        super().__init__()
        self.num_samples = num_samples  # Number of samples to log
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs
        self.train_every_n_epochs = train_every_n_epochs

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: L.LightningModule):
        # Skip for all other epochs
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Generate images
            fig_shots, fig_spec = get_and_plot_predictions(
                model=pl_module,
                data_set=trainer.val_dataloaders.dataset,  # type: ignore
                n=self.num_samples,
                title_base=f"{wandb.run.name} | Epoch {trainer.current_epoch}"
            )
            wandb.log({"val/prediction_plot": fig_shots, "val/spectogram_plot": fig_spec}, commit=False)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: L.LightningModule):
        if self.train_every_n_epochs:
            # Skip for all other epochs
            if trainer.current_epoch % self.train_every_n_epochs == 0:
                # Generate images
                fig_shots, fig_spec = get_and_plot_predictions(
                    model=pl_module,
                    data_set=trainer.train_dataloader.dataset,  # type: ignore
                    n=self.num_samples,
                    title_base=f"TRAINDATA | {wandb.run.name} | Epoch {trainer.current_epoch}"
                )
                wandb.log(
                    {
                        "train/prediction_plot": fig_shots,
                        "train/spectogram_plot": fig_spec
                    }, commit=False
                )
