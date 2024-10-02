"""Functions for evaluating and visualizing model performance."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch.utils import data
import lightning as L
import lightning.pytorch as pl
import torchaudio.transforms as audio_transforms

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
    Returns (pd.DataFrame): The dataframe with the predictions.
        Index: ShotNum
        Columns: t, Target variables predictions, ground truth target variables
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
            axis=1
        )
    return df


def plot_sample(df: pd.DataFrame, title="", cutoff_t=50):
    """Plot a shot of the data. Df shot be a single shot."""
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
        title=f"{wandb.run.name} | Predictions {title}"
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


def get_sine_wave(frequency, duration, sample_rate=20000):
    """Generate a sine wave with the given frequency and duration."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    x = np.sin(2 * np.pi * frequency * t)
    return x


def spectogram_plot(df: pd.DataFrame, title="", cutoff_t=50):
    """Plot the spectogram of the data. Df should be a single shot."""
    C = get_current_config()
    filtered_df = df[df['t'] <= 190]
    hop_length = 10
    win_length = 50
    to_spectrogram = audio_transforms.Spectrogram(
        n_fft=190, win_length=win_length, hop_length=hop_length, power=1, pad=0
    )
    sine = get_sine_wave(500, 0.0191, sample_rate=10000)
    sine += get_sine_wave(1000, 0.0191, sample_rate=10000)
    sine += get_sine_wave(4000, 0.0191, sample_rate=10000)
    # sine += get_sine_wave(4999, 1, sample_rate=10000)
    signal = sine
    signal = filtered_df[C.data.cols.x].values.astype('float64').squeeze()
    # spectogram = to_spectrogram(torch.tensor(sine).float()).numpy()
    spectogram = to_spectrogram(torch.tensor(signal)).numpy()
    freq_domain = torch.fft.fft(torch.tensor(signal)).numpy()
    amplitudes = np.abs(freq_domain)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Heatmap(
            z=np.log(spectogram),
            dx=hop_length,
            colorscale='Viridis',
            colorbar=dict(title='Log Value'),
            name=f'STFT spectrogram (Win: {win_length}, Hop: {hop_length})',
            showlegend=True
        ),
        secondary_y=False
    )

    fig.add_trace(
        px.line(
            x=np.arange(len(amplitudes)),  # t in sync with the hop windows
            y=amplitudes,
            labels={
                'x': 'Frequency',
                'y': 'Amplitude'
            },
            line_shape='linear',
            color_discrete_sequence=["rgb(255, 10, 10)"]
        ).data[0].update(name='Frequency Spectrum', showlegend=True),
        secondary_y=False,
    )

    fig.add_trace(
        px.line(
            x=np.arange(len(signal)),  # t in sync with the hop windows
            y=signal,
            labels={
                'x': 'Time',
                'y': 'Value'
            },
            line_shape='linear'
        ).data[0].update(name='Signal', showlegend=True),
        secondary_y=True
    )

    fig.update_layout(
        title=title,
        legend=dict(
            x=0.01,
            y=0.99,
            traceorder='normal',
            font=dict(family='sans-serif', size=12, color='black'),
            bgcolor='LightSteelBlue',
            bordercolor='Black',
            borderwidth=2
        )
    )
    fig.show()
    return fig


def get_and_plot_predictions(model, data_set, n=4, title_postfix=""):
    C = get_current_config()
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
        val_train_rollout = model.loss(x_out_pred[:, :model.train_rollout], x_out[:, :model.train_rollout])
        pred_out = observables.clone()
        # pred_out = torch.full_like(observables, fill_value=np.nan) # use if you want to start the forecast horizon cleanly.
        pred_out[:, -C.validation_rollout:] = x_out_pred[:, -C.validation_rollout:].cpu()
        df = build_output_df(shot_numbers, controls, observables, pred_out)
        fig = plot_sample(
            df,
            title=title_postfix + f" (TRO Loss: {val_train_rollout:.5f} / Full Loss: {loss:.5f})",
            cutoff_t=C.seq_length - C.validation_rollout
        )
    model.train()
    if wandb.run.disabled:
        fig.show()
    # Plot spectograms of each prediction pair in each shot
    # start with plotting: 1. the ground truth in the forecast horizon.
    fig = spectogram_plot(df.loc[df.index[0]], title_postfix, cutoff_t=C.seq_length - C.validation_rollout)
    # table = wandb.Table(dataframe=df.reset_index(names='ShotNum'))
    return fig


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
            fig = get_and_plot_predictions(
                model=pl_module,
                data_set=trainer.val_dataloaders.dataset,  # type: ignore
                n=self.num_samples,
                title_postfix=f"Epoch {trainer.current_epoch}"
            )
            wandb.log(
                {  #"predictions/val": table, 
                    "val/prediction_plot": fig
                },
                commit=False
            )

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: L.LightningModule):
        if self.train_every_n_epochs:
            # Skip for all other epochs
            if trainer.current_epoch % self.train_every_n_epochs == 0:
                # Generate images
                fig = get_and_plot_predictions(
                    model=pl_module,
                    data_set=trainer.train_dataloader.dataset,  # type: ignore
                    n=self.num_samples,
                    title_postfix=f"TRAINDATA - Epoch {trainer.current_epoch}"
                )
                wandb.log(
                    {  #"predictions/val": table, 
                        "train/prediction_plot": fig
                    },
                    commit=False
                )
