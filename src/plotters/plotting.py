"""Plotting functions for old Unet window prediction model and frequency domain Unet model.

Not updated yet to work with the new evaluation callback pipeline."""
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import torch

from src.config import get_current_config


def build_plotting_df_time(
    shot_numbers,
    controls: torch.Tensor,
    observables: torch.Tensor,
    model_predictions: torch.Tensor,
    time_last=False,
    isolated_prediction_line=False
):
    """Given the model outputs, build a dataframe with the predictions.

    Handles multiple shots (small batch) in one go.

    Args:
        shot_numbers (torch.Tensor): The shot numbers. (scalars)
        controls (torch.Tensor): The control variables. (batch_size, seq_length, n_controls)
        observables (torch.Tensor): The observable variables. (batch_size, seq_length, n_observables)
        model_predictions (torch.Tensor): The model outputs. (batch_size, seq_length, n_observables)
        isolated_prediction_line (bool): If True, the prediction line will not connect to the last observable
            value.
    Returns (pd.DataFrame): The dataframe with the predictions.
        Index: ShotNum
        Columns: t,
            Target variables predictions (^FIR, ^PD, ^DML, ^POHM, ^Z_axis),
            ground truth target variables (FIR, PD, DML, POHM, Z_axis)
            Removed: control variables (IP, gas_fringes, NBI, ECRH, a_minor, KAPPA, DELTA)
    """
    C = get_current_config()
    output_cols = [f"^{i}" for i in C.data.cols.x]
    time_dim = 2 if time_last else 1
    seq_length = observables.shape[time_dim]
    if isolated_prediction_line:
        prediction_traces = np.full_like(
            observables, fill_value=np.nan
        )  # use if you want to start the forecast horizon cleanly.
    else:
        prediction_traces = observables.clone().cpu().numpy()
    if time_last:
        time_seq = np.arange(seq_length)[np.newaxis, :]
        prediction_traces[:, :, -model_predictions.size(time_dim):] = model_predictions.cpu().numpy()
    else:
        prediction_traces[:, -model_predictions.size(time_dim):] = model_predictions.cpu().numpy()
        time_seq = np.arange(seq_length)[:, np.newaxis]

    df = pd.DataFrame(
        index=np.repeat(shot_numbers.cpu().numpy(), seq_length),  # ShotNum, each repeated seq_length times
        columns=["t"] + output_cols + C.data.cols.x,  # + C.data.cols.c,
        dtype=np.float32
    )
    for shot, output, control_seq, observable_seq in zip(
        shot_numbers,
        prediction_traces,
        controls,
        observables,
    ):
        df.loc[(shot.item())] = np.concatenate(
            [
                time_seq.copy(),  # t
                output,
                observable_seq.cpu().numpy(),
                # control_seq.numpy()
            ],
            axis=0 if time_last else 1  # we concatenate on the variables axis
        ).T  # TODO: make dependent on time_last

    df_stacked = df.reset_index(names='shot').melt(id_vars=['shot', 't'])
    df_stacked['is_prediction'] = df_stacked['variable'].str.startswith('^').map(
        {
            True: 'Predicted',
            False: 'Target'
        }
    )
    df_stacked['variable'] = df_stacked['variable'].str.replace('^', '')
    return df_stacked


def build_plotting_df_freq(x_pred_freq, x_target_freq, shot_numbers):
    # input (batch, variables (7), frequency bins)
    n_shots, n_vars, n_freqs = x_pred_freq.size()
    C = get_current_config()
    window_length = C.model.params.forecast_window
    sample_spacing = 1. / C.data.sample_rate
    # Transform the 3D array into a stacked dataframe in a vectorized manner
    shot_numbers_repeated = np.repeat(shot_numbers.cpu().numpy(), n_vars * n_freqs)
    variable_repeated = np.tile(np.repeat(C.data.cols.x, n_freqs), n_shots)
    frequency_bins = np.tile(np.fft.rfftfreq(window_length, d=sample_spacing), n_shots * n_vars)
    predicted_values = x_pred_freq.abs().cpu().numpy().flatten()
    target_values = x_target_freq.abs().cpu().numpy().flatten()

    df = pd.DataFrame(
        {
            'shot': shot_numbers_repeated,
            'variable': variable_repeated,
            'frequency_bin': frequency_bins,
            'Predicted': predicted_values,
            'Target': target_values
        }
    )
    df_stacked = df.melt(
        value_vars=['Predicted', 'Target'],
        var_name='is_prediction',
        value_name='amplitude',
        id_vars=['shot', 'variable', 'frequency_bin'],
    )
    return df_stacked


def plot_shot_batch(df: pd.DataFrame, title="", cutoff_t=50):
    """Plot a batch of shots into one plot, with predictions on the right in dotted lines."""

    fig = px.line(
        df,
        x='t',
        y='value',
        color='shot',
        symbol='variable',
        line_dash='is_prediction',
        line_shape='linear',
        category_orders={'is_prediction': ["Actual", "Predicted"]},
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
    fig.update_xaxes(range=[-0.5, C.data.seq_length])
    return fig


def format_losses(losses: dict):
    """Format the losses dictionary into a string."""
    if len(losses) < 5:
        return " / ".join([f"{k}: {v:.5f}" for k, v in losses.items()])
    else:
        first_line = max(len(losses) // 2, 4)
        str_list = [f"{k}: {v:.5f}" for k, v in losses.items()]
        return " / ".join(str_list[:first_line]) + "<br>" + " / ".join(str_list[first_line:])


def plot_signal_and_spectrum(df_stacked_time, df_freq, title, cutoff_t, subtitle=""):
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("Time-Domain Signal", "Frequency Spectrum"), vertical_spacing=0.1
    )

    # Time-domain signal plot
    time_fig = px.line(
        df_stacked_time,
        x='t',
        y='value',
        color='shot',
        symbol='variable',
        line_dash='is_prediction',
        line_shape='linear',
        category_orders={'is_prediction': ["Target", "Predicted"]},
        title=f"Signal Plot: {title}"
    )
    for trace in time_fig.data:
        shot, is_predicted, variable = trace.name.split(', ')
        trace.name = f"{is_predicted} {variable} time domain"
        trace.legendgroup = shot + is_predicted + variable
        trace.legendgrouptitle = {'text': f"Shot {shot}: {is_predicted} {variable} signal"}
        fig.add_trace(trace, row=1, col=1)

    # Frequency spectrum plot
    spectrum_fig = px.line(
        df_freq,
        x='frequency_bin',
        y='amplitude',
        color='shot',
        line_dash='is_prediction',
        category_orders={'is_prediction': ["Target", "Predicted"]},
        symbol='variable',
        line_shape='linear',
        markers=True,
        title="Frequency Spectrum (in forecast horizon)"
    )
    for trace in spectrum_fig.data:
        shot, is_predicted, variable = trace.name.split(', ')
        trace.name = f"{is_predicted} {variable} frequency spectrum"
        trace.legendgroup = shot + is_predicted + variable
        trace.legendgrouptitle = {'text': f"Shot {shot}: {is_predicted} {variable} signal"}
        if is_predicted == "Predicted":
            trace.fill = 'tonexty'
            rgb = tuple(str(int(trace.line.color[i:i + 2], 16)) for i in (1, 3, 5))
            trace.fillcolor = f'rgba({",".join(rgb)}, 0.1)'
        fig.add_trace(trace, row=2, col=1)

    # Add vertical rectangle to time-domain plot
    C = get_current_config()
    fig.add_vrect(
        x0=-0.5,
        x1=cutoff_t - 0.5,
        opacity=0.2,
        line_width=0,
        layer="below",
        fillcolor="LightSalmon",
        row=1,
        col=1
    )
    fig.update_xaxes(range=[-0.5, C.data.seq_length], row=1, col=1)
    title += "| Signal and Frequency Spectrum"
    if subtitle:
        title += f"<br><sub>{subtitle}</sub>"
    fig.update_layout(title_text=title, title_automargin=True, title_y=.97)
    # if subtitle:
    #     fig.update_layout(title_subtitle=dict(text=str(subtitle)))

    fig.update_yaxes(type="log", row=2, col=1)  # Set y-axis to log scale for the frequency spectrum plot
    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(args=[{
                        "yaxis2.type": "log"
                    }], label="Log", method="relayout"),
                    dict(args=[{
                        "yaxis2.type": "linear"
                    }], label="Linear", method="relayout"),
                    # Fill color buttons
                    dict(
                        args=[
                            {
                                "fill":
                                    [
                                        trace.fill if trace.name.startswith('Predicted') else None
                                        for trace in fig.data
                                    ]
                            }
                        ],
                        label="Fill",
                        method="update"
                    ),
                    dict(args=[{
                        "fill": [None for trace in fig.data]
                    }], label="No Fill", method="update"),
                ],
                pad={
                    "r": 0,
                    "t": 0
                },
                showactive=True,
                x=.995,
                xanchor="right",
                y=0.445,
                yanchor="top"
            ),
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(args=[{
                        "visible": [True] * len(fig.data)
                    }], label="All", method="update"),
                    dict(
                        args=[{
                            "visible": [trace.name.startswith('Predicted') for trace in fig.data]
                        }],
                        label="Predicted",
                        method="update"
                    ),
                    dict(
                        args=[{
                            "visible": [trace.name.startswith('Target') for trace in fig.data]
                        }],
                        label="Targets",
                        method="update"
                    ),
                ],
                pad={
                    "r": 0,
                    "t": 00
                },
                showactive=True,
                x=1.015,
                xanchor="left",
                y=1.01,
                yanchor="bottom"
            )
        ]
    )
    return fig


@torch.inference_mode()
def get_and_plot_predictions(module, batch, title_base=""):
    """Get and plot predictions for a batch of shots with traces and spectrogram.

    Plots n shots in the signal plot, and one signal from one shot in the spectogram plot.
    """
    C = get_current_config()

    shot_numbers, controls, observables = batch  # this should be in plot function
    losses, outputs = module.evaluate(batch)

    df = build_plotting_df_time(
        shot_numbers,
        controls,
        observables,
        outputs['x_pred_t'],
        time_last=C.data.time_last,
        isolated_prediction_line=False
    )
    df_freq = build_plotting_df_freq(outputs['x_pred_freq'], outputs['x_target_freq'], shot_numbers)
    # fig_shots = plot_shot_batch(df, title=title, cutoff_t=C.data.seq_length - C.validation_rollout)
    fig_time_and_freq = plot_signal_and_spectrum(
        df,
        df_freq=df_freq,
        title=title_base,
        subtitle=format_losses(losses),
        cutoff_t=C.data.seq_length - C.validation_rollout
    )

    return fig_time_and_freq
