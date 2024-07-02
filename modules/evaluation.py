"""Functions for evaluating and visualizing model performance."""
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.express as px

import torch
import wandb
from torch.utils import data




def build_output_df(shot_numbers, controls, observables, outputs):
    """Given the model outputs, build a dataframe with the predictions.

    Handles multiple shots (small batch) in one go.

    Args:
        shot_numbers (torch.Tensor): The shot numbers. (scalars)
        controls (torch.Tensor): The control variables. (batch_size, seq_length, n_controls)
        observables (torch.Tensor): The observable variables. (batch_size, seq_length, n_observables)
        outputs (torch.Tensor): The model outputs. (batch_size, seq_length, n_observables)

    """
    C = wandb.config
    output_cols = [f"^{i}" for i in C.data.cols.x]
    seq_length = outputs.shape[1]
        # DF will have columns: ShotNum, t, ^FIR, ^PD, ^DML, ^POHM, ^Z_axis, IP, gas_fringes, NBI, ECRH, a_minor, KAPPA, DELTA
    df = pd.DataFrame(
        index=np.repeat(
            shot_numbers.numpy().astype(int), seq_length
        ),  # ShotNum, each repeated seq_length times
        columns=["t"] + output_cols + C.data.cols.x + C.data.cols.c,
    )
    for shot, output, control_seq, observable_seq in zip(shot_numbers, outputs, observables, controls):
        df.loc[int(shot)] = np.concatenate([np.arange(seq_length)[:,np.newaxis],  # t
                                                output.numpy(),
                                                observable_seq.numpy(),
                                                control_seq.numpy()],
                                               axis=1)
    return df


def log_predictions(model, data_set, n=5):
    model.eval()
    with torch.no_grad():
        shot_numbers, controls, observables = next(iter(data.DataLoader(data_set, batch_size=n, shuffle=False)))
        inputs = torch.cat((controls, observables), dim=2) # (batch_size, seq_length, variables)
        outputs = model(inputs) #  (batch_size, seq_length, target_variables)
        df = build_output_df(shot_numbers, controls, observables, outputs)
        plot_sample(df.loc[df.index[0]], title="Predictions", show=False)
        table = wandb.Table(dataframe=df.reset_index(names='ShotNum'))
        wandb.log({"val/predictions": table})
    

def plot_sample(df: pd.DataFrame, title="", show=False):
    """Plot a shot of the data. Df shot be a single shot."""
    df_stacked = df.set_index('t').stack(future_stack=True).reset_index(name='value').rename(columns={'level_1': 'variable'})
    df_stacked['is_prediction'] = df_stacked['variable'].str.startswith('^')
    df_stacked['variable'] = df_stacked['variable'].str.replace('^', '')
    fig = px.line(df_stacked, x='t', y='value', color='variable', symbol='is_prediction', line_dash='is_prediction', markers='is_prediction', title=title)
    # fig.update_xaxes(rangeslider_visible=True)

    fig.show()