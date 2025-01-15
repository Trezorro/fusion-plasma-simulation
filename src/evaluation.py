"""Functions for evaluating and visualizing model performance."""
#%%
from typing import Callable, Sequence
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from torch.utils import data
import lightning as L
import lightning.pytorch as pl

import wandb

from src.plotting import get_and_plot_predictions
import src.flow_plots as fp

PlotFunction = Callable[[L.LightningModule, Sequence, str], plt.Figure | go.Figure | wandb.Image]


class PlotsCallback(L.Callback):
    """Calls a specified function, with the model and batch as arguments.

    Model is always put in eval mode, and the batch may be tensor or tuple, on some device.
    Plot function should take care of accepting any device.

    Args:
        plot_fn_key (str): The key of the function to call, also used as the wandb logging key.
        num_samples (int): The number of samples to plot at once.
        every_n_epochs (int): The interval of epochs to plot validation data.
        train_every_n_epochs (int): The interval of epochs to plot training data. If 0, don't plot training data.
    """

    # these functions should each accept the args: model, batch, title_base
    PLOT_FN_OPTIONS = {
        'spectogram_plot': get_and_plot_predictions,
        '2d_flow_plot': fp.plot_flow,
        'line_flow_plot': fp.plot_flow_and_lines_plotly,
    }

    def __init__(self, plot_fn_key: str, num_samples=8, every_n_epochs=5, train_every_n_epochs=20):
        super().__init__()
        self.plot_fn: PlotFunction = self.PLOT_FN_OPTIONS[plot_fn_key]  # Function to generate images
        self.plot_key = plot_fn_key  # Name of the plot, used as wandb logging key
        self.num_samples = num_samples  # Number of samples to log
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs
        self.train_every_n_epochs = train_every_n_epochs

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: L.LightningModule):
        # Skip for all other epochs
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Generate images
            data_set = trainer.val_dataloaders.dataset
            batch = next(
                iter(data.DataLoader(data_set, batch_size=self.num_samples, shuffle=False))
            )  # this can be in general function
            fig = self.plot_fn(
                module=pl_module,
                batch=batch,  # type: ignore
                title_base=f"TRAINDATA | {wandb.run.name} | Epoch {trainer.current_epoch}"
            )
            wandb.log({f"val/{self.plot_key}": fig, "trainer/global_step": trainer.global_step}, commit=False)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: L.LightningModule):
        if self.train_every_n_epochs and trainer.current_epoch % self.train_every_n_epochs == 0:
            # Generate images
            data_set = trainer.train_dataloader.dataset
            batch = next(
                iter(data.DataLoader(data_set, batch_size=self.num_samples, shuffle=False))
            )  # this can be in general function
            fig = self.plot_fn(
                module=pl_module,
                batch=batch,  # type: ignore
                title_base=f"TRAINDATA | {wandb.run.name} | Epoch {trainer.current_epoch}"
            )
            wandb.log(
                {
                    f"train/{self.plot_key}": fig,
                    "trainer/global_step": trainer.global_step
                }, commit=False
            )


# %% Utilities and metrics
def batch_variance(time_series_batch, mean_adjusted=False, reduce='mean'):
    """Calculate the variance in between a batch of time series, for each time step.

    Optionally, first normalize the time series by dividing by the mean of each series.

    Args:
        time_series_batch (torch.Tensor): The time series batch. (batch_size, n_variables, seq_length)
        mean_adjusted (bool): If True, divide each time series by its mean before calculating variance.

    """
    if mean_adjusted:
        time_series_batch = time_series_batch / (time_series_batch.mean(dim=2, keepdim=True) + 1e-8)
    variances = time_series_batch.var(dim=0)
    if reduce == 'mean':
        return variances.mean()
    elif reduce == 'sum':
        return variances.sum()
    else:
        return variances


def output_variance_per_input_variance(output_batch, input_batch, mean_adjusted=False):
    """Calculate the ratio of the output variance to the input variance."""
    output_var = batch_variance(output_batch, mean_adjusted=mean_adjusted)
    input_var = batch_variance(input_batch, mean_adjusted=mean_adjusted)
    return output_var / input_var
