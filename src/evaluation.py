"""Functions for evaluating and visualizing model performance."""
#%%
from typing import Any, Callable, Mapping
import logging
from venv import logger
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from torch.utils import data
import lightning as L
import lightning.pytorch as pl

import wandb

from src.plotting import get_and_plot_predictions
import src.flow_plots as fp

PlotFunction = Callable[[Any], plt.Figure | go.Figure | wandb.Image]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

    # these functions should each accept the args: n, title_base, and **kwargs
    PLOT_FN_OPTIONS = {
        # 'spectogram_plot': get_and_plot_predictions, # TODO update interface
        '2d_flow_plot': fp.plot_flow,
        'line_flow_plot': fp.plot_flow_and_lines_plotly,
        'multi_channel_lines': fp.multi_channel_lines_plotly,
    }

    def __init__(self, evaluation_config: Mapping):
        super().__init__()
        self.config = evaluation_config
        self.n_steps = evaluation_config.get("n_steps", 50)
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.val_every_n_epochs = self.config.get("val_every_n_epochs", 20)
        self.train_every_n_epochs = self.config.get("train_every_n_epochs", 20)
        self.plot_functions = self.config.get("plot_functions", [])
        if not self.plot_functions:
            logger.warning("No plot functions specified for evaluation callback.")
            self.max_n = 0
        else:
            self.max_n = max([func_c.get("n", 0) for func_c in self.plot_functions])

    def call_plot_functions(self, evaluation_output: dict, trainval: str, global_step: int, title_base: str):

        for func_c in self.plot_functions:
            key = func_c["key"]
            plot_fn: PlotFunction = self.PLOT_FN_OPTIONS[key]  # Function to generate images
            params = func_c.copy()
            n = params.pop("n", self.max_n)
            fig = plot_fn(**evaluation_output, n=n, title_base=title_base, **params)
            wandb.log({f"{trainval}/{key}": fig, "trainer/global_step": global_step}, commit=False)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: L.LightningModule):

        # Skip for all other epochs
        if trainer.current_epoch % self.val_every_n_epochs == 1:
            # Generate images
            logger.info(f"Generating validation plots for epoch {trainer.current_epoch}")
            data_set = trainer.val_dataloaders.dataset
            batch = next(
                iter(data.DataLoader(data_set, batch_size=self.max_n, shuffle=False))
            )  # this can be in general function
            logger.debug("Calling evaluate for validation data.")
            evaluation_output = pl_module.evaluate(batch, n_steps=self.n_steps)
            logger.debug("Evaluation done. Calling plotters.")
            title_base = f"{wandb.run.name} |  Epoch {trainer.current_epoch}"
            self.call_plot_functions(evaluation_output, "val", trainer.global_step, title_base)
            logger.debug("Plotters done.")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: L.LightningModule):
        if self.train_every_n_epochs and trainer.current_epoch % self.train_every_n_epochs == 0:
            logger.info(f"Generating training plots for epoch {trainer.current_epoch}")
            # Generate images
            data_set = trainer.train_dataloader.dataset
            batch = next(
                iter(data.DataLoader(data_set, batch_size=self.max_n, shuffle=False))
            )  # this can be in general function
            logger.debug("Calling evaluate for train data.")
            evaluation_output = pl_module.evaluate(batch, n_steps=self.n_steps)
            logger.debug("Evaluation done. Calling plotters.")
            title_base = f"TRAINDATA | {wandb.run.name} | Epoch {trainer.current_epoch}"
            self.call_plot_functions(evaluation_output, "train", trainer.global_step, title_base)
            logger.debug("Plotters done.")



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
