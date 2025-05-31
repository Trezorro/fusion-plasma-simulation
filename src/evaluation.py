"""Functions for evaluating and visualizing model performance."""
#%%
import logging
import time
from typing import Any, Callable, Mapping
from venv import logger

import lightning as L
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import torch
from torch.utils import data

import src.plotters.plot_animations
import src.plotters.plot_entropy as entropy
import src.plotters.flow_plots as fp
import wandb
from src.metrics.metrics import prefix_metrics
from src.plotters.histograms import plot_peak_prominences_histogram

PlotFunction = Callable[[Any], plt.Figure | go.Figure | wandb.Image]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def prune_online_checkpoints(run):
    api = wandb.Api()
    artifacts = api.run(run.path).logged_artifacts()
    for art in artifacts:
        if art.type == "model":
            if len(art.aliases) == 0 or not ("best" in art.aliases or "latest" in art.aliases):
                logger.warning("Deleting %s", art.name)
                art.delete()
            else:
                logger.info("Keeping %s \n with aliases %s", art.name, art.aliases)


class PlotsCallback(L.Callback):
    """Calls a specified function, with the model and batch as arguments.

    The plot functions should accept pytorch tensors as input and return some figure or image.

    Args in config.evaluation:
        n_steps (int): The number of steps to plot.
        val_every_n_epochs (int): The interval of epochs to plot validation data.
        train_every_n_epochs (int): The interval of epochs to plot training data. If 0, don't plot training data.
        scrutinize_epochs (int): Epochs equal or lower than this will always be fully evaluated. -1 to disable.
        plot_functions (list): A list of dictionaries, each specifying a function to call and its arguments.
            key: The key of the function to call, also used as the wandb logging key.
            n: The number of samples to plot. If not specified, the maximum number of samples over all plots or the batch size is used.
            Any other arguments are passed to the function.
    """

    # these functions should each accept the args: n, title_base, and **kwargs
    PLOT_FN_OPTIONS = {
        # 'spectogram_plot': get_and_plot_predictions, # TODO update interface
        '2d_flow_plot': fp.plot_flow,
        'line_flow_plot': fp.plot_flow_and_lines_plotly,
        'animated_traces': src.plotters.plot_animations.animated_trajectory_plotly,
        'multi_channel_lines': fp.multi_channel_lines_plotly,
        'entropy_plot': entropy.plot_entropy,
        'histogram': plot_peak_prominences_histogram,
    }

    def __init__(self, evaluation_config: Mapping):
        super().__init__()
        self.config = evaluation_config
        self.n_steps = evaluation_config.get("n_steps", 50)
        self.solve_method = evaluation_config.get("solve_method", "rk4")
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.val_every_n_epochs = self.config.get("val_every_n_epochs", 20)
        self.train_every_n_epochs = self.config.get("train_every_n_epochs", 20)
        self.scrutinize_epochs = self.config.get("scrutinize_epochs", 1)
        self.plot_functions = self.config.get("plot_functions", [])
        self.max_n = self.config.get("max_n", 0)
        if not self.max_n and self.plot_functions:
            self.max_n = max([func_c.get("n", 0) for func_c in self.plot_functions])
        elif not self.max_n:
            logger.warning(
                "No plot functions specified and no max n. Defaulting to 0. Set max_n in config to change this."
            )

    def call_plot_functions(self, evaluation_output: dict, trainval: str, global_step: int, title_base: str):
        if not self.plot_functions:
            return
        for func_c in self.plot_functions:
            key = func_c["key"]
            plot_fn: PlotFunction = self.PLOT_FN_OPTIONS[key]  # Function to generate images
            params = func_c.copy()
            log_key = params.pop("log_key", key)  # if log_key is present, use it instead of key
            n = params.pop("n", self.max_n)
            logger.debug(f"Calling plot function {key} with params {params} and n={n}. Will log as {log_key}.")
            fig = plot_fn(**evaluation_output, n=n, title_base=title_base, **params)
            wandb.log({f"{trainval}/{log_key}": fig, "trainer/global_step": global_step}, commit=False)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        logger.debug(f"on_validation_epoch_end called at epoch {trainer.current_epoch}")
        if not wandb.run.disabled:
            prune_online_checkpoints(wandb.run)
        # Skip for all other epochs
        if (trainer.current_epoch % self.val_every_n_epochs == 1) or trainer.current_epoch <= self.scrutinize_epochs:
            if torch.cuda.is_available():
                logger.info(torch.cuda.memory_summary(device=None, abbreviated=False))
            # Generate images
            logger.info(f"Evaluating model for EPOCH {trainer.current_epoch} on validation data.")
            data_set = trainer.datamodule.test_dataloader().dataset
            batch = next(
                iter(data.DataLoader(data_set, batch_size=self.max_n, shuffle=True))
            )  # this can be in general function
            logger.debug("Calling evaluate for validation data.")
            evaluation_output = pl_module.evaluate(batch, n_steps=self.n_steps, solve_method=self.solve_method)
            logger.debug("Evaluation done. Calling plotters.")
            title_base = f"{wandb.run.name} |  Epoch <b>{trainer.current_epoch}</b>"
            self.call_plot_functions(evaluation_output, "val", trainer.global_step, title_base)
            logger.debug("Plotters done.")
            val_metrics = prefix_metrics(evaluation_output['metrics'], prefix='val')  # Add prefix
            wandb.log(val_metrics | {"trainer/global_step": trainer.global_step}, commit=False)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        logger.debug(f"on_train_epoch_end called at epoch {trainer.current_epoch}")
        if self.train_every_n_epochs and (
            (trainer.current_epoch % self.train_every_n_epochs == 0) or trainer.current_epoch <= self.scrutinize_epochs
        ):
            logger.info(f"Evaluating model for EPOCH {trainer.current_epoch} on train data.")
            # Generate images
            data_set = trainer.train_dataloader.dataset
            batch = next(
                iter(data.DataLoader(data_set, batch_size=self.max_n, shuffle=False))
            )  # this can be in general function
            logger.debug("Calling evaluate for train data.")
            evaluation_output = pl_module.evaluate(batch, n_steps=self.n_steps, solve_method=self.solve_method)
            logger.debug("Evaluation done. Calling plotters.")
            title_base = f"TRAINDATA | {wandb.run.name} | Epoch {trainer.current_epoch}"
            self.call_plot_functions(evaluation_output, "train", trainer.global_step, title_base)
            logger.debug("Plotters done.")
            train_metrics = prefix_metrics(evaluation_output['metrics'], prefix='train')  # Add prefix
            wandb.log(train_metrics | {"trainer/global_step": trainer.global_step}, commit=False)

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        logger.info(f" =========== Starting training for EPOCH {trainer.current_epoch}=========== ")


class TrainStepMonitor(L.Callback):
    """Logs the number of train steps and train steps per minute to wandb."""

    def __init__(self):
        super().__init__()
        self.start_time: float = 0.0
        self.train_steps = 0
        self.samples_seen = 0

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self.start_time = time.time()

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx):
        # TODO: check if some reference is kept open here, leaking memory
        self.train_steps += 1
        self.samples_seen += len(batch[2]) * pl_module.batch_rematch_factor
        elapsed_time = (time.time() - self.start_time) / 60.0
        steps_per_minute = self.train_steps / elapsed_time if elapsed_time > 0 else 0
        samples_per_minute = self.samples_seen / elapsed_time if elapsed_time > 0 else 0
        self.log("trainer/samples_seen", self.samples_seen, prog_bar=True)
        self.log("trainer/samples_per_minute", samples_per_minute, prog_bar=True)
        self.log_dict(
            {
                "trainer/steps": trainer.global_step,
                "trainer/global_step": trainer.global_step,
                "trainer/steps_per_minute": steps_per_minute,
            },
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
