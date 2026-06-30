"""Functions for evaluating and visualizing model performance."""
#%%
import logging
import time
from typing import Any, Callable, Mapping
from venv import logger

import lightning as L
import plotly.graph_objects as go
import torch
from matplotlib import pyplot as plt
from torch.utils import data

import src.plotters.flow_plots as fp
import src.plotters.plot_animations
import src.plotters.plot_entropy as entropy
import wandb

from src.data_loaders import FusionShotDataModule, FusionShotDataset
from src.plotters.printing_plots import multi_sample_single_window_lines_plotly
from src.config import get_current_config
from src.to_pdf import dump_figure_to_pdfs
from src.metrics.metrics import prefix_metrics
from src.models.flow import FlowModule
from src.plotters.histograms import plot_peak_prominences_histogram

PlotFunction = Callable[[Any], plt.Figure | go.Figure | wandb.Image]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def prune_online_checkpoints(run):
    """Deletes wandb model artifacts that do not have a 'best' or 'latest' alias.

    Wandb logs every checkpoint as an artifact (log_model="all" in WandbLogger).
    This function keeps only the meaningful checkpoints and removes the rest to
    save cloud storage. Called at the end of each validation epoch and at run end.

    Args:
        run: Active wandb run object.
    """
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
        self.solve_method = evaluation_config.get("solve_method", "simple")
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
            try:
                fig = plot_fn(**evaluation_output, n=n, title_base=title_base, **params)
                wandb.log({f"{trainval}/{log_key}": fig, "trainer/global_step": global_step}, commit=False)
            except Exception as e:
                logger.error(f"At plot function {key}: {e}",)
                logger.debug("Plot function parameters: %s", params)
                logger.exception(e)
                logger.info("Continuing like nothing happened... (☞ﾟヮﾟ)☞")


    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        logger.debug(f"on_validation_epoch_end called at epoch {trainer.current_epoch}")
        try:
            if not wandb.run.disabled:
                prune_online_checkpoints(wandb.run)
            # Skip for all other epochs
            # Fires at epochs 1, 6, 11, ... (== 1 not == 0) so epoch 0 is only covered
            # by scrutinize_epochs, not the periodic trigger.
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
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.error("During single batch test set plotting: %s", e)
            logger.exception(e)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        logger.debug(f"on_train_epoch_end called at epoch {trainer.current_epoch}")
        if self.train_every_n_epochs and (
            (trainer.current_epoch % self.train_every_n_epochs == 0) or trainer.current_epoch <= self.scrutinize_epochs
        ):
            try:
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
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.error("During single batch TRAIN set plotting: %s", e)
                logger.exception(e)

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        logger.info(f" =========== Starting training for EPOCH {trainer.current_epoch}=========== ")


class TrainStepMonitor(L.Callback):
    """Logs the number of train steps and train steps per minute to wandb.

    samples_seen counts rematches: each batch is counted batch_rematch_factor
    times, reflecting actual gradient updates rather than data items.
    """

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



def evaluate_window_set(model: FlowModule, data_module: FusionShotDataModule, shot_t: list[tuple]):
    """Evaluates the model on a set of specific shot windows and writes PDF plots.

    Runs after every trainer.validate(), including reeval runs. Always executes
    regardless of other config flags; control which windows are evaluated by
    editing window_set in the config.

    Args:
        model: FlowModule instance (must already be on the target device).
        data_module: FusionShotDataModule with a prepared test_dataset.
        shot_t: List of [shot_number, time_seconds] pairs from config.window_set.
            Each pair identifies a specific moment in a TCV shot to evaluate.

    Outputs:
        PDF plots at output/pdfplots/{run_name}/qualitative_samples/ at 8 sizes,
        plus a "nolegend" variant and a JSON metadata file per window.
    """
    test_dataset = data_module.test_dataset
    logger.info("Running evaluate_window_set for %s shot windows", len(shot_t))
    for shot, t in shot_t:
        try:
            batch = test_dataset.quick_window(shot, t, repeat=4)
            if batch is None:
                continue
            output = model.evaluate(batch, data_module=data_module, n_steps=120 if torch.cuda.is_available() else 5)
            fig = multi_sample_single_window_lines_plotly(**output, title=f"Shot #{shot}")
            dump_figure_to_pdfs(
                fig,
                plot_name="qualitative_samples",
                subgroup='full',
                measure=f'{t}s',
                channel_name=shot,
                metadata=output['metrics'] # json friend
            )
            fig.update_layout(showlegend=False, margin=dict(l=10, r=0, t=0, b=10), title=None)
            dump_figure_to_pdfs(
                fig, plot_name="qualitative_samples", subgroup='nolegend', measure=f'{t}s', channel_name=shot
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(e)
        else:
            logger.info("Created pdfs for shot window %s : %s", shot, t)
