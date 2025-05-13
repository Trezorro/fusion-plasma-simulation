from functools import partial
from typing import Any, Literal, Optional
import lightning as L
from lightning.pytorch.core.optimizer import LightningOptimizer
import numpy as np
import torch
import torch.utils.data
import torchinfo
import torchmetrics
import wandb
from omegaconf import DictConfig

from src.models.flow_nets import VelocityNet
from src.models.unet_conditional import ConditionalUNet
from src.optimal_transport import OTPlanSampler
import src.metrics as metrics
from src.evaluate_modes import generate_surrogate_labels

import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FlowModule(L.LightningModule):

    TIME_DOMAIN_LOSS = torchmetrics.MeanAbsoluteError
    LOSS_OPTIONS = dict(
        MSELoss=torch.nn.MSELoss,
        L1Loss=torch.nn.L1Loss,
        # TODO: Old frequency errors for direct comparison
        # FrequencySpectrumMSESimple=FrequencySpectrumMSESimple,
        # FrequencyPhaseAmpMSE=FrequencyPhaseAmpMSE,
        # FrequencyPhaseLogAmpMSE=FrequencyPhaseLogAmpMSE,
        # FrequencyAmpMSE=FrequencyAmpMSE,
    )
    MODEL_OPTIONS = dict(
        # VelocityNet=VelocityNet,  # TODO needs to support channels, t, and conditioning
        # UNetModern=UNetModern,
        ConditionalUNet=ConditionalUNet,
    )
    SAMPLE_RATE = 10_000  # Hz
    PRIOR_OPTIONS = ["normal", "levy", "resample", "brownian", "copy"]

    def __init__(
        self,
        model: str = "VelocityNet",
        model_params: Optional[DictConfig | dict] = None,
        loss: str = "MSELoss",
        optimizer_params: Optional[DictConfig | dict] = None,
        prior: Literal["normal", "levy", "resample", "brownian", "copy"] = "normal",
        prior_sigma: float = 0.3,
        ot_method: Optional[str] = None,
        ot_replace: bool = False,
        batch_rematch_factor: int = 1,
        step_every_nth_match: Optional[int] = None,  # if None, step only after all matches.
        gradient_clip_val: float = 1.0,
        **kwargs: Any
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.optimizer_params = optimizer_params or dict()  # type: ignore
        if model_params is None:
            model_params = dict()
        self.model_params = model_params
        self.model = self.MODEL_OPTIONS[model](**model_params)  # type: ignore
        self.loss = self.LOSS_OPTIONS[loss]()
        self.prior = prior
        self.prior_sigma = prior_sigma
        self.step_fn = self.fwd_euler_step
        self.ot_method = ot_method
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=0.05) if ot_method else None
        self.ot_replace = ot_replace
        self.gradient_clip_val = gradient_clip_val
        self.batch_rematch_factor = batch_rematch_factor
        self.step_every_nth_match = step_every_nth_match or batch_rematch_factor
        self._validate_configuration()
        self.automatic_optimization = False
        self.register_buffer("sqrt_dt", torch.sqrt(torch.tensor(1 / self.SAMPLE_RATE)))

    def _validate_configuration(self):
        assert self.batch_rematch_factor > 0 and type(
            self.batch_rematch_factor
        ) == int, "batch_rematch_factor must be positive integer"
        assert self.batch_rematch_factor % self.step_every_nth_match == 0, (
            "batch_rematch_factor must be divisible by step_every_nth_match, or"
            " step_every_nth_match must be None. Otherwise, optimizer steps will"
            " be unbalanced."
        )
        logger.info(
            "Will do %d matches per batch, stepping every %d, so %d steps per batch.", self.batch_rematch_factor,
            self.step_every_nth_match, self.batch_rematch_factor // self.step_every_nth_match
        )
        assert self.prior in self.PRIOR_OPTIONS, f"Invalid prior: {self.prior}"
        assert (self.prior == "normal" or self.ot_method is None), "OT sampling only supported with fully random prior"
        assert ('position_sequence' in self.model_params.conditioning
               ) == (self.model_params["positional_encoding"]
                     is not None), ("Positional encoding must be configured for position_sequence conditioning")

    def forward(self, x, t, conditioning=None):
        return self.model(x, t, conditioning=conditioning)

    def training_step(self, batch, batch_idx):
        opt: LightningOptimizer = self.optimizers()  # type: ignore
        total_loss = 0
        opt.zero_grad()
        for match_i in range(1, self.batch_rematch_factor + 1):
            loss = self.batch_match(batch, batch_idx, match_i=match_i)
            self.manual_backward(loss)
            total_loss += loss.detach()
            if match_i % self.step_every_nth_match == 0:
                # See: on_before_optimizer_step() for gradient clipping
                opt.step()
                opt.zero_grad()
        total_loss /= self.batch_rematch_factor

        self.log("loss/train", total_loss, prog_bar=True)

    def batch_match(self, batch, batch_idx, match_i=0):
        t, samples_at_t, velocity, conditioning_input = self.interpolate_samples(batch)
        pred_velocity = self.model(samples_at_t, t, conditioning_input)
        loss = self.loss(pred_velocity, velocity)
        if loss.isnan().any():
            raise ValueError(f"Loss is NaN: {loss}")
        if loss > 1000:
            logger.warning(
                f"Loss is very high: {loss} at EPOCH {self.current_epoch}, step {self.global_step}\nfor batch {batch_idx} (match {match_i}) of size {velocity.size()}"
            )
            summary_str = lambda x: f"mean: {x.mean().item()}, std: {x.std().item()}, min: {x.min().item()}, max: {x.max().item()}"
            logger.debug("velocity: %s\n pred_velocity:%s", summary_str(velocity), summary_str(pred_velocity))
            logger.debug("Shots: %s", batch[0]['shot_number'])
        return loss

    def validation_step(self, batch, batch_idx):
        t, samples_at_t, velocity, conditioning_input = self.interpolate_samples(batch)
        pred_velocity = self.model(samples_at_t, t, conditioning_input)
        loss = self.loss(pred_velocity, velocity)
        self.log("loss/val", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # single scheduler
        lr_sched = self.lr_schedulers()
        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(lr_sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_sched.step(self.trainer.callback_metrics["loss/train"])

    @torch.no_grad()
    def interpolate_samples(self, batch):
        _meta, conditioning_inputs, target_samples = batch
        prior_samples = self.get_prior_samples(conditioning_inputs, target_samples.size())
        # TODO: just sample 100x more and pair up with many to one target samples, sample as many t's.
        if self.ot_sampler is not None:
            # sample from optimal transport plan based on prior and target samples in minibatch
            pi = self.ot_sampler.get_map(prior_samples, x1=target_samples)
            i, j = self.ot_sampler.sample_map(pi, prior_samples.shape[0], replace=self.ot_replace)
            prior_samples = prior_samples[i]
            target_samples = target_samples[j]
            # Fix for conditioning inputs to match their respective target X
            conditioning_inputs = {k: v[j] for k, v in conditioning_inputs.items()}

        # interpolate the probability path at t (making the example path)
        t = torch.rand(target_samples.size(0), device=self.device)  # TODO add safety non 1.0. see cfm.
        # if self.warp_fn is not None: # TODO warp option
        #     t = self.warp_fn(t)
        t_broadcast = t.view(-1, 1, 1)
        samples_at_t = prior_samples * (1 - t_broadcast) + target_samples * t_broadcast
        target_velocity = target_samples - prior_samples
        return t, samples_at_t, target_velocity, conditioning_inputs

    def get_prior_samples(self, conditioning_inputs, target_size: torch.Size):
        assert type(
            target_size
        ) == torch.Size and 2 <= len(target_size) <= 5, "target_size must be a torch.Size with 2-5 dimensions"
        seq_length = target_size[2]
        match self.prior:
            case "normal":
                prior_samples = torch.randn(target_size, device=self.device) * self.prior_sigma + 0.5
            case "copy":
                assert 'x_history' in conditioning_inputs, "x_history must be in conditioning for prior='copy'"
                # in case the x_history is longer than the target_samples, crop to seq_length
                prior_samples = conditioning_inputs['x_history'][:, :, -seq_length:].to(self.device)
            case "brownian":
                x_last = conditioning_inputs['x_history'][:, :, -1]
                prior_samples = self.generate_brownian_motion(x_last, seq_length)
            case "levy":
                x_last = conditioning_inputs['x_history'][:, :, -1]
                prior_samples = self.generate_levy_jump_process(x_last, seq_length)
            case "resample":
                assert 'x_history' in conditioning_inputs, "x_history must be in conditioning for prior='resample'"
                # in case the x_history is longer than the target_samples, crop to seq_length
                history = conditioning_inputs['x_history'].to(self.device)
                prior_samples = self.generate_history_resample(history, seq_length)
            case _:
                raise ValueError(f"Invalid prior: {self.prior}")
        return prior_samples

    def generate_brownian_motion(self, x_last, seq_length):
        """Generate a Brownian motion trajectory with fixed dt = 1/10000.

        Brownian motion follows the stochastic differential equation:
        dx_t = dB_t,
            where B_t is a standard Brownian motion with independent Gaussian increments:
            x_{t+dt} = x_t + \\sqrt{dt} * \\xi,  \\xi \\sim \\mathcal{N}(0,1).
        
        Parameters:
        x_last (torch.Tensor): Initial positions of shape (N, C), where N = batch size, C = number of channels.
        seq_length (int): Number of time steps in the sequence.
        device (str): Device to run computations on ("cpu" or "cuda").
        
        Returns:
        torch.Tensor: Brownian motion trajectory of shape (N, C, seq_length).
        """
        N, C = x_last.shape
        dB = self.sqrt_dt * torch.randn((N, C, seq_length), device=self.device) * self.prior_sigma
        x_t = x_last.unsqueeze(-1) + torch.cumsum(dB, dim=-1)
        return x_t

    def generate_levy_jump_process(self, x_last, seq_length, jump_prob=0.01):
        """Generate a Lévy jump process trajectory.

        The Lévy jump process is stationary for long periods and jumps to random levels
        with a given probability.

        Parameters:
        x_last (torch.Tensor): Initial positions of shape (N, C), where N = batch size, C = number of channels.
        seq_length (int): Number of time steps in the sequence.
        jump_prob (float): Probability of a jump at each time step.
        jump_scale (float): Scale of the random jumps.

        Returns:
        torch.Tensor: Lévy jump process trajectory of shape (N, C, seq_length).
        """
        N, C = x_last.shape
        jumps = torch.rand((N, C, seq_length), device=self.device) < jump_prob  # Jump mask
        jump_values = self.prior_sigma * torch.randn((N, C, seq_length), device=self.device)  # Random jump values
        jump_diff = jumps * jump_values
        x_t = x_last.unsqueeze(-1) + torch.cumsum(jump_diff, dim=-1)
        return x_t

    def generate_history_resample(self, history, target_seq_length):
        """Generate a prior by sampling from the history values.
        This is used for the 'resample' prior option.

        Parameters:
        history (torch.Tensor): History samples of shape (N, C, H), where N = batch size, C = number of channels,
            H = history length.
        target_seq_length (int): Number of time steps in the target sequence."""
        N, C, H = history.shape
        # Randomly sample indices from the history
        indices = torch.randint(0, H, (N, C, target_seq_length), device=self.device)
        # Gather the samples from the history
        prior_samples = history.gather(2, indices)
        return prior_samples

    test_step = validation_step

    @torch.inference_mode()
    def evaluate(
        self, batch: tuple[dict, dict, torch.Tensor], data_set: torch.utils.data.Dataset, n_steps=50, warp_fn=None
    ):
        """Evaluates the model on a given batch of data. Batch will be moved to the correct device.

        The model is set to evaluation mode, and the generated samples are compared to the target samples. 
        The function computes various metrics, including moment errors, entropy metrics, and peak metrics.
        The generated samples are obtained by integrating the prior samples using the model's velocity function.
        Outputs the generated samples, target samples, and various metrics and the trajectories of the generated samples.

        Args:
            batch (tuple[dict, dict, torch.Tensor]): A tuple containing metadata, conditioning input, 
                and target samples. The metadata is a dictionary, the conditioning input is a dictionary, 
                and the target samples are a tensor.
            n_steps (int, optional): The number of integration steps to perform. Defaults to 50.
            warp_fn (callable, optional): A function to warp the generated samples during integration. 
                Defaults to None.

        Returns:
            dict: A dictionary containing the following keys:
                - meta [dict]: Metadata associated with the batch.
                - conditioning_input [dict]: The conditioning input used for evaluation.
                - target_samples [Tensor]: The target samples from the batch.
                - prior_samples [Tensor]: Samples generated from the prior distribution.
                - generated_samples [Tensor]: Samples generated by the model.
                - trajectories [Tensor]: Trajectories of the generated samples during integration.
                - metrics [dict]: A dictionary of computed metrics, including moment errors, entropy metrics, 
                  and peak metrics.
                - peak_features [dict]: Features related to the peaks in the generated samples. Contains keys
                    pred_peaks and target_peaks. See `metrics.get_peak_metrics` for details. 
        """
        # TODO: may add matching, and (pred) velocity to the output  to debug and plot a training step.
        self.model.eval()
        # Use lightnings manner of moving to correct current device:
        meta, conditioning_input, target_samples = self._apply_batch_transfer_handler(batch)
        prior_samples = self.get_prior_samples(conditioning_input, target_samples.size())
        generated_samples, trajectories = self.integrate_path(
            prior_samples,
            conditioning_input=conditioning_input,
            n_steps=n_steps,
            warp_fn=warp_fn,
            save_trajectories=True
        )
        # Metrics
        metrics_out = metrics.get_moments_errors_per_channel(generated_samples, target_samples)
        self.model.train()  # Reset model to training mode

        meta, conditioning_input, target_samples, prior_samples, generated_samples, trajectories = self._apply_batch_transfer_handler(
            (meta, conditioning_input, target_samples, prior_samples, generated_samples, trajectories),
            device='cpu'  # type: ignore
        )
        # surrogate labels
        surr_labels_pred, surr_labels_target = generate_surrogate_labels(
            meta, generated_samples, target_samples, data_set=data_set
        )
        # TODO do metric calculation elsewhere
        metrics_out |= metrics.get_entropy_metrics(generated_samples, target_samples)
        peak_metrics, peak_features = metrics.get_peak_metrics(generated_samples, target_samples)
        return dict(
            meta=meta,
            conditioning_input=conditioning_input,
            target_samples=target_samples,
            prior_samples=prior_samples,
            generated_samples=generated_samples,
            trajectories=trajectories,
            metrics=metrics_out | peak_metrics,
            peak_features=peak_features,
            surr_labels_pred=surr_labels_pred,
            surr_labels_target=surr_labels_target
        )

    @staticmethod
    @torch.no_grad()
    def fwd_euler_step(velocity_model, current_points, current_t, dt):
        """
        Perform a forward Euler step.

        Args:
            model: The model to use for computing the velocity.
            current_points (torch.Tensor): Shape [batch_size, num_features]
            current_t (float): The current time step.
            dt (float): The time step size.

        Returns:
            torch.Tensor: Shape [batch_size, num_features]
        """
        # TODO change to self(x, t) for the model
        velocity = velocity_model(current_points, current_t)
        return current_points + velocity * dt

    @torch.inference_mode()
    def integrate_path(
        self,
        initial_points,
        conditioning_input=None,
        step_fn=fwd_euler_step,
        n_steps=100,
        save_trajectories=False,
        warp_fn=None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate a path using the given step function.

        Args:
            initial_points (torch.Tensor): Shape [batch_size, num_features]
            step_fn: The function to use for computing the next step.
            n_steps (int): The number of steps to integrate.
            save_trajectories (bool): Whether to save the trajectories.
            warp_fn: A function to warp the time steps.

        Returns:
            torch.Tensor: Shape [batch_size, num_features]
            torch.Tensor (optional): Shape [n_steps, batch_size, num_features] if save_trajectories is True
        """
        current_points = initial_points.clone()
        if conditioning_input is not None and self.model.conditioning:
            velocity_model = partial(self.model, conditioning_input=conditioning_input)
        else:
            velocity_model = self.model
        ts = torch.linspace(0, 1, n_steps, device=self.device)
        if warp_fn:
            ts = warp_fn(ts)
        if save_trajectories:
            trajectories = [current_points]
        logger.debug(f"Integrating path with {n_steps} steps")
        logger.debug(
            "Devices: timesteps: %s, current_points: %s, conditioning_input: %s", ts.device,
            current_points.device, {
                k: v.device for k, v in conditioning_input.items()
            } if conditioning_input is not None else None
        )
        # Integrate and use progress bar if running on CPU
        for i in tqdm(
            range(len(ts) - 1),
            disable=self.device.type != "cpu",
            desc="Integrating path",
        ):
            current_points = self.step_fn(velocity_model, current_points, ts[i], ts[i + 1] - ts[i])
            if save_trajectories:
                trajectories.append(current_points)
        if save_trajectories:
            return current_points, torch.stack(trajectories)
        return current_points

    def configure_optimizers(self):
        if self.optimizer_params is None:
            self.optimizer_params = dict()
        self.opt = torch.optim.Adam(self.parameters(), **self.optimizer_params)
        self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
        )

        return {
            'optimizer': self.opt,
            'lr_scheduler':
                {
                    # REQUIRED: The scheduler instance
                    "scheduler": self.reduce_lr_on_plateau,
                    # The unit of the scheduler's step size, could also be 'step'.
                    # 'epoch' updates the scheduler on epoch end whereas 'step'
                    # updates it after a optimizer update.
                    "interval": "epoch",
                    # How many epochs/steps should pass between calls to
                    # `scheduler.step()`. 1 corresponds to updating the learning
                    # rate after every epoch/step.
                    "frequency": 1,
                    # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                    "monitor": "loss/val",
                    # If using the `LearningRateMonitor` callback to monitor the
                    # learning rate progress, this keyword can be used to specify
                    # a custom logged name
                    # "name": None,
                }
        }

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        grads = [
            param.grad.detach().flatten()
            for param in self.model.parameters()
            if param.grad is not None and param.requires_grad
        ]
        total_norm = torch.cat(grads).norm()
        self.log("loss/grad_norm", total_norm, on_step=True, on_epoch=False, prog_bar=True)
        if total_norm > 1000:
            logger.warning(
                f"Gradient norm is very high: {total_norm} at EPOCH {self.current_epoch}, step {self.global_step}"
            )
        self.clip_gradients(
            optimizer, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm"
        )

    def log_summary(self, config: DictConfig):
        """
        Log a summary of the model.
        
        * `x` has shape `[batch_size, in_channels, *input_dims]`
        * `t` has shape `[batch_size]`

        Args:
            config (DictConfig): The global configuration object, mirroring the <config>.yaml file.
        """
        conditioning_shape = {
            'x_history':
                (config.batch_size, config.model.params.model_params.input_channels, config.data.seq_length),
            'position_sequence': (config.batch_size, config.data.seq_length * 2),
            'c':
                (
                    config.batch_size, config.model.params.model_params.c_channels,
                    config.data.history_length + config.data.seq_length
                ),
        }
        # Filter: only keep the conditioning inputs that are actually used in the model
        conditioning_shape = {
            k: v for k, v in conditioning_shape.items() if k in config.model.params.model_params.conditioning
        }
        x_shape = (config.batch_size, config.model.params.model_params.input_channels, config.data.seq_length)
        t_shape = (config.batch_size,)
        # this is passed to the forward of the UNet model(x, t, conditioning)
        expected_in_shape = [x_shape, t_shape, conditioning_shape]
        dummy_data = self.get_dummy_input_tensor(expected_in_shape, torch.float32, self.device)
        summary = torchinfo.summary(
            self.model,
            input_data=dummy_data,
            # batch_dim=0,
            col_names=[
                "input_size",
                "output_size",
                "kernel_size",
                "num_params",
                # "params_percent",
                "mult_adds",
                # "trainable"
            ],
            row_settings=("depth", "var_names"),
            verbose=1
        )  # (batch_size, seq_length, input_size)

        logger.info("Model expected input shape: %s", expected_in_shape)
        if __name__ == '__main__':  # Don't log to wandb in test mode, break now.
            return
        wandb.log(
            {
                "model/summary": str(summary),
                "model/trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "model/expected_input_shape": expected_in_shape,
            },
            step=0
        )

    def get_dummy_input_tensor(
        self,
        input_sizes: list[tuple[int, ...] | dict[str, tuple[int, ...]]],
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Get input_tensor with batch size 1 for use in model.forward()"""
        x = []
        for size_entry in input_sizes:
            if isinstance(size_entry, dict):
                input_tensor = {k: torch.rand(v, device=device, dtype=dtype) for k, v in size_entry.items()}
            else:
                input_tensor = torch.rand(size_entry, device=device, dtype=dtype)
            x.append(input_tensor)
        return x
