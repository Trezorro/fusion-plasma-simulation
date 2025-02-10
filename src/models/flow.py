from functools import partial
from typing import Any, Optional
import lightning as L
import torch
import torchinfo
import torchmetrics
import wandb
from omegaconf import DictConfig

from src.config import get_current_config
from src.models.flow_nets import VelocityNet
from src.models.unet_conditional import ConditionalUNet
from torchcfm.optimal_transport import OTPlanSampler

import logging

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

    def __init__(
        self,
        model: str = "VelocityNet",
        model_params: Optional[DictConfig | dict] = None,
        loss: str = "MSELoss",
        optimizer_params: Optional[DictConfig | dict] = None,
        prior: str = "normal",
        ot_method: Optional[str] = None,
        ot_replace: bool = False,
        **kwargs: Any
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_params = optimizer_params or dict()  # type: ignore
        if model_params is None:
            model_params = dict()
        self.model = self.MODEL_OPTIONS[model](**model_params)  # type: ignore
        self.loss = self.LOSS_OPTIONS[loss]()
        self.prior = prior
        self.step_fn = self.fwd_euler_step
        self.ot_method = ot_method
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=0.05) if ot_method else None
        self.ot_replace = ot_replace

    def forward(self, x, t, conditioning=None):
        return self.model(x, t, conditioning=conditioning)

    def training_step(self, batch, batch_idx):
        t, samples_at_t, velocity, conditioning_input = self.interpolate_samples(batch)
        pred_velocity = self.model(samples_at_t, t, conditioning_input)
        loss = self.loss(pred_velocity, velocity)
        self.log("loss/train", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        t, samples_at_t, velocity, conditioning_input = self.interpolate_samples(batch)
        pred_velocity = self.model(samples_at_t, t, conditioning_input)
        loss = self.loss(pred_velocity, velocity)
        self.log("loss/val", loss, prog_bar=True)
        return loss

    def interpolate_samples(self, batch):
        _meta, conditioning_inputs, target_samples = batch
        prior_samples = self.get_prior_samples(conditioning_inputs, target_samples.size())

        if self.ot_sampler is not None:
            # sample from optimal transport plan based on prior and target samples in minibatch
            prior_samples, target_samples = self.ot_sampler.sample_plan(
                prior_samples, target_samples, replace=self.ot_replace
            )
            # put back on correct device, because OT sampler may have moved them to cpu. TODO: verify on a gpu
            # prior_samples = prior_samples.to(self.device)
            # target_samples = target_samples.to(self.device)

        # interpolate the probability path at t (making the example path)
        t = torch.rand(target_samples.size(0), device=self.device)
        # if self.warp_fn is not None: # TODO warp option
        #     t = self.warp_fn(t)
        t_broadcast = t.view(-1, 1, 1)
        samples_at_t = prior_samples * (1 - t_broadcast) + target_samples * t_broadcast
        target_velocity = target_samples - prior_samples
        return t, samples_at_t, target_velocity, conditioning_inputs

    def get_prior_samples(self, conditioning_inputs, target_size: torch.Size):
        assert type(target_size) == torch.Size and 2 <= len(
            target_size
        ) <= 5, "target_size must be a torch.Size with 2-5 dimensions"
        match self.prior:
            case "normal":
                prior_samples = torch.randn(target_size, device=self.device)
            case "copy":
                assert 'x_history' in conditioning_inputs, "x_history must be in conditioning for prior='copy'"
                # in case the x_history is longer than the target_samples, crop to seq_length
                seq_length = target_size[2]
                prior_samples = conditioning_inputs['x_history'][:, :, -seq_length:]
            case _:
                raise ValueError(f"Invalid prior: {self.prior}")
        return prior_samples

    test_step = validation_step

    @torch.inference_mode()
    def evaluate(self, batch: tuple[dict, dict, torch.Tensor], n_steps=50, warp_fn=None, to_cpu=True):
        # TODO: may add matching, and (pred) velocity to the output  to debug and plot a training step.
        self.model.eval()
        meta, conditioning_input, target_samples = batch
        prior_samples = self.get_prior_samples(conditioning_input, target_samples.size())
        generated_samples, trajectories = self.integrate_path(
            prior_samples.to(self.device),
            conditioning_input=conditioning_input,
            n_steps=n_steps,
            warp_fn=warp_fn,
            save_trajectories=True
        )
        self.model.train()  # Reset model to training mode
        # trajectories shape: [n_steps, batch_size, channels, num_time_points]
        if to_cpu:
            prior_samples, generated_samples, target_samples = prior_samples.cpu(), generated_samples.cpu(
            ), target_samples.cpu()
            trajectories = trajectories.cpu()

        return dict(
            meta=meta,
            conditioning_input=conditioning_input,
            target_samples=target_samples,
            prior_samples=prior_samples,
            generated_samples=generated_samples,
            trajectories=trajectories
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
        for i in range(len(ts) - 1):
            current_points = self.step_fn(velocity_model, current_points, ts[i], ts[i + 1] - ts[i])
            if save_trajectories:
                trajectories.append(current_points)
        if save_trajectories:
            return current_points, torch.stack(trajectories)
        return current_points

    def configure_optimizers(self):
        if self.optimizer_params is None:
            self.optimizer_params = dict()
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_params)
        return optimizer

    def log_summary(self, config: DictConfig):
        """
        Log a summary of the model.
        
        * `x` has shape `[batch_size, in_channels, *input_dims]`
        * `t` has shape `[batch_size]`

        Args:
            config (DictConfig): The global configuration object, mirroring the <config>.yaml file.
        """
        dummy_data = self.get_dummy_input_tensor(
            (
                (config.batch_size, config.model.params.model_params.input_channels, config.data.seq_length),
                (config.batch_size,), {
                    'x_history':
                        (
                            config.batch_size, config.model.params.model_params.input_channels,
                            config.data.seq_length
                        )
                }
            ), torch.float32, self.device
        )
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

        if __name__ == '__main__':  # Don't log to wandb in test mode, break now.
            return
        wandb.log(
            {
                "model/summary": str(summary),
                "model/trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                # "model/minimum_input_length": self.convnet.minimum_input_length,
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
