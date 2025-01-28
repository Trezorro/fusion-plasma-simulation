from typing import Any, Optional
import lightning as L
import torch
import torchinfo
import torchmetrics
import wandb
from omegaconf import DictConfig

from src.models.flow_nets import VelocityNet
from src.models.unet_conditional import ConditionalUNet


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
        VelocityNet=VelocityNet,
        # UNetModern=UNetModern,
        ConditionalUNet=ConditionalUNet,
    )

    def __init__(
        self,
        model: str = "VelocityNet",
        model_params: Optional[DictConfig | dict] = None,
        loss: str = "MSELoss",
        optimizer_params: Optional[DictConfig | dict] = None,
        **kwargs: Any
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_params = optimizer_params or dict()  # type: ignore
        if model_params is None:
            model_params = dict()
        self.model = self.MODEL_OPTIONS[model](**model_params)  # type: ignore
        self.loss = self.LOSS_OPTIONS[loss]()
        self.step_fn = self.fwd_euler_step

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        t, samples_at_t, velocity = self.interpolate_samples(batch)
        pred_velocity = self.model(samples_at_t, t)
        loss = self.loss(pred_velocity, velocity)
        self.log("loss/train", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        t, samples_at_t, velocity = self.interpolate_samples(batch)
        pred_velocity = self.model(samples_at_t, t)
        loss = self.loss(pred_velocity, velocity)
        self.log("loss/val", loss, prog_bar=True)

        return loss

    def interpolate_samples(self, batch):
        _meta, source_samples, target_samples = batch
        t = torch.rand(source_samples.size(0), device=self.device)
        # if self.warp_fn is not None: # TODO warp option
        #     t = self.warp_fn(t)

        # interpolate the probability path at t (making the example path)
        t_broadcast = t.view(-1, 1, 1)
        samples_at_t = source_samples * (1 - t_broadcast) + target_samples * t_broadcast
        delta = target_samples - source_samples
        return t, samples_at_t, delta

    test_step = validation_step

    @torch.inference_mode()
    def evaluate(self, batch):
        self.model.eval()
        t, samples_at_t, velocity = self.interpolate_samples(batch)
        pred_velocity = self.model(samples_at_t, t)
        loss = self.loss(pred_velocity, velocity)
        losses = {
            "loss": loss
        }  # TODO: this loss makes no sense, but we could KLlosses for the KL divergence, or
        outputs = {"pred_velocity": pred_velocity, "samples_at_t": samples_at_t}
        self.model.train()  # Reset model to training mode
        return losses, outputs

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

    @torch.no_grad()
    def integrate_path(
        self, initial_points, step_fn=fwd_euler_step, n_steps=100, save_trajectories=False, warp_fn=None
    ):
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
        ts = torch.linspace(0, 1, n_steps).to(self.device)
        if warp_fn:
            ts = warp_fn(ts)
        if save_trajectories:
            trajectories = [current_points]
        for i in range(len(ts) - 1):
            current_points = self.step_fn(self.model, current_points, ts[i], ts[i + 1] - ts[i])
            if save_trajectories:
                trajectories.append(current_points)
        if save_trajectories:
            return current_points, torch.stack(trajectories).cpu()
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
        summary = torchinfo.summary(
            self.model,
            input_size=(
                (config.batch_size, config.model.params.model_params.input_channels, config.data.seq_length),
                (config.batch_size,)
            ),
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
