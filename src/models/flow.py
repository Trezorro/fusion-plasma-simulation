from functools import partial
from typing import Any, Literal, Optional
import lightning as L
from lightning.pytorch.core.optimizer import LightningOptimizer
import torch
import torchinfo
import torchmetrics
from torchmetrics.segmentation import DiceScore
import wandb
from omegaconf import DictConfig
from torchdiffeq import odeint

from src.models.unet_conditional import ConditionalUNet
from src.optimal_transport import OTPlanSampler
import src.metrics.metrics as metrics
from src.metrics.evaluate_modes import generate_surrogate_labels, generate_surrogate_labels_batched
from src.metrics.mode_metrics import ModeTransitionMetric
from src.metrics.peak_metric import PeakMetric

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
    PRIOR_OPTIONS = ["normal", "levy", "resample", "brownian", "copy", "constant"]

    def __init__(
        self,
        model: str = "VelocityNet",
        model_params: Optional[DictConfig | dict] = None,
        loss: str = "MSELoss",
        optimizer_params: Optional[DictConfig | dict] = None,
        prior: Literal["normal", "levy", "resample", "brownian", "copy", "constant"] = "normal",
        prior_sigma: float = 0.3,
        ot_method: Optional[str] = None,
        ot_replace: bool = False,
        batch_rematch_factor: int = 1,
        step_every_nth_match: Optional[int] = None,  # if None, step only after all matches.
        gradient_clip_val: float = 1.0,
        flow_steps=150,
        solve_method='simple',
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
        self.ot_method = ot_method
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=0.05) if ot_method else None
        self.ot_replace = ot_replace
        self.gradient_clip_val = gradient_clip_val
        self.batch_rematch_factor = batch_rematch_factor
        self.step_every_nth_match = step_every_nth_match or batch_rematch_factor
        self.flow_steps = flow_steps
        self.solve_method = solve_method
        self._validate_configuration()
        self.automatic_optimization = False
        self.register_buffer("sqrt_dt", torch.sqrt(torch.tensor(1 / self.SAMPLE_RATE)))
        self.init_metrics()
        self.test_cache_name = None
        self.test_cache = None
        self.test_cache_mode = "create"

    def set_cache(self, name: str, mode='create'):
        from src.hdf_cache import TestStepHDFCache
        self.test_cache_name = name
        self.test_cache_mode = mode
        self.test_cache = TestStepHDFCache(name, 'a')

    def set_integration_method(self, n_steps=None, method=None):
        self.flow_steps = n_steps or self.flow_steps
        self.solve_method = method or self.solve_method
        logger.debug("Integration method set to %s with %s steps", self.solve_method, self.flow_steps)

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
        """Create a velocity training sample by sampling the prior once and interpolating, then calculate the loss."""
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
        t_broadcast = t.view(-1, 1, 1)
        samples_at_t = prior_samples * (1 - t_broadcast) + target_samples * t_broadcast
        target_velocity = target_samples - prior_samples
        return t, samples_at_t, target_velocity, conditioning_inputs

    def get_prior_samples(self, conditioning_inputs, target_size: torch.Size):
        """Sample priors either around the mean of 0.5 or starting connected to the last value of Wh."""
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
            case "constant":
                prior_samples = torch.zeros(target_size, device=self.device) + 0.5
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

    def init_metrics(self):
        self.moments_metrics = metrics.MomentsErrorsMetric()
        self.mode_test_metrics = torchmetrics.MetricCollection(
            dict(
                any_Wh=ModeTransitionMetric('any_Wh'),
                mixed=ModeTransitionMetric('mixed'),
                L_only_Wh=ModeTransitionMetric('L_only_Wh'),
                D_only_Wh=ModeTransitionMetric('D_only_Wh'),
                H_only_Wh=ModeTransitionMetric('H_only_Wh'),
                L_in_Wh=ModeTransitionMetric('L_in_Wh'),
                D_in_Wh=ModeTransitionMetric('D_in_Wh'),
                H_in_Wh=ModeTransitionMetric('H_in_Wh'),
                L_not_in_Wh=ModeTransitionMetric('L_not_in_Wh'),
                D_not_in_Wh=ModeTransitionMetric('D_not_in_Wh'),
                H_not_in_Wh=ModeTransitionMetric('H_not_in_Wh'),
            ),
            prefix="/mode/",
        )
        self.dice_metric = DiceScore(3, input_format='index')
        self.peak_metrics = torchmetrics.MetricCollection(
            dict(
                any_Wh=PeakMetric('any_Wh'),
                mixed=PeakMetric('mixed'),
                L_only_Wh=PeakMetric('L_only_Wh'),
                D_only_Wh=PeakMetric('D_only_Wh'),
                H_only_Wh=PeakMetric('H_only_Wh'),
            ),
            prefix="/",
        )
        # self.mode_metrics = mode_metrics_collection
        # self.train_metrics = mode_metrics_collection.clone(prefix='train/')
        # self.mode_test_metrics = mode_metrics_collection.clone(prefix='test/')

    # def predict_step():
    #     pass

    def test_step(self, batch: tuple[dict, dict, torch.Tensor], batch_idx: int = -1):
        data_module = self.trainer.datamodule
        meta, conditioning_input, target_samples = batch
        generated_samples, surr_labels_pred, surr_labels_target = self.inference(batch, data_module)

        metrics_out = self.update_metrics(
            generated_samples, target_samples, conditioning_input, surr_labels_pred, surr_labels_target, data_module
        )
        metrics_out['_step'] = batch_idx
        self.log_dict(metrics.prefix_metrics(metrics_out, 'test/step'), prog_bar=True, on_step=True, on_epoch=False)
        return generated_samples, surr_labels_pred, surr_labels_target, metrics_out

    @torch.inference_mode()
    def inference(self, batch, data_module):
        meta, conditioning_input, target_samples = batch
        if self.test_cache is not None and self.test_cache_mode == 'use':
            generated_samples, surr_labels_pred, surr_labels_target = self.test_cache.get(
                meta['shot_number'].cpu(), meta['start_i'].cpu()
            )
            generated_samples = torch.tensor(generated_samples, device=self.device, dtype=torch.float32)
        else:
            prior_samples = self.get_prior_samples(conditioning_input, target_samples.size())
            generated_samples: torch.Tensor = self.integrate_path(
                prior_samples,
                conditioning_input=conditioning_input,
                n_steps=self.flow_steps,
                method=self.solve_method,
                save_trajectories=False
            )  # type: ignore
            surr_labels_pred, surr_labels_target = generate_surrogate_labels_batched(
                meta, generated_samples, target_samples, data_module=data_module
            )  # both pred and target have shape B, Wh+Wf, and
            if self.test_cache is not None and self.test_cache_mode == 'create':
                self.test_cache.set_from_batch(
                    meta['shot_number'].cpu(), meta['start_i'].cpu(), generated_samples.cpu(), surr_labels_pred,
                    surr_labels_target
                )

        return generated_samples, surr_labels_pred, surr_labels_target

    def update_metrics(
        self, generated_samples, target_samples, conditioning_input, surr_labels_pred, surr_labels_target, data_module
    ):
        pred_labels = torch.tensor(surr_labels_pred, device=self.device, dtype=torch.int)
        target_labels = torch.tensor(surr_labels_target, device=self.device, dtype=torch.int)

        # Metrics
        metrics_out = self.moments_metrics(generated_samples, target_samples)
        peak_metrics_out = self.peak_metrics(generated_samples, target_samples, conditioning_input['label'] - 1)
        metrics_out |= peak_metrics_out
        # label metrics:

        metrics_out |= self.mode_test_metrics(pred_labels, target_labels)  # requires full W to split off history itself
        # Ensure both inputs are long tensors with class indices for DiceScore
        WINDOW_OF_INFLUENCE_SPILL = 15  # the surrogate model looks ahead 15 steps past where it assigns a label.
        Wf_length = data_module.seq_length
        metrics_out['/dice'] = self.dice_metric(
            pred_labels[:, -Wf_length - WINDOW_OF_INFLUENCE_SPILL:].long(),
            target_labels[:, -Wf_length - WINDOW_OF_INFLUENCE_SPILL:].long()
        )

        return metrics_out

    def on_test_epoch_end(self):
        test_metrics = self.moments_metrics.compute()
        test_metrics |= self.mode_test_metrics.compute()
        test_metrics |= self.peak_metrics.compute()
        test_metrics['/dice'] = self.dice_metric.compute()
        epoch_metrics = metrics.prefix_metrics(test_metrics, 'test/final')
        self.log_dict(epoch_metrics, on_step=False, on_epoch=True)

        for sub_metric in self.peak_metrics.children():
            sub_metric.extract_df_all(self.test_cache)
        for sub_metric in self.mode_test_metrics.children():
            sub_metric.extract_df_all(self.test_cache)
        self.moments_metrics.reset()
        self.mode_test_metrics.reset()
        self.dice_metric.reset()

    @torch.inference_mode()
    def evaluate(
        self,
        batch: tuple[dict, dict, torch.Tensor],
        n_steps=50,
        solve_method="rk4",
        data_module: Optional[L.LightningDataModule] = None,
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
        self.model.eval()
        # Use lightnings manner of moving to correct current device:
        meta, conditioning_input, target_samples = self._apply_batch_transfer_handler(batch)
        logger.debug("Evaluating batch shape %s", target_samples.shape)
        prior_samples = self.get_prior_samples(conditioning_input, target_samples.size())
        generated_samples, trajectories = self.integrate_path(
            prior_samples,
            conditioning_input=conditioning_input,
            n_steps=n_steps,
            method=solve_method,
            save_trajectories=True
        )
        # surrogate labels
        if data_module is None:
            data_module = self.trainer.datamodule
        surr_labels_pred, surr_labels_target = generate_surrogate_labels_batched(
            meta, generated_samples, target_samples, data_module=data_module
        )
        self.model.train()  # Reset model to training mode
        # Metrics

        metrics_out = self.update_metrics(
            generated_samples, target_samples, conditioning_input, surr_labels_pred, surr_labels_target, data_module
        )
        # metrics_out = metrics.get_moments_errors_per_channel(generated_samples, target_samples)
        # meta, conditioning_input, target_samples, prior_samples, generated_samples, trajectories = self._apply_batch_transfer_handler(
        #     (meta, conditioning_input, target_samples, prior_samples, generated_samples, trajectories),
        #     device='cpu'  # type: ignore
        # )
        # TODO do metric calculation elsewhere
        metrics_out |= metrics.get_entropy_metrics(generated_samples, target_samples)
        peak_metrics, peak_features = metrics.get_peak_metrics(generated_samples, target_samples)
        self.init_metrics() # reset everything
        return dict(
            meta=meta,
            conditioning_input=conditioning_input,
            target_samples=target_samples,
            prior_samples=prior_samples,
            generated_samples=generated_samples,
            trajectories=trajectories,
            metrics=metrics_out,
            peak_features=peak_features,
            surr_labels_pred=surr_labels_pred,
            surr_labels_target=surr_labels_target
        )

    @staticmethod
    @torch.no_grad()
    def fwd_euler_step(ode_func, current_points, current_t, dt):
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
        velocity = ode_func(current_t, current_points)
        return current_points + velocity * dt

    @torch.inference_mode()
    def integrate_path(
        self,
        initial_points,
        conditioning_input=None,
        method='simple',  # Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
        n_steps=100,
        save_trajectories=False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate a path using the given step function.

        Args:
            initial_points (torch.Tensor): Shape [batch_size, num_features]
            step_fn: The function to use for computing the next step.
            n_steps (int): The number of steps to integrate.
            save_trajectories (bool): Whether to save the trajectories.

        Returns:
            torch.Tensor: Shape [batch_size, num_features]
            torch.Tensor (optional): Shape [n_steps, batch_size, num_features] if save_trajectories is True
        """
        current_points = initial_points.clone()
        if conditioning_input is not None and self.model.conditioning:

            def ode_func(t, x):
                return self.model(x=x.to(torch.float32), t=t.to(torch.float32), conditioning_input=conditioning_input)
        else:

            def ode_func(t, x):  # just swap around for torchdiffeq
                return self.model(x=x.to(torch.float32), t=t.to(torch.float32))

        time_grid = torch.linspace(0, 1, n_steps, device=self.device, dtype=torch.float64)

        logger.debug(f"Integrating path with {n_steps} steps with method {method}")
        # logger.debug(
        #     "Devices: timesteps: %s, current_points: %s, conditioning_input: %s", time_grid.device,
        #     current_points.device, {
        #         k: v.device for k, v in conditioning_input.items()
        #     } if conditioning_input is not None else None
        # )
        if method == "simple":
            # Integrate and use progress bar if running on CPU
            if save_trajectories:
                trajectories = [current_points]
            for i in tqdm(
                range(len(time_grid) - 1),
                disable=self.device.type != "cpu",
                desc="Integrating path",
            ):
                current_points = self.fwd_euler_step(
                    ode_func, current_points, time_grid[i], time_grid[i + 1] - time_grid[i]
                )
                if save_trajectories:
                    trajectories.append(current_points)
            logger.debug("Solved with %s steps.", len(time_grid))
            if save_trajectories:
                return current_points, torch.stack(trajectories)
            return current_points
        else:  # use torchdiffeq
            sol = odeint(
                ode_func,
                current_points,
                time_grid,
                method=method,
                options={'max_num_steps': 2000} if method == 'dopri5' else {},
                # atol=atol,
                # rtol=rtol,
            )
            logger.debug("Solved with %s steps.", len(sol))
            if save_trajectories:
                return sol[-1], sol
            else:
                return sol[-1]

    # def integrate_path_advanced(self, )

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
        self.clip_gradients(optimizer, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")

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
                (config.batch_size, config.model.params.model_params.input_channels, config.data.history_length),
            'position_sequence': (config.batch_size, config.data.history_length + config.data.seq_length),
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
