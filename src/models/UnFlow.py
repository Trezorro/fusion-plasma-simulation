from src.models.flow import FlowModule
import torch
import src.metrics.metrics as metrics
from src.metrics.evaluate_modes import generate_surrogate_labels_batched

import logging

logger = logging.getLogger(__name__)



class UnFlowModule(FlowModule):
    """
    Ablation: Directly predicts the output window from prior sample and conditioning, with t=1, no flow/interpolation.
    """

    def forward(self, x, t, conditioning_input=None):
        # Standard UNet interface: x, t, conditioning_input
        return self.model(x, t, conditioning=conditioning_input)

    def training_step(self, batch, batch_idx):
        opt: LightningOptimizer = self.optimizers()  # type: ignore
        total_loss = 0
        opt.zero_grad()
        for match_i in range(1, self.batch_rematch_factor + 1):
            # this is a batch matched with one sample from the prior:
            meta, conditioning_input, target_samples = batch
            noise_sample = self.get_prior_samples(conditioning_input, target_samples.size())
            t_dummy = torch.ones(target_samples.size(0), device=self.device)  # constant t=1
            pred = self.model(noise_sample, t_dummy, conditioning_input=conditioning_input)
            loss = self.loss(pred, target_samples)
            if loss.isnan().any():
                raise ValueError(f"Loss is NaN: {loss}")
            self.manual_backward(loss)
            total_loss += loss.detach()
            if match_i % self.step_every_nth_match == 0:
                # See: on_before_optimizer_step() for gradient clipping
                opt.step()
                opt.zero_grad()
        total_loss /= self.batch_rematch_factor
        self.log("loss/train", total_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        meta, conditioning_input, target_samples = batch
        noise_sample = self.get_prior_samples(conditioning_input, target_samples.size())
        t_dummy = torch.ones(target_samples.size(0), device=self.device)
        pred = self.model(noise_sample, t_dummy, conditioning_input=conditioning_input)
        loss = self.loss(pred, target_samples)
        self.log("loss/val", loss, prog_bar=True)
        return loss

    @torch.inference_mode()
    def inference(self, batch, data_module):
        meta, conditioning_input, target_samples = batch
        if self.test_cache is not None and self.test_cache_mode == 'use':
            generated_samples, surr_labels_pred, surr_labels_target = self.test_cache.get(
                meta['shot_number'].cpu(), meta['start_i'].cpu()
            )
            generated_samples = torch.tensor(generated_samples, device=self.device, dtype=torch.float32)
        else:
            noise_sample = self.get_prior_samples(conditioning_input, target_samples.size())
            t_dummy = torch.ones(target_samples.size(0), device=self.device)
            generated_samples = self.model(noise_sample, t_dummy, conditioning_input=conditioning_input)
            surr_labels_pred, surr_labels_target = generate_surrogate_labels_batched(
                meta, generated_samples, target_samples, data_module=data_module
            )  # both pred and target have shape B, Wh+Wf, and
            if self.test_cache is not None and self.test_cache_mode == 'create':
                self.test_cache.set_from_batch(
                    meta['shot_number'].cpu(), meta['start_i'].cpu(), generated_samples.cpu(), surr_labels_pred, surr_labels_target
                )

        return generated_samples, surr_labels_pred, surr_labels_target

    @torch.inference_mode()
    def evaluate(self, batch, n_steps=1, data_module=None, **kwargs):
        self.model.eval()
        meta, conditioning_input, target_samples = self._apply_batch_transfer_handler(batch, device=self.device)
        logger.debug(
            "Evaluating batch shape %s at target_samples.device %s", target_samples.shape, target_samples.device
        )
        logger.debug(
            "UnFlowModule on device %s", self.device
        )
        for k, v in (meta | conditioning_input).items():
            logger.debug("Meta/conditioning input %s on device %s", k, v.device if isinstance(v, torch.Tensor) else "Not tensor!")
        prior_samples = self.get_prior_samples(conditioning_input, target_samples.size())
        t_dummy = torch.ones(target_samples.size(0), device=self.device)
        logger.debug("prior.device %s", prior_samples.device)
        generated_samples = self.model(prior_samples, t_dummy, conditioning_input=conditioning_input)
        if data_module is None:
            data_module = getattr(self.trainer, 'datamodule', None)
            if data_module is None:
                raise RuntimeError("No datamodule found on trainer.")
        surr_labels_pred, surr_labels_target = generate_surrogate_labels_batched(
            meta, generated_samples, target_samples, data_module
        )
        self.model.train()  # Reset model to training mode
        metrics_out = self.update_metrics(
            generated_samples, target_samples, conditioning_input, surr_labels_pred, surr_labels_target, data_module
        )
        meta, conditioning_input, target_samples, prior_samples, generated_samples = self._apply_batch_transfer_handler(
            (meta, conditioning_input, target_samples, prior_samples, generated_samples),
            device='cpu'  # type: ignore
        )
        metrics_out |= metrics.get_entropy_metrics(generated_samples, target_samples)
        _peak_metrics, peak_features = metrics.cpu_batch_peak_metrics(generated_samples, target_samples)
        self.init_metrics() # reset everything
        trajectories = torch.stack((prior_samples, generated_samples))
        return dict(
            meta=meta,
            conditioning_input=conditioning_input,
            target_samples=target_samples,
            prior_samples=prior_samples,
            generated_samples=generated_samples,
            metrics=metrics_out,
            trajectories=trajectories,
            peak_features=peak_features,
            surr_labels_pred=surr_labels_pred,
            surr_labels_target=surr_labels_target
        )
