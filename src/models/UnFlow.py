from src.models.flow import FlowModule
import torch
import src.metrics.metrics as metrics
from src.metrics.evaluate_modes import generate_surrogate_labels_batched


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

    def test_step(self, batch, batch_idx):
        meta, conditioning_input, target_samples = batch
        noise_sample = self.get_prior_samples(conditioning_input, target_samples.size())
        t_dummy = torch.ones(target_samples.size(0), device=self.device)
        pred = self.model(noise_sample, t_dummy, conditioning_input=conditioning_input)
        # Fallback for datamodule
        data_module = getattr(self.trainer, 'datamodule', None)
        if data_module is None:
            raise RuntimeError("No datamodule found on trainer.")
        Wf_length = data_module.seq_length
        surr_labels_pred, surr_labels_target = generate_surrogate_labels_batched(
            meta, pred, target_samples, data_module
        )
        surr_labels_pred = torch.tensor(surr_labels_pred, device=self.device, dtype=torch.int)
        surr_labels_target = torch.tensor(surr_labels_target, device=self.device, dtype=torch.int)
        metrics_out = self.moments_metrics(pred, target_samples)
        metrics_out |= self.mode_test_metrics(surr_labels_pred, surr_labels_target)
        pred_labels = surr_labels_pred[:, -Wf_length:].long()
        target_labels = surr_labels_target[:, -Wf_length:].long()
        metrics_out['/dice'] = self.dice_metric(pred_labels, target_labels)
        self.log_dict(metrics.prefix_metrics(metrics_out, 'test'), prog_bar=True, on_step=True, on_epoch=False)

    @torch.inference_mode()
    def evaluate(self, batch, n_steps=1, warp_fn=None, data_module=None):
        self.model.eval()
        meta, conditioning_input, target_samples = self._apply_batch_transfer_handler(batch)
        prior_samples = self.get_prior_samples(conditioning_input, target_samples.size())
        t = torch.ones(target_samples.size(0), device=self.device)
        generated_samples = self.model(prior_samples, t, conditioning_input=conditioning_input)
        metrics_out = metrics.get_moments_errors_per_channel(generated_samples, target_samples)

        if data_module is None:
            data_module = getattr(self.trainer, 'datamodule', None)
            if data_module is None:
                raise RuntimeError("No datamodule found on trainer.")
        surr_labels_pred, surr_labels_target = generate_surrogate_labels_batched(
            meta, generated_samples, target_samples, data_module
        )
        meta, conditioning_input, target_samples, prior_samples, generated_samples = self._apply_batch_transfer_handler(
            (meta, conditioning_input, target_samples, prior_samples, generated_samples), device=torch.device('cpu'))
        metrics_out |= metrics.get_entropy_metrics(generated_samples, target_samples)
        peak_metrics, peak_features = metrics.get_peak_metrics(generated_samples, target_samples)
        trajectories = torch.stack((prior_samples, generated_samples))
        return dict(
            meta=meta,
            conditioning_input=conditioning_input,
            target_samples=target_samples,
            prior_samples=prior_samples,
            generated_samples=generated_samples,
            metrics=metrics_out | peak_metrics,
            trajectories=trajectories,
            peak_features=peak_features,
            surr_labels_pred=surr_labels_pred,
            surr_labels_target=surr_labels_target
        )
