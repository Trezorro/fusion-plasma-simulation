from typing import Optional
import lightning as L
import torch
import torch.nn as nn
import torchinfo
import torchmetrics
import wandb
from omegaconf import DictConfig

from src.fourier import FourierMSLE, FrequencySpectrumMSESimple, FrequencyPhaseAmpMSE, FrequencyAmpMSE


class MLP(nn.Sequential):

    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128]):
        super().__init__()
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            self.add_module(
                f'linear_{current_dim}_{hidden_dim}', nn.Linear(current_dim, hidden_dim, dtype=torch.float)
            )
            self.add_module(f'activation_{hidden_dim}', nn.ReLU())
            current_dim = hidden_dim
        self.add_module(
            f'linear_{current_dim}_{output_dim}', nn.Linear(current_dim, output_dim, dtype=torch.float)
        )


class ComplexReLU(nn.Module):

    def forward(self, input):
        return torch.complex(torch.relu(input.real), torch.relu(input.imag))


class ComplexMLP(nn.Sequential):

    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128]):
        super(ComplexMLP, self).__init__()
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            self.add_module(
                f'linear_{current_dim}_{hidden_dim}', nn.Linear(current_dim, hidden_dim, dtype=torch.cfloat)
            )
            self.add_module(f'activation_{hidden_dim}', ComplexReLU())
            current_dim = hidden_dim
        self.add_module(
            f'linear_{current_dim}_{output_dim}', nn.Linear(current_dim, output_dim, dtype=torch.cfloat)
        )


class FakeComplexMLP(nn.Sequential):
    """A fake complex MLP that treats complex numbers as two real numbers. All given dimensions are doubled.

    Works well with AmpPhase loss.
    """

    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            self.add_module(
                f'linear_{current_dim}*2_{hidden_dim}*2',
                nn.Linear(current_dim * 2, hidden_dim * 2, dtype=torch.float)
            )
            self.add_module(f'activation_{hidden_dim}*2', nn.ReLU())
            current_dim = hidden_dim
        self.add_module(
            f'linear_{current_dim}*2_{output_dim}*2',
            nn.Linear(current_dim * 2, output_dim * 2, dtype=torch.float)
        )

    def forward(self, complex_input):
        as_real = torch.concat((complex_input.real, complex_input.imag), dim=1)
        real_output = super().forward(as_real)
        return torch.complex(real_output[:, :self.output_dim], real_output[:, self.output_dim:])


class PolarComplexMLP(FakeComplexMLP):

    def forward(self, complex_input):
        as_polar = torch.concat((torch.log(complex_input.abs()), complex_input.angle()), dim=1)
        real_output = super(FakeComplexMLP, self).forward(as_polar)
        magnitude_log_space = real_output[:, :self.output_dim]
        # magnitude = torch.nn.Identity()(magnitude_log_space)  # Was torch.exp, but didn't learn.
        magnitude = torch.exp(magnitude_log_space)  # Was torch.exp, but didn't learn.
        angle_linear = real_output[:, self.output_dim:]
        angle = torch.tanh(angle_linear) * torch.pi
        return torch.polar(magnitude, angle)


class ComplexNet(L.LightningModule):

    TIME_DOMAIN_LOSS = torchmetrics.MeanAbsoluteError
    LOSS_OPTIONS = dict(
        MSELoss=torch.nn.MSELoss,
        L1Loss=torch.nn.L1Loss,
        FrequencySpectrumMSESimple=FrequencySpectrumMSESimple,
        FrequencyPhaseAmpMSE=FrequencyPhaseAmpMSE,
        FrequencyAmpMSE=FrequencyAmpMSE,
    )
    MODEL_OPTIONS = dict(
        MLP=MLP,
        ComplexMLP=ComplexMLP,
        FakeComplexMLP=FakeComplexMLP,
        PolarComplexMLP=PolarComplexMLP,
    )

    def __init__(
        self,
        c_channels: int,
        x_channels: int,
        out_channels: int,
        mlp_hidden_dims: list,
        warmup_window: int,
        forecast_window: int,
        model: str = "ComplexMLP",
        loss: str = "MSELoss",
        use_polar_pre_split: bool = False,
        mlp_activation: str = 'ReLU',
        output_activation: str = "Softplus",
        optimizer_params: Optional[dict] = None,
        **kwargs
    ):
        super().__init__()
        self.c_channels = c_channels
        self.x_channels = x_channels
        self.cx_channels = c_channels + x_channels
        self.out_channels = out_channels
        self.forecast_window = forecast_window
        self.forecast_window_freqs = (forecast_window // 2 +
                                      1) * (1 + use_polar_pre_split)  # double for magnitude angle pre-split
        self.warmup_window = warmup_window
        self.warmup_window_freqs = (warmup_window // 2 + 1) * (1 + use_polar_pre_split)
        self.val_rollout = forecast_window
        self.train_rollout = forecast_window
        self.loss = ComplexNet.LOSS_OPTIONS[loss]()
        self.loss_time_domain_train = self.TIME_DOMAIN_LOSS()
        self.loss_time_domain_val = self.TIME_DOMAIN_LOSS()
        self.optimizer_params = optimizer_params
        self.use_polar_pre_split = use_polar_pre_split
        self.net = ComplexNet.MODEL_OPTIONS[model](
            input_dim=(self.cx_channels) * self.warmup_window_freqs + c_channels * self.forecast_window_freqs,
            output_dim=out_channels * self.forecast_window_freqs,
            hidden_dims=mlp_hidden_dims
        )
        match output_activation.lower():
            case 'relu':
                self.out_activation = nn.ReLU()
            case 'softplus':
                self.out_activation = nn.Softplus(beta=2)
            case 'exp':
                self.out_activation = torch.exp
            case _:
                self.out_activation = nn.Identity()

    def forward(self, input_frequencies: torch.Tensor):
        """Predicts the window x_out_frequencies, which will be length T // 2 + 1.

        Inputs and outputs are in the shape (batch_size, variables, frequencies).
        """
        flattened_in_freqs = input_frequencies.reshape(input_frequencies.size(0), -1)
        x_out = self.net(flattened_in_freqs)
        x_out_flat = self.out_activation(x_out)  # (batch_size, variables x * forecast_window)
        x_frequencies_pred = x_out_flat.reshape(
            x_out_flat.size(0), self.out_channels, self.forecast_window_freqs
        )
        return x_frequencies_pred  # Returned as (batch_size, variables x, seq length)

    def split_and_prep_batch(self, batch):
        """Use self.forecast_window to split the batch into input and output, across C and X."""
        shot_number, controls, observables = batch
        c_in = controls[:, :, :-self.forecast_window].to(self.device)
        c_out = controls[:, :, -self.forecast_window:].to(self.device)
        x_in = observables[:, :, :c_in.size(2)].to(self.device)
        concat_xc = torch.cat((x_in, c_in, c_out), dim=1)  # (batch_size, variables c + x, seq_length)
        input_xc_freq = torch.fft.rfft(concat_xc, dim=2)
        if self.use_polar_pre_split:
            input_xc_mag = torch.log(input_xc_freq.abs() + 1)
            input_xc_freq = torch.concat((input_xc_mag, input_xc_freq.angle()), dim=2)

        if observables.size(2) < controls.size(2):
            return input_xc_freq, None, None  # for prediction without target
        # Else, we have a target
        x_target_t = observables[:, :, -self.forecast_window:].to(self.device)
        assert x_in.size(2) == c_in.size(2)
        assert x_target_t.size(2) == c_out.size(2)
        x_target_freq = torch.fft.rfft(x_target_t, dim=2)
        if self.use_polar_pre_split:
            x_target_mag = torch.log(x_target_freq.abs() + 1)
            x_target_freq = torch.concat(
                (x_target_mag, x_target_freq.angle()), dim=2
            )  # (batch_size, variables x, freq_bins*2)
        return input_xc_freq, x_target_freq, x_target_t

    def reverse_pre_split_to_complex(self, mag_angle_result):
        """Reverse the splitting of magnitude and angle."""
        num_actual_freqs = mag_angle_result.size(2) // 2
        mag = torch.exp(mag_angle_result[:, :, :num_actual_freqs]) - 1
        angle = mag_angle_result[:, :, num_actual_freqs:]
        return torch.polar(mag, angle)

    def prediction_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction function that skips loss computation."""
        # Do time split here to support autoregressive processing later. Don't do it in data loader.
        input_xc_freq, x_target_freq, x_target_t = self.split_and_prep_batch(batch)
        x_pred_freq = self(input_xc_freq)
        if self.use_polar_pre_split:
            x_pred_freq = self.reverse_pre_split_to_complex(x_pred_freq)
        x_pred_t = torch.fft.irfft(x_pred_freq, dim=2)
        return x_pred_t, x_pred_freq, x_target_t, x_target_freq

    def training_step(self, batch, batch_idx):
        input_xc_freq, x_target_freq, x_target_t = self.split_and_prep_batch(batch)
        x_pred_freq = self(input_xc_freq)  # call model forward
        loss = self.loss(x_pred_freq, x_target_freq)
        self.log("loss/train", loss, prog_bar=True)
        # Optional time domain loss:
        if self.use_polar_pre_split:
            x_pred_freq = self.reverse_pre_split_to_complex(x_pred_freq.detach())
        x_pred_t = torch.fft.irfft(x_pred_freq, dim=2)
        self.loss_time_domain_train(x_pred_t, x_target_t)
        self.log("loss/time_domain_train", self.loss_time_domain_train, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_xc_freq, x_target_freq, x_target_t = self.split_and_prep_batch(batch)
        x_pred_freq = self(input_xc_freq)  # call model forward
        loss = self.loss(x_pred_freq, x_target_freq)
        if self.use_polar_pre_split:
            x_pred_freq = self.reverse_pre_split_to_complex(x_pred_freq)
        x_pred_t = torch.fft.irfft(x_pred_freq, dim=2)
        self.log("loss/val", loss, prog_bar=True)
        self.loss_time_domain_val(x_pred_t, x_target_t)
        self.log("loss/time_domain_val", self.loss_time_domain_val, prog_bar=True)
        return dict(loss=loss, outputs=x_pred_t)

    test_step = validation_step

    def evaluate(self, batch):
        """Return losses and target and prediction outputs, as used by the model, for a batch."""
        self.eval()
        with torch.inference_mode():
            input_xc_freq, x_target_freq, x_target_t = self.split_and_prep_batch(batch)
            x_pred_freq = self(input_xc_freq)  # call model forward
            loss = self.loss(x_pred_freq, x_target_freq)
            if self.use_polar_pre_split:
                x_pred_freq = self.reverse_pre_split_to_complex(x_pred_freq)
                x_target_freq = self.reverse_pre_split_to_complex(x_target_freq)
            x_pred_t = torch.fft.irfft(x_pred_freq, dim=2)
            time_domain_loss = self.TIME_DOMAIN_LOSS().to(self.device)(x_pred_t, x_target_t)
            losses = dict(loss=loss, time_domain_loss=time_domain_loss)
            outputs = dict(
                # input_xc_freq=input_xc_freq,
                x_pred_freq=x_pred_freq,
                x_pred_t=x_pred_t,
                x_target_freq=x_target_freq,
                x_target_t=x_target_t
            )
        self.train()
        return losses, outputs

    def configure_optimizers(self):
        if self.optimizer_params is None:
            self.optimizer_params = dict()
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_params)
        return optimizer

    def log_summary(self, config: DictConfig):
        summary = torchinfo.summary(
            self,
            input_size=(
                2,
                self.c_channels * 2 + self.x_channels,
                self.forecast_window_freqs,
            ),
            dtypes=[torch.float if self.use_polar_pre_split else torch.cfloat],
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
                "model/trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
                # "model/minimum_input_length": self.convnet.minimum_input_length,
            },
            step=0
        )


if __name__ == '__main__':
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from src.config import load_config_from_file

    C = load_config_from_file(as_omega=True)
    model = ComplexNet(**C['model']['params'])
    model.log_summary(C)
