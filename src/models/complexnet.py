import lightning as L
import torch
import torch.nn as nn
import torchinfo
import torchmetrics
import wandb
from omegaconf import DictConfig

from src.fourier import FourierMSLE, FrequencySpectrumMSESimple


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


class ComplexNet(L.LightningModule):

    TIME_DOMAIN_LOSS = torchmetrics.MeanSquaredLogError

    def __init__(
        self,
        c_channels: int,
        x_channels: int,
        out_channels: int,
        mlp_hidden_dims: list,
        warmup_window: int,
        forecast_window: int,
        loss: str = "MSELoss",
        mlp_activation: str = 'ReLU',
        output_activation: str = "Softplus",
        optimizer_params: dict = {},
        **kwargs
    ):
        super().__init__()
        self.c_channels = c_channels
        self.x_channels = x_channels
        self.cx_channels = c_channels + x_channels
        self.out_channels = out_channels
        self.forecast_window = forecast_window
        self.forecast_window_freqs = forecast_window // 2 + 1
        self.warmup_window = warmup_window
        self.warmup_window_freqs = warmup_window // 2 + 1
        self.val_rollout = forecast_window
        self.train_rollout = forecast_window
        self.loss = FrequencySpectrumMSESimple()
        self.loss_time_domain_train = self.TIME_DOMAIN_LOSS()
        self.loss_time_domain_val = self.TIME_DOMAIN_LOSS()
        self.optimizer_params = optimizer_params
        self.net = ComplexMLP(
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

    def compute_frequency_loss(self, batch):
        c_in, c_out, x_in, x_out_true = self.split_batch_data(batch)
        concat_xc = torch.cat((x_in, c_in, c_out), dim=1)  # (batch_size, variables c + x, seq_length)
        input_frequencies = torch.fft.rfft(concat_xc, dim=2)
        output_frequencies = torch.fft.rfft(x_out_true, dim=2)
        x_out_pred_freqs = self(input_frequencies)
        x_out_pred_time = torch.fft.irfft(x_out_pred_freqs, dim=2)
        loss = self.loss(x_out_pred_freqs, output_frequencies)
        return loss, x_out_true, x_out_pred_time

    def prediction_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction function that skips loss computation."""
        # Do time split here to support autoregressive processing later. Don't do it in data loader.
        c_in, c_out, x_in, _ = self.split_batch_data(batch)
        concat_xc = torch.cat((x_in, c_in, c_out), dim=1)  # (batch_size, variables c + x, seq_length)
        input_frequencies = torch.fft.rfft(concat_xc, dim=2)
        x_out_pred_freqs = self(input_frequencies)
        x_out_pred_time = torch.fft.irfft(x_out_pred_freqs, dim=2)
        return x_out_pred_time, x_out_pred_freqs

    def training_step(self, batch, batch_idx):
        loss, x_out_true, x_out_pred_time = self.compute_frequency_loss(
            batch
        )  # Main loss in frequency domain
        self.loss_time_domain_train(x_out_pred_time, x_out_true)
        self.log("loss/train", loss, prog_bar=True)
        self.log(
            "loss/time_domain_train",
            self.loss_time_domain_train,
            prog_bar=True,
            on_epoch=True,
            on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, x_out_true, x_out_pred_time = self.compute_frequency_loss(
            batch
        )  # Main loss in frequency domain

        self.log("loss/val", loss, prog_bar=True)
        self.loss_time_domain_val(x_out_pred_time, x_out_true)
        self.log(
            "loss/time_domain_val", self.loss_time_domain_val, prog_bar=True, on_epoch=True, on_step=True
        )
        return dict(loss=loss, outputs=x_out_pred_time)

    test_step = validation_step

    def evaluate(self, batch):
        self.eval()
        with torch.inference_mode():
            loss, x_out_true, x_out_pred_time = self.compute_frequency_loss(batch)
            time_domain_loss = self.TIME_DOMAIN_LOSS().to(self.device)(x_out_pred_time, x_out_true)
            losses = dict(loss=loss, time_domain_loss=time_domain_loss)
        self.train()
        return x_out_pred_time, losses

    def split_batch_data(self, batch):
        """Use self.forecast_window to split the batch into input and output, across C and X."""
        shot_number, controls, observables = batch
        c_in = controls[:, :, :-self.forecast_window].to(self.device)
        c_out = controls[:, :, -self.forecast_window:].to(self.device)
        if observables.size(2) == controls.size(2):
            x_in = observables[:, :, :-self.forecast_window].to(self.device)
            x_out = observables[:, :, -self.forecast_window:].to(self.device)
            assert x_in.size(2) == c_in.size(2)
            assert x_out.size(2) == c_out.size(2)
        else:
            # support pre-masked input
            x_in = observables[:, :, :c_in.size(2)].to(self.device)
            x_out = None
        return c_in, c_out, x_in, x_out

    def configure_optimizers(self):
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
            dtypes=[torch.cfloat],
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
