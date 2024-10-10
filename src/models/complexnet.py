import lightning as L
import torch
import torch.nn as nn
import torchinfo
import wandb
from omegaconf import DictConfig

from src.fourier import FourierMSLE


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
        self.forecast_window = forecast_window
        self.forecast_window_freqs = forecast_window // 2 + 1
        self.warmup_window = warmup_window
        self.warmup_window_freqs = warmup_window // 2 + 1
        self.val_rollout = forecast_window
        self.train_rollout = forecast_window
        self.cx_channels = c_channels + x_channels
        self.out_channels = out_channels
        self.loss = getattr(torch.nn, loss)()
        self.fourier_loss_train = FourierMSLE()
        self.fourier_loss_val = FourierMSLE()
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

    def forward(self, x_in: torch.Tensor, c_in: torch.Tensor, c_out: torch.Tensor):
        """Predicts the window x_out, which will be the same length as the given c_out.

        Inputs and outputs are in the shape (batch_size, seq_length, channels).
        """
        with torch.no_grad():
            concat_xc = torch.cat((x_in, c_in, c_out), dim=2)  # (batch_size, seq_length, variables c + x)
            # put channel dimension second
            concat_xc = concat_xc.permute(0, 2, 1)  # (batch_size, variables c + x, seq_length)
            input_frequencies = torch.fft.rfft(concat_xc, dim=2)
            flattened_in_freqs = input_frequencies.reshape(input_frequencies.size(0), -1)

        x_out = self.net(flattened_in_freqs)
        x_out = self.out_activation(x_out)  # (batch_size, variables x * forecast_window)
        x_out = x_out.reshape(x_out.size(0), self.out_channels, self.forecast_window_freqs)
        x_out = torch.fft.irfft(x_out, dim=2)
        return x_out.permute(0, 2, 1)  # Returned as (batch_size, seq_length, variables x)

    def training_step(self, batch, batch_idx):
        c_in, c_out, x_in, x_out = self.split_batch_data(batch)
        x_out_pred = self(x_in, c_in, c_out)
        loss = self.loss(x_out_pred, x_out)
        self.log("loss/train", loss, prog_bar=True)
        self.fourier_loss_train(x_out_pred, x_out)
        self.log("fourier_loss/train", self.fourier_loss_train, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        c_in, c_out, x_in, x_out = self.split_batch_data(batch)
        x_out_pred = self(x_in, c_in, c_out)
        loss = self.loss(x_out_pred, x_out)
        self.log("loss/val", loss, prog_bar=True)
        self.fourier_loss_val(x_out_pred, x_out)
        self.log("fourier_loss/val", self.fourier_loss_val, prog_bar=True, on_epoch=True, on_step=True)
        return dict(loss=loss, outputs=x_out_pred)

    test_step = validation_step

    def split_batch_data(self, batch):
        shot_number, controls, observables = batch
        c_in = controls[:, :-self.forecast_window]
        c_out = controls[:, -self.forecast_window:]
        if observables.size(1) == controls.size(1):
            x_in = observables[:, :-self.forecast_window]
            x_out = observables[:, -self.forecast_window:]
            assert x_in.size(1) == c_in.size(1)
            assert x_out.size(1) == c_out.size(1)
        else:
            # support pre-masked input
            x_in = observables[:, :c_in.size(1)]
            x_out = None
        return c_in, c_out, x_in, x_out

    def prediction_step(self, batch, batch_idx, dataloader_idx=0):
        # Todo: move the following logic to the dataloaders
        c_in, c_out, x_in, _ = self.split_batch_data(batch)
        x_out_pred = self(x_in, c_in, c_out)
        return x_out_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_params)
        return optimizer

    def log_summary(self, config: DictConfig):
        summary = torchinfo.summary(
            self,
            input_size=[
                (2, config.seq_length - self.forecast_window, len(config.data.cols.x)),
                (2, config.seq_length - self.forecast_window, len(config.data.cols.c)),
                (2, self.forecast_window, len(config.data.cols.c)),
            ],
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
