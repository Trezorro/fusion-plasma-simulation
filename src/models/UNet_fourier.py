import lightning as L
import torch
import torch.nn as nn
import torchinfo
import wandb
from omegaconf import DictConfig

from src.fourier import FourierMSLE
from src.models.unet_components import DoubleConv, DownBlock, UpBlock, print_layer_line


class UNetFourier(L.LightningModule):

    def __init__(
        self,
        c_channels: int,
        x_channels: int,
        out_channels: int,
        forecast_window: int,
        kernel_size: int = 3,
        loss: str = "MSELoss",
        num_layers: int = 4,
        conv_activation: str = 'ReLU',
        upsample_at_fusing: bool = False,
        output_activation: str = "Softplus",
        optimizer_params: dict = {},
        **kwargs
    ):
        super().__init__()
        self.forecast_window = forecast_window
        self.val_rollout = forecast_window
        self.train_rollout = forecast_window
        self.cx_channels = c_channels + x_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.upsample_at_fusing = upsample_at_fusing
        self.loss = getattr(torch.nn, loss)()
        self.fourier_loss_train = FourierMSLE()
        self.fourier_loss_val = FourierMSLE()
        self.optimizer_params = optimizer_params
        # self.n_blocks = n_blocks
        self.in_conv_A = DoubleConv(c_channels, 64, kernel_size=kernel_size)  # L = 64
        # self.down_B = DownBlock(64, 128, kernel_size=kernel_size)  # L = 32
        self.down_C = DownBlock(64, 128, kernel_size=kernel_size)  # L = 16
        self.down_D = DownBlock(128, 256, kernel_size=kernel_size)  # L = 8
        enc_padding = 0
        self.state_encoder = nn.Sequential(  # Encodes Xt-1 and Ct-1 into a state matrix St-1.
            DoubleConv(self.cx_channels, 64, kernel_size=self.kernel_size, padding=enc_padding), # Min Length: 64
            DownBlock(64, 128, kernel_size=self.kernel_size, padding=enc_padding),  # Min Length: 32
            DownBlock(128, 256, kernel_size=self.kernel_size, padding=enc_padding),  # Min Length: 16
            DownBlock(256, 512, kernel_size=self.kernel_size, padding=enc_padding),  # Min Length: 8
            DownBlock(512, 1024, kernel_size=self.kernel_size, padding=enc_padding)  # Min Length: 4
        )

        self.up_D = UpBlock(
            1024,
            256,
            512,
            kernel_size=kernel_size,
            up_method=('conv' if self.upsample_at_fusing else 'identity')
        )  # Min Length: 8
        self.up_C = UpBlock(512, 128, 256, kernel_size=kernel_size, up_method="conv")  # Min Length: 16
        # self.up_B = UpBlock(256, 128, 128, kernel_size=kernel_size, up_method="conv")  # Min Length: 32
        self.up_A = UpBlock(256, 64, 64, kernel_size=kernel_size, up_method="conv")  # Min Length: 64
        self.out_conv = nn.Conv1d(64, self.out_channels, kernel_size=1)
        if output_activation.lower() == 'softplus':
            self.out_activation = nn.Softplus(beta=2)
        elif output_activation.lower() == 'exp':
            self.out_activation = torch.exp
        else:
            self.out_activation = nn.Identity()

    def forward(self, x_in: torch.Tensor, c_in: torch.Tensor, c_out: torch.Tensor):
        """Predicts the window x_out, which will be the same length as the given c_out.

        Inputs and outputs are in the shape (batch_size, seq_length, channels).
        """
        with torch.no_grad():
            warmup_window = torch.cat((x_in, c_in), dim=2)  # (batch_size, seq_length, variables c + x)
            # put channel dimension second
            warmup_window = warmup_window.permute(0, 2, 1)  # (batch_size, variables c + x, seq_length)
            c_out = c_out.permute(0, 2, 1)  # (batch_size, variables c, forecast_window)

        warmup_window = torch.fft.rfft(warmup_window, dim=2)
        c_out = torch.fft.rfft(c_out, dim=2)

        # Contracting path A to D
        ac = self.in_conv_A(c_out)
        # bc = self.down_B(ac)
        cc = self.down_C(ac)
        dc = self.down_D(cc)
        # State encoder conditioned on Xt-1 and Ct-1 (warmup window)
        max_s_length = dc.size(2) // 2 if self.upsample_at_fusing else dc.size(2)
        s = self.state_encoder(warmup_window)[:, :, -max_s_length:]  # Min Length: 4
        # Conditioned eXpanding path D to A
        dx = self.up_D(s, dc)  # Min Length: 8
        cx = self.up_C(dx, cc)  # Min Length: 16
        # bx = self.up_B(cx, bc)  # Min Length: 32
        ax = self.up_A(cx, ac)  # Min Length: 64
        # Output layer
        x_out = self.out_conv(ax)
        x_out = self.out_activation(x_out)  # (batch_size, variables x, seq_length)

        x_out = torch.fft.irfft(x_out, dim=2)
        return x_out.permute(0, 2, 1)  # Returned as (batch_size, seq_length, variables x)

    def training_step(self, batch, batch_idx):
        shot_number, controls, observables = batch
        x_in = observables[:, :-self.forecast_window]
        c_in = controls[:, :-self.forecast_window]
        x_out = observables[:, -self.forecast_window:]
        c_out = controls[:, -self.forecast_window:]
        x_out_pred = self(x_in, c_in, c_out)
        loss = self.loss(x_out_pred, x_out)
        self.log("loss/train", loss, prog_bar=True)
        self.fourier_loss_train(x_out_pred, x_out)
        self.log("fourier_loss/train", self.fourier_loss_train, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        shot_number, controls, observables = batch
        x_in = observables[:, :-self.forecast_window]
        c_in = controls[:, :-self.forecast_window]
        x_out = observables[:, -self.forecast_window:]
        c_out = controls[:, -self.forecast_window:]
        x_out_pred = self(x_in, c_in, c_out)
        loss = self.loss(x_out_pred, x_out)
        self.log("loss/val", loss, prog_bar=True)
        self.fourier_loss_val(x_out_pred, x_out)
        self.log("fourier_loss/val", self.fourier_loss_val, prog_bar=True)
        return dict(loss=loss, outputs=x_out_pred)

    test_step = validation_step

    def prediction_step(self, batch, batch_idx, dataloader_idx=0):
        shot_number, controls, observables = batch
        # Todo: move the following logic to the dataloaders
        c_in = controls[:, :-self.forecast_window]
        if observables.size(1) > c_in.size(1):
            # allow to receive the whole sequence, but don't cheat by using future data
            x_in = observables[:, :-self.forecast_window]
        else:
            x_in = observables
        c_out = controls[:, -self.forecast_window:]
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
        summary_layers = {l.var_name: l for l in summary.summary_list}
        # TODO: Print fuze lengths
        for layer in summary.summary_list:
            if 0 < layer.depth < 3:
                print_layer_line(layer)

        fuse_length = summary_layers['down_D'].output_size[2]
        print(f"Fuse length: {fuse_length}")
        print(
            "State encoding length in / out: ", summary_layers['state_encoder'].input_size[2], " / ",
            summary_layers['state_encoder'].output_size[2],
            f"with {summary_layers['state_encoder'].output_size[1]} out channels."
        )
        if __name__ == '__main__':  # Don't log to wandb in test mode, break now.
            return
        wandb.log(
            {
                "model/summary": str(summary),
                "model/fuse_length": fuse_length,
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
    model = UNetFourier(**C['model']['params'])
    model.log_summary(C)
