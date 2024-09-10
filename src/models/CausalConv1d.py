from typing import Optional
import torch
import torch.nn as nn
import lightning as L
import torchinfo
from omegaconf import DictConfig
import wandb


class CausalConv1d(nn.Module):
    """
    A causal 1D convolution. Will handle any sequence length, outputting the same number of outputs.

    Adapted from Tomczak, J. M. (2022). Deep Generative Modeling. Springer Nature.

    The general idea is the following: We take the built-in PyTorch Conv1D. Then, we must pick a proper padding,
    because we must ensure the convolutional is causal. Eventually, we must remove some final elements of the output,
    because we simply don't need them! Since CausalConv1D is still a convolution, we must define the kernel size,
    dilation , and whether it is option A (A=True) or option B (A=False). Remember that by playing with dilation we
    can enlarge the size of the memory.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, A=False, use_padding=True, **kwargs):
        """"""
        super(CausalConv1d, self).__init__()

        # attributes:
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.A = A

        # to make sure the tree branches have room to extend to the left of the start of the sequence
        self.padding = ((kernel_size - 1) * dilation + A * 1) * use_padding

        # module:
        self.conv1d = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            **kwargs,
        )

    def forward(self, x):
        if self.padding != 0:
            x = torch.nn.functional.pad(x, (self.padding, 0))
        conv1d_out = self.conv1d(x)
        if self.A:
            return conv1d_out[:, :, :-1]
        else:
            return conv1d_out


def minimum_input_for_conv_output(L_out, padding=0, dilation=1, kernel_size=3):
    """
    Calculate the minimum input size to get a desired output size.

    Note that this is a simplified version of the formula, which assumes stride=1.
    Padding is only applied to the left side of the input, and thus not multiplied by 2.

    See https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d for more details on the
    formula.

    Args:
        L_out (int): Desired output size.
        padding (int): Padding size.
        dilation (int): Dilation size.
        kernel_size (int): Kernel size.

    Returns:
        int: Minimum input size.
    """
    return L_out - padding + dilation * (kernel_size - 1)


def minimum_input_for_layers(padding=[0, 0, 0], dilation=[1, 2, 4], kernel_size=[3, 3, 3]):
    """
    Calculate the minimum input size to get a desired output size for a stack of convolutions.

    Args:
        padding (list): List of padding sizes, from lowest to highest layer.
        dilation (list): List of dilation sizes, idem.
        kernel_size (list): List of kernel sizes.

    Returns:
        int: Minimum input size.
    """
    L_out = 1
    for p, d, k in reversed(list(zip(padding, dilation, kernel_size))):
        if not p:  # if there is padding, we assume it maintains the size 1 to 1, otherwise:
            L_out = minimum_input_for_conv_output(L_out, padding=0, dilation=d, kernel_size=k)
    return L_out


class CausalConvNet(nn.Sequential):

    def __init__(self, in_channels=8, hidden_channels=64, kernel_size=5, num_layers=3, use_padding=True):
        self.hyperparams = locals()
        # First A layer:
        self.layers = [
            CausalConv1d(in_channels=in_channels,
                         out_channels=hidden_channels,
                         dilation=1,
                         kernel_size=kernel_size,
                         A=True,
                         use_padding=use_padding,
                         bias=True),
            nn.LeakyReLU(),
        ]
        #  Dilating layers:
        for i in range(1, num_layers):
            self.layers.extend([
                CausalConv1d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=i * kernel_size,
                    use_padding=use_padding,
                ),
                nn.SiLU(),
                # nn.BatchNorm1d(hidden_channels),
            ])
        super().__init__(*self.layers)
        self.minimum_input_length = minimum_input_for_layers(
            padding=[use_padding] * num_layers,
            dilation=[1] + [i * kernel_size for i in range(1, num_layers)],
            kernel_size=[kernel_size] * num_layers,
        )


class AutoRegressiveModel(L.LightningModule):
    """Model that uses a CausalConvNet to predict the next value in a sequence."""

    def __init__(self,
                 in_channels=1,
                 hidden_channels=64,
                 out_channels=1,
                 train_rollout=1,
                 validation_rollout=5,
                 kernel_size=5,
                 num_layers=3,
                 use_tanh_output=True,
                 use_padding=True,
                 loss="MSELoss",
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.out_channels = out_channels
        self.train_rollout = train_rollout
        self.val_rollout = validation_rollout
        self.loss = getattr(torch.nn, loss)()
        self.convnet = CausalConvNet(in_channels=in_channels,
                                     hidden_channels=hidden_channels,
                                     kernel_size=kernel_size,
                                     num_layers=num_layers,
                                     use_padding=use_padding)
        # Regression layers:
        self.mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels // 2), nn.SiLU(),
                                 nn.Linear(hidden_channels // 2, out_channels))
        if use_tanh_output:
            self.mlp.add_module("tanh", nn.Tanh())

    def forward(self, c, x, forecast_horizon: Optional[int] = None):
        """Returns the input sequence x with the last forecast_horizon elements filled with the model's own predictions.

        Returns:
            torch.Tensor: The input sequence with the last forecast_horizon elements filled with the model's own predictions. (batch_size, seq_length, variables)
        """
        input_seq_length = self.convnet.minimum_input_length
        if forecast_horizon is None:
            forecast_horizon = self.train_rollout

        seq = torch.cat((c, x), dim=2).detach().clone()  # (batch_size, seq_length, variables c + x)
        for t in range(seq.size(1) - forecast_horizon, seq.size(1)):
            input_part = seq[:, t - input_seq_length:t + 1].clone()
            a = input_part.permute(0, 2, 1)  # (batch_size, variable, seq_length)
            a = self.convnet(a)
            a = a.permute(0, 2, 1)  # (batch_size, forecast_seq_length, hidden_channels)
            x_t = self.mlp(a)  # (batch_size, seq_element, out_channels (1))
            seq[:, t, -self.out_channels:] = x_t.squeeze(dim=1)
        return seq[:, :, -self.out_channels:]

    def training_step(self, batch, batch_idx):
        shot_number, controls, observables = batch
        outputs = self(c=controls, x=observables)[:, -self.train_rollout:]
        f_x = observables[:, -self.train_rollout:]
        loss = self.loss(outputs, f_x)
        self.log("loss/train", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        shot_number, controls, observables = batch
        outputs = self(c=controls, x=observables, forecast_horizon=self.val_rollout)[:, -self.val_rollout:]
        f_x = observables[:, -self.val_rollout:]
        loss = self.loss(outputs, f_x)
        self.log("loss/val", loss, prog_bar=True)
        return loss

    test_step = validation_step

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def log_summary(self, config: DictConfig):
        summary = torchinfo.summary(
            self,
            input_size=[(config.seq_length, len(config.data.cols.c)),
                        (config.seq_length, len(config.data.cols.x))],
            batch_dim=0,
            col_names=[
                "input_size",
                "output_size",
                "kernel_size",
                "num_params",
                # "params_percent",
                "mult_adds",
                # "trainable"
            ],
        )  # (batch_size, seq_length, input_size)
        wandb.log(
            {
                "model/summary": str(summary),
                "model/trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            },
            step=0)
