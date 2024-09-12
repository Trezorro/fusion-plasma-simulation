from math import e
from typing import Optional
import torch
import torch.nn as nn
import lightning as L
import torchinfo
from omegaconf import DictConfig


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2

    Based on https://github.com/milesial/Pytorch-UNet
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        kernel_size=3,
        padding: int | str = 'same',
    ):
        """(convolution => [BN] => ReLU) * 2.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            mid_channels (int, optional): Number of channels in the middle layer. Defaults to out_channels.
            kernel_size (int, optional): Kernel size. Defaults to 3.
            padding (int | str, optional): Padding. Defaults to 'same'.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding,
                      bias=False),  # Bias was false in the original implementation
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Apllies maxpool then double conv. Padding is 'same' by default."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int | str = 'same'
    ):
        super().__init__()
        self.maxpool = nn.MaxPool1d(2)
        self.double_conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x


class UpBlock(nn.Module):

    def __init__(
        self, in_channels_left, in_channels_skip, out_channels, kernel_size=3, bilinear=True, padding='both'
    ):
        """Upscales the input and concats it with the skip connection. Then applies double convolutions.

        Args:
            in_channels_left (int): Number of channels in the input tensor from the expanding path.
            in_channels_skip (int): Number of channels in the skip connection. May be less than in_channels_left.
            out_channels (int): Number of output channels.
            kernel_size (int, optional): Kernel size. Defaults to 3.
            bilinear (bool, optional): Use upsampling with bilinear interpolation, otherwise use transposed convolutions.
            padding (str, optional): Padding strategy on expanding path input. 'left', 'both', 'right'.
                Defaults to 'both'.
        """
        super().__init__()
        self.padding = padding

        # if bilinear, use the normal convolutions to reduce the number of channels
        total_channels = in_channels_left + in_channels_skip
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(total_channels, out_channels, total_channels // 2, kernel_size=kernel_size)
        else:
            # idea: use this only in the upper expanding layers. In the lower layers, it hasn't conditioned on
            # c_out yet, so it doesn't know how to conditionally upscale.
            self.up = nn.ConvTranspose2d(in_channels_left, in_channels_left // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(
                in_channels_left // 2 + in_channels_skip, out_channels, kernel_size=kernel_size
            )

    def forward(self, expanding_input, skip):
        expanding_input = self.up(expanding_input)
        # input is CHW
        skip_length_extra = skip.size(2) - expanding_input.size(2)

        match self.padding:
            case 'both':
                expanding_input = nn.functional.pad(
                    expanding_input, [skip_length_extra // 2, skip_length_extra - skip_length_extra // 2]
                )
            case 'left':
                expanding_input = nn.functional.pad(expanding_input, (skip_length_extra, 0))
            case 'right':
                expanding_input = nn.functional.pad(expanding_input, (0, skip_length_extra))
            case _:
                raise ValueError(f"Padding {self.padding} is not a valid option in UpBlock.")
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([skip, expanding_input], dim=1)
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, c_channels: int, x_channels: int, out_channels: int, n_blocks: int):
        super().__init__()
        self.cx_channels = c_channels + x_channels
        self.kernel_size = 3
        self.out_channels = out_channels
        # self.n_blocks = n_blocks
        self.in_conv_A = DoubleConv(c_channels, 64)
        self.down_B = DownBlock(64, 128)
        self.down_C = DownBlock(128, 256)
        self.down_D = DownBlock(256, 512)
        enc_padding = 0
        self.state_encoder = nn.Sequential(  # Encodes Xt-1 and Ct-1 into a state matrix St-1.
            DoubleConv(self.cx_channels, 64, kernel_size=self.kernel_size, padding=enc_padding),
            DownBlock(64, 128, kernel_size=self.kernel_size, padding=enc_padding),
            DownBlock(128, 256, kernel_size=self.kernel_size, padding=enc_padding),
            DownBlock(256, 512, kernel_size=self.kernel_size, padding=enc_padding),
            DownBlock(512, 1024, kernel_size=self.kernel_size, padding=enc_padding),
        )

        self.up_D = UpBlock(1024, 512, 512)
        self.up_C = UpBlock(512, 256, 256)
        self.up_B = UpBlock(256, 128, 128)
        self.up_A = UpBlock(128, 64, 64)
        self.out_conv = nn.Conv1d(64, self.out_channels, kernel_size=1)

    def forward(self, x_in: torch.Tensor, c_in: torch.Tensor, c_out: torch.Tensor):
        """Predicts the window x_out, which will be the same length as the given c_out.

        Inputs and outputs are in the shape (batch_size, seq_length, channels).
        """
        with torch.no_grad():
            warmup_window = torch.cat((x_in, c_in), dim=2)  # (batch_size, seq_length, variables c + x)
            # put channel dimension second
            warmup_window = warmup_window.permute(0, 2, 1)
            c_out = c_out.permute(0, 2, 1)

        # Contracting path A to D
        ac = self.in_conv_A(c_out)
        bc = self.down_B(ac)
        cc = self.down_C(bc)
        dc = self.down_D(cc)
        # State encoder conditioned on Xt-1 and Ct-1 (warmup window)
        s = self.state_encoder(warmup_window)
        # Conditioned eXpanding path D to A
        dx = self.up_D(s, dc)
        cx = self.up_C(dx, cc)
        bx = self.up_B(cx, bc)
        ax = self.up_A(bx, ac)
        # Output layer
        x_out = self.out_conv(ax)

        return x_out.permute(0, 2, 1)  # Returned as (batch_size, seq_length, variables x)


class UNetAuto(L.LightningModule):

    def __init__(self):
        super().__init__()
