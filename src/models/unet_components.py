import torch
import torch.nn as nn


class DoubleConv(nn.Sequential):
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
        if not mid_channels:
            mid_channels = out_channels
        layers = [
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding,
                      bias=False),  # Bias was false in the original implementation
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super().__init__(*layers)


class DownBlock(nn.Module):
    """Apllies maxpool then double conv. Padding is 'same' by default."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int | str = 'same'
    ):
        super().__init__()
        self.maxpool = nn.MaxPool1d(2, padding=1)  # Use Padding so no information is lost close to the edges.
        self.double_conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x


class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels_left,
        in_channels_skip,
        out_channels,
        kernel_size=3,
        up_method: str | bool = 'upsample',
        padding='both'
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
        total_channels = in_channels_left + in_channels_skip
        if type(up_method) == bool:
            # map True to 'upsample' and False to 'identity'
            up_method = 'upsample' if up_method else 'identity'

        if up_method.lower() in ('upsample', 'bilinear'):
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            # if bilinear, use the normal convolutions to reduce the number of channels
            self.conv = DoubleConv(total_channels, out_channels, total_channels // 2, kernel_size=kernel_size)
        elif up_method.lower() in ('transposed', 'conv'):
            # idea: use this only in the upper expanding layers. In the lower layers, it hasn't conditioned on
            # c_out yet, so it doesn't know how to conditionally upscale.
            self.up = nn.ConvTranspose1d(in_channels_left, in_channels_left // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(
                in_channels_left // 2 + in_channels_skip, out_channels, kernel_size=kernel_size
            )
        else:
            self.up = nn.Identity()
            self.conv = DoubleConv(total_channels, out_channels, total_channels // 2, kernel_size=kernel_size)

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


def print_layer_line(layer_summary):
    """Prints a line of a layer summary.

    var_name
    class_name
    depth
    input_size
    output_size
    """
    _, in_channels, in_length = layer_summary.input_size
    _, out_channels, out_length = layer_summary.output_size
    if layer_summary.depth == 1:
        print("-" * 45)
        print(f"{layer_summary.var_name:<13}", end="")
    else:
        print(f"{layer_summary.var_name:>13}", end="")
    print(
        f" | C {in_channels:>4} > {out_channels:>4} | "
        f"L {in_length:>3} > {out_length:<3} | ({layer_summary.class_name})"
        # f"K : {str(layer_summary.kernel_size):>10} | Params {layer_summary.num_params:>10}"
    )
