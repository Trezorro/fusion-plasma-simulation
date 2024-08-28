import torch
import torch.nn as nn


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

    def __init__(self, in_channels, out_channels, kernel_size, dilation, A=False, **kwargs):
        """"""
        super(CausalConv1d, self).__init__()

        # attributes:
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.A = A

        # to make sure the tree branches have room to extend to the left of the start of the sequence
        self.padding = (kernel_size - 1) * dilation + A * 1

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
        x = torch.nn.functional.pad(x, (self.padding, 0))
        conv1d_out = self.conv1d(x)
        if self.A:
            return conv1d_out[:, :, :-1]
        else:
            return conv1d_out


kernel_size = 3


def make_causal_conv_net(in_channels, hidden_channels, out_channels=1, kernel_size=7, num_layers=4):
    # First A layer:
    layers = [
        CausalConv1d(in_channels=in_channels,
                     out_channels=hidden_channels,
                     dilation=1,
                     kernel_size=kernel_size,
                     A=True,
                     bias=True),
        nn.LeakyReLU(),
    ]
    #  Dilating layers:
    for i in range(1, num_layers):
        layers.extend([
            CausalConv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=i * kernel_size,
            ),
            nn.SiLU(),
            # nn.BatchNorm1d(hidden_channels),
        ])

    return nn.Sequential(*layers)


class AutoRegressiveModel(nn.Module):

    def __init__(self,
                 in_channels=1,
                 hidden_channels=64,
                 out_channels=1,
                 kernel_size=5,
                 num_layers=3,
                 use_tanh_output=True,
                 **kwargs):
        self.hyperparams = locals()
        super().__init__()
        self.convnet = make_causal_conv_net(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
        )
        # Regression layers:
        self.mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels // 2), nn.SiLU(),
                                 nn.Linear(hidden_channels // 2, out_channels))
        if use_tanh_output:
            self.mlp.add_module("tanh", nn.Tanh())

    def forward(self, c, x):
        x = torch.cat((c, x), dim=2)
        x = x.permute(0, 2, 1)
        x = self.convnet(x)
        x = x.permute(0, 2, 1)
        x = self.mlp(x)
        return x
