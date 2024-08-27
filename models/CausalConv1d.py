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


CausalConvNet = nn.Sequential(
    CausalConv1d(in_channels=1, out_channels=M, dilation=1, kernel_size=kernel, A=True, bias=True),
    nn.LeakyReLU(),
    CausalConv1d(in_channels=M, out_channels=M, dilation=2 * kernel, kernel_size=kernel, A=False, bias=True),
    nn.LeakyReLU(),
    CausalConv1d(in_channels=M, out_channels=M, dilation=4 * kernel, kernel_size=kernel, A=False, bias=True),
    nn.LeakyReLU(),
    CausalConv1d(in_channels=M, out_channels=M, dilation=8 * kernel, kernel_size=kernel, A=False, bias=True),
    nn.LeakyReLU(),
    CausalConv1d(in_channels=M,
                 out_channels=num_vals,
                 dilation=16 * kernel,
                 kernel_size=kernel,
                 A=False,
                 bias=True))
