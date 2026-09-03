""" (Auto generated:)
This module implements various neural network components for time-series and spatial data processing.

Classes:
- FNOLSTM: Combines a Fourier Neural Operator (FNO) embedding with an LSTM for sequence modeling. 
           It includes a convolutional feature extractor, an LSTM, and a final MLP for output generation.
- SpectralConv1d: Implements a 1D Fourier layer for spectral convolution, performing FFT, linear transformation, 
                  and inverse FFT.
- FNO_Layer: A single layer of the Fourier Neural Operator, combining spectral convolution and standard convolution 
             with optional activation.
- WindowFNOExtractor: Extracts features from a fixed-size time window using a stack of FNO layers and a dense layer.

Utility Functions:
- get_spectral_conv_with_right_spatial_dim: Returns a spectral convolution layer for the specified spatial dimension.
- get_conv_with_right_spatial_dim: Returns a standard convolution layer for the specified spatial dimension.
"""

import torch
from torch import nn



class FNOLSTM(nn.Module):
    """
    FNO embedding + LSTM class

    Args:
        n_in (int): Number of input channels.
        n_out (int): Number of output channels.
        tw (int): Time window size.
        h_c1, h_m1, h_c2, h_m2 (int): Hyperparameters for the convolutional layers.
        h_dropc (float): Dropout rate for convolutional layers.
        h_maxpool (int): Max pooling size.
        h_lstm_in (int): Input size for LSTM.
        h_lstm (int): Hidden size for LSTM.
        h_conv_res (bool): Whether to use residual connections in convolutional layers.
        h_mlp (int): Hidden size for the final MLP.
        h_dropmlp (float): Dropout rate for the final MLP.
        act: Activation function to use.
    """
    def __init__(self, n_in=3, n_out=3, tw=40,
                 h_c1=32, h_m1=8, h_c2=64, h_m2=8, h_dropc=.5, h_maxpool=2,
                 h_lstm_in=16, h_lstm=32,
                 h_conv_res=False,
                 h_mlp=8, h_dropmlp=.5,
                 act=nn.ReLU(), **kwargs):
        super().__init__()
        self.conv_extractor = WindowFNOExtractor(n_in=n_in, n_feature=h_lstm_in, tw=tw,
                                                 h_c=[h_c1, h_c2], h_m=[h_m1, h_m2],
                                                 h_drop=h_dropc, h_maxpool=h_maxpool, act=act)

        self.lstm = nn.LSTM(input_size=h_lstm_in, hidden_size=h_lstm)

        self.mlp = nn.Sequential(*[
            nn.Linear(h_lstm, h_mlp),
            act,
            nn.Dropout(h_dropmlp),
            nn.Linear(h_mlp, n_out),
            # nn.Softmax(dim=-1), <- logits as input for crossentropyloss
        ])

        self.conv_res = h_conv_res
        if self.conv_res:
            assert h_lstm_in == h_lstm

    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None):
        x = self.conv_extractor(x)
        if self.conv_res:
            x_conv = x
        x = x.unsqueeze(0)  # add single seq dim
        x, hidden = self.lstm(x, hidden)
        x = x.squeeze(0)  # squeeze away seq dim
        if self.conv_res:
            x = self.mlp(x + x_conv)
        else:
            x = self.mlp(x)
        return x, hidden


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes: tuple):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        '''
        Transform modes: 
        0 = Affine transformation, modeled as "activation * (1 + FiLM)" (parametrize delta)
        1 = Affine transformation, modeled as "activation * FiLM (parametrize transformation)
        '''

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes[0]  # Number of Fourier modes to multiply, at most floor(N/2) + 1,
        # where N is the number of points in our grid

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.complex64))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    # Complex multiplication, weights come in batch
    def compl_mul1d_batch(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,biox->box", input, weights)

    def forward(self, x, p=None):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)
        ft_size = x.size(-1) // 2 + 1
        out_ft = torch.zeros(batchsize, self.out_channels, ft_size, device=x.device, dtype=torch.complex64)

        # Multiply relevant Fourier modes
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


def get_spectral_conv_with_right_spatial_dim(spatial_dim, **kwargs):
    if spatial_dim == 1:
        conv = SpectralConv1d(**kwargs)
    else:
        raise NotImplementedError(f'only 0<x<=1d convs implemented so far, but found spatial dim {spatial_dim}!')
    return conv


def get_conv_with_right_spatial_dim(spatial_dim, **kwargs):
    if spatial_dim == 1:
        conv = nn.Conv1d(**kwargs)
    elif spatial_dim == 2:
        conv = nn.Conv2d(**kwargs)
    elif spatial_dim == 3:
        conv = nn.Conv3d(**kwargs)
    else:
        raise NotImplementedError(f'only 0<x<=3d convs implemented so far, but found spatial dim {spatial_dim}!')
    return conv


class FNO_Layer(nn.Module):
    '''
    Settings:
    0 = default (paper)

    Args:
        hidden_dim (int): Number of input channels.
        num_spatial_dims (int): Number of spatial dimensions.
        kernel_size (int): Kernel size for the convolutional layer.
        modes (int or tuple): Number of Fourier modes to use.
        activation (callable): Activation function to use.
        activation_params (dict): Parameters for the activation function.
        setting (int): Setting for the layer behavior.
        hidden_dim_out (int): Number of output channels.
    '''

    def __init__(self, hidden_dim, num_spatial_dims: int = 1,
                 kernel_size=1, modes=16, activation=nn.GELU, activation_params=None, setting=0,
                 hidden_dim_out=None):
        super(FNO_Layer, self).__init__()
        self.num_spatial_dims = num_spatial_dims
        if isinstance(modes, int):
            modes = tuple([modes for _ in range(num_spatial_dims)])  # tuple

        assert len(modes) == num_spatial_dims, 'modes should be int or tuple of ints with length equal to spatial dim!'

        if hidden_dim_out is None:
            hidden_dim_out = hidden_dim
        self.conv = get_spectral_conv_with_right_spatial_dim(spatial_dim=num_spatial_dims,
                                                             in_channels=hidden_dim, out_channels=hidden_dim_out,
                                                             modes=modes)
        self.w = get_conv_with_right_spatial_dim(spatial_dim=num_spatial_dims, in_channels=hidden_dim,
                                                 out_channels=hidden_dim_out, kernel_size=kernel_size, padding='same')
        if activation is None:
            self.act = None
        else:
            if activation_params is None:
                self.act = activation
            else:
                self.act = activation(**activation_params)

        self.setting = setting

    def forward(self, x, p=None):
        x1 = self.conv(x, p)
        x2 = self.w(x)
        if self.setting == 0:
            x = x1 + x2
        if self.act is not None:
            x = self.act(x)
        return x


class WindowFNOExtractor(nn.Module):
    """
    For some fixed-size time window 'tw', extracts 'n_feature' features using fno layers and a dense layer
    Args:
        n_in (int): Number of input channels.
        n_feature (int): Number of output features.
        tw (int): Time window size.
        h_c (list): List of hidden dimensions for FNO layers.
        h_m (list): List of modes for FNO layers.
        h_drop (float): Dropout rate.
        h_maxpool (int): Max pooling size.
        act: Activation function to use.
        act_final (bool): Whether to apply activation after the final layer.

    Architecture:
    Input -> FNO layers[] (-> Dropout) (-> MaxPool) -> Flatten -> Linear (-> Activation) -> Output
    
    """
    def __init__(self, n_in: int, n_feature: int, tw: int, h_c: list, h_m: list, h_drop: float, h_maxpool: int, act=nn.GELU(),
                 act_final=True):
        super().__init__()
        assert len(h_c) == len(h_m)
        assert len(h_c) > 0
        assert h_maxpool >= 1
        self.convs = [FNO_Layer(hidden_dim=n_in, hidden_dim_out=h_c[0], kernel_size=1, num_spatial_dims=1, modes=h_m[0],
                                activation=act)]
        for i in range(1, len(h_c)):
            self.convs.append(FNO_Layer(hidden_dim=h_c[i - 1], hidden_dim_out=h_c[i], kernel_size=1, num_spatial_dims=1,
                                        modes=h_m[i], activation=act))
        self.convs = nn.Sequential(*self.convs)

        self.feature_extr = []
        if h_drop > 0:
            self.feature_extr.append(nn.Dropout(h_drop))
        if h_maxpool > 1:
            self.feature_extr.append(nn.MaxPool1d(h_maxpool))
        self.feature_extr.extend([nn.Flatten(),
                                  nn.Linear(h_c[-1] * tw // h_maxpool, n_feature)])
        if act_final:
            self.feature_extr.append(act)
        self.feature_extr = nn.Sequential(*self.feature_extr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extr(self.convs(x))
