import math
import torch
import torch.nn as nn


class BasicRNN(nn.Module):

    def __init__(self, input_size, hidden_size=20, output_size=5, num_layers=1, batch_size=8, dropout=0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = dropout

        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size * 3),
            nn.SiLU(),
            nn.Linear(output_size * 3, output_size),
        )

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return torch.zeros(self.num_layers, batch_size,
                           self.hidden_size)  # num_layers, batch_size, hidden_size

    def forward(self, input, hidden_0=None):
        # input shape: (batch_size, seq_length, input_size)
        batch_size = input.size(0)
        if hidden_0 is None:
            hidden_0 = self.init_hidden(batch_size)

        out_z, hidded_t = self.rnn(input, hidden_0)  # output is last layer hidden state, for each time step
        output = self.fc(
            out_z)  # (batch_size, seq_length, hidden_size) -> (batch_size, seq_length, X_variables)

        return output


class BasicLSTM(nn.Module):

    def __init__(self, input_size, hidden_size=20, output_size=5, num_layers=1, batch_size=8, dropout=0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = dropout

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size * 3),
            nn.SiLU(),
            nn.Linear(output_size * 3, output_size),
        )

    # def init_hidden(self, batch_size=None):
    #     if batch_size is None:
    #         batch_size = self.batch_size
    #     return torch.zeros(self.num_layers, batch_size, self.hidden_size) # num_layers, batch_size, hidden_size

    def forward(self, input, hidden_0=None):
        # input shape: (batch_size, seq_length, input_size)
        batch_size = input.size(0)
        # if hidden_0 is None:
        #     hidden_0 = self.init_hidden(batch_size)

        out_z, hidded_t = self.rnn(input)  # output is last layer hidden state, for each time step
        output = self.fc(
            out_z)  # (batch_size, seq_length, hidden_size) -> (batch_size, seq_length, X_variables)

        return output


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


class ConvDownBlock(nn.Module):
    pass  # TODO


class ConvEncoder(nn.Module):
    """Encode either a window of controls or a window of observables or both.

    Encoder is completely flexible with regards to input length.
    """

    KERNEL_SIZE = [7, 7, 7]
    STRIDES = [5, 4, 3]

    def __init__(
        self,
        input_channels=12,
        hidden_channels=64,
        rnn_input_channels=32,
        hidden_size=64,
        rnn_layers=4,
        dropout=0.2,
        input_length=None,
    ):
        super().__init__()
        if input_length:
            self.calculate_compressed_length(input_length)

        self.ConvNet = nn.Sequential(
            nn.BatchNorm1d(input_channels),
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=hidden_channels,
                kernel_size=7,
                stride=5,
            ),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=7,
                stride=4,
            ),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=rnn_input_channels,
                kernel_size=7,
                stride=3,
            ),
            nn.SiLU(),
            nn.BatchNorm1d(rnn_input_channels),
        )
        self.lstm = nn.LSTM(
            input_size=rnn_input_channels,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, input, hidden_0=None):
        """Encode the warmup sequence into a single hidden state."""
        # input shape: (batch_size, seq_length, input_channels)
        input = input.permute(0, 2, 1)  # (batch_size, input_channels, seq_length)
        compressed = self.ConvNet(input)  # out (batch_size, conv_channels, compressed_length)
        compressed = compressed.permute(0, 2, 1)  # (batch_size, compressed_length, conv_channels)
        # RNN input should be (N, L, Hin) where Hin is input_size
        out_z, (ht, ct) = self.lstm(compressed)  # out_z is last layer hidden state, for each time step
        return ht, ct  # (num_layers 4, N, Hout=64)

    @staticmethod
    def conv_output_size(L_in, padding=0, dilation=1, kernel_size=3, stride=1):
        return (math.floor((L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)

    def calculate_compressed_length(self, input_length):
        for i in range(3):
            input_length = self.conv_output_size(
                input_length,
                kernel_size=self.KERNEL_SIZE[i],
                stride=self.STRIDES[i],
                padding=0,
                dilation=1,
            )
        return input_length


class Decoder(nn.Module):
    """Starts with initial hidden state and unrolls the LSTM."""

    def __init__(self, input_size, hidden_size=64, output_size=5, rnn_layers=4, dropout=0.3, rnn_type="LSTM"):
        super().__init__()
        self.rnn = getattr(nn, rnn_type)(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size * 3),
            nn.SiLU(),
            nn.Linear(output_size * 3, output_size),
        )

    def forward(self, input, hidden_0):
        """Decode the forecast horizon, given the hidden state from the encoder."""
        # input shape: (batch_size, seq_length, input_size)
        out_z, hidded_t = self.rnn(input, hidden_0)
        return self.fc(out_z)


class EncoderDecoder(nn.Module):

    def __init__(
        self,
        input_size=12,
        output_size=5,
        hidden_cnn_channels=64,
        rnn_input_channels=32,
        hidden_state_size=64,
        decoder_rnn_type="LSTM",
        num_layers=4,
        dropout=0.3,
        forecast_horizon=400,
        input_length=None,
    ):
        self.hyperparams = locals()
        super().__init__()
        self.encoder = ConvEncoder(
            input_length=input_length,  # Not used yet, but could be useful for debugging
            input_channels=input_size,
            hidden_channels=hidden_cnn_channels,
            rnn_input_channels=rnn_input_channels,
            hidden_size=hidden_state_size,
            dropout=dropout,
            rnn_layers=num_layers,
        )
        self.decoder = Decoder(input_size=7,
                               hidden_size=hidden_state_size,
                               output_size=output_size,
                               rnn_layers=num_layers,
                               rnn_type=decoder_rnn_type)
        self.forecast_horizon = forecast_horizon

    def forward(self, c, x):
        # input shape: (batch_size, seq_length, input_size)
        # this juggling should be in the train method
        with torch.no_grad():
            warmup = torch.cat((c[:, :-self.forecast_horizon], x[:, :-self.forecast_horizon]), dim=2)
            c_f = c[:, -self.forecast_horizon:]
        state_hc = self.encoder(warmup)
        if self.hyperparams["decoder_rnn_type"] == "LSTM":
            state = state_hc
        else:
            state = state_hc[0]
        return self.decoder(input=c_f, hidden_0=state)
