import math
from typing import Optional
import torch
import torch.nn as nn
import lightning as L
import torchinfo
from omegaconf import DictConfig
import wandb


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
        """Decode the forecast horizon, given the hidden state from the encoder.

        Input shape: (batch_size, f_seq_length, input_size)
        Output shape: (batch_size, f_seq_length, output_size)
        """
        out_z, hidded_t = self.rnn(input, hidden_0)
        return self.fc(out_z)


class EncoderDecoder(L.LightningModule):

    def __init__(
        self,
        warmup_input_size=12,
        conditional_input_size=7,
        output_size=5,
        hidden_cnn_channels=64,
        rnn_input_channels=32,
        hidden_state_size=64,
        decoder_rnn_type="LSTM",
        num_layers=4,
        dropout=0.3,
        train_rollout=400,
        val_rollout=400,
        loss="MSELoss",
        input_length=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.loss = getattr(torch.nn, loss)()
        self.encoder = ConvEncoder(
            input_length=input_length,  # Not used yet, but could be useful for debugging
            input_channels=warmup_input_size,
            hidden_channels=hidden_cnn_channels,
            rnn_input_channels=rnn_input_channels,
            hidden_size=hidden_state_size,
            dropout=dropout,
            rnn_layers=num_layers,
        )
        self.decoder = Decoder(
            input_size=conditional_input_size,  # c_f
            hidden_size=hidden_state_size,
            output_size=output_size,
            rnn_layers=num_layers,
            rnn_type=decoder_rnn_type)
        self.train_rollout = train_rollout
        self.val_rollout = val_rollout

    def forward(self, c, x, forecast_horizon: Optional[int] = None):
        # input shape: (batch_size, seq_length, input_size)
        if forecast_horizon is None:
            forecast_horizon = self.train_rollout

        with torch.no_grad():
            warmup = torch.cat((c[:, :-self.forecast_horizon], x[:, :-self.forecast_horizon]), dim=2)
            c_f = c[:, -self.forecast_horizon:]
        state_hc = self.encoder(warmup)
        if self.hyperparams["decoder_rnn_type"] == "LSTM":
            state = state_hc
        else:
            state = state_hc[0]
        return self.decoder(input=c_f, hidden_0=state)

    def training_step(self, batch, batch_idx):
        shot_number, controls, observables = batch
        outputs = self(c=controls, x=observables)[:, -self.train_rollout:]
        f_x = observables[:, -self.train_rollout:]
        loss = self.loss(outputs, f_x)
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        shot_number, controls, observables = batch
        outputs = self(c=controls, x=observables)[:, -self.val_rollout:]
        f_x = observables[:, -self.val_rollout:]
        loss = self.loss(outputs, f_x)
        train_loss = self.loss(outputs[:, :self.train_rollout], f_x[:, :self.train_rollout])
        self.log("loss/val", loss, prog_bar=True)
        self.log("loss/val_train_rollout", train_loss, prog_bar=True)
        return dict(loss=loss, val_train_rollout=train_loss, outputs=outputs)


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
                "num_params",
                # "params_percent",
                # "kernel_size",
                "mult_adds",
                # "trainable"
            ],
        )  # (batch_size, seq_length, input_size)
        compressed_length = self.encoder.calculate_compressed_length(config.seq_length - config.train_rollout)
        print(
            f"Compressed length: {compressed_length} for warmup window {config.seq_length - config.train_rollout}"
        )
        wandb.log(
            {
                "model/summary": str(summary),
                "model/trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "model/compressed_length": compressed_length,
                # TODO: maybe add model/minimum_input_length, but need to calculate it first
            },
            step=0)
