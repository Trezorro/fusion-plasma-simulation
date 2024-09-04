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
