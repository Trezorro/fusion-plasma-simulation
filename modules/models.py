import torch
import torch.nn as nn

class BasicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, batch_size=8, batch_first=True):
        super(BasicRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.batch_first = batch_first

        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return torch.zeros(self.num_layers, batch_size, self.hidden_size) # num_layers, batch_size, hidden_size

    def forward(self, input, hidden_0=None):
        # input shape: (batch_size, seq_length, input_size)
        batch_size = input.size(0)
        if hidden_0 is None:
            hidden_0 = self.init_hidden(batch_size)

        out_z, hidded_t = self.rnn(input, hidden_0) # output is last layer hidden state, for each time step
        output = self.fc(out_z)  # (batch_size, seq_length, hidden_size) -> (batch_size, seq_length, X_variables)

        return output

    def train_loop(self, input, target, optimizer, criterion, num_epochs):
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = self.forward(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

if __name__ == "__main__":
    # Example usage
    input_size = 10  # Number of columns in C + X
    hidden_size = 20
    output_size = 5  # Number of columns in X

    rnn = BasicRNN(input_size, hidden_size, output_size)

    # Create a random input tensor
    batch_size = 1
    seq_length = 3
    input = torch.randn(seq_length, batch_size, input_size)

    # Forward pass
    output = rnn(input)
    print(output)