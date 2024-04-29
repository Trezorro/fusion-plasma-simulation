import torch
import torch.nn as nn

class BasicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BasicRNN, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        hidden = self.init_hidden()

        output, hidden = self.rnn(input, hidden)
        output = self.fc(output)

        return output

    def init_hidden(self):
        return torch.zeros(1, 8, self.hidden_size)

    def train(self, input, target, optimizer, criterion, num_epochs):
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            hidden = self.init_hidden()
            output, hidden = self.rnn(input, hidden)
            output = self.fc(output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

if __name__ is "__main__":
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