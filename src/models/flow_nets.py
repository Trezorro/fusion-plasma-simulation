import torch
import torch.nn as nn
import torch.nn.functional as F


class VelocityNet(nn.Module):
    ACTIVATION_OPTIONS = dict(
        gelu=F.gelu,
        ReLU=nn.ReLU,
        SiLU=nn.SiLU,
        Softplus=nn.Softplus,
        Identity=nn.Identity,
        Tanh=nn.Tanh,
        Sigmoid=nn.Sigmoid,
    )

    def __init__(self, input_dim, h_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.fc_in = nn.Linear(input_dim + 1, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, h_dim)
        self.fc_out = nn.Linear(h_dim, input_dim)

    def forward(self, x, t, act=F.gelu):
        t = t.expand(x.size(0), 1)  # Ensure t has the correct dimensions
        x = torch.cat([x, t], dim=1)
        x = act(self.fc_in(x))
        x = act(self.fc2(x))
        x = act(self.fc3(x))
        return self.fc_out(x)
