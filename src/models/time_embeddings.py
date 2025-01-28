import torch
from torch import nn

import math


class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, n_channels: int, act: nn.Module = nn.SiLU()):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        assert self.n_channels % 2 == 0, "Number of time channels should be even"
        assert self.n_channels >= 8, "Number of time channels should be at least 8"
        self.D_HALF = self.n_channels // 8
        self.lin1 = nn.Linear(self.D_HALF * 2, self.n_channels)  # spread information over 4x the channels
        self.act = act
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

        # 10000 is how many unique values we can have in the time embeddings
        # devide the logarithmic scale up to 10_000 into 16 (or D_HALF) parts:
        log_part_size = math.log(10_000) / (self.D_HALF - 1)
        self.FREQUENCY_SCALES = torch.exp(
            torch.arange(self.D_HALF) * -log_part_size
        )  # TODO check if this goes to the right device

    def forward(self, t: torch.Tensor):
        """
        Create sinusoidal position embeddings
        [same as those from the transformer](../../transformers/positional_encoding.html)

        \\begin{align}
        PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{D - 1}}}\Bigg) \\
        PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{D - 1}}}\Bigg)
        \\end{align}

        where $D$ is `self.D_HALF`

        Arg:
            t (torch.Tensor): Shape `[batch_size] or scalar`
        """
        # If `t` is scalar,
        # Frequency scales (channels) is broadcasted across the batch size
        phase = t.view(-1, 1) * 10_000 * self.FREQUENCY_SCALES.to(t.device)  # [batch_size, half_dim]
        emb = torch.cat((phase.sin(), phase.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb  # [batch_size, n_channels]


class DummyTimeEmbedding(nn.Module):

    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels

    def forward(self, t: torch.Tensor):
        # Expand repeats the scalar `t` to the number of channels
        return t.view(-1, 1).expand(-1, self.n_channels)
