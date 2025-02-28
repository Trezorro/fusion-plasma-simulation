from typing import Optional
import torch
from torch import nn

import math


def build_time_embedder(time_embedding_str: str, d: int, max_value=1.0, suggested_projection_dim: int = 0):
    """
    Factory function to create a TimeEmbedding instance based on the given string.

    Args:
        time_embedding_str (str): The time embedding string, e.g., "sinusoidal+mlp".
        d (int): The dimension of the time embedding.
        suggested_projection_dim (int): The suggested projection dimension for the time embedding. This is used to determine the projection dimension if the time embedding is "sinusoidal+mlp".
        time_embedding_classes (dict): Dictionary of available time embedding classes.

    Returns:
        nn.Module: An instance of the appropriate time embedding class.
    """
    if time_embedding_str.startswith("dummy"):
        return DummyTimeEmbedding(d)
    elif time_embedding_str.startswith("sinusoidal"):
        if "+mlp" in time_embedding_str:
            projection_channels = suggested_projection_dim
        else:
            projection_channels = None  # will not use MLP head

    return TimeEmbedding(d=d, max_position=max_value, projection_channels=projection_channels)


class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(
        self,
        d: int,
        projection_channels: Optional[int] = None,
        act: nn.Module = nn.SiLU(),
        max_position: float = 1.0,
        resolution_base: int = 10_000
    ):
        """Create sinusoidal position embeddings for time or position `t`

        Args:
            d (int): Number of sinusoidal time channels. Should be even.
            projection_channels (int, optional): Number of channels in the learned time embedding. If None, no learned embedding is used, and output_dim is equal to d.
                Defaults to None.
            act (nn.Module, optional): Activation function. Defaults to nn.SiLU().
            max_position (float, optional): Maximum position value, where the last (slowest) channel is at phase 1.0.
                Defaults to 1.0.
            resolution_base (int, optional): Base of the resolution. 1 / resolution_base is the minimum translation that is covered by the first (fastest) channel within phase 0 to 1.
                Defaults to 10_000. 
        
        Provides output_dim as the number of channels in the time embedding.
        """
        super().__init__()
        assert d % 2 == 0, "Number of sinusoidal time channels should be even"
        assert d >= 4, "Number of sinusoidal time channels should be at least 4, giving D_half >= 2"
        self.d = d
        self.projection_channels = projection_channels
        self.resolution_base = resolution_base
        self.max_position = max_position
        self.scaler = self.resolution_base / self.max_position
        self.D_HALF = d // 2
        # 10000 is how many unique values we can have in the time embeddings
        # devide the logarithmic scale up to 10_000 into 16 (or D_HALF) parts:
        log_part_size = math.log(self.resolution_base) / (self.D_HALF - 1)
        self.FREQUENCY_SCALES = torch.exp(torch.arange(self.D_HALF) * -log_part_size).detach()

        # If projection channels are given, use an MLP to learn the final time embedding
        if self.projection_channels is not None:
            self.learned_embedding = True
            assert self.projection_channels % 2 == 0, "Number of learned time embedding channels should be even"
            assert self.projection_channels >= 2, "Number of learned time embedding channels should be at least 2"
            # Take the larger one of the two dimensions to maximize the information flow:
            self.hidden_dim = self.d if self.d > self.projection_channels else self.projection_channels
            self.lin1 = nn.Linear(self.d, self.hidden_dim)
            self.act = act
            self.lin2 = nn.Linear(self.hidden_dim, self.projection_channels)
            self.output_dim = self.projection_channels
        else:
            self.learned_embedding = False
            self.output_dim = self.d

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
            t (torch.Tensor): Shape `[batch_size], [batch_size, pos] or scalar`
        """
        # If `t` is scalar,
        # Frequency scales (channels) is broadcasted across the batch size
        scaled_t = t * self.scaler
        phase = scaled_t.view(-1, 1) * self.FREQUENCY_SCALES.to(t.device)  # [batch_size, half_dim]
        emb = torch.cat((phase.sin(), phase.cos()), dim=1)
        if not self.learned_embedding:
            return emb.to(torch.float32).view(*t.shape, self.d)  # [batch_size, d]

        # If using learned embedding, transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb.view(*t.shape, self.projection_channels)  # [batch_size, projection_channels]


class DummyTimeEmbedding(nn.Module):

    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.output_dim = n_channels

    def forward(self, t: torch.Tensor):
        # Expand repeats the scalar `t` to the number of channels
        return t.view(-1, 1).expand(-1, self.output_dim).view(*t.shape, self.output_dim)


def plot_time_embedding(min_value, max_value, d, use_mlp=False):
    if use_mlp:
        time_embedding = TimeEmbedding(d=d, projection_channels=d, max_position=max_value)
    else:
        time_embedding = TimeEmbedding(d=d, max_position=max_value)
    t = torch.linspace(min_value, max_value, 2341)
    print("t.shape: ", t.shape)
    # print("t: ", t)
    emb = time_embedding(t).detach().numpy()[:, :d]
    print("emb.shape: ", emb.shape)
    # print("emb: ", emb)
    # Test for duplicates:
    import pandas as pd
    df = pd.DataFrame(emb)
    print("Min max per channel:\nMin:", *df.min().values, "\nMax:", *df.max().values)
    num_duplicates = df.duplicated().sum()
    print("df.duplicated().sum(): ", num_duplicates)
    plt.figure(figsize=(10, 8))
    plt.matshow(
        emb.T, fignum=0, aspect='auto', origin='lower', extent=[min_value, max_value, 0, d], cmap='viridis'
    )
    plt.colorbar(label='Embedding Value')
    plt.xlabel('Time')
    plt.ylabel('Channels')
    plt.title(f'Time Embedding in [{min_value}-,{max_value}) and D={d} ({num_duplicates} duplicates)')
    # plt.show(block=False)


if __name__ == "__main__":
    # Test TimeEmbedding
    import matplotlib.pyplot as plt
    time_embedding = build_time_embedder("sinusoidal+mlp", 64, 1.0, 64)
    print("Testing TimeEmbedding")
    # gives D of 8
    plot_time_embedding(min_value=0, max_value=20000, d=64)
    plot_time_embedding(min_value=0, max_value=20000, d=32)
    plot_time_embedding(min_value=0, max_value=10_000, d=32)
    plot_time_embedding(min_value=0, max_value=1, d=32)
    plot_time_embedding(min_value=0, max_value=1, d=64)
    plot_time_embedding(min_value=0, max_value=1, d=64, use_mlp=True)
    plt.show(block=True)
