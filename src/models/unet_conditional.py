"""
---
title: U-Net model for Denoising Diffusion Probabilistic Models (DDPM)
summary: >
  UNet model for Denoising Diffusion Probabilistic Models (DDPM)
---

# U-Net model for [Denoising Diffusion Probabilistic Models (DDPM)](index.html)

This is a [U-Net](../../unet/index.html) based model to predict noise
$\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$.

U-Net is a gets it's name from the U shape in the model diagram.
It processes a given image by progressively lowering (halving) the feature map resolution and then
increasing the resolution.
There are pass-through connection at each resolution.

![U-Net diagram from paper](../../unet/unet.png)

This implementation contains a bunch of modifications to original U-Net (residual blocks, multi-head attention)
 and also adds time-step embeddings $t$.

Adapted from labml_nn/diffusion/ddpm/unet.py
https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/05321d644e4fed67d8b2856adc2f8585e79dfbee/labml_nn/diffusion/ddpm/unet.py 

Removed unused norm layer in AttentionBlock

Inspired by Koen Minartz https://openreview.net/forum?id=CCVsGbhFdj implementation:
- Added configurable activation and input dimensionalities.
- Cropping function to match encoder and decoder features
"""
import logging
from typing import Callable, Optional, Tuple, Union, List

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from src.models.time_embeddings import TimeEmbedding, DummyTimeEmbedding
from src.models.activations import ACTIVATION_OPTIONS

logger = logging.getLogger(__name__)

CONV_LAYERS = {
    1: nn.Conv1d,
    2: nn.Conv2d,
    3: nn.Conv3d,
}
CONV_T_LAYERS = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d,
}


class ConditionalUNet(nn.Module):
    """
    ## U-Net
    """
    TIME_EMBEDDING_CLASSES = {
        "dummy": DummyTimeEmbedding,
        "sinusoidal": TimeEmbedding,
    }

    def __init__(
        self,
        input_channels: int = 3,
        spatial_dim: int = 1,  # 2 for images, 1 for time series
        apex_hidden_channels: int = 64,
        time_embedding_channels: Optional[int] = None,
        time_embedding: str = "sinusoidal",
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Tuple[bool, ...] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        activation: str = "GELU",
        use1x1: bool = False,  # TODO implement 1x1 convolution
        norm_groups: int = 8,  # If 0, no group normalization
        conditioning: list[str] = [],
        conditioning_method: str = "sequence",  # sequence, channels, mid-sequence, mid-channels
        **kwargs
    ):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        if not spatial_dim == 1:
            # especially for conditioning concatenation, harder to implement for 2D
            raise NotImplementedError("Only 1D supported for now.")
        self.conditioning = conditioning
        self.use_x_history = "x_history" in conditioning
        self.conditioning_method = conditioning_method

        # Number of resolutions
        n_resolutions = len(ch_mults)
        # Time embedding layer. Time embedding has `n_channels * 4` channels
        if time_embedding_channels is None:
            time_embedding_channels = apex_hidden_channels * 4
        self.time_emb = self.TIME_EMBEDDING_CLASSES[time_embedding](time_embedding_channels)

        self.act = ACTIVATION_OPTIONS[activation]()
        self.ConvLayer = CONV_LAYERS[spatial_dim]
        # Project image into feature map
        # If conditioning, add a dummy channel to signal the presence of conditioning or previous input.
        if self.use_x_history:
            if conditioning_method == "sequence":
                in_channels = input_channels + 1
            elif conditioning_method == "channels":
                in_channels = input_channels * 2
            else:
                in_channels = input_channels
        self.image_proj = self.ConvLayer(in_channels, apex_hidden_channels, kernel_size=3, padding=1)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        in_channels = apex_hidden_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        time_embedding_channels,
                        is_attn[i],
                        act=self.act,
                        norm_groups=norm_groups,
                        spatial_dim=spatial_dim
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels, spatial_dims=spatial_dim))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(
            out_channels,
            time_embedding_channels,
            has_attn=mid_attn,
            act=self.act,
            norm_groups=norm_groups,
            spatial_dim=spatial_dim
        )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        time_embedding_channels,
                        is_attn[i],
                        act=self.act,
                        norm_groups=norm_groups,
                        spatial_dim=spatial_dim
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(
                UpBlock(
                    in_channels,
                    out_channels,  # out channels should also match the skip connection channels
                    time_embedding_channels,
                    is_attn[i],
                    act=self.act,
                    norm_groups=norm_groups,
                    spatial_dim=spatial_dim
                )
            )
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels, spatial_dims=spatial_dim))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(norm_groups, apex_hidden_channels) if norm_groups > 0 else nn.Identity()
        self.final = self.ConvLayer(
            in_channels, input_channels, kernel_size=3, padding=1
        )  # TODO: Option for 1x1 convolution

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning_input: Optional[dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        * `x` has shape `[batch_size, in_channels, *spatial dim (HxW | L)]`
        * `t` has shape `[batch_size]` or scalar
        """
        assert x.dim() == 2 + self.spatial_dim, (
            f"Expected batch, channel plus configured spacial dims, but got {x.dim()}"
        )  # [n, c, *spatial_dims]
        x_in_shape = x.shape
        n, c, seq_length = x.shape
        x_hist_emb = None  # Placeholder for x_history embeddings WIP
        # Get image projection
        if conditioning_input and self.use_x_history:
            # Add x_history as a condition
            x_history = conditioning_input["x_history"]
            if self.conditioning_method == "sequence":
                x = self.condition_sequentially(x_history, x)
            elif self.conditioning_method == "channels":
                x = self.condition_channel_wise(x_history, x)
            elif self.conditioning_method == "mid-sequence":  # shared weights TODO implement unshared
                x_hist_emb = self.image_proj(x_history)

        x = self.image_proj(x)  # Project input to hidden channels
        t = self.time_emb(t)  # Get flow time embeddings

        # `hidden features` will store outputs at each resolution for skip connection
        hid_features = [x]
        # First half of U-Net
        for down_layer in self.down:
            x = down_layer(x, t)
            hid_features.append(x)
            if x_hist_emb is not None:  # Prepare x_history embeddings for middle block
                x_hist_emb = down_layer(x_hist_emb, t)

        if x_hist_emb is not None:  # conditioning method is mid-sequence
            x = torch.cat((x_hist_emb, x), dim=2)
            # TODO implement mid-channels and id dummy channel for conditioning

        # Middle (bottom)
        x = self.middle(x, t)
        # Second half of U-Net

        for i, up_layer in enumerate(self.up):
            if isinstance(up_layer, Upsample):
                # print('--' * i, i, x.shape, type(up_layer).__name__, end=">")
                x = up_layer(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = hid_features.pop(
                )  # TODO figure out how this matches previous resolutions in that list, and how channels work
                # print(
                # i, repr(self.down[i - 1]), f"{x.shape=},{s.shape=}", repr(up_layer), sep=">>>", end=">>>"
                # )
                s_crop = self._crop_Nd(s, x)  # crop spatial dim to match features
                # print('--' * i, i, x.shape, s.shape, '>crop>', s_crop.shape, type(up_layer).__name__, end=">")

                x = torch.cat((x, s_crop), dim=1)
                x = up_layer(x, t)
            # print(x.shape)

        # Final normalization and convolution
        x = self.final(self.act(self.norm(x)))
        x = self._crop_right(x, x_in_shape)  # crop spatial dim to match features
        return x

    def condition_sequentially(self, x_history, x):
        n, c, seq_length = x.shape
        double_length = x_history.shape[-1] + seq_length
        cond_signal = torch.ones(n, 1, double_length, device=self.device)
        cond_signal[:, :, -seq_length:] = 0  # signal for current input
        x = torch.cat((x_history, x), dim=2)
        x = torch.cat((x, cond_signal), dim=1)
        return x

    def condition_channel_wise(self, x_history, x):
        n, c, seq_length = x.shape
        history_length = x_history.shape[-1]
        if history_length > seq_length:
            # if history is longer, pad the input on the left, so most relevant information is on the right
            padding_difference = history_length - seq_length
            x = F.pad(x, (padding_difference, 0))
        elif seq_length > history_length:
            # if input is longer, pad the history on the left, so most relevant information is on the right
            padding_difference = seq_length - history_length
            x_history = F.pad(x_history, (padding_difference, 0))

        x = torch.cat((x_history, x), dim=1)
        return x

    def _crop_Nd(
        self,
        enc_ftrs: torch.Tensor,
        target: torch.Tensor | np.ndarray | torch.Size | tuple,
    ) -> torch.Tensor:
        """Crop the encoder features to match the shape of the decoder features.

        Supports different spatial dimensions (1D, 2D, 3D) by dynamically calculating paddings based on the number of spatial dimensions.

        By K Minartz
        
        Args:
            enc_ftrs (torch.Tensor): Encoder features.
            shape (torch.Tensor): Shape to crop to.
        
        Returns:
            torch.Tensor: Cropped encoder features.
        """
        if isinstance(target, torch.Tensor) or isinstance(target, np.ndarray):
            target = target.shape
        target_spatial_shape = target[-self.spatial_dim:]
        input_spatial_shape = enc_ftrs.shape[-self.spatial_dim:]
        # first, calculate preliminary paddings - may contain non-integers ending in .5):
        pad_temp = np.repeat(np.subtract(target_spatial_shape, input_spatial_shape) / 2, 2)
        # to break the .5 symmetry to round one padding up and one down, we add a small pos/neg number respectively
        # note this will not impact the case where pad_temp[i] is integer since it is still rounded to that integer
        breaking_arr = np.tile([1, -1], int(len(pad_temp) / 2)) / 1000
        pad = tuple(map(lambda p: int(round(p)), pad_temp + breaking_arr))
        enc_ftrs = F.pad(enc_ftrs, pad)
        return enc_ftrs

    def _crop_right(
        self,
        enc_ftrs: torch.Tensor,
        target: torch.Tensor | np.ndarray | torch.Size | tuple,
    ) -> torch.Tensor:
        """Crop the encoder features to match the shape of the decoder features.

        Supports different spatial dimensions (1D, 2D, 3D) by dynamically calculating paddings based on the number of spatial dimensions.

        By K Minartz
        
        Args:
            enc_ftrs (torch.Tensor): Encoder features.
            shape (torch.Tensor): Shape to crop to.
        
        Returns:
            torch.Tensor: Cropped encoder features.
        """
        if isinstance(target, torch.Tensor) or isinstance(target, np.ndarray):
            target = target.shape
        target_spatial_shape = target[-1]
        input_spatial_shape = enc_ftrs.shape[-1]
        # first, calculate preliminary paddings - may contain non-integers ending in .5):
        padding_difference = np.subtract(target_spatial_shape, input_spatial_shape)
        logger.debug("Conforming spatial dimensions: %s -> %s to the right", input_spatial_shape, target_spatial_shape)
        # to break the .5 symmetry to round one padding up and one down, we add a small pos/neg number respectively
        # note this will not impact the case where pad_temp[i] is integer since it is still rounded to that integer
        enc_ftrs = F.pad(enc_ftrs, (padding_difference, 0))
        return enc_ftrs


class ResidualBlock(nn.Module):
    """
    ### Residual block

    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        norm_groups: int = 32,  # If 0, no group normalization
        spatial_dim: int = 2,
        act: nn.Module | Callable[[torch.Tensor], torch.Tensor] = nn.SELU()
    ):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `norm_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html) (default 32), if 0, no group normalization.
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(norm_groups, in_channels) if norm_groups > 0 else nn.Identity()
        self.act1 = act
        self.conv1 = CONV_LAYERS[spatial_dim](
            in_channels, out_channels, kernel_size=3, padding=1
        )  # TODO see if this works for 1D, or if we need to change the kernel size

        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(norm_groups, out_channels) if norm_groups > 0 else nn.Identity()
        self.act2 = act
        self.conv2 = CONV_LAYERS[spatial_dim](out_channels, out_channels, kernel_size=3, padding=1)

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = CONV_LAYERS[spatial_dim](in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb_adapter = nn.Linear(time_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))

        # Add time embeddings
        t_emb = self.time_emb_adapter(t)
        target_t_shape = list(t_emb.shape) + [1] * self.spatial_dim  # Broadcast to spatial dimensions
        h += t_emb.view(*target_t_shape)
        # Second convolution layer
        h = self.conv2(self.act2(self.norm2(h)))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    ### Attention block

    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `norm_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k**-0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(
            dim=2
        )  # TODO: check if this is the correct dimension, should be the sequence dimension. maybe it doesn't matter.
        # dim 1: voor bepaalde key, sommen alle query posities naar 1.
        # dim 2: voor een bepaalde query, sommen alle key posities naar 1.
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res


class DownBlock(nn.Module):
    """Combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        time_channels (int): Number of channels in the time step ($t$) embeddings
        has_attn (bool): Whether to use attention
        act (nn.Module | Callable[[torch.Tensor], torch.Tensor], optional): Activation function. Defaults to nn.SELU().
        norm_groups (int, optional): Number of groups for group normalization. Defaults to 32. If 0, no group normalization.
        spatial_dim (int, optional): Number of spatial dimensions. Defaults to 2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool,
        act: nn.Module | Callable[[torch.Tensor], torch.Tensor] = nn.SELU(),
        norm_groups: int = 32,
        spatial_dim: int = 2,
    ):
        super().__init__()
        self.res = ResidualBlock(
            in_channels,
            out_channels,
            time_channels,
            act=act,
            norm_groups=norm_groups,
            spatial_dim=spatial_dim
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """Combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        time_channels (int): Number of channels in the time step ($t$) embeddings
        has_attn (bool): Whether to use attention
        act (nn.Module | Callable[[torch.Tensor], torch.Tensor], optional): Activation function. Defaults to nn.SELU().
        norm_groups (int, optional): Number of groups for group normalization. Defaults to 32. If 0, no group normalization.
        spatial_dim (int, optional): Number of spatial dimensions. Defaults to 2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool,
        act: nn.Module | Callable[[torch.Tensor], torch.Tensor] = nn.SELU(),
        norm_groups: int = 32,
        spatial_dim: int = 2,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(
            in_channels + out_channels,
            out_channels,
            time_channels,
            act=act,
            norm_groups=norm_groups,
            spatial_dim=spatial_dim
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """
    ### Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(
        self,
        n_channels: int,
        time_channels: int,
        in_channels: Optional[int] = None,
        has_attn: bool = False,
        act: nn.Module | Callable[[torch.Tensor], torch.Tensor] = nn.SELU(),
        norm_groups: int = 32,
        spatial_dim: int = 2,
    ):
        super().__init__()
        if in_channels is None:
            in_channels = n_channels
        self.res1 = ResidualBlock(
            in_channels,
            n_channels,
            time_channels,
            act=act,
            norm_groups=norm_groups,
            spatial_dim=spatial_dim,
        )
        if has_attn:
            self.attn = AttentionBlock(n_channels)
        else:
            self.attn = nn.Identity()
        self.res2 = ResidualBlock(
            n_channels,
            n_channels,
            time_channels,
            act=act,
            norm_groups=norm_groups,
            spatial_dim=spatial_dim,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$

    `t` is not used, but it's kept in the arguments because for the attention layer function signature to
    match with `ResidualBlock`.
    """

    def __init__(self, n_channels, spatial_dims: int = 2):
        super().__init__()
        self.conv = CONV_T_LAYERS[spatial_dims](
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=(4,) * spatial_dims,
            stride=(2,) * spatial_dims,
            padding=(1,) * spatial_dims,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$

    `t` is not used, but it's kept in the arguments because for the attention layer function signature to
    match with `ResidualBlock`.
    """

    def __init__(self, n_channels, spatial_dims: int = 2):
        super().__init__()
        self.conv = CONV_LAYERS[spatial_dims](
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=(3,) * spatial_dims,
            stride=(2,) * spatial_dims,
            padding=(1,) * spatial_dims
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)
