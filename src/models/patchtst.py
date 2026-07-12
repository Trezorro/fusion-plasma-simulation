"""PatchTST baseline adapter for the PlasmaFlow / UnFlowModule harness.

Patch-tokenised, channel-independent transformer forecaster (Nie et al. 2023). Channel
independent, so it is fed X-history only (covariates cannot inform the X channels).

Network contract (matches what UnFlowModule calls):
    forward(x, t, conditioning_input) -> [B, x_channels, seq_length]
`x` (prior noise) and `t` are ignored; the forecast comes from conditioning_input["x_history"].
"""
from types import SimpleNamespace

import torch
import torch.nn as nn

from src.models.tslib.PatchTST import Model as _PatchTSTNet


class PatchTST(nn.Module):
    def __init__(
        self,
        seq_len=256,
        pred_len=256,
        input_channels=5,
        d_model=128,
        n_heads=8,
        e_layers=2,
        d_ff=256,
        dropout=0.1,
        factor=1,
        activation="gelu",
        patch_len=16,
        stride=8,
        use_norm=False,
        **kwargs,
    ):
        super().__init__()
        self.x_channels = input_channels  # config.update_model_input_channels sets this from data.cols.x
        configs = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=input_channels,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            factor=factor,
            activation=activation,
            use_norm=use_norm,
        )
        self.model = _PatchTSTNet(configs, patch_len=patch_len, stride=stride)

    def forward(self, x, t, conditioning_input=None):
        xh = conditioning_input["x_history"]        # [B, x_channels, Wh]
        x_enc = xh.transpose(1, 2)                   # [B, Wh, x_channels]
        out = self.model(x_enc, None, None, None)    # [B, pred_len, x_channels]
        return out.transpose(1, 2)[:, : self.x_channels, :]  # [B, x_channels, pred_len]


if __name__ == "__main__":
    net = PatchTST(seq_len=256, pred_len=256, input_channels=5)
    cond = {
        "x_history": torch.randn(2, 5, 256),
        "c": torch.randn(2, 3, 512),
    }
    y = net(torch.randn(2, 5, 256), torch.ones(2), cond)
    assert y.shape == (2, 5, 256), y.shape
    print("PatchTST adapter OK:", tuple(y.shape))
