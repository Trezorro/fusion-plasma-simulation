"""DLinear baseline adapter for the PlasmaFlow / UnFlowModule harness.

Channel-independent trend+remainder linear forecaster (Zeng et al. 2023). It is the
floor baseline: covariates cannot inform the X channels, so it is fed X-history only.

Network contract (matches what UnFlowModule calls):
    forward(x, t, conditioning_input) -> [B, x_channels, seq_length]
`x` (prior noise) and `t` are ignored; the forecast comes from conditioning_input["x_history"].
"""
from types import SimpleNamespace

import torch
import torch.nn as nn

from src.models.tslib.DLinear import Model as _DLinearNet


class DLinear(nn.Module):
    def __init__(self, seq_len=256, pred_len=256, input_channels=5, moving_avg=25, **kwargs):
        super().__init__()
        self.x_channels = input_channels  # config.update_model_input_channels sets this from data.cols.x
        configs = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=seq_len,      # lookback = x_history length
            pred_len=pred_len,    # horizon = target seq_length
            enc_in=input_channels,
            moving_avg=moving_avg,
        )
        self.model = _DLinearNet(configs, individual=True)

    def forward(self, x, t, conditioning_input=None):
        xh = conditioning_input["x_history"]        # [B, x_channels, Wh]
        x_enc = xh.transpose(1, 2)                   # [B, Wh, x_channels]
        out = self.model(x_enc, None, None, None)    # [B, pred_len, x_channels]
        return out.transpose(1, 2)[:, : self.x_channels, :]  # [B, x_channels, pred_len]


if __name__ == "__main__":
    net = DLinear(seq_len=256, pred_len=256, input_channels=5, moving_avg=25)
    cond = {
        "x_history": torch.randn(2, 5, 256),
        "c": torch.randn(2, 3, 512),
    }
    y = net(torch.randn(2, 5, 256), torch.ones(2), cond)
    assert y.shape == (2, 5, 256), y.shape
    print("DLinear adapter OK:", tuple(y.shape))
