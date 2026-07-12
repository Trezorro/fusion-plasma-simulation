"""iTransformer baseline adapter for the PlasmaFlow / UnFlowModule harness.

Inverted transformer that tokenises each variate as a whole and attends ACROSS variates
(Liu et al. 2024). This is the covariate-aware deterministic baseline: control covariates
enter as extra variate-tokens, so their (possibly future) values can inform the X forecast
through cross-variate attention.

Input window (mirrors the UNet's `condition_sequentially` variate stack, minus the binary
indicator channel):
    W = [B, V, L]  with  V = x_channels + c_channels,  L = seq_len = history + horizon
      - X channels: x_history over [0:history], ZEROS over the forecast span (no known X future)
      - C channels: c over the full [0:L] window (future covariates present -> the oracle path,
        active whenever timing-bearing covariates are configured in data.cols.c)

Network contract (matches what UnFlowModule calls):
    forward(x, t, conditioning_input) -> [B, x_channels, horizon]
`x` (prior noise) and `t` are ignored.
"""
from types import SimpleNamespace

import torch
import torch.nn as nn

from src.models.tslib.iTransformer import Model as _ITransformerNet


class ITransformer(nn.Module):
    def __init__(
        self,
        seq_len=512,       # history + horizon (the full stacked window)
        pred_len=256,      # horizon = target seq_length
        input_channels=5,
        c_channels=3,
        d_model=128,
        n_heads=8,
        e_layers=2,
        d_ff=256,
        dropout=0.1,
        factor=1,
        activation="gelu",
        embed="timeF",
        freq="h",
        use_norm=False,
        **kwargs,
    ):
        super().__init__()
        # config.update_model_input_channels sets input_channels/c_channels from data.cols.x/.c
        self.x_channels = input_channels
        self.c_channels = c_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        configs = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=input_channels + c_channels,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            factor=factor,
            activation=activation,
            embed=embed,
            freq=freq,
            use_norm=use_norm,
        )
        self.model = _ITransformerNet(configs)

    def _build_window(self, conditioning_input):
        xh = conditioning_input["x_history"]          # [B, Xc, Wh]
        B, Xc, Wh = xh.shape
        pad = self.seq_len - Wh
        if pad < 0:
            raise ValueError(f"seq_len {self.seq_len} < history length {Wh}")
        x_block = torch.cat(
            [xh, xh.new_zeros(B, Xc, pad)], dim=-1
        )                                             # [B, Xc, seq_len]

        c = conditioning_input.get("c")
        if self.c_channels > 0 and c is not None:
            if c.shape[-1] != self.seq_len:
                raise ValueError(
                    f"c length {c.shape[-1]} != seq_len {self.seq_len}; "
                    "iTransformer expects c over the full [history|horizon] window."
                )
            return torch.cat([x_block, c], dim=1)     # [B, Xc+Cc, seq_len]
        return x_block

    def forward(self, x, t, conditioning_input=None):
        window = self._build_window(conditioning_input)   # [B, V, seq_len]
        x_enc = window.transpose(1, 2)                      # [B, seq_len, V]
        out = self.model(x_enc, None, None, None)           # [B, pred_len, V]
        return out.transpose(1, 2)[:, : self.x_channels, :]  # [B, x_channels, pred_len]


if __name__ == "__main__":
    net = ITransformer(seq_len=512, pred_len=256, input_channels=5, c_channels=3)
    cond = {
        "x_history": torch.randn(2, 5, 256),
        "c": torch.randn(2, 3, 512),
    }
    y = net(torch.randn(2, 5, 256), torch.ones(2), cond)
    assert y.shape == (2, 5, 256), y.shape
    print("iTransformer adapter OK:", tuple(y.shape))

    # X-only degradation (no covariates)
    net2 = ITransformer(seq_len=256, pred_len=256, input_channels=5, c_channels=0)
    y2 = net2(torch.randn(2, 5, 256), torch.ones(2), {"x_history": torch.randn(2, 5, 256)})
    assert y2.shape == (2, 5, 256), y2.shape
    print("iTransformer X-only OK:", tuple(y2.shape))
