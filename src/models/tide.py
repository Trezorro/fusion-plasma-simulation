"""TiDE baseline adapter for the PlasmaFlow / UnFlowModule harness.

TiDE (Das et al., 2023) is an MLP encoder-decoder long-horizon forecaster whose distinctive
feature is a *temporal decoder*: each horizon step's decoded vector is combined with that same
step's projected covariates, a direct highway from the future covariate at step t to the
prediction at step t. This is the deterministic analogue of the flow UNet's time-aligned
covariate channels, and it is why TiDE is the covariate-aware baseline of choice here.

Covariate routing (the "mark slot"). In the Time-Series-Library, TiDE ingests dynamic
covariates through the `x_mark` tensors (upstream: calendar features). PlasmaFlow has no
calendar time, so we route the real control covariates C through that slot instead:
    x_mark_enc  = C over history  [B, Wh, c_channels]
    batch_y_mark = C over the full [history|horizon] window [B, Wh+Wf, c_channels]
TiDE's forward reconstructs the full covariate span and feeds its future portion to the
temporal decoder. The vendored model was edited so the mark width follows c_channels instead
of the fixed calendar-frequency map (see src/models/tslib/TiDE.py, "VENDOR EDIT").

Native lookback-to-horizon convention: seq_len = history (Wh), pred_len = horizon (Wf). No
seq_len = Wh + Wf spanning hack (that is only needed for iTransformer, whose covariate has no
native future path).

Network contract (matches what UnFlowModule calls):
    forward(x, t, conditioning_input) -> [B, x_channels, horizon]
`x` (prior noise) and `t` are ignored.
"""
from types import SimpleNamespace

import torch
import torch.nn as nn

from src.models.tslib.TiDE import Model as _TiDENet


class TiDE(nn.Module):
    def __init__(
        self,
        seq_len=256,       # lookback = data.history_length
        pred_len=256,      # horizon = data.seq_length
        input_channels=5,
        c_channels=3,
        d_model=256,
        d_ff=256,
        e_layers=2,
        d_layers=2,
        dropout=0.1,
        freq="h",          # fallback mark width only when c_channels == 0
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
            label_len=0,      # read but unused on the forecast path
            pred_len=pred_len,
            d_model=d_model,
            d_ff=d_ff,        # temporal-decoder hidden width
            e_layers=e_layers,
            d_layers=d_layers,
            c_out=1,          # per-channel internal decode width (channel-independent target)
            dropout=dropout,
            freq=freq,
            c_channels=c_channels,  # VENDOR EDIT: sets the mark (covariate) width
            use_norm=use_norm,
        )
        self.model = _TiDENet(configs)

    def forward(self, x, t, conditioning_input=None):
        xh = conditioning_input["x_history"]          # [B, Xc, Wh]
        x_enc = xh.transpose(1, 2)                     # [B, Wh, Xc]

        c = conditioning_input.get("c")
        if self.c_channels > 0 and c is not None:
            if c.shape[-1] != self.seq_len + self.pred_len:
                raise ValueError(
                    f"c length {c.shape[-1]} != history+horizon {self.seq_len + self.pred_len}; "
                    "TiDE expects c over the full [history|horizon] window."
                )
            c_full = c.transpose(1, 2)                 # [B, Wh+Wf, Cc]
            x_mark_enc = c_full[:, : self.seq_len, :]  # C over history
            batch_y_mark = c_full                      # forward slices [:, -pred_len:] as C_future
        else:
            x_mark_enc = None
            batch_y_mark = None                        # TiDE zero-fills the mark (plain TiDE)

        out = self.model(x_enc, x_mark_enc, None, batch_y_mark)  # [B, Wf, Xc]
        return out.transpose(1, 2)                     # [B, Xc, Wf]


if __name__ == "__main__":
    # Covariate path: real C over the full window.
    net = TiDE(seq_len=256, pred_len=256, input_channels=5, c_channels=3)
    cond = {"x_history": torch.randn(2, 5, 256), "c": torch.randn(2, 3, 512)}
    y = net(torch.randn(2, 5, 256), torch.ones(2), cond)
    assert y.shape == (2, 5, 256), y.shape
    print("TiDE adapter OK:", tuple(y.shape))

    # Arbitrary covariate count (mark slot freed from the calendar map).
    net8 = TiDE(seq_len=256, pred_len=256, input_channels=5, c_channels=8)
    cond8 = {"x_history": torch.randn(2, 5, 256), "c": torch.randn(2, 8, 512)}
    assert net8(torch.randn(2, 5, 256), torch.ones(2), cond8).shape == (2, 5, 256)
    print("TiDE 8-covariate OK")

    # X-only degradation (no covariates -> zero-filled calendar-width mark).
    net0 = TiDE(seq_len=256, pred_len=256, input_channels=5, c_channels=0)
    y0 = net0(torch.randn(2, 5, 256), torch.ones(2), {"x_history": torch.randn(2, 5, 256)})
    assert y0.shape == (2, 5, 256), y0.shape
    print("TiDE X-only OK")
