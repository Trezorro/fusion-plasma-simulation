# Deterministic forecasting baselines

This document explains the deterministic (point-forecast) baselines added to PlasmaFlow:
**DLinear**, **PatchTST**, **iTransformer**, and **TiDE**. It is written for reproducibility and
integrity first: every place where our conventions had to be reconciled with the original
models is stated explicitly, including the decisions we made and why, so a reader (or
reviewer) can check that the comparison is fair and that nothing about the published models
was silently altered.

## 1. Why these baselines exist (the argument they serve)

The scientific claim of the paper is that predicting the *timing* and *shape* of transient
events (ELMs) together is hard for models trained under a point-wise loss. Given a history,
the timing of the next event is uncertain, and the estimate that minimises expected squared
error is an average across the plausible event times. A deterministic regressor therefore
tends to blur or miss peaks whose timing it cannot know.

These baselines exist to demonstrate exactly that failure. They are expected to recover peak
shape only when the true timing is supplied to them through covariates (the *timing-oracle*
experiment), and to blur or miss peaks otherwise. The point is not to make them win: it is to
make them strong enough that their failure on peak timing is credible and cannot be dismissed
as a weak baseline.

Because the whole comparison lives in the evaluation layer, **the baselines are scored by the
same metric objects, on the same windows, with the same normalization as the flow model.**
This is the single most important integrity constraint. A baseline evaluated in a separate
harness or repo is not comparable and would defeat the purpose. Everything below follows from
this constraint.

## 2. The three models and the channel-independence caveat

These span the dominant design axes in current long-horizon multivariate forecasting, and
our window length (history 256 + horizon 256 by default) sits inside the 96 to 720 band these
models were designed for.

| Model | Reference | Role | Covariate-aware? |
|---|---|---|---|
| DLinear | Zeng et al., AAAI 2023 (arXiv:2205.13504) | Trend+remainder decomposition, per-channel linear map. The floor: simple enough that it has embarrassed many transformers, so it protects the comparison against the objection that the baselines were weak. | No (channel-independent) |
| PatchTST | Nie et al., ICLR 2023 (arXiv:2211.14730) | Patches each channel into tokens, processes channels independently with shared weights. Strong and stable. | No (channel-independent) |
| iTransformer | Liu et al., ICLR 2024 (arXiv:2310.06625) | Tokenises each variate as a whole and attends *across variates* rather than across time. | Yes (variate-collapsed) |
| TiDE | Das et al., TMLR 2023 (arXiv:2304.08424) | MLP encoder-decoder with a *temporal decoder* that feeds each horizon step's future covariate to the aligned prediction step. Time-aligned covariate handling. | Yes (time-aligned) |

**Honesty point that must survive into the writeup.** DLinear and PatchTST are
*channel-independent*: the covariate channels cannot inform the X channels. So a covariate
result from those two says nothing about covariates in general. **iTransformer is the only
member where covariate conditioning actually happens** (attention across variates). We keep
this distinction explicit in the code (DLinear/PatchTST are fed X-history only) and it must be
kept explicit in the paper.

## 3. Integrity principles applied

1. **Vendor, do not reimplement.** All three models are copied from the official
   [Time-Series-Library](https://github.com/thuml/Time-Series-Library) (thuml/TSlib) rather
   than reimplemented. Reimplementation risks subtle deviations from the published models that
   a reviewer could challenge. The vendored copies live in [src/models/tslib/](../src/models/tslib/)
   and are kept as close to upstream as possible; every edit is enumerated in §7.
2. **Same evaluation, unchanged.** The baselines reuse the flow model's evaluation pipeline in
   full (metrics, peak features, surrogate mode labels, HDF5 cache, plotters). We did not touch
   the metrics, the evaluation contract, the data pipeline, or the normalization.
3. **Every reconciliation is documented.** Where our data conventions did not match TSlib's
   native lookback-to-horizon convention, the choice we made is stated below (§5, §6, §8), with
   the reasoning, so the numbers are traceable to explicit decisions.

## 4. How the baselines plug in (harness reuse)

We did **not** write a new training module. PlasmaFlow already contains a deterministic
regression harness: [`UnFlowModule`](../src/models/UnFlow.py), an ablation that subclasses
[`FlowModule`](../src/models/flow.py). It trains a network by a single forward pass at constant
`t=1` and regresses the output directly to the future window with MSE, and it already produces
the full `evaluation_output` dict, drives the HDF5 cache, and works with the plotters.

The baselines are therefore just **networks** selected inside `UnFlowModule`. The network is
called as:

```python
# in UnFlowModule.training_step / inference / evaluate
pred = self.model(noise_sample, t_dummy, conditioning_input=conditioning_input)
```

so each baseline network implements this exact signature:

```python
def forward(self, x, t, conditioning_input=None) -> Tensor:  # -> [B, x_channels, horizon]
```

- `x` (the prior sample; `constant` prior gives a fixed `0.5` tensor) and `t` are **ignored**
  by the deterministic backbones.
- `conditioning_input` is the dict from the data module: `x_history` `[B, x_channels, Wh]`,
  `c` `[B, c_channels, Wh+Wf]`, `position_sequence`, `label`.
- The output is the X channels in the same order as `data.cols.x`, shape `[B, x_channels, Wf]`.
  Channel order matters: the mode classifier locates `PD` positionally via
  `data.cols.x.index("PD")`, so channels are never permuted or dropped.

The networks are registered in `FlowModule.MODEL_OPTIONS` ([flow.py](../src/models/flow.py)) as
`DLinear`, `PatchTST`, `ITransformer`, exactly like `ConditionalUNet`, and selected through the
config's `model.params.model` string.

Adapter files (thin wrappers around the vendored TSlib `Model`):

| Adapter | File | Vendored model |
|---|---|---|
| DLinear | [src/models/dlinear.py](../src/models/dlinear.py) | `tslib/DLinear.py` |
| PatchTST | [src/models/patchtst.py](../src/models/patchtst.py) | `tslib/PatchTST.py` |
| iTransformer | [src/models/itransformer.py](../src/models/itransformer.py) | `tslib/iTransformer.py` |
| TiDE | [src/models/tide.py](../src/models/tide.py) | `tslib/TiDE.py` |

## 5. Input representation, and how it was reconciled with the UNet

This is the crux of comparability, so it is spelled out in full.

### How the flow UNet stacks its inputs

The conditional UNet ([unet_conditional.py](../src/models/unet_conditional.py),
`condition_sequentially`, the `conditioning_method: sequence` used in the paper) builds its
input like this:

- concatenate `x_history` and the forecast-region tensor along **time** into a single
  `Wh + Wf` (= 512) sequence;
- add one binary indicator channel (1 over history, 0 over the forecast region);
- concatenate `c` (which spans the full 512 window) along **channels**;
- append positional-embedding channels.

Crucially, `c` carries real values over the **entire** `[history | horizon]` window in the
data module, so the UNet already sees future covariate values whenever such covariates are
configured.

### How each baseline is fed

We mirror the UNet's *information access*, adapted to each model's native convention:

- **DLinear and PatchTST** are channel-independent and are native lookback-to-horizon
  forecasters. They are fed **X-history only**: lookback = `x_history` `[B, 5, 256]`, predict
  horizon `[B, 5, 256]`. No covariates, no indicator channel. This is their native TSlib
  convention and, given channel independence, feeding them covariates would be misleading
  (the covariate channels could never inform the X channels anyway).

- **iTransformer** attends across variates, so we assemble a window that matches the UNet's
  variate stack:

  ```
  W = [B, V, L]   with V = x_channels + c_channels,  L = seq_len = Wh + Wf = 512
    X channels: x_history over [0:Wh],  ZEROS over [Wh:L]   (no known X future;
                                                             the UNet fills this with the prior x)
    C channels: c over the full [0:L]   (future covariate values present)
  ```

  This is transposed to `[B, L, V]`, run through iTransformer (`seq_len=512, pred_len=256,
  enc_in=V`), transposed back, and the first `x_channels` are sliced out to `[B, 5, 256]`.
  Because the future covariate values live in the C variate-tokens, cross-variate attention
  lets them inform the X projections. This is the mechanism that makes the timing-oracle
  experiment possible.

  We deliberately omit the UNet's explicit binary history/forecast indicator variate: the
  zero-padding of the X future already marks the boundary. It can be added later if it ever
  measurably matters.

- **TiDE** is fed in its native lookback-to-horizon form, with no spanning-window hack. The
  target lookback is `x_history` `[B, 5, 256]` and the horizon is the 256-step forecast. The
  control covariates enter through TiDE's *mark* channel (see below), carrying real values over
  the full `[history | horizon]` window. Because TiDE forecasts each target channel
  independently, all 5 X channels share the same covariate context and are stacked to
  `[B, 5, 256]` on output.

### TiDE covariate routing and the temporal decoder (method summary)

This is the property that makes TiDE the covariate-aware baseline worth running, so it is
stated at method-section altitude.

TiDE (Das et al., 2023) is an MLP encoder-decoder. A dense encoder maps the target lookback
together with the projected dynamic covariates over the whole `[history | horizon]` span into a
latent vector; a dense decoder expands it to one vector per horizon step; and a *temporal
decoder* combines each horizon step's decoded vector with the projected covariate of that same
step to produce the prediction. A global linear residual maps the lookback directly to the
horizon and is added on top. The temporal decoder is a direct path from the covariate at
horizon step `t` to the prediction at step `t`: the deterministic counterpart of the flow
UNet's time-aligned covariate channels, delivered by a standard long-horizon model.

In the Time-Series-Library, TiDE ingests dynamic covariates through the model's *mark* tensors,
which upstream carry calendar features (hour, weekday, ...). PlasmaFlow has no meaningful
calendar time, so we route the real control covariates `c` through that same slot:

```
x_mark_enc   = c over history                 [B, Wh, c_channels]
batch_y_mark = c over [history | horizon]      [B, Wh+Wf, c_channels]
```

TiDE's forward reconstructs the full covariate span and passes its future portion
(`batch_y_mark[:, -Wf:]`) to the temporal decoder. There is no separate "oracle" switch and no
masking: as everywhere in these baselines, whichever covariates are listed in `data.cols.c` are
fed in full, over both history and horizon (§6). Supplying a timing-bearing covariate (e.g.
IPLA) therefore lets the temporal decoder recover the event at the aligned step; withholding it
leaves the estimator to average over plausible event times. TiDE's own paper demonstrates this
directly: on a semi-synthetic series with injected event features, the model with the temporal
decoder recovers the events after a single epoch and the model without it misses them.

One vendored edit makes this possible. Upstream fixes the mark width to a calendar-frequency
map (`feature_dim = freq_map[freq]`); we free it to follow the configured covariate count
(`feature_dim = c_channels`), so any number of control covariates can pass through the temporal
decoder (§7, edit 4). When no covariates are configured (`c_channels == 0`) the model falls
back to the upstream zero-filled calendar-width mark, degrading to plain TiDE.

Rationale for the split: comparability within PlasmaFlow matters more than matching TSlib's
defaults, but where a model is natively a lookback-to-horizon forecaster (DLinear, PatchTST)
we respect that so we do not misrepresent what the published model does.

## 6. The timing-oracle experiment (how it is run)

There is **no masking flag** and no separate "oracle on/off" switch buried in the code. The
oracle is controlled entirely by **which covariates are listed in `data.cols.c`**:

- Whatever covariates are configured are fed to iTransformer in full, over the whole
  `[history | horizon]` window, as extra variate-tokens (§5).
- To run the timing-oracle experiment, add covariates that carry ELM-timing information (for
  example the non-reference plasma current IP, or shape-related signals that are not truly
  exogenous). iTransformer can then exploit their future values.
- To run the honest, non-oracle case, list only genuinely exogenous covariates (or none).

Channel counts follow the data automatically: `config.update_model_input_channels`
([config.py](../src/config.py)) sets `model_params.input_channels = len(data.cols.x)` and
`model_params.c_channels = len(data.cols.c)` before the model is built, so `enc_in` for
iTransformer always tracks the configured covariate set. DLinear/PatchTST ignore `c_channels`.

This design keeps the experiment a data-config change, not a code change, which is exactly
what we want for reproducibility: the oracle condition is a line in the config, visible in the
logged run config.

## 7. Vendoring: exact provenance and the edits we made

Upstream: `thuml/Time-Series-Library` (submodule at `giants/Time-Series-Library`), pinned at
commit **`4e938a1767106324dd753b2a44832bf870a0252e`** (2026-04-18).

Vendored subset under [src/models/tslib/](../src/models/tslib/):

| File | Purpose | Upstream origin |
|---|---|---|
| `DLinear.py` | DLinear `Model` | `models/DLinear.py` |
| `PatchTST.py` | PatchTST `Model` | `models/PatchTST.py` |
| `iTransformer.py` | iTransformer `Model` | `models/iTransformer.py` |
| `TiDE.py` | TiDE `Model` (self-contained, no layer deps) | `models/TiDE.py` |
| `Autoformer_EncDec.py` | `series_decomp` (DLinear dep) | `layers/Autoformer_EncDec.py` |
| `Transformer_EncDec.py` | `Encoder`, `EncoderLayer` | `layers/Transformer_EncDec.py` |
| `SelfAttention_Family.py` | `FullAttention`, `AttentionLayer` | `layers/SelfAttention_Family.py` |
| `Embed.py` | `PatchEmbedding`, `DataEmbedding_inverted` | `layers/Embed.py` |
| `masking.py` | `TriangularCausalMask`, `ProbMask` | `utils/masking.py` |

**The only edits made to upstream code**, so a reviewer can diff against `4e938a1`:

1. **Import paths.** `from layers.X` / `from utils.masking` rewritten to package-relative
   imports (`from .X` / `from .masking`). No logic change.
2. **`use_norm` gate (PatchTST, iTransformer, TiDE).** See §8. Upstream *these ports* hardcode
   an always-on instance normalization with no flag; we wrapped the normalize/de-normalize
   blocks in `if self.use_norm:` and read `self.use_norm = getattr(configs, 'use_norm', True)`.
   Default preserves upstream behavior; the config turns it off. For TiDE this matters extra:
   the inline norm interacts with the global linear residual, so we disable it (§8).
3. **Trim in `SelfAttention_Family.py`.** Removed the `ReformerLayer` and
   `TwoStageAttentionLayer` classes and their two module-level imports (`reformer_pytorch`,
   `einops`). Neither class is used by PatchTST/iTransformer (which only use `FullAttention`
   and `AttentionLayer`), and the imports would otherwise force two extra pip dependencies on
   the cluster. After the trim the vendored set needs only `torch` and `numpy`.

4. **Covariate mark width (TiDE only).** Upstream fixes TiDE's dynamic-covariate ("mark")
   width to a calendar-frequency map (`feature_dim = freq_map[freq]`, e.g. 4 for hourly). We
   changed it to `feature_dim = getattr(configs, 'c_channels', 0) or freq_map[freq]`, so the
   temporal decoder accepts an arbitrary number of PlasmaFlow control covariates. This is the
   edit that lets real covariates flow through TiDE's temporal decoder (§5). No math changed;
   only the width of an existing projection.

Everything else (the model math, the attention, the embeddings, the decomposition, the TiDE
temporal decoder) is copied verbatim.

### Adapter details worth knowing

- The TSlib models are constructed from a `Configs` namespace built by each adapter from the
  config's `model_params` (a `types.SimpleNamespace`). `task_name` is fixed to
  `'long_term_forecast'` (any other value makes the upstream `forward` return `None`).
- `patch_len`/`stride` (PatchTST) and `individual=True` (DLinear) are **constructor
  arguments**, not config fields, matching upstream. `individual=True` gives DLinear its
  per-channel (channel-independent) linear heads.
- iTransformer and PatchTST are always called with `x_mark_enc=None`; the models tolerate this
  (PatchTST never uses marks; `DataEmbedding_inverted` handles `None`).

## 8. Instance normalization (RevIN) decision

The data module already min-max normalizes signals to roughly `[0, 1]` using train-split
statistics. PatchTST, iTransformer, and TiDE in TSlib apply an additional per-window
"non-stationary" normalization (subtract the window mean, divide by std, then de-normalize the
output back to scale). For TiDE there is an extra reason to disable it: the inline norm and the
model's global linear residual (lookback-to-horizon) can work against each other, so we keep
`use_norm: false` and let the single train-split normalization stand.

**Decision: disable it by default** (`use_norm: false` in the baseline configs), for
comparability with the flow model, which has no such internal normalization. Keeping it would
give the transformers an internal per-window instance normalization the flow model lacks,
which a reviewer could reasonably call an apples-to-oranges comparison.

Reproducibility note: because this TSlib port dropped the upstream `use_norm` flag, we
re-introduced it ourselves (§7, edit 2). The gate defaults to `True` (published behavior) at
the code level, but the shipped baseline configs set it to `false`. Set `use_norm: true` in a
config to reproduce the published-model behavior. Disabling it also removes a minor artifact:
for iTransformer the assembled window zero-pads the X future, which would otherwise pollute the
per-window mean/std of the X channels.

## 9. Loss

Training uses **MSE** (`loss: MSELoss`), because the mode-averaging argument (§1) is
specifically about the squared-error estimator. Reporting L1 as a secondary metric is fine, but
MSE is the objective that carries the argument. This is inherited unchanged from
`UnFlowModule`.

## 10. Deterministic single-sample behavior

A deterministic model produces one prediction per window. The per-window plots that repeat a
window (the `window_set` path draws several copies of a batch) will therefore show identical
overlaid traces with zero spread. **This is expected and is itself informative**: it visualises
the collapse to a mean. The evaluation path used by `UnFlowModule` does not divide by the
across-sample variance, so identical samples do not cause errors. (The one latent
`output_var / input_var` division lives in the disabled `complexnet.py` and is not on this
path.)

## 11. How to run

The baselines are selected via the config's model-include mechanism (the same one the flow
model uses). In `configs/plasmaflow.yaml`, set the `model:` key to one of the baseline configs:

```yaml
# configs/plasmaflow.yaml
model: configs/dlinear.yaml       # or configs/patchtst.yaml / configs/itransformer.yaml
```

Then run as usual:

```bash
python run.py run_name=dlinear_baseline
# or on Snellius:
bash src/HPC_setup/submit_remote_job_snellius.sh dlinear_baseline
```

Shipped baseline configs:

| Config | Model | Notes |
|---|---|---|
| [configs/dlinear.yaml](../configs/dlinear.yaml) | DLinear | X-history only; `moving_avg` kernel is the only structural knob |
| [configs/patchtst.yaml](../configs/patchtst.yaml) | PatchTST | X-history only; `patch_len`/`stride`, `d_model`, `n_heads`, `e_layers`, `use_norm: false` |
| [configs/itransformer.yaml](../configs/itransformer.yaml) | iTransformer | covariate-aware (variate-collapsed); `seq_len: 512`, `use_norm: false`; oracle driven by `data.cols.c` |
| [configs/tide.yaml](../configs/tide.yaml) | TiDE | covariate-aware (time-aligned); native `seq_len: 256`, `use_norm: false`; oracle driven by `data.cols.c` |

Each config keeps `Class: UnFlowModule`, `prior: 'constant'`, `loss: MSELoss`, and the same
optimizer block as `configs/unflow.yaml`.

### Quick shape self-check (no data needed)

Each adapter has a `__main__` that runs a synthetic forward pass and asserts the output shape:

```bash
python -m src.models.dlinear
python -m src.models.patchtst
python -m src.models.itransformer
python -m src.models.tide
```

## 12. Known couplings and caveats

- **iTransformer `seq_len` must equal `Wh + Wf`** (512 with the default history 256 + horizon
  256), because its assembled window concatenates `x_history` and the zero-padded future, and
  `c` must span that full length. If `data.history_length` or `data.seq_length` change, update
  `seq_len`/`pred_len` in `configs/itransformer.yaml` to match. This is a hardcoded coupling,
  noted in the config.
- **Channel-count auto-sync.** `input_channels`/`c_channels` in the baseline `model_params` are
  overwritten by `config.update_model_input_channels` from `data.cols`. The values in the YAML
  are documentation/fallback; the data config is the source of truth.
- **HDF5 cache assumes 5 X-channels** on the read path (`hdf_cache.py`). The baselines emit 5
  X-channels, so their caches interoperate with the flow model's in the eval notebooks
  (`kwali_plots.py`, `peak_analysis.py`). Changing the X-channel count would require updating
  that constant.
- **`conditioning` is documentation, not a data gate, for the baselines.** In `model_params`,
  `conditioning` is a *ConditionalUNet-internal* field (it tells the UNet which keys to stack).
  The baseline adapters ignore it and read `conditioning_input["x_history"]` (and `c` for
  iTransformer) directly. Whether those keys exist is decided by the **data** config
  (`data.history_length > 0`, `data.cols.c`), inherited from `plasmaflow.yaml`, not by
  `model_params.conditioning`. The baseline configs still list the keys the adapter consumes
  (`["x_history"]`, or `["x_history", "c"]` for iTransformer) so the config is honest, and set
  `positional_encoding: null`; this pair also satisfies `FlowModule._validate_configuration`,
  which asserts `('position_sequence' in conditioning) == (positional_encoding is not None)`.
  The adapters swallow `conditioning`/`positional_encoding` (and any other unused keys) via
  `**kwargs`.

## 13. Summary of decisions (traceability)

| # | Topic | Decision | Where |
|---|---|---|---|
| 1 | Covariates for DLinear/PatchTST | X-history only (channel-independent) | §2, §5 |
| 2 | Oracle mechanism | Driven by `data.cols.c`; no masking flag; covariate-aware models (iTransformer, TiDE) | §6 |
| 3 | Instance norm | Disabled by default (`use_norm: false`); flag re-added on top of the port; TiDE also fights its linear residual | §8 |
| 4 | Loss | MSE | §9 |
| 5 | Single-sample plots | Expected zero-spread; no code change | §10 |
| 6 | Sourcing | Vendored from TSlib `4e938a1`, four enumerated edits | §7 |
| 7 | TiDE covariate path | Routed through the mark slot; mark width freed to `c_channels`; native lookback/horizon | §5, §7 |
