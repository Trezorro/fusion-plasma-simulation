# Configuration Reference

This document is the complete config reference for PlasmaFlow. The main config file is `configs/plasmaflow.yaml`. Configs are written in YAML and loaded through OmegaConf, which supports variable interpolation: a value like `${data.seq_length}` reads the value of another key at resolution time, so related settings stay in sync without duplication.

## How config loading works

Config assembly happens in `src/config.py` and proceeds in order:

1. `load_config_from_file('plasmaflow')` loads `configs/plasmaflow.yaml` into an OmegaConf object.
2. If the top-level `model` key is a string path, that model YAML is loaded and merged in. Currently the model block is inline in `plasmaflow.yaml`, not a separate file, so this step is a no-op for the default config.
3. `OmegaConf.from_cli()` reads CLI arguments in `key=value` or `key.subkey=value` form and merges them on top of the file values. CLI values override file values.
4. `update_model_input_channels(conf)` auto-updates `model.params.model_params.input_channels` to `len(data.cols.x)` and `c_channels` to `len(data.cols.c)`. You do not have to keep these in sync manually.
5. `wandb.init(config=conf)` uploads the config. After init, `get_current_config(wandb_only=True)` is the canonical source of truth: it picks up any values injected by wandb sweeps.

### CLI override examples

```bash
python run.py run_name=experiment_A epochs=200
python run.py model.params.prior=brownian data.batch_size=64
python run.py reeval=True  # triggers reeval mode
```

Any key in the config tree can be overridden from the CLI using its dotted path. CLI overrides take precedence over the YAML file but can still be overridden by a wandb sweep.

## Top-level run parameters

| Key | Type | Default | Description |
|---|---|---|---|
| `run_name` | str | `testrunAAA` | Identifier for this run; used in wandb, the checkpoint directory name, and PDF plot paths |
| `epochs` | int | 120 | Maximum training epochs |
| `limit_train_batches` | int | 150 | Number of batches per training epoch (limits epoch length) |
| `limit_val_batches` | int | 30 | Number of batches per validation epoch |
| `batch_size` | int | 128 | Training batch size |
| `patience` | int | 20 | Early stopping patience (epochs without val loss improvement) |
| `gradient_clip_val` | float | 10 | Global gradient norm clip threshold; the high value is a safety net, not a tuning parameter |
| `skip_log_summary` | bool | false | If true, skips printing the model summary at startup |
| `test_cache_name` | str or null | null | If set, enables HDF5 test result caching under this name |
| `test_cache_mode` | str | `create` | `create` to generate and save; `use` to load from an existing cache |
| `resume_id` | str | (absent) | wandb run ID to resume; triggers resume mode |
| `resume_name` | str | (absent) | wandb run name for resume (must match `resume_id`) |
| `features` | list | (see yaml) | Feature flags (list of strings); controls which optional behaviors are active |

## window_set

A list of `[shot_number, time_seconds]` pairs consumed by `evaluate_window_set()`. These specific shot windows are always evaluated after training and produce PDF plots.

```yaml
window_set:
  - [57013, 0.94]   # L-H transition example
  - [77604, 0.72]   # high NBI example
```

Remove entries to suppress specific plots. Set the key to `[]` to skip all window-set evaluation.

## evaluation

Controls the `PlotsCallback` and `model.evaluate()` behavior.

| Key | Type | Default | Description |
|---|---|---|---|
| `n_steps` | int | 100 | ODE integration steps for all evaluation calls (including window_set and test) |
| `solve_method` | str | `simple` | Integration method; only `simple` (forward Euler) is active; torchdiffeq adaptive solvers are preserved but bypassed |
| `max_n` | int | 64 | Max samples drawn from the test set for validation plots |
| `val_every_n_epochs` | int | 5 | Full evaluation fires at epochs where `epoch % val_every_n_epochs == 1` (so 1, 6, 11, ...) |
| `train_every_n_epochs` | int | 20 | Same logic applied to train-data plots; set to 0 to disable |
| `scrutinize_epochs` | int | 3 | All epochs `<=` this value always get full evaluation regardless of `val_every_n_epochs` |
| `plot_functions` | list | (see yaml) | List of plot function specs; each entry has a `key` and optional `n`, `log_key`, and function-specific kwargs |

### plot_functions entries

Each entry specifies one plot to generate:

```yaml
- key: animated_traces     # function name from PlotsCallback.PLOT_FN_OPTIONS
  n: 5                      # number of samples to plot
- key: multi_channel_lines
  n: 16
  buttons: true            # adds interactive channel-selection buttons in plotly
- key: histogram
  log_key: peak_histogram/PD_count   # overrides the wandb log key
  channel_name: PD
  measure: count
  n: 64
- key: entropy_plot
  method: app_entropy        # or spectral_entropy, perm_entropy
  log_key: approximate_entropy_plot
  n: 64
```

Available plot keys: `2d_flow_plot`, `line_flow_plot`, `animated_traces`, `multi_channel_lines`, `entropy_plot`, `histogram`.

## rollout

Autoregressive rollout evaluation, run after `trainer.test()` (see [run-lifecycle.md](run-lifecycle.md)). The gate is the presence of this block: remove or comment it out and the feature is fully off; nothing else references it. Base runs predate the feature, so reeval runs only get rollouts through the block in `configs/reeval.yaml`.

| Key | Type | Default | Description |
|---|---|---|---|
| `start_fractions` | list | `[0.10, 0.25, 0.50, 0.75, 0.90]` | Rollout start points as fractions of each test shot's length; clamped to the viable window range, duplicates after clamping are dropped |
| `step` | int | `${data.seq_length}` | Samples to advance per generation. Equal to `seq_length` = non-overlapping chaining; smaller values enable overlapped chaining (the first `step` samples of each generated window are kept) |
| `n_samples` | int | 1 | Stochastic samples per start point; cache keys gain a sample index |
| `n_steps` | int | `${evaluation.n_steps}` | ODE integration steps per generated window |
| `max_batch` | int | 128 | Rollouts advance in lockstep; window k of every unfinished rollout is batched together up to this size |
| `clamp_history` | bool | false | Clamp the fed-back generated history to [0,1] before conditioning on it |
| `cache_name` | str | `${test_cache_name}_rollout` | HDF5 rollout cache name (see [outputs.md](outputs.md)) |
| `cache_mode` | str | `create` | `create` generates and caches (resumable: cached rollouts are skipped); `use` reads the cache and skips generation, so plots/metrics rebuild without a GPU |
| `html_shots` | list | 4 window_set shots | Shots included in the interactive rollout browser HTML (all test shots are still rolled out and cached) |
| `analysis` | bool | true | Also export the horizon figures/tables in-run to `output/pdfplots/{run_name}/rollout_analysis/` (same code as `eval_notebooks/rollout_analysis.py`, which can redo them from the cache) |

## model

```yaml
model:
  Class: FlowModule   # must be registered in src/models/__init__.py
  params:             # passed as **kwargs to FlowModule.__init__
    ...
```

### Flow matching parameters

| Key | Type | Default | Description |
|---|---|---|---|
| `prior` | str | `normal` | Prior distribution: `normal`, `brownian`, `levy`, `resample`, `copy`, `constant` |
| `prior_sigma` | float | 0.3 | Std dev for the normal prior; 0.3 puts roughly 90% of samples in [0,1] given normalized targets |
| `ot_method` | str or null | null | Optimal transport pairing method; null disables OT (OT only works with `prior=normal`) |
| `ot_replace` | bool | true | Whether OT sampling uses replacement |
| `batch_rematch_factor` | int | 5 | Number of prior-target rematch iterations per batch (gradient accumulation over multiple pairings) |
| `step_every_nth_match` | int | 1 | Take a gradient step every N matches; must divide `batch_rematch_factor` evenly |
| `gradient_clip_val` | float | (from top-level) | Gradient clipping inside the manual optimization loop |
| `solve_method` | str | (from evaluation) | Passed to `integrate_path()`; currently inactive (always Euler) |
| `flow_steps` | int | (from evaluation.n_steps) | Integration steps for test and inference |
| `loss` | str | `MSELoss` | Loss function: `MSELoss` or `L1Loss` |

### model_params (ConditionalUNet architecture)

| Key | Type | Default | Description |
|---|---|---|---|
| `conditioning` | list | `[x_history, c, position_sequence]` | Which conditioning signals to use |
| `conditioning_method` | str | `channels` | How conditioning is applied: `channels` (concatenated as extra channels), `sequence` (prepended to sequence), `mid-sequence`, `mid-channels` |
| `spatial_dim` | int | 1 | Spatial dimensions (1 for time series) |
| `input_channels` | int | 5 | Auto-set from `len(data.cols.x)`; do not set manually |
| `c_channels` | int | 0 | Auto-set from `len(data.cols.c)`; 0 disables c conditioning |
| `apex_hidden_channels` | int | 64 | Hidden dim for the apex (bottleneck) block |
| `time_embedding` | str | `sinusoidal+mlp` | Time t embedding: `sinusoidal`, `sinusoidal+mlp`, `dummy` |
| `time_embedding_d` | int | 32 | Dimension of the time embedding |
| `positional_encoding` | str | `sinusoidal+mlp` | Positional encoding for `position_sequence` conditioning |
| `positional_encoding_d` | int | 32 | Dimension of the positional encoding |
| `positional_encoding_c` | int | 8 | MLP compression dim for positional encoding (only used with `+mlp`) |
| `ch_mults` | list | `[2,2,2,2]` | Channel multiplier per UNet level; length determines depth |
| `is_attn` | list | `[true,false,false,false]` | Attention at each level |
| `mid_attn` | bool | true | Attention in the bottleneck |
| `attn_heads` | int | 2 | Number of attention heads |
| `n_blocks` | int | 2 | Residual blocks per UNet level |
| `norm_groups` | int | 4 | Groups for GroupNorm |
| `use1x1` | bool | false | Whether to use 1x1 convolutions |

The positional encoding embeds `position_sequence`, which is the raw shot time in **seconds** (not normalized, unlike `X` and `C`). Its embedder is hard-built with `max_value=2.0` in `unet_conditional.py` to span that range; this is not a config key. If shots ever exceed 2 s, that `max_value` must be raised in code or distant timesteps alias onto the same phase.
### optimizer_params

| Key | Type | Default | Description |
|---|---|---|---|
| `lr` | float | 0.00005 | Learning rate |
| `weight_decay` | float | 0 | L2 regularization |
| `amsgrad` | bool | false | Use the AMSGrad variant of Adam |

## data

IMPORTANT: The column names in `cols.x`, `cols.c`, and `cols.label` must exactly match column names in the parquet file. There is no validation at load time; a mismatch causes a silent KeyError or NaN values during normalization.

Current required columns in the parquet (as of `2026_06_29-TCV_shots_V2.parquet`): `ShotNum`, `time`, `FIR_LIDs_core`, `PD`, `DML`, `POHM`, `Z_axis`, `IP`, `LHD_label`.

| Key | Type | Default | Description |
|---|---|---|---|
| `Class` | str | `FusionShotDataModule` | Data module class |
| `dir` | str | `./data/` | Directory containing the parquet file |
| `file` | str | `2026_06_29-TCV_shots_V2.parquet` | Parquet filename |
| `batch_size` | int | (from top-level) | Batch size for all dataloaders |
| `seq_length` | int | 256 | Length of the future window Wf (in samples at 10 kHz = 25.6 ms) |
| `history_length` | int | 256 | Length of the history window Wh; defaults to `seq_length` via `${data.seq_length}` |
| `crop_margin` | int | 1024 | Minimum distance from shot start/end to any window; must be `>=` `history_length` |
| `overfit_on_shots` | list or null | null | If set, restricts all splits to only these shot numbers (debugging) |
| `allowed_start_indices` | list or null | null | If set, only uses these specific start indices across all shots |
| `sample_rate` | int | 10000 | Sampling rate in Hz (used for Brownian motion sqrt_dt and positional encoding) |
| `pre_shuffle` | bool | true | Shuffle training and validation sets before batching; the test set is always sorted by index |

### data.cols

```yaml
cols:
  meta: [ShotNum, time]   # loaded but not used as features; used for indexing
  x: [FIR_LIDs_core, PD, DML, POHM, Z_axis]   # observable channels (model input/output)
  c: [IP]                 # conditioning channels (history only, not predicted)
  label: LHD_label        # integer label column (0=L-mode, 1=D-mode, 2=H-mode)
```

Adding or removing entries from `x` or `c` automatically updates the model input/output sizes via `update_model_input_channels`. The `PD` column is additionally required by the mode classifier; see `src/metrics/evaluate_modes.py`.

### data.train_shots / val_shots / test_shots

Hard-coded lists of TCV shot numbers. These are fixed for paper reproducibility. The train/val/test split is at the shot level: no shot appears in more than one split.

Viable sample counts (approximate, at `seq_length=256`, `crop_margin=1024`):

- Train: 237 shots, ~1.96M viable windows
- Val: 12 shots, ~104K viable windows
- Test: 25 shots, ~217K viable windows (stride=10 for index precomputation)

## configs/reeval.yaml

Used when `reeval=True` is passed. This file is merged on top of the config downloaded from the base run, so any key set here overrides the base run's config. Typical use: change `batch_size` to 512 for faster inference, override `window_set`, and set `test_cache_name`.

```yaml
base_run: <run_name_or_id>     # the run to re-evaluate
prefer_model_alias: best       # which checkpoint to use: best or latest
test_cache_name: ${run_name}
test_cache_mode: create
batch_size: 512
```

## Shorthand aliases

At the bottom of `plasmaflow.yaml`, top-level keys like `Prior`, `Ot_method`, and `Seq_L` are OmegaConf interpolations pointing to deeply nested params. They exist solely for convenient display in the wandb config table, which only shows top-level keys. Do not set them directly; set the underlying keys they point to.
