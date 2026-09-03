# PlasmaFlow Output Artifacts

Complete inventory of every output artifact a PlasmaFlow run produces: where it goes, what it contains, what triggers it, and how to control it. Use this doc to answer "where do I find X?" for any output type.

There are two destinations:

1. **Local filesystem** under the repo's `output/` directory (also used on the cluster, inside `~/fusion-plasma-simulation/output/`).
2. **Wandb cloud** (project `plasmaflow`, entity `deep-learning-course-team`), synced through the `WandbLogger`.

On the cluster, several environment variables redirect a subset of these paths to scratch storage. See [Cluster-specific paths](#cluster-specific-paths).

---

## Filesystem outputs (local)

### Model checkpoints

| Property | Value |
|---|---|
| Path | `output/models/{dated_run_name}/` |
| Filename | `{dated_run_name}-Epoch={epoch:02d}-step={step}-val_loss={loss/val:.2f}.ckpt` plus `last.ckpt` |
| `dated_run_name` | `{YYYY-MM-DD}-{wandb_run_name}` |
| Trigger | `ModelCheckpoint` callback, fires every epoch; saves when `loss/val` improves, and always updates `last.ckpt` |
| Contents | Full Lightning checkpoint: model weights, optimizer state, hyperparameters, epoch/step counters |

**Config control:**

- `patience`: early stopping patience on `loss/val`.
- `epochs`: maximum number of epochs.

The set kept on the wandb side is pruned to best + latest by `prune_online_checkpoints()` (see [Wandb artifacts](#wandb-artifacts-model-checkpoints)). Local checkpoint files are **not** automatically pruned; they accumulate on disk until removed manually.

### PDF plots from `evaluate_window_set`

| Property | Value |
|---|---|
| Path | `output/pdfplots/{run_name}/qualitative_samples/{subgroup}/{WxH}/{shot}_{time}s.pdf` |
| `subgroup` | `full` (with legend) or `nolegend` (cleaner for figures) |
| Trigger | `evaluate_window_set()`, which runs after every `trainer.validate()`, including reeval runs |

One file is written per size in:

```
[(300, 250), (400, 450), (600, 500), (800, 500),
 (1200, 600), (1300, 910), (800, 1200), (600, 1000)]
```

Additional files in each subgroup directory:

| File | Description |
|---|---|
| `atom_{subgroup}_{shot}_{time}s.pdf` | "Atom" variant at 500x400 with serif font and no margins |
| `throwaway.pdf` | Empty file written first to flush mathjax artifacts from plotly's kaleido renderer |
| `batch_metrics_{subgroup}_{shot}_{time}s.json` | Per-window metric metadata for the plotted sample |

**Config control:** entries in `window_set`. Remove entries to suppress specific shots from being plotted.

### HDF5 test cache

| Property | Value |
|---|---|
| Path (local) | `output/test_cache/{test_cache_name}.h5` |
| Path (cluster) | `$TEST_CACHE_DIR/{test_cache_name}.h5` if the env var is set |
| Trigger | `trainer.test()` when `test_cache_name` is set in config and `test_cache_mode: create` |

**HDF5 structure** (per generated window):

| Dataset | Dtype | Shape |
|---|---|---|
| `/{shot_number}/{start_idx}/generated_x` | float32 | channels x seq_length |
| `/{shot_number}/{start_idx}/surr_labels_gen` | int16 | history_length + seq_length |
| `/{shot_number}/{start_idx}/surr_labels_target` | int16 | history_length + seq_length |

**Config control:**

- Omit `test_cache_name` from config to disable caching entirely.
- `test_cache_mode: create`: generate the cache during `trainer.test()`.
- `test_cache_mode: use`: read from an existing cache instead of re-generating.

**Caveat:** the channel count in the cache is hardcoded to 5 (the length of `data.cols.x`). Changing the channel list without re-creating the cache produces mismatched data.

### HDF5 cache metadata JSON

| Property | Value |
|---|---|
| Path | Same directory as the `.h5` file, named `{test_cache_name}.json` |
| Contents | Aggregated `test/final/*` metrics for the run, JSON-serializable (Tensors converted to scalars) |
| Trigger | `on_test_epoch_end()` when the cache is configured |

### HDF5 rollout cache

| Property | Value |
|---|---|
| Path (local) | `output/test_cache/{rollout.cache_name}.h5`, by default `{test_cache_name}_rollout.h5` |
| Path (cluster) | `$TEST_CACHE_DIR/{rollout.cache_name}.h5` if the env var is set |
| Trigger | `run_rollouts()` after `trainer.test()`, only when a `rollout:` block is in the config |

A separate file from the window test cache on purpose: rollout traces have variable length, and the window-cache readers iterate `{shot}/{start_idx}` groups assuming fixed `(5, seq_length)` shapes.

**HDF5 structure** (per rollout; T = total generated samples, variable per rollout):

| Dataset | Dtype | Shape |
|---|---|---|
| `/{shot_number}/{start_idx}/{sample_idx}/generated_x` | float32 | channels x T (normalized [0,1]) |
| `/{shot_number}/{start_idx}/{sample_idx}/surr_labels_gen` | int16 | history_length + T |
| `/{shot_number}/{start_idx}/{sample_idx}/surr_labels_real` | int16 | history_length + T |

Leaf group attrs: `start_frac`, `start_i`, `t_start`, `t_end`, `n_windows`, `seq_length`, `history_length`, `step`. Root attrs: `start_fractions`, `n_samples`, `cols_x`, `run_name`. Labels use the unshifted surrogate convention (0=L, 1=D, 2=H). Real observables, controls, and true labels are NOT stored; notebooks re-derive them positionally from the parquet at `start_i`. Read via `RolloutHDFCache` / `src.rollout.load_results_from_cache`.

A `{rollout.cache_name}.json` sidecar holds the small in-run summary (`rollout/final/*`) plus the skipped shot/fraction combinations.

### Rollout browser HTML

| Property | Value |
|---|---|
| Path | `output/htmlplots/{run_name}/rollouts.html` |
| Contents | Interactive per-rollout browser (see [plots.md](plots.md)) over the `rollout.html_shots` subset |
| Trigger | `run_rollouts()`; also logged to wandb as `rollout/browser` |

Rebuild locally from a cache with `eval_notebooks/rollout_browser.py` (writes to `output/htmlplots/local/`).

### Rollout paper PDFs and horizon tables (notebooks, local)

| Property | Value |
|---|---|
| Paths | `output/pdfplots/paper_rollout/{WxH}/{shot}_{frac}.pdf`, `output/pdfplots/rollout_analysis/*.pdf`, `output/tables/rollout_horizon_{cache}.csv` and `.tex` |
| Trigger | Manual: `eval_notebooks/paper_rollout.py` and `eval_notebooks/rollout_analysis.py` against a fetched rollout cache |

### Slurm output logs (cluster, rsynced back)

| Property | Value |
|---|---|
| Path | `output/snellius/slurms/gpujob-{jobid}-{jobname}.out` |
| Contents | Combined stdout + stderr from the entire `run.py` execution on the cluster |
| Trigger | Rsynced back by `submit_remote_job_snellius.sh` during and after job execution |

Not available for local runs.

### Lightning fallback logs

| Property | Value |
|---|---|
| Path | `output/lightning_logs/version_{N}/` |
| Contents | Lightning's default checkpoint and `hparams.yaml` (fallback, mostly superseded by `WandbLogger` + `ModelCheckpoint`) |
| Trigger | Always created by Lightning unless explicitly redirected |

---

## Wandb outputs (cloud synced)

### Training metrics (every step)

| Key | Meaning | Summary |
|---|---|---|
| `loss/train` | Training loss (MSELoss over predicted vs. target velocity) | min |
| `loss/val` | Validation loss | min |
| `loss/grad_norm` | Gradient norm before clipping | |
| `trainer/samples_seen` | Cumulative samples processed (accounts for `batch_rematch_factor`) | |
| `trainer/samples_per_minute` | Sample throughput | |
| `trainer/steps_per_minute` | Gradient step throughput | |
| `trainer/global_step` | Lightning global step counter | |

### Validation metrics

Logged conditionally, when `epoch % val_every_n_epochs == 1` OR `epoch <= scrutinize_epochs`.

**Moment errors** (per channel and mean across channels):

```
val/error/magnitude_mean_mse/{channel}
val/error/magnitude_var_mse/{channel}
val/error/magnitude_skew_mse/{channel}
val/error/magnitude_kurtosis_mse/{channel}
val/error/diff_mean_mse/{channel}        # same statistics computed on first-differences
val/error/diff_var_mse/{channel}
val/error/diff_skew_mse/{channel}
val/error/diff_kurtosis_mse/{channel}
val/error/magnitude_*_mse/mean           # mean across channels
```

All 8 combinations of magnitude/diff x moment (mean, var, skew, kurtosis) x channel are emitted, plus the `/mean` rollups.

**Entropy metrics** (per channel and mean):

```
val/error/app_entropy_{mse|mae|msd|wasserstein}/{channel}
val/error/spectral_entropy_{mse|mae|msd|wasserstein}/{channel}
val/error/perm_entropy_{mse|mae|msd|wasserstein}/{channel}   # if enabled
```

**Peak metrics:**

```
val/peak_{measure}_{stat}/{channel}
val/peak_count_{stat}
```

where `measure` is one of `height`, `prominence`, `base`, `width`, and `stat` is one of `mean_pairwise_wasserstein`, `marginal_wasserstein`, `count_mse`.

**Mode transition metrics:**

```
val/mode/{condition}/{from}_{to}_{stat}
```

Example: `val/mode/any_Wh/from_L_to_H_expect_target`.

**Dice score:**

```
val/dice
```

**Plots** (logged as `wandb.Image` or `wandb.Html`):

```
val/animated_traces
val/multi_channel_lines
val/peak_histogram/PD_count
val/approximate_entropy_plot
val/spectral_entropy_plot
```

Train-data equivalents are logged under the `train/` prefix every `train_every_n_epochs` epochs.

### Test metrics (after `trainer.test()`)

All the same metrics as validation, under the `test/final/` prefix:

```
test/final/error/*
test/final/peak_*
test/final/mode/*
test/final/dice
```

Per-step metrics are logged under `test/step/` during test batches (`on_step=True`).

### Wandb artifacts (model checkpoints)

- Logged as model artifacts by `WandbLogger` (`log_model="all"`).
- Aliases: `best` (checkpoint with lowest val loss) and `latest` (most recent).
- `prune_online_checkpoints()` runs at the end of a run and deletes all wandb artifacts that carry neither the `best` nor the `latest` alias, to save cloud storage.

---

## Cluster-specific paths

On Snellius, these environment variables redirect outputs to scratch storage:

```bash
WANDB_DIR=/scratch-shared/mtresoor/wandb            # wandb run directory
WANDB_CACHE_DIR=/scratch-shared/mtresoor/wandb/cache
WANDB_ARTIFACT_DIR=/scratch-shared/mtresoor/artifacts
TEST_CACHE_DIR=/scratch-shared/mtresoor/final_cache  # HDF5 cache location
```

Model checkpoints and PDF plots still land in the repo's `output/` directory on the cluster (inside `~/fusion-plasma-simulation/output/`). To retrieve them locally, use rsync manually or check what the submit script syncs.

---

## What does NOT get auto-retrieved

The submit script only rsyncs Slurm logs back from the cluster. The following must be pulled manually if you need them locally:

| Artifact | Cluster location |
|---|---|
| Model checkpoints | `output/models/` in the cluster repo |
| PDF plots | `output/pdfplots/` in the cluster repo |
| HDF5 caches | `$TEST_CACHE_DIR/` on scratch |

Wandb artifacts are synced to the cloud and are accessible through the wandb UI without any manual retrieval.
