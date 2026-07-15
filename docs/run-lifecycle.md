# Run Lifecycle

This document answers one question: what happens, step by step, when you submit a job? It traces a run from the moment `run.py` starts through training, evaluation, and teardown. It is the canonical reference for understanding control flow in PlasmaFlow.

## Overview

`run.py` is the single entry point for training and evaluation. It supports three distinct execution paths, selected by config:

1. **Fresh training**: train a new model from scratch.
2. **Resume training**: continue an interrupted run from its wandb checkpoint.
3. **Reeval**: re-evaluate an already-trained run without retraining.

All three paths share the same evaluation tail (`trainer.validate()`, `evaluate_window_set()`, `trainer.test()`). They differ only in setup and in whether `trainer.fit()` runs.

```
run.py
├── Config loading (OmegaConf + wandb.init)
├── DataModule init
├── Model init (fresh / load_from_checkpoint)
├── wandb_logger.watch
├── set_cache / set_integration_method
├── Trainer + callbacks
├── trainer.fit()          [SKIPPED on reeval]
│     └── per epoch: training_step → validation_step → epoch-end callbacks
├── wandb_logger.unwatch
├── trainer.validate()     [always]
├── evaluate_window_set()  [always]
├── trainer.test()         [always]
├── run_rollouts()         [only if a rollout: block is in the config]
├── prune_online_checkpoints
└── run.finish()
```

## Path 1: Fresh training

```bash
python run.py run_name=myrun
```

The sequence is:

1. **Config loading.** OmegaConf loads `configs/plasmaflow.yaml`, then merges CLI overrides. `update_model_input_channels` auto-syncs channel counts: it reads `data.cols.x` and writes the count into `model.params.model_params.input_channels`. `wandb.init()` is then called. From this point on, `get_current_config(wandb_only=True)` is the canonical config object, since wandb has merged everything (defaults, file, CLI).
2. **DataModule.** `FusionShotDataModule(**C.data)` is constructed. Note that `prepare_data()` and `setup()` are not called here; Lightning calls them later, inside `trainer.fit()`.
3. **Model.** `FlowModule(**C.model.params)` is constructed (or via `load_from_checkpoint` when resuming, see Path 2).
4. **Gradient logging.** `wandb_logger.watch(model, log="all", log_freq=50)` registers hooks to log gradients and parameters.
5. **Cache.** If `test_cache_name` is present in config, `model.set_cache(name, mode)` configures the HDF5 evaluation cache.
6. **Integration method.** If `evaluation` is present in config, `model.set_integration_method(n_steps, solve_method)` sets the integration parameters used by `model.evaluate()`.
7. **Trainer.** `trainer = L.Trainer(...)` is built with the callback set: `PlotsCallback`, `EarlyStopping`, `LearningRateMonitor`, `ModelCheckpoint`, `TrainStepMonitor`.
8. **Fit.** `trainer.fit()` runs the training loop (see [Training loop](#training-loop-trainerfit)).
9. **Unwatch.** `wandb_logger.experiment.unwatch(model)` removes the gradient hooks.
10. **Validate.** `trainer.validate()` runs one final validation pass.
11. **Window set.** `evaluate_window_set(model, data_module, C.window_set)` always runs (see [evaluate_window_set](#evaluate_window_set)).
12. **Test.** `trainer.test()` runs the final test pass (see [Test loop](#test-loop-trainertest)).
13. **Rollouts.** If the config has a `rollout:` block, `src.rollout.run_rollouts()` performs autoregressive rollout evaluation (see [Rollout evaluation](#rollout-evaluation)). No block, no rollouts.
14. **Prune.** `prune_online_checkpoints(run)` deletes wandb artifacts that carry neither the `best` nor the `latest` alias.
15. **Finish.** `run.finish()` closes the wandb run.

## Path 2: Resume training

Add `resume_id` and `resume_name` to the config (or pass them via CLI):

```bash
python run.py resume_id=<wandb_run_id> resume_name=<run_name>
```

Differences from Path 1:

- `find_and_download_model(resume_name)` downloads the checkpoint from wandb.
- `run_name` is overridden to `resume_name`.
- `wandb.init(id=resume_id, resume='allow')` resumes the existing run rather than creating a new one.
- The model is built via `load_from_checkpoint` instead of a fresh `FlowModule(...)`.
- `trainer.fit()` continues from the checkpoint epoch.

The evaluation tail (validate, window set, test, prune, finish) is identical to Path 1.

## Path 3: Reeval

```bash
python run.py reeval=True run_name=myrun
```

Re-evaluates a previously trained run without retraining. Differences from Path 1:

- `is_reeval_run()` detects the `reeval=True` CLI flag.
- `consolidate_base_reeval_configs()` loads `configs/reeval.yaml`, finds the base run by its `base_run` name, and downloads the checkpoint using `prefer_model_alias` (either `best` or `latest`).
- The wandb run is tagged `reeval`.
- `trainer.fit()` is **skipped**.
- `trainer.validate()`, `evaluate_window_set()`, and `trainer.test()` all still run.

This path exists to regenerate metrics and figures from a finished model, for example after changing a metric or plotter.

## Training loop (`trainer.fit`)

Each epoch runs train batches, then validation batches, then the epoch-end callbacks.

### 1. Training step

`training_step(batch, batch_idx)` runs for `limit_train_batches` batches. It uses **manual optimization**:

1. `opt.zero_grad()`.
2. Run `batch_rematch_factor` iterations of `batch_match()`. Each `batch_match()`:
   - Calls `interpolate_samples()`: draw the prior, optionally apply OT pairing, sample a random `t`, then compute the interpolated point and the velocity target (see [interpolate_samples in detail](#interpolate_samples-in-detail)).
   - Runs `model(samples_at_t, t, conditioning_input)` to get the predicted velocity.
   - Computes MSE loss between predicted and target velocity.
   - Calls `manual_backward(loss)`.
   - Every `step_every_nth_match` matches: `opt.step()`, then `opt.zero_grad()`.
3. Logs `loss/train`.

### 2. Validation step

`validation_step(batch, batch_idx)` runs for `limit_val_batches` batches. Same interpolation as the training step, but no backward pass. Logs `loss/val`.

### 3. Train epoch end

`on_train_epoch_end()`:

- Steps the LR scheduler (`ReduceLROnPlateau` monitoring `loss/train`).
- Has a parallel trigger for train-data plots: fires when `epoch % train_every_n_epochs == 0`.

### 4. Validation epoch end

`on_validation_epoch_end()` (in `PlotsCallback`):

- Prunes wandb checkpoints, keeping `best` and `latest`.
- **Conditional full evaluation.** Runs the full evaluation only if `(current_epoch % val_every_n_epochs == 1)` OR `current_epoch <= scrutinize_epochs`.
  - **Note:** the condition uses `== 1`, not `== 0`. It therefore fires at epochs 1, 6, 11, ... and not at 0, 5, 10. Epoch 0 is covered only by `scrutinize_epochs`.
- When it fires: draws a batch from the **test** dataset (not val), runs `model.evaluate()`, calls `call_plot_functions()`, and logs metrics under the `val/` prefix.

`EarlyStopping` monitors `loss/val` with patience taken from config.

## evaluate_window_set

Always runs after `trainer.validate()` and before `trainer.test()`. There is no config flag to skip it; the only way to disable it is to remove entries from `config.window_set`.

For each `[shot_number, time_seconds]` pair in `config.window_set`:

1. `test_dataset.quick_window(shot, t, repeat=4)`: finds the nearest window to that time and repeats the batch 4 times.
2. `model.evaluate(batch, n_steps=120)`: full evaluation, including metrics.
3. `multi_sample_single_window_lines_plotly(...)`: builds a Plotly figure.
4. `dump_figure_to_pdfs(fig, plot_name="qualitative_samples", subgroup="full", ...)`: saves the figure at 8 sizes for thesis figure fitting: 300x250, 400x450, 600x500, 800x500, 1200x600, 1300x910, 800x1200, 600x1000.
5. Dumps a `nolegend` version as well.
6. Saves `batch_metrics_{subgroup}_{shot}_{t}s.json` alongside the PDFs.

PDF output path:

```
output/pdfplots/{run_name}/qualitative_samples/{full|nolegend}/{WxH}/{shot}_{t}s.pdf
```

## Test loop (`trainer.test`)

### 1. Test step

`test_step(batch, batch_idx)` runs for each batch:

- `inference()`:
  - If cache mode is `use`: load samples from the HDF5 cache.
  - Otherwise: generate samples via `integrate_path()`, compute surrogate labels, and write to the cache if mode is `create`.
- `update_metrics()`: compute moments, peak metrics, mode transition metrics, and dice score.
- Logs `test/step/{metrics}` per step.

### 2. Test epoch end

`on_test_epoch_end()`:

- Aggregates all metrics and logs them under `test/final/{metrics}`.
- `test_cache.save_json_friend(epoch_metrics)`: writes a `.json` summary.
- For each `ModeTransitionMetric`: `extract_df_all(cache)` appends DataFrames to the HDF5 file.
- For each `PeakMetric`: `extract_df_all(cache)`, then `export_2d_NBI_distributions()`, then `make_histograms(peak_dfs)`.
- Resets all metrics.

## Rollout evaluation (`src/rollout.py`)

Runs after `trainer.test()` only when the config contains a `rollout:` block (see [configuration.md](configuration.md)); removing the block disables the feature entirely. The whole stage is wrapped in try/except like `animate_window_set`, so a failure here cannot kill a finished run.

1. **Plan.** `compute_rollout_specs()` turns `rollout.start_fractions` into per-shot start indices, clamped to `[crop_margin, shot_len - crop_margin - seq_length]`. On short shots multiple fractions can clamp onto the same index; duplicates are dropped and recorded as skipped.
2. **Generate.** `_generate_rollouts()` chains windows: the real history at the start point conditions the first generation, after which each generated window becomes the next `x_history` while `c` and `position_sequence` keep coming from the real shot data (`label` is carried too, but it is not a model input). All rollouts advance in lockstep; window k of every unfinished rollout is batched together (up to `rollout.max_batch`), and finished rollouts drop out. The advance per generation is `rollout.step` (default `seq_length`, i.e. non-overlapping).
3. **Label.** `label_rollout()` runs the FNOLSTM surrogate classifier over each full generated trace and over the real trace of the same span (full real pre-rollout history prepended, PD channel, denormalized). One rollout per call: padding different-length rollouts into one batch would leak garbage labels into the shorter ones. Labels stay in the unshifted 0=L, 1=D, 2=H convention.
4. **Cache.** Each rollout is written to `{rollout.cache_name}.h5` (see [outputs.md](outputs.md)). `cache_mode: create` is resumable (existing rollouts are skipped); `cache_mode: use` reads the cache back and skips generation, so metrics and plots can be redone without a GPU.
5. **Summarize.** `summarize_rollouts()` logs a small `rollout/final/*` summary to wandb and writes it as the cache's `.json` sidecar. The horizon-resolved analysis lives in `eval_notebooks/rollout_analysis.py`, computed from the cache.
6. **Browser.** The interactive rollout browser HTML is written for the `rollout.html_shots` subset (see [plots.md](plots.md)).

Standalone smoke test (untrained model, two shots, CPU): `PYTHONPATH=. python src/rollout.py`.

## interpolate_samples in detail

The core of the flow matching objective. Straight-line (conditional optimal transport) interpolation:

```python
t = rand(batch_size)             # random interpolation time in [0, 1]
x_t = x0 * (1 - t) + x1 * t      # linear interpolation
velocity = x1 - x0               # constant velocity target (straight-line flow)
```

When OT pairing is active:

- `pi = ot_sampler.get_map(prior, x1)` computes the transport plan.
- `i, j = ot_sampler.sample_map(pi, ...)` gives the reordering indices.
- `prior = prior[i]; x1 = x1[j]` reorders prior and targets.
- **Important:** the conditioning must be reordered to match the reordered targets:

  ```python
  conditioning_inputs = {k: v[j] for k, v in conditioning_inputs.items()}
  ```

  This is easy to miss. If conditioning is not reordered with `j`, each target is paired with the wrong conditioning.

## integrate_path

Generates samples by integrating the learned velocity field from `t=0` to `t=1`.

Uses **forward Euler** integration with `n_steps` steps.

**Note on the `method` parameter.** `method` is accepted but currently inactive. The torchdiffeq adaptive solver path (`dopri5`, `midpoint`, and others) was used in an earlier version, but were found to cause buggy generation results. These methods are preserved as commented-out code behind an `if True:` guard at the top of the integration loop. Only forward Euler is active. The `solve_method` config key flows into `model.evaluate()` (which passes it to `integrate_path`), but it has no effect on the solver actually used during test or inference.

## Callback epoch schedule

| Callback | Trigger |
|---|---|
| `EarlyStopping` | every epoch (monitors `loss/val`) |
| `LearningRateMonitor` | every epoch |
| `ModelCheckpoint` | every epoch (saves if `loss/val` improved) |
| `PlotsCallback.on_validation_epoch_end` | `epoch % val_every_n_epochs == 1` OR `epoch <= scrutinize_epochs` |
| `PlotsCallback.on_train_epoch_end` | `epoch % train_every_n_epochs == 0` OR `epoch <= scrutinize_epochs` |
| `TrainStepMonitor` | every training batch |
