# Plots

This document describes every plot type PlasmaFlow produces: what each one shows, when it fires, how to configure or disable it, and where it ends up. Two distinct mechanisms generate plots:

1. The **dispatch system**: plots driven by config under `evaluation.plot_functions`, logged to wandb during validation (and optionally training) epochs.
2. **`evaluate_window_set` plots**: qualitative figures for hand-picked shot windows, written to disk as PDFs after every `trainer.validate()`.

Both are covered below.

## Plot dispatch system

Plots in the dispatch system are configured in `configs/plasmaflow.yaml` under `evaluation.plot_functions`. Each entry is a small dict with these fields:

| Field | Meaning |
|---|---|
| `key` | The plot function identifier. Maps to an entry in `PlotsCallback.PLOT_FN_OPTIONS` (`src/evaluation.py`). |
| `n` | Number of samples to draw from the test dataset for this plot. |
| `log_key` | The wandb key to log the figure under. Defaults to `key` if not specified. |
| (other) | Any additional keys are passed straight through to the plot function as kwargs. |

### Available keys

| Key | Function | Source file |
|---|---|---|
| `animated_traces` | `animated_trajectory_plotly` | `src/plotters/plot_animations.py` |
| `multi_channel_lines` | `multi_channel_lines_plotly` | `src/plotters/flow_plots.py` |
| `entropy_plot` | `plot_entropy` | `src/plotters/plot_entropy.py` |
| `histogram` | `plot_peak_prominences_histogram` | `src/plotters/histograms.py` |
| `2d_flow_plot` | `plot_flow` | `src/plotters/flow_plots.py` |
| `line_flow_plot` | `plot_flow_and_lines_plotly` | `src/plotters/flow_plots.py` |

The canonical key-to-function mapping lives in `PlotsCallback.PLOT_FN_OPTIONS` (`src/evaluation.py`).

## When plots fire

All `plot_functions` entries fire during `PlotsCallback.on_validation_epoch_end()`, and optionally during `on_train_epoch_end()`.

### Epoch trigger

Validation plots fire when:

```
(current_epoch % val_every_n_epochs == 1) OR (current_epoch <= scrutinize_epochs)
```

Worked examples with the defaults in `plasmaflow.yaml`:

- With `val_every_n_epochs: 5`, the first clause fires at epochs 1, 6, 11, 16, and so on.
- With `scrutinize_epochs: 3`, the second clause additionally fires at epochs 0, 1, 2, 3 (every early epoch gets the full treatment).

The two clauses are OR-ed, so early epochs are covered by `scrutinize_epochs` and later epochs by the modulo cadence.

### Training-data plots

Plots for the training set (logged under the `train/` prefix instead of `val/`) fire only if `train_every_n_epochs > 0`. Set `train_every_n_epochs: 0` to skip training-data plots entirely. The default in `plasmaflow.yaml` is `20`.

## Individual plot types

### animated_traces

**What it shows:** an animated Plotly figure of `n=5` generated sample trajectories over time. Each frame of the animation is one step of the integration path, from the prior (noise) to the final generated trajectory. You can scrub through it and watch the model flow from noise into a realistic trajectory.

**Config:**

```yaml
- key: animated_traces
  n: 5
```

**Wandb key:** `val/animated_traces`, logged as an HTML iframe.

**How to disable:** remove this entry from `plot_functions`.

### multi_channel_lines

**What it shows:** a multi-panel Plotly figure with one panel per channel. Each panel overlays `n=16` generated sample trajectories with the target, and also shows the history window (the conditioning context). With `buttons: true`, the figure gains a dropdown for selecting which samples to show.

**Config:**

```yaml
- key: multi_channel_lines
  n: 16
  buttons: true
```

**Wandb key:** `val/multi_channel_lines`, logged as HTML.

**How to disable:** remove this entry from `plot_functions`.

### histogram

**What it shows:** a histogram comparing the distribution of a peak property between generated and target samples, for a single channel and a single measure.

**Config:**

```yaml
- key: histogram
  log_key: peak_histogram/PD_count  # custom wandb key
  channel_name: PD                   # which channel to analyze
  measure: count                     # which peak property (see below)
  n: 64
```

`measure` selects which peak property to histogram. Supported values: `count`, `prominence`, `height`, `base`, `width`, `energy_delta`, `energy_ratio`, `pd_prominence`.

You can add multiple `histogram` entries for different channel/measure combinations. Several examples are present but commented out in `plasmaflow.yaml` (for instance DML `energy_delta`, DML `pd_prominence`, FIR_core `count`).

**Wandb key:** whatever `log_key` specifies.

**How to disable:** remove the entry (or entries) from `plot_functions`.

### entropy_plot

**What it shows:** a comparison of entropy distributions between generated and target samples. It renders per-sample entropy values as histograms or scatter plots, so you can visually check whether the model produces sequences with the right complexity.

**Config:**

```yaml
- key: entropy_plot
  method: app_entropy          # or spectral_entropy, perm_entropy
  log_key: approximate_entropy_plot
  n: 64
```

`method` selects the entropy estimator: `app_entropy` (approximate entropy), `spectral_entropy`, or `perm_entropy` (permutation entropy).

**Wandb key:** whatever `log_key` specifies.

**How to disable:** remove the entry from `plot_functions`.

### 2d_flow_plot and line_flow_plot

These are experimental and diagnostic plots for visualizing the flow in 2D. They are useful for toy experiments where the state is low-dimensional, not for the full TCV dataset. Both are currently commented out in `plasmaflow.yaml` and are not part of the normal paper-run plotting set.

| Key | Function | Source file |
|---|---|---|
| `2d_flow_plot` | `plot_flow` | `src/plotters/flow_plots.py` |
| `line_flow_plot` | `plot_flow_and_lines_plotly` | `src/plotters/flow_plots.py` |

## evaluate_window_set plots (always produced)

Separate from the dispatch system, `evaluate_window_set()` runs after every `trainer.validate()` call (including reeval runs) and generates qualitative plots for the specific shot windows listed in `config.window_set`. These are the figures that go into the paper.

**What it shows:** a multi-panel Plotly figure spanning all channels for one shot window, containing:

- The history window (`Wh`, the conditioning context).
- Multiple generated future trajectories (4 samples, since `repeat=4`).
- The true observed future (the target).
- Color-coded mode labels from two sources: the FNOLSTM classifier (the surrogate) and the ground-truth `LHD_label`. `add_mode_bars` indexes `["Unknown","L","D","H"]` by label value, i.e. the `+1`-shifted `LHD_label` convention. Surrogate labels are unshifted (`0=L,1=D,2=H`), so callers add `+1` first (`printing_plots.py:554`). Wrong convention = silently mislabelled modes. See [evaluation-metrics.md](evaluation-metrics.md).

**Output path:** `output/pdfplots/{run_name}/qualitative_samples/`. The layout underneath is:

- `full/`: figures that include the legend.
- `nolegend/`: the same figures without the legend, cleaner for inclusion in the paper.
- Inside each of those, one folder per render size: `300x250/`, `400x450/`, `600x500/`, `800x500/`, `1200x600/`, `1300x910/`, `800x1200/`, `600x1000/`.
- Each size folder contains files named `{shot_number}_{time}s.pdf`.

Additionally, per window:

- `atom_{subgroup}_{shot}_{time}s.pdf`: a compact 500x400 variant with a serif font and minimal margins, written into the subgroup folder.
- `batch_metrics_{subgroup}_{shot}_{time}s.json`: the metric dict computed for that window.

**How to configure:** edit `window_set` in `configs/plasmaflow.yaml`. Remove individual entries to suppress specific windows. Set `window_set: []` to skip window-set plotting entirely.

**Integration steps:** `evaluate_window_set()` uses `n_steps=120` when a GPU is available and `n_steps=5` on CPU. These values are hardcoded inside `evaluate_window_set()` and are not exposed in config.

### Animated window-set figure

Alongside the static PDFs, `animate_window_set()` runs right after `evaluate_window_set()` (also driven by `config.window_set`) and produces a single interactive HTML animation. It keeps the flow animation of `animated_trajectory_plotly` but adds the context of the window-set plots: static history and ground-truth overlays plus filters over every curated window.

**What it shows:** one single-panel Plotly figure spanning all windows and channels, containing:

- The history window (`Wh`), drawn left of `x=0` in the yellow-shaded region, as a static overlay.
- The true observed future (the target), as a static overlay.
- The generated future trajectories (`repeat=4` stochastic samples per window), as dotted lines. These are the only traces that animate.

**Animation axis:** the integration step, not time. Pressing Play sweeps the generated (dotted) traces across the integration `trajectories`, flowing from prior noise toward the sample; the history and target overlays stay fixed across every frame.

**Controls:**

- Shot dropdown: isolate a single window (or "All windows").
- Signal dropdown: isolate a single channel (or "All signals").
- Samples toggle: switch between `1 sample` and `{repeat} samples`; it only touches the extra-sample traces, so it composes with the active dropdown.
- Legend clicks: every trace is its own legend entry (no shared legend group), so clicking entries isolates down to a single `(channel, sample, window)` trace. The dropdowns coarse-filter; the legend composes on top. Note that the two dropdowns themselves do not compose with each other (each overwrites the full visibility state), which is exactly why per-trace isolation lives in the legend.

**Output path:** `output/htmlplots/{run_name}/animated_window_set.html`, and logged to wandb under `val/animated_window_set`.

**Integration steps:** same rule as `evaluate_window_set`: `n_steps=120` on GPU, `n_steps=5` on CPU, hardcoded.

### Rollout browser

`rollout_browser_plotly()` (`src/plotters/rollout_plots.py`) is the interactive companion of the rollout evaluation stage (see [run-lifecycle.md](run-lifecycle.md)). Written by `run_rollouts()` when a `rollout:` block is in the config; rebuildable locally from a cache with `eval_notebooks/rollout_browser.py`.

**What it shows:** one rollout at a time, in stacked shared-x rows: one row per observable channel (generated in vermillion vs real in black, real history in the yellow-shaded region), a controls row (real `c`), and a mode row with the surrogate labels of the generated and real traces as step-lines (unshifted 0=L, 1=D, 2=H).

**Controls:**

- Rollout dropdown: one entry per (shot, start fraction) over the `rollout.html_shots` subset. Each button swaps trace visibility AND the per-rollout layout shapes: the yellow `W_H` rect, the goldenrod rollout-start line, dotted window-boundary lines, and the x range.
- Rangeslider minimap on the bottom axis: the label step-lines double as an overview of the mode structure; drag to scroll through the shot. The signal rows are Scattergl (for the 10 kHz point counts) and do not render inside the rangeslider preview; the SVG label traces do, which is why they live on that axis.
- x axis is actual shot time in seconds, so positions match the shot overview plots.

**Output path:** `output/htmlplots/{run_name}/rollouts.html`, and logged to wandb under `rollout/browser`. Expect tens of MB for 20 full-length rollouts; trace y values are rounded to 4 decimals and share `x0`/`dx` instead of explicit x arrays to keep it manageable.

**Printable counterpart:** `eval_notebooks/paper_rollout.py` renders one PDF per rollout (all X channels + C + gen/real mode bars) in the `paper_single_variate.py` style; see [notebooks.md](notebooks.md).

## Error handling

Every plot function call in the dispatch system is wrapped in a try/except inside `call_plot_functions()` (`src/evaluation.py`). If a plot raises, the error is logged and the run continues rather than crashing. The log line reports the original exception followed by:

```
Continuing like nothing happened... (☞ﾟヮﾟ)☞
```

So a broken plot will never take down a training run; check the logs for that string if a figure goes missing from wandb.

## Adding a new plot type

1. Write a function in `src/plotters/` that accepts `**evaluation_output` kwargs plus `n: int`, `title_base: str`, and any custom kwargs you want to expose through config. Return a `plt.Figure`, a `go.Figure`, or a `wandb.Image`.
2. Register it in `PlotsCallback.PLOT_FN_OPTIONS` in `src/evaluation.py`, mapping a new string key to the function.
3. Add an entry to `evaluation.plot_functions` in `configs/plasmaflow.yaml` referencing that key.

### The evaluation_output dict

Plot functions receive the `evaluation_output` dict (produced by `model.evaluate()`) spread as kwargs. The available keys:

| Key | Type / shape | Contents |
|---|---|---|
| `meta` | dict | `shot_number`, `start_i`, `end_i` |
| `conditioning_input` | dict | `x_history`, `c`, `position_sequence`, `label` |
| `target_samples` | Tensor `(batch, channels, seq_length)` | the true future |
| `prior_samples` | Tensor `(batch, channels, seq_length)` | sampled prior (noise) |
| `generated_samples` | Tensor `(batch, channels, seq_length)` | model output |
| `trajectories` | Tensor `(n_steps, batch, channels, seq_length)` | full integration path |
| `metrics` | dict | computed metric values |
| `peak_features` | dict | `pred_peaks` and `target_peaks` |
| `surr_labels_pred` | ndarray `(batch, history+future)` | surrogate-classifier mode labels for the prediction |
| `surr_labels_target` | ndarray `(batch, history+future)` | surrogate-classifier mode labels for the target |

A new plot function can read whichever of these it needs and ignore the rest, since they all arrive as keyword arguments.
