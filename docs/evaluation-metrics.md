# PlasmaFlow Evaluation Metrics

This doc explains every metric PlasmaFlow computes during evaluation: what it measures physically, how it is computed, when it fires, and where it shows up in wandb. The goal is that when you stare at a wandb dashboard you know exactly what each number means and why it moved.

The model generates future plasma diagnostic trajectories conditioned on a history window. Every metric here is fundamentally a comparison between a batch of **generated** futures and the matching batch of **target** (ground-truth) futures. Some metrics compare summary statistics of the two distributions; others compare classifier-derived mode labels; one compares raw sequence shape directly.

---

## When metrics are computed

| Phase | When | Wandb prefix |
|---|---|---|
| Validation (conditional) | `epoch % val_every_n_epochs == 1` OR `epoch <= scrutinize_epochs` | `val/` |
| Test epoch end | after `trainer.test()` | `test/final/` |
| Test step | per test batch | `test/step/` |
| Evaluate (window_set) | after `validate()`, for each shot window | not logged separately; written to PDF metadata JSON |

All metrics are computed inside `model.evaluate()` (validation and the window_set evaluation path) or `update_metrics()` (the test path). The `evaluate()` path calls `self.init_metrics()` at the end to reset all torchmetrics state, so each evaluation starts clean.

The `scrutinize_epochs` clause means the early epochs of a run get full metric computation every epoch, so you can watch a model converge in detail before metrics settle into the cheaper `val_every_n_epochs` cadence.

---

## 1. Moment errors (`MomentsMetric`)

Computed in `src/metrics/metrics.py`. This measures the distributional distance between generated and target sequences using statistical moments. It does not care about temporal alignment of individual samples; it asks whether the *distribution* of values the model produces at each time step has the right shape.

For each channel in `data.cols.x`, and for both the raw signal and its first difference (the step-to-step change):

- Compute the mean, variance, skewness, and kurtosis across the batch, at each time step.
- Measure MSE between the generated distribution's moment and the target's moment.

That gives 8 combinations per channel: {magnitude, diff} x {mean, var, skew, kurtosis}.

**Wandb keys:**

- `{prefix}/error/magnitude_mean_mse/{channel}` (raw signal moments)
- `{prefix}/error/magnitude_var_mse/{channel}`
- `{prefix}/error/magnitude_skew_mse/{channel}`
- `{prefix}/error/magnitude_kurtosis_mse/{channel}`
- `{prefix}/error/diff_mean_mse/{channel}` (first-difference moments)
- `{prefix}/error/diff_var_mse/{channel}`
- `{prefix}/error/diff_skew_mse/{channel}`
- `{prefix}/error/diff_kurtosis_mse/{channel}`
- `{prefix}/error/magnitude_mean_mse/mean` (and the `/mean` variant for every other key: mean across all channels)

**Physical interpretation:** the moments factorize "is the model right?" into distinct failure modes. If the model captures the correct mean trajectory but overestimates spread, you see low `magnitude_mean_mse` but high `magnitude_var_mse`. The diff moments are sensitive to the *roughness* of the signal: a model that produces correct values but too-smooth or too-jagged transitions shows up in `diff_var_mse` even when the magnitude moments look fine.

---

## 2. Entropy metrics

Computed in `src/metrics/metrics.py` using the `antropy` library. These measure the complexity and irregularity of the generated time series, which is a different question from "does it have the right mean and variance". A signal can have correct moments and still be too regular or too noisy.

Three entropy methods (all enabled in the current config):

- **Approximate entropy** (`app_entropy`): quantifies regularity and predictability of the series. Lower means more predictable.
- **Spectral entropy** (`spectral_entropy`): entropy of the power spectral density. Higher means more broadband noise; lower means energy concentrated in a few frequencies.
- **Permutation entropy** (`perm_entropy`): entropy of ordinal patterns (the relative ordering of nearby samples) in the series.

For each channel x method combination, four distance metrics are computed between the batch distribution of entropies (generated vs target):

- MSE, MAE, MSD (mean signed difference), and Wasserstein distance.

The MSD is signed, so it tells you the *direction* of the error: whether the model is systematically generating too much or too little entropy, not just how far off it is.

**Wandb key pattern:** `{prefix}/error/{method}_{distance}/{channel}` and the `{prefix}/error/{method}_{distance}/mean` aggregate.

**Note on sampling frequency:** the spectral entropy implementation uses `sf=100` as the sampling frequency parameter, not the actual 10 kHz sample rate. This reflects a downsampling assumption made when the metric was introduced. It does not affect the validity of relative comparisons (model A vs model B, or generated vs target), but the absolute spectral entropy values should be read with this in mind.

---

## 3. Peak metrics (`PeakMetric`)

Computed in `src/metrics/peak_metric.py`. This measures whether the model produces the right number, shape, and energy of peaks in each channel. Peaks matter physically because many of the events we care about (ELMs, mode transitions, bursts) show up as peaks in the diagnostic signals, so a model can have good moments and good entropy while still getting the *event structure* wrong.

### Detection, and the two synthetic channels

Every channel is analyzed independently with `scipy.signal.find_peaks` at `evaluation.peaks.prominence` (0.001), `width=0`, `rel_height=1.0` (`batch_get_peakprops`, `metrics.py`). On a [0,1]-normalized signal that prominence is deliberately at noise level, so PD and FIR windows hold tens to hundreds of peaks. Thresholds come from config via `get_peak_thresholds`, with the historical values as fallback.

`PeakMetric.CHANNEL_NAMES` is `data.cols.x` plus two **synthetic** channels appended by `batch_get_peakprops` in `SYNTHETIC_CHANNELS` order. Both are ELM-focused views of a real channel, and both are *additive*: the raw `PD` and `DML` channels keep every peak.

| Channel | What it is | Peaks/window (test set) |
|---|---|---|
| `PD` | raw, at `prominence` | ~113 |
| `DML` | raw, at `prominence` | ~58 |
| `PD large peaks` | the PD trace at `elm_pd_prominence` (0.1): ELM-scale H-alpha bursts only | ~7.5 |
| `DML ELM peaks` | DML peaks that coincide with one of those PD bursts | ~7.9 |

**The `DML ELM peaks` gate.** ELMs show up as H-alpha (photodiode) bursts, so a DML excursion is treated as an ELM candidate only if a large PD burst happens at the same time. PD peaks are found at `elm_pd_prominence`, their prominences are summed over a window spanning each DML peak's width (extended to the next DML peak), and the DML peak is kept only if that sum clears `elm_pd_prominence`. The energy properties are attached to the survivors, and only to them.

This gate is why the channel is sparse: it removes ~87% of DML peaks. So `DML ELM peaks` count is not "peaks in DML", it is "DML peaks coincident with an ELM-scale PD burst". Zero in a window is normal, including in ground truth.

A consequence worth holding onto: a model scores on `DML ELM peaks` only if it generates **both** a DML excursion and a coincident ELM-scale PD burst. A model whose PD output is too smooth to clear `elm_pd_prominence` scores zero by construction, whatever its DML channel does. Deterministic baselines do exactly this, so their whole `DML ELM peaks` column can collapse to the zero-peak sentinel path (see the sentinel caveat below), and two unrelated baselines can then produce bit-identical scores.

> **Changed 2026-07.** The gate used to be applied to the DML channel *in place*, so `count`/`prominence`/`width`/`base` for DML silently described only the ~13% of peaks that survived it, and meant something different from the same measures on every other channel. The raw DML view was not recorded anywhere. It is now a separate channel, mirroring how `PD large peaks` had always been done additively. The ELM-burst threshold was also split across two call sites (0.1 for `PD large peaks`, 0.15 for the gate) and is now the single `elm_pd_prominence` key. **Caches written before this change are not comparable on DML**, and their DML numbers are the gated subset despite the label.

**Measured properties for each peak:**

- `height`: absolute signal value at the peak.
- `prominence`: peak prominence (how much the peak stands out from its surroundings).
- `base`: signal value at the base of the peak.
- `width`: peak width at half prominence.

**ELM properties**, on `DML ELM peaks` only (they are undefined without a coincident PD burst):

- `energy_delta`: DML value at the peak minus the minimum in its window (the assumed base energy).
- `pd_prominence`: summed prominence of the coincident PD burst.
- `energy_ratio`: `pd_prominence / energy_delta`.

### Conditions, and how much data backs each one

Peaks are scored separately per mode composition of the **history** window: `L_only_Wh`, `D_only_Wh`, `H_only_Wh` (history entirely in one mode), `mixed` (more than one mode), and `any_Wh` (everything). The four disjoint conditions partition `any_Wh` exactly.

Splitting this way matters because the peak behavior a model *should* produce depends on the regime it starts from. But the conditions are wildly unbalanced, and **`D_only_Wh` is thin enough that it must not be read like the others**:

| Condition | Windows | Share |
|---|---|---|
| `any_Wh` | 61459 | 100% |
| `L_only_Wh` | 39015 | 63.5% |
| `H_only_Wh` | 16460 | 26.8% |
| `mixed` | 4717 | 7.7% |
| `D_only_Wh` | **1267** | **2.1%** |

Pure-dithering history windows are rare in the data: a D box in a boxplot, or a D row in a table, rests on ~2% of the test set and its tails are far less trustworthy than L or H. Treat single-model D differences as suggestive, not decisive.

It compounds on DML. In those 1267 D windows the ground truth has **zero** `DML ELM peaks` in 78.6% of windows (mean 1.80), because D mode has few ELMs, so the effective sample carrying any DML ELM information is only a few hundred windows. A DML/D cell is close to uninformative, and when models also produce nothing there it degenerates entirely into the sentinel.

Read these counts off the `total_hits` keys in a run's JSON friend. They are a property of the test split rather than the model, and are stable across the 2026-07 dataset update (which added covariate columns, not shots): every cache reports the same 61459/39015/16460/4717/1267.

Note the mode-transition metrics report *different* counts for the same-named conditions (`D_only_Wh` 858, 1.4%), because `PeakMetric` conditions on the **true** `LHD_label` (`flow.py` passes `conditioning_input['label'] - 1`) while `ModeTransitionMetric` conditions on **surrogate** classifier labels. Both partition 61459; they disagree on which windows are pure.

**Statistics per property per condition:**

- `mean_pairwise_wasserstein`: average pairwise Wasserstein distance between generated and target peak distributions, window by window.
- `marginal_wasserstein`: Wasserstein distance between the marginal distributions, pooled over all windows.
- `count_mse`: MSE of peak count between generated and target.

**Wandb key pattern:** `{prefix}/peak_{measure}_{stat}/{channel}`

### Caveat: the zero-peak sentinel

`PeakProps.__sub__` (`metrics.py`) is the overloaded subtraction that produces every pairwise distance. When **either** side has no peaks in a window, it does not return a distance between prediction and target. It returns the mean of the *other* side's property, a sentinel:

- both empty: `0.0` for every property.
- prediction empty: `np.mean(target.<prop>)`, a value derived **only from the ground truth**.
- target empty: `np.mean(prediction.<prop>)`.

So a model that predicts nothing in a window is not scored against its prediction, and any two models that predict nothing in the same windows receive mathematically identical scores. That is not a hypothetical: on `DML ELM peaks` in the D condition, deterministic baselines can predict zero in 100% of windows, making their scores a ground-truth constant, bit-identical across unrelated architectures.

This mostly bites the sparse ELM channels. Raw `PD`, `FIR_LIDs_core` and `DML` hold tens of peaks per window for every model, so their pairwise numbers are genuine comparisons. When reading a `DML ELM peaks` or `PD large peaks` cell, check the peak counts before believing the ranking.

Boxplots of raw per-peak values do not suffer from this, since they plot the distributions directly and an empty prediction simply contributes nothing.

**Note on label indexing:** two conventions coexist. Check which one you hold before indexing anything by label value.

| Source | Values | Shift |
|---|---|---|
| `LHD_label` col / `conditioning_input['label']` | `0=Unknown, 1=L, 2=D, 3=H` | `+1` applied in `prepare_data` (`data_loaders.py:478`) |
| `surr_labels_gen` / `surr_labels_target` (HDF5 cache) | `0=L, 1=D, 2=H` | none; raw FNOLSTM `argmax`, 3 classes, never Unknown |

`PeakMetric.test_condition` uses `0=L, 1=D, 2=H`, so `flow.py:443` passes `conditioning_input['label'] - 1`. Mode/transition metrics take surrogate labels unshifted (`flow.py:448`). `add_mode_bars` expects the *shifted* convention, so plotters add `+1` to surrogate labels (`printing_plots.py:554`). That off-by-one is the thing to be careful about.

---

## 4. Mode transition metrics (`ModeTransitionMetric`)

Computed in `src/metrics/mode_metrics.py`. This measures whether the model generates the right *sequence of confinement mode transitions* (L-mode, D-mode/dithering, H-mode). This is arguably the metric group that matters most physically: getting the trajectory shape right is necessary, but the scientifically interesting question is whether the model reproduces the correct mode dynamics.

### Surrogate labels

The true LHD_label is only available for the history (conditioning) window, not for the future the model is supposed to predict. So to score mode behavior on the future, we run the mode classifier (FNOLSTM, the same model used in preprocessing) on both the generated and the target future windows to produce *surrogate* mode labels. This happens in `generate_surrogate_labels_batched()` in `src/metrics/evaluate_modes.py`.

Surrogate label generation details:

- Input: PD (H-alpha) channel only.
- Sliding window of `TW=40` samples with `STRIDE=10` and `OFFSET_PRED=20`.
- Output: one integer label per 10-sample stride.
- The label is shifted by `OFFSET_PRED=20` steps. The LSTM has a 20-step prediction horizon, and this offset aligns the label with the time it actually describes.
- `WINDOW_OF_INFLUENCE_SPILL = min(15, history_length)` (15 in practice): the first and last 15 index positions of the surrogate label sequence are cropped when computing transition metrics, because the classifier has a "spill" where nearby context bleeds into the boundary predictions. Cropping removes those unreliable edge labels.

### Transition matrix

For each sample, a 3x3 transition count matrix is computed from the surrogate label sequence (row = from-mode, col = to-mode). This matrix is then expanded to 4x4 by adding "any" rows and columns, so that transitions *from any mode* or *to any mode* can be queried as a single aggregate.

### Conditions on the history window (Wh)

| Condition | Meaning |
|---|---|
| `L_only_Wh` | history contains only L-mode |
| `D_only_Wh` | history contains only D-mode |
| `H_only_Wh` | history contains only H-mode |
| `L_in_Wh` | history contains at least one L-mode step |
| `D_in_Wh` | history contains at least one D-mode step |
| `H_in_Wh` | history contains at least one H-mode step |
| `any_Wh` | no condition (all samples) |

The `_only_` conditions isolate clean starting regimes; the `_in_` conditions are looser and catch mixed histories.

### Statistics per (condition, from, to) combination

- `expect_target`: expected number of transitions in the target.
- `expect_pred`: expected number of transitions in the generated output.
- `expect_error`: the difference between the two.
- `p_gt0_target`: probability of *any* such transition occurring in the target.
- `p_gt0_pred`: same, for generated.
- `p_gt0_error`: the difference.

The `expect_` family asks "how many transitions"; the `p_gt0_` family asks "does the transition happen at all". A model can get the rate of common transitions right while still missing rare ones, and these two views separate those cases.

**Wandb key pattern:** `{prefix}/mode/{condition}/{from}_{to}_{stat}`

Example: `val/mode/any_Wh/from_L_to_H_expect_target` is the expected number of L->H transitions across all samples.

---

## 5. Dice score

A simple classification-accuracy metric comparing the surrogate mode labels of generated vs target sequences. It is computed on the future window (Wf) only, with the `WINDOW_OF_INFLUENCE_SPILL` crop applied so that unreliable boundary labels do not pollute the score.

**Wandb key:** `{prefix}/dice`

**Interpretation:** 1.0 is perfect mode-sequence reproduction, 0.0 is random. Where the transition metrics measure whether the model gets the right *counts and rates* of transitions, the Dice score measures whether it gets the *timing* right: it rewards the model for putting the right mode at the right time step, not just for producing the right number of transitions overall. A high Dice score means the model generates trajectories that are both plausible-looking and correctly timed.

---

## 6. SoftDTW

Computed in `src/metrics/metrics.py` using `pysdtw` with numba CUDA JIT. Active on GPU only.

Soft dynamic time warping between generated and target sequences. Unlike the moment and entropy metrics (which compare distributions and ignore alignment), SoftDTW compares sequence *shape* directly while being robust to temporal shifts: a generated trajectory that is correct but slightly early or late is not heavily penalized. It is used as a secondary metric.

**Important dependency:** SoftDTW requires `numba >= 0.61` on the cluster. The Pipfile pins an older numba for local compatibility (`ydata-profiling`), so the cluster venv has the correct version installed manually. With `numba 0.58.1` the kernel launch segfaults under the cluster's driver. See the repo CLAUDE.md and the bottom of the Pipfile for the full story.

---

## Metric computation flow

```
evaluate() or update_metrics()
  |
  |-- moments_metrics(generated_samples, target_samples)      # GPU, all channels
  |-- peak_metrics(generated_samples, target_samples, ...)    # GPU, per condition
  |-- mode_test_metrics(pred_labels, target_labels)           # GPU, surrogate labels
  |-- dice_metric(pred_labels, target_labels)                 # GPU
  |
  [move to CPU]
  |
  |-- get_entropy_metrics(generated_samples, target_samples)  # CPU, antropy
  |-- cpu_batch_peak_metrics(generated_samples, ...)          # CPU, scipy
```

The GPU-resident metrics run first while the tensors are still on device; the batch is then moved to CPU for the metrics that rely on CPU-only libraries (`antropy` and `scipy.signal`). SoftDTW, when active, runs on GPU as part of the GPU block.

---

## Quick reference: which metric answers which question

| Question | Metric group |
|---|---|
| Does the output have the right value distribution (mean/spread/shape) per time step? | Moment errors |
| Is the signal the right amount of regular vs noisy? | Entropy metrics |
| Does it produce the right peaks/events (count, shape, energy)? | Peak metrics |
| Does it produce the right mode transitions (rates and probabilities)? | Mode transition metrics |
| Does it get the mode timing right? | Dice score |
| Is the overall sequence shape right, allowing for small time shifts? | SoftDTW |
