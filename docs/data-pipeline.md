# PlasmaFlow Data Pipeline

How raw TCV shot data becomes training batches. There are three stages:

1. **Raw shot parquets to one combined parquet** (`src/run_processing.py`, local only).
2. **Combined parquet to windowed samples** (`FusionShotDataset` in `src/data_loaders.py`).
3. **Lightning DataModule** wiring (`FusionShotDataModule`).

The end product the model trains on is a set of `(history window, future window)` pairs sliced out of long shot time-series, normalized to `[0, 1]`, conditioned on control signals and a confinement-mode label.

---

## Signals at a glance

A TCV shot is a multi-channel time-series sampled at 10 kHz (one sample every 0.1 ms). The columns the model uses are split into observables (`X`, what we generate) and conditioning controls (`C`, what we condition on). Physical meaning of the notable signals:

| Column | Role | Physical meaning |
|---|---|---|
| `FIR_LIDs_core` | X | Far-infrared interferometer, core line: line-integrated electron density along the central chord. |
| `PD` | X | H-alpha photodiode on the divertor: H-alpha light intensity, a proxy for recycling and edge/divertor activity. The mode classifier reads this channel. |
| `DML` | X | Diamagnetic loop: magnetic response that correlates with the stored energy in the plasma. |
| `POHM` | X | Ohmic heating power: power dissipated by plasma resistivity (ohmic friction). |
| `Z_axis` | X | Vertical position of the plasma magnetic axis; deviation from the reference is what matters. |
| `IP` | C | Plasma current (reference). |
| `PNBI`, `PNBI2` | C | Neutral beam injection heating power, beam 1 and beam 2 (two separate beams in the new dataset). |
| `PECRH` | C | Electron cyclotron resonance heating power (the microwave/magnetron heating). |
| `MINRAD` | C | Plasma minor radius (horizontal half-width of the plasma cross-section). |
| `KAPPA` | C | Plasma elongation (vertical shape parameter). |
| `DELTA_TOP`, `DELTA_BOTTOM` | C | Upper and lower triangularity (plasma cross-section shape, split into two parameters). |

Which columns are actually loaded as `X` and `C` is set in `configs/plasmaflow.yaml` under `data.cols`. The paper config uses `X = [FIR_LIDs_core, PD, DML, POHM, Z_axis]` and `C = [IP]`, with the other control signals commented out.

The confinement-mode label is `LHD_label`, an integer per timestep: `0 = L-mode`, `1 = D-mode (dithering)`, `2 = H-mode`.

---

## Stage 1: Raw data to combined parquet

**Script:** `src/run_processing.py`. Entry point is `combine_public_dataset()`, called from `__main__`.

This stage runs **locally only**. It imports `ydata_profiling` at module top (used by `generate_report()`), which is a dev-only dependency and is not installed on the cluster.

### Inputs

| Input | Path | Notes |
|---|---|---|
| Individual shot parquets | `data/public_data_set/data/TCV_confstate_*.parquet` | One file per shot; shot number parsed from the filename. |
| Column-to-LaTeX map | `data/public_data_set/metadata/column_to_latex.json` | 67-column name mapping (for plotting and reference). |
| Data splits | `data/public_data_set/metadata/data_splits.json` | Published train/val/test split definitions. |
| Experiment-to-shot map | `data/public_data_set/metadata/experiment_to_shot.json` | Maps experiment identifiers to shot numbers. |

The shot parquets come from Yoeri Poels' published TCV confinement-state dataset (Poels, Venturini et al., *Nuclear Fusion* 2025), available on Zenodo at <https://zenodo.org/records/16631053>. The dataset is tracked as a git submodule at `giants/TCV_confstate_data/`.

### Column selection and renaming (old to new dataset)

The new public dataset uses different column names than the older local dataset PlasmaFlow was first built on. `run_processing.py` selects the columns listed in `ALL_SIG_COLLS` (= `COLS_META + COLS_CONTROL + COLS_DATA`) and applies one explicit rename. Key correspondences:

| Old name | New name | Note |
|---|---|---|
| `FIR_core` | `FIR_LIDs_core` | Density interferometer, core line. |
| `PD` | `Halpha1` (in source), renamed **back** to `PD` | H-alpha divertor photodiode. The processing step renames `Halpha1 -> PD` for compatibility with the older code and metrics; `plasmaflow.yaml` therefore keeps `PD`. |
| `DML` | `DML` | Unchanged. |
| `POHM` | `POHM` | Unchanged. |
| `Z_axis` | `Z_axis` | Unchanged. |
| `IP` | `IP` | Unchanged (reference plasma current). |
| `NBI` | `PNBI`, `PNBI2` | NBI split into two separate beams. |
| `ECRH` | `PECRH` | Heating power. |
| `a_minor` | `MINRAD` | Minor radius. |
| `DELTA` | `DELTA_TOP`, `DELTA_BOTTOM` | Triangularity split top/bottom. |

The single rename actually performed in code is `Halpha1 -> PD`; the other rows above describe how the new dataset's native column names map to the historical names, and the new names are what the config and processing code reference directly.

### Per-shot processing

For each `TCV_confstate_*.parquet`, `combine_public_dataset()`:

1. Inserts a `ShotNum` column from the filename and renames the source label column `label_conf -> LHD_label`.
2. **Minimum length filter:** skips shots shorter than `min_steps_filter` (default 5000 samples).
3. Checks time consistency: requires a monotonically increasing time index at roughly 10 kHz; skips shots whose sampling is too irregular.
4. Selects `ALL_SIG_COLLS`; skips a shot if any required column is missing.
5. Downcasts float64 columns to float32, then renames `Halpha1 -> PD`.
6. **NaN handling:** fills missing `PNBI` (beam power) values with `0`, per Yoeri's guidance (no beam means zero power). Then it finds the longest contiguous NaN-free window across the remaining columns and slices the shot to that window. If any column still has too few usable consecutive samples (its "consecutive ratio" falls below 0.3), the whole shot is dropped.

### The LHD label and the mode classifier

`LHD_label` (confinement mode `0 = L`, `1 = D`, `2 = H`) ships **with** the public dataset as the `label_conf` column and is simply renamed during Stage 1. It is not recomputed at preprocessing time in the current `combine_public_dataset` path.

The confinement-mode classifier itself, Yoeri Poels' **FNOLSTM** model, is used downstream at **evaluation** time, in `src/metrics/evaluate_modes.py`, to segment generated rollouts into L/D/H modes so generated mode transitions can be compared against ground truth. It is defined in `src/metrics/LDH_model.py` and loaded from:

| Asset | Path |
|---|---|
| Weights | `configs/MHD_model_yoerie/weights_PD.pt` |
| Normalization stats | `configs/MHD_model_yoerie/stats_PD.json` |

The classifier operates on the `PD` (H-alpha divertor photodiode) channel only. Its sliding-window settings (in `evaluate_modes.py`) are `TW = 40` samples (input window), `STRIDE = 10` samples, and `OFFSET_PRED = 20` (it predicts 20 steps ahead relative to its input window). The `PD` channel is located by name via `C.data.cols.x.index("PD")`, so removing or renaming `PD` in `cols.x` breaks the mode metrics at runtime.

### Output

| Property | Value |
|---|---|
| Path | `data/{DATE}-TCV_shots_V2.parquet` (e.g. `2026_06_29-TCV_shots_V2.parquet`) |
| Size | ~206 MB |
| Index | `time` (seconds) |
| Columns | `ShotNum`, `time_step`, `LHD_label`, and all selected `X` and `C` signal columns; all shots concatenated. |

---

## Window geometry

Stages 2 and 3 carve each shot into fixed windows. The relevant config values (`configs/plasmaflow.yaml`, under `data`) are:

| Param | Paper value | Meaning |
|---|---|---|
| `seq_length` | 256 | Length of the **future** window `X` the model generates. |
| `history_length` | 256 (`= seq_length`) | Length of the **history** window used for conditioning. `0`/null disables history conditioning. |
| `crop_margin` | 1024 | Unused guard band at the start and end of each shot. Must be `>= history_length`. |

A start index `start_i` is valid only when both the history window before it and the future window after it fit inside the cropped region:

```
crop_margin  <=  start_i  <=  shot_len - crop_margin - seq_length
```

So the **minimum viable shot length** is `seq_length + 2 * crop_margin`. At `seq_length = 256`, `crop_margin = 1024` that is **2304 samples** (0.23 s at 10 kHz). Shots shorter than this contribute no windows and are logged as skipped.

---

## Stage 2: Combined parquet to windowed samples

**Class:** `FusionShotDataset` in `src/data_loaders.py`. One instance is created per split (train, val, test).

### Precomputing window indices

At construction, `precompute_indices()` builds `self.viable_indices`, a flat list of `(shot_number, start_i)` pairs, by walking every shot in the split's dataframe and enumerating every valid `start_i` in the range above.

- **Stride.** For the **test** set the start indices are strided by 10 (every 10th index kept); for train and val the stride is 1 (all indices). This is a performance trade-off in the precompute step: at stride 1 the test set has 2M+ windows, which is too much to precompute and iterate; stride 10 brings it down to ~217K. (For reference, the configured test shots yield ~217K windows at stride 10; train yields ~1.96M at stride 1.)
- **Shuffling.** Train and val index lists are shuffled in place when `pre_shuffle=True`. The test list is always sorted by `start_i` so test windows come out in time order.
- Shots too short to yield any window are logged with a warning and skipped.

### Splitting

After Stage 3 normalizes the combined dataframe, it is subset by shot number into three dataframes and handed to three `FusionShotDataset` instances. The split is **by hard-coded shot lists** in `configs/plasmaflow.yaml` (`data.train_shots`, `data.val_shots`, `data.test_shots`). There is no random splitting; a shot belongs to exactly one split.

### `__getitem__(idx)`

Looks up `(shot_number, start_i)` from `self.viable_indices[idx]` and returns a 3-tuple `(meta, conditioning_input, x)`:

- **`meta`** (dict): bookkeeping for plotting and metrics, including `shot_number`, `start_i`, `end_i` (= `start_i + seq_length`), the corresponding `start`/`end` times, `full_history_start`, and (when history is used) `history_start` / `history_start_i`.
- **`conditioning_input`** (dict), keyed by what the `conditioning` config requests:
  - `x_history`: the history window, observables `X` over `[start_i - history_length : start_i]`. Shape `(x_channels, history_length)`.
  - `c`: control signals over the conditioning window. Shape `(c_channels, window_length)`.
  - `label`: integer `LHD_label` values (0/1/2) over the window.
  - `position_sequence`: time in seconds for each timestep in the window (float32). This is the **raw** time index, passed through unnormalized.
- **`x`** (the target): the **future** window, observables `X` over `[start_i : start_i + seq_length]`. Shape `(x_channels, seq_length)`.

Time is the last axis throughout (channels first). A final assertion guards against NaNs leaking into the target window (usually a sign of a constant column causing a divide-by-zero in normalization).

### Column dependencies (hard requirements)

- Every name in `cols.x` and `cols.c` must exist in the parquet, spelled exactly.
- `cols.label` must exist and contain integers `0`, `1`, `2` (L/D/H).
- `PD` must be present **and** must stay in `cols.x`: the mode classifier in `src/metrics/evaluate_modes.py` locates it positionally with `C.data.cols.x.index("PD")`. Renaming or dropping `PD` raises a runtime error in the mode metrics.

---

## Stage 3: Lightning DataModule

**Class:** `FusionShotDataModule` in `src/data_loaders.py`, implementing the Lightning `DataModule` interface.

### `prepare_data()`

- Reads the combined parquet with `pd.read_parquet(dir + file)`.
- Casts `ShotNum` to `int32` (smaller, faster indexing).
- **Forward-fills NaNs** across the whole dataframe with `df.ffill()`: the last valid measurement is carried forward in time. (Stage 1 already trims to NaN-free windows, so this mainly catches residual gaps.)
- Creates derived columns on demand, only if their name appears in the config:
  - `DML-r` = `DML * -1` (reversed diamagnetic loop).
  - `NBI-median` = per-shot centered rolling median of `NBI` over 100 samples.
  - `ECRH-median` = per-shot centered rolling median of `ECRH` over 100 samples.

### `setup(stage)` and normalization

`setup()` calls `normalize_xc_data()`, then builds the three `FusionShotDataset` instances.

`normalize_xc_data()`:

- Computes `min` and `max` **from the TRAIN split only**, per column, to avoid leaking validation/test statistics into the scaling.
- Applies min-max scaling `(x - min) / (max - min)` to all `X` and `C` columns, **globally** across the entire dataframe (every shot scaled by the same train-derived min/max, not per shot).
- Stores `self.min` and `self.max` as pandas `Series` indexed by column name, and logs them to the wandb config under `data.train_stats.min` / `data.train_stats.max`.
- Caches `self.min_vals_x` and `self.max_vals_x` as GPU tensors (shaped for broadcasting over the channel axis) for fast `denormalize()` at test time.

Because scaling is global and train-derived, any generated sample must be inverted with `data_module.denormalize(x)` (which copies the input, moves the cached min/max to the input's device, and applies `x * (max - min) + min`). Do not re-derive per-shot statistics.

**Not everything is normalized.** Only `X` and `C` columns are min-max scaled. `position_sequence` and `label` are passed through untouched:

### DataLoaders

| Method | Split | Notes |
|---|---|---|
| `train_dataloader()` | train | `DataLoader(..., shuffle=True, num_workers=2 on GPU)`. The index list is also pre-shuffled in `precompute_indices`. |
| `val_dataloader()` | val | Same batch size, no extra shuffle. Asserts normalization has run. |
| `test_dataloader()` | test | No shuffle; supports a `batch_size_override`. Windows arrive in time order. |

`num_workers` is `2` when CUDA is available, else `0`.

---

## Quick reference: one window's journey

1. `run_processing.py` writes `data/{DATE}-TCV_shots_V2.parquet` (per-shot cleaned, label-renamed, NaN-trimmed, concatenated).
2. `prepare_data()` loads it, casts `ShotNum`, forward-fills, adds any derived columns.
3. `normalize_xc_data()` scales `X`/`C` to `[0, 1]` using train-split min/max.
4. `precompute_indices()` enumerates `(shot, start_i)` windows per split (stride 10 for test, 1 otherwise).
5. `__getitem__` returns `(meta, conditioning_input, x)`: history window plus controls/label/positions as conditioning, future window as the generation target.
6. At evaluation, denormalize generated `x` with `data_module.denormalize`, and segment the `PD` channel into L/D/H modes with the FNOLSTM classifier in `evaluate_modes.py`.
