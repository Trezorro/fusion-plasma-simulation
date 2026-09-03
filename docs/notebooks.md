# Analysis Notebooks

This is the analysis layer: the cell-marked `.py` notebooks (and the central `evaluate.ipynb`) that turn finished-run artifacts into the thesis figures and LaTeX/Excel tables. They were built in an intense sprint in June 2025, lay dormant for four months, and got a final polish on 2025-10-11 ("Final table and plot scripts"). They are powerful but fragile: hardcoded cache names, a hardcoded `exit()`, three notebooks writing the same filenames, a ground-truth alias baked in, and a dependency on wandb CSV exports that are not tracked in git.

If you only read one thing: the notebooks do not run the model (mostly). They read what a finished run left behind. Get those artifacts local first (see [Workflow](#end-to-end-workflow)), then run the notebook that makes the figure you want.

## The two artifact families

Everything downstream comes from one of two places (see [outputs.md](outputs.md) and [evaluation-metrics.md](evaluation-metrics.md) for how they are produced):

1. **HDF5 test caches**: `output/test_cache/{name}.h5` (locally) or `$TEST_CACHE_DIR/{name}.h5` (Snellius: `/scratch-shared/mtresoor/final_cache/`). Written during `trainer.test()`. Two levels of content:
   - Per-window groups `{shot}/{start_idx}/` with `generated_x` (float32, `(channels, seq_length)`), `surr_labels_gen`, `surr_labels_target` (int16, `(history_length + seq_length,)`, **unshifted `0=L, 1=D, 2=H`**, never Unknown; see [evaluation-metrics.md](evaluation-metrics.md) on the two label conventions). Read via `TestStepHDFCache.quick_window(...)`. `start_idx` is **positional within the shot**; `quick_window` maps a time to it via the index, which holds physical time in seconds.
   - Aggregated DataFrames written by `FlowModule.on_test_epoch_end` via the metrics' `extract_df_all(cache)`, at keys `/modes/{condition}/{measure}` (4x4 from/to transition matrices) and `/peaks/{condition}` (peak property rows). Read via `pd.read_hdf(path, key=...)`.
2. **JSON metric friends**: `output/test_cache/{name}.json`, written by `save_json_friend`. Flat `test/final/{condition}/{property}/{statistic}/{channel}` keys. Parsed by a local `parse_metrics_json()` in `peak_analysis.py` and `peaks_tables.py`.

A third, softer input is the **wandb run table** exported by hand to `output/X2.csv` / `output/XR-overview.csv`. These map cache names to model config (prior, history length, class) and to human-readable model names. They are not in git; re-export them from wandb when caches change.

## Notebook reference

| Notebook | Scientific output | Reads | Writes | Status |
|---|---|---|---|---|
| [evaluate.ipynb](../evaluate.ipynb) | Qualitative single-window rollouts, dataset overview histograms, surrogate-label overlays (live from the model) | wandb checkpoint (`find_and_download_model`), run config, datamodule, optional cache | `output/pdfplots/*.pdf`, `output/plots/multiplot/*.html`, mostly inline | Central, reworked 2025-10-11; several dead cells |
| `eval_notebooks/kwali_plots.py` | Multi-model qualitative sample panels ("sequence vs channel") | `fm_toy` config, test df, many `output/test_cache/*.h5` via `quick_window` | `output/pdfplots/seqVSchannel/multimodel_{WxH}/kwali_{SUBNAME}_{shot}_{t}.pdf` (10 sizes) | Fragile (hardcoded cache/model lists) |
| `eval_notebooks/mode_analysis.py` | Mode-transition (L/D/H) tables and bar plots | all `*.h5` `/modes/*`, `output/X2.csv` | `output/tables/mode/{cond}.tex`, `output/pdfplots/{cond}_{measure}.pdf`, appends `modes all.tex` | Fragile (rigid HDF5 keys, GT alias) |
| `eval_notebooks/peak_analysis.py` | Peak/ELM boxplots; peak + window-metric tables (second half) | all `*.h5` `/peaks/*`, `*.json` | `output/pdfplots/peak_boxplot_with_base_DMLratio/.../boxplots_{channel}.pdf`, `output/tables/peak_props_{channel}.xlsx`, `base_metrics_detail.{tex,xlsx}` | Moderate; `exit()` splits the file |
| `eval_notebooks/peaks_tables.py` | Canonical peak-property and window-metric LaTeX/Excel tables (Wasserstein marginal/pairwise, MSE) | `*.json` via `parse_metrics_json` | `output/tables/peak_props_{channel}.xlsx`, `peaks_overview[_split]_{channel}.tex` (6 channels) | Robust, self-contained (run last) |
| `eval_notebooks/paper_peak_tables.py` | Paper peak tables: pairwise-only (count MSE + Wasserstein), PD and DML, for the `paper_single_variate.py` model set | `output/test_cache/{cache}.json` for the 5 paper models (rsynced from Snellius if absent) | `output/tables/paper_peaks_{PD,DML}.tex` | Robust, self-contained; simplified descendant of `peaks_tables.py` |
| `eval_notebooks/moments.py` | None. Wandb run renaming + cache-to-human-name mapping | wandb runs (tag `final_reeval`), CSV | Renames wandb runs in place (`.update()`); no files | Housekeeping; misnamed (see gotchas) |
| `eval_notebooks/model_cache_overview.py` | None. Cache inventory + rsync from Snellius | `output/X2.csv` / `XR-overview.csv`, hardcoded cache list | Console tables; a guarded `rsync` command | Very fragile (stale list, buggy cell) |
| `eval_notebooks/plot_shot_plotly.py` | Interactive single-shot overview: all scoped signals over the L/D/H background | `data/public_data_set/` parquet + `column_to_latex.json` (no caches, no wandb) | `eval_notebooks/shot_{SHOT}_overview.html` | Self-contained; reads the public dataset, not run artifacts |
| `eval_notebooks/rollout_tables.py` | **The rollout result tables.** Per-window peak/mode metrics and the pooled depth table, as LaTeX + CSV, plus the depth-curve PDFs | every rollout cache in its `MODELS` map, data module for real traces | `output/paper_tables/*.tex`, `*.csv`, `depth/{WxH}/*.pdf` | Robust; see [Rollout analysis](#rollout-analysis) |
| `eval_notebooks/paper_rollout.py` | Paper rollout figure: all X + C + gen/real surrogate mode bars, one PDF per rollout | rollout cache `{name}_rollout.h5` (autofetched), data module | `output/pdfplots/paper_rollout/{WxH}/{shot}_{frac}_s{sample_idx}.pdf` | Robust; env-overridable (`ROLLOUT_CACHE_NAME`, `ROLLOUT_PDF_DIR`); `MAX_SAMPLES_PER_START` caps samples printed per start point |
| `eval_notebooks/paper_rollout_models.py` | Appendix figure: PD only, one row per main-table model, same shot and time axis | the caches in `rollout_tables.MODELS` (autofetched) | `output/pdfplots/paper_rollout_models/{WxH}/{shot}_s{sample_idx}.pdf` | Imports its styling from `paper_rollout.py`; `ROLLOUT_MODELS_PDF_DIR` |
| `eval_notebooks/paper_rollout_compare.py` | Timing-vs-capacity panel: PD only, three models overlaid on one ground truth at `f=0.75` | three caches in `MODEL_CACHES` (autofetched) | `output/pdfplots/paper_rollout_compare/{WxH}/{shot}_0.75.pdf` | `ROLLOUT_COMPARE_PDF_DIR` |
| `eval_notebooks/rollout_browser.py` | Interactive rollout browser HTML, rebuilt locally from a cache | rollout cache (autofetched), data module | `output/htmlplots/local/rollouts_{cache}.html` | Thin wrapper around `src/plotters/rollout_plots.py`; `ROLLOUT_CACHE_NAME`, `ROLLOUT_HTML_DIR` |
| `eval_notebooks/rollout_evaluation_script.py` | Exploratory twin of `rollout_tables.py`: cache walkthrough, horizon figures, scratch space | one rollout cache, data module | whatever you make it write | Onboarding / scratch, not a paper artifact. `rollout_cache_explorer.ipynb` is the same thing as a notebook |
| `eval_notebooks/elm_interval_stats.py` | Console stats: real ELM inter-peak intervals on PD, for sanity-checking the OT lambda | reference rollout cache, data module | console only | Tiny, self-contained |

## End-to-end workflow

From a finished run to thesis figures:

0. **Produce the artifacts.** Finish a run or a reeval so the HDF5 cache and its JSON friend exist. On the cluster they land under `/scratch-shared/mtresoor/final_cache/`. See [run-lifecycle.md](run-lifecycle.md) and [hpc-snellius.md](hpc-snellius.md).
1. **Pull caches local.** Use the rsync cell in `model_cache_overview.py` (or an equivalent one-liner) to sync `snellius:/scratch-shared/mtresoor/final_cache/` into `output/test_cache/`. The JSON friends come along with the `.h5` files.
2. **Export the wandb run table.** `mode_analysis.py` and `model_cache_overview.py` need `output/X2.csv` (and/or `output/XR-overview.csv`): the wandb run list with config columns, exported by hand. Skip this only if you are running purely JSON-based notebooks (`peaks_tables.py`).
3. **Run the notebook for the output you want:**

   | You want | Run | Notes |
   |---|---|---|
   | Qualitative sample comparison panels | `kwali_plots.py` | Edit the model-name lists to match caches you actually have local. |
   | One-off live rollout / dataset overview | `evaluate.ipynb` | Generates from the model, not the cache; needs a checkpoint. |
   | Mode-transition tables and bar plots | `mode_analysis.py` | Needs `X2.csv`; ground truth is read only from `FM-Sequence-Gaussian`. |
   | Peak/ELM boxplots | `peak_analysis.py` | Boxplot half runs before the `exit()`. |
   | Peak and window-metric tables (canonical) | `peaks_tables.py` | Run this last: it is the source of truth for `peak_props_*` and `peaks_overview_*`. |
   | Cache inventory / sync | `model_cache_overview.py` | Bookkeeping and rsync only. |
   | Anything rollout (tables, figures, browser) | see [Rollout analysis](#rollout-analysis) | All need a `{name}_rollout.h5` cache from a run with a `rollout:` block; each autofetches it. |

### Gotchas

- **Running one as a plain script needs `PYTHONPATH=.`** These are Jupytext `# %%` cells; interactively VSCode puts the workspace root on `sys.path`, but `python eval_notebooks/x.py` does not, so `import src` fails with `ModuleNotFoundError`. Use `PYTHONPATH=. python eval_notebooks/x.py` from the repo root.
- **Cache labels are unshifted (`0=L, 1=D, 2=H`); the `LHD_label` column is `+1` shifted (`0=Unknown, 1=L, 2=D, 3=H`).** Indexing a colour or name list with the wrong one mislabels modes with no error. See [evaluation-metrics.md](evaluation-metrics.md).
- **`surr_labels_target` is not strictly model-independent.** Same target window, different caches, can differ by a few timesteps (classifier boundary flips). Fine to read the target labels from one cache, but it is that model's view of the target, not a canonical one.
- **`moments.py` does not compute moments.** Despite the name, it pulls wandb runs and renames them; it writes no tables. The moment / window-metric error tables (mean, var, skew, kurtosis of magnitude and first-difference) come out of the JSON branch of `peaks_tables.py` (and `peak_analysis.py`).
- **Filename collision.** `peak_analysis.py` and `peaks_tables.py` both write `output/tables/peak_props_{channel}.xlsx` and `peaks_overview_{channel}.tex`. Last run wins. Treat `peaks_tables.py` as canonical and run it last.
- **`peak_analysis.py` has a hardcoded `exit()`** (around line 359, under `__main__`) that separates the boxplot half from the table half. Comment it out to reach the table cells.
- **Ground-truth alias is baked in.** `mode_analysis.py` and `peak_analysis.py` treat the cache named `FM-Sequence-Gaussian` as ground truth. Rename that model and the "Real" distribution disappears.
- **`MODEL_ORDER` is reversed in `peak_analysis.py`** relative to the other notebooks; do not assume a shared ordering.
- **`model_cache_overview.py` is stale by construction:** a hardcoded ~30-name cache list, a buggy `get_cache_overview(CSV, group_cols, cache_col, caches)` call (the function takes one argument), and a `CACHE_DIR = 'ouptut/test_cache'` typo. Use the rsync cell, ignore the rest.
- **`X2.csv` / `XR-overview.csv` are not in git.** Re-export from wandb; a stale CSV silently drops or mislabels models.
- **`evaluate.ipynb` has dead and destructive cells:** a `from src.evaluate_modes import get_mode_predictions` import that no longer resolves (module is `src.metrics.evaluate_modes`, function refactored), label loops referencing an undefined `val_set` (should be `data_module.test_dataset`), trailing plotters referencing an undefined `simulated_shot`, and a wandb artifact-cleanup loop that calls `art.delete()`. Read before you run.

## Per-notebook detail

### evaluate.ipynb
The original, central post-hoc evaluation notebook. It loads a trained checkpoint from wandb, rebuilds the exact test datamodule from that run's config, and drives single-window and batched rollouts through `FlowModule.evaluate` to make qualitative figures and dataset-overview plots. The `eval_notebooks/` siblings were split out of workflows first prototyped here. Live sections: dataset histograms, "Convenient window query" (single-window rollout), "Run Evaluation and Generate Plots" (batched `model.evaluate` with `rk4`), and the surrogate-label pipeline. Inputs: a checkpoint via `find_and_download_model(selected_run, prefer_alias='latest')` (run picked by name, e.g. `FIND_RUN = "seq_normal_smC_BIG4"`, project `flowtoy`), the run config re-hydrated through `wandb.init(mode="disabled")` + `get_current_config()`, and the parquet datamodule. Outputs: `output/pdfplots/shot_length_histogram.pdf`, per-shot single-window PDFs, and an interactive `output/plots/multiplot/slabels-*.html`; most figures are inline. Fragility: hardcoded run names, project, and probe windows (`57013 @ 1.08s`, `64770 @ 1.4s`), plus the dead cells listed under gotchas.

### kwali_plots.py
Qualitative multi-model comparison: for a window, it stacks the ground truth against N generated samples from each model, with history context and control signals overlaid. Reads model caches through `TestStepHDFCache(model_name).quick_window(shot, t, TEST_DF)`, so every model in its hardcoded lists must be present in `output/test_cache/`. `SUBNAME` is reassigned across cells ("seq_vs_channel", "new", "allC"), which changes the output path; each window is exported at 10 sizes.

### mode_analysis.py
Reads `/modes/{condition}/{measure}` from every cache (11 conditions x 9 measures, each a 4x4 from/to matrix), joins the wandb metadata from `X2.csv` for human names, and emits per-condition LaTeX transition tables plus multi-facet bar and stacked-bar plots. Appends to `output/pdfplots/modes all.tex` without clearing it first. Was renamed from `mode_transtions.py`.

### peak_analysis.py
Two halves separated by a hardcoded `exit()`. First half: `box_plot_peaks()` builds a models x measures grid of ELM/peak boxplots (count, prominence, width, base, energy ratio) with a semi-transparent ground-truth "ghost" overlay and condition-highlight rectangles, exported at several sizes. Second half (after `exit()`): parses the JSON friends into peak and window metric tables.

### peaks_tables.py
The robust, canonical table generator. Parses JSON friends via `parse_metrics_json()`, pivots per channel, and writes Excel plus auto-split LaTeX (chunked at 10 columns) with bold-min highlighting and detailed Wasserstein-marginal / Wasserstein-pairwise / MSE captions. Self-contained: no wandb, no CSV, no HDF5 required for the table export.

### moments.py
Housekeeping. Pulls wandb runs (tag `final_reeval`), maps cache names to human names, and renames runs in place via `find_wandb_run(...).update()`. Produces no files. Named `moments.py` for historical reasons; the actual moment tables live in `peaks_tables.py`.

### model_cache_overview.py
Cache inventory and sync. The one genuinely useful cell builds a selective `rsync` include-pattern command to pull named caches from Snellius into `output/test_cache/`. The rest (existence check against a hardcoded list, config audit) is stale.

### plot_shot_plotly.py
The odd one out in `eval_notebooks/`: it reads the **public confstate dataset** (`data/public_data_set/`), not run caches or wandb, so it needs no finished run. It is the interactive plotly counterpart of `plot_discharge()` (matplotlib, 3 signals) in `giants/TCV-confstate-data/utils/overview.py`. Cell-marked (`# %%`); run it directly to write `shot_{SHOT}_overview.html` next to the script, or step through the cells.

What it draws: one shot, all 46 scoped variables from `column_to_latex.json`, as a tall stack of `make_subplots` rows (one row per category: shaping, magnetics, density, temperature, power, ...), sharing the time x-axis, over the L/D/H confinement-state background.

Design decisions worth knowing:
- **Category = subplot row.** Iterating `column_to_latex` keeps the paper's variable grouping and ordering; a missing parquet column is skipped rather than erroring (e.g. `Halpha_fft` is absent from the parquet).
- **`NORMALISE` (default on)** z-scores each signal independently so variables with wildly different magnitudes stay legible on one shared row; y-axis title switches between "z-scored" and "raw value". Flip to `False` for physical units.
- **State background via contiguous runs.** `add_state_background()` scans the label array, coalesces equal-label runs into a single `add_vrect` span (`layer="below"`), instead of one rect per timestep, keeping the figure light. Colours match `plot_discharge` (L/D/H, plus QCE-H). Because vrects carry no legend, one dummy `Scatter` per present state is added purely for a legend swatch.
- **Legends and hover.** Traces are grouped by category (`legendgroup` + `groupclick="togglegroup"`) so a whole category toggles at once; `hovermode="x unified"` with a per-trace `hovertemplate` shows raw column name, time, and value. `latex_to_name()` strips `$...$`/LaTeX escapes down to a readable legend label.
- **Config at the top:** `SHOT`, `NORMALISE`, `LABEL_COLUMN` (`label_conf`, or `label_conf_qce` for the four QCE shots `[61056, 71344, 78069, 83049]`). Excludes `time`/`label_conf` (axis + background) and the 30 derived Halpha FFT-window columns, which are engineered features, not raw diagnostics.

## Rollout analysis

Everything that reads a `{name}_rollout.h5` cache. A rollout is the model running free on its own
output: real history in, generated window out, generated window back in as the next history, to
the end of the shot. Controls and the time axis always come from the real shot. See
[evaluation-metrics.md](evaluation-metrics.md) and `src/rollout.py`.

**Get the caches first.** They are written on the cluster under `/scratch-shared/mtresoor/final_cache/`
and every script below autofetches the ones it needs with `rsync` into `output/test_cache/`, so the
usual first run is just slow, not broken. Cache names are grid-cell names: see [run_grid.md](run_grid.md).

| Script | Gives you |
|---|---|
| `rollout_tables.py` | The paper's rollout tables and depth plots |
| `paper_rollout.py` | One rollout, everything on it (5 observables, controls, mode bars) |
| `paper_rollout_models.py` | One shot, PD only, all five main-table models stacked |
| `paper_rollout_compare.py` | The timing argument in one panel: flow vs U-Net vs leak-oracle |
| `rollout_browser.py` | A clickable HTML browser over a whole cache |
| `rollout_evaluation_script.py` | A place to poke at a cache yourself |

All of them are Jupytext `# %%` files: step through the cells in VSCode, or run them as scripts with
`PYTHONPATH=. python eval_notebooks/<name>.py` from the repo root.

### rollout_tables.py

The one that produces paper numbers. Reads every cache in its `MODELS` map, measures peaks and mode
labels on each rollout, aggregates, and writes the tables.

```bash
PYTHONPATH=. python eval_notebooks/rollout_tables.py              # everything
PYTHONPATH=. python eval_notebooks/rollout_tables.py --shots 3    # smoke run, ~one shot per model
PYTHONPATH=. python eval_notebooks/rollout_tables.py --force      # ignore the cached parquet
PYTHONPATH=. python eval_notebooks/rollout_tables.py --models main  # skip the appendix ablations
```

The file itself is configuration plus the paper's captions. The work lives in:

| Module | Does |
|---|---|
| `src/rollout_cache.py` | Reads a cache's stamped config, rebuilds the data module from it, refuses mismatched caches |
| `src/metrics/rollout_peaks.py` | Per-window and pooled peak statistics, Dice |
| `src/metrics/rollout_aggregate.py` | The aggregation ladder (window to sample to shot) |
| `src/metrics/ot_peak_error.py` | The unbalanced optimal-transport peak error |
| `src/plotters/latex_tables.py` | Stacked-table assembly, ranking, formatting |
| `src/plotters/rollout_depth.py` | The depth-curve figures |

Outputs land in `output/paper_tables/`: `rollout_results_table.tex` (main),
`rollout_depth_table.tex` (depth-stratified, the pooled OT table), `*_appendix_table.tex`, the CSVs
behind them, and `depth/{WxH}/*.pdf`. Sensitivity variants are written beside the reported ones with
a suffix (`_elmscale`, `_lam30`, `_long22`, `_large_scale`); the reported table keeps the bare name so
the paper's `\input` path never moves.

Two intermediates are cached: `rollout_slice_metrics.parquet` (per window) and
`rollout_pool_metrics.parquet` (per depth stratum), with a `.meta.json` recording what they were
computed for. Change a model, a start fraction or a threshold and they are recomputed automatically;
`--force` does it by hand. Reading the caches is the slow part, rendering is free.

Two measurement choices are deliberate and documented in `src/metrics/rollout_peaks.py`: peaks are
detected once over the whole rollout (not inside each 256-sample window, which would truncate every
prominence and width walk at the boundaries), and the Wasserstein columns are reported only where both
peak sets are non-empty (the empty-side sentinel would reward a model that emits nothing).

### The figure scripts

`paper_rollout.py` is the reference one: pick a cache with `ROLLOUT_CACHE_NAME`, pick a shot and start
fraction in the config block, get one PDF per rollout at several figure sizes.
`paper_rollout_models.py` and `paper_rollout_compare.py` import their styling from it and only change
the layout. Each takes a `ROLLOUT_*_PDF_DIR` env var, so test renders can go to `output/testplots/`
instead of over the paper figures.

### rollout_browser.py

Rebuilds the interactive browser that a run writes to `output/htmlplots/{run_name}/rollouts.html`,
locally, from the cache. Use it to find the shots worth putting in a figure. Set `ROLLOUT_CACHE_NAME`,
open the HTML it prints.

## Reusable snippets worth extracting

Real functionality currently stuck in notebooks that could move into `src/`:

- **Load-run-to-model helper** (`evaluate.ipynb`): select run, download checkpoint, `wandb.init(disabled)`, `get_current_config`, build `FlowModule` + datamodule. The clearest win; every eval notebook re-implements a slice of this. Suggested `src/eval_setup.py::load_run(name) -> (model, data_module, C)`.
- **PD-rollout to surrogate labels** (`evaluate.ipynb`): `get_full_history` + `denormalize` + concat + mode prediction into `surr_labels_target`/`surr_labels_pred`. Suggested `src/metrics/evaluate_modes.py`.
- **wandb artifact prune** (`evaluate.ipynb`): deletes non-best/non-latest model artifacts. Overlaps `prune_online_checkpoints`; consolidate into `src/config.py` or a new `src/wandb_utils.py`.
- **JSON friend parser** `parse_metrics_json()` (`peak_analysis.py`, `peaks_tables.py`): duplicated in two notebooks; belongs in `src/metrics/`.
- **HDF5 introspection** `get_h5_tree`/`print_h5_tree`/`list_mode_keys` (`mode_analysis.py`): handy for any cache debugging.
- **`box_plot_peaks` and `export_pdf`** (`peak_analysis.py`, `kwali_plots.py`): multi-size PDF export and the ghost-overlay boxplot grid.
- **`split_and_write_latex_table`** (`peaks_tables.py`): generic wide-table column chunker for LaTeX.
- **Shot-length histogram** (`evaluate.ipynb`): dataset overview by split. Suggested `src/plotters/data_overview.py`.
- From the reference notebooks (below): `plot_signal_and_spectrum` (dual-axis spectrum), the mode-shaded `plot_shot`, and the sample-entropy tests (`get_sample_entropy`, `ks_test_sample_entropy`).

## Reference and scratch appendix

Prototype/legacy material in `notebooks_and_reference/`. Not maintained; kept for reference and for the handy snippets noted. The `.py` files carry a short module docstring; the `.ipynb` files are cataloged here only.

Cell-marked `.py`:

- `fourier_transform_test.py`: round-trips synthetic signals through torch FFT. Handy: `plot_signals` dual-axis comparison.
- `generate_shot_plot.py`: loads one TCV shot, renders time/spectrum plotly to HTML. Handy: `plot_signal_and_spectrum` (log/linear toggle).
- `metric_test.py`: compares torchmetrics MSE/MSLE against `FourierMSLE` on synthetic signals. Handy: metric-sweep helpers.
- `preview_nb.py`: explores shot timing consistency, renders per-shot observable/control/spectrogram PDFs. Handy: `plot_shot` (mode-shaded trace), `check_time_consistency`.
- `fusion_naive_rnn.py`: early RNN/causal-conv autoregressive baseline. Handy: `ParquetDataset`, `CausalConv1d`. Superseded by `src/data_loaders.py`.
- `fusion_convnet_example.py`: PixelCNN-style causal-conv baseline (Tomczak digits adapted to TCV). Handy: `CausalConv1d`, `ARM`. Superseded by `src/models/`.
- `test_entropy.py`: compares sample-entropy distributions via KS / Wasserstein / Jensen-Shannon. Handy: `get_sample_entropy`, `normalized_entropies`, `ks_test_sample_entropy`; worth lifting into `src/metrics/`.
- `test_on_data_TEMPLATE.py`: minimal `FusionShotDataModule` load + single-shot plot. A good smoke-test starting point.

Jupyter `.ipynb` (not edited):

- `hdf5_testing.ipynb`: benchmarks HDF5 results caching (write / random-lookup timing).
- `FlowModels_colab.ipynb`: external Colab flow-matching tutorial (Scott H. Hawley), reference only, not TCV-specific.
- `preview_proposal_april.ipynb`: early shot/label exploration from the proposal stage.
- `peaks.ipynb`: develops the peak-based ELM metric algorithms later formalized in `src/metrics/peak_metric.py`.

## Public confstate dataset tooling

Standalone helpers that read the public TCV confstate dataset (`data/public_data_set/`: parquet-per-shot, `column_to_latex.json`, `data_splits.json`); they do not touch the model, wandb, or the eval caches.

- `giants/TCV-confstate-data/data_overview.ipynb`: the dataset walkthrough. Prints the split/experiment/variable metadata and calls `utils/overview.py::plot_discharge` (matplotlib) to show one shot as 3 signals on twinned axes over an L/D/H-shaded background.
- `eval_notebooks/plot_shot_plotly.py`: its interactive plotly counterpart, plotting all 46 scoped variables at once. Detailed under [Per-notebook detail](#plot_shot_plotlypy) above.

There is also a `flowmodels_colab.py` export in that directory, left untouched.
