# Analysis Notebooks

This is the analysis layer: the cell-marked `.py` notebooks (and the central `evaluate.ipynb`) that turn finished-run artifacts into the thesis figures and LaTeX/Excel tables. They were built in an intense sprint in June 2025, lay dormant for four months, and got a final polish on 2025-10-11 ("Final table and plot scripts"). They are powerful but fragile: hardcoded cache names, a hardcoded `exit()`, three notebooks writing the same filenames, a ground-truth alias baked in, and a dependency on wandb CSV exports that are not tracked in git.

If you only read one thing: the notebooks do not run the model (mostly). They read what a finished run left behind. Get those artifacts local first (see [Workflow](#end-to-end-workflow)), then run the notebook that makes the figure you want.

## The two artifact families

Everything downstream comes from one of two places (see [outputs.md](outputs.md) and [evaluation-metrics.md](evaluation-metrics.md) for how they are produced):

1. **HDF5 test caches**: `output/test_cache/{name}.h5` (locally) or `$TEST_CACHE_DIR/{name}.h5` (Snellius: `/scratch-shared/mtresoor/final_cache/`). Written during `trainer.test()`. Two levels of content:
   - Per-window groups `{shot}/{start_idx}/` with `generated_x` (float32), `surr_labels_gen`, `surr_labels_target` (int16 L/D/H labels). Read via `TestStepHDFCache.quick_window(...)`.
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
| `eval_notebooks/moments.py` | None. Wandb run renaming + cache-to-human-name mapping | wandb runs (tag `final_reeval`), CSV | Renames wandb runs in place (`.update()`); no files | Housekeeping; misnamed (see gotchas) |
| `eval_notebooks/model_cache_overview.py` | None. Cache inventory + rsync from Snellius | `output/X2.csv` / `XR-overview.csv`, hardcoded cache list | Console tables; a guarded `rsync` command | Very fragile (stale list, buggy cell) |

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

### Gotchas

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

There is also a `flowmodels_colab.py` export in that directory, left untouched.
