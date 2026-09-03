# Handoff: paper-quality peak boxplot script

Task for a fresh agent. Rewrite `box_plot_peaks()` from `eval_notebooks/peak_analysis.py` into a clean,
runnable paper script, the same treatment already given to the tables and the single-variate plots.

Read this whole document before touching code. Section 3 is a blocker: **every existing cache is obsolete**
and the whole experiment grid is being retrained. You can write and structure the script now, but you cannot
produce the final figure until the new caches land, and the channel set has changed under you.

## 1. Goal

Produce `eval_notebooks/paper_peak_boxplots.py`:

- Driven by a `MODELS = [(cache_name, print_name), ...]` list, exactly like `paper_single_variate.py:305` and
  `paper_peak_tables.py:28`. Milan supplies cache names and print names; nothing else should need editing to
  change the model set.
- Runnable as `PYTHONPATH=. python eval_notebooks/paper_peak_boxplots.py` from the repo root.
- Self-contained: no wandb, no `output/X2.csv`, no `MODEL_ORDER` global, no hardcoded `exit()`.
- **Pick new export sizes.** The thesis figure had ~16 model rows; the paper has ~5, so it is a much shorter,
  wider figure and the thesis aspect ratios no longer fit. Choose a `sizes` tuple suited to few rows rather
  than copying the thesis list, and drop the tall `_vertical*` shapes that only made sense at 16 rows.

The two existing paper scripts are the style reference. Match them, do not invent a third style:

| Script | What to copy from it |
|---|---|
| [eval_notebooks/paper_peak_tables.py](../eval_notebooks/paper_peak_tables.py) | Module docstring shape (`Scientific output/Inputs/Outputs/Usage/Caveat`), `MODELS` list, `fetch_missing()` rsync-from-Snellius helper, plain `if __name__ == "__main__"` |
| [eval_notebooks/paper_single_variate.py](../eval_notebooks/paper_single_variate.py) | `_export(fig, subdir, ..., sizes)` multi-size PDF helper (`:194`), `_REPO_ROOT` sys.path bootstrap (`:47`), serif/muted print style, `PDF_DIR` layout |

## 2. Reference output and source

- **Target look**: `output/pdfplots/peak_boxplot/peak_boxplots_15x14_horiz/boxplots_PD.pdf` (thesis figure,
  Jun 17 2025). Milan is happy with this; the job is to reproduce it cleanly, not redesign it.
  Note the reference lives under `peak_boxplot/`, while the current code writes to
  `peak_boxplot_with_base_DMLratio/` (`peak_analysis.py:133`). The `_with_base_DMLratio` directory is the
  later variant that added the `base` and `energy_ratio` columns. Pick one output dir and say so in the docstring.
- **Source function**: `box_plot_peaks()` at [eval_notebooks/peak_analysis.py:143-355](../eval_notebooks/peak_analysis.py#L143-L355).
- **Data loader**: `iter_peak_properties_per_model()` at [eval_notebooks/peak_analysis.py:73-109](../eval_notebooks/peak_analysis.py#L73-L109).
- Background on the notebook family: [docs/notebooks.md](notebooks.md).

### What the figure is

A grid of horizontal boxplots. One **row per model**, one **column per peak measure**
(`count`, `prominence`, `width`, `base`, plus the ELM energy measures on the ELM channel only). Within each
panel, one box per history-window mode condition (L, D, H, mixed, all), color-coded. Every non-ground-truth
row also carries a semi-transparent "ghost" copy of the ground-truth boxes behind it, so each model is read
against the target.

Design details worth preserving:
- `condition_palette` L `#1f77b4`, D `#ff7f0e`, H `#d62728`, mixed `purple`, all `#444444` (`peak_analysis.py:148`).
  These match `paper_peak_tables.py` and `peaks_tables.py`; keep them consistent across the paper.
- `MEASURE_LABEL` (`peak_analysis.py:52`) holds the LaTeX column titles.
- `showfliers=False`, `sns.despine`, y-ticklabels blanked, measure title only on the top row.

**The old code's `if channel == "DML": measures.append('energy_ratio')` (`peak_analysis.py:153`) is now wrong**,
see section 3.2. The energy measures live on `DML ELM peaks`, not on `DML`. This also settles the old
`energy_ratio` vs `energy_delta` label confusion between the two notebooks: they are different measures, both
defined only on the ELM channel, and `energy_ratio` is literally `pd_prominence / energy_delta`. Pick one,
label it for what it is, and do not carry the other notebook's label across.

## 3. BLOCKER: every existing cache is obsolete

Three independent things invalidate the current caches. Do not build the final figure against them, and do
not spend time salvaging them. **Wait for the re-run**, then point `MODELS` at the new names.

### 3.1 The whole grid is being retrained under a new naming scheme

The old caches straddle two datasets: the Jul 1-8 runs predate a parquet that added leakage covariates, the
Jul 13+ runs use it. Normalization is derived from the train split, so those are not comparable and re-eval
cannot fix it. Milan is retraining the full grid on the current dataset under new names.

**Read [docs/run-grid.md](run-grid.md) for the naming scheme and the grid.** Cache names will look like
`C0-cfm-noleak-normal-s03` (headline CFM), `U0-unet-noleak`, `I3-itransformer-triple`, and so on. Ask Milan
which cells the figure should show rather than guessing; the grid has 10 cells and the figure wants ~5 rows.

### 3.2 The channel set changed: DML is now two channels

This lands directly on the boxplot code. `PeakMetric.CHANNEL_NAMES` is now `data.cols.x` plus **two**
synthetic channels, `SYNTHETIC_CHANNELS = ["PD large peaks", "DML ELM peaks"]`, so seven in total.

| Channel | Was | Is now |
|---|---|---|
| `DML` | silently ELM-gated: only ~13% of peaks, ~7/window | **raw**, every peak, ~58/window |
| `DML ELM peaks` | did not exist | the ELM-gated subset, ~7.9/window, carries the energy measures |
| `PD large peaks` | PD at prominence 0.1 | unchanged in meaning |

Until 2026-07 the ELM gate was applied to the DML channel **in place**, so DML's `count`/`prominence`/`width`/
`base` described only the gated subset and meant something different from the same measures on every other
channel, with the raw view recorded nowhere. It is now additive, mirroring how `PD large peaks` always worked.
The ELM-burst threshold was also split across two call sites (0.1 and 0.15) and is now one config key,
`evaluation.peaks.elm_pd_prominence`. See [docs/evaluation-metrics.md](evaluation-metrics.md).

Consequences for you:
- The energy measures (`energy_delta`, `pd_prominence`, `energy_ratio`) key off `DML ELM peaks`, **not** `DML`.
- A `DML` boxplot from a new cache will look nothing like one from an old cache. That is expected, not a bug.
- Decide with Milan whether the figure shows `DML`, `DML ELM peaks`, or both. They answer different questions:
  raw DML is "does the model get the signal's peak structure right", ELM DML is "does it put ELMs in the right
  places". The paper's argument is about the latter, but the former is the one comparable to PD and FIR.

### 3.3 Root cause of the missing `/peaks/*` groups (historical, do not re-investigate)

Kept because it explains the old caches and the `-e2` re-eval convention in [run-grid.md](run-grid.md). Three
of the five old paper caches held only `/peaks/D_only_Wh` instead of all five conditions.

Every cache written on or before 2026-07-08 has only `D_only_Wh`; every cache from 2026-07-13 onward has all
five. The mechanism:

1. `torchmetrics.MetricCollection` iterates its keys **alphabetically**, not in declaration order. For
   `flow.py:377-381` that is `D_only_Wh, H_only_Wh, L_only_Wh, any_Wh, mixed`, so **`D_only_Wh` runs first**.
2. In the pre-fix code, `flow.py`'s `on_test_epoch_end` called `sub_metric.export_2d_NBI_distributions()`
   *inside* the `for sub_metric in self.peak_metrics.children()` loop, right after `extract_df_all()`.
3. That call raised on the first iteration, killing the whole test hook before the other four conditions were
   ever written. From `output/snellius/slurms/gpujob-24505063-R-NormalMidAttSig03_anim.out`:

   ```
   15:05:41 src.models.flow[INFO]: Mode metrics saved! Extracting peak metrics to cache.
   Traceback (most recent call last):
     File ".../src/models/flow.py", line 470, in on_test_epoch_end
       sub_metric.export_2d_NBI_distributions()
     File ".../src/metrics/peak_metric.py", line 292, in get_nbi_pd_count_per_window_sample
       pred_joint_nbi_pd_window = torch.stack(
   RuntimeError: stack expects each tensor to be equal size, but got [0] at entry 0 and [1267] at entry 1
   ```

   `1267` is exactly the D_only_Wh window count, and `[0]` is the empty NBI window-means state: this run had
   no usable NBI column, so the joint NBI/PD histogram had nothing to stack.
4. Commit `db234b3` ("Hotfix nbi histgram bug (missing nbi column)", 2026-07-08 13:07) commented that call out.
   The R- job was already running at 13:00:35 that day, so it executed pre-fix code and crashed at 15:05.

**Current `main`/`paper` code no longer has this bug** (`flow.py:479` is the commented-out line). A fresh
test/eval run of the three affected models will write all five groups.

The fix is already in (`flow.py:479` is the commented-out call), so the re-run writes all five groups. The
lesson worth carrying: that was an **eval-side** crash with perfectly good weights, which is why
[run-grid.md](run-grid.md) distinguishes a re-eval (`-e2` suffix) from a retrain (`v2` in the ID).

## 4. Statistical caveat: the D condition is thin

Milan flagged this and it is confirmed. **Pure-D history windows are rare**: `D_only_Wh` is 1267 windows out of
61459 in the test set (2.1%), against L_only 39015 (63.5%), H_only 16460 (26.8%), mixed 4717 (7.7%). Every D box
is built from ~2% of the data, so its whiskers are far less trustworthy than L or H. These counts come from the
`total_hits` keys in any JSON friend, are a property of the test split rather than the model, and are unchanged
by the dataset update (which added columns, not shots).

It compounds on the ELM channel: in those 1267 D windows the ground truth itself has zero `DML ELM peaks` in
78.6% of them, because D mode has few ELMs. A DML-ELM/D box is close to uninformative.

**Do not generalize that into "the models generate no peaks".** They do, abundantly, which is exactly why the
raw `DML` channel now exists. Measured per-window generated counts in those 1267 D windows:

| Model | PD | DML (raw) | FIR_LIDs_core |
|---|---|---|---|
| CFM (ours) | 66.9 mean, 0% zero | ~58 mean, 0% zero | 21.7 mean, 0% zero |
| U-Net (non-oracle) | 52.1 mean, 0% zero | ~58 mean, 0% zero | 53.1 mean, 0% zero |
| iTransformer (oracled) | 78.5 mean, 0% zero | ~58 mean, 0% zero | 72.7 mean, 0% zero |
| *Ground truth* | *76.7 mean, 0% zero* | *~58 mean, 0% zero* | *21.7 mean, 0% zero* |

Noise means peaks. The PD and FIR figures above are measured per model on the old caches; the raw DML figure is
the population average over random test windows, since the old caches never recorded raw DML. Re-measure per
model once the new caches exist.

The sentinel caveat (zero-peak windows score against a target-only constant, so unrelated models can come out
bit-identical) is documented in [docs/evaluation-metrics.md](evaluation-metrics.md). It applies to the sparse
ELM channels and to `paper_peak_tables.py`, not to boxplots of raw per-peak values: those plot distributions
directly and an empty prediction simply contributes nothing. It only becomes your problem if you add a derived
distance panel.

## 5. Cruft to remove when porting

`box_plot_peaks()` carries thesis-era baggage. All of the following should go or be reconsidered:

- **`MODEL_ORDER` reversed at definition** then sliced `[1:]` (`peak_analysis.py:29-51`). Replace with the flat
  `MODELS` list. Milan's order is the display order, top to bottom, no reversing.
- **Ground-truth aliasing**: `iter_peak_properties_per_model` hardcodes `FM-Sequence-Gaussian.h5` as the source of
  the `"Ground Truth"` row (`peak_analysis.py:78`). That cache is thesis-era and is not in the paper set. Take the
  ground truth from the `distribution == "Real"` rows of one of the actual paper caches, and say which one in the
  docstring. Caveat already documented in `paper_single_variate.py`: the target is *nearly* but not exactly
  model-independent (surrogate-label boundary flips differ slightly between caches).
- **Ghost-overlay ordering dependency**: the ghost logic assumes `i == 0` is Ground Truth and stashes
  `ground_truth_data` (`peak_analysis.py:193-195`). If the first model is not GT, `ground_truth_data` is undefined
  and it raises `NameError` on the next row. Load the GT explicitly before the loop instead.
- **`if "seq" in model.lower()`** draws a lightblue "Sequential Conditioning" rectangle (`peak_analysis.py:216-231`).
  That is a thesis variant distinction (sequence vs channel conditioning) which does not exist in the paper model
  set. Drop it, or, if a highlight is still wanted, key it off something real such as oracle vs non-oracle. Note it
  also depends on `fig.subplotpars` arithmetic that breaks when the figure is later resized for export.
- **`if 'rect' in locals()`** (`peak_analysis.py:289`) to decide whether to add the rectangle to the legend. Fragile;
  goes away with the rectangle.
- **`'unet' in model.lower()` forces italic** (`peak_analysis.py:268`). Paper print names include
  "U-Net (elm oracle)", so this would italicise some rows and not others by accident. `.claude/CLAUDE.md` does say
  italics for model/variant names, so make it an explicit per-model property, not a substring sniff.
- **Seven copy-pasted export blocks** (`peak_analysis.py:316-351`), each mutating `fig.set_size_inches` then saving,
  with inconsistent dir naming (`{w}x{h}` vs `{w:.0f}x{h:.0f}`, `_vertical`/`_vertical2`/`_vertical25`/`_horiz`).
  Replace with one `sizes=(...)` tuple and the `_export` helper from `paper_single_variate.py:194`.
  The reference figure is `15x14_horiz`, so keep that size in the list.
- **`box_plot_peaks(models, 'DML')` hardcoded at module level** (`peak_analysis.py:361`) with the other channels
  commented out. Loop over the channels Milan wants under `__main__`, like `paper_peak_tables.py` does.
- **The hardcoded `exit()`** at `peak_analysis.py:369`. It exists only to stop the table half of that notebook from
  running. The new script has no table half.
- **`sample`/`dummy_df` debug args** in the loader, plus the module-level smoke-test call at `peak_analysis.py:112`
  that runs on import. Drop both; a `sample` fraction is genuinely useful while iterating, so keep it only if it is
  an explicit opt-in argument.

Do not touch `peak_analysis.py` itself. It is a frozen thesis artifact and `docs/notebooks.md` documents it as such.
Write the new file alongside it.

## 6. Data schema

`pd.read_hdf(cache, key="peaks/{condition}")` gives a long-format frame, one row per peak (or per window for
`count`):

| Column | Values |
|---|---|
| `condition` | the condition, matching the key |
| `channel_name` | `FIR_LIDs_core`, `PD`, `DML`, `POHM`, `Z_axis`, `PD large peaks`, **`DML ELM peaks`** (7, was 6) |
| `measure` | `count`, `height`, `prominence`, `base`, `width`; **`DML ELM peaks`** adds `energy_delta`, `pd_prominence`, `energy_ratio` |
| `distribution` | `Generated` (the model) or `Real` (the target) |
| `value` | the property of one peak, or the peak count for that window |

Written by `PeakMetric.extract_df_all` ([src/metrics/peak_metric.py](../src/metrics/peak_metric.py)). Values are
denormalized already. The channel order is `data.cols.x` then `SYNTHETIC_CHANNELS`; do not hardcode it, read
`PeakMetric.CHANNEL_NAMES` or the frame itself.

These frames are large and about to get larger. `D_only_Wh` alone was ~1.06M rows per model *before* raw DML was
recorded; raw DML adds ~58 peaks per window where the gated channel contributed ~7, so expect a substantial jump.
The L and any conditions are roughly 30x D_only_Wh. Load one channel at a time and filter early, as the existing
generator does, and do not assume a whole condition fits comfortably in memory.

## 7. Conventions

- **No em dashes anywhere.** Non-negotiable, see `.claude/CLAUDE.md`. Strip them from anything you copy.
- Module docstring in the house format: `Scientific output / Inputs / Outputs / Usage / Limits or Caveat / History`.
- Comments explain constraints and quirks, not what the next line does.
- `PYTHONPATH=. python eval_notebooks/...` from the repo root; the shell is already inside the pipenv venv.
- Add a row to the table in [docs/notebooks.md](notebooks.md) when done.
- Mode-label conventions are a known trap in this repo (two coexisting conventions). The peak frames are already
  reduced to a `condition` string, so this does not bite here, but do not assume that elsewhere.

## 8. Definition of done

- `PYTHONPATH=. python eval_notebooks/paper_peak_boxplots.py` runs clean from the repo root and writes
  `boxplots_{channel}.pdf` at the agreed sizes.
- Changing `MODELS` alone changes the figure, with no other edits.
- The PD output is recognizably the reference figure, at a size chosen for ~5 rows rather than the thesis's ~16.
- **Fails loudly, never silently.** If a cache is missing a `/peaks/*` group or a requested channel, the script
  must say which cache and which key, not skip the row the way `iter_peak_properties_per_model` currently does
  with `print(f"Skipping model {model}...")`. A quietly missing model row is the failure mode that would let an
  obsolete or half-written cache reach a paper figure, which is exactly what happened before.
- The D-column thinness from section 4 is visible to a reader somewhere: an annotation, an `n=` label, or at minimum
  a documented note. A reader should not read the D box as being as solid as the L box.
- The channel is named honestly in the output. A `DML` figure and a `DML ELM peaks` figure are different claims;
  the filename and the title should not let them be confused, especially against pre-2026-07 caches where `DML`
  meant the gated subset.

## 9. Suggested order of work

The re-run gates the final figure, not the code. A sensible sequence:

1. Read [run-grid.md](run-grid.md) and [evaluation-metrics.md](evaluation-metrics.md), then agree with Milan on
   the model set and on which DML channel(s) the figure shows.
2. Write the script against the two old caches that do have all five conditions (`T2CUnFlow_tripleLeak`,
   `T1BiTransformer_tripleLeak`) purely to exercise the layout and export. Treat any figure from these as
   throwaway: wrong dataset, wrong DML semantics, obsolete names.
3. When the new caches land, point `MODELS` at the new names and regenerate. Expect raw `DML` to look completely
   different from anything you saw in step 2.
