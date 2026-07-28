# Run naming and the experiment grid

The naming scheme and full grid for our experiments is documented here.
wandb runs and caches also follow these names `output/test_cache/{name}.h5`.

## Naming scheme

    {ID}-{backbone}-{cov}[-{prior}][-{note}]

The ID is a short distinguisher and comes first; everything after it is human-readable meaning and can
be reworded without breaking the handle.

### The ID is the grid cell

    {backbone letter}{leak letter}[{variant letter}][v{revision}]

| Part | Values |
|---|---|
| backbone letter | `C` CFM, `U` U-Net deterministic, `I` iTransformer, `T` TiDE, `D` DLinear, `P` PatchTST |
| leak letter | `N` noleak, `S` single leak (ipla), `F` full leak (triple). Omitted for covariate-blind backbones |
| variant letter | `b`, `c`, ... a second run in the same cell (a prior ablation, an attention ablation) |
| `v{revision}` | absent = v1. `v2`, `v3`: the cell was **retrained**, see below |

`grep '^C'` gets every CFM run, `grep -- '-triple'` gets a full-leak run, and the ID sorts into backbone
groups.

**Why a letter and not the old digit (`0`/`1`/`3`):** the digit read as a covariate *count*, which was
wrong even for the cell it was supposed to describe cleanly (`noleak` still has three covariates, `IP`,
`PNBI`, `PECRH`; the digit was actually counting *informative* covariates, not covariates). A letter that
means "leak level" instead of a digit that looks like "covariate count" removes the ambiguity outright,
since a letter carries no numeric-count connotation to misread.

### Redoing a run

Which marker you use depends on one question: **did the trained weights change?**

| What happened | Marker | Example |
|---|---|---|
| Bugfix in training, model, or data. New weights. | bump the ID revision | `CN` becomes `CNv2` |
| Same checkpoint, eval re-run only (an eval-side crash, a new metric) | suffix at the very end | `CN-cfm-noleak-normal-s05-e2` |
| Same cell, deliberately different hyperparameter | variant letter | `CNb` |
| A variant that then needs a retrain | both, letter before `v` | `CNbv2` |

The retrain/re-eval split is the one that matters. A re-eval produces identical science from identical
weights, so it must not look like a new model. This is not hypothetical: the whole Jul 1-8 cache
generation needed an eval re-run (`-e2`) and nothing more, because the models were fine and only the
peak-extraction hook crashed. See [handoff-peak-boxplots.md](handoff-peak-boxplots.md).

Useful side effect: a dataset change forces a retrain of every cell, so the whole grid bumps to `v2`
together and `v` doubles as a grid generation. Runs from different generations then cannot be silently
compared, which is exactly the failure the old names allowed. Do not put the dataset in the name;
`data.file` is already in the wandb config.

Avoid `.` in IDs: several notebooks parse names via `Path.stem` and string splits.

## The covariate axis (the oracle experiment)

Set by `data.cols.c` in `configs/plasmaflow.yaml`. This is the whole leakage/oracle experiment.

| leak letter | `cols.c` | Carries ELM timing? |
|---|---|---|
| `N` noleak | `IP`, `PNBI`, `PECRH` | No. `IP` is the programmed current *reference* |
| `S` ipla (single leak) | `IPLA`, `PNBI`, `PECRH` | Yes. `IPLA` is the realized current and pulses at ELMs |
| `F` triple (full leak) | `IPLA`, `PNBI`, `PECRH`, `KAPPA`, `VOL` | Yes, three informative signals (current plus two shape signals, to be unambiguous) |

`c_channels` is auto-synced from `len(cols.c)` by `update_model_input_channels` (`config.py:124`), so the
`c_channels: 0` sitting in `configs/unflow.yaml` is dead text and is overwritten. Do not trust it; read
`cols.c`.

**DLinear and PatchTST are covariate-blind by construction** and so have no leak letter. `dlinear.py:4`:
"floor baseline: covariates cannot inform the X channels, so it is fed X-history only." Giving them a
leak cell is meaningless.

**Leak coverage is not the same for every backbone.** CFM gets the two extremes only (`N`, `F`): the
question for our own model is a switch test, not a trend. U-Net gets all three points (`N`, `S`, `F`):
it is the cheapest backbone to run three ways (deterministic, no prior/attention axis of its own), so it
is the one that shows whether the covariate effect is a step or a dose-response. iTransformer and TiDE
get the lighter two-point check (`N`, `S`): they are supporting baselines, not the focus of the leak
narrative, so the coarser single-leak comparison is enough.

## The attention and prior axis (CFM only)

Two more ways to vary the same `ConditionalUNet` backbone that CFM already uses, both scoped to the
`noleak` cell so the attention/prior ablation and the leak ablation never get confounded together (a
full 2 prior x 2 attention x 2 leak factorial was considered and dropped as unnecessary cost for the
questions being asked here).

| Axis | Values | Config |
|---|---|---|
| Attention | on (default, unmarked) / off (`-noatt`) | `model.params.model_params.mid_attn`: `true` (default) / `false`. `is_attn` stays all-`false` either way (attention was already bottleneck-only) |
| Prior | normal (default, unmarked) / brownian (`-brownian`) | `model.params.prior` |
| Prior sigma | normal `0.5` (`-s05`) / brownian `0.7` (`-s07`) | `model.params.prior_sigma`. Brownian sigma is a *terminal* std, whose time-averaged std is `sigma/sqrt(2)`; `0.7` was picked so brownian's time-averaged spread (`0.7/sqrt(2) = 0.495`) matches normal's constant `0.5`, making the two priors comparable in spread rather than in the raw sigma number |

This replaces the old open item about the brownian sigma value (previously unresolved, "confirm before
launching"): `0.7` is now the number, chosen by the sqrt(2) matching argument above, not carried over from
whatever brownian run happened to exist before.

## The grid

All cells train on the **current** dataset (`data.file`, `2026_07_13-TCV_shots_V2.parquet`). This is the
point of the re-run: the previous caches straddle two datasets, and normalization is derived from the
train split, so cross-dataset comparison is confounded. Nothing here can be salvaged by re-eval alone.

| ID | Run name | Backbone | Leak | Why it exists |
|---|---|---|---|---|
| `CN` | `CN-cfm-noleak-normal-s05` | CFM, normal prior, attention on | noleak | **Headline model** |
| `CNb` | `CNb-cfm-noleak-normal-s05-noatt` | CFM, normal prior, attention off | noleak | Does the bottleneck self-attention matter for ELM-scale transient timing, or is the conv backbone alone enough? |
| `CNc` | `CNc-cfm-noleak-brownian-s07` | CFM, brownian prior, attention on | noleak | Prior ablation, attention on |
| `CNd` | `CNd-cfm-noleak-brownian-s07-noatt` | CFM, brownian prior, attention off | noleak | Prior ablation, attention off. Together with `CNc`, checks whether the prior and the attention axis interact |
| `CF` | `CF-cfm-triple-normal-s05` | CFM, normal prior, attention on | triple | **New cell.** Does CFM also improve when handed ELM timing? |
| `UN` | `UN-unet-noleak` | U-Net deterministic | noleak | Same architecture as CFM, no flow. Isolates the flow itself |
| `US` | `US-unet-ipla` | U-Net deterministic | single | **New cell.** Middle point of the dose-response: does one leaky signal move the needle, or does it take all three? |
| `UF` | `UF-unet-triple` | U-Net deterministic | triple | Top of the dose-response, and the same ablation `CF` runs on CFM |
| `IN` | `IN-itransformer-noleak` | iTransformer | noleak | Strong deterministic baseline: capacity is not the issue |
| `IS` | `IS-itransformer-ipla` | iTransformer | single | Oracle check, light version |
| `TN` | `TN-tide-noleak` | TiDE | noleak | Covariate-aware baseline |
| `TS` | `TS-tide-ipla` | TiDE | single | Oracle check, light version |
| `D` | `D-dlinear` | DLinear | blind | Floor baseline. No leak letter |

Thirteen trainings (up from the previous ten): CFM grew from 2 cells to 5 (the attention/prior ablation),
U-Net grew from 2 cells to 3 (the dose-response middle point), iTransformer and TiDE swapped their `triple`
cell for the lighter `single` cell (net zero change in count for those two).

### What the grid buys, per claim

- CFM vs `UN`: identical `ConditionalUNet` backbone, the only difference is flow matching vs a
  deterministic point prediction. This is the cleanest possible isolation of the contribution.
- CFM vs `IN`/`TN`/`D`: model capacity and architecture family are not the bottleneck.
- `CN` vs `CNb`: does attention specifically matter for CFM, independent of the leak question.
- `CN` vs `CNc`/`CNd`: prior choice (normal vs brownian), with and without attention.
- `UN` -> `US` -> `UF`: does a deterministic model's use of ELM timing scale with how much of it it is
  handed, or does it only respond once the signal is unambiguous? This is the load-bearing evidence that
  timing, not capacity or spectral bias, is what is missing, now shown as a trend rather than a switch.
- `IN`/`TN` vs `IS`/`TS`: the same switch test as before, at lower cost since these are supporting
  baselines rather than the focus of the leak narrative.
- `CN` vs `CF`: the same test applied to our own model, which closes the obvious reviewer question of
  whether CFM uses covariates at all.

### Deliberately excluded

- The `triple` cell for iTransformer and TiDE: replaced by `single` (see above); keeping both would add
  two more trainings for baselines that are not the paper's focus.
- The full attention x prior x leak factorial for CFM: would be 8 CFM cells instead of 5; the two axes
  are kept as separate single-axis ablations off the `noleak` headline instead (see "The attention and
  prior axis" above).
- `PatchTST`: config exists at `configs/patchtst.yaml`, never run. Would need its own leak letter like the
  other covariate-aware baselines if it gets added.
- Positional-encoding axis (`yesPos`/`noPos`): settled. Removed for the paper to prevent shortcut
  learning, so it is not an axis any more. See `.claude/CLAUDE.md`.
