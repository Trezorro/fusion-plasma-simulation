# Run naming and the experiment grid

The naming scheme and final grid for the paper are documented here.
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
| `v{revision}` | absent = v1. `v2`, `v3`: the cell was **retrained** |

`grep '^C'` gets every CFM run, `grep -- '-triple'` gets a full-leak run, and the ID sorts into backbone
groups.

### Redoing a run

Which marker to use depends on one question: **did the trained weights change?**

| What happened | Marker | Example |
|---|---|---|
| Bugfix in training, model, or data. New weights. | bump the ID revision | `CN` becomes `CNv2` |
| Same checkpoint, eval re-run only (an eval-side crash, a new metric) | suffix at the very end | `CN-cfm-noleak-normal-s05-e2` |
| Same cell, deliberately different hyperparameter | variant letter | `CNb` |
| A variant that then needs a retrain | both, letter before `v` | `CNbv2` |

A re-eval produces identical science from identical weights, so it must not look like a new model.

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

## The attention and prior axis (CFM and U-Net)

Two more ways to vary the `ConditionalUNet` backbone that both CFM and the deterministic U-Net cells use.

| Axis | Values | Config |
|---|---|---|
| Attention | on / off (`-noatt`) | `model.params.model_params.mid_attn`. `is_attn` stays all-`false` either way (attention was already bottleneck-only) |
| Prior (CFM only) | normal (default, unmarked) / brownian (`-brownian`) | `model.params.prior` |
| Prior sigma (CFM only) | normal `0.5` (`-s05`) / brownian `0.7` (`-s07`) | `model.params.prior_sigma`. Brownian sigma is a *terminal* std, whose time-averaged std is `sigma/sqrt(2)`; `0.7` matches normal's constant `0.5` time-averaged spread |

**Final work uses no-attention (`mid_attn: false`) throughout, for both PlasmaFlow (CFM) and UnFlow
(the deterministic U-Net).** No-attention was found to be competitive-or-better, so it is the backbone
used in every main-text cell below. Attention-on and Brownian-prior cells remain available as ablations.

## Main-text grid (Table \ref{tab:grid})

All cells train on `data.file: 2026_07_13-TCV_shots_V2.parquet`. Cache file = `output/test_cache/<name>_rollout.{h5,json}`;
matching directory under `output/htmlplots/<name>/`.

| Model | ID | Covariates | Role | Cache (`<name>`) |
|---|---|---|---|---|
| PlasmaFlow (CFM) | `CNb` | noleak | headline model | `R-CNb-cfm-noleak-normal-s05-noatt-e2` |
| PlasmaFlow (CFM) | `CSb` | single-leak | timing oracle | `CSb-cfm-ipla-normal-s05-noatt` |
| UnFlow | `UNb` | noleak | flow-vs-no-flow isolation | `UNb-unet-noleak-noatt` |
| UnFlow | `USb` | single-leak | timing oracle | `USb-unet-ipla-noatt` |
| TiDE | `TN` | noleak | covariate-aware baseline | `R-TN-tide-noleak-e2` |
| TiDE | `TS` | single-leak | timing oracle, light | `R-TS-tide-ipla-e2` |
| iTransformer | `IN` | noleak | capacity check | `R-IN-itransformer-noleak-e2` |
| iTransformer | `IS` | single-leak | timing oracle, light | `R-IS-itransformer-ipla-e2` |
| DLinear | `D` | — | floor baseline | `D-dlinear-r2` |

- CFM vs `UNb`: identical `ConditionalUNet` backbone, the only difference is flow matching vs a
  deterministic point prediction. Cleanest isolation of the flow-matching contribution.
- CFM vs `IN`/`TN`/`D`: model capacity and architecture family are not the bottleneck.
- `CNb` vs `CSb`, `UNb` vs `USb`, `IN` vs `IS`, `TN` vs `TS`: does a single leaky covariate
  (realized current, `IPLA`) already give the model what it needs to time ELMs. This is the
  paper's central timing-vs-capacity argument, run across every backbone.

## Appendix ablations (Appendix~\ref{app:ablations})

Attention-on and Brownian-prior variants of the `noleak`/`single` CFM and U-Net cells above.

| Model | ID | Covariates | What it isolates | Cache (`<name>`) |
|---|---|---|---|---|
| PlasmaFlow (CFM) | `CN` | noleak | attention on (vs `CNb`) | `R-CN-cfm-noleak-normal-s05-e2` |
| PlasmaFlow (CFM) | `CNc` | noleak | Brownian prior, attention on | `R-CNc-cfm-noleak-brownian-s07-e2` |
| PlasmaFlow (CFM) | `CNd` | noleak | Brownian prior, attention off | `R-CNd-cfm-noleak-brownian-s07-noatt-e2` |
| UnFlow | `UN` | noleak | attention on (vs `UNb`) | `UN-unet-noleak` |
| UnFlow | `US` | single-leak | attention on (vs `USb`) | `US-unet-ipla` |

## High leakage dose cells (triple leak, not in either table above)

Attention-on only; kept for a possible supplementary dose-response figure (`N` -> `S` -> `F`), not
part of the main text or the ablations appendix.

| Model | ID | Covariates | Cache (`<name>`) |
|---|---|---|---|
| PlasmaFlow (CFM) | `CF` | triple-leak | `R-CF-cfm-triple-normal-s05-e2` |
| UnFlow | `UF` | triple-leak | `UFv2-unet-triple` |

## Do not use

| ID | Cache (`<name>`) | Why |
|---|---|---|
| `UF-unet-triple` reeval | `R-UF-unet-triple-e2` | Corrupted: a Snellius submission race silently trained an ITransformer/noleak model under this label. Superseded by `UFv2-unet-triple` above |

Everything else under `output/test_cache/`/`output/htmlplots/` (pre-e2 superseded attempts, pre-grid
runs, smoke/debug caches) is archived under `output/_archive/` and not part of the paper; see its
`README.md` if you need the reason a specific one was set aside.

## Deliberately excluded

- The `triple` cell for iTransformer and TiDE: replaced by `single`; keeping both would add two more
  trainings for baselines that are not the paper's focus.
- The full attention x prior x leak factorial for CFM: kept as separate single-axis ablations instead.
- `PatchTST`: config exists at `configs/patchtst.yaml`, never run.
- Positional-encoding axis (`yesPos`/`noPos`): removed for the paper to prevent shortcut learning, not
  an axis any more. See `.claude/CLAUDE.md`.
