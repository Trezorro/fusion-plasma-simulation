# Run naming and the experiment grid

The naming scheme for runs (and therefore for wandb runs and `output/test_cache/{name}.h5`, since
`test_cache_name: ${run_name}`), plus the current grid to train.

## Naming scheme

    {ID}-{backbone}-{cov}[-{prior}][-{note}]

The ID is a short distinguisher and comes first; everything after it is human-readable meaning and can
be reworded without breaking the handle.

### The ID is the grid cell

    {backbone letter}{covariate digit}[{variant letter}][v{revision}]

| Part | Values |
|---|---|
| backbone letter | `C` CFM, `U` U-Net deterministic, `I` iTransformer, `T` TiDE, `D` DLinear, `P` PatchTST |
| covariate digit | `0` noleak, `1` ipla, `3` triple. Omitted only for covariate-blind backbones |
| variant letter | `b`, `c`, ... a second run in the same cell (a prior ablation, a different sigma) |
| `v{revision}` | absent = v1. `v2`, `v3`: the cell was **retrained**, see below |

`grep '^C'` gets every CFM run, `grep -- '-triple'` gets the oracle column, and the ID sorts into
backbone groups.

### Redoing a run

Which marker you use depends on one question: **did the trained weights change?**

| What happened | Marker | Example |
|---|---|---|
| Bugfix in training, model, or data. New weights. | bump the ID revision | `C0` becomes `C0v2` |
| Same checkpoint, eval re-run only (an eval-side crash, a new metric) | suffix at the very end | `C0-cfm-noleak-normal-s03-e2` |
| Same cell, deliberately different hyperparameter | variant letter | `C0b` |
| A variant that then needs a retrain | both, letter before `v` | `C0bv2` |

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

| `cov` | `cols.c` | Carries ELM timing? |
|---|---|---|
| `noleak` | `IP`, `PNBI`, `PECRH` | No. `IP` is the programmed current *reference* |
| `ipla` | `IPLA`, `PNBI`, `PECRH` | Yes. `IPLA` is the realized current and pulses at ELMs |
| `triple` | `IPLA`, `PNBI`, `PECRH`, `KAPPA`, `VOL` | Yes, three informative signals |

`c_channels` is auto-synced from `len(cols.c)` by `update_model_input_channels` (`config.py:124`), so the
`c_channels: 0` sitting in `configs/unflow.yaml` is dead text and is overwritten. Do not trust it; read
`cols.c`.

**DLinear and PatchTST are covariate-blind by construction** and so have no covariate digit. `dlinear.py:4`:
"floor baseline: covariates cannot inform the X channels, so it is fed X-history only." Giving them a
`triple` cell is meaningless.

## The grid

All cells train on the **current** dataset (`data.file`, `2026_07_13-TCV_shots_V2.parquet`). This is the
point of the re-run: the previous caches straddle two datasets, and normalization is derived from the
train split, so cross-dataset comparison is confounded. Nothing here can be salvaged by re-eval alone.

| ID | Run name | Backbone | `cov` | Why it exists |
|---|---|---|---|---|
| `C0` | `C0-cfm-noleak-normal-s03` | CFM, normal prior sigma 0.3 | noleak | **Headline model** |
| `C3` | `C3-cfm-triple-normal-s03` | CFM, normal prior sigma 0.3 | triple | **New cell.** Does CFM also improve when handed ELM timing? |
| `C0b` | `C0b-cfm-noleak-brownian-s08-long` | CFM, brownian prior | noleak | Prior ablation. Trained extra long, lower sigma (confirm the value) |
| `U0` | `U0-unet-noleak` | U-Net deterministic | noleak | Same architecture as CFM, no flow. Isolates the flow itself |
| `U3` | `U3-unet-triple` | U-Net deterministic | triple | The same ablation, handed the oracle |
| `I0` | `I0-itransformer-noleak` | iTransformer | noleak | Strong deterministic baseline: capacity is not the issue |
| `I3` | `I3-itransformer-triple` | iTransformer | triple | Oracle column |
| `T0` | `T0-tide-noleak` | TiDE | noleak | Covariate-aware baseline |
| `T3` | `T3-tide-triple` | TiDE | triple | Oracle column |
| `D0` | `D0-dlinear` | DLinear | blind | Floor baseline. No covariate digit |

Ten trainings. `C0b` is the extra-long, lower-variance brownian run.

### Open item on C0b

"Slightly less variance" than the sigma 1.0 brownian used so far most likely means `prior_sigma: 0.8`,
which is the other brownian that was run (`BrownianMidAttSig08`). Confirm before launching, and put the
real number in the name: `-s08` is a placeholder. Note `plasmaflow.yaml` documents that brownian sigma
is a *terminal* std whose time-averaged std is `sigma/sqrt(2)`, so brownian sigma is not directly
comparable to normal sigma.

### What the grid buys, per claim

- CFM vs `U0`: identical `ConditionalUNet` backbone, the only difference is flow matching vs a
  deterministic point prediction. This is the cleanest possible isolation of the contribution.
- CFM vs `I0`/`T0`/`D0`: model capacity and architecture family are not the bottleneck.
- `*0` vs `*3`: hand a model ELM timing and watch it improve. This is the load-bearing evidence that
  timing, not capacity or spectral bias, is what is missing.
- `C0` vs `C3`: the same test applied to our own model, which closes the obvious reviewer question of
  whether CFM uses covariates at all.

### Deliberately excluded

- `ipla` (single-leak) column, for the 4 covariate-aware backbones: would turn the oracle contrast into
  a dose-response (none, IPLA only, triple). 4 more trainings. Worth adding if the `noleak` to `triple`
  jump needs to be shown as a trend rather than a switch.
- `PatchTST` (`P0`): config exists at `configs/patchtst.yaml`, never run.
- Positional-encoding axis (`yesPos`/`noPos`): settled. Removed for the paper to prevent shortcut
  learning, so it is not an axis any more. See `.claude/CLAUDE.md`.
