# PlasmaFlow — Repo Overview

Flow Matching model for data-driven simulation of plasma confinement diagnostics (TCV tokamak). Master thesis project by Milan Tresoor. Active branch for paper results: `paper`.

## What this is
**PlasmaFlow** learns a conditional flow matching model over plasma diagnostic time-series windows. Given a history of signals, it generates plausible future trajectories. The model is trained on real shot data from TCV and evaluated against ground-truth using mode transition metrics, peak metrics, and SoftDTW.

## Key files
| File | Role |
|---|---|
| `run.py` | Entry point — training + evaluation, driven by wandb + OmegaConf config |
| `src/models/flow.py` | Lightning module: all flow matching logic, training/val/test steps |
| `src/models/unet_conditional.py` | Main architecture (conditional UNet) |
| `src/evaluation.py` | Evaluation callbacks called during training |
| `src/data_loaders.py` | `ShotWindowDataset` + Lightning datamodule |
| `src/config.py` | Config loading/consolidation (OmegaConf + wandb) |
| `src/metrics/` | Metrics: mode transitions (`evaluate_modes.py`), SoftDTW (`metrics.py`), peaks |
| `src/plotters/` | All plotting code for evaluation figures |
| `configs/plasmaflow.yaml` | Main config for paper runs |

## HPC (Snellius)
- Jobs submitted via `run_snellius_job.sh` (SBATCH), orchestrated locally via `src/HPC_setup/submit_remote_job_snellius.sh` and `.vscode/tasks.json`
- Venv at `~/fusion/` on cluster, **not** installed from Pipfile — set up manually via pip commands documented at the bottom of `Pipfile`
- Modules loaded: `Python/3.11.3-GCCcore-12.3.0`, `PyTorch/2.1.2-foss-2023a-CUDA-12.1.1`
- Data uploaded via rsync: `rsync -vz ./data/*.parquet snellius:fusion-plasma-simulation/data/`
- Slurm output logs land in `output/snellius/slurms/`

## Data
- Raw data: TCV plasma shots, stored as `.parquet` in `data/` after preprocessing with `src/run_processing.py`
- New dataset (TCV confstate) accompanying code: `giants/TCV_confstate_data/`
- Evaluation caches stored in `/scratch-shared/mtresoor/final_cache/` on cluster
- ...

## Wandb
- Project: `plasmaflow`, entity: `deep-learning-course-team`
- Config is synced through wandb; `get_current_config(wandb_only=True)` is the canonical config object after init


# Non-negotiable rule for CLAUDE or other LLMS:
### No thought dashes (EM DASHES ARE BANNED)
NEVER write the "—" character (U+2014, EM dash) anywhere: not in new text, not when copying or paraphrasing existing content, not in headings, not in inline annotations. No exceptions, ever.

When you feel the urge to write " — ", stop and rewrite the sentence. Use instead:
- a colon (":") to introduce a clarification
- a comma or parentheses for an aside
- a semicolon for two related clauses

**This applies when copying content from existing files.** If source material contains em dashes, remove them during the copy. Do NOT carry them over.

Clean up any em dash you encounter in existing docs, even if you didn't write it. The only permitted occurrence of the "—" character in this entire codebase is in the example in this rule.
