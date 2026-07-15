# PlasmaFlow: Repo Overview
Conditional Flow Matching model for data-driven simulation of plasma confinement diagnostics (TCV tokamak).

## What this is
 Codebase for Master thesis by Milan Tresoor, now positioned for a KI (Künstliche Intelligenz) journal paper, Technical Contribution format. Active paper branch: `paper`.
**PlasmaFlow** learns a conditional flow matching model over plasma diagnostic time-series windows. Given a history window of signals plus control covariates, it generates a distribution over plausible future diagnostic windows. The model is trained on real shot data from TCV and evaluated against ground-truth using mode transition metrics, peak metrics, and SoftDTW.

Physics is context, not the contribution. CFM+UNet is an established backbone; novelty is the diagnosis and evaluation. Central claim: ELM-caused stochastic transient event *timing* is the bottleneck, not spectral bias, not model capacity.

## READ FIRST: docs index
`docs/README.md` = canonical index. Execution diagram, output-artifact table, config-knob table, repo tree, architectural history. Individual docs:

| Doc | Covers |
|---|---|
| [docs/run-lifecycle.md](../docs/run-lifecycle.md) | Step-by-step tree of everything a run does |
| [docs/configuration.md](../docs/configuration.md) | Every `plasmaflow.yaml` key explained |
| [docs/data-pipeline.md](../docs/data-pipeline.md) | Raw TCV shots → training batches |
| [docs/evaluation-metrics.md](../docs/evaluation-metrics.md) | Every metric: definition, wandb key, timing |
| [docs/baselines.md](../docs/baselines.md) | Deterministic baselines (DLinear, PatchTST, iTransformer) |
| [docs/plots.md](../docs/plots.md) | Plot types + config |
| [docs/outputs.md](../docs/outputs.md) | Every output artifact: path, trigger, control |
| [docs/hpc-snellius.md](../docs/hpc-snellius.md) | Snellius: submit, sync, debug, venv |
| [docs/notebooks.md](../docs/notebooks.md) | Which notebook makes which thesis figure/table |

## Task
- Input: history window `x_WH` (observables) + control covariates `c_WH`, `c_WF`. Output: future window `x_WF`.
- Generate by integrating learned velocity field from prior (τ=0) to data (τ=1). Flow time τ∈[0,1] ≠ physical time t.
- Interpolation: `x_τ=(1-τ)x0+τx1`, target velocity `v=x1-x0`. Forward Euler only (see quirks).
- Windows: seq_length(future)=256, history_length=256 @ 10kHz. Sample rate hardcoded 10000Hz in FlowModule.
- Modes: L, D(dithering), H → `LHD_label` ints 0/1/2. Raw source is 1-indexed, shifted -1 in PeakMetric (off-by-one risk).
- ELMs (edge-localized modes) = key stochastic transient of interest.

## Key files
| File | Role |
|---|---|
| `run.py` | Entry point: training + eval, driven by OmegaConf + wandb |
| `src/models/flow.py` | `FlowModule` (Lightning): all flow matching, train/val/test steps |
| `src/models/unet_conditional.py` | `ConditionalUNet` architecture |
| `src/data_loaders.py` | `FusionShotDataset` + `FusionShotDataModule` |
| `src/evaluation.py` | `PlotsCallback` + `evaluate_window_set` |
| `src/config.py` | Config load/consolidate; `get_current_config(wandb_only=True)` = canonical config after init |
| `src/metrics/` | `evaluate_modes.py` (FNOLSTM surrogate labels), `mode_metrics.py`, `peak_metric.py`, `metrics.py` (moments/entropy/SoftDTW) |
| `src/plotters/` | All eval plotting |
| `configs/plasmaflow.yaml` | Main paper config |

## Architecture (`unet_conditional.py`)
- 1D conv U-Net, hierarchical encoder-decoder, skip connections. Not a novelty claim.
- Conditioning concatenated along channel dim: history+future windows, control covariates, binary history/future indicator channel, positional encoding.
- Positional encoding = raw physical time in seconds, unnormalized. Hardcoded `max_value=2.0` (not a config key). Shots >2s alias; check if data changes. Removed for paper results to prevent shortcut learning.
- Knobs: `ch_mults`, `is_attn` per level, `mid_attn`, `attn_heads`, `n_blocks`, `norm_groups`. Defaults → 4-level UNet, attention at bottleneck.

## Training (`flow.py`, `FlowModule`, Lightning)
- **Manual optimization loop** (not Lightning automatic). Grad clip done manually inside loop.
- Batch rematching: each batch makes N training pairs (default `batch_rematch_factor=5`) by resampling prior + τ. Optimizer steps every `step_every_nth_match` (must evenly divide rematch factor).
- Priors: normal, brownian, levy, resample, copy, constant. Normal σ default 0.3.
- **OT pairing only with normal prior.** If OT active and conditioning reordering skipped, conditioning gets mismatched to targets. Mostly not applied, did not help.
- Loss: MSE or L1 on predicted vs target velocity.

## Data pipeline (`data_loaders.py`, `run_processing.py`)
- Stages: raw shot parquets → combined parquet (local only, `run_processing.py`) → windowed samples (`FusionShotDataset`) → `FusionShotDataModule`.
- Source: Poels/Venturini et al. TCV confinement-state dataset (Nuclear Fusion 2025), git submodule `giants/TCV_confstate_data/`.
- Current parquet: `data/2026_06_29-TCV_shots_V2.parquet` (~206MB).
- **`PD` column is load-bearing:** FNOLSTM classifier finds it via `cols.x.index("PD")`. Rename/drop → silent break or crash.
- Column renames (old→new): `FIR_core`→`FIR_LIDs_core`, `Halpha1`→`PD`(kept as PD), `NBI`→`PNBI`+`PNBI2`(2 beams), `ECRH`→`PECRH`, `a_minor`→`MINRAD`, `DELTA`→`DELTA_TOP`+`DELTA_BOTTOM`. No equivalent for legacy `gas_fringes`. `Halpha_fft` status uncertain.
- Norm: min-max→[0,1], computed from **TRAIN split only**, applied globally (not per-shot). Position sequence + label NOT normalized.
- Splits = hardcoded shot lists in config (not random).
- Window valid if start within `crop_margin` of both shot boundaries. Min shot length = seq_length + 2·crop_margin.
- Test set: stride-10 windows (perf). Train/val: stride 1.

## Evaluation metrics (`src/metrics/`)
No single scalar. Groups answer different questions:
- Moment errors: right mean/variance per timestep.
- Entropy (antropy lib): right regularity/complexity.
- **PeakMetric**: right peak count/shape/energy, split by history-mode (L_only_Wh, D_only_Wh, H_only_Wh). ELM-relevant.
- ModeTransitionMetric: right transition rates/probabilities. Dice score: right mode-sequence timing.
- SoftDTW: shape similarity robust to small shifts (secondary).
- Future-window mode labels are **surrogate**: FNOLSTM classifier run on both generated + target futures (true labels only exist for history).
- All eval denormalizes with train-derived min/max first.

## Known quirks (DO NOT "fix" without checking git history)
- `solve_method` / integration config key is **inert**. Forward Euler always. Adaptive solver path is dead/commented.
- Spectral entropy uses `sf=100`, not real 10kHz. Legacy; fine for relative comparisons.
- SoftDTW needs numba≥0.61 on cluster; Pipfile pins older numba for local `ydata-profiling`. Per-env mismatch is intentional.
- `ydata_profiling` import in `run_processing.py` is local-only dev dep; that stage never runs on cluster.

## HPC (Snellius)
- Submit: `bash src/HPC_setup/submit_remote_job_snellius.sh <run_name>` (tags+pushes, sbatch via SSH, polls squeue, rsyncs logs). SBATCH script: `run_snellius_job.sh`. Orchestration also via `.vscode/tasks.json`.
- Venv `~/fusion/` on cluster, NOT from Pipfile: manual pip per commands at bottom of `Pipfile`.
- Modules: `Python/3.11.3-GCCcore-12.3.0`, `PyTorch/2.1.2-foss-2023a-CUDA-12.1.1`.
- Upload data: `rsync -vz ./data/*.parquet snellius:fusion-plasma-simulation/data/`. Eval caches: `/scratch-shared/mtresoor/final_cache/`. Slurm logs: `output/snellius/slurms/`.
- Details: [docs/hpc-snellius.md](../docs/hpc-snellius.md).

## Wandb
Project `plasmaflow`, entity `deep-learning-course-team`. Config synced through wandb.

## Writing conventions (docs + paper)
- Docs: for new readers/reviewers to grasp code fast. No fluff, no selling/defending. Do explain specific decisions, quirks, workarounds.
- Paper prose: no bullet points inside actual paper/discussion text (bullets fine in notes/plans/this file). Hedge interpretive claims ("we hypothesize..."). No overclaiming beyond tested domain. Concession-then-differentiation when comparing to prior work.
- LaTeX: `\gls{}` for domain terms, `\autoref{}` for figures, italics for model/variant names, `\fillin{}` placeholder for unsupported claims.
- Co-author comms: short, scannable, structured (Teams-style).

# Non-negotiable rule for CLAUDE or other LLMS:
### No thought dashes (EM DASHES ARE BANNED)
NEVER write the "—" character (U+2014, EM dash) anywhere: not in new text, not when copying or paraphrasing existing content, not in headings, not in inline annotations. No exceptions, ever.

When you feel the urge to write " — ", stop and rewrite the sentence. Use instead:
- a colon (":") to introduce a clarification
- a comma or parentheses for an aside
- a semicolon for two related clauses

**This applies when copying content from existing files.** If source material contains em dashes, remove them during the copy. Do NOT carry them over.

Clean up any em dash you encounter in existing docs, even if you didn't write it. The only permitted occurrence of the "—" character in this entire codebase is in the example in this rule.
