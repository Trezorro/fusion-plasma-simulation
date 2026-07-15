# PlasmaFlow

PlasmaFlow learns a conditional flow matching model over plasma diagnostic time-series windows from the TCV tokamak. Given a window of observed signals (the history Wh), it generates a distribution over plausible future trajectories (the future Wf). The model is a conditional UNet trained with flow matching (straight-line probability paths from a prior to the data distribution). It is evaluated on whether it captures mode transitions (L/D/H confinement modes), peak statistics, and signal entropy. The current training data uses the public TCV confinement-state dataset by Poels et al. (Nuclear Fusion, 2025).

## Quick start

```bash
python run.py run_name=my_experiment
# or to submit to Snellius:
bash src/HPC_setup/submit_remote_job_snellius.sh my_experiment
```

## Execution overview

```
┌──────────────────────────────────────────────────────────────────┐
│ User: .vscode/tasks.json: Run Snellius Experiment                │
└─────────────────────────────┬────────────────────────────────────┘
                              │
          ┌───────────────────┴────────────────────────┐
          │ ./submit_remote_job_snellius.sh <run_name> │
          │   ──────────────────────────────────────   │
          │ 1. Assert no unpushed local commits        │
          │ 2. git tag <run_name>; git push            │
          │ 3. SSH + run sbatch on snellius cluster    │
          │ └──> Job <run_name> is put in slurm queue  │
          │ 4. Poll `squeue` + rsync slurm logs.       │
          │                            (continuous)    │
          └───────────────────┬────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│ Snellius cluster (SLURM, gpu_mig, 1 GPU, 9 CPUs)                    │
│ ┌───────────────────────────────────────────────────────────────┐   │
│ │ run_snellius_job.sh (SBATCH)                                  │   │
│ │  - Load modules: Python/3.11.3, PyTorch/2.1.2-CUDA-12.1.1     │   │
│ │  - source ~/fusion/bin/activate                               │   │
│ │  - git checkout tags/<run_name>                               │   │
│ │  - export WANDB_DIR, TEST_CACHE_DIR, WANDB_ARTIFACT_DIR       │   │
│ │  - srun python run.py run_name=<name> [reeval=True]           │   │
│ └───────────────────────────┬───────────────────────────────────┘   │
│                             │                                       │
│ ┌───────────────────────────▼───────────────────────────────────┐   │
│ │ run.py                                                        │   │
│ │  - Load config (plasmaflow.yaml + CLI overrides + wandb)      │   │
│ │  - FusionShotDataModule  (parquet -> normalized windows)      │   │
│ │  - FlowModule            (ConditionalUNet + flow matching)    │   │
│ │  - trainer.fit()         (120 epochs, rematch_factor=5)       │   │
│ │  - trainer.validate()                                         │   │
│ │  evaluate_window_set() ────────────────────────► output/pdfplots/ │
│ │  trainer.test()        ──────────────► HDF5 cache (if configured) │
│ │  - prune_online_checkpoints()                                 │   │
│ └───────────────────────────┬───────────────────────────────────┘   │
│                             │                                       │
│ ┌───────────────────────────▼───────────────────────────────────┐   │
│ │ Outputs on cluster:                                           │   │
│ │  output/models/<run_name>/*.ckpt                              │   │
│ │  output/pdfplots/<run_name>/qualitative_samples/              │   │
│ │  /scratch-shared/mtresoor/final_cache/<name>.h5               │   │
│ │  output/snellius/slurms/gpujob-<id>-<name>.out                │   │
│ └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │  rsync (auto: slurm logs)
                              │  rsync (manual: pdfplots, cache)
                              │  wandb cloud (auto: all metrics + ckpts)
┌─────────────────────────────▼───────────────────────────────────────┐
│ Local results                                                       │
│  output/snellius/slurms/    <- slurm logs (auto-synced each poll)   │
│  output/pdfplots/           <- PDF plots (sync via VS Code task)    │
│  output/test_cache/         <- HDF5 cache (manual pull)             │
│  wandb.ai/.../plasmaflow    <- all metrics, checkpoints, figures    │
└─────────────────────────────────────────────────────────────────────┘
```

## Docs index

| File | What it covers |
|---|---|
| [run-lifecycle.md](run-lifecycle.md) | Full step-by-step tree of operations during any run |
| [outputs.md](outputs.md) | Every output artifact: path, trigger, config control |
| [configuration.md](configuration.md) | All config keys in plasmaflow.yaml explained |
| [data-pipeline.md](data-pipeline.md) | From raw TCV shots to training batches |
| [evaluation-metrics.md](evaluation-metrics.md) | Every metric computed: definition, wandb key, timing |
| [baselines.md](baselines.md) | Deterministic baselines (DLinear, PatchTST, iTransformer, TiDE): approach, integrity, how to run |
| [plots.md](plots.md) | All plot types: what they show, how to configure |
| [hpc-snellius.md](hpc-snellius.md) | Snellius HPC: submit, sync, debug, venv setup |
| [notebooks.md](notebooks.md) | Analysis notebooks: which one makes which thesis figure/table, inputs/outputs, run order |

## Key output artifacts

| Artifact | Path | Trigger | Config control |
|---|---|---|---|
| Model checkpoints | `output/models/{date}-{run_name}/*.ckpt` | Every epoch (best + last kept) | `patience`, `epochs` |
| PDF qualitative plots | `output/pdfplots/{run_name}/qualitative_samples/` | After every validate() | `window_set` entries |
| HDF5 test cache | `$TEST_CACHE_DIR/{test_cache_name}.h5` | trainer.test() if cache enabled | `test_cache_name` in config |
| Wandb metrics | wandb cloud | During training/val/test | `plot_functions`, `val_every_n_epochs` |
| Slurm logs | `output/snellius/slurms/gpujob-*.out` | Cluster run (auto-rsynced) | n/a |

## Key config knobs

| Knob | Key | Default | Effect |
|---|---|---|---|
| Training epochs | `epochs` | 120 | Hard cap on training |
| Early stopping | `patience` | 20 | Epochs without val loss improvement |
| Prior distribution | `model.params.prior` | `normal` | `normal`, `brownian`, `levy`, `resample`, `copy`, `constant` |
| Optimal transport | `model.params.ot_method` | null | null disables OT; requires prior=normal |
| Batch rematching | `model.params.batch_rematch_factor` | 5 | Gradient accumulation over multiple pairings per batch |
| Eval frequency | `evaluation.val_every_n_epochs` | 5 | Full evaluation fires at epoch % 5 == 1 |
| Test cache | `test_cache_name` | (absent) | Set to enable HDF5 caching of test outputs |
| Window-set plots | `window_set` | 18 shot windows | List of [shot, time] pairs for PDF plot generation |

## Repo structure

```
experiments/
  run.py                          # entry point
  run_snellius_job.sh             # SBATCH script (runs on cluster)
  configs/
    plasmaflow.yaml               # main config for paper runs
    reeval.yaml                   # re-evaluation config
    MHD_model_yoerie/             # FNOLSTM mode classifier weights
    models/                       # model architecture yamls (legacy)
  src/
    models/
      flow.py                     # FlowModule (Lightning): training + inference
      unet_conditional.py         # ConditionalUNet architecture
    metrics/
      evaluate_modes.py           # surrogate mode label generation (FNOLSTM)
      mode_metrics.py             # ModeTransitionMetric
      peak_metric.py              # PeakMetric (ELM/peak detection)
      metrics.py                  # moment metrics, entropy, SoftDTW
    plotters/                     # all plotting code
    data_loaders.py               # FusionShotDataModule + FusionShotDataset
    evaluation.py                 # PlotsCallback + evaluate_window_set
    config.py                     # OmegaConf + wandb config loading
    hdf_cache.py                  # HDF5 test result cache
    to_pdf.py                     # dump figures to PDF at multiple sizes
    run_processing.py             # preprocessing: raw shots -> parquet
    HPC_setup/                    # cluster submission scripts
  data/
    2026_06_29-TCV_shots_V2.parquet  # current training data (~206MB)
    public_data_set/              # Poels et al. individual shot parquets + metadata
  giants/
    TCV_confstate_data/           # Poels et al. git submodule (full dataset)
    LDH_demo/                     # reference LDH model from Poels (model weights)
  output/
    models/                       # local model checkpoints
    pdfplots/                     # PDF evaluation plots
    test_cache/                   # HDF5 test caches (local)
    snellius/slurms/              # rsynced cluster logs
  eval_notebooks/                 # analysis notebooks
  evaluate.ipynb                  # main evaluation notebook
```

## Architectural history highlights

Run-specific commits are excluded; only structural changes are listed.

1. **2024-03-19** (`0f0d838`): Initial ML scaffold with data loading, model, training, and evaluation modules.
2. **2024-07-03** (`34710d3`): LSTM/EncoderDecoder paradigm and OmegaConf config system (this era is preserved on the `pre-lightning` branch).
3. **2024-09-04** (`444d321`): PyTorch Lightning refactor; code moved into `src/` with LightningModules and a Trainer.
4. **2024-09-13** (`a0f4792`): First UNet backbone; `src/models/unet.py` and `configs/models/UNet.yaml` introduced.
5. **2025-01-08** (`bd74dec`): Core pivot to flow matching; `src/models/flow.py` and vector-field networks introduced. The project becomes generative.
6. **2025-01-21** (`dee7c9f`): Time-conditioned UNet (`src/models/unet_conditional.py`) replaces earlier backbone; multi-channel conditioning added.
7. **2025-02-20** (`397cd70`): Optimal transport pairing added (`src/optimal_transport.py`); enables minibatch CFM with OT coupling.
8. **2025-03-10** (`352387e`): Conditioning channels (x_history, c) wired into data loaders and UNet.
9. **2025-04-08** (`5ceccd4`): Peak metrics (ELM/DML Wasserstein distances) introduced in `src/metrics/`.
10. **2025-05-08 to 05-16** (`f215d01`, `da81702`, `0a7ba07`): Lightning DataModule, FNOLSTM surrogate mode labels, and `src/metrics/` submodule. Mode transition metrics ("BOOYAH mode metrics") land.
11. **2025-06-02** (`de5aa03`): HDF5 test cache (`src/hdf_cache.py`) and reeval config overhaul.
12. **2026-05-27** (`ba6f54c`): Yoeri Poels' TCV confinement-state dataset imported as `giants/TCV_confstate_data/` git submodule.
13. **2026-06-30** (`5f3726b`): Switch to the public Poels et al. dataset and a fresh wandb project (`plasmaflow`) for clean paper results. The `paper` branch marks this transition; it will merge back into `main`.

## Branch notes

- `paper`: current active branch; marks the pivot to the public dataset and clean paper experiments. Will merge into `main`.
- `main`: primary development branch; contains the full history above.
- `pre-lightning`: frozen at the 2024-09-02 divergence point; preserves the RNN/LSTM forecasting era for reference.
