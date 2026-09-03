# PlasmaFlow
A *Probabilistic Learning Architecture for Simulating Mode-dependent, Aperiodic Fluctuations in Long Output Windows*

Generative simulation of tokamak plasma diagnostics, from the TCV confinement-state dataset.

PlasmaFlow samples a future window of five diagnostic channels given a history window and three
programmed actuator covariates, using conditional flow matching over a 1D U-Net. Chaining the
samples autoregressively gives full-length synthetic discharges.


- Task: `x_WF ~ p(x_WF | x_WH, c_WH:WF)`, windows of L=256 samples (25.6 ms) at 10 kHz.
- Channels generated: `FIR_LIDs_core`, `PD` (H-alpha), `DML`, `POHM`, `Z_axis`.
- Covariates: `IP` (programmed current reference), `PNBI`, `PECRH`.


## Repo map

| Path | What it is |
|---|---|
| `run.py` | Entry point: config, data module, train, validate, rollouts, test |
| `configs/plasmaflow.yaml` | The paper config; everything else is included from here |
| `configs/{cfm,unflow,itransformer,tide,dlinear,patchtst}.yaml` | Model family definitions |
| `src/models/flow.py` | `FlowModule`: all flow-matching logic, manual optimization loop |
| `src/models/UnFlow.py` | `UnFlowModule`: deterministic MSE ablation, also hosts the baselines |
| `src/models/unet_conditional.py` | `ConditionalUNet` backbone |
| `src/models/tslib/` | Vendored Time-Series-Library baselines, kept close to upstream |
| `src/data_loaders.py` | `FusionShotDataset`, `FusionShotDataModule`, normalization, splits |
| `src/run_processing.py` | Raw shot parquets to the combined parquet (local only) |
| `src/evaluation.py` | `PlotsCallback`, `evaluate_window_set` |
| `src/rollout.py`, `src/rollout_cache.py` | Autoregressive rollouts and their cache |
| `src/metrics/` | Peak, mode-transition, moment and entropy metrics, FNOLSTM surrogate labels |
| `src/hdf_cache.py` | HDF5 test cache and the JSON metric friend |
| `src/HPC_setup/` | Snellius submission scripts |
| `eval_notebooks/` | Jupytext analysis scripts that turn caches into paper tables and figures |
| `docs/` | The real documentation; start at `docs/README.md` |

## Docs
This codebase is fully documented! Have a look at the docs index: [docs/README.md](docs/README.md).


| Doc | Theme |
|---|---|
| [docs/run-lifecycle.md](docs/run-lifecycle.md) | Everything a run does, step by step |
| [docs/configuration.md](docs/configuration.md) | Every key in `plasmaflow.yaml` |
| [docs/data-pipeline.md](docs/data-pipeline.md) | Raw shots to training batches |
| [docs/evaluation-metrics.md](docs/evaluation-metrics.md) | Metric definitions and wandb keys |
| [docs/baselines.md](docs/baselines.md) | Baselines: vendoring, reconciliations, integrity |
| [docs/plots.md](docs/plots.md) | Plot types and configuration |
| [docs/outputs.md](docs/outputs.md) | Every output artifact and what triggers it |
| [docs/hpc-snellius.md](docs/hpc-snellius.md) | Cluster submit, sync, debug, venv |
| [docs/run_grid.md](docs/run_grid.md) | Run naming and the full experiment grid |
| [docs/notebooks.md](docs/notebooks.md) | Which notebook makes which figure or table |


## Requirements
Requirements are maintained in `Pipfile` using pipenv for local use. The same file describes the pip commands needed for 
setup on a cluster.
Local (macOS/Linux, Python 3.11.3), using pipenv:
```bash
pipenv install --dev
pipenv shell
```

Cluster (Snellius), no pipenv; module loads plus a plain venv, per the notes at the bottom of `Pipfile`:

```bash
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load plotly.py/5.16.0-GCCcore-12.3.0
module unload SciPy-bundle/2023.07-gfbf-2023a
python -m venv ~/fusion && source ~/fusion/bin/activate
pip install pyarrow torchinfo omegaconf lightning wandb pytest pot antropy "plotly<6.0.0" \
            kaleido torchdiffeq pysdtw tables datashader h5py scipy
```


Runs log to Weights and Biases. For new runs, the full config is managed through wandb, so `wandb` is not optional, but you can set it to offline mode using `WANDB_MODE=disabled` in your env.

## Data

- We train PlasmaFlow on the public TCV confinement-state dataset, Poels/Venturini et al., *Nuclear Fusion* 2025.
  Zenodo: <https://zenodo.org/records/16631053>. Tracked as submodule `giants/TCV-confstate-data`.
- Raw per-shot parquets are expected under `data/public_data_set/data/`, with the metadata JSONs
  (`column_to_latex.json`, `data_splits.json`, `experiment_to_shot.json`) under `.../metadata/`.
- Preprocessing (local only, imports the pipfile's dev-only `ydata_profiling`):

  ```bash
  PYTHONPATH=. python src/run_processing.py   # entry point: combine_public_dataset()
  ```

- Output: one combined parquet in `data/`. The config points at `2026_07_13-TCV_shots_V2.parquet`
  (replace `data.file` in `configs/plasmaflow.yaml`).
- Windowing, normalization and splits: `src/data_loaders.py`. Min-max to `[0,1]`, statistics from
  the train split only, applied globally. Splits are hardcoded shot lists in the config
  228 train / 30 val / 44 test shots. Only 43 test shots are long enough to contribute
  rollouts, which is the number the paper reports.
  See [docs/data-pipeline.md](docs/data-pipeline.md).
- Upload to the cluster before submitting jobs:

  ```bash
  rsync -vz ./data/*.parquet snellius:~/fusion-plasma-simulation/data/
  ```

## Pretrained models

- Every run pushes checkpoints to wandb as artifacts; `src/config.py:find_and_download_model`
  pulls them back by run name.
- Local copies land in `output/models/{date}-{run_name}/*.ckpt`. Headline checkpoint is 791 MB
  (weights plus optimizer state), 69.1 M trainable parameters.

## Quickstart

Train (or re-evaluate) with the main config on your local machine:

```bash
python run.py run_name=my_experiment              # train + validate + rollouts + test
python run.py reeval=true run_name=my_experiment  # reuse the trained weights, redo evaluation
python run.py resume_id=<wandb_id> resume_name=<run_name>
```

... Or on the cluster (tags the commit, pushes, sbatches over SSH, polls `squeue`, rsyncs slurm logs):

```bash
bash src/HPC_setup/submit_remote_job_snellius.sh my_experiment
bash src/HPC_setup/submit_remote_job_snellius.sh my_experiment epochs=50   # CLI overrides
```

This needs an ssh host alias `snellius` in `~/.ssh/config`:

```
Host snellius
    HostName snellius.surf.nl
    User mtresoor
    AddKeysToAgent yes
    UseKeychain yes
    IdentityFile ~/.ssh/snellius
```
###  Config files
See [the main plasmaflow config at `configs/plasmaflow.yaml`](configs/plasmaflow.yaml)

Switching model family is a one-line config edit: `model:` in `configs/plasmaflow.yaml` points at
one of `configs/{cfm,unflow,itransformer,tide,dlinear,patchtst}.yaml`. Covariate leakage is set by
`data.cols.c` in the same file. Check that line before you train: it is whatever the last grid cell
left it at, not necessarily `cfm.yaml`.

### Where results are stored

| Artifact | Path |
|---|---|
| Checkpoints | `output/models/{date}-{run_name}/*.ckpt` (and wandb) |
| Qualitative PDF plots | `output/pdfplots/{run_name}/qualitative_samples/` |
| Per-window test cache | `$TEST_CACHE_DIR/{test_cache_name}.h5` plus a `.json` friend with recorded metrics |
| Rollout cache and browser | `{name}_rollout.h5`, `output/htmlplots/{run_name}/rollouts.html` |
| Metrics, figures, gradients | wandb |
| Slurm logs (auto-rsynced) | `output/snellius/slurms/gpujob-*.out` |

On Snellius the caches go to `/scratch-shared/mtresoor/final_cache/`; pull them down before running
any analysis notebook. Full list in [docs/outputs.md](docs/outputs.md).

## Evaluation

Evaluation runs inside `run.py` (validate, `evaluate_window_set`, rollouts, `trainer.test`). The
paper tables are then regenerated offline (that means manually, after the run) from the produced rollout caches:

```bash
PYTHONPATH=. python eval_notebooks/rollout_tables.py            # CSVs + LaTeX tables + depth plots
PYTHONPATH=. python eval_notebooks/rollout_tables.py --shots 3  # smoke run
PYTHONPATH=. python eval_notebooks/paper_rollout.py             # qualitative rollout PDFs
```

Outputs go to `output/paper_tables/`. Note that `--shots N` writes its smoke-run numbers to the same
filenames, so it overwrites the real tables; keep a copy or rerun in full afterwards. The rollout
caches are fetched from the cluster automatically on first use. Every rollout script is covered in
[docs/notebooks.md](docs/notebooks.md#rollout-analysis).

The `eval_notebooks/*.py` files are Jupytext cell scripts and assume the repo root on `sys.path`,
hence the `PYTHONPATH=.`.

Metric families (details in [docs/evaluation-metrics.md](docs/evaluation-metrics.md)):

| Metric | What it answers |
|---|---|
| `E_OT` (unbalanced OT over peaks, lambda=10 ms) | Are peaks in the right place, with the right mass |
| `\|dN\|` peak-count error | Does the model emit the right number of excursions |
| Dice `D` on surrogate L/D/H labels | Does the confinement-mode sequence line up in time |
| Wasserstein-1 on peak prominence/width | Peak shape distributions |
| Moment errors, entropy, SoftDTW | Secondary sanity checks |

Mode labels for future windows are surrogate: the FNOLSTM classifier of Poels et al. is run on both
generated and real futures. Note that two label conventions coexist in the code (shifted and
unshifted).

## Experiment grid

Naming: `{ID}-{backbone}-{cov}[-{prior}][-{note}]`. ID letters: `C` CFM, `U` deterministic U-Net,
`I` iTransformer, `T` TiDE, `D` DLinear; then `N` noleak, `S` single leak (`IPLA`), `F` triple leak;
then a variant letter. Full rules in [docs/run_grid.md](docs/run_grid.md).

Main text (all trained on `2026_07_13-TCV_shots_V2.parquet`, attention off):

| Model | ID | Covariates | Role | Cache stem |
|---|---|---|---|---|
| PlasmaFlow (CFM) | `CNb` | noleak | headline model | `R-CNb-cfm-noleak-normal-s05-noatt-e2` |
| PlasmaFlow (CFM) | `CSb` | single leak | timing oracle | `CSb-cfm-ipla-normal-s05-noatt` |
| UnFlow (U-Net) | `UNb` | noleak | flow vs no flow | `UNb-unet-noleak-noatt` |
| UnFlow (U-Net) | `USb` | single leak | timing oracle | `USb-unet-ipla-noatt` |
| TiDE | `TN` / `TS` | noleak / single | covariate-aware baseline | `R-TN-tide-noleak-e2`, `R-TS-tide-ipla-e2` |
| iTransformer | `IN` / `IS` | noleak / single | capacity check | `R-IN-itransformer-noleak-e2`, `R-IS-itransformer-ipla-e2` |
| DLinear | `D` | none | floor baseline | `D-dlinear-r2` |

Appendix ablations: `CN` (attention on), `CNc`/`CNd` (Brownian prior, attention on/off), `UN`, `US`.
Triple-leak dose cells `CF` and `UFv2`. Caches under `output/_archive/` are not part of the paper;
`R-UF-unet-triple-e2` is corrupted and must not be used.

## Results

Autoregressive rollouts from `f=0.5`, first 12 generated windows split into two halves of 6.
Peaks at noise-scale prominence (0.001) on the `[0,1]` signal. Lower is better except Dice.
Source: `output/paper_tables/rollout_depth_table.tex`.

Early half (windows 0-5):

| Model | PD `E_OT` | PD `\|dN\|` | DML `E_OT` | DML `\|dN\|` | Dice |
|---|---|---|---|---|---|
| **PlasmaFlow** | **66.7** | **63.6** | **2.30** | **21.0** | **0.799** |
| U-Net (UnFlow) | 149.3 | 165.0 | 8.25 | 196.4 | 0.750 |
| TiDE | 121.2 | 125.0 | 10.72 | 166.3 | 0.537 |
| iTransformer | 141.0 | 96.8 | 13.08 | 245.5 | 0.671 |
| DLinear | 163.0 | 302.7 | 4.68 | 106.3 | 0.778 |


Late half (windows 6-11):

| Model | PD `E_OT` | PD `\|dN\|` | DML `E_OT` | DML `\|dN\|` | Dice |
|---|---|---|---|---|---|
| **PlasmaFlow** | **108.9** | **83.3** | **2.44** | **27.4** | **0.735** |
| U-Net (UnFlow) | 205.5 | 166.9 | 9.63 | 204.2 | 0.643 |
| TiDE | 190.2 | 118.6 | 10.67 | 164.4 | 0.380 |
| iTransformer | 207.7 | 103.5 | 15.48 | 258.0 | 0.507 |
| DLinear | 221.6 | 301.0 | 4.85 | 120.6 | 0.533 |

Means over 43 test shots. The standard deviations across shots are large and are in the LaTeX
table; ranking is on means. PlasmaFlow wins every column of both halves.
A per-start-fraction version (f=0.05 and f=0.75, with Wasserstein columns) is in
`output/paper_tables/rollout_results_table.tex`.

### Compute cost

Headline `CNb` on one A100-SXM4-40GB partitioned as a `3g.20gb` MIG slice, float32, PyTorch 2.1.2.

| | Quantity | Value |
|---|---|---|
| Model | Trainable parameters | 69.1 M |
| Training | Epochs to early stop (best) | 63 (42) |
| Training | Wall time to best / total | 4.5 h / 6.6 h |
| Training | Peak reserved GPU memory | 19.7 GiB |
| Inference | Per generated window (batch 512, 120 Euler steps) | 60 ms |
| Inference | Full test set, 61,459 windows | 1.6 h |
| Inference | Rollouts, 6,450 chains / 245,790 windows | 4.8 h |

Two config details behind those numbers: reported results come from `reeval=true` runs, and
`configs/reeval.yaml` sets `evaluation.n_steps: 120` (in-training evaluation uses the 100 in
`configs/plasmaflow.yaml`). The 10x batch rematching is `configs/cfm.yaml`, not the code default of 1.

## Contributing, licence, citation

- Work by Milan Tresoor, Simon Koop and Mahdi Mehmanchi, supervised by Vlado Menkovski
  (Eindhoven University of Technology).
- Issues and pull requests are welcome, but this is a research repo: the analysis scripts have  sharp-edges, long text and several carry hardcoded run names.
- Licence pending MIT licence
