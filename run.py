import logging
from src.config import consolidate_base_reeval_configs
from src.logging_util import handler

logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger('src').setLevel(logging.DEBUG)
logger.info("Starting run.py with wandb init.")
import wandb
import wandb.env
from src.config import get_current_config, load_config_from_file, find_and_download_model, pretty_config, is_reeval_run

logger.debug(
    "Wandb dirs: \n  main: %s, \n  data dir: %s, \n  artifacts: %s", wandb.env.get_dir(),
    wandb.env.get_data_dir(), wandb.env.get_artifact_dir()
)

PROJECT = "flowtoy"
if is_reeval_run(): # Based on cli arguments
    logger.info("Re-evaluating previous run, consolidating configs.")
    base_run, conf = consolidate_base_reeval_configs()
    base_checkpoint_path = find_and_download_model(base_run, prefer_alias=conf.get('prefer_model_alias', 'latest'))
else:
    conf = load_config_from_file('fm_toy')
    if "resume_id" in conf:
        logger.info("RESUME MODE. This job %s is resuming run %s with ID: %s.", conf.get('run_name', '?'), conf.get('resume_name', '?'), conf.get('resume_id') )
        base_checkpoint_path = find_and_download_model(conf.get('resume_name'))
        conf['run_name'] = conf['resume_name']
    else:
        logger.info("Training new model, initializing new wandb run.", )
        base_checkpoint_path = None

run = wandb.init(
    name=conf.get("run_name", None), # None: Wandb picks a name for us
    project=PROJECT,
    id=conf.get("resume_id", None),
    resume='allow',
    config=conf,
    # mode="offline",
)
RUN_ID = wandb.run.id
RUN_NAME = wandb.run.name
C = get_current_config(wandb_only=True) # clean synchronized config object from wandb
logger.info("Running with config:\n%s", pretty_config(C))

logger.info("Run initialized, importing torch and lightning.")
import torch
from torch.utils import data
import lightning as L
import lightning.pytorch.callbacks as pl_callbacks
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelCheckpoint

# if torch.cuda.is_available():
#     logger.info("CUDA is available, logging GPU memory.")
#     torch.cuda.memory._record_memory_history()

logger.info("Torch and lightning imported, importing src modules.")

import src.evaluation
from datetime import datetime
import src.models
import src.data_loaders
import src.metrics.metrics as metrics

logger.info("Imports complete, init wandb logger and loading model and data.")

logger.info("Trying to init data module:")
fusion_data_module = src.data_loaders.FusionShotDataModule(**C.data)


metrics.define_error_metrics("val")
metrics.define_error_metrics("train")
wandb.define_metric("loss/train", summary="min")
wandb.define_metric("loss/val", summary="min")
current_date: str = datetime.now().strftime("%Y-%m-%d")
dated_run_name = current_date + '-' + RUN_NAME
wandb_logger = WandbLogger(
    log_model="all",
    experiment=run,
    # save_dir="output/models/",  # where to save the model checkpoints, will get lighting_logs/ appended
    checkpoint_name=dated_run_name,  # name of the wandb artifact
)

ModelClass: src.models.FlowModule = getattr(src.models, C.model.Class)
if base_checkpoint_path is not None:
    model = ModelClass.load_from_checkpoint(base_checkpoint_path)
else:
    model = ModelClass(**C.model.params)  # fresh model
    wandb_logger.watch(model, log="all", log_freq=50)  # log gradients

if 'test_cache_name' in C:
    model.set_cache(C.test_cache_name, C.test_cache_mode)
if 'evaluation' in C:
    model.set_integration_method(C.evaluation.n_steps, C.evaluation.get("solve_method", None))
if "skip_log_summary" not in C or not C["skip_log_summary"]:
    logger.info("Model loaded, summary:")
    model.log_summary(C)
# log weights for analysis in W&B
logger.info("Model loaded, loading data.")

logger.info("Data loaded, initializing trainer.")
# torch.profiler.profile(
#     activities=[
#         torch.profiler.ProfilerActivity.CPU,
#         torch.profiler.ProfilerActivity.CUDA],
#     schedule=torch.profiler.schedule(
#         wait=1,
#         warmup=1,
#         active=10),
#     on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir, worker_name='worker0'),
#     record_shapes=True,
#     profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
#     with_stack=True

trainer = L.Trainer(
    default_root_dir="output/",
    enable_progress_bar=wandb.run.disabled,
    max_epochs=C["epochs"],
    logger=wandb_logger,
    fast_dev_run=2,
    profiler='simple',
    limit_train_batches=C.limit_train_batches,
    limit_val_batches=C.limit_val_batches,
    limit_test_batches=C.get("limit_test_batches", None),
    max_time={"hours": 10},
    benchmark=True,
    # num_sanity_val_steps=1,
    log_every_n_steps=1,
    check_val_every_n_epoch=1,  # May validate less often
    # gradient_clip_val=C["gradient_clip_val"],  # gradient_clip_algorithm='norm' by default
    callbacks=[
        src.evaluation.PlotsCallback(C.evaluation),
        pl_callbacks.EarlyStopping(monitor="loss/val", patience=C.patience, mode="min"),
        pl_callbacks.LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            monitor="loss/val",
            mode="min",
            save_last=True,
            verbose=True,
            dirpath="output/models/" + dated_run_name,  # lightning_logs by default
            filename=dated_run_name + '-Epoch={epoch:02d}-step={step}-val_loss={loss/val:.2f}',
            auto_insert_metric_name=False
        ),
        src.evaluation.TrainStepMonitor(),
    ]
)

if not is_reeval_run():  # Train fresh model
    logger.info("Starting model fit...")
    trainer.fit(model=model, datamodule=fusion_data_module)
    logger.info("Finished training.")
    wandb_logger.experiment.unwatch(model)

# if torch.cuda.is_available():
#     logger.info("Dumping CUDA memory snapshot...")
#     memory_trace_file = f'output/traces/{RUN_NAME}_memory_trace.pickle'
#     torch.cuda.memory._dump_snapshot(filename=memory_trace_file)
#     logger.info("CUDA memory snapshot dumped at %s", memory_trace_file)

# model.solve_method = "rk4"
# model.flow_steps = 20
logger.info("Starting model validation...")
trainer.validate(model=model, datamodule=fusion_data_module)
logger.info("Starting final testing...")
trainer.test(model=model, datamodule=fusion_data_module)
logger.info("Finished testing.")


logger.info("Run finished, deleting redundant artifacts.")
if not wandb.run.disabled:
    src.evaluation.prune_online_checkpoints(run)

logger.info("Goodbye!")
run.finish()
