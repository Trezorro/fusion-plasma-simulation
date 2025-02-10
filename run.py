import logging
from src.logging_util import handler

logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info("Starting run.py with imports")
import wandb
import torch
from torch.utils import data
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch.callbacks as pl_callbacks

from src.config import get_current_config, load_config_from_file
from src.evaluation import PlotsCallback
import src.models
import src.data_loaders

logger.info("Imports complete, Loading config and initializing wandb.")

conf = load_config_from_file('fm_toy')
run = wandb.init(
    name=conf.get("run_name", None),
    tags=conf.get("tags", None),
    project="flowtoy",
    config=conf,
    dir="./output/wandb",
    # mode="offline",
)
C = get_current_config()
wandb.define_metric("loss/train", summary="min")
wandb.define_metric("loss/val", summary="min")
wandb.define_metric("loss/time_domain_train", summary="min")
wandb.define_metric("loss/time_domain_val", summary="min")

wandb.define_metric("val/time_pred_batch_variance", summary="max")
wandb.define_metric("val/time_pred_batch_var_mean_adjusted", summary="max")
wandb.define_metric("val/freq_pred_batch_variance", summary="max")
wandb.define_metric("val/freq_pred_batch_var_mean_adjusted", summary="max")
wandb.define_metric("val/freq_pred_batch_input_variance_ratio", summary="max")

wandb.define_metric("val/time_target_batch_variance", summary="max")
wandb.define_metric("val/time_target_batch_var_mean_adjusted", summary="max")
wandb.define_metric("val/freq_target_batch_variance", summary="max")
wandb.define_metric("val/freq_target_batch_var_mean_adjusted", summary="max")
wandb.define_metric("val/freq_target_batch_input_variance_ratio", summary="max")

wandb_logger = WandbLogger(
    log_model=True,
    experiment=run,
    save_dir="output/",
)
logger.info("Config and wandb initialized, loading model and data.")
ModelClass = getattr(src.models, C.model.Class)
model = ModelClass(**C.model.params)
if not C.skip_log_summary:
    model.log_summary(C)
# log weights for analysis in W&B
wandb_logger.watch(model, log="all", log_freq=50)
logger.info("Model loaded, loading data.")
DataSetClass = getattr(src.data_loaders, C.data.Class)
data_set = DataSetClass(**C.data)

train_set, val_set = data.random_split(data_set, [0.9, 0.1], generator=torch.Generator().manual_seed(42))

train_loader = data.DataLoader(train_set, batch_size=C.batch_size, shuffle=True)
val_loader = data.DataLoader(val_set, batch_size=C.batch_size, shuffle=False)
val_loader.dataset.random_start = False  # TODO this doesn't work, wrong dataset attribute.

logger.info("Data loaded, initializing trainer.")
trainer = L.Trainer(
    default_root_dir="output/",
    enable_progress_bar=wandb.run.disabled,
    max_epochs=C["epochs"],
    logger=wandb_logger,
    # num_sanity_val_steps=1,
    log_every_n_steps=1,
    check_val_every_n_epoch=1,  # May validate less often
    gradient_clip_val=C["gradient_clip_val"],  # gradient_clip_algorithm='norm' by default
    callbacks=[
        pl_callbacks.EarlyStopping(monitor="loss/val", patience=C.patience, mode="min"),
        PlotsCallback(C.evaluation)
    ]
)
logger.info("Starting training with first validation...")
# trainer.validate(model=model, dataloaders=val_loader)
logger.info("Starting model fit...")
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
logger.info("Starting final validation...")
trainer.test(model=model, dataloaders=val_loader)
logger.info("Finished training.")
