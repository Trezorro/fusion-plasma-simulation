import wandb
import torch
from torch.utils import data
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch.callbacks as pl_callbacks

from src.config import get_current_config, load_config_from_file
from src.evaluation import PlotPredictionsCallback
import src.models
from src.data_loaders import MyDataset

conf = load_config_from_file()
run = wandb.init(
    name=conf.get("run_name", None),
    project="plasma",
    notes="Unet, now with transposed convolutions",
    tags=[],
    config=conf,
    dir="./output/wandb"
)
C = get_current_config()
wandb.define_metric("loss/train", summary="min")
wandb.define_metric("loss/val", summary="min")
wandb.define_metric("loss/val_train_rollout", summary="min")

wandb_logger = WandbLogger(
    log_model=False,
    experiment=run,
    save_dir="output/",
)

ModelClass = getattr(src.models, C.model.Class)
model = ModelClass(**C.model.params)
model.log_summary(C)
# log weights for analysis in W&B
wandb_logger.watch(model, log="all", log_freq=50)

data_set = MyDataset(
    file_path=C.data.dir + C.data.file,
    columns_C=list(C.data.cols.c),
    columns_X=list(C.data.cols.x),
    seq_length=C.seq_length,
    crop_margin=C.crop_margin,
    random_start=C.random_start,
)
train_set, val_set = data.random_split(data_set, [0.9, 0.1], generator=torch.Generator().manual_seed(42))

train_loader = data.DataLoader(train_set, batch_size=C.batch_size, shuffle=True)
val_loader = data.DataLoader(val_set, batch_size=C.batch_size, shuffle=False)
val_loader.dataset.random_start = False  # TODO this doesn't work, wrong dataset attribute.

trainer = L.Trainer(
    default_root_dir="output/",
    enable_progress_bar=False,
    max_epochs=C["epochs"],
    logger=wandb_logger,
    log_every_n_steps=1,
    check_val_every_n_epoch=1,  # May validate less often
    callbacks=[
        pl_callbacks.EarlyStopping(monitor="loss/val", patience=C.patience, mode="min"),
        PlotPredictionsCallback(num_samples=5, every_n_epochs=5, train_every_n_epochs=20),
    ]
)

trainer.validate(model=model, dataloaders=val_loader)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(model=model, dataloaders=val_loader)
