import torch
from torch.utils import data
import wandb
from rich.progress import track

from src.config import get_current_config, load_config_from_file
from src.evaluation import log_predictions
import src.models
from src.training import validate
import src.utils as utils
from src.data_loaders import MyDataset

from torchinfo import summary

wandb.login()
conf = load_config_from_file()
run = wandb.init(
    name=conf.get("run_name", None),
    project="plasma",
    notes=
    "Loss was always 0 before because of the wrong loss function. Fixed now. Also added a random crop to the data loader.",
    tags=["SiLu", "BatchNorm", "random crop"],
    config=conf,
    dir="./output/wandb")
C = get_current_config()

ModelClass = getattr(src.models, C.model.Class)
model = ModelClass(**C.model.params)
model_summary = summary(
    model,
    input_size=[(C.seq_length, len(C.data.cols.c)), (C.seq_length, len(C.data.cols.x))],
    batch_dim=0,
    col_names=[
        "input_size",
        "output_size",
        "num_params",
        # "params_percent",
        # "kernel_size",
        "mult_adds",
        # "trainable"
    ],
)  # (batch_size, seq_length, input_size)
# compressed_length = model.encoder.calculate_compressed_length(C.seq_length - C.forecast_horizon)
# print(f"Compressed length: {compressed_length} for warmup window {C.seq_length - C.forecast_horizon}")
wandb.log(
    {
        "model_summary": str(model_summary),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        # "compressed_length": compressed_length,
    },
    step=0)

# log weights for analysis in W&B
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
CriterionClass = getattr(torch.nn, C.loss)
criterion = CriterionClass()  # note, this is influenced by seq length
eval_citerion = CriterionClass(reduction="sum")  # manually average over batches
wandb.watch(model, criterion=criterion, log="all")

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
val_loader = data.DataLoader(val_set, batch_size=C.batch_size * 2, shuffle=False)
val_loader.dataset.random_start = False

mean_loss = validate(model, val_loader, eval_citerion)
wandb.log({"loss/val": mean_loss}, step=0)

with utils.progress as progress:
    for epoch in progress.track(
            range(1, C["epochs"] + 1),
            description="Epoch",
            total=C["epochs"],
    ):

        model.train()
        for batch_idx, (shot_number, controls, observables) in enumerate(train_loader):
            optimizer.zero_grad()
            # partial_observables = torch.zeros_like(observables)
            # partial_observables[:, :-C["forecast_horizon"]] = observables[:, :-C["forecast_horizon"]]
            # Concatenate controls and observables for model input
            # input:(batch_size, seq_length, input_size)
            # inputs = torch.cat((controls, partial_observables), dim=2)
            # Forward pass
            outputs = model(x=observables, c=controls)[:, -C["forecast_horizon"]:]

            # Compute loss
            f_x = observables[:, -C["forecast_horizon"]:]
            loss = criterion(outputs, f_x)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()
            with torch.no_grad():
                future_loss = criterion(
                    outputs[:, -C["forecast_horizon"]:],
                    observables[:, -C["forecast_horizon"]:],
                )

            # Log metrics to wandb
        mean_loss = validate(model, val_loader, eval_citerion)
        fig = log_predictions(model, val_set, title=f"{wandb.run.name} - Epoch {epoch}", n=5)
        progress.console.print(f"Epoch {epoch:03d}: Loss {loss:.5f} | Val Loss {mean_loss:.5f}")
        # if (epoch - 1) % 33 == 0:
        if epoch == 1:
            fig.show()
        wandb.log({
            "epoch": epoch,
            "loss/train": loss.item(),
            "loss/val": mean_loss
            # "train/future_loss": future_loss.item(),
            # "train/weights": model.state_dict(),
        })
