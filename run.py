import random
import torch
from torch.utils import data
import wandb
from rich.progress import track

from config import get_current_config, load_config_dict
from modules.evaluation import log_predictions
import modules.models
from modules.data_loaders import MyDataset

from torchinfo import summary

wandb.login()
conf = load_config_dict()
run = wandb.init(
    name=conf.get("run_name", None),
    project="plasma",
    notes=
    "Loss was always 0 before because of the wrong loss function. Fixed now. Also added a random crop to the data loader.",
    tags=["SiLu", "BatchNorm", "random crop"],
    config=conf,
)
C = get_current_config()

ModelClass = getattr(modules.models, C.model.Class)
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
        "kernel_size",
        "mult_adds",
        # "trainable"
    ],
)  # (batch_size, seq_length, input_size)
compressed_length = model.encoder.calculate_compressed_length(C.seq_length - C.forecast_horizon)
print(f"Compressed length: {compressed_length} for warmup window {C.seq_length - C.forecast_horizon}")
wandb.log({
    "model_summary": str(model_summary),
    "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "compressed_length": compressed_length,
})

# log weights for analysis in W&B
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()  # note, this is influenced by seq length
eval_citerion = torch.nn.MSELoss(reduction="sum")  # manually average over batches
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


def validate(model, data_loader, criterion):
    n_samples = len(data_loader.dataset)
    loss = 0
    future_loss = 0
    model.eval()
    with torch.inference_mode():
        for batch_idx, (shot_number, controls, observables) in enumerate(data_loader):
            partial_observables = torch.zeros_like(observables)
            partial_observables[:, :-C["forecast_horizon"]] = observables[:, :-C["forecast_horizon"]]
            # Input: (batch_size, seq_length, variables)
            inputs = torch.cat((controls, partial_observables), dim=2)
            outputs = model(controls, observables)
            loss += criterion(outputs, observables[:, -C["forecast_horizon"]:]).item()
            future_loss += criterion(
                outputs[:, -C["forecast_horizon"]:],
                observables[:, -C["forecast_horizon"]:],
            ).item()

        mean_loss = loss / n_samples
        mean_future_loss = future_loss / n_samples
        wandb.log({"val/loss": mean_loss, "val/future_loss": mean_future_loss})


for epoch in track(range(1, C["epochs"] + 1), description="Epoch", total=C["epochs"]):
    validate(model, val_loader, eval_citerion)
    fig = log_predictions(model, val_set, title=f"{wandb.run.name} - Epoch {epoch}", n=5)
    if (epoch - 1) % 33 == 0:
        fig.show()
    model.train()
    # log metrics to wandb
    for batch_idx, (shot_number, controls, observables) in enumerate(train_loader):
        # Zero the gradients
        optimizer.zero_grad()
        partial_observables = torch.zeros_like(observables)
        partial_observables[:, :-C["forecast_horizon"]] = observables[:, :-C["forecast_horizon"]]
        # Concatenate controls and observables for model input
        # input:(batch_size, seq_length, input_size)
        # inputs = torch.cat((controls, partial_observables), dim=2)
        # Forward pass
        outputs = model(x=observables, c=controls)

        # Compute loss
        f_x = observables[:, -C["forecast_horizon"]:]
        loss = criterion(outputs, f_x)

        # Backward passs
        loss.backward()

        # Update weights
        optimizer.step()
        with torch.no_grad():
            future_loss = criterion(
                outputs[:, -C["forecast_horizon"] :],
                observables[:, -C["forecast_horizon"] :],
            )

        # Log metrics to wandb
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": loss.item(),
                "train/future_loss": future_loss.item(),
                "train/weights": model.state_dict(),
            }
        )
