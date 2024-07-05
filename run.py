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
run = wandb.init(
    name="EncoderDecoder test 3 BatchNorm",
    project="plasma",
    notes="Added 4 batchnorm layers because the outputs were always lower than the input",
    tags=[
        "SiLu",
        "BatchNorm",
    ],
    config=load_config_dict(),
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
wandb.log({"model_summary": str(model_summary)})
# log weights for analysis in W&B
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()  # note, this is influenced by seq length
eval_citerion = torch.nn.MSELoss(reduction="sum")  # manually average over batches
wandb.watch(model, criterion=criterion, log="all")

data_set = MyDataset(
    file_path=C.data.dir + C.data.file,
    columns_C=list(C.data.cols.c),
    columns_X=list(C.data.cols.x),
)
train_set, val_set = data.random_split(data_set, [0.9, 0.1], generator=torch.Generator().manual_seed(42))

train_loader = data.DataLoader(train_set, batch_size=C.batch_size, shuffle=True)
val_loader = data.DataLoader(val_set, batch_size=C.batch_size, shuffle=False)


def validate(model, data_loader, criterion):
    n_samples = len(data_loader.dataset)
    loss = 0
    future_loss = 0
    model.eval()
    with torch.no_grad():
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


for epoch in track(range(1, C["epochs"] + 1), description="Epoch"):
    validate(model, val_loader, eval_citerion)
    fig = log_predictions(model, val_set, n=5)  # Todo pass epoch for titles
    if epoch % 5 == 0:
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
        f_x = outputs[:, -C["forecast_horizon"]:]
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
