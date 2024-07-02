import torch
from torch.utils import data
import wandb
from modules.evaluation import log_predictions
from modules.models import BasicRNN
from modules.data_loaders import MyDataset

from torchinfo import summary

config_data = dict(
    dir="./data/",
    file="2024_05_01-NaNsFiltered.parquet",
    cols=dict(
        meta=[
            "ShotNum",
            "time",
        ],
        x=[
            # "FIR",  # density lijn Interferometer
            "FIR_core",  # For the March dataset of 260 shots, the FIR_core signal is the same as FIR.
            "PD",  # photodiode lijn op de divertor
            "DML",  # Magnetische respons  correleert met de energie in het plasma
            "POHM",  # Gemeten power waarde meet de power die uit wrijving komt
            "Z_axis",  # center Plasma positie in de verticale lijn. deviation van reference is betekenis.
        ],
        c=[
            "IP",  # Current (niet reference lijn voor controller, maar de ware input. Dan laat je control bij control)
            "gas_fringes",  # Ingepompte gas
            "NBI",  # manieren om te verhitten: colliding Neutral beam injection
            "ECRH",  # magnetron.
            "a_minor",  # reel gemeten plasma shape a k d (horizontale radius
            "KAPPA",
            "DELTA",  # linkerbovenhoek nar links vanuit hetmidden
        ],
        label=["LHD_label"],
    ),
)
config_dict = dict(
    data=config_data,
    seq_length=2000,
    forecast_horizon=200,
    epochs=4,
    batch_size=16,
)


# setup wandb
wandb.login()
run = wandb.init(
    project="plasma",
    tags=[
        #  "throwaway",
    ],
    notes="seq2seq partial observables",
    config=config_dict,
)
C = wandb.config


model = BasicRNN(
    input_size=len(C.data.cols.c) + len(C.data.cols.x),  # 12
    hidden_size=20,
    output_size=len(C.data.cols.x),
    batch_size=C["batch_size"],
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()  # note, this is influenced by seq length
eval_citerion = torch.nn.MSELoss(reduction="sum")  # manually average over batches
data_set = MyDataset(
    file_path=C.data.dir + C.data.file,
    columns_C=C.data.cols.c,
    columns_X=C.data.cols.x,
)
train_set, val_set = data.random_split(
    data_set, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
)

train_loader = data.DataLoader(train_set, batch_size=C.batch_size, shuffle=True)
val_loader = data.DataLoader(val_set, batch_size=C.batch_size, shuffle=False)

model_summary = summary(
    model,
    input_size=(C.data_seq_length, len(C.data_c_columns) + len(C.data_x_columns)),
    batch_dim=0,
    col_names=[  # "input_size",
        "output_size",
        "num_params",
        # "params_percent",
        # "kernel_size",
        "mult_adds",
        # "trainable"
    ],
)  # (batch_size, seq_length, input_size)
wandb.log({"model_summary": str(model_summary)})
# log weights for analysis in W&B
wandb.watch(model, criterion=criterion, log="all")


def validate(model, data_loader, criterion):
    n_samples = len(data_loader.dataset)
    loss = 0
    future_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (shot_number, controls, observables) in enumerate(data_loader):
            partial_observables = torch.zeros_like(observables)
            partial_observables[:, : -C["forecast_horizon"]] = observables[
                :, : -C["forecast_horizon"]
            ]
            # Input: (batch_size, seq_length, variables)
            inputs = torch.cat((controls, partial_observables), dim=2)
            outputs = model(inputs)
            loss += criterion(outputs, observables).item()
            future_loss += criterion(
                outputs[:, -C["forecast_horizon"] :],
                observables[:, -C["forecast_horizon"] :],
            ).item()

        mean_loss = loss / n_samples
        mean_future_loss = future_loss / n_samples
        wandb.log({"val/loss": mean_loss, "val/future_loss": mean_future_loss})


for epoch in range(1, C["epochs"] + 1):
    validate(model, val_loader, eval_citerion)
    log_predictions(model, val_set, n=5)
    model.train()
    # log metrics to wandb
    for batch_idx, (shot_number, controls, observables) in enumerate(train_loader):
        # Zero the gradients
        optimizer.zero_grad()
        partial_observables = torch.zeros_like(observables)
        partial_observables[:, : -C["forecast_horizon"]] = observables[
            :, : -C["forecast_horizon"]
        ]
        # Concatenate controls and observables for model input
        # input:(batch_size, seq_length, input_size)
        inputs = torch.cat((controls, partial_observables), dim=2)
        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, observables)

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
