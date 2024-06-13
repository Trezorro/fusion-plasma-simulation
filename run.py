import numpy as np
import pandas as pd
import torch
from torch.utils import data
import wandb
from modules.models import BasicRNN
from modules.data_loaders import MyDataset

from torchinfo import summary


COLS_META = [
    "ShotNum",
    "time",
]
COLS_CONTROL = [
    "IP",  # Current (niet reference lijn voor controller, maar de ware input. Dan laat je control bij control)
    "gas_fringes",  # Ingepompte gas
    "NBI",  # manieren om te verhitten: colliding Neutral beam injection
    "ECRH",  # magnetron.
    "a_minor",  # reel gemeten plasma shape a k d (horizontale radius
    "KAPPA",
    "DELTA"  # inkerbovenhoek nar links vanuit hetmidden
]
COLS_DATA = [
    # "FIR",  # density lijn Interferometer
    "FIR_core",  # For the March dataset of 260 shots, the FIR_core signal is the same as FIR.
    "PD",  # photodiode lijn op de divertor
    "DML",  # Magnetische respons  correleert met de energie in het plasma
    "POHM",  # Gemeten power waarde meet de power die uit wrijving komt
    "Z_axis"  # center Plasma positie in de verticale lijn. deviation van reference is betekenis. 
]
COLS_LABEL = ["LHD_label"]

# setup wandb
wandb.login()
run = wandb.init(project="plasma", 
                 tags=[
                    #  "throwaway",
                       ],
                 notes="seq2seq cleaner data",
                 config=dict(
                     data_dir = './data/',
                     data_file = '2024_05_01-NaNsFiltered.parquet',
                     data_columns = COLS_META + COLS_CONTROL + COLS_DATA,
                     data_x_columns = COLS_DATA,
                     data_c_columns = COLS_CONTROL,
                     data_seq_length = 2000,
                     epochs = 3,
                     batch_size = 16,
                 ),
                 
)
C = wandb.config

model = BasicRNN(input_size=len(C['data_c_columns'])+len(C['data_x_columns']), # 12
                 hidden_size=20, 
                 output_size=len(C['data_x_columns']),
                 batch_size=C['batch_size'])

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()  # note, this is influenced by seq length
eval_citerion = torch.nn.MSELoss(reduction='sum')  # manually average over batches
data_set = MyDataset(file_path=C['data_dir'] + C['data_file'],

                     columns_C=C['data_c_columns'],
                     columns_X=C['data_x_columns'])
train_set, val_set = data.random_split(data_set, [0.9, 0.1], generator=torch.Generator().manual_seed(42))

train_loader = data.DataLoader(train_set, batch_size=C.batch_size, shuffle=True)
val_loader = data.DataLoader(val_set, batch_size=C.batch_size, shuffle=False)

model_summary = summary(model, 
                        input_size=(C.data_seq_length, len(C.data_c_columns)+len(C.data_x_columns)), batch_dim=0, 
                        col_names=[# "input_size", 
                                   "output_size", "num_params",
                                   #"params_percent",
                                   # "kernel_size",
                                   "mult_adds",
                                   # "trainable" 
                                   ]) # (batch_size, seq_length, input_size)
wandb.log({"model_summary": str(model_summary)})
# log weights for analysis in W&B
wandb.watch(model, criterion=criterion, log="all")

def validate(model, data_loader, criterion):
    n_samples = len(data_loader.dataset)
    loss = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (shot_number, controls, observables) in enumerate(data_loader):
            inputs = torch.cat((controls, observables), dim=2) # (batch_size, seq_length, variables)
            outputs = model(inputs)
            loss += criterion(outputs, observables).item()

        mean_loss = loss / n_samples
        wandb.log({"val_loss": mean_loss})

def log_predictions(model, data_set, n=5):
    model.eval()
    with torch.no_grad():
        shot_numbers, controls, observables = next(iter(data.DataLoader(data_set, batch_size=n, shuffle=False)))
        inputs = torch.cat((controls, observables), dim=2) # (batch_size, seq_length, variables)
        outputs = model(inputs) #  (batch_size, seq_length, target_variables)
        output_cols = [f"^{i}" for i in C['data_x_columns']]
        seq_length = outputs.shape[1]
        df = pd.DataFrame(index=np.repeat(shot_numbers.numpy().astype(int), seq_length),
                          columns=['t'] + output_cols +C['data_x_columns']+ C['data_c_columns'])
        for shot, output, control_seq, observable_seq in zip(shot_numbers, outputs, observables, controls):
            df.loc[int(shot)] = np.concatenate([np.arange(seq_length)[:,np.newaxis], 
                                                output.numpy(),
                                                observable_seq.numpy(),
                                                control_seq.numpy()],
                                               axis=1)
        table = wandb.Table(dataframe=df.reset_index(names='ShotNum'))
        wandb.log({"val_predictions": table})

for epoch in range(1, C['epochs']+1):
    validate(model, val_loader, eval_citerion)
    log_predictions(model, val_set, n=5)
    model.train()
    # log metrics to wandb
    for batch_idx, (shot_number, controls, observables) in enumerate(train_loader):
        # Zero the gradients
        optimizer.zero_grad()
        # Concatenate controls and observables for model input
        inputs = torch.cat((controls, observables), dim=2) # (batch_size, seq_length, input_size)
        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, observables)

        # Backward passs
        loss.backward()

        # Update weights
        optimizer.step()

        # Log metrics to wandb
        wandb.log({"epoch": epoch, "loss": loss.item(), "weights":model.state_dict()})

    

