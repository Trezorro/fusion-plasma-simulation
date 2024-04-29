import torch
import wandb
from modules.models import BasicRNN
from modules.data_loaders import MyDataset

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
ALL_SIG_COLLS = COLS_META + COLS_CONTROL + COLS_DATA

# setup wandb
wandb.login()
run = wandb.init(project="plasma", 
                 tags=[
                     "throwaway",
                       ],
                 notes="First seq2seq",
                 config=dict(
                     data_dir = './data/',
                     data_file = '2024_04_23-all_preprocessed.parquet',
                     data_columns = ALL_SIG_COLLS,
                     data_x_columns = COLS_DATA,
                     data_c_columns = COLS_CONTROL,
                     epochs = 10,
                     batch_size = 8,
                 )
)
C: dict = wandb.config



model = BasicRNN(input_size=len(C['data_c_columns'])+len(C['data_x_columns']), 
                 hidden_size=20, 
                 output_size=len(C['data_x_columns']))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()
data_set = MyDataset(file_path=C['data_dir'] + C['data_file'],
                     columns_C=C['data_c_columns'],
                     columns_X=C['data_x_columns'])
data_loader = torch.utils.data.DataLoader(data_set, batch_size=8, shuffle=True)


for epoch in range(0, C['epochs']):
    
    # log metrics to wandb
    for batch_idx, (controls, observables) in enumerate(data_loader):
        # Zero the gradients
        optimizer.zero_grad()
        # Concatenate controls and observables for model input
        inputs = torch.cat((controls, observables), dim=2)
        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, observables)

        # Backward passs
        loss.backward()

        # Update weights
        optimizer.step()

        # Log metrics to wandb
        wandb.log({"loss": loss.item()}, step=batch_idx)
    

