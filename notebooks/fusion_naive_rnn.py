# %% Orignal by Jakub Tomczak for the book Deep Generative Models
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
import pandas as pd
import tqdm

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# %% [markdown]

# %%
data_dir = '../data/LHD_labeled_TCV/'
sig_all_names = glob.glob(data_dir + 'TCV_DATA*.parquet')
# use regex to get sample number:
sig_all = {int(re.findall(r'\d+', x)[0]): x for x in sig_all_names}

label_all = glob.glob(data_dir + 'TCV_*_apau_labeled.csv')
label_all = {int(x.split("TCV_")[1].split("_apau_labeled.csv")[0]): x for x in label_all}
shot_no_list = list(sig_all.keys())

print(f"All shots: {shot_no_list}")
print("Amount of shots: ", len(shot_no_list))
print(f"Labels: {len(label_all)}")


# %%
# example
shotno = shot_no_list[0]

sig = pd.read_parquet(sig_all[shotno])
label = pd.read_csv(label_all[shotno])
assert 'FIR_core' in sig.columns.tolist(), "FIR_core not in columns"
sig.columns


# %%
sig

# %%
def check_time_consistency(signal_df):
    """Check if the time steps are consistent for a shot.

    Should give a frequency of about 10 kHz for TCV data.
    """
    time_diff = signal_df['time'].diff()
    frequency = 1 / time_diff.mean()
    is_consistent = time_diff.std() < 1e-7
    inconsistent_steps = ~np.isclose(time_diff, time_diff.mean(), atol=1e-7, equal_nan=True)
    # print(f"Frequency: {frequency}, is broadly consistent: {is_consistent}")
    # print(f"{inconsistent_steps.sum()} steps out of {len(inconsistent_steps)} were not exaclty the same as the mean step size.")
    return is_consistent, inconsistent_steps.sum(), frequency
# %%
for i, shotno in enumerate(shot_no_list):
    sig = pd.read_parquet(sig_all[shotno])
    print(f"[{i+1}] Shot {shotno} has {len(sig)} timesteps from {sig['time'].min()} to {sig['time'].max()}")
# %%
C_COLS = [
    "IP",  # Current (niet reference lijn voor controller, maar de ware input. Dan laat je control bij control)
    "gas_fringes",  # Ingepompte gas
    # "NBI",  # manieren om te verhitten: colliding Neutral beam injection
    # "ECRH",  # magnetron.
    # "a_minor",  # reel gemeten plasma shape a k d (horizontale radius
    # "KAPPA",
    # "DELTA"  # inkerbovenhoek nar links vanuit hetmidden
]
X_COLS = [  
    # "FIR",  # density lijn Interferometer
    "FIR_core",  # For the March dataset of 260 shots, the FIR_core signal is the same as FIR.
    "PD",  # photodiode lijn op de divertor
    # "DML",  # Magnetische respons  correleert met de energie in het plasma
    # "POHM",  # Gemeten power waarde meet de power die uit wrijving komt
    # "Z_axis"  # center Plasma positie in de verticale lijn. deviation van reference is betekenis. 
]
all_cols = C_COLS + X_COLS # 7 and 5 columns
TIME_COL = "time"
QUANTIZATION_LEVELS = 50

# %%
class ParquetDataset(Dataset):
    """Custom dataset for loading data from parquet files."""

    def __init__(self, data_dir, sig_all, label_all, transforms=None):
        self.data_dir = data_dir
        self.sig_all = sig_all
        self.label_all = label_all
        self.transforms = transforms

        self.shot_no_list = list(sig_all.keys())
        print(f"Loaded {len(self.shot_no_list)} shots")

        # Calculate mean and standard deviation of y values
        self.y_mean = None
        self.y_std = None
        # self.calculate_y_stats()

    # def calculate_y_stats(self):
    #     """Set y_mean and y_std based on all y values in the dataset.

    #     Improvements:
    #         - [ ] Use the same temporal window to normalize
    #     """
    #     y_values = []
    #     self.y_min = float('inf')
    #     self.y_max = -float('inf')
    #     for shotno in self.shot_no_list:
    #         sig = pd.read_parquet(self.sig_all[shotno])
    #         y = sig[X_COLS].values
    #         # update min and max
    #         if np.min(y) < self.y_min:
    #             self.y_min = np.min(y)
    #         if np.max(y) > self.y_max:
    #             self.y_max = np.max(y)
    #     print(f"y_min: {self.y_min}, y_max: {self.y_max}")

    # def normalize_y(self, y): # TODO: do much smarter quantization
    #     normalized = (y - self.y_min) / (self.y_max - self.y_min)
    #     quantized = (normalized * (QUANTIZATION_LEVELS - 1))
    #     return quantized

    def __len__(self):
        return len(self.shot_no_list)

    def __getitem__(self, idx):
        shotno = self.shot_no_list[idx]

        sig = pd.read_parquet(self.sig_all[shotno])

        time_steps = len(sig)
        start = time_steps // 2 - 2000
        end = start + 4000

        c = sig[C_COLS].values.astype(np.float32)[start:end]
        x = sig[X_COLS].values.astype(np.float32)[start:end]
        assert c.shape[0] == 4000, f"Expected 4000 time steps, got {c.shape[0]} for shot {shotno}"
        assert x.shape[0] == 4000, f"Expected 4000 time steps, got {x.shape[0]} for shot {shotno}"
        c = torch.tensor(c)
        x = torch.tensor(x)
        return c, x
# %%

#%%

dataset = ParquetDataset(data_dir, sig_all, label_all)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

#%%
def plot_batch(batch: torch.Tensor):
    c, x = batch
    c = c.detach().numpy()
    fig, ax = plt.subplots(1, 1)
    # set y limits
    # ax.set_ylim([0, 5])
    ax.grid(True)
    for i, timeseries in enumerate(c):
        ax.plot(timeseries, label=f"{i}")
        ax.set_ylabel(C_COLS)
        break
    plt.show()

    # plt.savefig('test.pdf', bbox_inches='tight')
    plt.close()



plot_batch(next(iter(dataloader)))


# %% [markdown] Simple RNN

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)# .to(x.device)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out



# %% [markdown]
# ### Auxiliary functions: training, evaluation, plotting

# %%
def evaluation(test_loader, model_path=None, model_best=None, epoch=None):
    # EVALUATION
    if model_best is None:
        # load best performing model
        if model_path is None:
            raise ValueError('Either model_path or model_best should be provided.')
        model_best = torch.load(model_path + '.model')

    model_best.eval()
    loss = 0.
    N = 0.
    for indx_batch, test_batch in enumerate(test_loader):
        loss_t = model_best.forward(test_batch, reduction='sum')
        loss = loss + loss_t.item()
        N = N + test_batch.shape[0]
    loss = loss / N

    if epoch is None:
        print(f'FINAL LOSS: nll={loss}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}')

    return loss


def samples_real(name, test_loader: DataLoader):
    # REAL-------
    x = next(iter(test_loader)).detach().numpy()
    assert test_loader.batch_size is not None
    fig, ax = plt.subplots(test_loader.batch_size, 1, figsize=(12, 8))
    for i, ax in enumerate(ax.flatten()):
        ax.plot(x[i], linewidth='3', color='black', label=f"{i}")
        ax.set_ylabel(X_COLS)
        ax.set_ylim([0, QUANTIZATION_LEVELS])
        ax.grid()
    plt.tight_layout()
    plt.savefig(name+'_real_images.pdf', bbox_inches='tight')
    plt.show()
    
    plt.close()


def samples_generated(name, data_loader: DataLoader, start_sequence_length=0, extra_name=''):
    real_sequences = next(iter(data_loader)).detach()
    assert data_loader.batch_size is not None

    # GENERATIONS-------
    model_best: ARM = torch.load(name + '.model')
    model_best.eval()

    x = model_best.sample(data_loader.batch_size, start_sequences=real_sequences[:, :start_sequence_length])
    x = x.detach().numpy()
    

    fig, ax = plt.subplots(data_loader.batch_size, figsize=(12, 8))
    for i, ax in enumerate(ax.flatten()):
        ax.plot(real_sequences[i].numpy(), linewidth='3', color='black', label=f"{i}")
        ax.plot(x[i], linewidth='2', color='red', label=f"{i}", alpha=0.7)
        ax.set_ylabel(X_COLS)
        ax.set_ylim([0, QUANTIZATION_LEVELS])
        ax.grid()
    plt.tight_layout()
    plt.savefig(name + '_generated' + extra_name + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


def plot_curve(name, nll_val):
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('nll')
    plt.savefig(name + '_nll_val_curve.pdf', bbox_inches='tight')
    plt.close()

# %%
def training(name, max_patience, num_epochs, model: SimpleRNN, optimizer, training_loader, val_loader):
    nll_val = []
    best_nll = 1000.
    patience = 0

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        for indx_batch, batch in enumerate(training_loader):
            if hasattr(model, 'dequantization'):
                if model.dequantization:
                    batch = batch + torch.rand(batch.shape)
            loss = model.forward(batch)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Validation
        loss_val = evaluation(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_val)  # save for plotting

        if e == 0:
            print('saved!')
            torch.save(model, name + '.model')
            best_nll = loss_val
        else:
            if loss_val < best_nll:
                print('saved!')
                torch.save(model, name + '.model')
                best_nll = loss_val
                patience = 0

                # samples_generated(name, val_loader, extra_name="_epoch_" + str(e))
            else:
                patience = patience + 1

        if patience > max_patience:
            break

    nll_val = np.asarray(nll_val)

    return nll_val

# %% [markdown]
# ### Initialize dataloaders

# %%
# train_data = Digits(mode='train')
# val_data = Digits(mode='val')
# test_data = Digits(mode='test')

# training_loader = DataLoader(train_data, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
# test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# %% [markdown]
# ### Hyperparams
# %%

result_dir = 'results_fusion/'
if not(os.path.exists(result_dir)):
    os.mkdir(result_dir)
name = 'fusion_bigger'

D = 4000   # input dimension (time steps)
M = 256  # the number of neurons in scale (s) and translation (t) nets

lr = 1e-3 # learning rate
num_epochs = 100 # max. number of epochs
max_patience = 20 # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped

# %% [markdown]
# ### Initialize ARM

# %%
likelihood_type = 'categorical'

num_vals = QUANTIZATION_LEVELS

kernel = 7

net = nn.Sequential(
    CausalConv1d(in_channels=1, out_channels=M, dilation=1, kernel_size=kernel, A=True, bias=True),
    nn.LeakyReLU(),
    CausalConv1d(in_channels=M, out_channels=M, dilation=2*kernel, kernel_size=kernel, A=False, bias=True),
    nn.LeakyReLU(),
    CausalConv1d(in_channels=M, out_channels=M, dilation=4*kernel, kernel_size=kernel, A=False, bias=True),
    nn.LeakyReLU(),
    CausalConv1d(in_channels=M, out_channels=M, dilation=8*kernel, kernel_size=kernel, A=False, bias=True),
    nn.LeakyReLU(),
    CausalConv1d(in_channels=M, out_channels=num_vals, dilation=16*kernel, kernel_size=kernel, A=False, bias=True)
    )


# Print the summary (like in Keras)
print(summary(model, torch.zeros(1, 4000), show_input=False, show_hierarchical=False))

# %% [markdown]
# ### Let's play! Training

# %%
# OPTIMIZER
optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)

# %%
# Training procedure
nll_val = training(name=result_dir + name, max_patience=max_patience, num_epochs=num_epochs, model=model, optimizer=optimizer,
                       training_loader=dataloader, val_loader=dataloader)

# %%
test_loss = evaluation(model_path=result_dir + name, test_loader=dataloader)
f = open(result_dir + name + '_test_loss.txt', "w")
f.write(str(test_loss))
f.close()

samples_real(result_dir + name, dataloader)
samples_generated(result_dir + name, dataloader, extra_name="_conditioned", start_sequence_length=3000)

# plot_curve(result_dir + name, nll_val)

# %%



