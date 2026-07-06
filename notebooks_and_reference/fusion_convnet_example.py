"""Autoregressive causal-conv (PixelCNN-style) baseline for single-channel TCV signal, adapted from Tomczak's Deep Generative Models Digits example. Prototype/legacy, unmaintained.

Inputs/Outputs: reads /Users/milan/Code/fusion/experiments/shots/TCV_DATA*clean.parquet and TCV_*_apau_labeled.csv; writes model checkpoints/PDFs under results_fusion/.
Handy: CausalConv1d and the ARM (log_categorical + autoregressive sampling) class are reusable building blocks, though superseded by the flow matching model in src/models/.

Orignal code by Jakub Tomczak for the book Deep Generative Models
"""
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
# **DISCLAIMER**
# 
# The presented code is not optimized, it serves an educational purpose. It is written for CPU, it uses only fully-connected networks and an extremely simplistic dataset. However, it contains all components that can help to understand how an autoregressive model (ARM) works, and it should be rather easy to extend it to more sophisticated models. This code could be run almost on any laptop/PC, and it takes a couple of minutes top to get the result.

# %% [markdown]
# ### Dataset

# %% [markdown]
# In this example, we go wild and use a dataset that is simpler than MNIST! We use a scipy dataset called Digits. It consists of ~1500 images of size 8x8, and each pixel can take values in $\{0, 1, \ldots, 16\}$.
# 
# The goal of using this dataset is that everyone can run it on a laptop, without any gpu etc.

# %%
data_dir = "/Users/milan/Code/fusion/experiments/shots/"
sig_all_names = glob.glob(data_dir + 'TCV_DATA*clean.parquet')
# sig_all = {int(x.split("DATAno")[1].split("clean.parquet")[0]): x for x in sig_all_names} # get sample number
# use regex to get sample number:
sig_all = {int(re.findall(r'\d+', x)[0]): x for x in sig_all_names}

label_all = glob.glob(data_dir + 'TCV_*_apau_labeled.csv')
label_all = {int(x.split("TCV_")[1].split("_apau_labeled.csv")[0]): x for x in label_all}
shot_no_list = list(sig_all.keys())

print(f"All shots: {shot_no_list}")
print("Amount of shots: ", len(shot_no_list))

# %%
# example
shotno = shot_no_list[0]

sig = pd.read_parquet(sig_all[shotno])
label = pd.read_csv(label_all[shotno])
sig.columns[0:40]


# %%
sig

# %%
for i, shotno in enumerate(shot_no_list):
    sig = pd.read_parquet(sig_all[shotno])
    print(f"[{i+1}] Shot {shotno} has {len(sig)} timesteps from {sig['time'].min()} to {sig['time'].max()}")
# %%
X_COL = "IP"
Y_COL = "FIR"
time_col = "time"
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
        self.calculate_y_stats()

    def calculate_y_stats(self):
        """Set y_mean and y_std based on all y values in the dataset.

        Improvements:
            - [ ] Use the same temporal window to normalize
        """
        y_values = []
        self.y_min = float('inf')
        self.y_max = -float('inf')
        for shotno in self.shot_no_list:
            sig = pd.read_parquet(self.sig_all[shotno])
            y = sig[Y_COL].values
            # update min and max
            if np.min(y) < self.y_min:
                self.y_min = np.min(y)
            if np.max(y) > self.y_max:
                self.y_max = np.max(y)
        print(f"y_min: {self.y_min}, y_max: {self.y_max}")

    def normalize_y(self, y): # TODO: do much smarter quantization
        normalized = (y - self.y_min) / (self.y_max - self.y_min)
        quantized = (normalized * (QUANTIZATION_LEVELS - 1))
        return quantized

    def __len__(self):
        return len(self.shot_no_list)

    def __getitem__(self, idx):
        shotno = self.shot_no_list[idx]

        sig = pd.read_parquet(self.sig_all[shotno])

        time_steps = len(sig)
        start = time_steps // 2 - 2000
        end = start + 4000

        x = sig[X_COL].values.astype(np.float32)[start:end]
        y = sig[Y_COL].values.astype(np.float32)[start:end]

        if self.transforms:
            x = self.transforms(x)

        y = self.normalize_y(y)

        return y
# %%

#%%

data_dir = "/Users/milan/Code/fusion/experiments/shots/"
sig_all_names = glob.glob(data_dir + 'TCV_DATA*clean.parquet')
sig_all = {int(re.findall(r'\d+', x)[0]): x for x in sig_all_names}

label_all = glob.glob(data_dir + 'TCV_*_apau_labeled.csv')
label_all = {int(x.split("TCV_")[1].split("_apau_labeled.csv")[0]): x for x in label_all}

dataset = ParquetDataset(data_dir, sig_all, label_all)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


#%%
def plot_batch(batch: torch.Tensor):
    x = batch.detach().numpy()
    fig, ax = plt.subplots(1, 1)
    # set y limits
    ax.set_ylim([0, QUANTIZATION_LEVELS])
    ax.grid(True)
    for i, timeseries in enumerate(batch):
        ax.plot(timeseries, label=f"{i}")
        ax.set_ylabel(Y_COL)
    plt.show()

    # plt.savefig('test.pdf', bbox_inches='tight')
    plt.close()



plot_batch(next(iter(dataloader)))
# %% Example digits data set from Jakub's book
class Digits(Dataset):
    """Scikit-Learn Digits dataset."""

    def __init__(self, mode='train', transforms=None):
        digits = load_digits()
        if mode == 'train':
            self.data = digits.data[:1000].astype(np.float32)
        elif mode == 'val':
            self.data = digits.data[1000:1350].astype(np.float32)
        else:
            self.data = digits.data[1350:].astype(np.float32)

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transforms:
            sample = self.transforms(sample)
        return sample

# %% [markdown]
# ### ARM code

# %% [markdown]
# Please see the blogpost for details.

# %%
class CausalConv1d(nn.Module):
    """
    A causal 1D convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, A=False, **kwargs):
        super(CausalConv1d, self).__init__()

        # attributes:
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.A = A
        
        self.padding = (kernel_size - 1) * dilation + A * 1

        # module:
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels,
                                      kernel_size, stride=1,
                                      padding=0,
                                      dilation=dilation,
                                      **kwargs)

    def forward(self, x):
        # We do padding only from the left! This is more efficient implementation.
        x = torch.nn.functional.pad(x, (self.padding, 0))
        conv1d_out = self.conv1d(x)
        if self.A:
            # Remember , we cannot be dependent on the current component; therefore, the last element is removed
            return conv1d_out[:, :, : -1]
        else:
            return conv1d_out

# %%
EPS = 1.e-5

def log_categorical(x, p, num_classes=256, reduction=None, dim=None):
    """
    Compute the logarithm of categorical cross-entropy loss.

    Args:
        x (Tensor): The target tensor.
        p (Tensor): The predicted probabilities tensor.
        num_classes (int): The number of classes.
        reduction (str): The reduction method. Options are 'avg', 'sum', or None.
        dim (int): The dimension along which to perform the reduction.

    Returns:
        Tensor: The logarithm of categorical cross-entropy loss.

    """
    x_one_hot = F.one_hot(x.long(), num_classes=num_classes)
    log_p = x_one_hot * torch.log(torch.clamp(p, EPS, 1. - EPS))
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

# %% Auto Regressive Model
class ARM(nn.Module):
    def __init__(self, net, D=1000, num_vals=256):
        super(ARM, self).__init__()

        print('ARM by JT.')

        self.net = net
        # This is how many values a pixel can take.
        self.num_vals = num_vals
        # This is the problem dimentionality (the number of pixels)
        self.D = D

    def f(self, x):
        # First, we apply causal convolutions.
        h = self.net(x.unsqueeze(1))
        # In channels , we have the number of values. Therefore, we change the order of dims.
        h = h.permute(0, 2, 1)
        # We apply softmax to calculate probabilities.
        p = torch.softmax(h, 2)
        return p
        
    def forward(self, x, reduction='avg'):
        if reduction == 'avg':
            return -(self.log_prob(x).mean())
        elif reduction == 'sum':
            return -(self.log_prob(x).sum())
        else:
            raise ValueError('reduction could be either `avg` or `sum`.')

    def log_prob(self, x):
        mu_d = self.f(x)
        log_p = log_categorical(x, mu_d, num_classes=self.num_vals, reduction='sum', dim=-1).sum(-1)

        return log_p

    def sample(self, batch_size, start_sequences=None):
        x_new = torch.zeros((batch_size, self.D))
        prediction_start = 0
        if start_sequences is not None:
            prediction_start = start_sequences.shape[1]
            x_new[:, :prediction_start] = start_sequences

        for d in tqdm.trange(prediction_start, self.D, desc='sampling', leave=False):
            p = self.f(x_new)
            x_new_d = torch.multinomial(p[:, d, :], num_samples=1)
            x_new[:, d] = x_new_d[:,0]

        return x_new

# %% [markdown]
# ### Auxiliary functions: training, evaluation, plotting

# %% [markdown]
# It's rather self-explanatory, isn't it?

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
        ax.set_ylabel(Y_COL)
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
        ax.set_ylabel(Y_COL)
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
def training(name, max_patience, num_epochs, model: ARM, optimizer, training_loader, val_loader):
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

model = ARM(net, D=D, num_vals=num_vals)

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
