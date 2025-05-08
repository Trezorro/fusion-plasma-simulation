# %%
import sys
import pathlib
from pathlib import Path
# Add the notebook's directory to the Python path
project_dir = pathlib.Path().resolve()  # vscode executes in the workspace root
notebook_dir = project_dir / 'LDH_demo'
print(f"Project dir: {project_dir}")
print(f"Notebook dir: {notebook_dir}")
sys.path.append(str(notebook_dir))
sys.path

# %%
from model import FNOLSTM
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.interpolate import interp1d
import os

# %%
# global model settings
TW = 40
OFFSET_PRED = 20
STRIDE = 10

# %%
with open(notebook_dir/'train.txt', 'r') as f:
    train_shots = json.load(f)
with open(notebook_dir/'test.txt', 'r') as f:
    test_shots = json.load(f)

# %%
dataset_path = '/Users/milan/Code/fusion/data/LHD_labeled_TCV/'
signal_template = dataset_path + 'TCV_DATAno{shot}.parquet'
label_template = dataset_path + 'TCV_{shot}_apau_labeled.csv'


# %%
def create_input(sig, signal_list, stats):
    assert all([s in stats for s in signal_list])
    columns = []
    for s in signal_list:
        columns.append(torch.tensor((sig[s].values - stats[s]["mean"])/ stats[s]["sd"]))
    return torch.tensor(sig.time), torch.stack(columns)

# %%
def pred_sample_slidingwindow(model, t, x, device, tw, stride, offset_pred, i_start=0):
    model.eval()
    x = x.to(device)
    model.to(device)
    hidden = None
    i_max = x.shape[-1] - tw + 1  # max index for the sliding window
    y_preds = []
    y_gts = []
    y_times = []
    for k in range(i_start, i_max, stride):
        i_s = k # start index
        i_e = k + tw # end index
        i_time = k + tw - 1 - offset_pred
        with torch.no_grad():
            y_pred, hidden = model(x[:, i_s:i_e].unsqueeze(0), hidden)
        y_times.append(t[i_time])

        y_preds.append(y_pred.squeeze(0))
    y_preds = torch.stack(y_preds).argmax(dim=1).to('cpu')
    y_times = torch.tensor(y_times, device='cpu')

    return y_times, y_preds

# %%
def predict_and_plot(shot, model, signal_list, stats, title=None, file_out=None):
    if not os.path.exists(signal_template.format(shot=shot)):
        print(f"Missing parquet for #{shot}")
        return
    if not os.path.exists(label_template.format(shot=shot)):
        print(f"Missing csv for #{shot}")
        return

    sig = pd.read_parquet(signal_template.format(shot=shot))
    label = pd.read_csv(label_template.format(shot=shot))

    t_input, sig_input = create_input(sig, signal_list, stats)
    t_out, y_out = pred_sample_slidingwindow(model, t_input, sig_input, device='cpu', tw=TW, stride=STRIDE, offset_pred=OFFSET_PRED, i_start=0)

    fig, ax = plt.subplots()
    fig.set_size_inches((8, 3))
    ax.plot(sig.time, sig.PD, color='black', linewidth=.5)
    if title is not None:
        ax.set_title(title)

    LHD_valid = ~(label.LHD_label == 0)  # 0 labels make no sense, just interpolate to nearest as easy fix
    LHD_label = interp1d(label.time[LHD_valid], label.LHD_label[LHD_valid], bounds_error=False, fill_value=np.nan, kind='nearest')(label.time)

    ax2 = ax.twinx()
    ax2.plot(label.time, LHD_label, label='labels')
    ax2.plot(t_out, y_out + 1, linestyle='--', linewidth=2, alpha=1, label='prediction')  # labels in dataset are 1, 2, 3, model outputs are 0, 1, 2 --> adjust
    plt.legend()
    if file_out is not None:
        out_path = Path('output/mode_predictions') / file_out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), bbox_inches='tight')
        plt.close(fig)
        return
    plt.show()


# %%
#  PD model
model_PD = FNOLSTM(n_in=1, n_out=3, tw=TW,
               h_c1=32, h_c2=64, h_m1=8, h_m2=8, h_dropc=0.5, h_maxpool=2,
               h_lstm_in=32, h_lmst=32, h_mlp=8, m_dropmlp=0.5)
model_PD.load_state_dict(torch.load(notebook_dir/"models_milan/weights_PD.pt"))
with open(notebook_dir/'models_milan/stats_PD.json', 'r') as f:
    stats_PD = json.load(f)


def get_mode_predictions(pd_traces: torch.Tensor, timeline: np.ndarray):
    # dim should be (N, T)
    COLNAME = 'PD'
    normalized = torch.tensor((pd_traces - stats_PD[COLNAME]["mean"]) / stats_PD[COLNAME]["sd"])

    t_out, y_out = pred_sample_slidingwindow(
        model_PD, t=timeline, x=normalized, device='cpu', tw=TW, stride=STRIDE, offset_pred=OFFSET_PRED, i_start=0
    )
    return t_out, y_out
