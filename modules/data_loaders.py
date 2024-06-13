import pandas as pd
import wandb
import torch
from torch.utils import data

C = wandb.config


class MyDataset(data.Dataset):

    def __init__(self, file_path, columns_C, columns_X):
        self.data = pd.read_parquet(file_path)
        self.shot_numbers = self.data['ShotNum'].unique()
        self.columns_C = columns_C
        self.columns_X = columns_X
        self.min = self.data[columns_C + columns_X].min()
        self.max = self.data[columns_C + columns_X].max()
        self.data[columns_C + columns_X] = (self.data[columns_C + columns_X] - self.min) / (self.max - self.min)

    def __len__(self):
        return len(self.shot_numbers)

    def __getitem__(self, idx):
        shot_number = self.shot_numbers[idx]
        shot_data = self.data[self.data['ShotNum'] == shot_number] # indexed by time
        C = shot_data[self.columns_C].iloc[1000:3000].values
        X = shot_data[self.columns_X].iloc[1000:3000].values
        return shot_number, C, X # C and X shapes: (seq_length, variables)


def slice_data(data, start_time, end_time):
    return data[(data['Time'] >= start_time) & (data['Time'] <= end_time)]


# def normalize_data(data):
#     raw_sig = data.copy()
#     sig[ALL_SIG_COLLS] = (sig[ALL_SIG_COLLS] -
#                         sig[ALL_SIG_COLLS].mean()) / sig[ALL_SIG_COLLS].std()
#     return (data - data.mean()) / data.std()

