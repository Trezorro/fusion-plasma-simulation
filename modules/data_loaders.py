import pandas as pd
import wandb
import torch
from torch.utils.data import Dataset, DataLoader

C = wandb.config


class MyDataset(Dataset):

    def __init__(self, file_path, columns_C, columns_X):
        self.data = pd.read_parquet(file_path)
        self.shot_numbers = self.data['ShotNum'].unique()
        self.columns_C = columns_C
        self.columns_X = columns_X

    def __len__(self):
        return len(self.shot_numbers)

    def __getitem__(self, idx):
        shot_data = self.data[self.data['ShotNum'] == self.shot_numbers[idx]] # indexed by time
        C = shot_data[self.columns_C].iloc[1000:3000].values
        X = shot_data[self.columns_X].iloc[1000:3000].values
        return C, X


def slice_data(data, start_time, end_time):
    return data[(data['Time'] >= start_time) & (data['Time'] <= end_time)]


# def normalize_data(data):
#     raw_sig = data.copy()
#     sig[ALL_SIG_COLLS] = (sig[ALL_SIG_COLLS] -
#                         sig[ALL_SIG_COLLS].mean()) / sig[ALL_SIG_COLLS].std()
#     return (data - data.mean()) / data.std()


if __name__ == '__main__':
    dataset = MyDataset(file_path=C.data_dir + C.data_file,
                        columns_C=C.data_c_columns,
                        columns_X=C.data_x_columns)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for C, X in dataloader:
        # Perform operations on C and X
        # ...
        pass
