import pandas as pd
from torch.utils import data
import random


class MyDataset(data.Dataset):

    def __init__(self, file_path, columns_C, columns_X, seq_length=2000, crop_margin=1000, random_start=True):
        self.file_path = file_path
        self.columns_C = columns_C
        self.columns_X = columns_X
        self.seq_length = seq_length
        self.crop_margin = crop_margin
        self.random_start = random_start

        self.data = pd.read_parquet(file_path)
        self.shot_numbers = self.data['ShotNum'].unique()
        self.min = self.data[columns_C + columns_X].min()
        self.max = self.data[columns_C + columns_X].max()
        self.data[columns_C +
                  columns_X] = (self.data[columns_C + columns_X] - self.min) / (self.max - self.min)

    def __len__(self):
        return len(self.shot_numbers)

    def __getitem__(self, idx):
        shot_number = self.shot_numbers[idx]
        shot_data = self.data[self.data['ShotNum'] == shot_number]  # indexed by time
        shot_len = len(shot_data)  # Should be at minumum 5000 based on preprocessing
        viable_start_max = shot_len - self.crop_margin - self.seq_length
        assert viable_start_max > self.crop_margin, (
            f"Shot {shot_number} is too short (T{shot_len}) for desired "
            f"seq_length {self.seq_length} and crop_margin {self.crop_margin}")
        start = random.randint(self.crop_margin, viable_start_max) if self.random_start else self.crop_margin
        end = start + self.seq_length
        C = shot_data[self.columns_C].iloc[start:end].values
        X = shot_data[self.columns_X].iloc[start:end].values
        return shot_number, C, X  # C and X shapes: (seq_length, variables)


def slice_data(data, start_time, end_time):
    return data[(data['Time'] >= start_time) & (data['Time'] <= end_time)]


# def normalize_data(data):
#     raw_sig = data.copy()
#     sig[ALL_SIG_COLLS] = (sig[ALL_SIG_COLLS] -
#                         sig[ALL_SIG_COLLS].mean()) / sig[ALL_SIG_COLLS].std()
#     return (data - data.mean()) / data.std()
