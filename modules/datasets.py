"""Defines torch datasets that load data from files."""
import wandb
from wandb.sdk.wandb_run import Run
from torch.utils.data import Dataset, DataLoader
import pandas as pd


X_COL = "IP"
Y_COL = "FIR"
time_col = "time"
QUANTIZATION_LEVELS = 50

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
