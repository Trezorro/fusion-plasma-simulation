import numpy as np
from omegaconf import DictConfig
import pandas as pd
from torch.utils import data
import random


class ShotWindowDataset(data.Dataset):

    def __init__(
        self,
        dir: str,
        file: str,
        cols: DictConfig,
        seq_length=2000,
        crop_margin=1000,
        random_start=True,
        time_last=False,
        force_mean_zero=False,
        **kwargs
    ):
        super().__init__()
        self.file_path = dir + file
        self.columns_C = list(cols.c)
        self.columns_X = list(cols.x)
        self.seq_length = seq_length
        self.crop_margin = crop_margin
        self.random_start = random_start
        self.time_last = time_last
        self.force_mean_zero = force_mean_zero

        self.data = pd.read_parquet(self.file_path)
        self.data['ShotNum'] = self.data['ShotNum'].astype(
            np.int32
        )  # Reduce memory usage and quicker indexing
        self.shot_numbers = self.data['ShotNum'].unique()
        self.min = self.data[self.columns_C + self.columns_X].min()
        self.max = self.data[self.columns_C + self.columns_X].max()
        self.data[self.columns_C + self.columns_X] = (self.data[self.columns_C + self.columns_X] -
                                                      self.min) / (self.max - self.min)

    def __len__(self):
        return len(self.shot_numbers)

    def __getitem__(self, idx):
        shot_number = self.shot_numbers[idx]
        shot_data = self.data[self.data['ShotNum'] == shot_number]  # indexed by time
        shot_len = len(shot_data)  # Should be at minumum 5000 based on preprocessing
        viable_start_max = shot_len - self.crop_margin - self.seq_length
        assert viable_start_max > self.crop_margin, (
            f"Shot {shot_number} is too short (T{shot_len}) for desired "
            f"seq_length {self.seq_length} and crop_margin {self.crop_margin}"
        )
        start = random.randint(self.crop_margin, viable_start_max) if self.random_start else self.crop_margin
        end = start + self.seq_length
        C = shot_data[self.columns_C].iloc[start:end].values
        X = shot_data[self.columns_X].iloc[start:end].values
        if self.force_mean_zero:
            C = C - C.mean(axis=0)
            X = X - X.mean(axis=0)
        if self.time_last:
            C = C.transpose()
            X = X.transpose()

        return shot_number, C, X  # C and X shapes: (seq_length, variables)


class DummyDataSet(data.Dataset):
    """A data set that returns a sine wave translated up and down by a random value."""
    SAMPLE_RATE = 10000

    def __init__(
        self,
        n_columns_C,
        n_columns_X,
        seq_length=128,
        random_start=True,
        time_last=False,
        n_frequencies=1,
        aligned_frequencies=False,
        sample_rate=10000,
        **kwargs
    ):
        super().__init__()
        self.n_columns_C = n_columns_C
        self.n_columns_X = n_columns_X
        self.seq_length = seq_length
        self.random_start = random_start
        self.time_last = time_last
        self.n_frequencies = n_frequencies
        self.sample_rate = sample_rate
        self.aligned_frequencies = aligned_frequencies
        if self.aligned_frequencies:
            window_length = self.seq_length // 2
            sample_spacing = 1. / self.sample_rate
            self.freq_bins = np.fft.rfftfreq(window_length, d=sample_spacing)
        self.time_points = self.generate_timepoints()

    def __len__(self):
        if self.aligned_frequencies:
            return len(self.freq_bins)
        return 250

    def generate_timepoints(self):
        duration_secs = self.seq_length / self.sample_rate
        t = np.linspace(0, duration_secs, self.seq_length, endpoint=False)
        return t

    @staticmethod
    def get_sine_wave(t, frequency, amplitude=1.0, phase=0.0, y_offset=0.0):
        """Generate a sine wave with the given frequency and duration."""
        x = np.sin(2 * np.pi * (frequency * t + phase), dtype=np.float32) * amplitude + y_offset
        return x

    @staticmethod
    def get_multi_sine_wave(t, frequencies, amplitudes, phases, y_offsets):
        """Generate a sum of sine waves with the given frequencies and durations."""
        x = np.sum(
            [
                DummyDataSet.get_sine_wave(t, f, a, p, y)
                for f, a, p, y in zip(frequencies, amplitudes, phases, y_offsets)
            ],
            axis=0
        )
        return x

    def __getitem__(self, idx):
        if self.aligned_frequencies:
            idx = idx % len(self.freq_bins)
            frequency_x = self.freq_bins[idx]
            frequency_c = frequency_x + 10
            idx = frequency_x  # for logging and plotting
        else:
            frequency_c = idx + 10  # should maximally be 260, which gives about 3 waves in 128 time points
            frequency_x = idx * 10 + 1  # should maximally be 2500, which gives 30 waves in 128 time points
        C = np.array(
            [
                self.get_sine_wave(
                    self.time_points,
                    frequency_c,
                    amplitude=np.random.rand() * 0.5,
                    phase=np.random.rand() if self.random_start else 0
                ) for _ in range(self.n_columns_C)
            ]
        )
        if self.n_frequencies > 1:
            frequencies = [frequency_x * (i + 1) for i in range(self.n_frequencies)]
            amplitudes = np.random.rand(self.n_frequencies)
            phases = np.random.rand(self.n_frequencies) if self.random_start else np.zeros(self.n_frequencies)
            y_offsets = (np.random.rand(self.n_frequencies) / self.n_frequencies) / 10
            X = np.array(
                [
                    self.get_multi_sine_wave(self.time_points, frequencies, amplitudes, phases, y_offsets)
                    for _ in range(self.n_columns_X)
                ]
            )
        else:
            X = np.array(
                [
                    self.get_sine_wave(
                        self.time_points,
                        frequency_x,
                        amplitude=np.random.rand(),
                        phase=np.random.rand() if self.random_start else 0,
                        y_offset=np.random.rand()
                    ) for _ in range(self.n_columns_X)
                ]
            )
        if not self.time_last:
            C = C.transpose()
            X = X.transpose()

        return idx, C, X
