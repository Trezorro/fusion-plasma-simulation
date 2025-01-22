from typing import Optional, Sequence
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch
from torch.utils import data
import random
from src.data_generators import create_gaussian_data, create_square_data, create_spiral_data, create_heart_data, create_two_gaussians_data, create_smiley_data
import logging


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


class SourceTargetDS(data.Dataset):

    DISTRIBUTION_OPTIONS = {
        'gaussian': create_gaussian_data,
        'two_gaussians': create_two_gaussians_data,
        'spiral': create_spiral_data,
        'square': create_square_data,
        'heart': create_heart_data,
        'smiley': create_smiley_data
    }

    def __init__(self, source_distribution: str, target_distribution: str, n=1000, **kwargs):
        super().__init__()
        self.source_distribution = source_distribution
        self.target_distribution = target_distribution
        self.n = n
        self.source_gen_fn = self.DISTRIBUTION_OPTIONS[source_distribution]
        self.target_gen_fn = self.DISTRIBUTION_OPTIONS[target_distribution]
        self.source_data = self.source_gen_fn(n)
        self.target_data = self.target_gen_fn(n)

    def regenerate_data(self):
        self.source_data = self.source_gen_fn(self.n)
        self.target_data = self.target_gen_fn(self.n)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.source_data[idx], self.target_data[idx]


class ShotFlowDS(data.Dataset):
    """

    Args:
        dir: Directory where the data file is located.
        file: Name of the data file.
        cols: OmegaConf DictConfig with keys 'c' and 'x' containing lists of column names.
        seq_length: Length of the sequence to extract from each shot.
        crop_margin: Minimum distance from the start and end of the shot to the start and end of the sequence.
        random_start: If True, the sequence will start at a random point within the shot.
        time_last: If True, the time dimension will be the last dimension of the output.
        force_mean_zero: If True, the mean of the sequence will be zero.
        force_fixed_shot: If not None, the dataset will always return the same shot. Useful for debugging by overfitting.
        force_start: If not None, the dataset will always return the same start timestep index. May be modified by random_start.
        **kwargs: Additional arguments that are not used.
    """

    def __init__(
        self,
        dir: str,
        file: str,
        cols: DictConfig,
        seq_length=200,
        crop_margin=1000,
        random_start: bool | list[int] = True,
        time_last=False,
        force_mean_zero=False,
        force_fixed_shot: Optional[int] = None,
        force_start: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.file_path = dir + file
        self.columns_C = list(cols.get('c', []))
        if self.columns_C:
            logging.warning("Warning: columns_C will not be used right now.")
        self.columns_X = list(cols.x)[:1]
        if len(cols.x) > 1:
            logging.warning("Only one column_X is supported right now. Will use the first one.")
        self.seq_length = seq_length
        self.crop_margin = crop_margin
        self.random_start = random_start
        self.time_last = time_last
        self.force_mean_zero = force_mean_zero
        self.force_fixed_shot = force_fixed_shot
        self.force_start = force_start
        assert random_start is not True or force_start is None, "Cannot have random_start and force_start at the same time, unless random_start is a list of int options."

        self.data = pd.read_parquet(self.file_path)
        self.data['ShotNum'] = self.data['ShotNum'].astype(
            np.int32
        )  # Reduce memory usage and quicker indexing
        self.shot_numbers = self.data['ShotNum'].unique()
        # Normalize within 0-1:
        self.min = self.data[self.columns_X].min()
        self.max = self.data[self.columns_X].max()
        self.data[self.columns_X] = (self.data[self.columns_X] - self.min) / (self.max - self.min)

    def __len__(self):
        return len(self.shot_numbers)

    def __getitem__(self, idx):
        if self.force_fixed_shot is not None and self.force_fixed_shot < len(self.shot_numbers):
            idx = self.force_fixed_shot

        shot_number = self.shot_numbers[idx]
        shot_data = self.data[self.data['ShotNum'] == shot_number]  # indexed by time
        shot_len = len(shot_data)  # Should be at minumum 5000 based on preprocessing
        viable_start_max = shot_len - self.crop_margin - self.seq_length
        assert viable_start_max > self.crop_margin, (
            f"Shot {shot_number} is too short (T{shot_len}) for desired "
            f"seq_length {self.seq_length} and crop_margin {self.crop_margin}"
        )
        if self.force_start is not None:
            start = self.force_start
            if isinstance(self.random_start, Sequence):
                start += random.choice(self.random_start)
        elif self.random_start:
            if isinstance(self.random_start, Sequence):
                start = self.crop_margin + random.choice(self.random_start)
            else:
                start = random.randint(self.crop_margin, viable_start_max)
        else:
            start = self.crop_margin
        end = start + self.seq_length
        X = shot_data[self.columns_X].iloc[start:end].values.T

        if self.force_mean_zero:
            X = X - X.mean(axis=1, keepdims=True)
            X = X / X.std(axis=1, keepdims=True)

        prior_distribution_sample = torch.randn(len(self.columns_X), self.seq_length)

        return prior_distribution_sample, X  # prior and X shapes: (variables, seq_length)
