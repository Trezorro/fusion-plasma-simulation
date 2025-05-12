from typing import List, Optional, Sequence
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch
from torch.utils import data
import random

import wandb
from src.data_generators import create_gaussian_data, create_square_data, create_spiral_data, create_heart_data, create_two_gaussians_data, create_smiley_data
import logging

logger = logging.getLogger(__name__)


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
        self.data['ShotNum'] = self.data['ShotNum'].astype(np.int32)  # Reduce memory usage and quicker indexing
        self.shot_numbers = self.data['ShotNum'].unique()
        self.min = self.data[self.columns_C + self.columns_X].min()
        self.max = self.data[self.columns_C + self.columns_X].max()
        self.data[self.columns_C +
                  self.columns_X] = (self.data[self.columns_C + self.columns_X] - self.min) / (self.max - self.min)

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
    """Just for testing flow matching with simple 2d data sets."""

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
    
    Returns:
        meta, conditioning and x window for a single shot.
            meta: Dictionary with keys 'shot_number', 'start', 'end'.
            conditioning: Dictionary with optional keys 'x_history' containing the history data.
            x: Selected window of observables from the shot.
        
    Args:
        dir: Directory where the data file is located.
        file: Name of the data file.
        cols: OmegaConf DictConfig with keys 'c' and 'x' containing lists of column names.
        seq_length: Length of the sequence to extract from each shot.
        crop_margin: Minimum distance from the start and end of the shot to the start and end of the sequence.
        random_start: If True, the sequence will start at a random point within the shot.
        time_last: If True, the time dimension will be the last dimension of the output.
        force_mean_zero: If True, the mean of the sequence will be zero. [DEPRECATED]
            (Caused division by zero when window was constant for a sensor)
        force_fixed_shot: If not None, the dataset will always return the same shot. Useful for debugging by overfitting.
        force_start: If not None, the dataset will always return the same start timestep index. May be modified by random_start.
        **kwargs: Additional arguments that are not used.
    """

    def __init__(
        self,
        dir: str,
        file: str,
        cols: DictConfig,
        train_shots: list[int],
        test_shots: list[int],
        seq_length=200,
        crop_margin=0,
        history_length: Optional[int] = 0,  # 0 or None, when no history conditioning is used.
        allowed_start_indices: Optional[list[int]] = None,
        # overfitting helpers:
        overfit_on_shots: Optional[list[int]] = None,
        train=True,
        pre_shuffle=True,
        **kwargs
    ):
        super().__init__()
        self.file_path = dir + file
        self.columns_C = list(cols.get('c', []))
        self.columns_X = list(cols.x)
        self.label = cols.get('label', None)
        self.seq_length = seq_length
        self.history_length = history_length or 0
        self.crop_margin = crop_margin or 0
        self.overfit_on_shots = overfit_on_shots
        self.allowed_start_indices = allowed_start_indices
        self.train = train
        self.shot_numbers = train_shots if train else test_shots
        self.pre_shuffle = pre_shuffle

        assert self.crop_margin >= self.history_length, "crop_margin must be greater than or equal to history_length to provide enough context."

        self.load_and_filter_data()
        self.normalize_columns()
        self.precompute_indices()

    def precompute_indices(self):
        """Precompute all viable sample indices for each shot."""
        self.viable_indices = []
        viable_shots = set()
        if self.overfit_on_shots:
            # Overfit on a specific set of shots
            self.shot_numbers = self.overfit_on_shots
        for shot_number in self.shot_numbers:
            shot_data = self.data[self.data['ShotNum'] == shot_number]
            shot_len = len(shot_data)
            viable_start_max = shot_len - self.crop_margin - self.seq_length
            if viable_start_max < self.crop_margin:
                logger.warning(
                    f"Shot {shot_number} is too short (T{shot_len}) for desired "
                    f"seq_length {self.seq_length} and crop_margin {self.crop_margin}"
                )
                continue  # Skip this shot if it's too short
            if self.allowed_start_indices:
                # Use the allowed start indices if provided
                for start_idx in self.allowed_start_indices:
                    if self.crop_margin <= start_idx and start_idx + self.seq_length <= viable_start_max:
                        self.viable_indices.append((shot_number, start_idx))
                        viable_shots.add(shot_number)
            else:
                # Use all possible start indices within the viable range and crop margin
                for start_idx in range(self.crop_margin, viable_start_max + 1):
                    self.viable_indices.append((shot_number, start_idx))
                    viable_shots.add(shot_number)
        logger.info(f"Precomputed {len(self.viable_indices)} viable samples across all shots.")
        logger.info(f"Included {len(viable_shots)} shots of {len(self.shot_numbers)} specified, in the dataset.")
        self.shot_numbers = list(viable_shots)
        if self.pre_shuffle:
            random.shuffle(self.viable_indices)

    def load_and_filter_data(self):
        self.data = pd.read_parquet(self.file_path)
        # Reduce memory usage and quicker indexing:
        self.data['ShotNum'] = self.data['ShotNum'].astype(np.int32)
        self.data = self.data[self.data['ShotNum'].isin(self.shot_numbers)]
        # Fill NaNs with forward fill
        self.data = self.data.ffill()
        logger.info(f"Using {len(self.shot_numbers)} shots from {self.file_path}")
        if "DML-r" in self.columns_X:
            logger.info("Adding reversed DML to data.")
            self.data["DML-r"] = self.data["DML"] * -1

    def normalize_columns(self):
        """Normalize within 0-1"""
        target_cols = self.columns_X + self.columns_C
        self.min = self.data[target_cols].min()
        self.max = self.data[target_cols].max()
        self.data[target_cols] = (self.data[target_cols] - self.min) / (self.max - self.min)
        # Prepare these for denormalize method (used for downstream models):
        self.max_vals_x: np.ndarray = self.max[self.columns_X].values[..., np.newaxis]
        self.min_vals_x: np.ndarray = self.min[self.columns_X].values[..., np.newaxis]

        # log the min and max of the target cols to wandb config
        stats_key = 'stats_' + ("train" if self.train else "val")
        wandb.run.config['data'] |= {stats_key: {'min': self.min.to_dict(), 'max': self.max.to_dict()}}
        # Summarize statistics per column
        for column in self.data.columns:
            col_data = self.data[column]
            logger.info(
                "Column '%-15s': min=%-11.4f max=%-11.4f mean=%-11.4f std=%-11.4f nans=%-10d", column, col_data.min(),
                col_data.max(), col_data.mean(), col_data.std(),
                col_data.isna().sum()
            )

    def denormalize(self, x: np.ndarray | torch.Tensor):
        """Makes a copy of the input and denormalizes it from [0,1] to original."""
        if isinstance(x, torch.Tensor):
            x = x.clone()
        elif isinstance(x, np.ndarray):
            x = torch.tensor(x)
        else:
            raise TypeError(f"Unsupported type {type(x)}. Expected np.ndarray or torch.Tensor.")
        x = (x * (self.max_vals_x - self.min_vals_x)) + self.min_vals_x
        return x

    def __len__(self):
        return len(self.viable_indices)

    def get_full_history(self, shot_number: int | Sequence[int],
                         start_i: int | Sequence[int]) -> np.ndarray | List[np.ndarray]:
        """Get the full history of a shot up to the start index.

        Supports both scalar input as well as batched input. Returns (*input shape, T).
        """
        if isinstance(shot_number, (Sequence, torch.Tensor)) and isinstance(start_i, (Sequence, torch.Tensor)):
            assert len(shot_number) == len(
                start_i
            ), f"shot_number and start_i must have the same length. Got {len(shot_number)} and {len(start_i)}."
            shots = shot_number
            starts = start_i
            histories = []
            for shot, start in zip(shots, starts):
                shot_data = self.data[self.data['ShotNum'] == shot]
                full_history = shot_data[self.columns_X].iloc[:start].values.T
                histories.append(full_history)
            return histories
        elif isinstance(shot_number, int) and isinstance(start_i, int):
            shot_data = self.data[self.data['ShotNum'] == shot_number]
            return shot_data[self.columns_X].iloc[:start_i].values.T
        else:
            raise TypeError(
                f"Unsupported type {type(shot_number)=} and {type(start_i)=}. Expected int or Sequence[int]."
            )

    def __getitem__(self, idx):
        shot_number, start_i = self.viable_indices[idx]
        shot_data = self.data[self.data['ShotNum'] == shot_number]
        end_i = start_i + self.seq_length
        x = shot_data[self.columns_X].iloc[start_i:end_i].values.T
        meta = {
            'shot_number': shot_number,
            'start': shot_data.index[start_i],  # future window start time
            'end': shot_data.index[end_i],  # future window end time
            # 'full_history': full_history,  # full history of the shot up until the start of the future window
            'full_history_start': shot_data.index[0],  # shots don't always start at time 0
            'start_i': start_i,
            'end_i': end_i,
            # history end is the start of the current window
        }
        conditioning_input = {}

        if self.history_length:  # if we are conditioning on X history
            history_start = start_i - self.history_length
            assert history_start >= 0, "Not enough history data available."
            meta['history_start'] = shot_data.index[history_start]
            meta['history_start_i'] = history_start
            history_end_i = start_i
            x_history = shot_data[self.columns_X].iloc[history_start:history_end_i].values.T
            conditioning_input['x_history'] = x_history
            start_i = history_start

        conditioning_input['position_sequence'] = shot_data.index[start_i:end_i].values.astype(np.float32)
        if self.columns_C:
            conditioning_input['c'] = shot_data[self.columns_C].iloc[start_i:end_i].values.T
        if self.label:
            conditioning_input['label'] = shot_data[self.label].iloc[start_i:end_i].values.T
        assert not np.isnan(x).any(
        ), "NaNs found in the output window. Often caused by division by zero in normalization."
        return meta, conditioning_input, x
