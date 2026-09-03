from typing import List, Optional, Sequence
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch
from torch.utils import data
import random
import lightning as L
from torch.utils.data import default_collate
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


class FusionShotDataset(data.Dataset):
    """Used to construct the respective train, val and test data sets.
    
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

    Note:
        The PD column (H-alpha divertor photodiode) is required by the mode classifier
        in src/metrics/evaluate_modes.py and is looked up by name via
        C.data.cols.x.index("PD"). Renaming or removing PD from cols.x will cause
        a runtime error during mode metric computation.
    """

    def __init__(
        self,
        subset_df: pd.DataFrame,
        cols: DictConfig,
        seq_length=200,
        crop_margin=0,
        history_length: Optional[int] = 0,  # 0 or None, when no history conditioning is used.
        allowed_start_indices: Optional[list[int]] = None,
        pre_shuffle=True,
        name="Train",
        **kwargs
    ):
        super().__init__()
        self.data: pd.DataFrame = subset_df
        self.columns_C = list(cols.get('c', []))
        self.columns_X = list(cols.x)
        self.label = cols.get('label', None)
        self.seq_length = seq_length
        self.history_length = history_length or 0
        self.crop_margin = crop_margin or 0
        self.allowed_start_indices = allowed_start_indices
        self.pre_shuffle = pre_shuffle
        self.name = name

        assert self.crop_margin >= self.history_length, "crop_margin must be greater than or equal to history_length to provide enough context."
        self.precompute_indices()

    def __len__(self):
        return len(self.viable_indices)

    def precompute_indices(self):
        """Precompute all viable sample indices, indexing each possible window of each shot."""
        self.viable_indices = []
        self.shot_numbers = self.data['ShotNum'].unique()
        viable_shots = set()
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
                # Test set uses stride=10 to cap index count; at stride=1 the test set has 2M+ viable windows
                stride = 10 if self.name == "Test" else 1
                # Use all possible start indices within the viable range and crop margin
                for start_idx in range(self.crop_margin, viable_start_max + 1, stride):
                    self.viable_indices.append((shot_number, start_idx))
                    viable_shots.add(shot_number)
        logger.info(f"{self.name} set: Precomputed {len(self.viable_indices)} viable samples across all shots.")
        logger.info(f"Included {len(viable_shots)} shots of {len(self.shot_numbers)} specified, in the dataset.")
        self.shot_numbers = list(viable_shots)
        if self.pre_shuffle and self.name != "Test":
            random.shuffle(self.viable_indices)
        elif self.name == "Test":
            # Test set is sorted by start index for deterministic and cache-friendly ordering
            self.viable_indices.sort(key=lambda x: x[1])
            logger.info(f"Sorted test set on time index.")

    def __getitem__(self, idx):
        shot_number, start_i = self.viable_indices[idx]
        return self.get_shot_window(shot_number, start_i)

    def get_shot_window(self, shot_number, start_i):
        """Given a window W_i,j, provide X, C and Y together with metadata for plotting in a tuple."""
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

        # raw time index in seconds, deliberately NOT normalized; the positional embedding's max_value=2.0 spans this range
        conditioning_input['position_sequence'] = shot_data.index[start_i:end_i].values.astype(np.float32)
        if self.columns_C:
            conditioning_input['c'] = shot_data[self.columns_C].iloc[start_i:end_i].values.T
        if self.label:
            conditioning_input['label'] = shot_data[self.label].iloc[start_i:end_i].values.T
        assert not np.isnan(x).any(
        ), "NaNs found in the output window. Often caused by division by zero in normalization."
        return meta, conditioning_input, x

    def quick_window(self, shot_number, time, limit_to_idx: Optional[Sequence] = None, repeat=1):
        """Convenience function for plotting a specific window Wh and Wf around time t. """
        shot_data_index = self.data[self.data['ShotNum'] == shot_number].index
        if len(shot_data_index) == 0:
            return None
        start_idx = shot_data_index.get_indexer([time], method='nearest')[0]
        logger.info("Shot %s at t=%s is at start index %s", shot_number, time, start_idx)
        if limit_to_idx is not None:
            start_idx = limit_to_idx[pd.Index(limit_to_idx).get_indexer([start_idx], method='nearest')[0]]
            logger.info(
                "Adjusted to available indeces, we get idx %s with time %.4f", start_idx, shot_data_index[start_idx]
            )
        sample = self.get_shot_window(shot_number, start_idx)
        return default_collate([sample] * repeat)

    def window_set_batch(self, shot_t, repeat=4):
        """Build a single collated batch from every window in a window_set.

        For each [shot, time] pair, resolves the nearest available index (same logic
        as quick_window) and appends the resolved window `repeat` times. Missing shots
        are skipped (mirrors quick_window's None behaviour). The `repeat` copies share
        conditioning but get different stochastic priors at evaluation time, so their
        generated samples / trajectories differ; window `w` occupies collated rows
        [w*repeat : (w+1)*repeat].

        Args:
            shot_t: Iterable of [shot_number, time_seconds] pairs (e.g. config.window_set).
            repeat: Number of copies per window (stochastic samples to animate).

        Returns:
            A collated batch (meta, conditioning_input, target_samples) over all windows.
        """
        samples = []
        for shot, t in shot_t:
            shot_data_index = self.data[self.data['ShotNum'] == shot].index
            if len(shot_data_index) == 0:
                logger.warning("Shot %s not present in dataset, skipping in window_set_batch", shot)
                continue
            start_idx = shot_data_index.get_indexer([t], method='nearest')[0]
            logger.info("Window set: shot %s at t=%s is at start index %s", shot, t, start_idx)
            sample = self.get_shot_window(shot, start_idx)
            samples.extend([sample] * repeat)
        return default_collate(samples)


class FusionShotDataModule(L.LightningDataModule):
    """
    Lightning DataModule for TCV plasma shot data.

    Loads a combined parquet file of TCV shots, normalizes signal columns to [0,1]
    using train-split statistics, and constructs train/val/test FusionShotDataset
    instances. Shot assignments are fixed lists (not random splits) for reproducibility.

    Column names in cols.x, cols.c, and cols.label must exactly match the parquet
    column names. There is no validation at load time; a mismatch causes a KeyError
    or silent NaN values during normalization.

    Args:
        dir: Directory containing the parquet file.
        file: Parquet filename.
        cols: OmegaConf DictConfig with keys 'x' (observable channels), 'c' (conditioning
            channels), 'label' (mode label column), and 'meta' (metadata columns).
        train_shots: Shot numbers to use for training.
        val_shots: Shot numbers to use for validation.
        test_shots: Shot numbers to use for testing.
        batch_size: Batch size for all DataLoaders.
        seq_length: Length of the future window Wf in samples.
        crop_margin: Minimum distance from shot boundaries to any window start;
            must be >= history_length to provide enough context for x_history.
        history_length: Length of the history window Wh. 0 disables x_history conditioning.
        allowed_start_indices: If set, restricts sampling to only these start indices.
        overfit_on_shots: If set, restricts all splits to these shot numbers (for debugging).
        pre_shuffle: Shuffle training and validation index lists at init. Test set is
            always sorted by index regardless of this flag.
    """

    def __init__(
        self,
        dir: str,
        file: str,
        cols: DictConfig,  # x, c, label
        train_shots: list[int],
        val_shots: list[int],
        test_shots: list[int],
        batch_size: int,
        seq_length=64,
        crop_margin=64,
        history_length: Optional[int] = 0,
        allowed_start_indices: Optional[list[int]] = None,
        overfit_on_shots: Optional[list[int]] = None,
        pre_shuffle=True,
        **kwargs
    ):
        super().__init__()
        self.dir = dir
        self.file = file
        self.cols = cols
        self.train_shots = train_shots
        self.val_shots = val_shots
        self.test_shots = test_shots
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.crop_margin = crop_margin or 0
        self.history_length = history_length or 0
        assert self.crop_margin >= self.history_length, "crop_margin must be greater than or equal to history_length to provide enough context."

        self.allowed_start_indices = allowed_start_indices
        self.overfit_on_shots = overfit_on_shots
        self.pre_shuffle = pre_shuffle  # overridden by test set

        self.data_is_normalized = False
        self.num_workers = 2 if torch.cuda.is_available() else 0

    def prepare_data(self):
        """Load the parquet file, forward-fill NaN values, and create any derived columns.

        Forward-fill (ffill) handles gaps in sensor readings by propagating the last
        valid measurement forward. Derived columns (DML-r, NBI-median, ECRH-median)
        are created only if their names appear in cols.x or cols.c.
        """
        self.data = pd.read_parquet(self.dir + self.file)
        self.data_is_normalized = False
        self.data['ShotNum'] = self.data['ShotNum'].astype(np.int32)
        self.data['LHD_label'] = self.data['LHD_label'] + 1 # Hardcoded compatibility shortcut for new data set (which has labels L: 0, D: 1, H, 2)
        self.data = self.data.ffill()
        logger.info(f"Loaded {self.data['ShotNum'].nunique()} shots from {self.file}")
        # add artificial columns if found in the config:
        if "DML-r" in self.cols.x:
            self.data["DML-r"] = self.data["DML"] * -1
        if "NBI-median" in self.cols.get('c', []):
            self.data["NBI-median"] = self.data.groupby('ShotNum')["NBI"].transform(
                lambda x: x.rolling(100, min_periods=1, center=True).median()
            ).astype(np.float32)
        if "ECRH-median" in self.cols.get('c', []):
            self.data["ECRH-median"] = self.data.groupby('ShotNum')["ECRH"].transform(lambda x: x.rolling(100, min_periods=1, center=True).median()).astype(np.float32)

    def setup(self, stage: Optional[str] = None):
        """Split data into train, validation, and test sets and compute normalization stats."""
        self.normalize_xc_data()
        train_df = self.data[self.data['ShotNum'].isin(self.train_shots)]
        # Todo check if values train_df change
        self.train_dataset = FusionShotDataset(
            train_df,
            cols=self.cols,
            seq_length=self.seq_length,
            crop_margin=self.crop_margin,
            history_length=self.history_length,
            allowed_start_indices=self.allowed_start_indices,
            pre_shuffle=self.pre_shuffle,
            name='Train'
        )

        assert self.data_is_normalized, "Normalization stats not computed. Call setup() first."
        val_df = self.data[self.data['ShotNum'].isin(self.val_shots)]
        self.val_dataset = FusionShotDataset(
            val_df,
            cols=self.cols,
            seq_length=self.seq_length,
            crop_margin=self.crop_margin,
            history_length=self.history_length,
            allowed_start_indices=self.allowed_start_indices,
            pre_shuffle=self.pre_shuffle,
            name='Val'
        )

        assert self.data_is_normalized, "Normalization stats not computed. Call setup() first."
        test_df = self.data[self.data['ShotNum'].isin(self.test_shots)]
        self.test_dataset = FusionShotDataset(
            test_df,
            cols=self.cols,
            seq_length=self.seq_length,
            crop_margin=self.crop_margin,
            history_length=self.history_length,
            allowed_start_indices=self.allowed_start_indices,
            pre_shuffle=self.pre_shuffle,
            name='Test'
        )

    def normalize_xc_data(self):
        """Normalize X and C columns to [0, 1] using min/max computed from the TRAIN split only.

        Computing stats from the train split prevents data leakage. The same min/max
        is applied to val and test splits. Stats are logged to wandb config under
        data.train_stats and stored as GPU tensors for fast denormalize() calls.
        """
        assert not self.data_is_normalized, "Data already normalized. Call setup() first."
        target_cols = list(self.cols.x + self.cols.get('c', []))
        train_df = self.data[self.data['ShotNum'].isin(self.train_shots)]
        self.min = train_df[target_cols].min()
        self.max = train_df[target_cols].max()
        logger.info(f"Normalizing columns {target_cols} with min {self.min} and max {self.max}")
        self.data[target_cols] = ((self.data[target_cols] - self.min) / (self.max - self.min))
        self.data_is_normalized = True
        if wandb.run is not None:
            wandb.run.config['data'] |= {'train_stats': {'min': self.min.to_dict(), 'max': self.max.to_dict()}}
        # Prepare these for denormalize method (used for downstream models):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_vals_x = torch.tensor(self.max[self.cols.x].values, device=device).unsqueeze(-1)
        self.min_vals_x = torch.tensor(self.min[self.cols.x].values, device=device).unsqueeze(-1)

        # log the min and max of the target cols to wandb config
        # Summarize statistics per column
        for column in self.data.columns:
            col_data = self.data[column]
            logger.info(
                "Column '%-15s': min=%-11.4f max=%-11.4f mean=%-11.4f std=%-11.4f nans=%-10d", column, col_data.min(),
                col_data.max(), col_data.mean(), col_data.std(),
                col_data.isna().sum()
            )

    def denormalize(self, x: np.ndarray | torch.Tensor, to_device: Optional[str]=None):
        """Makes a copy of the input and denormalizes it from [0,1] to original."""
        if isinstance(x, torch.Tensor):
            x = x.clone()
        elif isinstance(x, np.ndarray):
            x = torch.tensor(x)
        else:
            raise TypeError(f"Unsupported type {type(x)}. Expected np.ndarray or torch.Tensor.")
        if to_device is not None:
            x = x.to(to_device)
        min_vals = self.min_vals_x.to(x.device)
        max_vals = self.max_vals_x.to(x.device)
        # logger.debug("Devices: x - %s, self.max_vals_x %s", x.device, self.max_vals_x.device)
        x = (x * (max_vals - min_vals)) + min_vals
        return x

    def get_full_history(self, shot_number: int | Sequence[int],
                         start_i: int | Sequence[int]) -> np.ndarray | List[np.ndarray]:
        """Get the full history of a shot up to the start index.

        Supports both scalar input as well as batched input. Returns (*input shape, T).

        Note: Returns data in the normalized [0,1] space; call denormalize() before
        passing to downstream models that expect physical units.
        """
        if isinstance(shot_number, (torch.Tensor)) and isinstance(start_i, (torch.Tensor)):
            assert len(shot_number) == len(
                start_i
            ), f"shot_number and start_i must have the same length. Got {len(shot_number)} and {len(start_i)}."
            shots = shot_number
            starts = start_i
            histories = []
            for shot, start in zip(shots, starts):
                shot_data = self.data[self.data['ShotNum'] == shot.item()]
                full_history = shot_data[self.cols.x].iloc[:start.item()].values.T
                histories.append(full_history)
            return histories
        elif isinstance(shot_number, int) and isinstance(start_i, int):
            shot_data = self.data[self.data['ShotNum'] == shot_number]
            return shot_data[self.cols.x].iloc[:start_i].values.T
        else:
            raise TypeError(
                f"Unsupported type {type(shot_number)=} and {type(start_i)=}. Expected int or Sequence[int]."
            )

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        assert self.data_is_normalized, "Normalization stats not computed. Call setup() first."
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self, batch_size_override=None, shuffle=False):
        batch_size = batch_size_override or self.batch_size
        return data.DataLoader(self.test_dataset, batch_size=batch_size, num_workers=self.num_workers, shuffle=shuffle)
