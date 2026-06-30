import os
from typing import Literal
import h5py
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import logging
from src.config import get_current_config
import json

logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    """
    Determine the cache directory based on cluster environment or default to local.
    """
    # Check for cluster environment variables
    if "TEST_CACHE_DIR" in os.environ:
        base_dir = Path(os.environ["TEST_CACHE_DIR"])
    else:
        # Default to local directory
        base_dir = Path("output/test_cache")
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


class TestStepHDFCache:
    """A class to handle caching of test step results in HDF5 format.

    Modes:
        w: Write: Will overwrite any ancountered present keys during setting.
        r: Read: No setting allowed, just reading. If a key is not present, a whole batch get will fail.
        a: Setting will skip any present values without overwriting, saving a little bit of I/O. Reading same as r.

    HDF5 layout:
        Groups are nested as ``{shot_number}/{start_idx}/``. Each leaf group holds
        three datasets: ``generated_x``, ``surr_labels_gen``, and ``surr_labels_target``.

    Resumability:
        ``mode='a'`` is safe for re-runs: it skips any group that already has
        ``generated_x``, so a partial run can be resumed without losing existing data.

    Cache location:
        ``get_cache_dir()`` checks the ``TEST_CACHE_DIR`` environment variable
        (set to ``/scratch-shared/mtresoor/final_cache/`` on Snellius); it falls back
        to ``output/test_cache/`` locally.
    """

    def __init__(self, cache_filename: str = "test_step_cache", mode: Literal['w', 'r', 'a'] = "w"):
        self.base_dir = get_cache_dir()
        self.cache_filename = cache_filename
        self.h5_path = self.base_dir / (cache_filename + '.h5')
        self.mode = mode.lower()
        logger.info("Initialized HDF5 cache at %s in mode '%s'", self.h5_path, self.mode)

    def save_json_friend(self, dict):
        """Write a JSON-serializable copy of a metrics dict alongside the HDF5 cache.

        Tensors are converted to scalars; values that fail json.dumps are stringified.
        Writes to the same directory as the HDF5 file as {cache_filename}.json.

        Args:
            dict: Metrics dict from on_test_epoch_end. Modified in-place during serialization.
        """
        json_file = self.base_dir / (self.cache_filename + '.json')
        dict = dict.copy()
        for k,v in dict.items():
            try:
                if isinstance(v, torch.Tensor):
                    dict[k] = v.item()
                else:
                    json.dumps(v)
            except Exception:
                dict[k] = str(v)
        try:
            with open(json_file, "w") as f:
                json.dump(dict, f, indent=2)
            logger.info("Saved JSON-friendly cache metadata to %s", json_file)
        except Exception as e:
            logger.error("Failed to save JSON-friendly cache metadata: %s", e)


    def set_from_batch(self, shot_nums, start_idxs, generated_x, surr_labels_gen, surr_labels_target):
        """
        Store batch results in the HDF5 cache.
        shot_nums: array-like, shape (batch,)
        start_idxs: array-like, shape (batch,)
        generated_x: np.ndarray, shape (batch, channels, timesteps)
        surr_labels_gen: np.ndarray, shape (batch, channels, timesteps)
        surr_labels_target: np.ndarray, shape (channels, timesteps)

        In ``mode='a'`` the group is only skipped if ``generated_x`` already exists;
        if ``generated_x`` is missing, all three datasets are written even in append mode.

        ``generated_x`` is stored as float32; the label datasets are stored as int16
        (saves space).
        """
        match self.mode:
            case 'r':
                raise RuntimeError("Cannot set values in read-only mode.")
            case "a":
                with h5py.File(self.h5_path, "a") as f:
                    for i, (shot_num, start_idx) in enumerate(zip(shot_nums, start_idxs)):
                        group_path = f"{shot_num}/{start_idx}"
                        grp = f.require_group(group_path)
                        # Store generated_x
                        if not "generated_x" in grp:
                            grp.create_dataset("generated_x", data=generated_x[i], dtype="f")
                            # Store surr_labels_gen
                            if not "surr_labels_gen" in grp:
                                grp.create_dataset("surr_labels_gen", data=surr_labels_gen[i], dtype="i2")
                            # Store surr_labels_target (same for all in batch)
                            if not "surr_labels_target" in grp:
                                grp.create_dataset("surr_labels_target", data=surr_labels_target[i], dtype="i2")
            case "w":
                with h5py.File(self.h5_path, "a") as f:
                    for i, (shot_num, start_idx) in enumerate(zip(shot_nums, start_idxs)):
                        group_path = f"{shot_num}/{start_idx}"
                        grp = f.require_group(group_path)
                        # Store generated_x
                        if "generated_x" in grp:
                            del grp["generated_x"]
                        grp.create_dataset("generated_x", data=generated_x[i], dtype="f")
                        # Store surr_labels_gen
                        if "surr_labels_gen" in grp:
                            del grp["surr_labels_gen"]
                        grp.create_dataset("surr_labels_gen", data=surr_labels_gen[i], dtype="i2")
                        # Store surr_labels_target (same for all in batch)
                        if "surr_labels_target" in grp:
                            del grp["surr_labels_target"]
                        grp.create_dataset("surr_labels_target", data=surr_labels_target[i], dtype="i2")
        logger.debug("Set %s entries in cache.", i + 1)

    def get(self, shot_nums, start_idxs):
        """ Retrieve cached data for a given shot_num and start_idx.
        
        Returns a dict with keys: generated_x, surr_labels_gen, surr_labels_target

        Raises KeyError if the cache for the given shot_num and start_idx does not exist.
        """
        C = get_current_config()
        C.data.history_length
        C.data.seq_length
        batch_size = len(shot_nums)
        # 5 is hardcoded here because the cache was created with 5 X-channels; this must match the training config
        generated_x_batch = np.zeros((batch_size, 5, C.data.seq_length), dtype=np.float32)
        surr_labels_gen_batch = np.zeros((batch_size, C.data.history_length + C.data.seq_length), dtype=np.int16) - 1
        surr_labels_target_batch = np.zeros_like(surr_labels_gen_batch) - 1
        with h5py.File(self.h5_path, "r") as f:
            for i, (shot_num, start_idx) in enumerate(zip(shot_nums, start_idxs)):
                group_path = f"{shot_num}/{start_idx}"
                if group_path not in f:
                    raise KeyError(f"No cache for {group_path}")
                shot_t_group = f[group_path]
                shot_t_group["generated_x"].read_direct(generated_x_batch, np.s_[:], np.s_[i, :, :])
                shot_t_group["surr_labels_gen"].read_direct(surr_labels_gen_batch, np.s_[:], np.s_[i, :])
                shot_t_group["surr_labels_target"].read_direct(surr_labels_target_batch, np.s_[:], np.s_[i, :])
            logger.debug("Got %s entries from cache.", i + 1)
        return torch.tensor(generated_x_batch), surr_labels_gen_batch, surr_labels_target_batch

    def find_cached_idxs(self, shot_number):
        """
        Return all cached start indices for a given shot number.

        Args:
            shot_number: TCV shot number (int or string).

        Returns:
            Sorted list of integer start indices.
        """
        with h5py.File(self.h5_path, "r") as f:
            if str(shot_number) not in f:
                return []
            shot_group = f[str(shot_number)]
            return sorted(int(k) for k in shot_group.keys() if k.isdigit())

    def quick_window(self, shot_number, time, dataset, repeat=1):
        """Retrieve the generated samples and surrogate labels for the cached window nearest to a given time.

        Looks up the closest available cached start index to the given time (by finding the
        nearest index in the parquet DataFrame, then snapping to the nearest cached index).

        Args:
            shot_number: TCV shot number.
            time: Target time in seconds (relative to shot start).
            dataset: Full parquet DataFrame with 'ShotNum' and 'time' columns.
            repeat: Number of times to tile the generated_x sample along axis 0
                (for visualization with multiple trajectories).

        Returns:
            Tuple of (sample, labels_gen, labels_real, start_idx).
        """
        shot_data_index = dataset[dataset['ShotNum'] == shot_number].index
        if len(shot_data_index) == 0:
            raise ValueError(f"Shot {shot_number} not found in dataset.")
        start_idx = shot_data_index.get_indexer([time], method='nearest')[0]
        logger.info("Shot %s at t=%s is at start index %s", shot_number, time, start_idx)
        limit_to_idx = self.find_cached_idxs(shot_number)
        start_idx = limit_to_idx[pd.Index(limit_to_idx).get_indexer([start_idx], method='nearest')[0]]
        logger.info(
            "Adjusted to available indeces, we get idx %s with time %.4f", start_idx, shot_data_index[start_idx]
        )
        with h5py.File(self.h5_path, "r") as f:
            group_path = f"{shot_number}/{start_idx}"
            if group_path not in f:
                raise KeyError(f"No cache for {group_path}")
            shot_t_group = f[group_path]
            sample = shot_t_group["generated_x"][:]
            labels_gen = shot_t_group["surr_labels_gen"][:]
            labels_real = shot_t_group["surr_labels_target"][:]
        return sample, labels_gen, labels_real, start_idx
