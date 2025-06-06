import os
from typing import Literal
import h5py
from pathlib import Path
import numpy as np
import logging
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
    """
    def __init__(self, cache_filename: str = "test_step_cache", mode: Literal['w', 'r', 'a'] = "w"):
        self.base_dir = get_cache_dir()
        self.h5_path = self.base_dir / (cache_filename + '.h5')
        self.mode = mode.lower()
        logger.info("Initialized HDF5 cache at %s in mode '%s'", self.h5_path, self.mode)

    def set_from_batch(self, shot_nums, start_idxs, generated_x, surr_labels_gen, surr_labels_target):
        """
        Store batch results in the HDF5 cache.
        shot_nums: array-like, shape (batch,)
        start_idxs: array-like, shape (batch,)
        generated_x: np.ndarray, shape (batch, channels, timesteps)
        surr_labels_gen: np.ndarray, shape (batch, channels, timesteps)
        surr_labels_target: np.ndarray, shape (channels, timesteps)
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
                        grp.create_dataset("surr_labels_target", data=surr_labels_target, dtype="i2")
        logger.debug("Set %s entries in cache.", i+1)

    def get(self, shot_nums, start_idxs):
        """
        Retrieve cached data for a given shot_num and start_idx.
        Returns a dict with keys: generated_x, surr_labels_gen, surr_labels_target

        Raises KeyError if the cache for the given shot_num and start_idx does not exist.
        """
        batch_size = len(shot_nums)
        generated_x_batch = np.zeros((batch_size, 5, 256), dtype=np.float32)
        surr_labels_gen_batch = np.zeros((batch_size, 256*2), dtype=np.int16) - 1
        surr_labels_target_batch = np.zeros((batch_size, 256*2), dtype=np.int16) -1
        with h5py.File(self.h5_path, "r") as f:
            for i, (shot_num, start_idx) in enumerate(zip(shot_nums, start_idxs)):
                group_path = f"{shot_num}/{start_idx}"
                if group_path not in f:
                    raise KeyError(f"No cache for {group_path}")
                shot_t_group = f[group_path]
                shot_t_group["generated_x"].read_direct(generated_x_batch, np.s_[:], np.s_[i, :, :])
                shot_t_group["surr_labels_gen"].read_direct(surr_labels_gen_batch, np.s_[:], np.s_[i, :, :])
                shot_t_group["surr_labels_target"].read_direct(surr_labels_target_batch, np.s_[:], np.s_[i, :, :])
            logger.debug("Got %s entries from cache.", i+1)
        return generated_x_batch, surr_labels_gen_batch, surr_labels_target_batch
