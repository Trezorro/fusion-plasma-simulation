import torch
import pytest


def batch_variance(time_series_batch, mean_adjusted=False):
    if mean_adjusted:
        time_series_batch = time_series_batch / (time_series_batch.mean(dim=2, keepdim=True) + 1e-8)
    return time_series_batch.var(dim=0)


def test_normal_case():
    time_series_batch = torch.tensor(
        [[[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]], [[3, 4, 5], [6, 7, 8]]], dtype=torch.float32
    )
    result = batch_variance(time_series_batch)
    assert result.shape == (2, 3), "Output shape mismatch"


def test_identical_case():
    identical_batch = torch.tensor(
        [[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]], dtype=torch.float32
    )
    result = batch_variance(identical_batch)
    assert torch.all(result == 0), "Variance should be zero for identical time series"


def test_single_time_step():
    single_step_batch = torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]]], dtype=torch.float32)
    result = batch_variance(single_step_batch)
    expected_result = torch.tensor([[4], [4]], dtype=torch.float32)
    assert torch.allclose(result, expected_result), "Variance for single time step is incorrect"


def test_zero_mean_case():
    zero_mean_batch = torch.tensor(
        [[[-1, 0, 1], [2, 3, 4]], [[-1, 0, 1], [2, 3, 4]], [[-1, 0, 1], [2, 3, 4]]], dtype=torch.float32
    )
    result = batch_variance(zero_mean_batch, mean_adjusted=True)
    assert torch.all(result == 0), "Variance should be zero after mean adjustment"


def test_randomized_case():
    random_batch = torch.rand(5, 3, 10)
    result = batch_variance(random_batch)
    assert result.shape == (3, 10), "Output shape mismatch for randomized case"
