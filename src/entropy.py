#%% Imports
from functools import partial
from typing import Callable, List, Tuple
import numpy as np
import torch
import antropy as ant
import time
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

VALID_FUNCS = {
    "app_entropy": ant.app_entropy,
    "perm_entropy": partial(ant.perm_entropy, normalize=True),
    "spectral_entropy": partial(ant.spectral_entropy, sf=100, normalize=True),
    # "sample_entropy":
    #     ant.sample_entropy,
    # "svd_entropy":
    #     ant.svd_entropy,
    # "hjorth_params": ant.hjorth_params,
    # "num_zerocross": ant.num_zerocross,
    # "lziv_complexity": ant.lziv_complexity,
    # "spectral_entropy32":
    #     partial(ant.spectral_entropy, sf=32, normalize=True),
    # "spectral_entropy100welch16":
    #     partial(ant.spectral_entropy, sf=100, normalize=True, method="welch", nperseg=16),
}


#%% Definitions
def get_sample_entropy(batched_time_series: torch.Tensor) -> np.ndarray:
    """Calculate the sample entropy for a batch of time series.
    
    See: https://en.wikipedia.org/wiki/Sample_entropy

    An input of shape (N, C, T) will return an output of shape (N, C).
    """
    batched_time_series_np = batched_time_series.numpy()
    return np.apply_along_axis(ant.sample_entropy, -1, batched_time_series_np, order=2)


def batch_entropy(batched_time_series: torch.Tensor, func: Callable, *args, **kwargs) -> np.ndarray:
    """Calculate the entropy for a batch of time series using a given function.

    An input of shape (N, C, T) will return an output of shape (N, C).
    """
    batched_time_series_np = batched_time_series.numpy()
    return np.apply_along_axis(func, -1, batched_time_series_np, *args, **kwargs)


def get_normalized_entropies(
    pred_batch: torch.Tensor,
    target_batch: torch.Tensor,
    func: Callable = ant.sample_entropy,
    *args,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Used for functions without a normalization option, like sample_entropy."""
    entropy_target = batch_entropy(target_batch, func, *args, **kwargs)
    # plot_inf_entropy_timeseries(entropy_target, target_batch)
    entropy_pred = batch_entropy(pred_batch, func, *args, **kwargs)
    # assert not np.isnan(entropy_pred).any(), "NaN values in entropy_pred."
    # assert not np.isnan(entropy_target).any(), "NaN values in entropy_target."
    target_means = np.nanmean(np.where(np.isinf(entropy_target), np.nan, entropy_target), axis=0)
    target_std = np.nanstd(np.where(np.isinf(entropy_target), np.nan, entropy_target), axis=0) + 1e-10
    entropy_target_normalized = (entropy_target - target_means) / target_std
    entropy_pred_normalized = (entropy_pred - target_means) / target_std
    # assert not np.isnan(entropy_target_normalized).any(), "NaN values in entropy_target_normalized."
    return entropy_target_normalized, entropy_pred_normalized


def plot_inf_entropy_timeseries(entropy: np.ndarray, timeseries_batch):
    """Plot a trace for each time series that has inf or neg entropy in plotly and show.

    Args: 
        entropy (np.ndarray): A batch of time series entropies (BxC)
        timeseries_batch (torch.Tensor): A batch of time series. (BxCxT)
    """
    # Get the indices of the time series with inf or neg entropy
    inf_indices = np.where(np.isinf(entropy) | (entropy < 0))  # returns zipped
    # Get the time series with inf or neg entropy
    inf_timeseries = timeseries_batch[inf_indices]
    # Plot each time series with inf or neg entropy
    print("Plotting inf or neg entropy time series...")
    print(f"Found: {len(inf_indices[0])} such time series.")
    fig = go.Figure()
    for i, ts in enumerate(inf_timeseries):
        fig.add_trace(
            go.Scatter(
                y=ts.numpy(),
                mode='lines',
                name=f"Time Series {inf_indices[i]} entropy {entropy[*inf_indices[i]]}"
            )
        )
        x = ts.numpy()
        print(f"Time Series: {inf_indices[i]}", x)
        print("Sample Entropy:", ant.sample_entropy(x, order=2))
        print("Sample Entropy (Handled):", ant.sample_entropy(x) if len(x) > 10 else "N/A")
        print("First diff sample entropy:", ant.sample_entropy(np.diff(x), order=2))
        print("Permutation Entropy:", ant.perm_entropy(x, normalize=True))
        print("Spectral Entropy:", ant.spectral_entropy(x, sf=100, method='welch', normalize=True))
        print("SVD Entropy:", ant.svd_entropy(x, normalize=True))
        print("Approximate Entropy:", ant.app_entropy(x))
        print("Hjorth Parameters:", ant.hjorth_params(x))
        print("Zero-Crossings:", ant.num_zerocross(x))
        print(
            "Lempel-Ziv Complexity:",
            ant.lziv_complexity(''.join(map(str,
                                            np.sign(x).astype(int))), normalize=True)
        )
    # plot all other time series in a seperate legend group:
    # others = timeseries_batch[~(np.isinf(entropy) | (entropy < 0))].numpy()
    # entropies = entropy[~(np.isinf(entropy) | (entropy < 0))]
    # # Get the indices of the time series with inf or neg entropy
    # for i, ts in enumerate(others):
    #     fig.add_trace(
    #         go.Scatter(
    #             y=ts,
    #             mode='lines',
    #             name=f"{entropies[i]} entropy Time Series {i} ",
    #             # hoverinfo="name",
    #             legendgroup="Other",
    #             showlegend=True
    #         )
    #     )

    fig.update_layout(title=f"Time Series with Inf/Neg Entropy", xaxis_title="Time", yaxis_title="Value")
    fig.show()


#%%


def test_sample_entropy(shape=(10, 5, 100)):
    """Test the sample entropy calculation for a given shape."""
    x = torch.randn(*shape)
    start_time = time.time()
    res = get_sample_entropy(x)
    end_time = time.time()
    print(
        f"Sample entropy calculation took {end_time - start_time:4.5f} seconds for shape {shape} ({shape[0] * shape[1]} samples)."
    )
    print(f"Result shape: {res.shape}")
    print(res)
    pass
