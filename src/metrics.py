from typing import Tuple
from collections import namedtuple

import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance

import src.entropy as entropy
import wandb
from src.config import get_current_config
"""
## Absolute metrics (targets and predicted are both viable inputs on their own)
- moments:
    - mean
    - variance
    - skewness
    - kurtosis
    

## Conditional Error metrics (targets and predicted are compared, should be paired), often aggragated to a single value with mse.
- moments:
    - mean
    - variance
    - skewness
    - kurtosis
- Element wise magnitude
- Element wise squared magnitude


Torchmetrics:
- audio.SignalNoiseRatio


## Marginal distribution distance metrics (targets and predicted batches are compared)


Wandb structure

val/

val/predictions/distributions/mean
val/predictions/distributions/...

val/predictions/moments/mean

"""

#### -- Old Metrics -- ####
# wandb.define_metric("loss/time_domain_train", summary="min")
# wandb.define_metric("loss/time_domain_val", summary="min")

# wandb.define_metric("val/time_pred_batch_variance", summary="max")
# wandb.define_metric("val/time_pred_batch_var_mean_adjusted", summary="max")
# wandb.define_metric("val/freq_pred_batch_variance", summary="max")
# wandb.define_metric("val/freq_pred_batch_var_mean_adjusted", summary="max")
# wandb.define_metric("val/freq_pred_batch_input_variance_ratio", summary="max")

# wandb.define_metric("val/time_target_batch_variance", summary="max")
# wandb.define_metric("val/time_target_batch_var_mean_adjusted", summary="max")
# wandb.define_metric("val/freq_target_batch_variance", summary="max")
# wandb.define_metric("val/freq_target_batch_var_mean_adjusted", summary="max")
# wandb.define_metric("val/freq_target_batch_input_variance_ratio", summary="max")
#### -- New Metrics -- ####

METRIC_NAMES = ["mean", "var", "skew", "kurtosis"]


def define_error_metrics(main_prefix: str):
    # wandb.define_metric("val/error/magnitude_mean_mse", summary="min")
    C = get_current_config()
    for metric_name in METRIC_NAMES:
        wandb.define_metric(f"{main_prefix}/error/magnitude_{metric_name}_mse", summary='min')
        wandb.define_metric(f"{main_prefix}/error/diff_{metric_name}_mse", summary='min')
    for entropy_method in entropy.VALID_FUNCS.keys():
        for channel_name in C.data.cols.x:
            wandb.define_metric(f"{main_prefix}/error/{entropy_method}_mse/{channel_name}", summary='min')
            wandb.define_metric(f"{main_prefix}/error/{entropy_method}_mae/{channel_name}", summary='min')
            wandb.define_metric(
                f"{main_prefix}/error/{entropy_method}_wasserstein/{channel_name}", summary='min'
            )
        wandb.define_metric(f"{main_prefix}/error/{entropy_method}_mse/mean", summary='min')
        wandb.define_metric(f"{main_prefix}/error/{entropy_method}_mae/mean", summary='min')
        wandb.define_metric(f"{main_prefix}/error/{entropy_method}_wasserstein/mean", summary='min')


def first_difference(batched_time_series: torch.Tensor):
    return batched_time_series[:, :, 1:] - batched_time_series[:, :, :-1]


def moments(batched_time_series: torch.Tensor):
    mean = torch.mean(batched_time_series, -1, keepdim=True)
    diffs = batched_time_series - mean
    var = torch.mean(torch.pow(diffs, 2), -1, keepdim=True)
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0), -1, keepdim=True)
    kurtoses = torch.mean(torch.pow(zscores, 4.0), -1, keepdim=True) - 3.0
    return dict(mean=mean, var=var, skew=skews, kurtosis=kurtoses)


def get_moments_errors(pred: torch.Tensor, target: torch.Tensor):
    mse = lambda x, y: torch.mean(torch.pow(x - y, 2))
    magnitudes_pred = moments(pred)
    magnitudes_target = moments(target)
    mag_out = {
        f"/error/magnitude_{time_agg}_mse": mse(vpredict, vtarget).item()
        for (time_agg, vpredict), (_, vtarget) in zip(magnitudes_pred.items(), magnitudes_target.items())
    }
    first_diff_pred = moments(first_difference(pred))
    first_diff_target = moments(first_difference(target))
    first_diff_out = {
        f"/error/diff_{time_agg}_mse": mse(vpredict, vtarget).item()
        for (time_agg, vpredict), (_, vtarget) in zip(first_diff_pred.items(), first_diff_target.items())
    }
    return {**mag_out, **first_diff_out}


def get_entropy_metrics(pred: torch.Tensor, target: torch.Tensor):
    """Return a dict with the sample entropy metrics:

    All entropy values are normalized to the target entropy mean and standard deviation.
    The wasserstein distance is the only metric for which the mean cannot be calculated from combining all channels, it is calculated per channel first. Then averaged.
    The other mean values are calculated by averaging over all values in the batch, irrespective of channel (as one big bag of values).

    - /error/app_entropy_wasserstein/<channelname>:
         The wasserstein distance between the target and predicted normalized app entropy batch distributions, per channel.
    - /error/app_entropy_wasserstein/mean: 
        The mean of the wasserstein distances between the target and predicted normalized app entropy batch distributions, averaged over all channels.
    - /error/app_entropy_msd/<channelname>:
         The mean signed difference (MSD) between the paired target and predicted app entropy, per channel. Essentially estimating the bias of the prediction.
    - /error/app_entropy_mse/<channelname>:
         The mean squared error between the paired target and predicted normalized app entropy batch distributions, per channel.
    - /error/app_entropy_mse/mean: 
        The mean of the mean squared errors between the paired target and predicted normalized app entropy batch distributions, averaged over all channels.
    - /error/app_entropy_mae/<channelname>:
         The mean absolute error between the paired target and predicted normalized app entropy batch distributions, per channel.
    - /error/app_entropy_mae/mean: 
        The mean of the mean absolute errors between the paired target and predicted normalized app entropy batch distributions, averaged over all channels.
    - /error/kl_divergence/<channelname>:
         The Kullback-Leibler divergence between the target and predicted timeseries, per channel.
    """

    C = get_current_config()
    channel_names = C.data.cols.x
    metrics = {}
    for method, func in entropy.VALID_FUNCS.items():
        entropy_target = entropy.batch_entropy(target, func)
        entropy_pred = entropy.batch_entropy(pred, func)
        pair_errors = entropy_pred - entropy_target
        msd_channel = np.mean(pair_errors, axis=0)
        msd = np.mean(pair_errors)
        mse_channel = np.mean(np.power(pair_errors, 2), axis=0)
        mse = np.mean(np.power(pair_errors, 2))
        mae_channel = np.mean(np.abs(pair_errors), axis=0)
        mae = np.mean(np.abs(pair_errors))

        wasserstein_sum = 0  # Wasserstein distance needs to be averaged after the channel loop
        for i, channel_name in enumerate(channel_names):
            wasserstein_per_channel = wasserstein_distance(entropy_target[:, i], entropy_pred[:, i])
            wasserstein_sum += wasserstein_per_channel
            metrics[f"/error/{method}_wasserstein/{channel_name}"] = wasserstein_per_channel
            metrics[f"/error/{method}_msd/{channel_name}"] = msd_channel[i]
            metrics[f"/error/{method}_mse/{channel_name}"] = mse_channel[i]
            metrics[f"/error/{method}_mae/{channel_name}"] = mae_channel[i]
        metrics[f"/error/{method}_wasserstein/mean"] = wasserstein_sum / len(channel_names)
        metrics[f"/error/{method}_msd/mean"] = msd
        metrics[f"/error/{method}_mse/mean"] = mse
        metrics[f"/error/{method}_mae/mean"] = mae
    return metrics


PeakWasserstein = namedtuple(
    "PeakWasserstein",
    [
        "height",  # Peak heights
        "prominence",
        "base",  # Height at which the width is measured. Width is right_ip - left_ip.
        "width",
    ]
)


class PeakProps(
    namedtuple(
        "PeakProps",
        [
            "X",  # Peak positions
            "Y",  # Peak heights
            "prominences",
            "bases",  # Height at which the width is measured. Width is right_ip - left_ip.
            "left_ips",  # Left (interpolated) x positions of the peak base. 
            "right_ips",  # Right (interpolated) x positions of the peak base.
        ]
    )
):
    """
    A named tuple subclass to store peak properties with a factory method for creation.

    Provides widths, length, and addition/subtraction operations for peak properties.
    Note: The addition operator concatenates the peak properties of two instances.
    Note: The subtraction operator calculates the Wasserstein distance between two PeakProps instances and returns a PeakWasserstein instance.
    """
    __slots__ = ()  # Prevents the creation of a per-instance dictionary, keeping it lightweight.

    @property
    def widths(self):
        """
        Calculate the widths of the peaks based on left and right interpolated positions.
        """
        return self.right_ips - self.left_ips

    def __len__(self):
        """
        Returns the number of peaks.
        """
        return len(self.X)

    def __repr__(self) -> str:
        return f"<PeakProps n={len(self.X)} peaks>"

    def __str__(self) -> str:
        return f"<PeakProps n={len(self.X)} peaks>"

    def __getitem__(self, i):
        """
        Allows indexing into the PeakProps instance to get individual peak properties.
        """
        return PeakProps(
            X=self.X[i].item(),
            Y=self.Y[i].item(),
            prominences=self.prominences[i].item(),
            bases=self.bases[i].item(),
            left_ips=self.left_ips[i].item(),
            right_ips=self.right_ips[i].item(),
        )

    def __iter__(self):
        for i in range(len(self)):
            yield PeakProps(
                X=self.X[i].item(),
                Y=self.Y[i].item(),
                prominences=self.prominences[i].item(),
                bases=self.bases[i].item(),
                left_ips=self.left_ips[i].item(),
                right_ips=self.right_ips[i].item(),
            )

    @classmethod
    def from_find_peaks(cls, trace, prominence=0.001, width=0, rel_height=1.0) -> "PeakProps":
        """
        Factory method to create a PeakProps instance from the output of scipy's find_peaks.
        """
        peak_positions, props = find_peaks(trace, prominence=prominence, width=width, rel_height=rel_height)
        return cls(
            X=peak_positions,
            Y=trace[peak_positions].numpy(),
            prominences=props["prominences"],
            bases=props["width_heights"],
            left_ips=props["left_ips"],
            right_ips=props["right_ips"],
        )

    def __add__(self, other: "PeakProps") -> "PeakProps":
        """
        Overload the addition operator to combine two PeakProps instances.
        """
        if not isinstance(other, PeakProps):
            raise TypeError("Can only add PeakProps instances.")
        return PeakProps(
            X=np.concatenate((self.X, other.X)),
            Y=np.concatenate((self.Y, other.Y)),
            prominences=np.concatenate((self.prominences, other.prominences)),
            bases=np.concatenate((self.bases, other.bases)),
            left_ips=np.concatenate((self.left_ips, other.left_ips)),
            right_ips=np.concatenate((self.right_ips, other.right_ips)),
        )

    def __sub__(self, other: "PeakProps") -> PeakWasserstein:
        """
        The wasserstein distance between two PeakProps instances.

        If one of the instances is empty, the distance is calculated against a sentinel value. Since 
        If both instances are empty, the distance is 0.
        """
        SENTINEL_VALUE = -1.0
        if not isinstance(other, PeakProps):
            raise TypeError("Can only subtract PeakProps instances.")
        if len(self) == 0 and len(other) == 0:
            return PeakWasserstein(
                height=0.0,
                prominence=0.0,
                base=0.0,
                width=0.0,
            )
        if len(self) == 0:
            return PeakWasserstein(
                height=wasserstein_distance([SENTINEL_VALUE], other.Y),
                prominence=wasserstein_distance([SENTINEL_VALUE], other.prominences),
                base=wasserstein_distance([SENTINEL_VALUE], other.bases),
                width=wasserstein_distance([SENTINEL_VALUE], other.widths),
            )
        if len(other) == 0:
            return PeakWasserstein(
                height=wasserstein_distance(self.Y, [SENTINEL_VALUE]),
                prominence=wasserstein_distance(self.prominences, [SENTINEL_VALUE]),
                base=wasserstein_distance(self.bases, [SENTINEL_VALUE]),
                width=wasserstein_distance(self.widths, [SENTINEL_VALUE]),
            )
        return PeakWasserstein(
            height=wasserstein_distance(self.Y, other.Y),
            prominence=wasserstein_distance(self.prominences, other.prominences),
            base=wasserstein_distance(self.bases, other.bases),
            width=wasserstein_distance(self.widths, other.widths),
        )


def batch_get_peakprops(batch: torch.Tensor,
                        prominence=0.001,
                        width=0,
                        rel_height=1.0) -> list[list[PeakProps]]:
    """Find peaks in a batch of time series data.

    Args:
        batch (torch.Tensor): A batch of time series data with shape (B, C, T).

    Returns:
        list: A transposed list of PeakProps namedtuples for each time series in the batch. (C, B)
    """
    peak_results = [
        [
            PeakProps.from_find_peaks(
                sample[channel_idx], prominence=prominence, width=width, rel_height=rel_height
            ) for sample in batch  # Iterate over the batch dimension (B)
        ] for channel_idx in range(batch.shape[1])  # Iterate over the channel dimension (C)
    ]
    return peak_results


def pairwise_wasserstein_distance(
    pred_peaks: list[list[PeakProps]], target_peaks: list[list[PeakProps]]
) -> dict:
    """Calculate the mean pairwise Wasserstein distance for each measure."""
    C = get_current_config()
    channel_names = C.data.cols.x
    metrics = {}
    measures = ["height", "prominence", "base", "width"]

    for channel, pred_ch_samples, target_ch_samples in zip(channel_names, pred_peaks, target_peaks):
        pairwise_channel_metrics = {
            f"/error/peak_{measure}/pairwise_wasserstein/{channel}": 0. for measure in measures
        }
        marginal_pred = {measure: [] for measure in measures}
        marginal_target = {measure: [] for measure in measures}
        for pred_sample, target_sample in zip(pred_ch_samples, target_ch_samples):
            pair_wasserstein = pred_sample - target_sample  # Subtraction operator is overloaded for PeakProps
            for measure in measures:
                pairwise_channel_metrics[f"/error/peak_{measure}/pairwise_wasserstein/{channel}"] += getattr(
                    pair_wasserstein, measure
                )
                marginal_pred[measure].extend(getattr(pred_sample, measure))
                marginal_target[measure].extend(getattr(target_sample, measure))

        # Average over the batch
        for measure in measures:
            pairwise_channel_metrics[f"/error/peak_{measure}/pairwise_wasserstein/{channel}"] /= len(
                pred_ch_samples
            )

        metrics.update(pairwise_channel_metrics)

    return metrics


def get_peak_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[dict, dict]:
    """Return a dict with the sample peak metrics and a dict with intermediate results for plotting.

    For measure in {'height', 'prominence', 'width', 'base'} we will calculate the wasserstein distance between the target and predicted distributions of that measure:
        - /error/peak_{measure}/pairwise_wasserstein/<channelname>:
         The batch mean wasserstein distance between the target and predicted heights/prominences/widths/bases, per channel.
        - /error/peak_{measure}/marginal_wasserstein/<channelname>: The wasserstein distance between the target and predicted distributions of that measure, per channel.

         """
    MEASURES = ["height", "prominence", "base", "width"]
    C = get_current_config()
    CHANNEL_NAMES = C.data.cols.x
    BATCH_SIZE = pred.shape[0]
    pred_peaks = batch_get_peakprops(pred)  # (C, B)
    target_peaks = batch_get_peakprops(target)  # (C, B)

    metrics_out = {}

    for channel, pred_ch_samples, target_ch_samples in zip(CHANNEL_NAMES, pred_peaks, target_peaks):
        pairwise_distances = {
            f"/error/peak_{measure}/pairwise_wasserstein/{channel}": 0. for measure in MEASURES
        }
        marginal_dist_pred = {measure: [] for measure in MEASURES}
        marginal_dist_target = {measure: [] for measure in MEASURES}
        for pred_sample, target_sample in zip(pred_ch_samples, target_ch_samples):
            pair_wasserstein = pred_sample - target_sample  # Subtraction operator is overloaded for PeakProps
            for measure in MEASURES:
                pairwise_distances[f"/error/peak_{measure}/pairwise_wasserstein/{channel}"] += getattr(
                    pair_wasserstein, measure
                )
                match measure:
                    case "height":
                        marginal_dist_pred[measure].extend(pred_sample.Y)
                        marginal_dist_target[measure].extend(target_sample.Y)
                    case "prominence":
                        marginal_dist_pred[measure].extend(pred_sample.prominences)
                        marginal_dist_target[measure].extend(target_sample.prominences)
                    case "base":
                        marginal_dist_pred[measure].extend(pred_sample.bases)
                        marginal_dist_target[measure].extend(target_sample.bases)
                    case "width":
                        marginal_dist_pred[measure].extend(pred_sample.widths)
                        marginal_dist_target[measure].extend(target_sample.widths)
                    case _:
                        raise ValueError(f"Unknown measure: {measure}")

        # Average over the batch
        for measure in MEASURES:
            pairwise_distances[f"/error/peak_{measure}/pairwise_wasserstein/{channel}"] /= len(
                pred_ch_samples
            )
            metrics_out[f"/error/peak_{measure}/marginal_wasserstein/{channel}"] = wasserstein_distance(
                marginal_dist_pred[measure] or [-1 * BATCH_SIZE],
                marginal_dist_target[measure] or [-1 * BATCH_SIZE],  # distance to a sentinel value
            )

        metrics_out.update(pairwise_distances)
    peak_features = {
        "pred_peaks": pred_peaks,
        "target_peaks": target_peaks,
    }
    return metrics_out, peak_features
