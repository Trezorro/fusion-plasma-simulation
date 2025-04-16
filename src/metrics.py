from typing import Optional, Tuple
from collections import namedtuple, abc
import logging
from venv import logger

import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance

import src.entropy as entropy
import wandb
from src.config import get_current_config

logger = logging.getLogger(__name__)
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
        "energy_delta",  # Energy difference between the peak and adjusted base of DML.
        "pd_prominence",  # Peak prominence of matched PD peaks
        "energy_ratio",  # Ratio of the energy delta to the pd prominence.
    ],
    defaults=(None, None, None)
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
            # Optional properties:
            "energy_delta",  # Energy difference between the peak and adjusted base.
            "energy_base_x",  # Position of the base used to calculate the energy delta.
            "pd_prominence",  # Peak prominence of matched PD peaks
        ],
        defaults=(None, None, None),
    )
):
    """
    A named tuple subclass to store peak properties with a factory method for creation.

    Allows storage and retrieval of the properties of multiple peaks, from one trace, so that they can be 
    compared to other arrays of peak properties easily.

    Iteration over the PeakProps instance yields the properties in the order they are defined like a tuple.
    Using the iter_peaks() method will iterate over the peaks in the arrays, returning a new PeakProps instance for each peak.

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

    @property
    def energy_ratio(self):
        """
        Calculate the ratio of the PD prominence to the energy delta.
        """
        if self.pd_prominence is None or self.energy_delta is None:
            return None
        return self.pd_prominence / self.energy_delta

    # Measure Aliases:
    @property
    def width(self):
        return self.widths

    @property
    def height(self):
        return self.Y

    @property
    def prominence(self):
        return self.prominences

    @property
    def base(self):
        return self.bases

    def num_peaks(self):
        """
        Returns the number of peaks.
        """
        if isinstance(self.X, abc.Collection):
            return len(self.X)
        if self.X is None:
            return 0
        return 1

    def __repr__(self) -> str:
        if self.X is None:
            return "<PeakProps None>"
        if not isinstance(self.X, abc.Collection):
            return f"PeakProps{tuple(self)}"
        return f"<PeakProps :" + ", ".join(
            f'{f}[{len(v) if v is not None else None}]' for f, v in self.items()
        ) + ">"

    def items(self):
        """
        Returns the properties as a list of tuples.
        """
        return [(field, getattr(self, field)) for field in self._fields]

    def _asdict(self):
        """
        Returns the properties as a dictionary.
        """
        return {field: getattr(self, field) for field in self._fields}

    def __getitem__(self, i):
        """
        Allows indexing into the PeakProps instance to get individual peak properties.
        """
        if type(i) is str:
            return getattr(self, i)
        if isinstance(self.X, abc.Collection):
            # assuming all properties are the same length
            return PeakProps(
                X=self.X[i].item(),
                Y=self.Y[i].item(),
                prominences=self.prominences[i].item(),
                bases=self.bases[i].item(),
                left_ips=self.left_ips[i].item(),
                right_ips=self.right_ips[i].item(),
                energy_delta=self.energy_delta[i].item() if self.energy_delta is not None else None,
                energy_base_x=self.energy_base_x[i].item() if self.energy_base_x is not None else None,
                pd_prominence=self.pd_prominence[i].item() if self.pd_prominence is not None else None,
            )
        return self

    def iter_peaks(self):
        if not isinstance(self.X, abc.Collection):
            return self
        for i in range(self.num_peaks()):
            yield self[i]

    @classmethod
    def from_find_peaks(
        cls,
        trace,
        prominence=0.001,
        width=0,
        rel_height=1.0,
        pd_trace: Optional[torch.Tensor] = None
    ) -> "PeakProps":
        """
        Factory method to create a PeakProps instance from the output of scipy's find_peaks.
        """
        peak_positions, props = find_peaks(trace, prominence=prominence, width=width, rel_height=rel_height)
        # for every peak, find the trace minimum in the range of the peak width
        if pd_trace is None:
            return cls(
                X=peak_positions,
                Y=trace[peak_positions].numpy(),
                prominences=props["prominences"],
                bases=props["width_heights"],
                left_ips=props["left_ips"],
                right_ips=props["right_ips"]
            )
        else:  # This means that this is a DML signal and we should calculate the DML-PD energy relationship
            peak_widths = props["widths"]
            half_width = peak_widths / 2
            window_l = (peak_positions - half_width).astype(np.int32)
            window_r = np.ceil(peak_positions + half_width).astype(np.int32)
            # If the next peak is even further away, we will extend the energy delta window to the next peak
            right_neighbor_pos = np.append(peak_positions[1:], len(trace))
            extended_window_r = np.max(np.vstack((window_r, right_neighbor_pos)), axis=0).astype(np.int32)
            # Ensure the windows are within the bounds of the trace
            window_l = np.clip(window_l, 0, len(trace) - 1)
            extended_window_r = np.clip(extended_window_r, 0, len(trace))
            # Find the minimum in the window for each peak. This is the assumed base energy position.
            energy_base_pos = np.array(
                [
                    window_l[i] + trace[window_l[i]:extended_window_r[i]].argmin(0)
                    for i in range(len(peak_positions))
                ]
            )
            energy_delta = trace[peak_positions] - trace[energy_base_pos]
            # get peaks from the pd_trace
            pd_peak_positions, pd_props = find_peaks(
                pd_trace, prominence=prominence, width=width, rel_height=rel_height
            )
            pd_prominence_sums = []
            # sum up the PD prominences of peaks in the same window
            for window in zip(window_l, extended_window_r):
                # get the peaks in the window
                pd_peaks_in_window = (pd_peak_positions >= window[0]) & (pd_peak_positions < window[1])
                # sum up the prominences of the peaks in the window
                pd_prominence_sums.append(np.sum(pd_props["prominences"][pd_peaks_in_window], axis=0))
            pd_prominence_sums = np.array(pd_prominence_sums)
            return cls(
                X=peak_positions,
                Y=trace[peak_positions].numpy(),
                prominences=props["prominences"],
                bases=props["width_heights"],
                left_ips=props["left_ips"],
                right_ips=props["right_ips"],
                energy_delta=energy_delta,
                energy_base_x=energy_base_pos,
                pd_prominence=pd_prominence_sums,
            )

    def __add__(self, other: "PeakProps") -> "PeakProps":
        """Didnt get used.
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
        energy_delta_w = pd_prominence_w = energy_ratio_w = None
        if self.energy_delta is not None and other.energy_delta is not None:
            # special case for wassersteins between DML peaks
            energy_delta_w = wasserstein_distance(self.energy_delta, other.energy_delta)
            pd_prominence_w = wasserstein_distance(self.pd_prominence, other.pd_prominence)
            energy_ratio_w = wasserstein_distance(self.energy_ratio, other.energy_ratio)
        if self.num_peaks() == 0 and other.num_peaks() == 0:
            return PeakWasserstein(
                height=0.0,
                prominence=0.0,
                base=0.0,
                width=0.0,
            )
        if self.num_peaks() == 0:
            return PeakWasserstein(
                height=wasserstein_distance([SENTINEL_VALUE], other.Y),
                prominence=wasserstein_distance([SENTINEL_VALUE], other.prominences),
                base=wasserstein_distance([SENTINEL_VALUE], other.bases),
                width=wasserstein_distance([SENTINEL_VALUE], other.widths),
            )
        if other.num_peaks() == 0:
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
            energy_delta=energy_delta_w,
            pd_prominence=pd_prominence_w,
            energy_ratio=energy_ratio_w
        )


def batch_get_peakprops(
    batch: torch.Tensor,
    prominence=0.001,
    width=0,
    rel_height=1.0,
    dml_channel_index=None,
    pd_channel_index=None
) -> list[list[PeakProps]]:
    """Find peaks in a batch of time series data.

    Args:
        batch (torch.Tensor): A batch of time series data with shape (B, C, T).

    Returns:
        list: A transposed list of PeakProps namedtuples for each time series in the batch. (C, B)
    """
    logger.debug(f"Finding peaks in batch of shape {batch.shape}")
    if dml_channel_index is None or pd_channel_index is None:
        peak_results = [
            [
                PeakProps.from_find_peaks(
                    channel_trace, prominence=prominence, width=width, rel_height=rel_height
                ) for channel_trace in sample  # Iterate over the channel dimension (C)  
            ] for sample in batch  # Iterate over the batch dimension (B)
        ]
    else:
        peak_results = []
        for sample in batch:  # Iterate over the batch dimension (B)
            sample_channel_results = []
            peak_results.append(sample_channel_results)
            for channel_i, trace in enumerate(sample):
                if channel_i == dml_channel_index:  # For dml channel, we want to calculate the energy delta
                    sample_channel_results.append(
                        PeakProps.from_find_peaks(
                            trace,
                            prominence=prominence,
                            width=width,
                            rel_height=rel_height,
                            pd_trace=sample[pd_channel_index]
                        )
                    )
                else:
                    sample_channel_results.append(
                        PeakProps.from_find_peaks(
                            trace, prominence=prominence, width=width, rel_height=rel_height
                        )
                    )
    return peak_results


def get_peak_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[dict, dict]:
    """Return a dict with the sample peak metrics and a dict with intermediate results for plotting.

    For measure in {'height', 'prominence', 'width', 'base'} we will calculate the wasserstein distance between the target and predicted distributions of that measure:
        - /error/peak_{measure}/pairwise_wasserstein/<channelname>:
         The batch mean wasserstein distance between the target and predicted heights/prominences/widths/bases, per channel.
        - /error/peak_{measure}/marginal_wasserstein/<channelname>: The wasserstein distance between the target and predicted distributions of that measure, per channel.

         """
    BASE_MEASURES = [
        "height",
        "prominence",
        "base",
        "width",
    ]  # + count with pairwise mse and marginal wasserstein
    C = get_current_config()
    CHANNEL_NAMES = C.data.cols.x
    dml_channel_index = CHANNEL_NAMES.index("DML") if "DML" in CHANNEL_NAMES else None
    pd_channel_index = CHANNEL_NAMES.index("PD") if "PD" in CHANNEL_NAMES else None
    pred_peaks_batch = batch_get_peakprops(
        pred, dml_channel_index=dml_channel_index, pd_channel_index=pd_channel_index
    )  # (B, C)
    target_peaks_batch = batch_get_peakprops(
        target, dml_channel_index=dml_channel_index, pd_channel_index=pd_channel_index
    )  # (B, C)
    BATCH_SIZE = len(target_peaks_batch)

    # plot results of dml delta vs pd prominence
    # if dml_channel_index is not None and pd_channel_index is not None:
    #     plot_delta_prominence_scatter(dml_channel_index, pred_peaks_batch)

    logger.debug(f"Analyzing peak distributions and calculating difference metrics for {BATCH_SIZE} samples")
    metrics_out = {}
    for channel_i, channel_name in enumerate(CHANNEL_NAMES):
        measures = BASE_MEASURES
        if channel_name == "DML":
            measures = measures + ["energy_delta", "pd_prominence", "energy_ratio"]
        pairwise_distances = {
            f"/error/peak_{measure}/pairwise_wasserstein/{channel_name}": 0. for measure in measures
        }
        counts_target = []
        counts_pred = []
        marginal_dist_pred = {measure: [] for measure in measures}
        marginal_dist_target = {measure: [] for measure in measures}
        for sample_i in range(BATCH_SIZE):
            pred_sample = pred_peaks_batch[sample_i][channel_i]
            target_sample = target_peaks_batch[sample_i][channel_i]
            pair_wasserstein = pred_sample - target_sample  # Subtraction operator is overloaded with wasserstein distance
            counts_pred.append(pred_sample.num_peaks())
            counts_target.append(target_sample.num_peaks())
            for measure in measures:
                pairwise_distances[f"/error/peak_{measure}/pairwise_wasserstein/{channel_name}"] += getattr(
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
                    case "energy_delta":
                        marginal_dist_pred[measure].extend(pred_sample.energy_delta)
                        marginal_dist_target[measure].extend(target_sample.energy_delta)
                    case "pd_prominence":
                        marginal_dist_pred[measure].extend(pred_sample.pd_prominence)
                        marginal_dist_target[measure].extend(target_sample.pd_prominence)
                    case "energy_ratio":
                        marginal_dist_pred[measure].extend(pred_sample.energy_ratio)
                        marginal_dist_target[measure].extend(target_sample.energy_ratio)
                    case _:
                        raise ValueError(f"Unknown measure: {measure}")

        metrics_out[f"/error/peak_count/marginal_wasserstein/{channel_name}"] = wasserstein_distance(
            counts_pred, counts_target
        )
        metrics_out[f"/error/peak_count/pairwise_mse/{channel_name}"] = (
            np.mean(np.power(np.array(counts_pred) - np.array(counts_target), 2))
        )
        # Average over the batch
        for measure in measures:
            pairwise_distances[f"/error/peak_{measure}/pairwise_wasserstein/{channel_name}"] /= BATCH_SIZE
            metrics_out[f"/error/peak_{measure}/marginal_wasserstein/{channel_name}"] = wasserstein_distance(
                marginal_dist_pred[measure] or [-1 * BATCH_SIZE],
                marginal_dist_target[measure] or [-1 * BATCH_SIZE],  # distance to a sentinel value
            )

        metrics_out.update(pairwise_distances)
    peak_features = {
        "pred_peaks": pred_peaks_batch,
        "target_peaks": target_peaks_batch,
    }
    return metrics_out, peak_features


def plot_delta_prominence_scatter(dml_channel_index, pred_peaks_batch):
    dml_peaks = [sample[dml_channel_index] for sample in pred_peaks_batch]
    import plotly.express as px
    dml_peaks = [peak for sample in dml_peaks for peak in sample.iter_peaks()]
    sample_nums = [[i] * peaks.num_peaks() for i, peaks in enumerate(dml_peaks)]
    sample_nums = [item for sublist in sample_nums for item in sublist]
    fig = px.scatter(
        x=[peak.energy_delta for peak in dml_peaks],
        y=[peak.pd_prominence for peak in dml_peaks],
        color=sample_nums,
        opacity=0.5,
        title="DML energy delta vs PD prominence",
        labels={
            "x": "DML energy delta",
            "y": "PD prominence",
        },
        color_discrete_sequence=px.colors.qualitative.Plotly,
        # show color as discrete traces in legend
    )
    fig.show()
