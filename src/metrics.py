import numpy as np
from sklearn import metrics
import torch
import wandb
from scipy.special import rel_entr  # Import for KL divergence
from scipy.stats import wasserstein_distance
import src.entropy as entropy
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
    for metric_name in METRIC_NAMES:
        wandb.define_metric(f"{main_prefix}/error/magnitude_{metric_name}_mse", summary='min')
        wandb.define_metric(f"{main_prefix}/error/diff_{metric_name}_mse", summary='min')
    wandb.define_metric(f"{main_prefix}/error/sample_entropy", summary='min')


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

    - /error/<channelname>/sample_entropy_wasserstein/ The wasserstein distance between the target and predicted normalized sample entropy batch distributions, per channel.
    - /error/sample_entropy_wasserstein: The mean of the wasserstein distances between the target and predicted normalized sample entropy batch distributions, averaged over all channels.
    - /error/<channelname>/sample_entropy_msd/ The mean signed difference (MSD) between the paired target and predicted sample entropy, per channel. Essentially estimating the bias of the prediction.
    - /error/<channelname>/sample_entropy_mse/ The mean squared error between the paired target and predicted normalized sample entropy batch distributions, per channel.
    - /error/sample_entropy_mse: The mean of the mean squared errors between the paired target and predicted normalized sample entropy batch distributions, averaged over all channels.
    - /error/<channelname>/sample_entropy_mae/ The mean absolute error between the paired target and predicted normalized sample entropy batch distributions, per channel.
    - /error/sample_entropy_mae: The mean of the mean absolute errors between the paired target and predicted normalized sample entropy batch distributions, averaged over all channels.
    - /error/<channelname>/kl_divergence/ The Kullback-Leibler divergence between the target and predicted timeseries, per channel.
    """

    C = get_current_config()
    channel_names = C.data.cols.x
    entropy_target, entropy_pred = entropy.get_normalized_entropies(pred, target)
    pair_errors = entropy_pred - entropy_target
    msd_channel = np.mean(pair_errors, axis=0)
    msd = np.mean(pair_errors)
    mse_channel = np.mean(np.power(pair_errors, 2), axis=0)
    mse = np.mean(np.power(pair_errors, 2))
    mae_channel = np.mean(np.abs(pair_errors), axis=0)
    mae = np.mean(np.abs(pair_errors))

    wasserstein_sum = 0  # Wasserstein distance needs to be averaged after the channel loop
    kl_divergence_sum = 0  # KL divergence needs to be averaged after the channel loop
    metrics = {}
    for i, channel_name in enumerate(channel_names):
        wasserstein_per_channel = wasserstein_distance(entropy_target[:, i], entropy_pred[:, i])
        # kl_divergence_per_channel = np.sum(rel_entr(target[:, i], pred[:, i]))
        wasserstein_sum += wasserstein_per_channel
        # kl_divergence_sum += kl_divergence_per_channel
        metrics[f"/error/{channel_name}/sample_entropy_wasserstein"] = wasserstein_per_channel
        # metrics[f"/error/{channel_name}/kl_divergence"] = kl_divergence_per_channel
        metrics[f"/error/{channel_name}/sample_entropy_msd"] = msd_channel[i]
        metrics[f"/error/{channel_name}/sample_entropy_mse"] = mse_channel[i]
        metrics[f"/error/{channel_name}/sample_entropy_mae"] = mae_channel[i]
    metrics["/error/sample_entropy_wasserstein"] = wasserstein_sum / len(channel_names)
    # metrics["/error/kl_divergence"] = kl_divergence_sum / len(channel_names)
    metrics["/error/sample_entropy_msd"] = msd
    metrics["/error/sample_entropy_mse"] = mse
    metrics["/error/sample_entropy_mae"] = mae
    return metrics
