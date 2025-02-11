import torch
# import torchmetrics.audio as audio
import wandb
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
wandb.define_metric("loss/train", summary="min")
wandb.define_metric("loss/val", summary="min")

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
