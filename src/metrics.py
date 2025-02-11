#%%
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
wandb.define_metric("loss/time_domain_train", summary="min")
wandb.define_metric("loss/time_domain_val", summary="min")

wandb.define_metric("val/time_pred_batch_variance", summary="max")
wandb.define_metric("val/time_pred_batch_var_mean_adjusted", summary="max")
wandb.define_metric("val/freq_pred_batch_variance", summary="max")
wandb.define_metric("val/freq_pred_batch_var_mean_adjusted", summary="max")
wandb.define_metric("val/freq_pred_batch_input_variance_ratio", summary="max")

wandb.define_metric("val/time_target_batch_variance", summary="max")
wandb.define_metric("val/time_target_batch_var_mean_adjusted", summary="max")
wandb.define_metric("val/freq_target_batch_variance", summary="max")
wandb.define_metric("val/freq_target_batch_var_mean_adjusted", summary="max")
wandb.define_metric("val/freq_target_batch_input_variance_ratio", summary="max")
#### -- New Metrics -- ####
wandb.define_metric("val/error/mean_magnitude_mse", summary="min")
wandb.define_metric("val/error/var_magnitude_mse", summary="min")
wandb.define_metric("val/error/skew_magnitude_mse", summary="min")
wandb.define_metric("val/error/kurtosis_magnitude_mse", summary="min")


def moments(batched_time_series: torch.Tensor):
    mean = torch.mean(batched_time_series, -1, keepdim=True)
    diffs = batched_time_series - mean
    var = torch.mean(torch.pow(diffs, 2), -1, keepdim=True)
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0), - 1, keepdim=True)
    kurtoses = torch.mean(torch.pow(zscores, 4.0), -1, keepdim=True) - 3.0
    return mean, var, skews, kurtoses

def get_moments_errors(pred: torch.Tensor, target: torch.Tensor):
    mean_pred, var_pred, skew_pred, kurtosis_pred = moments(pred)
    mean_target, var_target, skew_target, kurtosis_target = moments(target)
    mean_mse = torch.mean(torch.pow(mean_pred - mean_target, 2))
    var_mse = torch.mean(torch.pow(var_pred - var_target, 2))
    skew_mse = torch.mean(torch.pow(skew_pred - skew_target, 2))
    kurtosis_mse = torch.mean(torch.pow(kurtosis_pred - kurtosis_target, 2))
    return {
        "/error/mean_magnitude_mse": mean_mse.item(),
        "/error/var_magnitude_mse": var_mse.item(),
        "/error/skew_magnitude_mse": skew_mse.item(),
        "/error/kurtosis_magnitude_mse": kurtosis_mse.item(),
    }
