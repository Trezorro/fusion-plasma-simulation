# %%
import json
import pathlib

import numpy as np
import pandas as pd
import torch

from src.models.LDH_model import FNOLSTM
from src.config import get_current_config
import logging

logger = logging.getLogger(__name__)
from tqdm import tqdm

# %%
# global model settings
MODEL_METADATA_DIR = pathlib.Path("configs/MHD_model_yoerie")
TW = 40
OFFSET_PRED = 20
STRIDE = 10
C = get_current_config()
train_shots = C.data.train_shots
test_shots = C.data.test_shots

with open(MODEL_METADATA_DIR / 'stats_PD.json', 'r') as f:
    stats_PD = json.load(f)
# %% Initialize mode segmentation model

model_PD = FNOLSTM(
    n_in=1,
    n_out=3,
    tw=TW,
    h_c1=32,
    h_c2=64,
    h_m1=8,
    h_m2=8,
    h_dropc=0.5,
    h_maxpool=2,
    h_lstm_in=32,
    h_lmst=32,
    h_mlp=8,
    m_dropmlp=0.5
)
model_PD.load_state_dict(torch.load(MODEL_METADATA_DIR / "weights_PD.pt"))


# %%
def normalize_input_multichannel(sig, signal_list, stats):
    assert all([s in stats for s in signal_list])
    columns = []
    for s in signal_list:
        columns.append(torch.tensor((sig[s].values - stats[s]["mean"]) / stats[s]["sd"]))
    return torch.tensor(sig.time), torch.stack(columns)


# %%
def pred_sample_slidingwindow(model, t, x, device, tw, stride, offset_pred, i_start=0):
    model.eval()
    x = x.to(device)
    if x.dim() == 2:
        x = x.unsqueeze(0)  # add the Batch dimension if it isn't there
    model.to(device)
    hidden = None
    i_max = x.shape[-1] - tw + 1  # max index for the sliding window
    y_preds = []
    y_times = []
    for k in range(i_start, i_max, stride):
        i_s = k  # start index
        i_e = k + tw  # end index
        i_time = k + tw - 1 - offset_pred
        with torch.no_grad():
            y_pred, hidden = model(x[:, :, i_s:i_e], hidden)
        y_times.append(t[i_time])

        y_preds.append(y_pred)
    y_preds = torch.stack(y_preds, dim=1).argmax(dim=-1).to('cpu')
    y_times = torch.tensor(y_times, device='cpu')

    return y_times, y_preds


def clean_labels(label_t, surr_labels, history_length, seq_length):
    """Resample the labels such that we have a label for every step from -history to +seq_length"""
    resampled_series = pd.Series(data=surr_labels, index=label_t.numpy())
    # Extend the index to the full range and forward fill the series
    # cut to around the target timeline, interpolate nearest,
    # ffill the end (model doesn't predict over the edge),
    # #then cut to the exact plotting range:
    resampled_series = resampled_series.reindex(range(-history_length - STRIDE, seq_length)
                                               ).interpolate(method='nearest'
                                                            ).ffill().reindex(range(-history_length, seq_length))
    return resampled_series


def get_mode_predictions(
    pd_rollout_pred: torch.Tensor,
    pd_rollout_target: torch.Tensor,
    timeline: np.ndarray,
    history_length=0,
    seq_length=64
):
    """Get interpolated mode predictions ranging from 1 to 3 for the range from provided timeline around -history_length to + seq_length."""
    # dim should be (N, T)
    x = torch.stack((pd_rollout_pred, pd_rollout_target))
    COLNAME = 'PD'
    normalized_rollout = (x - stats_PD[COLNAME]["mean"]) / stats_PD[COLNAME]["sd"]
    if normalized_rollout.dim() == 2:
        normalized_rollout = normalized_rollout.unsqueeze(1)  # add the input channels dimension back

    t_out, y_out = pred_sample_slidingwindow(
        model_PD,
        t=timeline,
        x=normalized_rollout,
        device='cpu',
        tw=TW,
        stride=STRIDE,
        offset_pred=OFFSET_PRED,
        i_start=0
    )
    y_out_pred, y_out_target = y_out.unbind(0)
    surr_labels_pred = clean_labels(
        label_t=t_out, surr_labels=y_out_pred, history_length=history_length, seq_length=seq_length
    )
    surr_labels_target = clean_labels(
        label_t=t_out, surr_labels=y_out_target, history_length=history_length, seq_length=seq_length
    )
    return surr_labels_pred, surr_labels_target


def generate_surrogate_labels(meta, generated_samples, target_samples, data_module):
    """Dataset is needed for get history and denormalization."""
    C = get_current_config()
    shot_numbers = meta['shot_number'].cpu()
    prediction_window_starts_idx = meta['start_i']
    PD_index = C.data.cols.x.index("PD")
    history_length = C.data.history_length
    seq_length = C.data.seq_length
    target_samples_denorm = data_module.denormalize(target_samples.to('cpu'))
    generated_samples_denorm = data_module.denormalize(generated_samples.to('cpu'))
    surr_labels_pred = []
    surr_labels_target = []
    logger.info("Generating surrogate labels for batch of %d samples", len(shot_numbers))

    for i, shot_num in enumerate(
        tqdm(
            shot_numbers,
            disable=torch.cuda.is_available(),
            desc="Getting surrogate labels ",
        )
    ):
        full_history_raw = data_module.get_full_history(shot_num.item(), prediction_window_starts_idx[i].item())
        full_history = data_module.denormalize(full_history_raw)  # type: ignore
        target_pd_rollout = torch.concat((full_history, target_samples_denorm[i]), dim=-1)[PD_index]  # type: ignore
        predicted_pd_rollout = torch.concat((full_history, generated_samples_denorm[i]),
                                            dim=-1)[PD_index]  # type: ignore
        idx_timeline = np.arange(-prediction_window_starts_idx[i], seq_length)
        surr_labels_pred_i, surr_labels_target_i = get_mode_predictions(
            pd_rollout_pred=predicted_pd_rollout,
            pd_rollout_target=target_pd_rollout,
            timeline=idx_timeline,
            history_length=history_length,
            seq_length=seq_length
        )
        surr_labels_pred.append(surr_labels_pred_i.values)
        surr_labels_target.append(surr_labels_target_i.values)

    surr_labels_pred = np.stack(surr_labels_pred)
    surr_labels_target = np.stack(surr_labels_target)
    return surr_labels_pred, surr_labels_target


# surr_labels_target, surr_labels_pred = generate_surrogate_labels(get_mode_predictions, evaluation_output)
