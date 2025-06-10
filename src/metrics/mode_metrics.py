#%%
import pandas as pd
import torch
from src.config import get_current_config
import torchmetrics


#%%
class ModeTransitionMetric(torchmetrics.Metric):
    """Metric name scheme:
    `<data_set>/<on_condition>/<from>/<to>/<statistic>/<view>`

    - `data_set` ∈ {`train`, `val`, `test`}
    - `on_condition`: Logical condition on the history window Wh, e.g. `L_only_Wh`, `D_in_Wh`, `H_not_in_Wh` or 'any_Wh'
    - `from`, `to`: Initial and resulting mode states (e.g. `from_L`, `to_H`, `from_any`)
    - `statistic` ∈ {`expect`, `p_gt0`} — representing expected number of transitions or probability of any transition occurring
    - `view` ∈ {`target`, `pred`, `error`} — indicates whether the value is computed from the ground truth, model prediction, or their difference

    Instead of statistic and view, we may see the divergence as wasserstein or kl divergence.
    """
    full_state_update = False  # metric state of one batch is independent of the state of other batches

    CONDITION_OPTONS = [
        "L_only_Wh", "D_only_Wh", "H_only_Wh", "L_in_Wh", "D_in_Wh", "H_in_Wh", "L_not_in_Wh", "D_not_in_Wh",
        "H_not_in_Wh", "any_Wh"
    ]
    FROM_OPTIONS = ["from_L", "from_D", "from_H", "from_any"]
    TO_OPTIONS = ["to_L", "to_D", "to_H", "to_any"]
    STAT_OPTIONS = [
        "expect",
        "p_gt0",
        "div"  # divergence
    ]
    VIEW_OPTIONS = ["target", "pred", "error", "sq_err"]

    def __init__(
        self,
        condition_history: str = 'any_Wh',
    ):
        super().__init__()
        self.condition = condition_history
        self.C = get_current_config()
        WINDOW_OF_INFLUENCE_SPILL = min(
            15, self.C.data.history_length
        )  # the surrogate model looks ahead 15 steps past where it assigns a label.
        self.history_length = self.C.data.history_length - WINDOW_OF_INFLUENCE_SPILL
        self.future_length = self.C.data.seq_length + WINDOW_OF_INFLUENCE_SPILL
        self.add_state(f"transition_counts_pred", default=torch.zeros(4, 4), dist_reduce_fx="sum")
        self.add_state(f"transition_gt0_pred", default=torch.zeros(4, 4), dist_reduce_fx="sum")
        self.add_state(f"transition_counts_target", default=torch.zeros(4, 4), dist_reduce_fx="sum")
        self.add_state(f"transition_gt0_target", default=torch.zeros(4, 4), dist_reduce_fx="sum")
        self.add_state(f"transition_counts_sq_error", default=torch.zeros(4, 4), dist_reduce_fx="sum")
        self.add_state(f"total_hits", default=torch.zeros(1, dtype=torch.int32), dist_reduce_fx="sum")

    def update(self, surr_labels_pred, surr_labels_target):
        # history should of course be the same as it is conditioning information
        history = surr_labels_target[:, :self.history_length]
        future_pred_y = surr_labels_pred[:, self.history_length:]
        future_target_y = surr_labels_target[:, self.history_length:]

        # Get mask for samples that satisfy the condition
        sample_mask = self.test_condition(history)

        # Only keep samples where mask is True and at least one sample is selected
        if sample_mask.any():
            # Compute transition matrices for pred and target
            pred_trans = transition_matrix(future_pred_y[sample_mask])  # (M, 3, 3)
            target_trans = transition_matrix(future_target_y[sample_mask])  # (M, 3, 3)
            expanded_pred = self._expand_transition_matrix(pred_trans)  # (M, 4 ,4)
            expanded_target = self._expand_transition_matrix(target_trans)  # (M, 4 ,4)

            # Sum over batch and add to state
            self.transition_counts_pred += expanded_pred.sum(dim=0)
            self.transition_gt0_pred += (expanded_pred > 0).sum(dim=0)
            self.transition_counts_target += expanded_target.sum(dim=0)
            self.transition_gt0_target += (expanded_target > 0).sum(dim=0)
            self.transition_counts_sq_error += torch.square(expanded_pred - expanded_target).sum(
                dim=0
            )  # calculate to/from any totals before calculating error
            self.total_hits += sample_mask.sum()

    def test_condition(self, history):
        """Return a batch mask, where 1 is a sample that satisfies the history condition."""
        if self.condition in ("any_Wh", "any"):
            return torch.ones(history.size(0), dtype=torch.bool, device=history.device)  # All True
        if self.condition == "mixed":  # if there are multiple modes in the history
            first_entry = history[:, 0:1]
            is_mixed = (history[:, 1:] != first_entry).any(dim=1)
            return is_mixed
        # e.g., "L_only_Wh" -> mode="L", op="only_Wh"
        mode, op = self.condition.split("_", 1)
        mode_map = {"L": 0, "D": 1, "H": 2}
        mode_idx = mode_map[mode]
        mode_presence_mask = (history == mode_idx)
        if op.startswith("in"):
            # Check if mode_idx is present anywhere in each sample's history
            return mode_presence_mask.any(dim=1)
        elif op.startswith("not_in"):
            # Check if mode_idx is not present anywhere in each sample's history
            return ~mode_presence_mask.any(dim=1)
        elif op.startswith("only"):
            # Check if mode_idx is present in all time steps of each sample's history
            return mode_presence_mask.all(dim=1)
        else:
            raise ValueError(f"Unknown condition operation: {op}")

    @staticmethod
    def _expand_transition_matrix(transition_counts_matrix: torch.Tensor):
        """Add a 4th row/column for from_any/to_any. Supports optional batch dim.

        Only off-diagonal transitions are counted. i.e. Not L->L, D->D, H->H
        """
        if transition_counts_matrix.dim() == 2:
            # Single matrix, shape (3,3)
            mat = transition_counts_matrix.clone()
            mat.fill_diagonal_(0)
            expanded = torch.zeros(4, 4, dtype=mat.dtype, device=mat.device)
            expanded[:3, :3] = mat
            expanded[3, :3] = mat.sum(dim=0)
            expanded[:3, 3] = mat.sum(dim=1)
            expanded[3, 3] = mat.sum()
            return expanded
        elif transition_counts_matrix.dim() == 3:
            # Batched, shape (B,3,3)
            B = transition_counts_matrix.shape[0]
            mat = transition_counts_matrix.clone()
            # Zero diagonals for each batch
            idx = torch.arange(3, device=mat.device)
            mat[:, idx, idx] = 0
            expanded = torch.zeros(B, 4, 4, dtype=mat.dtype, device=mat.device)
            expanded[:, :3, :3] = mat
            expanded[:, 3, :3] = mat.sum(dim=1)
            expanded[:, :3, 3] = mat.sum(dim=2)
            expanded[:, 3, 3] = mat.sum(dim=(1, 2))
            return expanded
        else:
            raise ValueError("Input must be 2D or 3D tensor of shape (3,3) or (B,3,3)")

    def compute(self):
        """Output expected number of transitions and P(eta>0) for each from-to pair, for pred, target, and squared error views."""
        # Reduce over the super-batch dimension (concatenate dim) and calculate from/to any totals
        total_hits = self.total_hits.item() if hasattr(self.total_hits, 'item') else float(self.total_hits)
        out = {f"{self.condition}/total_hits": total_hits}
        if total_hits == 0:
            return out
        n = max(total_hits, 1)

        # Difference in Proportions with confidence interval
        p1 = self.transition_gt0_pred / n
        p2 = self.transition_gt0_target / n
        delta = p1 - p2
        # Manual SE + CI
        SE = torch.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
        # CI_lower = delta - 1.96 * SE
        # CI_upper = delta + 1.96 * SE

        for from_idx, from_name in enumerate(["L", "D", "H", "any"]):
            for to_idx, to_name in enumerate(["L", "D", "H", "any"]):
                key_base = f"{self.condition}/from_{from_name}/to_{to_name}"
                pred_sum_entry = self.transition_counts_pred[from_idx, to_idx].item()
                sq_err_val = self.transition_counts_sq_error[from_idx, to_idx].item()
                target_sum_entry = self.transition_counts_target[from_idx, to_idx].item()
                pred_gt0_entry = self.transition_gt0_pred[from_idx, to_idx].item()
                target_gt0_entry = self.transition_gt0_target[from_idx, to_idx].item()
                delta_entry = delta[from_idx, to_idx].item()
                SE_entry = SE[from_idx, to_idx].item()
                # Expectation (mean number of transitions per sample)
                out[f"{key_base}/expect/pred"] = pred_sum_entry / n
                out[f"{key_base}/expect/target"] = target_sum_entry / n
                out[f"{key_base}/expect/sq_err"] = sq_err_val / n
                # Probability of at least one transition (gt0)
                out[f"{key_base}/p_gt0/pred"] = pred_gt0_entry / n
                out[f"{key_base}/p_gt0/target"] = target_gt0_entry / n
                out[f"{key_base}/p_gt0/delta_CI_lower"] = delta_entry - 1.96 * SE_entry
                out[f"{key_base}/p_gt0/delta_CI_upper"] = delta_entry + 1.96 * SE_entry
                out[f"{key_base}/p_gt0/delta"] = delta_entry
                out[f"{key_base}/p_gt0/match"] = abs(delta_entry) <= 1.96 * SE_entry
        return out

    def extract_df_all(self, cache_obj=None):
        if self.total_hits == 0:
            return pd.DataFrame(columns=["L", "D", "H", "any"], index=["L", "D", "H", "any"])
        p1 = self.transition_gt0_pred / self.total_hits
        p2 = self.transition_gt0_target / self.total_hits
        delta = p1 - p2
        SE = torch.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / self.total_hits)
        lower = delta - 1.96 * SE
        upper = delta + 1.96 * SE
        # Manual SE + CI
        # Cache delta, SE, lower, upper if cache_obj is provided
        if cache_obj is not None:
            for name, arr in zip(['delta', 'SE', 'CI_lower', 'CI_upper'], [delta, SE, lower, upper]):
                df = pd.DataFrame(
                    columns=["L", "D", "H", "any"],
                    index=["L", "D", "H", "any"],
                    data=arr.cpu().numpy() if hasattr(arr, 'cpu') else arr
                )
                df.to_hdf(cache_obj.h5_path, key=f'modes/{self.condition}/{name}', mode='a')
        for state in [
            "transition_counts_pred", "transition_counts_target", "transition_counts_sq_error", "transition_gt0_pred",
            "transition_gt0_target"
        ]:
            df = pd.DataFrame(
                columns=["L", "D", "H", "any"],
                index=["L", "D", "H", "any"],
                data=getattr(self, state) / self.total_hits
            )
            if cache_obj is not None:
                df.to_hdf(cache_obj.h5_path, key=f'modes/{self.condition}/{state}', mode='a')

#%%
def transition_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Compute per-sample 3x3 transition matrices.
    
    x: LongTensor of shape (N, T), values in {0, 1, 2}
    Returns: LongTensor of shape (N, 3, 3)
    """
    N, T = x.shape

    assert T >= 2, "Need at least two time steps to compute transitions"

    from_vals = x[:, :-1]  # (N, T-1) in {0,1,2}
    to_vals = x[:, 1:]  # (N, T-1)  in {0,1,2}

    flat_idx = from_vals * 3 + to_vals  # (N, T-1)
    sample_idx = torch.arange(N, device=x.device).repeat_interleave(T - 1) # make a flat

    counts = torch.bincount(sample_idx * 9 + flat_idx.flatten(), minlength=N * 9)
    counts = counts.view(N, 3,3)

    return counts

# %%
def test_transition_matrix():

    x = torch.tensor([
        [0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [2, 2, 1, 1, 2, 2],
    ], dtype=torch.long)

    out = transition_matrix(x)
    print(out)
    assert (out == torch.tensor(
        [[[2, 1, 0],
         [0, 2, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 5, 0],
         [0, 0, 0]],

        [[5, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 1, 1],
         [0, 1, 2],],],)).all()
# test_transition_matrix()
# %%
