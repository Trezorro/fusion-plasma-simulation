#%%
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
        self.history_length = self.C.data.history_length
        self.future_length = self.C.data.seq_length
        self.add_state(f"transition_counts_pred", default=torch.zeros(1,3,3), dist_reduce_fx="cat")
        self.add_state(f"transition_counts_target", default=torch.zeros(1,3,3), dist_reduce_fx="cat")
        self.add_state(f"transition_counts_sq_error", default=torch.zeros(3,3), dist_reduce_fx="sum")
        self.add_state(f"total_hits", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, surr_labels_pred, surr_labels_target):
        # history should of course be the same as it is conditioning information
        history = surr_labels_target[:, :self.history_length]
        future_pred = surr_labels_pred[:, self.history_length:]
        future_target = surr_labels_target[:, self.history_length:]

        # Get mask for samples that satisfy the condition
        mask = self.test_condition(history)

        # Only keep samples where mask is True and at least one sample is selected
        if mask.any():
            # Compute transition matrices for pred and target
            pred_trans = transition_matrix(future_pred[mask])  # (M, 3, 3)
            target_trans = transition_matrix(future_target[mask])  # (M, 3, 3)
            sq_error = torch.square(pred_trans-target_trans)

            # Sum over batch and add to state
            self.transition_counts_pred = torch.concat((self.transition_counts_pred, pred_trans), dim=0)
            self.transition_counts_target = torch.concat((self.transition_counts_target, target_trans), dim=0)
            # self.transition_counts_pred += pred_trans.sum(dim=0, keepdim=True)
            # self.transition_counts_target += target_trans.sum(dim=0, keepdim=True)
            self.transition_counts_sq_error += sq_error.sum(dim=0)
            self.total_hits += mask.sum()


    def test_condition(self, history):
        """Return a batch mask, where 1 is a sample that satisfies the history condition."""
        if self.condition in ("any_Wh", "any"):
            return torch.ones(history.size(0), dtype=torch.bool, device=history.device)  # All True
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

    def compute(self):
        """Output expected number of transitions and P(eta>0) for each from-to pair, for pred, target, and squared error views."""
        out = {}
        # Remove the batch dimension (cat) if present
        pred_counts = self.transition_counts_pred.sum(dim=0)
        target_counts = self.transition_counts_target.sum(dim=0)
        sq_error_counts = self.transition_counts_sq_error  # already summed in update
        total_hits = self.total_hits.item() if hasattr(self.total_hits, 'item') else float(self.total_hits)

        for from_idx, from_name in enumerate(["L", "D", "H"]):
            for to_idx, to_name in enumerate(["L", "D", "H"]):
                key_base = f"{self.condition}/from_{from_name}/to_{to_name}"
                pred_val = pred_counts[from_idx, to_idx].item()
                target_val = target_counts[from_idx, to_idx].item()
                sq_err_val = sq_error_counts[from_idx, to_idx].item()
                # Expectation (mean number of transitions per sample)
                out[f"{key_base}/expect/pred"] = pred_val / max(total_hits, 1)
                out[f"{key_base}/expect/target"] = target_val / max(total_hits, 1)
                out[f"{key_base}/expect/sq_err"] = sq_err_val / max(total_hits, 1)
                # Probability of at least one transition (gt0)
                out[f"{key_base}/p_gt0/pred"] = float(pred_val > 0)
                out[f"{key_base}/p_gt0/target"] = float(target_val > 0)
                out[f"{key_base}/p_gt0/sq_err"] = float(sq_err_val > 0)
        return out

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
