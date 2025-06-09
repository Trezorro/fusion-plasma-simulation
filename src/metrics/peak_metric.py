#%%
import logging
import numpy as np
import torch
import torchmetrics
from src.config import get_current_config
from src.metrics.metrics import batch_get_peakprops, prefix_metrics
from scipy.stats import wasserstein_distance
import pandas as pd

logger = logging.getLogger(__name__)


#%%
class PeakMetric(torchmetrics.Metric):
    """Metric name scheme:
    `<data_set>/<on_condition>/peak_<measure>/<statistic>/<channel_name>`

    - `data_set` ∈ {`train`, `val`, `test`}
    - `on_condition`: Logical condition on the history window Wh, e.g. `L_only_Wh`, `mixed`, `H_not_in_Wh` or 'any_Wh'
    - `statistic` ∈ {`mean_pairwise_wasserstein`, `marginal_wasserstein`} — 
    - `view` ∈ {`target`, `pred`, `error`} — indicates whether the value is computed from the ground truth, model prediction, or their difference

    Instead of statistic and view, we may see the divergence as wasserstein or kl divergence.
    """
    full_state_update = False  # metric state of one batch is independent of the state of other batches

    CONDITION_OPTONS = [
        "L_only_Wh", "D_only_Wh", "H_only_Wh", "L_in_Wh", "D_in_Wh", "H_in_Wh", "L_not_in_Wh", "D_not_in_Wh",
        "H_not_in_Wh", "any_Wh", "mixed"
    ]
    VIEW_OPTIONS = ["target", "pred", "error", "sq_err"]
    BASE_MEASURES = [
        "height",
        "prominence",
        "base",
        "width",
    ]  # + count with pairwise mse and marginal wasserstein

    def __init__(
        self,
        condition_history: str = 'any_Wh',
    ):
        super().__init__()
        self.condition = condition_history
        self.C = get_current_config()
        self.CHANNEL_NAMES = self.C.data.cols.x
        self.dml_channel_index = self.CHANNEL_NAMES.index("DML") if "DML" in self.CHANNEL_NAMES else None
        self.pd_channel_index = self.CHANNEL_NAMES.index("PD") if "PD" in self.CHANNEL_NAMES else None
        self.history_length = self.C.data.history_length
        self.future_length = self.C.data.seq_length
        for channel_i, channel_name in enumerate(self.CHANNEL_NAMES):
            measures = self.BASE_MEASURES
            if channel_name == "DML":
                measures = measures + ["energy_delta", "pd_prominence", "energy_ratio"]
            # pairwise_distances = {f"/error/peak_{measure}/pairwise_wasserstein/{channel_name}": 0. for measure in measures}
            for measure in measures:
                self.add_state(f"pair_dists_sum:{channel_name}/{measure}", default=torch.zeros(1), dist_reduce_fx="sum")
                self.add_state(f"prop_list_pred:{channel_name}/{measure}", default=torch.zeros(0), dist_reduce_fx="cat")
                self.add_state(
                    f"prop_list_target:{channel_name}/{measure}", default=torch.zeros(0), dist_reduce_fx="cat"
                )

            self.add_state(
                f"list:{channel_name}/counts_pred", default=torch.zeros(0, dtype=torch.int32), dist_reduce_fx="cat"
            )
            self.add_state(
                f"list:{channel_name}/counts_target", default=torch.zeros(0, dtype=torch.int32), dist_reduce_fx="cat"
            )
            self.add_state(f"sum:{channel_name}/counts_sq_error", default=torch.zeros(1), dist_reduce_fx="sum")

        self.add_state(f"total_hits", default=torch.zeros(1, dtype=torch.int32), dist_reduce_fx="sum")

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

    def add_to_state(self, state_name: str, new_value):
        setattr(self, state_name, getattr(self, state_name) + new_value)

    def append_to_state(self, state_name: str, new_value):
        getattr(self, state_name).append(new_value)

    def cat_with_state(self, state_name: str, new_value):
        setattr(self, state_name, torch.cat((getattr(self, state_name), torch.tensor(new_value).view(-1))))

    def update(self, pred, target, labels):
        # history should of course be the same as it is conditioning information
        history_labels = labels[:, :self.history_length]

        # Get mask for samples that satisfy the condition
        sample_mask = self.test_condition(history_labels)
        num_selected_samples = int(sample_mask.sum().item())
        if num_selected_samples == 0:
            return
        self.total_hits += num_selected_samples
        # Only process samples where mask is True and at least one sample is selected
        logger.debug(
            "Computing pred and target peak properties for %d samples satisfying condition '%s'.", num_selected_samples,
            self.condition
        )
        pred_peaks_batch = batch_get_peakprops(
            pred[sample_mask], dml_channel_index=self.dml_channel_index, pd_channel_index=self.pd_channel_index
        )  # (B, C)
        target_peaks_batch = batch_get_peakprops(
            target[sample_mask], dml_channel_index=self.dml_channel_index, pd_channel_index=self.pd_channel_index
        )  # (B, C)
        logger.debug("Done. Updating state.")
        for channel_i, channel_name in enumerate(self.CHANNEL_NAMES):
            measures = self.BASE_MEASURES
            if channel_name == "DML":
                measures = measures + ["energy_delta", "pd_prominence", "energy_ratio"]
            # pairwise_distances = {f"/error/peak_{measure}/pairwise_wasserstein/{channel_name}": 0. for measure in measures}
            # counts_target = []
            # counts_pred = []
            # marginal_dist_pred = {measure: [] for measure in measures}
            # marginal_dist_target = {measure: [] for measure in measures}
            for sample_i in range(num_selected_samples):
                pred_sample = pred_peaks_batch[sample_i][channel_i]
                target_sample = target_peaks_batch[sample_i][channel_i]
                pair_wasserstein = pred_sample - target_sample  # Subtraction operator is overloaded with wasserstein distance
                pred_sample_num_peaks = pred_sample.num_peaks()
                target_sample_num_peaks = target_sample.num_peaks()
                self.cat_with_state(f"list:{channel_name}/counts_pred", pred_sample_num_peaks)
                self.cat_with_state(f"list:{channel_name}/counts_target", target_sample_num_peaks)
                self.add_to_state(
                    f"sum:{channel_name}/counts_sq_error", np.square(pred_sample_num_peaks - target_sample_num_peaks)
                )
                for measure in measures:
                    self.add_to_state(f"pair_dists_sum:{channel_name}/{measure}", getattr(pair_wasserstein, measure))
                    match measure:
                        case "height":
                            self.cat_with_state(f"prop_list_pred:{channel_name}/{measure}", pred_sample.Y)
                            self.cat_with_state(f"prop_list_target:{channel_name}/{measure}", target_sample.Y)
                        case "prominence":
                            self.cat_with_state(f"prop_list_pred:{channel_name}/{measure}", pred_sample.prominences)
                            self.cat_with_state(f"prop_list_target:{channel_name}/{measure}", target_sample.prominences)
                        case "base":
                            self.cat_with_state(f"prop_list_pred:{channel_name}/{measure}", pred_sample.bases)
                            self.cat_with_state(f"prop_list_target:{channel_name}/{measure}", target_sample.bases)
                        case "width":
                            self.cat_with_state(f"prop_list_pred:{channel_name}/{measure}", pred_sample.widths)
                            self.cat_with_state(f"prop_list_target:{channel_name}/{measure}", target_sample.widths)
                        case "energy_delta":
                            self.cat_with_state(f"prop_list_pred:{channel_name}/{measure}", pred_sample.energy_delta)
                            self.cat_with_state(
                                f"prop_list_target:{channel_name}/{measure}", target_sample.energy_delta
                            )
                        case "pd_prominence":
                            self.cat_with_state(f"prop_list_pred:{channel_name}/{measure}", pred_sample.pd_prominence)
                            self.cat_with_state(
                                f"prop_list_target:{channel_name}/{measure}", target_sample.pd_prominence
                            )
                        case "energy_ratio":
                            self.cat_with_state(f"prop_list_pred:{channel_name}/{measure}", pred_sample.energy_ratio)
                            self.cat_with_state(
                                f"prop_list_target:{channel_name}/{measure}", target_sample.energy_ratio
                            )
                        case _:
                            raise ValueError(f"Unknown measure: {measure}")
        logger.debug("State updated.")

    def compute(self):
        metrics_out = {}
        if self.total_hits == 0:
            return metrics_out
        logger.debug("Computing metrics for %d hits for condition %s", self.total_hits.item(), self.condition)

        for channel_i, channel_name in enumerate(self.CHANNEL_NAMES):
            measures = self.BASE_MEASURES
            if channel_name == "DML":
                measures = measures + ["energy_delta", "pd_prominence", "energy_ratio"]
            #  overal distribution of peak counts in windows:
            counts_pred = getattr(self, f"list:{channel_name}/counts_pred").cpu().numpy()
            counts_target = getattr(self, f"list:{channel_name}/counts_target").cpu().numpy()
            counts_pred = [0] if counts_pred.size == 0 else counts_pred
            counts_target = [0] if counts_target.size == 0 else counts_target
            metrics_out[f"peak_count/marginal_wasserstein/{channel_name}"] = wasserstein_distance(
                counts_pred, counts_target
            )
            # conditional error of peak counts:
            total_summed_counts_errors = getattr(self, f"sum:{channel_name}/counts_sq_error")
            metrics_out[f"peak_count/pairwise_mse/{channel_name}"] = total_summed_counts_errors / self.total_hits
            metrics_out[f"peak_count/pairwise_rmse/{channel_name}"] = torch.sqrt(
                total_summed_counts_errors / self.total_hits
            )
            # Average over the batch
            for measure in measures:
                metrics_out[f"peak_{measure}/pairwise_wasserstein/{channel_name}"
                           ] = getattr(self, f"pair_dists_sum:{channel_name}/{measure}") / self.total_hits
                # Overall distributions per channel per measure:
                pred_property_list = getattr(self, f"prop_list_pred:{channel_name}/{measure}")
                target_property_list = getattr(self, f"prop_list_target:{channel_name}/{measure}")

                # Convert to numpy arrays if they are tensors
                if torch.is_tensor(pred_property_list):
                    pred_property_list = pred_property_list.cpu().numpy()
                if torch.is_tensor(target_property_list):
                    target_property_list = target_property_list.cpu().numpy()

                # Handle empty or 0-d arrays
                if pred_property_list.size == 0 or pred_property_list.ndim == 0:
                    pred_property_list = np.array([0])
                if target_property_list.size == 0 or target_property_list.ndim == 0:
                    target_property_list = np.array([0])

                metrics_out[f"peak_{measure}/marginal_wasserstein/{channel_name}"] = wasserstein_distance(
                    pred_property_list, target_property_list
                )
        metrics_out['total_hits'] = self.total_hits.item()
        logger.debug("Done %d hits for condition %s", self.total_hits.item(), self.condition)
        return prefix_metrics(metrics_out, self.condition)

    def extract_df_all(self, cache_obj=None):
        base_df = pd.DataFrame(columns=['condition', 'channel_name', 'measure', 'distribution', 'value'])
        if self.total_hits == 0:
            return base_df
        for channel_i, channel_name in enumerate(self.CHANNEL_NAMES):
            measures = self.BASE_MEASURES
            if channel_name == "DML":
                measures = measures + ["energy_delta", "pd_prominence", "energy_ratio"]
            for measure in measures:
                base_df = pd.concat((base_df, self.extract_df(channel_name, measure)))
        if cache_obj is not None:
            base_df.to_hdf(cache_obj.h5_path, key=f'peaks/{self.condition}', mode='a')
        return base_df

    def extract_df(self, channel_name, measure):
        """
        Extract a DataFrame with the peak properties for the given channel and measure.
        Returns a DataFrame with columns: 'pred', 'target', 'error' (if applicable).
        """

        if measure.startswith('count'):
            pred_list = getattr(self, f"list:{channel_name}/counts_pred").cpu().numpy()
            target_list = getattr(self, f"list:{channel_name}/counts_target").cpu().numpy()
        else:
            pred_list = getattr(self, f"prop_list_pred:{channel_name}/{measure}").cpu().numpy()
            target_list = getattr(self, f"prop_list_target:{channel_name}/{measure}").cpu().numpy()
        # Data frame: Distribution, condition, value
        pred_df = pd.DataFrame(
            dict(
                condition=self.condition,
                channel_name=channel_name,
                measure=measure,
                distribution='Generated',
                value=pred_list
            )
        )
        target_df = pd.DataFrame(
            dict(
                condition=self.condition,
                channel_name=channel_name,
                measure=measure,
                distribution='Real',
                value=target_list
            )
        )
        out_df = pd.concat([pred_df, target_df], ignore_index=True)
        return out_df

    def make_histogram(
        self,
        channel_name,
        measure,
        df=None,
    ):
        import plotly.express as px
        import wandb
        from src.plotters.helpers import as_wandb_image
        COLOR_SCALE = ['#636EFA', '#00CC96', '#EF553B', "#999999"]
        if df is None:
            df = self.extract_df(channel_name, measure)
        subtitle = ''
        fig = px.histogram(
            df,
            x="value",
            color="condition",
            barmode="stack",
            facet_col="distribution",
            histnorm="probability density",
            color_discrete_sequence=COLOR_SCALE,
            title=f"Histogram of Peak <b>{measure.capitalize()}s</b> "
            f"for Channel: <b>{channel_name}</b>{subtitle}",
            labels={
                "value": f"<b>{measure.capitalize()}s</b>",
                "condition": "condition $W_H$",
                "distribution": "Distribution",
            },
            marginal="violin",
            nbins=100,
            category_orders={
                "condition": ['L_only_Wh', 'D_only_Wh', 'H_only_Wh', 'mixed'],
                "distribution": ["Generated", "Real"],
            },
        )

        fig.update_layout(
            hovermode="x",
            # legend_title_text="Mode",
            margin=dict(l=20, r=20, t=120, b=20),
            violingap=0,
            violingroupgap=0,
            violinmode='overlay',
            title=dict(font=dict(size=18)),
            width=1300,  # Increase plot width
            height=910,  # Increase plot height
        )
        fig.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[1]}</b>") if '=' in a.text else None)
        wandb_image = as_wandb_image(fig, format="png", show=wandb.run.disabled)
        return wandb_image
