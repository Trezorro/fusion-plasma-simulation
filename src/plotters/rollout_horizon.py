"""Horizon-resolved rollout analysis: metrics per autoregressive depth k, shared logic.

Used from two places with identical output:
  * in-run, by src.rollout.run_rollouts (rollout.analysis config key), right after the
    rollout cache is written, so a finished run comes back with the figures ready;
  * post-hoc, by eval_notebooks/rollout_analysis.py, to re-slice or overlay several
    models from their caches without a GPU.

Definitions, aggregation rationale (facet per start fraction, never pool shot phases),
and a reading guide live in docs/evaluation-metrics.md section 7 and docs/plots.md.
"""
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

# Print styling applied around figure creation (rc_context, so nothing global leaks),
# matching eval_notebooks/paper_single_variate.py.
RC_PARAMS = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.6,
    "axes.edgecolor": "0.3",
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.2,
    "figure.dpi": 120,
}

# Okabe-Ito, one color per model; black is reserved for the real reference line.
MODEL_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

# Per-panel sizes (width_per_panel, height) in inches; exported once per size so fonts
# scale relative to the figure, like the other paper exports.
PANEL_SIZES = ((2.6, 3.2), (3.6, 4.4), (5.0, 6.0))


def window_metric_rows(record: dict, channel_names, pd_index, elm_prominence) -> list[dict]:
    """One row of metrics per autoregressive window index k of one rollout record.

    Definitions (all against the real trace over the same span, normalized [0,1]):
    abs_mean_err/{ch} = |mean(gen) - mean(real)|, abs_std_err/{ch} = |std(gen) - std(real)|,
    label_agreement = fraction of samples where FNOLSTM(gen) == FNOLSTM(real) (unshifted),
    elm_peaks_* = find_peaks count on the window's PD at the ELM prominence.
    """
    step = int(record["step"])
    n_windows = int(record["n_windows"])
    # seq_length recovered from the kept-trace layout: T = (n_windows-1)*step + seq_length
    seq_length = record["generated_x"].shape[-1] - (n_windows - 1) * step
    hl = int(record["history_length"])
    gen = record["generated_x"]
    real = record["real_x"][:, hl:]
    lab_gen = record["surr_labels_gen"][hl:]
    lab_real = record["surr_labels_real"][hl:]
    rows = []
    for k in range(n_windows):
        sl = slice(k * step, k * step + seq_length)
        row = {
            "shot": record["shot_number"],
            "start_frac": record["start_frac"],
            "sample_idx": record["sample_idx"],
            "k": k,
            "label_agreement": float((lab_gen[sl] == lab_real[sl]).mean()),
            "elm_peaks_gen": len(find_peaks(gen[pd_index, sl], prominence=elm_prominence)[0]),
            "elm_peaks_real": len(find_peaks(real[pd_index, sl], prominence=elm_prominence)[0]),
        }
        for ch, name in enumerate(channel_names):
            row[f"abs_mean_err/{name}"] = float(abs(gen[ch, sl].mean() - real[ch, sl].mean()))
            row[f"abs_std_err/{name}"] = float(abs(gen[ch, sl].std() - real[ch, sl].std()))
        rows.append(row)
    return rows


def build_horizon_df(model_records: list[tuple[str, list[dict]]], channel_names, pd_index,
                     elm_prominence) -> pd.DataFrame:
    """Long per-(model, rollout, k) dataframe over one or more models' rollout records."""
    frames = []
    for model_name, records in model_records:
        frame = pd.DataFrame([
            row for record in records
            for row in window_metric_rows(record, channel_names, pd_index, elm_prominence)
        ])
        frame["model"] = model_name
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["abs_elm_peaks_err"] = (df["elm_peaks_gen"] - df["elm_peaks_real"]).abs()
    return df


def aggregate_horizon(df: pd.DataFrame):
    """Median + IQR per (model, start fraction, depth k), plus the population per panel.

    Returns (med, q25, q75, counts, fracs). counts is per (start_frac, k) and identical
    across models since the rollout specs are shared.
    """
    metric_cols = [c for c in df.columns if c not in ("model", "shot", "start_frac", "sample_idx", "k")]
    groups = df.groupby(["model", "start_frac", "k"])[metric_cols]
    med, q25, q75 = groups.median(), groups.quantile(0.25), groups.quantile(0.75)
    first_model = df["model"].iloc[0]
    counts = df[df["model"] == first_model].groupby(["start_frac", "k"]).size()
    fracs = sorted(df["start_frac"].unique())
    return med, q25, q75, counts, fracs


def horizon_figures(med, q25, q75, counts, fracs, model_names, channel_names, elm_prominence,
                    pdf_dir, panel_sizes=PANEL_SIZES, wandb_prefix=None):
    """Write every faceted horizon figure at each panel size; optionally log to wandb.

    Panels = start fractions (shot phases are never pooled); line = median over that
    panel's rollouts at depth k, one line per model; band = IQR over the same rollouts;
    grey step (right axis) = n(k). See docs/plots.md for the reading guide.
    """
    pdf_dir = Path(pdf_dir)

    def _one_figure(col, title, fname, ylabel, real_col=None):
        for size_i, (panel_w, height) in enumerate(panel_sizes):
            fig, axes = plt.subplots(
                1, len(fracs), sharey=True, figsize=(panel_w * len(fracs) + 1.2, height), squeeze=False
            )
            for ax, frac in zip(axes[0], fracs):
                for model_name, color in zip(model_names, MODEL_COLORS):
                    try:
                        m = med.loc[(model_name, frac)][col]
                    except KeyError:
                        continue  # this model has no rollouts at this start fraction
                    ax.plot(m.index, m.values, color=color, label=model_name)
                    ax.fill_between(
                        m.index, q25.loc[(model_name, frac)][col], q75.loc[(model_name, frac)][col],
                        color=color, alpha=0.15, linewidth=0,
                    )
                if real_col is not None:
                    r = med.loc[(model_names[0], frac)][real_col]
                    ax.plot(r.index, r.values, color="black", linestyle="--", linewidth=1.0, label="real")
                ax_n = ax.twinx()
                n = counts.loc[frac]
                ax_n.step(n.index, n.values, where="mid", color="0.65", linewidth=0.8)
                ax_n.set_ylim(bottom=0)
                ax_n.spines["top"].set_visible(False)
                if ax is axes[0][-1]:
                    ax_n.set_ylabel("n rollouts at depth $k$", color="0.45", fontsize=8)
                ax_n.tick_params(axis="y", colors="0.45", labelsize=7)
                ax.set_title(f"start at {frac:.0%} of shot", fontsize=9)
                ax.set_xlabel("depth $k$")
                ax.set_facecolor("#F7F7F7")
                ax.grid(color="white", linewidth=1.0)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
            axes[0][0].set_ylabel(ylabel)
            axes[0][0].legend(frameon=False, fontsize=7)
            fig.suptitle(f"{title}\nper start fraction; line = median over rollouts, band = IQR", fontsize=10)
            fig.tight_layout()
            w, h = fig.get_size_inches()
            out = pdf_dir / f"{w:.0f}x{h:.0f}" / f"{fname}.pdf"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, bbox_inches="tight")
            logger.info("Wrote horizon figure %s", out)
            if wandb_prefix and size_i == 0:
                import wandb
                if wandb.run is not None and not wandb.run.disabled:
                    wandb.log({f"{wandb_prefix}/{fname}": wandb.Image(fig)}, commit=False)
            plt.close(fig)

    with mpl.rc_context(RC_PARAMS):
        for ch_name in channel_names:
            _one_figure(f"abs_mean_err/{ch_name}", f"{ch_name}: mean error vs rollout depth",
                        f"abs_mean_err_{ch_name}", r"$|\,\mu_{gen} - \mu_{real}\,|$ (norm. units)")
            _one_figure(f"abs_std_err/{ch_name}", f"{ch_name}: spread error vs rollout depth",
                        f"abs_std_err_{ch_name}", r"$|\,\sigma_{gen} - \sigma_{real}\,|$ (norm. units)")
        _one_figure("label_agreement", "Surrogate mode-label agreement vs rollout depth",
                    "label_agreement", "fraction of window where\nFNOLSTM(gen) = FNOLSTM(real)")
        _one_figure("elm_peaks_gen", "ELM-scale PD peak rate vs rollout depth",
                    "elm_peaks", f"peaks per window (prominence {elm_prominence})",
                    real_col="elm_peaks_real")


def horizon_table(med, counts, fracs, main_model, depths=(0, 1, 2, 4, 8, 16, 32)) -> pd.DataFrame:
    """Compact table for one model: rows = (start fraction, selected depths)."""
    table_cols = ["label_agreement", "abs_mean_err/PD", "abs_std_err/PD", "elm_peaks_gen", "elm_peaks_real"]
    rows = []
    for frac in fracs:
        block = med.loc[(main_model, frac)]
        for k in (d for d in depths if d in block.index):
            rows.append({"start_frac": frac, "k": k, "n": counts.loc[(frac, k)], **block.loc[k, table_cols]})
    return pd.DataFrame(rows).set_index(["start_frac", "k"])


def export_horizon_analysis(model_records, channel_names, pd_index, elm_prominence, pdf_dir,
                            table_dir=None, wandb_prefix=None) -> pd.DataFrame:
    """Full pipeline: df -> aggregates -> figures + csv + tex. Returns the per-(rollout, k) df.

    Args:
        model_records: list of (model_name, records) pairs; records from
            src.rollout.build_rollout_records.
        channel_names, pd_index, elm_prominence: data/config context.
        pdf_dir: figures go to {pdf_dir}/{WxH}/{metric}.pdf.
        table_dir: csv + tex destination; defaults to pdf_dir.
        wandb_prefix: if set, figures (smallest size) are also logged to wandb.
    """
    table_dir = Path(table_dir or pdf_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    df = build_horizon_df(model_records, channel_names, pd_index, elm_prominence)
    med, q25, q75, counts, fracs = aggregate_horizon(df)
    med.join(counts.rename("n_rollouts"), on=["start_frac", "k"]).to_csv(table_dir / "rollout_horizon.csv")
    horizon_figures(med, q25, q75, counts, fracs, [name for name, _ in model_records],
                    channel_names, elm_prominence, pdf_dir, wandb_prefix=wandb_prefix)
    table = horizon_table(med, counts, fracs, main_model=model_records[0][0])
    table.to_latex(table_dir / "rollout_horizon.tex", float_format="%.3f")
    logger.info("Wrote horizon csv + tex to %s", table_dir)
    return df
