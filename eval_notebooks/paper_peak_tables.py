"""Paper peak-metric LaTeX tables: pairwise-only, for the single-variate model set.

Scientific output: the paper's peak-metric tables, a simplified descendant of the thesis tables in
    peaks_tables.py. Two differences from the thesis version: the marginal ("global") Wasserstein
    columns are dropped, so only the paired per-window metrics remain, and only three channels are
    exported instead of all six.
Inputs:  output/test_cache/{cache_name}.json (JSON metric friends) for the MODELS below, the same
    caches paper_single_variate.py plots. Fetched from Snellius if missing.
Outputs: output/tables/paper_peaks_{PD,DML,DML_ELM_peaks}.tex
Usage:   PYTHONPATH=. python eval_notebooks/paper_peak_tables.py  (self-contained: no wandb, no CSV, no HDF5)
Layout:  rows = peak property x history-window condition, columns = models. Best (lowest) model per
    row is bolded. peak_count is a paired MSE on counts; every other property is a paired
    1-Wasserstein distance, so units differ per property block and only within-row comparison is meaningful.

Channels: 'DML' and 'DML ELM peaks' answer different questions and are both exported.
    'DML' is raw (every peak, ~58/window, comparable to PD): does the model get the signal's peak
    structure right. 'DML ELM peaks' is the ELM-gated subset (~8/window, carries energy_delta): does
    the model put ELMs in the right places, which is the paper's argument.
    **Caches written before 2026-07 do not have this split**: their 'DML' channel is really the gated
    subset despite the label, and has no 'DML ELM peaks' key at all. Against such a cache this script
    warns and emits a DML table that is silently the ELM channel. See docs/evaluation-metrics.md.

Caveat: when a model predicts zero peaks in a window, PeakProps.__sub__ does not return a distance
    against the prediction; it returns the mean of the *target* property, a sentinel that depends only
    on the ground truth. Any two models that predict nothing in the same window therefore score
    identically, and can come out bit-identical. This bites the sparse ELM channels, not the raw ones:
    on 'DML ELM peaks' in the D condition, the U-Net (non-oracle) and iTransformer produce nothing in
    100% of windows, so their whole row is a ground-truth constant; even CFM is ~90% sentinel there,
    against a ground truth that itself has no ELM-coincident DML peaks in 78.6% of those windows. Read
    that row as "no model finds ELMs in D mode", not as a ranking. Raw PD and raw DML are unaffected:
    every model produces tens of peaks per window on both.
    Compounding it, the D condition is only 2.1% of the test set (1267 of 61459 windows).
"""
# %%
import json
import subprocess
from pathlib import Path

import pandas as pd

CACHE_DIR = Path("output/test_cache")
OUT_DIR = Path("output/tables")
SNELLIUS_CACHE = "snellius:/scratch-shared/mtresoor/final_cache"

# Same models and caches as paper_single_variate.py; order here is the column order.
MODELS = [
    ("R-NormalMidAttSig03_anim", "CFM (ours)"),
    ("R-BrownianMidAttSig1_anim", "CFM brownian"),
    ("T2CUnFlow_tripleLeak", "U-Net (elm oracle)"),
    ("PC3UnFlow_Normal256_noPos", "U-Net (non-oracle)"),
    ("T1BiTransformer_tripleLeak", "iTransformer (oracled)"),
]

ANY_NAME = r"$\forall\mathbf{y}_{W_H}$"
CONDITION_LABELS = {
    "L_only_Wh": "L",
    "D_only_Wh": "D",
    "H_only_Wh": "H",
    "mixed": "mixed",
    "any_Wh": ANY_NAME,
}
COND_ORDER = ["L", "D", "H", "mixed", ANY_NAME]

# Property -> (the one pairwise statistic it carries, row label). Counts are integers compared per
# window, hence MSE; the other properties are distributions of peaks within a window, hence W_pair.
PROPERTIES = {
    "peak_count": ("pairwise_mse", r"$\mathbf{N}^\text{Peaks}$ (MSE)"),
    "peak_prominence": ("pairwise_wasserstein", "Prominence"),
    "peak_width": ("pairwise_wasserstein", "Width"),
    "peak_base": ("pairwise_wasserstein", "Base"),
    "peak_energy_delta": ("pairwise_wasserstein", r"$\approx$ELM $\mathbf{E}\Delta$"),
}
# The energy properties exist only on the ELM-gated channel: they are undefined without a coincident
# H-alpha burst. Raw DML carries the same base measures as every other channel, and nothing more.
BASE = ["peak_count", "peak_prominence", "peak_width", "peak_base"]
CHANNEL_PROPERTIES = {
    "PD": BASE,
    "DML": BASE,
    "DML ELM peaks": BASE + ["peak_energy_delta"],
}

# How each channel is named in its caption. Only PD and DML are glossary acronyms
# (\newacronym{PD}{PD}{Photodiode}, \newacronym{DML}{DML}{Diamagnetic Loop}), so "ELM" is written
# plainly here; swap it for \gls{ELM} if that key exists in the paper's glossary.
CHANNEL_CAPTION_NAME = {
    "PD": r"the \gls{PD} signal",
    "DML": r"the raw \gls{DML} signal",
    "DML ELM peaks": r"ELM-coincident \gls{DML} peaks",
}


# %% Fetch any missing JSON friends
def fetch_missing():
    missing = [n for n, _ in MODELS if not (CACHE_DIR / f"{n}.json").exists()]
    for name in missing:
        cmd = f"rsync -vz {SNELLIUS_CACHE}/{name}.json {CACHE_DIR}/"
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    if not missing:
        print("all JSON friends present")


# %% Load
def load_pairwise(cache_name, print_name, channel):
    """Pull the pairwise peak metrics for one model and channel out of its JSON friend.

    Returns (rows_df, missing_keys). Every key listed in CHANNEL_PROPERTIES is expected to be present,
    so an absent one means a stale cache or a renamed channel, not an inapplicable property. Missing
    keys are reported rather than dropped: a silently absent key removes a whole property block from
    the table via the dropna in build_pivot, with no other symptom.
    """
    with open(CACHE_DIR / f"{cache_name}.json", encoding="utf-8") as f:
        data = json.load(f)
    rows, missing = [], []
    for prop in CHANNEL_PROPERTIES[channel]:
        stat, prop_label = PROPERTIES[prop]
        for cond, cond_label in CONDITION_LABELS.items():
            key = f"test/final/{cond}/{prop}/{stat}/{channel}"
            if key not in data:
                missing.append(key)
                continue
            rows.append(
                {
                    "model": print_name,
                    "property": prop_label,
                    "condition": cond_label,
                    "value": data[key],
                }
            )
    return pd.DataFrame(rows), missing


def is_pre_split_cache(cache_name):
    """True if the cache predates the raw/ELM DML split, i.e. has no 'DML ELM peaks' channel at all."""
    with open(CACHE_DIR / f"{cache_name}.json", encoding="utf-8") as f:
        data = json.load(f)
    return not any(k.endswith("/DML ELM peaks") for k in data)


def build_pivot(channel):
    frames, missing = [], {}
    for cn, pn in MODELS:
        df, absent = load_pairwise(cn, pn, channel)
        frames.append(df)
        if absent:
            missing[cn] = absent

    # A pre-split cache still answers every raw-DML key, so nothing above would notice. But its 'DML'
    # channel is the ELM-gated subset despite the label, which is the one way this table can be wrong
    # while looking right.
    if channel == "DML":
        stale = [cn for cn, _ in MODELS if is_pre_split_cache(cn)]
        if stale:
            print(
                f"  WARNING [DML]: {len(stale)} cache(s) predate the raw/ELM DML split: "
                f"{', '.join(stale)}\n"
                f"           Their 'DML' channel is the ELM-GATED subset (~8 peaks/window), not raw "
                f"(~58). This table is labeled raw DML but is not. Regenerate the caches."
            )
    if missing:
        total = sum(len(v) for v in missing.values())
        example = next(iter(missing.values()))[0]
        print(
            f"  WARNING [{channel}]: {total} expected keys absent across {len(missing)} cache(s): "
            f"{', '.join(missing)}\n"
            f"           e.g. {example}\n"
            f"           Stale cache or renamed channel? Pre-2026-07 caches have no 'DML ELM peaks' "
            f"and carry the energy properties on 'DML'. The affected rows are omitted from the table."
        )
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        print(f"  SKIP [{channel}]: no data in any cache, no table written.")
        return None
    pivot = df.pivot_table(index=["property", "condition"], columns="model", values="value")

    # Impose the declared orders: properties as listed in CHANNEL_PROPERTIES, conditions L/D/H/mixed/all,
    # models as listed in MODELS.
    prop_order = [PROPERTIES[p][1] for p in CHANNEL_PROPERTIES[channel]]
    pivot = pivot.reindex(
        index=pd.MultiIndex.from_product([prop_order, COND_ORDER], names=["property", "condition"]),
        columns=[pn for _, pn in MODELS],
    ).dropna(how="all")
    return pivot


# %% Export
def block_scale(block):
    """Pick a power-of-ten multiplier so a property block prints readably at fixed decimals.

    The pairwise Wasserstein distances on DML prominence and energy_delta live around 1e-4, which
    a plain %.3f flattens to 0.000 and destroys the ranking. Scale so the largest value in the
    block lands in [1, 10); leave blocks that are already >= 1 alone.
    """
    largest = block.abs().to_numpy().max()
    if largest <= 0 or largest >= 1:
        return 0
    exponent = 0
    while largest * 10**exponent < 1:
        exponent += 1
    return exponent


def fmt_block(block):
    """Scale a property block to a readable magnitude and bold each row's best (lowest) model."""
    exponent = block_scale(block)
    scaled = block * 10**exponent

    def fmt_row(row):
        # Decimals are chosen per row, for ~4 significant digits on the row's largest entry. A single
        # per-block choice would print the DML count row spanning 1.1..161 at one decimal, collapsing
        # 1.118 and 1.121 to an identical "1.1" while bolding only one of them.
        largest = row.abs().max()
        int_digits = len(f"{int(largest)}") if largest >= 1 else 1
        decimals = min(3, max(0, 4 - int_digits))
        best = row.min()
        return row.apply(
            lambda v: ""
            if pd.isna(v)
            else (rf"\textbf{{{v:.{decimals}f}}}" if v == best else f"{v:.{decimals}f}")
        )

    return scaled.apply(fmt_row, axis=1), exponent


def to_latex(pivot, channel):
    """Format each property block independently, then emit the table."""
    blocks, labels = [], {}
    for prop in pivot.index.get_level_values("property").unique():
        block = pivot.xs(prop, level="property", drop_level=False)
        formatted_block, exponent = fmt_block(block)
        blocks.append(formatted_block)
        # Rename by mapping, not set_levels: set_levels assigns positionally against pandas' own
        # level ordering, which silently pairs each label with the wrong block.
        labels[prop] = prop if exponent == 0 else rf"{prop} ($\times 10^{{-{exponent}}}$)"

    formatted = pd.concat(blocks).rename(index=labels, level="property")
    formatted.index.names = ["Property", "Condition"]

    caption = (
        rf"Paired peak metrics on {CHANNEL_CAPTION_NAME[channel]}. "
        r"All metrics compare a generated window against its own ground-truth window, so lower is better "
        r"throughout and the best model per row is \textbf{highlighted}. Rows are grouped by peak property "
        r"and by the mode composition of the history window (L, D, H: single-mode histories; mixed: multiple "
        r"modes; $\forall\mathbf{y}_{W_H}$: all samples). $\mathbf{N}^\text{Peaks}$ is scored with the mean "
        r"squared error $\frac{\|N - \hat{N}\|_2^2}{n}$ between predicted and true peak counts per window; "
        r"every other property is scored with the pairwise 1-Wasserstein distance between the predicted and "
        r"real peak distributions of the same window, averaged over windows. Units differ per property, so "
        r"values are comparable across a row and not down a column. Models marked oracle receive covariates "
        r"that leak ELM timing information."
    )
    if channel == "DML ELM peaks":
        caption += (
            r" Only \gls{DML} peaks coinciding with an ELM-scale \gls{PD} burst are counted here, so this "
            r"channel is sparse (around 8 peaks per window against 58 for raw \gls{DML}) and a model scores "
            r"only if it generates both the \gls{DML} excursion and the coincident burst. Where a model "
            r"produces no peaks at all, the metric falls back to a target-only constant, so near-identical "
            r"values in the D column reflect that fallback rather than a genuine tie."
        )

    # pandas already separates the multirow property blocks with \cline.
    return formatted.to_latex(
        index=True,
        escape=False,
        multicolumn_format="c",
        caption=caption,
        label=f"tab:paper_peaks_{slug(channel)}",
        position="h",
    )


def slug(channel):
    """Filename/label-safe channel name: 'DML ELM peaks' -> 'DML_ELM_peaks'."""
    return channel.replace(" ", "_")


def export(channel):
    print(f"\n=== {channel} ===")
    pivot = build_pivot(channel)
    if pivot is None:
        return None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"paper_peaks_{slug(channel)}.tex"
    path.write_text(to_latex(pivot, channel))
    print(pivot.round(3).to_string())
    print(f"LaTeX table saved to {path}")
    return pivot


if __name__ == "__main__":
    fetch_missing()
    for ch in CHANNEL_PROPERTIES:
        export(ch)
