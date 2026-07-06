"""Canonical peak-property LaTeX/Excel tables (Wasserstein marginal/pairwise plus MSE) per channel, auto-splitting wide tables.

Scientific output: thesis peak-metric tables (appendix), with detailed W_marg/W_pair/MSE captions.
Inputs:  output/test_cache/*.json via parse_metrics_json(); *.h5 /peaks/{condition} keys are globbed for MODEL_ORDER context but the table export is JSON-based; MODEL_ORDER reversed at definition, sliced [1:] (drops Ground Truth) for the pivot; ground truth handled as the 'Real' rows in the JSON.
Outputs: output/tables/peak_props_{channel}.xlsx and output/tables/peaks_overview_{channel}.tex plus peaks_overview_split_{channel}.tex, for each of FIR_core, PD, PD large peaks, DML, POHM, Z_axis.
Usage:   highly robust and self-contained; run top to bottom; treat as the canonical peak-table generator (run last of the three peak notebooks).
Limits:  shares peak_props_*/peaks_overview_* output filenames with moments.py and peak_analysis.py (last run wins); SettingWithCopy-style in-place edits on channel_df.
Handy:   split_and_write_latex_table() column chunker (max 10 cols); pivot_peaks_df(); export_tables_for_channel() with min-value bold highlighting; parse_metrics_json(). All good src/ candidates.
History: created Jun 17 2025 ("Finish awesome boxplots and all tables"), single commit, never edited (frozen finalized artifact).
"""
#%%# Peak Metrics Analysis from Archived HDF5 Files

# This notebook loads peak property DataFrames directly from HDF5 archives created by PeakMetric, combines them, and explores their dimensions for pivot table analysis.

import os
import glob
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
from pathlib import Path
# Find all HDF5 files in the output or cache directories
h5_pattern = "*.h5"
CACHE_DIR = Path('output/test_cache/')
h5_files = list(CACHE_DIR.glob(h5_pattern))
print(f"Found {len(h5_files)} HDF5 files:", h5_files)
EXAMPLE_MODEL = h5_files[0]

MODEL_ORDER = list(
    reversed(
        [
            'Unet-Channel-Brownian',
            'Unet-Sequence-Brownian',
            'Unet-Sequence-AllCov-Brownian',
            'FM-Sequence-AllCov-Brownian',
            # 'FM-Channel-AllCov-Brownian',
            'FM-Sequence-Constant',
            'FM-Channel-CP',
            'FM-Sequence-CP',
            'FM-Channel-Resampled',
            'FM-Sequence-Resampled',
            'FM-Channel-Brownian',
            'FM-Sequence-Brownian',
            'FM-Sequence-Tiny-Gaussian',
            'FM-Sequence-2x-Gaussian',
            'FM-Sequence-Gaussian',
            'FM-Channel-Gaussian',
            'Ground Truth'  # sourced from 'FM-Sequence-Gaussian' -> distribution == Real
        ]
    )
)
MEASURE_LABEL = dict(
    count=r'$\text{Window } \mathbf{N}^\text{Peaks}$',
    prominence='Peak Prominence',
    width='Peak Width',
    base='Peak Base',
    energy_delta=r'$\approx\text{ELM }\mathbf{E}\Delta$'
)
#%%

#%%## Getting Precalculated distances from config
"""
The combined Table should, for each channel:
- row: 'Model' = Name of model / Ground Truth
- `condition`: Logical condition in {L (b) D (orange) H (r)  mixed (purple) <ANY_NAME> (darkgrey) }
# - `channel_name`: Signal/channel (e.g., DML, PD, etc.)
- `measure`: Peak property in {} count, prominence, width. For DML: also energy_ratio }
- `distribution`: 'Generated' or 'Real'. Real will be taken as the first model row.
- `value`: The measured property value of one peak, or per window for 'count'


Magnitude Window Mean:
$ \mu_{\text{window}}^{\text{magnitude}} $

Diff Window Mean:
$ \mu_{\text{window}}^{\text{diff}} $

Magnitude Window Variance:
$ \sigma_{\text{window}}^{2, \text{magnitude}} $

Diff Window Variance:
$ \sigma_{\text{window}}^{2, \text{diff}} $

Magnitude Window Skewness:
$ \text{Skew}_{\text{window}}^{\text{magnitude}} $

Diff Window Skewness:
$ \text{Skew}_{\text{window}}^{\text{diff}} $

Magnitude Window Kurtosis:
$ \text{Kurt}_{\text{window}}^{\text{magnitude}} $

Diff Window Kurtosis:
$ \text{Kurt}_{\text{window}}^{\text{diff}} $


"""
from pathlib import Path

h5_pattern = "*.json"
json_files = list(Path('output/test_cache/').glob(h5_pattern))
print(f"Found {len(json_files)} json files:", json_files)
EXAMPLE_MODEL = json_files[0]

####

import pandas as pd
import json


def parse_metrics_json(json_path):
    """
    Parse a JSON containing hierarchical keys into a long-format DataFrame.

    Args:
        json_data (dict): JSON data with hierarchical keys.

    Returns:
        pd.DataFrame: Long-format DataFrame with columns:
                      ['condition', 'peak_property', 'statistic', 'channel', 'value']
    """
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    print("Opened JSON file:", json_path)
    # Initialize a list to store parsed rows
    rows = []

    # Iterate over the JSON keys and values
    for key, value in json_data.items():
        # Split the hierarchical key into components
        key = key.replace('test/final/', '')
        parts = key.split('/')
        if parts[0] == 'mode':
            continue
        elif len(parts) == 4:
            condition, property, statistic, channel = parts
            if "NBI" in property:
                property = "ELM "+  channel.split('_')[-1]
                channel = "PDxNBI"
        elif len(parts) == 3:
            property, statistic, channel = parts
            if '_' in statistic:
                property, moment, statistic = statistic.split('_')
                property = property + ' ' + 'window ' + moment
            else:
                property = 'magnitude'
            condition = 'any_Wh'
        elif parts[-1] == 'dice':
            condition, property, statistic, channel = 'any_Wh', 'mode labels', 'Dice', 'mean'
        elif parts[-1] == 'softDTW':
            condition, property, statistic, channel = 'any_Wh', 'magnitude', 'DTW', 'mean'
        elif len(parts) == 2:
            condition, hits = parts
            if 'total_hits' == hits:
                print(f"Total hits on {condition.split('_')[0]} was {value}")
            else:
                print(parts)
        else:
            print(parts)
        # Append the parsed row
        rows.append(
            {
                'model': json_path.stem,
                'condition': condition,
                'property': property,
                'statistic': statistic,
                'channel': channel,
                'value': value
            }
        )

    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(rows)

    return df


parse_metrics_json(json_files[0]).query("'2d_wasserstein'== statistic")

#%% Concatenate all parsed DataFrames from the JSON files
all_json_dfs = [parse_metrics_json(jf) for jf in json_files]
metrics_df = pd.concat(all_json_dfs, ignore_index=True)
print(f"Combined metrics DataFrame shape: {metrics_df.shape}")
metrics_df
ANY_NAME = r"$\forall\mathbf{y}_{W_H}$"

CONDS = [
    'L_only_Wh',
    'D_only_Wh',
    'H_only_Wh',
    'mixed',
    'any_Wh',
]
condition_labels = {"L_only_Wh": "L", "D_only_Wh": "D", "H_only_Wh": "H", "mixed": "mixed", "any_Wh": ANY_NAME}
condition_palette = {'L': '#1f77b4', 'D': '#ff7f0e', 'H': '#d62728', 'mixed': 'purple', ANY_NAME: '#444444'}

COND_ORDER = ["L", "D", "H", "mixed", ANY_NAME]
# Map 'condition' column to labels for plotting and analysis
metrics_df['condition'] = metrics_df['condition'].map(condition_labels)
# Print unique values for every column in metrics_df

# Replace underscores with spaces and capitalize in 'property' column
metrics_df['statistic'] = metrics_df['statistic'].str.replace('_', ' ').str.title().str.replace('Mse', 'MSE')
for col in metrics_df.columns:
    print(f"Unique values in '{col}': {metrics_df[col].unique()}\n")
#%% Split metrics
window_metrics_df = metrics_df.loc[~metrics_df['property'].str.contains("peak")]
peaks_metrics_df = metrics_df.loc[metrics_df['property'].str.contains("peak") & (metrics_df['statistic'] != 'Pairwise Rmse')]
for col in peaks_metrics_df.columns:
    print(f"Export: Unique values in '{col}': {peaks_metrics_df[col].unique()}\n")

#%%
def export_tables_for_channel(peaks_metrics_df, channel):
    channel_df = peaks_metrics_df.query(f'channel == "{channel}"')
    channel_df['channel'] = channel_df['channel'].str.replace('_', ' ').str.title()
    for col in channel_df.columns:
        if col == 'value': continue
        print(f"Export: Unique values in '{col}': {channel_df[col].unique()}\n")

    pivot = pivot_peaks_df(channel_df)

    pivot.to_excel(f"output/tables/peak_props_{channel}.xlsx", index=True, float_format="%.3f", merge_cells=True)
    pivot.columns = pivot.columns.set_levels(
    [r'$\mathcal{W}_{\text{marg}}$', r'MSE', r'$\mathcal{W}_{\text{pair}}$'],
    level='statistic'
    )
    # Automatically split wide tables into two LaTeX tables if too many columns
    MAX_COLS = 10  # Adjust as needed for your document


    caption = f"Model comparison by peaks on signal {channel.replace('_', ' ').upper()}. " + r"""Lower values indicate better model performance for all metrics and the best score per condition and metric is \textbf{highlighted}. Note that the models using all covariates (AllCov) benefit from more ELM timing information.
The sub tables are organized with a three-level column hierarchy: (1) Peak Property (2) History window conditions based on mode composition (L: L-mode only, D: D-mode only, H: H-mode only, mixed: multiple modes, $\forall\mathbf{y}_{W_H}$: all samples), and (3) Evaluation metrics. $\mathcal{W}_{\text{marg}}$ measures marginal 1-Wasserstein distance between property distributions, $\mathcal{W}_{\text{pair}}$ measures pairwise 1-Wasserstein distance between predicted and real peaks of the same window, averaged over all windows, and MSE $\frac{\|N - \hat{N}\|_2^2}{n}$ represents the mean squared error between predicted and true peak counts per window across paired samples. """

    def highlight_min(s, col):
        return f"\\textbf{{{s:.3f}}}" if s == pivot[col].min() else f"{s:.3f}"


    # Apply formatting to each column
    formatted_pivot = pivot.copy()
    for col in pivot.columns:
        formatted_pivot[col] = pivot[col].apply(lambda s: highlight_min(s, col))

    latex_str = formatted_pivot.replace(0.0, '').fillna('').to_latex(
        index=True,
        escape=False,
        float_format="%.3f",
        longtable=True,
        position='h',
        caption=caption,
        multicolumn_format='c',
    )

    # Save the LaTeX string to a file
    output_path = Path(f"output/tables/peaks_overview_{channel}.tex")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex_str)
    print(f"LaTeX table saved to {output_path}")

    def split_and_write_latex_table(pivot, path, max_cols=MAX_COLS):
        n_cols = pivot.shape[1]
        with open(path, "w") as f:
            f.write("")
        for i, start in enumerate(range(0, n_cols, max_cols)):
            end = min(start + max_cols, n_cols)
            chunk = pivot.iloc[:, start:end]
            latex_str = chunk.replace(0.0, '').fillna('').round(2).to_latex(
                index=True,
                escape=False,
                float_format="%.3f",
                longtable=False,
                # formatters=formatters
                index_names=False,
                position='h',
                caption=caption
                if start == 0 else f"Part {i+1} of {channel.replace('_', ' ').upper()} peaks overview.",
                multicolumn_format='c',
                label=f"tab:peaks_overview_{channel}:part{i+1}" if n_cols > max_cols else f"tab:peaks_overview_{channel}",
            )
            with open(path, "a") as f:
                f.write(latex_str)

    split_and_write_latex_table(formatted_pivot, f"output/tables/peaks_overview_split_{channel}.tex")

def pivot_peaks_df(channel_df):
    channel_df.loc[:, 'property'] = channel_df['property'].str.replace('peak_', '', regex=False, )
    property_order = ['count', 'prominence', 'width', 'base', 'energy_delta']
    channel_df = channel_df.loc[channel_df['property'].isin(property_order)]
    pivot = channel_df.pivot_table(
    columns=["property", "condition", "statistic"], values="value", index=['model']
    )
    # Sort columns by CONDS order for 'condition' level
    if "condition" in pivot.columns.names:
        # Get current MultiIndex columns as DataFrame for sorting
        cols_df = pivot.columns.to_frame(index=False)
        # Create a categorical type for 'condition' with CONDS order
        cols_df['condition'] = pd.Categorical(cols_df['condition'], categories=COND_ORDER, ordered=True)
        # Sort by all column levels, but 'condition' will use CONDS order
        # Rebuild MultiIndex with renamed condition
        pivot.columns = pd.MultiIndex.from_frame(cols_df)
        sorted_cols = cols_df.sort_values(list(cols_df.columns)).set_index(list(cols_df.columns)).index
        pivot = pivot.reindex(columns=sorted_cols)
        # Sort index (models) by MODEL_ORDER
        model_order_index = pd.CategoricalIndex(MODEL_ORDER[1:], name='model')
        pivot = pivot.reindex(model_order_index)
        # Map 'property' level to remove 'peak_' prefix and sort as: count, prominence, width, base
        if "property" in pivot.columns.names:
            cols_df = pivot.columns.to_frame(index=False)
            print(cols_df['property'].unique())
            # Set property as categorical for sorting
            cols_df['property'] = pd.Categorical(cols_df['property'], categories=property_order, ordered=True)

            # Rebuild MultiIndex and sort
            pivot.columns = pd.MultiIndex.from_frame(cols_df)
            sorted_cols = cols_df.sort_values(list(cols_df.columns)).set_index(list(cols_df.columns)).index
            pivot = pivot.reindex(columns=sorted_cols)
            # Replace underscores with spaces and capitalize in 'property' column labels of the pivot table
            new_columns = []
            for col in pivot.columns:
                col = list(col)
                # Replace underscores with spaces and capitalize for 'property' level (assumed to be index 0)
                col[0] = col[0].replace('_', ' ').capitalize()
                new_columns.append(tuple(col))
            pivot.columns = pd.MultiIndex.from_tuples(new_columns, names=pivot.columns.names)
    if isinstance(pivot.columns, pd.MultiIndex):
        for level, name in enumerate(pivot.columns.names):
            unique_vals = pivot.columns.get_level_values(level).unique().values
            print(f"Unique values in column level '{name}': {unique_vals}\n")
    else:
        print(f"Unique values in columns: {pivot.columns.unique()}\n")
    return pivot

export_tables_for_channel(peaks_metrics_df,'FIR_core')
export_tables_for_channel(peaks_metrics_df,'PD')
export_tables_for_channel(peaks_metrics_df,'PD large peaks')
export_tables_for_channel(peaks_metrics_df,'DML')
export_tables_for_channel(peaks_metrics_df,'POHM')
export_tables_for_channel(peaks_metrics_df,'Z_axis')

# latex_str = pivot.replace(0.0, '').fillna('').round(2).to_latex(
#     index=True,
#     escape=False,
#     float_format="%.3f",
#     longtable=True,
#     position='h',
#     multicolumn_format='c',
# )
# pivot

# # %%
# pivoted_T = pivoted.reindex(
#     pd.MultiIndex.from_product([MODEL_ORDER], names=['Model'])
# ).reindex(pd.MultiIndex.from_product([sort_order_trans, sort_order_cols_inner], names=['From', '']),
#           axis='columns').dropna(how='all')

# #%%# Example: Pivot table for a specific channel (e.g., 'DML')
# #%%
# #%%# Example: Loop over all channels and display summary tables
# channels = combined_df["channel_name"].unique()
# for channel in channels:
#     print(f"=== Channel: {channel} ===")
#     display(
#         combined_df[combined_df["channel_name"] == channel].pivot_table(
#             index=["condition", "measure", "distribution"], values="value", aggfunc=["mean", "std", "count"]
#%%# %%
#%%# %%
