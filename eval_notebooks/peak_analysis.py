"""Peak/ELM statistics boxplots (count/prominence/width/base/energy_ratio) across models and mode conditions, with a ghost ground-truth overlay; second half exports peak tables.

Scientific output: thesis ELM/peak boxplot figures and peak-metric tables.
Inputs:  all output/test_cache/*.h5 via /peaks/{condition} keys; plus output/test_cache/*.json (parse_metrics_json) for the table half; MODEL_ORDER is reversed here; ground truth is the 'Real' distribution, sourced from FM-Sequence-Gaussian.h5.
Outputs: output/pdfplots/peak_boxplot_with_base_DMLratio/peak_boxplots_{WxH}/boxplots_{channel}.pdf (multiple sizes); output/tables/base_metrics_detail.{tex,xlsx}; output/tables/peak_props_{channel}.xlsx and peaks_overview_split_{channel}.tex (the final peaks_overview_split write is commented out at bottom).
Usage:   run the boxplot half, then a hardcoded exit() under __main__ (~line 358), then the JSON table half (comment out exit() to reach it); box_plot_peaks(models, 'DML') is the active call, other channels commented.
Limits:  hardcoded exit() blocks the table half by default; MODEL_ORDER reversed vs the other notebooks; peak_props_*.xlsx filename collides with peaks_tables.py (last run wins); uses energy_ratio label here vs energy_delta in peaks_tables.py.
Handy:   iter_peak_properties_per_model() generator; box_plot_peaks() grid with ghost-GT overlay, condition-highlight rectangles, and multi-size PDF export; parse_metrics_json() hierarchical-key parser. All good src/ candidates.
History: created Jun 16 2025 ("Basic boxplot" then "Good boxplot"), major Jun 17 ("Finish awesome boxplots and all tables", +592 lines), heavy Oct 11 2025 rework (+175).
"""
#%%# Peak Metrics Analysis from Archived HDF5 Files

# This notebook loads peak property DataFrames directly from HDF5 archives created by PeakMetric, combines them, and explores their dimensions for pivot table analysis.

import os
import glob
import numpy as np
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
    energy_ratio=r'$\approx\text{ELM }\mathbf{E}\Delta$ ratio'
)


#%%
#%%# Helper: List all /peaks/* keys in an HDF5 file
def list_peak_keys(h5_path):
    with pd.HDFStore(h5_path, "r") as store:
        return [k for k in store.keys() if k.startswith("/peaks/")]


list_peak_keys(EXAMPLE_MODEL)
#%%# Collect all DataFrames from all HDF5 files and all /peaks/* keys
CONDITION_KEYS = ['/peaks/D_only_Wh', '/peaks/H_only_Wh', '/peaks/L_only_Wh', '/peaks/any_Wh', '/peaks/mixed']


def iter_peak_properties_per_model(models: list[str], channel_name, sample=False, dummy_df=None):
    for model in (pbar := tqdm(models, desc="Loading models peaks")):
        pbar.set_postfix(model=model)
        all_peak_dfs = []
        if model == "Ground Truth":
            h5_path = CACHE_DIR / ('FM-Sequence-Gaussian' + '.h5')
            distribution = "Real"
        else:
            h5_path = CACHE_DIR / (model + '.h5')
            distribution = "Generated"
        for cond_idx, condition_key in enumerate(CONDITION_KEYS):
            pbar.set_postfix(model=model, condition_key=condition_key)
            try:
                # tqdm.write(f"Processing file: {h5_path.name}, condition: {condition_key}")
                if dummy_df is not None:
                    df = dummy_df.sample(frac=0.5)
                else:
                    df = pd.read_hdf(h5_path, key=condition_key)
                if sample:
                    df = df.sample(frac=0.5 if sample is True else sample)
                df["human_name"] = model
                df["condition"] = condition_key.split('/')[2]
                all_peak_dfs.append(df.query(f"distribution=='{distribution}' & channel_name=='{channel_name}'"))
            except KeyError as e:
                tqdm.write(f"Missing key: {condition_key} in {h5_path.name} ({e})")
                continue
            except Exception as e:
                tqdm.write(f"Error processing {h5_path.name} ({condition_key}): {e}")
                continue

        if all_peak_dfs:
            combined_df = pd.concat(all_peak_dfs, ignore_index=True)
            tqdm.write(f"Combined DataFrame shape: {combined_df.shape}")
            del all_peak_dfs
            yield model, combined_df
        else:
            yield model, None


m, combined_df = next(iter_peak_properties_per_model(MODEL_ORDER[1:2], 'DML', 0.01))

#%%## DataFrame Structure and Available Dimensions
"""
The combined DataFrame includes:
- `human_name`: Name of model / Ground Truth
- `condition`: Logical condition in {L_only_Wh (b) D_only_Wh (orange) H_only_Wh (r)  mixed (purple) any_Wh (darkgrey) }
- `channel_name`: Signal/channel (e.g., DML, PD, etc.)
- `measure`: Peak property (e.g., height, prominence, count, etc.)
  - relevant: count, prominence, width. For DML: also energy_ratio
- `distribution`: 'Generated' or 'Real'. Real will be taken as a model row.
- `value`: The measured property value of one peak, or per window for 'count'
"""

# for one channel, one measure per column, and the human_name model name, create a box plot, split out in colors per condition.
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch, Rectangle
import os
from matplotlib.legend import Legend

PDF_DIR = Path("output/pdfplots/peak_boxplot_with_base_DMLratio")
PDF_DIR.mkdir(exist_ok=True)

#%%
# def plot_facet_boxplots(df):
# Define color palette for conditions

models = MODEL_ORDER


def box_plot_peaks(models, channel='DML'):
    model_iterator_for_channel = iter_peak_properties_per_model(models, channel, dummy_df=None)
    ANY_NAME = r"$\forall\mathbf{y}_{W_H}$"
    condition_order = ["L", "D", "H", "mixed", ANY_NAME]
    condition_labels = {"L_only_Wh": "L", "D_only_Wh": "D", "H_only_Wh": "H", "mixed": "mixed", "any_Wh": ANY_NAME}
    condition_palette = {'L': '#1f77b4', 'D': '#ff7f0e', 'H': '#d62728', 'mixed': 'purple', ANY_NAME: '#444444'}
    # df = df[df['condition'].isin(condition_order)].copy()
    # cond_label_order = [condition_labels[c] for c in condition_order]

    measures = ['count', 'prominence', 'width', 'base']
    if channel == "DML":  #'energy_delta' in df['measure'].unique():
        measures.append('energy_ratio')
    n_rows = len(models)
    n_cols = len(measures)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 2 * n_rows),
        sharex='col',
        gridspec_kw={
            'hspace': 0.4,
            'wspace': 0.05
        },  #layout="constrained"  # Add this argument
    )
    # abnded rows:
    """
    fig.subplotpars.left  # Left edge of the subplots
    fig.subplotpars.right  # Right edge of the subplots
    fig.subplotpars.top  # Top edge of the subplots
    fig.subplotpars.bottom  # Bottom edge of the subplots
    fig.subplotpars.hspace  # Vertical spacing between subplots
    fig.subplotpars.wspace  # Horizontal spacing between subplots
    """
    x = fig.subplotpars.left
    width = fig.subplotpars.right - fig.subplotpars.left
    height = (fig.subplotpars.top - fig.subplotpars.bottom) / n_rows * 0.99
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for i, (model, df) in enumerate(model_iterator_for_channel):
        if df is None:
            print(f"Skipping model {model} due to missing data.")
            continue

        tqdm.write(f"Plotting model: {model} with shape {df.shape}")
        df['cond_label'] = df['condition'].map(condition_labels)

        is_ground_truth = model == "Ground Truth"
        if i == 0 and is_ground_truth:
            ground_truth_data = df.copy()  # Save Ground Truth data for later rows
        else:
            # Add ghost versions of the first model row (Ground Truth) to all other rows
            for j, measure in enumerate(measures):
                ax = axes[i][j]
                ghost_data = ground_truth_data[ground_truth_data['measure'] == measure]
                if not ghost_data.empty:
                    sns.boxplot(
                        data=ghost_data,
                        x='value',
                        y='cond_label',
                        hue='cond_label',
                        order=condition_order,
                        palette=condition_palette,
                        width=0.7,  # Reduced width for ghost boxes
                        ax=ax,
                        showfliers=False,
                        orient='h',
                        boxprops=dict(edgecolor='none', alpha=0.2),  # No borders, semi-transparent
                        whiskerprops=dict(color='gray', linewidth=2, alpha=0.05)  # Whiskers match fill color
                    )
        if "seq" in model.lower():
            subplot_rop_adj = fig.subplotpars.top * 1.015
            print(subplot_rop_adj, fig.subplotpars.bottom, fig.subplotpars.hspace)
            rowy = subplot_rop_adj - (i + 1) * (subplot_rop_adj - fig.subplotpars.bottom) / n_rows
            print(rowy)
            rect = Rectangle(
                (x, rowy),
                width,
                height,
                color='lightblue',
                alpha=0.3,
                zorder=-1,
                transform=fig.transFigure,
                label='Sequential\nConditioning'
            )
            fig.patches.append(rect)

    # Plot the actual data for the current model
        for j, measure in enumerate(measures):
            ax = axes[i][j]
            ax.set_facecolor('none')
            data = df[(df['measure'] == measure)]
            if not data.empty:
                sns.boxplot(
                    data=data,
                    x='value',
                    y='cond_label',
                    hue='cond_label',
                    width=0.9 if not is_ground_truth else 0.7,
                    order=condition_order,
                    palette=condition_palette,
                    boxprops=dict(edgecolor='black', linewidth=0.5)
                    if not is_ground_truth else dict(edgecolor='none', alpha=0.7),
                    ax=ax,
                    showfliers=False,
                    orient='h'
                )
            if j == 0:
                # Always show the model label as a text annotation on the left of each row, regardless of x-axis ticks
                # ax.annotate(
                #     model,
                #     xy=(0, 0.5),
                #     xycoords=('axes fraction', 'axes fraction'),
                #     xytext=(-ax.yaxis.labelpad - 10, 0),
                #     textcoords='offset points',
                #     ha='right',
                #     va='center',
                #     rotation=0,
                #     fontsize=8,
                #     fontweight='bold'
                # )
                # Annotate model name, italic if "unet" in model name (case-insensitive)
                fontstyle = 'italic' if 'unet' in model.lower() else 'normal'
                ax.annotate(
                    model,
                    xy=(0, 1.05),
                    xycoords=('axes fraction', 'axes fraction'),
                    xytext=(-ax.yaxis.labelpad - 10, 0),
                    textcoords='offset points',
                    ha='left',
                    va='bottom',
                    rotation=0,
                    fontsize=7,
                    fontweight='bold',
                    fontstyle=fontstyle
                )
            if j == n_cols - 1 and i == n_rows - 1:
                # Remove any existing legend
                # Add a new legend positioned at the top right outside the axes
                handles = [
                    Patch(facecolor=condition_palette[label], edgecolor='black', label=label)
                    for label in condition_order
                ]
                if 'rect' in locals():
                    handles.append(rect)
                ax.legend(
                    handles=handles,
                    loc='lower left',
                    bbox_to_anchor=(1.01, 1.0),
                    borderaxespad=0.0,
                    fontsize='small',
                    title='Modes in $W_H$',
                    title_fontsize='small',
                    frameon=False
                )
            if i == 0:
                ax.set_title(MEASURE_LABEL[measure], pad=12)
            else:
                ax.set_title('')
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_yticklabels([])
            sns.despine(ax=ax)
    # Add custom legend for conditions to the right of the plot
    # fig.legend(handles=handles, title='Modes in $W_H$', loc='lower right', bbox_to_anchor=(1.0, .93),    borderaxespad=0.0,
    # fontsize='small',
    # title_fontsize='small',
    # frameon=False
    #  )
    # plt.show()
    w, h = fig.get_size_inches()
    pdf_path = PDF_DIR / f"peak_boxplots_{w}x{h}" / f"boxplots_{channel}.pdf"
    pdf_path.parent.mkdir(parents=False, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches='tight')
    # Save a more vertical version (portrait orientation)
    w, h = 5.77 * 1.2, 9.69 * 1.2
    pdf_path = PDF_DIR / f"peak_boxplots_{w:.0f}x{h:.0f}_vertical" / f"boxplots_{channel}.pdf"
    pdf_path.parent.mkdir(parents=False, exist_ok=True)
    fig.set_size_inches(w, h)  # Swap width and height for vertical
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    w, h = 5.77 * 2, 9.69 * 2
    pdf_path = PDF_DIR / f"peak_boxplots_{w}x{h}_vertical2" / f"boxplots_{channel}.pdf"
    pdf_path.parent.mkdir(parents=False, exist_ok=True)
    fig.set_size_inches(w, h)  # Swap width and height for vertical
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    w, h = 5.77 * 2.5, 9.69 * 2.5
    fig.set_size_inches(w, h)  # Swap width and height for vertical
    pdf_path = PDF_DIR / f"peak_boxplots_{w:.0f}x{h:.0f}_vertical25" / f"boxplots_{channel}.pdf"
    pdf_path.parent.mkdir(parents=False, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)

    w, h = 8, 6
    fig.set_size_inches(w, h)  # Swap width and height for vertical
    pdf_path = PDF_DIR / f"peak_boxplots_{w:.0f}x{h:.0f}_horiz" / f"boxplots_{channel}.pdf"
    pdf_path.parent.mkdir(parents=False, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    w, h = 15, 14
    fig.set_size_inches(w, h)  # Swap width and height for vertical
    pdf_path = PDF_DIR / f"peak_boxplots_{w:.0f}x{h:.0f}_horiz" / f"boxplots_{channel}.pdf"
    pdf_path.parent.mkdir(parents=False, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    w, h = 16, 12
    fig.set_size_inches(w, h)  # Swap width and height for vertical
    pdf_path = PDF_DIR / f"peak_boxplots_{w:.0f}x{h:.0f}_horiz" / f"boxplots_{channel}.pdf"
    pdf_path.parent.mkdir(parents=False, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    # Restore original size for further use
    # fig.set_size_inches(w, h)
    # fig.savefig(pdf_path, bbox_inches='tight')
    return fig


# box_plot_peaks(models, 'FIR_core')
# box_plot_peaks(models, 'PD')
# box_plot_peaks(models, 'PD large peaks')
box_plot_peaks(models, 'DML')
# box_plot_peaks(models, 'POHM')
# box_plot_peaks(models, 'Z_axis')
# plt.gcf()
# fig.set_size_inches(w, h)  # Swap width and height for vertical
# fig.savefig(vertical_pdf_path, bbox_inches='tight', dpi=300)
#%%
if __name__ == "__main__":
    exit()

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
                property = "ELM " + channel.split('_')[-1]
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


parse_metrics_json(json_files[0])  #.query("'2d_wasserstein'== statistic")

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

condition_order = ["L", "D", "H", "mixed", ANY_NAME]
# Map 'condition' column to labels for plotting and analysis
metrics_df['condition'] = metrics_df['condition'].map(condition_labels)
# Print unique values for every column in metrics_df

# Replace underscores with spaces and capitalize in 'property' column
metrics_df['statistic'] = metrics_df['statistic'].str.replace('_', ' ').str.title().str.replace('Mse', 'MSE')
metrics_df['channel'] = metrics_df['channel']  #.str.replace('_', ' ')#.str.title()
for col in metrics_df.columns:
    print(f"Unique values in '{col}': {metrics_df[col].unique()}\n")
#%%
channel = "FIR core"
peaks_metrics_df = metrics_df.loc[metrics_df['property'].str.contains("peak") &
                                  (metrics_df['statistic'] != 'Pairwise Rmse')]
for col in peaks_metrics_df.columns:
    print(f"Unique values in '{col}': {peaks_metrics_df[col].unique()}\n")
window_metrics_df = metrics_df.loc[~metrics_df['property'].str.contains("peak")]
#%%
peaks_metrics_df.loc[:, 'property'] = peaks_metrics_df['property'].str.replace(
    'peak_',
    '',
    regex=False,
)
property_order = ['count', 'prominence', 'width', 'base', 'energy_ratio']
peaks_metrics_df = peaks_metrics_df.loc[peaks_metrics_df['property'].isin(property_order)]
pivot = peaks_metrics_df.query(
    f'channel == "{channel.title()}"'
).pivot_table(  # TODO FIX 
    columns=["property", "condition", "statistic"], values="value", index=['model']
)
# Sort columns by CONDS order for 'condition' level
if "condition" in pivot.columns.names:
    # Get current MultiIndex columns as DataFrame for sorting
    cols_df = pivot.columns.to_frame(index=False)
    # Create a categorical type for 'condition' with CONDS order
    cols_df['condition'] = pd.Categorical(cols_df['condition'], categories=condition_order, ordered=True)
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
pivot
#%%
import numpy as np
channel = ''
base_metrics_df = window_metrics_df.query("property not in ['ELM count', 'ELM prominence', 'ELM width']")
base_metrics_df_log = base_metrics_df.copy()
base_metrics_df_log['value'] = np.log10(base_metrics_df['value'])
pivot_window = base_metrics_df.pivot_table(
    columns=[
        # "condition",
        "property",
        "statistic",
    ],
    values="value",
    index=['channel', 'model']
)
# Specify the desired channel order
desired_channels = [
    "mean",
    # "FIR_core",

    "PD",
    "DML",
    "POHM",
    # "Z_axis",
]
all_properties = pivot_window.columns.get_level_values(0)
disired_properties = [
    'magnitude',
    'magnitude',
    'mode labels',
    'magnitude window mean',
    'magnitude window var',
    'magnitude window skew',
    'magnitude window kurtosis',
    'diff window mean',
    'diff window var',
    'diff window skew',
    'diff window kurtosis',
]
desired_order_mapping = {channel: i for i, channel in enumerate(desired_channels)}
desired_order_mapping_props = {channel: i for i, channel in enumerate(disired_properties)}
# Filter the pivot_window to include only the desired channels
filtered_pivot_window = pivot_window.loc[pivot_window.index.get_level_values(0).isin(desired_channels)]
# Create a mapping for the desired order
# filtered_pivot_window = filtered_pivot_window.loc[
#     filtered_pivot_window.columns.get_level_values(0).isin(disired_properties)]

# # Create a mapping for the desired order

# Sort the filtered pivot_window based on the desired order for the first level
sorted_pivot_window = filtered_pivot_window.sort_index(key=lambda idx: idx.map(desired_order_mapping),
                                                       level=0).sort_index(
                                                           key=lambda idx: idx.map(desired_order_mapping_props),
                                                           level=0,
                                                           axis=1
                                                       )

sorted_pivot_window


# def highlight_min(s, col):
#     return f"\\textbf{{{s:.3f}}}" if s == pivot[col].min() else f"{s:.3f}"


# # Apply formatting to each column
# formatted_pivot = sorted_pivot_window.copy()
# for col in sorted_pivot_window.columns:
#     formatted_pivot[col] = sorted_pivot_window[col].apply(lambda s: highlight_min(s, col))

latex_str = sorted_pivot_window.replace(0.0, '').fillna('').to_latex(
    index=True,
    escape=False,
    float_format="%.3e",  # Use scientific notation with 2 decimal places
    longtable=True,
    position='h',
    caption="Pairwise errors and magnitude mse",
    multicolumn_format='c',
)

# Save the LaTeX string to a file
output_path = Path(f"output/tables/base_metrics_detail.tex")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    f.write(latex_str)
sorted_pivot_window.to_excel(f"output/tables/base_metrics_detail.xlsx", index=True, 
# float_format="%.3f",
    float_format="%.4e",  # Use scientific notation with 2 decimal places
                             merge_cells=True)
#%%
# FORMATTING
# condition_labels = {"L_only_Wh": "L", "D_only_Wh": "D", "H_only_Wh": "H", "mixed": "mixed", "any_Wh": ANY_NAME}
# condition_palette = {'L': '#1f77b4', 'D': '#ff7f0e', 'H': '#d62728', 'mixed': 'purple', ANY_NAME: '#444444'}
# # df = df[df['condition'].isin(condition_order)].copy()
# # cond_label_order = [condition_labels[c] for c in condition_order]

# measures = ['count', 'prominence', 'width']
# if  channel == "DML":#'energy_ratio' in df['measure'].unique():
#     measures.append('energy_ratio')
# n_rows = len(models)
# n_cols = len(measures)
# Print unique values for each column level in the pivot table
if isinstance(pivot.columns, pd.MultiIndex):
    for level, name in enumerate(pivot.columns.names):
        unique_vals = pivot.columns.get_level_values(level).unique().values
        print(f"Unique values in column level '{name}': {unique_vals}\n")
else:
    print(f"Unique values in columns: {pivot.columns.unique()}\n")

pivot.to_excel(f"output/tables/peak_props_{channel}.xlsx", index=True, float_format="%.3f", merge_cells=True)
# pivot.to_excel()
pivot.columns = pivot.columns.set_levels(
    [r'$\mathcal{W}_{\text{marg}}$', r'pair MSE', r'$\mathcal{W}_{\text{pair}}$'], level='statistic'
)
# Automatically split wide tables into two LaTeX tables if too many columns
MAX_COLS = 10  # Adjust as needed for your document

caption = f"Model comparison by peaks on signal {channel}. " + r"""
The table is organized with a three-level column hierarchy: (1) Peak properties measured within each window (Count, Prominence, Width, Base, Energy Delta Ratio), (2) History window conditions based on mode composition (L: L-mode only, D: D-mode only, H: H-mode only, mixed: multiple modes, $\forall\mathbf{y}_{W_H}$: all samples), and (3) Evaluation metrics. $\mathcal{W}_{\text{marg}}$ measures marginal 1-Wasserstein distance between property distributions, $\mathcal{W}_{\text{pair}}$ measures pairwise 1-Wasserstein distance between predicted and real peaks of the same window, averaged over all windows, and MSE $\frac{\|N - \hat{N}\|_2^2}{n}$ 
$\frac{\|N - \hat{N}\|_2^2}{n}$ represents the mean squared error between predicted and true peak counts per window across paired samples. Lower values indicate better model performance for all metrics."""


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
    # formatters=formatters,
    position='h',
    caption=caption,
    multicolumn_format='c',
)

# Save the LaTeX string to a file
output_path = Path(f"output/tables/peaks_overview_{channel}.tex")
output_path.parent.mkdir(parents=True, exist_ok=True)
# with open(output_path, "w") as f:
#     f.write(latex_str)
print(f"LaTeX table saved to {output_path}")


def split_and_write_latex_table(pivot, path, max_cols=MAX_COLS):
    n_cols = pivot.shape[1]
    if n_cols <= max_cols:
        with open(path, "w") as f:
            f.write(pivot.to_latex(escape=False, index=True))
    else:
        with open(path, "w") as f:
            f.write("")
        for start in range(0, n_cols, max_cols):
            end = min(start + max_cols, n_cols)
            chunk = pivot.iloc[:, start:end]
            latex_str = chunk.replace(0.0, '').fillna('').round(2).to_latex(
                index=True,
                escape=False,
                float_format="%.3f",
                longtable=False,
                # formatters=formatters,
                position='h',
                caption=caption if start == 0 else None,
                multicolumn_format='c',
            )
            with open(path, "a") as f:
                f.write(latex_str)


# split_and_write_latex_table(formatted_pivot, f"output/tables/peaks_overview_split_{channel}.tex")

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
#
#%%# %%
#%%# %%
