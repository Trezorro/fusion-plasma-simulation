#%% Imports
import pandas as pd
from pathlib import Path
import h5py
from sympy import Id
from tqdm import tqdm
import matplotlib.pyplot as plt

CACHE_DIR = Path('output/test_cache')  # contains .h5 and .jsons
# List all .h5 files in the CACHE_DIR
CACHED_H5_LIST = list(CACHE_DIR.glob('*.h5'))
CACHED_H5_LIST
#%% show keys available in mode key for a cache
"""From serialization function
for name, arr in zip(['delta', 'SE', 'CI_lower', 'CI_upper'], [delta, SE, lower, upper]):
                df = pd.DataFrame(columns=["L", "D", "H", "any"], index=["L", "D", "H", "any"], data=arr.cpu().numpy())
                df.to_hdf(cache_obj.h5_path, key=f'modes/{self.condition}/{name}', mode='a')
        for state in [
            "transition_counts_pred", "transition_counts_target", "transition_counts_sq_error", "transition_gt0_pred",
            "transition_gt0_target"
        ]:
            value = getattr(self, state) / self.total_hits
            df = pd.DataFrame(columns=["L", "D", "H", "any"], index=["L", "D", "H", "any"], data=value.cpu().numpy())
            if cache_obj is not None:
                df.to_hdf(cache_obj.h5_path, key=f'modes/{self.condition}/{state}', mode='a')
"""


def get_h5_tree(g, prefix='', level=1):
    contents = []
    if level == 0:
        return contents
    for key in g.keys():
        item = g[key]
        if isinstance(item, h5py.Group):
            contents.extend(get_h5_tree(item, prefix + key + '/', level=level - 1))
    if not contents:
        return [g.name]
    else:
        return contents


def print_h5_tree(g, prefix='', level=1):
    if level == 0:
        return
    for key in g.keys():
        item = g[key]
        print(prefix + key)
        if isinstance(item, h5py.Group):
            print_h5_tree(item, prefix + key + '/', level=level - 1)


def list_mode_keys(h5path, tree=2):
    with h5py.File(h5path, 'r') as f:
        if tree:
            print_h5_tree(f['modes'], level=tree)
        else:
            return list(f['modes'].keys())


def get_mode_keys(h5path):
    with h5py.File(h5path, 'r') as f:
        full_keys = get_h5_tree(f['modes'], level=3)


list_mode_keys('output/test_cache/seq_CONST_smC.h5', 1)
get_mode_keys('output/test_cache/seq_CONST_smC.h5')
#%% Load a subkey into a df
EXAMPLE_CACHE = CACHED_H5_LIST[0]
EXAMPLE_KEY = 'modes/H_only_Wh/transition_gt0_pred'

df = pd.read_hdf(EXAMPLE_CACHE, key=EXAMPLE_KEY)
"""
	L	D	H	any
L	0.000000	0.035862	0.043863	0.079726
D	0.022003	0.000000	0.053722	0.075725
H	0.061009	0.060294	0.000000	0.121303
any	0.083012	0.096157	0.097585	0.276754

"""
df

#%% Change to long format.
long = df.reset_index(names='from').melt(id_vars='from', var_name='to', value_name='value')
long.assign(cache=EXAMPLE_CACHE.stem)
#%% For input, Add cols for the meta dimensions: cache_name, measure quantity, etc

MEASURES = [
    'transition_counts_pred',
    'transition_counts_target',
    'transition_counts_sq_error',
    'CI_lower',
    'CI_upper',
    'SE',
    'delta',
    'transition_gt0_pred',
    'transition_gt0_target',
]
CONDS = [
    'D_in_Wh',
    'D_not_in_Wh',
    'D_only_Wh',
    'H_in_Wh',
    'H_not_in_Wh',
    'H_only_Wh',
    'L_in_Wh',
    'L_not_in_Wh',
    'L_only_Wh',
    'any_Wh',
    'mixed',
]


def df_from_cache(h5_path):
    dfs = []
    for cond in CONDS:
        for measure in MEASURES:
            try:
                df = pd.read_hdf(h5_path, key=f'/modes/{cond}/{measure}')
                long = df.reset_index(names='from').melt(id_vars='from', var_name='to', value_name='value')
                if '_target' in measure:
                    if h5_path.stem != 'seq_normal_smC_BIG4_reeval_rk440':
                        continue  # only take ground truth values from one of the models. They match.
                    else:
                        details = long.assign(
                            name="Ground Truth", condition=cond, measure=measure.replace('_target', '')
                        )
                else:
                    details = long.assign(name=h5_path.stem, condition=cond, measure=measure.replace('_pred', ''))
                dfs.append(details)
            except KeyError as e:
                tqdm.write(f"Missing key: /modes/{cond}/{measure} in {h5_path.name} ({e})")
                continue
    return pd.concat(dfs, ignore_index=True)


df_from_cache(EXAMPLE_CACHE).sort_values('name')
df_from_cache(Path('output/test_cache') / 'seq_normal_smC_BIG4_reeval_rk440.h5').sort_values('name')


#%% loop through all caches and concatenate
def combine_caches(cache_list):
    dfs = []
    for cache in tqdm(
        cache_list,
        desc="Processing caches",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}: {postfix}"
    ):
        tqdm.write(f"Processing {cache.name}")
        dfs.append(df_from_cache(cache))
    return pd.concat(dfs, ignore_index=True)


combined_df = combine_caches(CACHED_H5_LIST)

#%% Map cache name to the pretty thesis name and Conditioning Dim, Prior, C Variation,

CSV = "output/XR-overview.csv"


def get_cache_overview(csvpath):
    df = pd.read_csv(csvpath)
    group_cols = [
        'model.Class', 'data.history_length', 'model.params.model_params.c_channels', 'Cond_dim', 'model.params.prior'
    ]

    # Add checkmark column
    # df['cache_exists'] = df['test_cache_name'].apply(lambda v: (v + '.h5') in caches)
    # Drop columns starting with 'test/'
    df = df.loc[:, ~df.columns.str.startswith('test/')]
    # Reorder columns: group_cols + [cache_col] + rest
    important_cols = group_cols + ["Name", 'test_cache_name', 'base_run', 'run_name', 'Created', 'test_cache_mode']
    other_cols = [col for col in df.columns if col not in important_cols]
    df = df[important_cols + other_cols]
    rename_map = {
        'model.Class': 'Model',
        'data.history_length': 'History Length',
        'model.params.model_params.c_channels': 'Full Covariates',
        'Cond_dim': 'WH concat dim',
        'model.params.prior': 'Prior'
    }
    # Rename 'model.Class' values for display
    df['model.Class'] = df['model.Class'].replace({'FlowModule': 'Flow Matching', 'UnFlowModule': 'Base Unet'})
    group_cols = list(rename_map.values())
    df = df.rename(columns=rename_map)
    # Show possible unique group combinations
    possible_groups = df.set_index('test_cache_name')[group_cols].drop_duplicates().sort_values(group_cols)
    print("Possible groups:")
    print(possible_groups.to_string(index=True))
    possible_groups['human_name'] = """
            Unet-Channel-Brownian  
            Unet-Sequence-Brownian  
            Unet-Sequence-AllCov-Brownian  
            FM-Sequence-Tiny-Gaussian  
            FM-Channel-Brownian  
            FM-Channel-CP  
            FM-Channel-Gaussian  
            FM-Channel-Resampled  
            FM-Sequence-Brownian  
            FM-Sequence-Constant  
            FM-Sequence-CP  
            FM-Sequence-Gaussian  
            FM-Sequence-Resampled  
            FM-Channel-AllCov-Brownian  
            FM-Sequence-AllCov-Brownian  
            FM-Sequence-2x-Gaussian  """.split()
    return possible_groups


runs = get_cache_overview(CSV)
runs.loc['Ground Truth'] = ['-', '-', '-', '-', '-', 'Ground Truth']
runs

#%% map the cache name to the rows in runs
final_df = combined_df.join(runs, on='name')

#%%
MODEL_ORDER =  [
    'FM-Sequence-Tiny-Gaussian',
    'FM-Sequence-Gaussian',
    'FM-Sequence-2x-Gaussian',
    'FM-Sequence-Brownian',
    'FM-Sequence-Constant',
    'FM-Sequence-CP',
    'FM-Sequence-Resampled',
    'FM-Channel-Gaussian',
    'FM-Channel-Brownian',
    'FM-Channel-CP',
    'FM-Channel-Resampled',
    'FM-Channel-AllCov-Brownian',
    'FM-Sequence-AllCov-Brownian',
    # 'Unet-Sequence-AllCov-Brownian',
    'Unet-Sequence-Brownian',
    'Unet-Channel-Brownian',
    'Ground Truth'
]

#%% Pivot to Latex format
CONDITION = 'any_Wh'
pivoted = final_df.query(f"condition=='{CONDITION}'"
                        ).pivot_table(index=['from', 'human_name'], columns=['to', 'measure'], values='value')

# Sort by Model, History Length, Full Covariates, WH concat dim, Prior
sort_cols = ['Model', 'History Length', 'Full Covariates', 'WH concat dim', 'Prior']
if all(col in final_df.columns for col in sort_cols):
    # Get the sort order for human_name based on those columns
    sort_order = (
        final_df.drop_duplicates('human_name').set_index('human_name')[sort_cols].sort_values(sort_cols).index
    )
    sort_order = [
        'FM-Sequence-Tiny-Gaussian',
        'FM-Sequence-Gaussian',
        'FM-Sequence-2x-Gaussian',
        'FM-Sequence-Brownian',
        'FM-Sequence-Constant',
        'FM-Sequence-CP',
        'FM-Sequence-Resampled',
        'FM-Channel-Gaussian',
        'FM-Channel-Brownian',
        'FM-Channel-CP',
        'FM-Channel-Resampled',
        'FM-Channel-AllCov-Brownian',
        'FM-Sequence-AllCov-Brownian',
        # 'Unet-Sequence-AllCov-Brownian',
        'Unet-Sequence-Brownian',
        'Unet-Channel-Brownian',
        'Ground Truth'
    ]
    sort_order_trans = ['L', 'D', 'H', 'any']
    sort_order_cols_inner = [
        'transition_counts',
        'transition_counts_sq_error',
        'transition_gt0',
        'delta',
        # 'SE',
        # 'CI_lower',
        # 'CI_upper',
    ]

    # Reindex pivoted to match the sorted human_name order
    pivoted = pivoted.reindex(
        pd.MultiIndex.from_product([sort_order_trans, sort_order], names=['From', ''])
    ).reindex(pd.MultiIndex.from_product([sort_order_trans, sort_order_cols_inner], names=['To', '']),
              axis='columns').dropna(how='all')
pivoted
#%%
pivoted.to_excel(f"output/tables/mode_transitions{CONDITION}.xlsx", index=True, float_format="%.3f")
# %%
# df_latex = pivoted.round(3)
# df_latex.replace(0, '', inplace=True)


# Map measure names to LaTeX symbols
measure_latex = {
    'transition_counts': r'\mathbb{E}',
    'transition_counts_sq_error': r'\mathrm{SE}^2',
    'transition_gt0': r'P(T>0)',
    'delta': r'\Delta',
    'SE': r'\pm',
}

# Rename columns in df_latex and pivoted
def rename_cols(df):
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            to, measure = col
            measure_sym = measure_latex.get(measure, measure)
            new_cols.append((to, measure_sym))
        else:
            new_cols.append(measure_latex.get(col, col))
    df.columns = pd.MultiIndex.from_tuples(new_cols, names=df.columns.names)
    return df

latex_df = rename_cols(pivoted)
latex_str = latex_df.replace(0.0, '').fillna('').round(2).to_latex(
    index=True,
    escape=False,
    float_format="%.3f",
    longtable=True,
    # formatters={'transition_counts': lambda num: f"\mathbb{{E}}"},
    position='h',
    multicolumn_format='c',
)
# print(latex_str)
# Save the LaTeX string to a file
output_path = Path(f"output/tables/{CONDITION}.tex")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    f.write(latex_str)
print(f"LaTeX table saved to {output_path}")


# %%


# %%
# %%
pivoted_T = final_df.query(f"condition=='{CONDITION}'"
                          ).pivot_table(index=['to', 'human_name'], columns=['from', 'measure'], values='value')

# Sort by Model, History Length, Full Covariates, WH concat dim, Prior
sort_cols = ['Model', 'History Length', 'Full Covariates', 'WH concat dim', 'Prior']
if all(col in final_df.columns for col in sort_cols):
    # Get the sort order for human_name based on those columns
    sort_order = (
        final_df.drop_duplicates('human_name').set_index('human_name')[sort_cols].sort_values(sort_cols).index
    )
    sort_order = [
        'FM-Sequence-Tiny-Gaussian',
        'FM-Sequence-Gaussian',
        'FM-Sequence-2x-Gaussian',
        'FM-Sequence-Brownian',
        'FM-Sequence-Constant',
        'FM-Sequence-CP',
        'FM-Sequence-Resampled',
        'FM-Channel-Gaussian',
        'FM-Channel-Brownian',
        'FM-Channel-CP',
        'FM-Channel-Resampled',
        'FM-Channel-AllCov-Brownian',
        'FM-Sequence-AllCov-Brownian',
        # 'Unet-Sequence-AllCov-Brownian',
        'Unet-Sequence-Brownian',
        'Unet-Channel-Brownian',
        'Ground Truth'
    ]
    sort_order_trans = ['L', 'D', 'H', 'any']
    sort_order_cols_inner = [
        'transition_counts',
        'transition_counts_sq_error',
        'transition_gt0',
        'delta',
        # 'SE',
        # 'CI_lower',
        # 'CI_upper',
    ]

    # Reindex pivoted_T to match the sorted human_name order
    pivoted_T = pivoted_T.reindex(pd.MultiIndex.from_product([sort_order_trans, sort_order], names=['To', ''])).reindex(
        pd.MultiIndex.from_product([sort_order_trans, sort_order_cols_inner], names=['From', '']), axis='columns'
    ).dropna(how='all')
pivoted_T

# %%


# %%
def plot_multifacet_bar(pivoted, measure='transition_counts'):
    """
    Plots a multifacet horizontal bar plot for each FROM state (facet),
    with models as bars. Only one measure per transition is shown.
    Args:
        pivoted: MultiIndex DataFrame (to, model) x (from, measure)
        measure: str, which measure to plot (must match column level 1)
    """
    # Define colors for each from_state
    color_map = {'L': '#1f77b4', 'D': '#ff7f0e', 'H': '#d62728', 'any': '#444444'}
    from_states = list("LDH")  # columns in pivoted (level 0)
    to_states = from_states
    models = MODEL_ORDER

    n_facets = len(from_states)
    fig, axes = plt.subplots(1, n_facets, figsize=(5 * n_facets, 6), sharey=True)
    if n_facets == 1:
        axes = [axes]

    # Find global max for xlim
    global_max = pivoted.xs(measure, level=1, axis=1).max().max()

    for i, from_state in enumerate(from_states):
        ax = axes[i]
        # For this facet, plot all TO states (rows) for this FROM (col)
        data = pivoted[(from_state, measure)]  # index: to_state, columns: model
        for j, to_state in enumerate(to_states):
            vals = data.loc[to_state].reindex(models)
            y_pos = [y + j * 0.8 / len(to_states) for y in range(len(models))]
            color = color_map.get(to_state, '#888888')
            ax.barh(y_pos, vals, left=0, height=0.8 / len(to_states), label=f"{to_state}", color=color)
        ax.set_title(f"From: {from_state}")
        ax.set_xlim(0, global_max * 1.05)
        ax.set_yticks([y + 0.4 for y in range(len(models))])
        ax.set_yticklabels(models)
        if i == 0:
            ax.set_ylabel("Model")
        else:
            ax.set_ylabel("")
        ax.set_xlabel(measure)
        ax.legend(title="To state")
    plt.tight_layout()
    plt.show()


# Example usage (uncomment to use):
plot_multifacet_bar(pivoted_T, measure='transition_counts')

# %%
