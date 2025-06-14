#%%
import pandas as pd
import os
import argparse

CSV = "output/wand_overview.csv"
CACHE_DIR = 'ouptut/test_cache'
# Read CSV
#%%
df = pd.read_csv(CSV)

# Relevant columns
group_cols = [
    'model.Class',
    'model.params.model_params.c_channels',
    'data.history_length',
    'Cond_dim',
    'model.params.prior'
]

cache_col = 'test_cache_name'

caches = """
chan_brown_smC_BIG.h5
chan_brown_smC-rk4_40.h5
channel_levy_smC.h5
chan_normal_smC_BIG_retry2.h5
chan_normal_smC-rk4_40.h5
chan_resample_smC_BIG.h5
double_history_seq_normal_BIG2_reeval_rk440
doublehistory_seq_normal_smC_BIG2_cache2.h5
doublehistory_seq_normal_smC_BIG2.h5
DoubleHistory_seq_normal_smC_BIG.h5
Flow_brown_normal_allC_BIG.h5
Flow_seq_brown_normal_allC_BIG.h5
R2_seq_normal_smC_rk4_40.h5
R_seq_levy_smC_BIG.h5
seq_brown-sig06_smC.h5
seq_brown_smC-2-rk4_40.h5
seq_brown_smC_BIG.h5
seq_brown_smC_LONG.h5
seq_brown_smC-rk4_40.h5
seq_constant_smC_BIG.h5
seq_CONST_smC.h5
seq_levy_smC-rk4_40.h5
seq_normal_smC_BIG4.h5
seq_normal_smC_BIG4_reeval_rk440.h5
seq_normal_smC_rk4_40.h5
seq_resample_smC_BIG.h5
seq_resample_smC-rk4_40.h5
test_cache_tinyhistory.h5
test_cache_unflow.h5
test_seq_normal_smC_rk4_40.h5
test_tinyhistory.h5
tinyhistory_seq_normal_smC_BIG.h5
UnFlow_brown_normal_allC_BIG.h5
UnFlow_brown_normal_smC_BIG.h5
UnFlow_seq_brown_smC.h5""".split()


# def cache_exists(cache_name, cache_dir):
#     if pd.isna(cache_name):
#         return False
#     cache_path = os.path.join(cache_dir, f'{cache_name}.h5')
#     return os.path.isfile(cache_path)


# Add checkmark column
df['cache_exists'] = df['test_cache_name'].apply(lambda v: (v+'.h5') in caches)
# Drop columns starting with 'test/'
df = df.loc[:, ~df.columns.str.startswith('test/')]
# Reorder columns: group_cols + [cache_col] + rest
important_cols = ['cache_exists'] +group_cols + ["Name",  cache_col, 'test_cache_mode', "epoch"]
other_cols = [col for col in df.columns if col not in important_cols]
df = df[important_cols + other_cols]
df
#%%

# Check for cache existence

#%%
# Group and print
groups = df.groupby(group_cols)
pd.set_option('display.max_colwidth', 200)

for group_keys, group_df in groups:
    print(f'\nGroup:')
    for col, val in zip(group_cols, group_keys):
        print(f'  {col}: {val}')
    print('  Caches:')
    for _, row in group_df.iterrows():
        print(f"    {row[cache_col]} {'✅' if row['cache_exists'] else ''}")


# %%
# Create a pivot table summarizing available caches per group
def summarize_caches(group):
    available = group[group['cache_exists']][cache_col].tolist()
    if available:
        return '✅'+', ✅'.join(available)
    else:
        return '⚠️ No cache available!'

pivot = df.groupby(group_cols).apply(summarize_caches).reset_index(name='Available Caches')

pivot
# %%
