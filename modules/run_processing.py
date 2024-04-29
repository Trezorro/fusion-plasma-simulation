# %%
import pandas as pd
import glob
import numpy as np
import re
from ydata_profiling import ProfileReport

# %%
# data_dir = 'shots/'
data_dir = '../data/LHD_labeled_TCV/'
COLS_META = [
    "ShotNum",
    "time",
]
COLS_CONTROL = [
    "IP",  # Current (niet reference lijn voor controller, maar de ware input. Dan laat je control bij control)
    "gas_fringes",  # Ingepompte gas
    "NBI",  # manieren om te verhitten: colliding Neutral beam injection
    "ECRH",  # magnetron.
    "a_minor",  # reel gemeten plasma shape a k d (horizontale radius
    "KAPPA",
    "DELTA"  # inkerbovenhoek nar links vanuit hetmidden
]
COLS_DATA = [
    # "FIR",  # density lijn Interferometer
    "FIR_core",  # For the March dataset of 260 shots, the FIR_core signal is the same as FIR.
    "PD",  # photodiode lijn op de divertor
    "DML",  # Magnetische respons  correleert met de energie in het plasma
    "POHM",  # Gemeten power waarde meet de power die uit wrijving komt
    "Z_axis"  # center Plasma positie in de verticale lijn. deviation van reference is betekenis. 
]
COLS_LABEL = ["LHD_label"]
ALL_SIG_COLLS = COLS_META + COLS_CONTROL + COLS_DATA


# %%
def get_shot_index(data_dir: str) -> tuple[dict[int, str], dict[int, str]]:
    if data_dir[-1] != '/':
        data_dir += '/'
    sig_all_names = glob.glob(data_dir + 'TCV_DATA*.parquet')
    # use regex to get sample number:
    sig_all = {int(re.findall(r'\d+', x)[0]): x for x in sig_all_names}
    shot_no_list = list(sig_all.keys())
    label_all = glob.glob(data_dir + 'TCV_*_apau_labeled.csv')
    label_all = {
        int(x.split("TCV_")[1].split("_apau_labeled.csv")[0]): x
        for x in label_all
    }
    label_no_list = list(label_all.keys())
    assert set(shot_no_list) <= set(
        label_no_list
    ), f"Not all shots have labels: {set(shot_no_list) - set(label_no_list)}"
    print(f"All shots: {shot_no_list}")
    print("Amount of shots: ", len(shot_no_list))
    print(f"Labels: {len(label_all)}")
    return sig_all, label_all


def check_time_consistency(signal_df, frequency_tolerance=3e-3):
    time_diff = signal_df['time'].iloc[1:-2].diff()
    t_start = signal_df['time'].iloc[0]
    t_end = signal_df['time'].iloc[-1]
    frequency = 1 / time_diff.mean()
    step_size_std = time_diff.std()
    is_consistent = step_size_std < 1e-7 and abs(frequency - 1e4) < frequency_tolerance
    inconsistent_steps = ~np.isclose(
        time_diff[1:], time_diff.mean(), atol=1e-7, rtol=1e-2, equal_nan=True)
    if 0< inconsistent_steps.sum() < 6:
        print(f"Inconsistent steps at: {signal_df['time'].iloc[2:-2][inconsistent_steps].to_list()}")
    # print(f"Frequency: {frequency}, is broadly consistent: {is_consistent}")
    # print(f"{inconsistent_steps.sum()} steps out of {len(inconsistent_steps)} were not exaclty the same as the mean step size.")
    return is_consistent, inconsistent_steps.sum(), frequency, step_size_std, t_start, t_end


#%% Function definition: Combine all shots into one dataframe
def combine_all_shots(sig_all: dict[int, str],
                      label_all: dict[int, str],
                      min_steps_filter: int = 5000,
                      frequency_tolerance=3e-3) -> pd.DataFrame:
    """Load and combine all shots into one dataframe. Check for consistency and discrepancies. 

    Guarantuees a frequency of 10 kHz and that the shot is at least min_steps_filter long.
    
    Args:
        sig_all (dict[int, str]): _description_
        label_all (dict[int, str]): _description_
        min_steps_filter (int, optional): _description_. Defaults to 5000.

    Returns:
        pd.DataFrame: A dataframe with all shots combined. Index is time. 
    """
    # init counters for discrepancies and memory usage
    time_discrepancy, shot_num_discrepancy, label_shot_num_discrepancy, length_discrepancy, too_short = 0, 0, 0, 0, 0
    time_inconsistency = 0
    nans_replaced = 0
    memory = 0

    all_shot_dfs = []  # list to store loaded and processed dataframes
    for shotno in sig_all.keys():
        sig = pd.read_parquet(sig_all[shotno])
        label = pd.read_csv(label_all[shotno])
        if len(sig) < min_steps_filter:
            print(
                f"Skipping shot {shotno} because it has less than {min_steps_filter} steps ({len(sig)})"
            )
            too_short += 1
            continue

        ### Assertions for consistency and discrepancies ###
        if len(sig) != len(label):
            print(
                f"Length of signal and label do not match for shot {shotno}: {len(sig)} != {len(label)}. ({len(sig) - len(label):+d})"
            )
            length_discrepancy += 1
        elif not np.allclose(sig["time"], label["time"]):
            print(f"Time values do not match for shot {shotno}")
            time_discrepancy += 1
        # check monotonicity of time
        if not sig["time"].is_monotonic_increasing:
            print(
                f"Time is not monotonically increasing for shot signal {shotno}"
            )
        if not label["time"].is_monotonic_increasing:
            print(f"Time is not monotonically increasing for label {shotno}")

        # Check shot number column consistency. Should probably be fine, but always good to check.
        if sig["ShotNum"].iloc[0] != shotno:
            print(f"Shot number does not match for shot {shotno}")
            shot_num_discrepancy += 1

        # check time consistency
        is_consistent, n_inconsistent, freq, step_size_std, t_start, t_end = check_time_consistency(
            sig, frequency_tolerance=frequency_tolerance)
        if not is_consistent or n_inconsistent > 10:
            print(f"Time is not consistent for shot {shotno}: Frequency: {freq}, Standard deviation of steps: {step_size_std}. {n_inconsistent} steps have a different time step.")
            print("Skipping shot.")
            time_inconsistency += 1
            continue
        ### End of consistency checks ### 

        
        # extract columns from signal
        shot_out = sig[ALL_SIG_COLLS].reset_index(
            names='time_step').set_index("time")
        # resample labels with ffill to time steps of signal
        label = label.set_index("time")
        label = label.reindex(shot_out.index, method='nearest', tolerance=0.01)
        # add labels
        shot_out.join(label[COLS_LABEL], on="time")
        # count amount of columns with any nans 
        nan_cols = shot_out[COLS_CONTROL + COLS_DATA].isnull().any(axis=0).sum()
        if nan_cols > 0:
            print(f"Found {nan_cols} cols with NaN in shot {shotno}")
            nans_replaced += 1
            # replace nans with 0 if they occur in the X or C columns
            shot_out[COLS_CONTROL + COLS_DATA] = shot_out[COLS_CONTROL + COLS_DATA].fillna(0)
        all_shot_dfs.append(shot_out)
        # print all_data size in memory
        memory += shot_out.memory_usage().sum()
        print(f"Memory usage: {memory / 1e6} MB")
    print(
        f"Total shots: {len(sig_all)} of which {too_short} had less than {min_steps_filter} steps. Output total: {len(all_shot_dfs)}"
    )
    print(f"Length discrepancy: {length_discrepancy}")
    print(f"Time discrepancy: {time_discrepancy} (of shots without length discrepancy)")
    print(f"Time inconsistency: {time_inconsistency}")
    print(f"Shot number discrepancy: {shot_num_discrepancy}")
    print(f"Label shot number discrepancy: {label_shot_num_discrepancy}")
    print(f"NaNs replaced: {nans_replaced}")
    return pd.concat(all_shot_dfs)


def load_shot(shotno: int, sig_all: dict[int, str], label_all: dict[int, str]):
    """Simple helper function to load a shot from the dataset.

    Args:
        shotno (int): shot number
        sig_all (dict[int,str]): dictionary of shot number to signal file path
        label_all (dict[int,str]): dictionary of shot number to label file path
    """
    sig_df = pd.read_parquet(sig_all[shotno])
    label_df = pd.read_csv(label_all[shotno])
    return sig_df, label_df

def generate_report(data_path: str):
    data_df = pd.read_parquet(data_path)
    profile = ProfileReport(data_df, title="Profiling Report")
    profile.to_file("data_report.html")


#%% Run!

if __name__ == "__main__":
    sig_all, label_all = get_shot_index(data_dir)
    data_df = combine_all_shots(sig_all, label_all)
    date = pd.Timestamp.now().strftime("%Y_%m_%d")
    file_name = f"./data/{date}-all_preprocessed.parquet"
    data_df.to_parquet(file_name)
    print("Saved to ", file_name)
    


# %%
generate_report("data/2024_04_23-all_preprocessed.parquet")
# %%
data_df = pd.read_parquet("data/2024_04_23-all_preprocessed.parquet")
#%%

def analyze_nans(df):
    """Per shotNo, analyze which columns have NaNs and how many.

    Also checks and counts whether there are consecutive non-NaNs in the columns with Nan values, so they can still be used.
    """
    nan_cols = df.columns[df.isnull().any()]
    nan_counts = df[nan_cols].isnull().sum()
    consecutive_non_nans = {}
    for col in nan_cols:
        consecutive_non_nans[col] = df[col].notnull().astype(int).groupby(df[col].isnull().cumsum()).sum().max()
    summary = pd.DataFrame({
        "ShotNum": df["ShotNum"].iloc[0],
        "NaNs": nan_counts, 
                            "NaN ratio": nan_counts / len(df),
                            "Consecutive non-NaNs": consecutive_non_nans})
    summary['Consecutive ratio'] = summary['Consecutive non-NaNs'] / len(df)
    summary["Small C-ratio"] = summary["Consecutive ratio"] < 0.4
    summary['Total'] = len(df)
    return summary

results = []
shots_n = data_df["ShotNum"].nunique()
for group, df in data_df.groupby("ShotNum"):
    summmary = analyze_nans(df)
    if summmary.empty:
        continue
    results.append(summmary)
    print(f"Shot {group}:")
    print(summmary)
res_df= pd.concat(results)
# %% What columns are most often unusable?
print("These are the columns that are most often unusable (they have no usable window without NaNs):")
res_df["Small C-ratio"].groupby(level=0).sum().sort_values(ascending=False)
# %% How many shots have unusable columns? And how many columns are unusable?
unusable_counts = res_df.groupby("ShotNum")["Small C-ratio"].sum().value_counts()
print(f"Out of {shots_n} shots, {unusable_counts.drop(0).sum()} have unusable columns.")
print(unusable_counts)

# %%
