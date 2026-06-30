"""Data preprocessing for PlasmaFlow training data.

Reads individual TCV shot parquet files from the Poels et al. public dataset,
generates surrogate confinement mode labels using the FNOLSTM classifier on the
PD (H-alpha divertor photodiode) channel, and writes a single combined parquet
file used for all training runs.

Run locally only: requires ydata-profiling, which is not installed on Snellius.

Input:  {DATA_INPUT_DIR}/*.parquet  (one file per TCV shot)
Output: data/{DATE}-{DATA_SET_NAME}.parquet     (all shots, with LHD_label column)

LHD_label is the confinement mode integer label (0=L, 1=D, 2=H) produced by
the FNOLSTM classifier (configs/MHD_model_yoerie/weights_PD.pt). The classifier
output column label_conf is renamed to LHD_label in the combined output file.

Column names in the output must match data.cols in configs/plasmaflow.yaml exactly.
"""
# To be run on local machine. Preferably in vscode with correct cell execution.
# %%
import glob
import re
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import rich
import rich.progress
import rich.traceback
from ydata_profiling import ProfileReport

rich.traceback.install()

# %%
# data_dir = 'shots/'
DATA_INPUT_DIR = Path("./data/public_data_set/data")
DATA_SET_NAME = "TCV_shots_V2"
DATE = pd.Timestamp.now().strftime("%Y_%m_%d")

COLS_META = [
    "ShotNum",
    "time",
    "LHD_label",
]

##### Old data set ######
COLS_CONTROL = [
    "IP",  # Current (niet reference lijn voor controller, maar de ware input. Dan laat je control bij control), 
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
##### New data set ######
COLS_CONTROL = [
    "IP",          # I_p,ref: reference current (or swap for IPLA)
    # gas_fringes missing; consider GWfr as partial proxy
    "PNBI",        # replaces NBI (beam 1)
    "PNBI2",       # new: beam 2 now separated
    "PECRH",       # replaces ECRH
    "MINRAD",      # replaces a_minor
    "KAPPA",       # same
    "DELTA_TOP",   # replaces DELTA (now split)
    "DELTA_BOTTOM",
]

COLS_DATA = [
    "FIR_LIDs_core",  # replaces FIR_core
    "Halpha1",        # replaces PD (H-alpha divertor photodiode)
    "DML",            # same
    "POHM",           # same
    "Z_axis",         # same
]


COLS_LABEL = ["LHD_label"]
ALL_SIG_COLLS = COLS_META + COLS_CONTROL + COLS_DATA



# %%
def index_shot_names(data_dir: str) -> tuple[dict[int, str], dict[int, str]]:
    if data_dir[-1] != '/':
        data_dir += '/'
    sig_all_names = glob.glob(data_dir + 'TCV_DATA*.parquet')
    # use regex to get sample number:
    sig_all = {int(re.findall(r'\d+', x)[0]): x for x in rich.progress.track(sig_all_names, "Indexing shots")}
    shot_no_list = list(sig_all.keys())
    label_all = glob.glob(data_dir + 'TCV_*_*_labeled.csv')
    label_all = {int(x.split("TCV_")[1].split("_")[0]): x for x in rich.progress.track(label_all, "Indexing labels")}
    label_no_list = list(label_all.keys())
    assert set(shot_no_list
              ) <= set(label_no_list), f"Not all shots have labels: {set(shot_no_list) - set(label_no_list)}"
    print(f"All shots: {shot_no_list}")
    print("Amount of shots: ", len(shot_no_list))
    print(f"Labels: {len(label_all)}")
    return sig_all, label_all


def analyze_nans(one_shot_df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[int, int]]:
    """Per shotNo, analyze which columns have NaNs and how many.

    Also checks and counts whether there are consecutive non-NaNs in the columns with Nan values,
    so they can still be used.

    Assumes time_step column is present in the dataframe.

    Returns a summary dataframe and a tuple of the first and last time step of the longest consecutive non-NaNs.
        The dataframe has the following columns:
        - ShotNum: The shot number
        - NaNs: The amount of NaNs per column
        - NaN ratio: The ratio of NaNs per column
        - Consecutive non-NaNs: The maximum amount of consecutive non-NaNs per column
        - Consecutive ratio: The ratio of consecutive non-NaNs per column
        - Small C-ratio: A boolean indicating whether the consecutive ratio is below 0.3
        - Total: The total amount of rows in the dataframe
        The index is the original shot df column names, where NaNs were present.
        If the dataframe is empty, it means there were no NaNs in the dataframe.
    """
    one_shot_df = one_shot_df.copy()
    nan_cols = one_shot_df.columns[one_shot_df.isnull().any()]
    nan_counts = one_shot_df[nan_cols].isnull().sum()
    nan_in_row = one_shot_df[nan_cols].isnull().any(axis=1)
    one_shot_df['nan_splits'] = nan_in_row.cumsum()
    consecutive_non_nans = {}
    for col in nan_cols:
        # Split groups by NaNs via cumsum, then get the maximum length of consecutive non-NaNs
        around_nan_windows = one_shot_df[col].notnull().astype(int).groupby(one_shot_df[col].isnull().cumsum())
        consecutive_non_nans[col] = around_nan_windows.sum().max()
    # get start and end step of the longest window of non-NaNs over all columns:
    usable_rows = one_shot_df.dropna()
    if usable_rows.empty:
        first_step, last_step = (0, 0)
    else:
        longest_window = usable_rows['nan_splits'].mode().values[0]
        viable_time_steps = one_shot_df.loc[one_shot_df['nan_splits'] == longest_window, 'time_step']
        first_step = viable_time_steps.iloc[0]
        last_step = viable_time_steps.iloc[-1]
    summary = pd.DataFrame(
        {
            "ShotNum": one_shot_df["ShotNum"].iloc[0],
            "NaNs": nan_counts,
            "NaN ratio": nan_counts / len(one_shot_df),
            "Consecutive non-NaNs": consecutive_non_nans
        }
    )
    summary['Consecutive ratio'] = summary['Consecutive non-NaNs'] / len(one_shot_df)
    summary["Small C-ratio"] = summary["Consecutive ratio"] < 0.3
    summary['Total'] = len(one_shot_df)
    return summary, (first_step, last_step)


def analyze_nans_over_all_shots(data_df: pd.DataFrame):
    if "Small C-ratio" in data_df.columns:
        # Got the summary already
        res_df = data_df
    else:
        # Got the full dataframe of all shots, to still analyze:
        results = []
        for group, df in data_df.groupby("ShotNum"):
            summmary, window = analyze_nans(df)
            if summmary.empty:
                continue
            results.append(summmary)
        res_df = pd.concat(results)
    # What columns are most often unusable?
    print("These are the columns that are most often unusable (they have no usable window without NaNs):")
    print(res_df["Small C-ratio"].groupby(level=0).sum().sort_values(ascending=False))
    # How many shots have some unusable columns? And how many columns are then unusable?
    shots_n = data_df["ShotNum"].nunique()
    unusable_counts = res_df.groupby("ShotNum")["Small C-ratio"].sum().value_counts()
    n_unusable_columns_per_shot = unusable_counts.drop(0).sum()
    print(
        f"Out of {shots_n} shots, {n_unusable_columns_per_shot} shots have 1 or more unusable columns. ({n_unusable_columns_per_shot/shots_n:.2%})"
    )
    print(unusable_counts)


def check_time_consistency(signal_df, frequency_tolerance=3e-3):
    time_diff = signal_df['time'].iloc[1:-2].diff()
    t_start = signal_df['time'].iloc[0]
    t_end = signal_df['time'].iloc[-1]
    frequency = 1 / time_diff.mean()
    step_size_std = time_diff.std()
    is_consistent = step_size_std < 1e-7 and abs(frequency - 1e4) < frequency_tolerance
    inconsistent_steps = ~np.isclose(time_diff[1:], time_diff.mean(), atol=1e-7, rtol=1e-2, equal_nan=True)
    if 0 < inconsistent_steps.sum() < 6:
        print(f"Inconsistent steps at: {signal_df['time'].iloc[2:-2][inconsistent_steps].to_list()}")
    # print(f"Frequency: {frequency}, is broadly consistent: {is_consistent}")
    # print(f"{inconsistent_steps.sum()} steps out of {len(inconsistent_steps)} were not exaclty the same as the mean step size.")
    return is_consistent, inconsistent_steps.sum(), frequency, step_size_std, t_start, t_end


#%% Function definition: Combine all shots into one dataframe
def combine_all_shots(
    sig_all: dict[int, str],
    label_all: dict[int, str],
    min_steps_filter: int = 5000,
    frequency_tolerance=3e-3
) -> pd.DataFrame:
    """Load and combine all shots into one dataframe. Check for consistency and discrepancies. 

    Guarantuees a frequency of 10 kHz and that the shot is at least min_steps_filter long.

    Replaces NaNs in the NBI column with 0, as per Yoeri's suggestion.
    Slices the dataframe to only include the longest window of non-NaNs for all other columns.
    Will not include shots that have too many NaNs in the X or C columns for a long usable window.
    
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
    nan_rejects = 0
    memory = 0

    all_shot_dfs = []  # list to store loaded and processed dataframes
    nan_summaries = []
    for shotno in rich.progress.track(sig_all.keys(), description="Loading and processing shots"):
        sig = pd.read_parquet(sig_all[shotno])
        label = pd.read_csv(label_all[shotno])
        rich.print("Reading shot", shotno, "Length:", len(sig), "Label length:", len(label))
        if len(sig) < min_steps_filter:
            rich.print(f"Skipping shot {shotno} because it has less than {min_steps_filter} steps ({len(sig)})")
            too_short += 1
            continue

        ### Assertions for consistency and discrepancies ###
        if len(sig) != len(label):
            # rich.print(
            #     f"Length of signal and label do not match for shot {shotno}: {len(sig)} != {len(label)}. ({len(sig) - len(label):+d})"
            # )
            length_discrepancy += 1
        elif not np.allclose(sig["time"], label["time"]):
            print(f"Time values do not match for shot {shotno}")
            time_discrepancy += 1
        # check monotonicity of time
        if not sig["time"].is_monotonic_increasing:
            print(f"Time is not monotonically increasing for shot signal {shotno}")
        if not label["time"].is_monotonic_increasing:
            print(f"Time is not monotonically increasing for label {shotno}")

        # Check shot number column consistency. Should probably be fine, but always good to check.
        if sig["ShotNum"].iloc[0] != shotno:
            print(f"Shot number does not match for shot {shotno}")
            shot_num_discrepancy += 1

        # check time consistency
        is_consistent, n_inconsistent, freq, step_size_std, t_start, t_end = check_time_consistency(
            sig, frequency_tolerance=frequency_tolerance
        )
        if not is_consistent or n_inconsistent > 10:
            rich.print(
                f"Time is not consistent for shot {shotno}: Frequency: {freq}, Standard deviation of steps: {step_size_std}. {n_inconsistent} steps have a different time step."
            )
            print("Skipping shot.")
            time_inconsistency += 1
            continue
        ### End of consistency checks ###

        # extract columns from signal
        shot_out = sig[ALL_SIG_COLLS].reset_index(names='time_step').set_index("time")
        # resample labels with ffill to time steps of signal
        label = label.set_index("time")
        label = label.reindex(shot_out.index, method='nearest', tolerance=0.01)
        # add labels
        shot_out = shot_out.join(label[COLS_LABEL], on="time")
        # print all_data size in memory
        memory += shot_out.memory_usage().sum()

        ### Handle NaNs ###
        # count amount of columns with any nans
        raw_nan_summary, longest_window = analyze_nans(shot_out)
        # Replace nans in NBI with 0, as per yoeri's suggestion
        shot_out.loc[shot_out["NBI"].isnull(), "NBI"] = 0
        nan_summary, (start_usable, end_usable) = analyze_nans(shot_out)
        # print a representation of the usable window in 100 steps
        last_time_step = shot_out["time_step"].iloc[-1]
        print(
            "Using", start_usable, "to", end_usable, ":", "-" * ceil(start_usable / 200) + "X" *
            ((end_usable - start_usable) // 200) + "-" * ceil((last_time_step - end_usable) / 200)
        )
        # Other columns with too many NaNs are problmeatic, so we drop the shot
        if not nan_summary.empty and nan_summary["Small C-ratio"].any():
            nan_summaries.append(nan_summary)  # for later analysis
            rich.print(
                f"Shot {shotno} has columns with too many NaNs ({nan_summary.index.tolist()}). Dropping the shot."
            )
            nan_rejects += 1
            continue
        # slice the dataframe to only include the longest window of non-NaNs, using the time_step start and end
        shot_out = shot_out.loc[shot_out["time_step"].between(start_usable, end_usable)]
        all_shot_dfs.append(shot_out)
    print(
        f"Total shots: {len(sig_all)} of which {too_short} had less than {min_steps_filter} steps. Output total: {len(all_shot_dfs)}"
    )
    print(f"Length discrepancy: {length_discrepancy}")
    print(f"Time discrepancy: {time_discrepancy} (of shots without length discrepancy)")
    print(f"Time inconsistency: {time_inconsistency}")
    print(f"Shot number discrepancy: {shot_num_discrepancy}")
    print(f"Label shot number discrepancy: {label_shot_num_discrepancy}")
    print(f"Rejected shots due to too many NaNs: {nan_rejects}")
    rich.print(f"Memory usage: {memory / 1e6} MB")
    nan_summary_df = pd.concat(nan_summaries)
    rich.print(nan_summary_df.sort_index())
    rich.print(nan_summary_df.index.value_counts())
    return pd.concat(all_shot_dfs)

def combine_public_dataset(
    shots_dir: Path,
    min_steps_filter: int = 5000,
    frequency_tolerance=1
) -> pd.DataFrame:
    """Load and combine all shots into one dataframe. Check for consistency and discrepancies. 

    Modified version for loading the TCV confinement state database https://zenodo.org/records/16631053

    Guarantuees a frequency of 10 kHz and that the shot is at least min_steps_filter long.

    Renames Halpha1 to PD for backwards compatibility with old code. And it is a simpler name (photodiode).

    Replaces NaNs in the NBI column with 0, as per Yoeri's suggestion.
    Slices the dataframe to only include the longest window of non-NaNs for all other columns.
    Will not include shots that have too many NaNs in the X or C columns for a long usable window.
    
    Args:
        min_steps_filter (int, optional): _description_. Defaults to 5000.

    Returns:
        pd.DataFrame: A dataframe with all shots combined. Index is time. 
    """
    all_shot_files = list(shots_dir.glob("TCV_confstate_*.parquet"))
    rich.print(f"Found {len(all_shot_files)} shot files to combine in dir '{shots_dir}'")
    if not all_shot_files:
        raise RuntimeError("No input parquets found in %s", shots_dir)
    # init counters for discrepancies and memory usage
    time_discrepancy, shot_num_discrepancy, label_shot_num_discrepancy, length_discrepancy, too_short = 0, 0, 0, 0, 0
    time_inconsistency = 0
    nan_rejects = 0
    memory = 0

    all_shot_dfs = []  # list to store loaded and processed dataframes
    nan_summaries = []
    for shot_file in rich.progress.track(all_shot_files, description="Loading and processing shots"):
        shot_num = re.search(r"(\d+)", str(shot_file.name)).group(0)
        sig = pd.read_parquet(shot_file)
        sig.insert(0, 'ShotNum', shot_num)
        # rename to LHD_label to match the column name expected by data_loaders.py and plasmaflow.yaml
        sig.rename(columns={"label_conf": COLS_LABEL[0]}, inplace=True)

        rich.print("Reading shot", shot_num, "Length:", len(sig))
        if len(sig) < min_steps_filter:
            rich.print(f"Skipping shot {shot_num} because it has less than {min_steps_filter} steps ({len(sig)})")
            too_short += 1
            continue

        ### Assertions for consistency and discrepancies ###
        # check monotonicity of time
        if not sig["time"].is_monotonic_increasing:
            rich.print(f"Time is not monotonically increasing for shot signal {shot_num}")
            time_discrepancy += 1

        # check time consistency
        is_consistent, n_inconsistent, freq, step_size_std, t_start, t_end = check_time_consistency(
            sig, frequency_tolerance=frequency_tolerance
        )
        if not is_consistent or n_inconsistent > 10:
            rich.print(
                f"Time is not consistent for shot {shot_num}: Frequency: {freq}, Standard deviation of steps: {step_size_std}. {n_inconsistent} steps have a different time step."
            )
            rich.print("Skipping shot.")
            time_inconsistency += 1
            continue
        ### End of consistency checks ###

        # extract columns from signal
        missing_cols=[]
        for target_col in ALL_SIG_COLLS:
            if target_col not in sig.columns:
                rich.print(f"{target_col} column MISSING for shot {shot_num}. SKIPPING")
                missing_cols.append(target_col)
        if missing_cols:
            continue
        shot_out = sig[ALL_SIG_COLLS].reset_index(names='time_step').set_index("time")
        float_cols = shot_out.select_dtypes(include="float64").columns
        shot_out[float_cols] = shot_out[float_cols].astype("float32")
        shot_out.rename(columns={'Halpha1': 'PD'}, inplace=True)
        # resample labels with ffill to time steps of signal
        memory += shot_out.memory_usage().sum()
        rich.print("Memory:", memory)
        

        ### Handle NaNs ###
        # count amount of columns with any nans
        raw_nan_summary, longest_window = analyze_nans(shot_out)
        # Replace nans in NBI with 0, as per yoeri's suggestion
        shot_out.loc[shot_out["PNBI"].isnull(), "PNBI"] = 0
        nan_summary, (start_usable, end_usable) = analyze_nans(shot_out)
        # print a representation of the usable window in 100 steps
        last_time_step = shot_out["time_step"].iloc[-1]
        rich.print(
            "Using", start_usable, "to", end_usable, ":", "-" * ceil(start_usable / 200) + "X" *
            ((end_usable - start_usable) // 200) + "-" * ceil((last_time_step - end_usable) / 200)
        )
        # Other columns with too many NaNs are problmeatic, so we drop the shot
        if not nan_summary.empty and nan_summary["Small C-ratio"].any():
            nan_summaries.append(nan_summary)  # for later analysis
            rich.print(
                f"Shot {shot_num} has columns with too many NaNs ({nan_summary.index.tolist()}). Dropping the shot."
            )
            nan_rejects += 1
            continue
        # slice the dataframe to only include the longest window of non-NaNs, using the time_step start and end
        shot_out = shot_out.loc[shot_out["time_step"].between(start_usable, end_usable)]
        all_shot_dfs.append(shot_out)

    rich.print("Float 64 columns converted before saving:", float_cols)
    
    print(
        f"Total shots: {len(all_shot_files)} of which {too_short} had less than {min_steps_filter} steps. Output total: {len(all_shot_dfs)}"
    )
    print(f"Length discrepancy: {length_discrepancy}")
    print(f"Time discrepancy: {time_discrepancy} (of shots without length discrepancy)")
    print(f"Time inconsistency: {time_inconsistency}")
    print(f"Shot number discrepancy: {shot_num_discrepancy}")
    print(f"Rejected shots due to too many NaNs: {nan_rejects}")
    rich.print(f"Memory usage: {memory / 1e6} MB")
    if nan_summaries:
        nan_summary_df = pd.concat(nan_summaries)
        rich.print(nan_summary_df.sort_index())
        rich.print(nan_summary_df.index.value_counts())
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


def generate_report(data: str | pd.DataFrame):
    """Profile a dataframe from a saved parquet file or a freshly made dataframe directly.

    When using a dataframe, the name of the dataset is assumed to be DATA_SET_NAME.
    """
    if isinstance(data, str):
        name = data.split("/")[-1]
        data_df = pd.read_parquet(data)
    else:
        name = DATA_SET_NAME
        data_df = data
    profile = ProfileReport(data_df, title="Profiling Report " + name, explorative=True)
    profile.to_file(name + ".html")
    print("Saved report to ", name + ".html")


#%% CHeck multiple data sets and compare them
def compare_data_sets(input_dirs: list[str] = [], shot_lists: dict[str, list[int]]|None = None):
    """Compare multiple datasets and check for duplicates."""
    shot_sets = {} if shot_lists is None else shot_lists.copy()
    for data_dir in input_dirs:
        data_name = data_dir.split("/")[-1]
        sig_all, label_all = index_shot_names(data_dir)
        print(data_name)
        print(" - Duplicate shots: ", len(label_all) - len(set(label_all.keys())))
        shot_sets[data_name] = list(label_all.keys())
    total_set = set()
    for name, shot_set in shot_sets.items():
        new_elements = set(shot_set) - total_set
        total_set |= set(shot_set)
        print(f"Total shots in {name}: {len(shot_set)} (+ adding {len(new_elements)} new shots to the total.)")
    print("Total unique shot set: ", len(total_set))
    # Compare the sets
    for i, (name, shot_set) in enumerate(shot_sets.items()):
        for name2, shot_set2 in list(shot_sets.items())[i + 1:]:
            print(f"Comparing {name} and {name2}")
            print("Lengths of sets: ", len(shot_set), len(shot_set2))
            left_unique = set(shot_set) - set(shot_set2)
            right_unique = set(shot_set2) - set(shot_set)
            print(" - Only in", name, "but not in", name2, "", len(left_unique), left_unique)
            print(" - Only in", name2, "but not in", name, "", len(right_unique), right_unique)
            print(" - In both sets: ", len(set(shot_set) & set(shot_set2)))

    # data_df = combine_all_shots(sig_all, label_all)
    # generate_report(data=data_df)
    # result = {}


#%% Run!

if __name__ == "__main__":
    # sig_all, label_all = index_shot_names(DATA_INPUT_DIR)
    data_df = combine_public_dataset(DATA_INPUT_DIR)
    out_path = f"./data/{DATE}-{DATA_SET_NAME}.parquet"
    data_df.to_parquet(out_path)
    # generate_report(data=data_df)
    print("Saved to ", out_path)

    #%%
    exit(0)
#%%
generate_report("data/" + DATE + DATA_SET_NAME + ".parquet")

#%%
data_set_dirs = [
    "/Users/milan/Code/fusion/data/LHD_labeled_TCV", "/Users/milan/Code/fusion/experiments/data/shots"
    # "/Users/milan/Downloads/tcv_data/ffelici",
    # "/Users/milan/Downloads/tcv_data/maurizio",
    # "/Users/milan/Downloads/tcv_data/labit",
]
compare_data_sets(data_set_dirs)

# %% Notebook style testing and profiling
# generate_report("data/2024_04_23-all_preprocessed.parquet")
# %%
# data_df = pd.read_parquet("data/2024_04_23-all_preprocessed.parquet")

#%%
import json

with open("./giants/LDH_demo/train.txt", 'r') as f:
    train_shots = json.load(f)
with open('./giants/LDH_demo/test.txt', 'r') as f:
    test_shots = json.load(f)

# %%
compare_data_sets(
    ["/Users/milan/Code/fusion/data/LHD_labeled_TCV"], shot_lists={
        "train": train_shots,
        "test": test_shots
    }
)

# %%
compare_data_sets(
    ["/Users/milan/Code/fusion/data/LHD_labeled_TCV"], shot_lists={
        "all_yoeri": train_shots + test_shots,
    }
)

# %%
