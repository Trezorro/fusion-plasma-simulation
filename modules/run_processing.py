# %%
import pandas as pd
import glob
import numpy as np
import re

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


sig_all, label_all = get_shot_index(data_dir)

#%% Combine all shots into one dataframe


def combine_all_shots(sig_all: dict[int, str],
                      label_all: dict[int, str],
                      min_steps_filter: int = 5000):
    # init counters for discrepancies
    time_discrepancy, shot_num_discrepancy, label_shot_num_discrepancy, length_discrepancy, too_short = 0, 0, 0, 0, 0
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

        # assertions for consistency
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

        # not too interesting, but good to check: shot number column consistency
        if sig["ShotNum"].iloc[0] != shotno:
            print(f"Shot number does not match for shot {shotno}")
            shot_num_discrepancy += 1
        # extract columns from signal
        shot_out = sig[ALL_SIG_COLLS].reset_index(
            names='time_step').set_index("time")
        # resample labels with ffill to time steps of signal
        label = label.set_index("time")
        label = label.reindex(shot_out.index, method='nearest', tolerance=0.01)
        # add labels
        shot_out.join(label[COLS_LABEL], on="time")

        all_shot_dfs.append(shot_out)
        # print all_data size in memory
        memory += shot_out.memory_usage().sum()
        print(f"Memory usage: {memory / 1e6} MB")
    print(
        f"Total shots: {len(sig_all)} of which {too_short} had less than {min_steps_filter} steps. Output total: {len(all_shot_dfs)}"
    )
    print(f"Length discrepancy: {length_discrepancy}")
    print(f"Time discrepancy: {time_discrepancy}")
    print(f"Shot number discrepancy: {shot_num_discrepancy}")
    print(f"Label shot number discrepancy: {label_shot_num_discrepancy}")
    return pd.concat(all_shot_dfs)


data_df = combine_all_shots(sig_all, label_all)
date = pd.Timestamp.now().strftime("%Y_%m_%d")
data_df.to_parquet(f"./data/{date}-all_preprocessed.parquet")

