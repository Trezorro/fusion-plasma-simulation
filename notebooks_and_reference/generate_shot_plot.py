# %%
import pandas as pd
from pathlib import Path
import glob
import numpy as np
import re
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.tools import mpl_to_plotly
import datapane as dp

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SHOT_N = 1
# %%
# data_dir = 'shots/'
data_dir = '../data/LHD_labeled_TCV/'
cols_meta = [
    "ShotNum",
    "time",
]
cols_control = [
    "IP",  # Current (niet reference lijn voor controller, maar de ware input. Dan laat je control bij control)
    "gas_fringes",  # Ingepompte gas
    "NBI",  # manieren om te verhitten: colliding Neutral beam injection
    "ECRH",  # magnetron.
    "a_minor",  # reel gemeten plasma shape a k d (horizontale radius
    "KAPPA",
    "DELTA"  # inkerbovenhoek nar links vanuit hetmidden
]
cols_data = [
    # "FIR",  # density lijn Interferometer
    "FIR_core",  # For the March dataset of 260 shots, the FIR_core signal is the same as FIR.
    "PD",  # photodiode lijn op de divertor
    "DML",  # Magnetische respons  correleert met de energie in het plasma
    "POHM",  # Gemeten power waarde meet de power die uit wrijving komt
    "Z_axis"  # center Plasma positie in de verticale lijn. deviation van reference is betekenis. 
]
cols_label = ["LHD_label"]
ALL_SIG_COLLS = cols_meta + cols_control + cols_data


# %%
def get_shot_index(data_dir: str) -> tuple[dict[int, str], dict[int, str]]:
    sig_all_names = glob.glob(data_dir + 'TCV_DATA*.parquet')
    # use regex to get sample number:
    sig_all = {int(re.findall(r'\d+', x)[0]): x for x in sig_all_names}
    shot_no_list = list(sig_all.keys())
    label_all = glob.glob(data_dir + 'TCV_*_apau_labeled.csv')
    label_all = {int(x.split("TCV_")[1].split("_apau_labeled.csv")[0]): x for x in label_all}
    label_no_list = list(label_all.keys())
    assert set(shot_no_list
              ) <= set(label_no_list), f"Not all shots have labels: {set(shot_no_list) - set(label_no_list)}"
    print(f"All shots: {shot_no_list}")
    print("Amount of shots: ", len(shot_no_list))
    print(f"Labels: {len(label_all)}")
    return sig_all, label_all


sig_all, label_all = get_shot_index(data_dir)

#%%
# %%
# example
shot_no_list = list(sig_all.keys())
shotno = shot_no_list[SHOT_N]

sig = pd.read_parquet(sig_all[shotno])
label = pd.read_csv(label_all[shotno])
print(sig.columns.tolist())  #[0:40]
sig[cols_control + cols_data] = (sig[cols_control + cols_data] -
                                 sig[cols_control + cols_data].mean()) / sig[cols_control + cols_data].std()
shot_out = sig[ALL_SIG_COLLS].reset_index(names='time_step')

assert all(
    [x in sig.columns for x in ALL_SIG_COLLS]
), f"Missing columns {set(ALL_SIG_COLLS) - set(sig.columns)}"
shot_out
# %%
'FIR_core' in sig.columns.tolist()

# %%
c_df = sig[cols_meta + cols_control]
x_df = sig[cols_meta + cols_data]
x_plot_df = x_df.melt(
    # value_vars=['Predicted', 'Target'],
    # var_name='is_prediction',
    # value_name='amplitude',
    id_vars=['ShotNum', 'time'],
)
c_plot_df = c_df.melt(id_vars=['ShotNum', 'time'],)

# %%
# lengths_array = np.array(lengths)
# plt.hist(lengths_array, bins=50)
# plt.title("Length of signals")
# lengths_array = np.array(lengths)

# mean_length = np.mean(lengths_array)
# median_length = np.median(lengths_array)
# mode_length = np.argmax(np.bincount(lengths_array))
# std_length = np.std(lengths_array)
# q1_length = np.percentile(lengths_array, 25)
# q3_length = np.percentile(lengths_array, 75)

# print(f"Mean length: {mean_length}")
# print(f"Median length: {median_length}")
# print(f"Mode length: {mode_length}")
# print(f"Standard deviation: {std_length}")
# print(f"1st quartile: {q1_length}")
# print(f"3rd quartile: {q3_length}")

# least_k = 20
# shortest_k = np.argsort(lengths_array)[:least_k]
# print(f"Shortest {least_k} signals: {shortest_k}")

# %%

# %% [markdown]
# ```
#
# cols_label = ["LHD_label"]
#
# For the labels, 0 = no label, 1 = low confinement mode, 2 = dithering ('inbetween'), 3 = high confinement mode
#
# This is data for full shots (shot = pulse = experiment). We do not want to model the full shot, but rather an area just around a transition in the LHD_label, so zooming in on say ~100ms before/after a transition in the LHD_label
# ```

# %%
sig

# %%
# normalize data
# raw_sig = sig.copy()

# # %%
# label

# # %%
# # Plot observables
# fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
# plt.title(f"Shot #{shotno} - Observables")
# ax.set_ylabel("Observables (normalized)")
# ax.plot(
#     sig["time"],
#     sig[cols_data],
# )
# # ax.set_ylim([-3, 5.3])
# # ax.set_xlim([0.6, 0.8])

# ax2 = ax.twinx()

# # interpolate missing labels (label==0) to previous
# labelvals = np.array(label["LHD_label"])
# mask = labelvals == 0
# labelvals[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), labelvals[~mask])
# # plot filled area:
# # ax2.fill_between(label["time"], labelvals, color='red', alpha=.2, )
# # ax2.plot(label["time"], labelvals, color='red', alpha=.3, )
# # set axis limits:
# ax2.set_ylim([0.99, 3])
# ax2.set_ylabel("Confinement Mode")

# # Change y-axis tick positions
# ax2.set_yticks([1, 2, 3])
# # Change y-axis tick labels
# ax2.set_yticklabels(['L', 'D', 'H'])
# legend = ax.legend(cols_data, loc='upper left')
# p_fig = mpl_to_plotly(fig)
# # plt.show()
# # p_fig.write_html('plots/first_figure.html', auto_open=True)
# report = dp.Blocks(
#     # title="Plots",
#     blocks=[
#         dp.DataTable(sig[ALL_SIG_COLLS], label="Data"),
#         # dp.DataTable(sig.describe(), label="Summary"),
#         dp.Plot(p_fig, label="Observables"),
#         dp.Plot(fig, label="Observables (mpl)")
#     ],
# )

# dp.save_report(report, path=f'output/plots/report_{shotno}.html', open=True)

# # %%
# fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
# plt.title(f"Shot #{shotno} - control")

# ax.set_ylabel("Controls (normalized)")
# ax.set_xlabel("Time (s)")

# ax.plot(sig["time"], sig[cols_control])
# ax.set_ylim([-2, 1.5])

# ax2 = ax.twinx()

# # interpolate missing labels (label==0) to previous
# labelvals = np.array(label["LHD_label"])
# mask = labelvals == 0
# labelvals[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), labelvals[~mask])
# # plot filled area:
# # ax2.fill_between(label["time"], labelvals, color='red', alpha=.2)
# ax2.set_ylim([.99, 3])
# ax2.set_ylabel("Confinement Mode")
# # ax2.set_xlim([0.6, 0.8])
# # ax2.set_xlim([sig["time"].min(), sig["time"].max()])

# # Change y-axis tick positions
# ax2.set_yticks([1, 2, 3])
# # Change y-axis tick labels
# ax2.set_yticklabels(['L', 'D', 'H'])

# legend = ax.legend(cols_control, loc='best')

# # %%
# fig, axs = plt.subplots(10, 1, figsize=(10, 20))

# shot_keys = list(sig_all.keys())
# # plt.suptitle("FIR response with red-marked H-mode for 10 shots")

# for i, shotno in enumerate(shot_keys[0:10]):
#     sig = pd.read_parquet(sig_all[shotno])
#     label = pd.read_csv(label_all[shotno])
#     sig[ALL_SIG_COLLS] = (sig[ALL_SIG_COLLS] - sig[ALL_SIG_COLLS].mean()) / sig[ALL_SIG_COLLS].std()
#     is_dithering = label["LHD_label"] == 2

#     axs2 = axs[i].twinx()
#     axs2.fill_between(label["time"], label["LHD_label"], color='red', alpha=.3)
#     ax2.plot(
#         label.loc[is_dithering, "time"],
#         label.loc[is_dithering, "LHD_label"],
#         color='purple',
#         alpha=.3,
#     )
#     axs[i].plot(sig["time"], sig[cols_control])
#     # axs[i].set_ylabel("Observables (normalized)")
#     axs[i].set_ylabel("Controls (normalized)")

#     plt.title(f"Shot #{shotno} - Controls")
#     # plt.title(f"Shot #{shotno} - Observables")
#     # ax.set_ylim([-3, 5.3])
#     # ax.set_xlim([0.6, 0.8])

#     axs2.set_ylim([0.99, 3])
#     axs2.set_ylabel("confinement label")

#     # Set x-axis limits
#     axs[i].set_xlim([sig["time"].min(), sig["time"].max()])

#     # Add time label to x-axis
#     axs[i].set_xlabel("Time (s)")

# plt.tight_layout()
# plt.savefig('10shots.pdf')

# plt.show()

# # %%
# sig.reset_index(drop=False).melt(
#     value_vars=['Predicted', 'Target'],
#     var_name='is_prediction',
#     value_name='amplitude',
#     id_vars=['shot', 'variable', 'frequency_bin'],
# )


#%%
def plot_signal_and_spectrum(df_stacked_time, df_freq, title, cutoff_t, subtitle=""):
    # Create subplots
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Observables", "Controls"), vertical_spacing=0.1)

    # Time-domain signal plot
    time_fig = px.line(
        df_stacked_time,
        x='time',
        y='value',
        color='variable',
        symbol='variable',
        # line_dash='is_prediction',
        # line_shape='linear',
        # category_orders={'is_prediction': ["Target", "Predicted"]},
        title=f"Signal Plot: {title}"
    )
    for trace in time_fig.data:
        # shot, is_predicted, variable = trace.name.split(', ')
        # trace.name = f"{is_predicted} {variable} time domain"
        # trace.legendgroup = shot + is_predicted + variable
        # trace.legendgrouptitle = {'text': f"Shot {shot}: {is_predicted} {variable} signal"}
        fig.add_trace(trace, row=1, col=1)

    # Frequency spectrum plot
    spectrum_fig = px.line(
        df_freq,
        x='time',
        y='value',
        color='variable',
        # line_dash='is_prediction',
        # category_orders={'is_prediction': ["Target", "Predicted"]},
        # symbol='variable',
        line_shape='linear',
        markers=True,
        title="Controls"
    )
    for trace in spectrum_fig.data:
        # shot, is_predicted, variable = trace.name.split(', ')
        # trace.name = f"{is_predicted} {variable} frequency spectrum"
        # trace.legendgroup = shot + is_predicted + variable
        # trace.legendgrouptitle = {'text': f"Shot {shot}: {is_predicted} {variable} signal"}
        fig.add_trace(trace, row=2, col=1)

    # Add vertical rectangle to time-domain plot
    # fig.add_vrect(
    #     x0=-0.5,
    #     x1=cutoff_t - 0.5,
    #     opacity=0.2,
    #     line_width=0,
    #     layer="below",
    #     fillcolor="LightSalmon",
    #     row=1,
    #     col=1
    # )
    # fig.update_xaxes(range=[-0.5, C.data.seq_length], row=1, col=1)
    title += "| Signal and Controls "
    if subtitle:
        title += f"<br><sub>{subtitle}</sub>"
    fig.update_layout(title_text=title, title_automargin=True, title_y=.97)
    # if subtitle:
    #     fig.update_layout(title_subtitle=dict(text=str(subtitle)))

    fig.update_yaxes(type="linear", row=2, col=1)  # Set y-axis to log scale for the frequency spectrum plot
    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(args=[{
                        "yaxis2.type": "log"
                    }], label="Log", method="relayout"),
                    dict(args=[{
                        "yaxis2.type": "linear"
                    }], label="Linear", method="relayout")
                ],
                pad={
                    "r": 0,
                    "t": 0
                },
                showactive=True,
                x=.995,
                xanchor="right",
                y=0.44,
                yanchor="top"
            ),
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(args=[{
                        "visible": [True] * len(fig.data)
                    }], label="All", method="update"),
                    dict(
                        args=[{
                            "visible": [trace.name.startswith('Predicted') for trace in fig.data]
                        }],
                        label="Predicted",
                        method="update"
                    ),
                    dict(
                        args=[{
                            "visible": [trace.name.startswith('Target') for trace in fig.data]
                        }],
                        label="Targets",
                        method="update"
                    ),
                ],
                pad={
                    "r": 10,
                    "t": 10
                },
                showactive=True,
                x=1.005,
                xanchor="left",
                y=1.02,
                yanchor="bottom"
            ),
        ]
    )
    return fig


# %%
plot_signal_and_spectrum(x_plot_df, c_plot_df, f"Shot #{shotno}", 0.6, "Observables and Controls").show()
# %%
