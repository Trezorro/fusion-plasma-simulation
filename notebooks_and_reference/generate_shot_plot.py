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
# print(sig.columns.tolist())  #[0:40]
sig[cols_control + cols_data] = (sig[cols_control + cols_data] -
                                 sig[cols_control + cols_data].mean()) / sig[cols_control + cols_data].std()
shot_out = sig[ALL_SIG_COLLS].reset_index(names='time_step')

assert all(
    [x in sig.columns for x in ALL_SIG_COLLS]
), f"Missing columns {set(ALL_SIG_COLLS) - set(sig.columns)}"
shot_out
# %% change all numbered columns to a general name
variables_projection = sig.columns
# replace any numbers with x
variables_projection = set(re.sub(r'\d+', 'x', x) for x in variables_projection)

#%% Add rolling mean (left and right) to a column


def add_rolling_mean(df: pd.DataFrame, cols: list, window: int = 5):
    for col in cols:
        original_col_index = df.columns.get_loc(col)
        df.insert(
            original_col_index, f"{col}_rolling",
            df[col].rolling(window=window, min_periods=window, center=True).mean()
        )


# %%
c_df = sig[cols_meta + cols_control]
x_df = sig[cols_meta + cols_data]
add_rolling_mean(x_df, ['DML'], window=10)

x_plot_df = x_df.melt(
    # value_vars=['Predicted', 'Target'],
    # var_name='is_prediction',
    # value_name='amplitude',
    id_vars=['ShotNum', 'time'],
).sort_values(by=['ShotNum', 'time'])
c_plot_df = c_df.melt(id_vars=['ShotNum', 'time'],).sort_values(by=['ShotNum', 'time'])


#%%
def plot_signal_and_spectrum(df_stacked_time, df_freq, title, cutoff_t, subtitle="", buttons=True):
    if not (df_stacked_time is None or df_stacked_time.empty or df_freq is None or df_freq.empty):
        double_plot = True
        # Create subplots
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Observables", "Controls"), vertical_spacing=0.1)
    else:
        double_plot = False
        fig = go.Figure()

    if df_stacked_time is not None and not df_stacked_time.empty:
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
            # trace.name = f"{is_predicted} {variable} time domain"
            # trace.legendgroup = shot + is_predicted + variable
            # trace.legendgrouptitle = {'text': f"Shot {shot}: {is_predicted} {variable} signal"}
            if double_plot:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace)

    if df_freq is not None and not df_freq.empty:
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
            if double_plot:
                fig.add_trace(trace, row=2, col=1)
            else:
                fig.add_trace(trace)
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
    if double_plot:
        fig.update_yaxes(
            type="linear", row=2, col=1
        )  # Set y-axis to log scale for the frequency spectrum plot

    if not buttons:
        return fig
    # Add buttons
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


plot_signal_and_spectrum(x_plot_df, None, f"Shot #{shotno}", 0.6, "Observables",
                         buttons=False).write_html('output/first_figure.html', auto_open=True)

# %%
plot_signal_and_spectrum(x_plot_df, c_plot_df, f"Shot #{shotno}", 0.6, "Observables and Controls").show()
# %%
# %%
