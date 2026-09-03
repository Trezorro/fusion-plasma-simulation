import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objects as go
import torch
import wandb
from plotly import colors as plt_colors

from src.config import get_current_config

def animated_trajectory_plotly(
    meta: dict[str, torch.Tensor],
    trajectories: torch.Tensor,
    n=5,
    title_base="",
    subtitle="",  # Subtitle for the plot
    **kwargs
):
    """Create an animated line plot showing trajectories over time.

    Args:
        meta (dict): Dictionary containing metadata about the samples.
        trajectories (torch.Tensor): Trajectories, shape: [n_steps, num_samples, num_channels, num_timepoints].
        n (int, optional): Number of traces to visualize. Defaults to 5.
        title_base (str, optional): Base title for the plot. Defaults to "".
        subtitle (str, optional): Subtitle for the plot. Defaults to "".
    """
    C = get_current_config()
    n_steps, num_samples, n_channels, num_timepoints = trajectories.size()
    CHANNEL_NAMES = C.data.cols.x
    n_channels = len(CHANNEL_NAMES)
    num_samples = min(n, num_samples)  # Limit the number of traces to visualize
    COLOR_SCALE = plt_colors.qualitative.Plotly

    # Initialize the figure with the first frame's data
    fig = go.Figure()

    for sample_i in range(num_samples):
        for channel_i in range(n_channels):
            channel_color = COLOR_SCALE[((n_channels * sample_i) + channel_i) % len(COLOR_SCALE)]
            channel_name = CHANNEL_NAMES[channel_i]

            # Add the initial trace for each sample and channel
            fig.add_trace(
                go.Scatter(
                    x=np.arange(num_timepoints),
                    y=trajectories[0, sample_i, channel_i, :].numpy(),
                    mode='lines',
                    line=dict(color=channel_color, width=2),
                    name=f'{channel_name} (Sample {sample_i + 1})',
                    legendgroup=f'Sample {sample_i + 1} - {channel_name}',
                )
            )

    # Create frames for the animation
    frames = []
    for step in range(n_steps):
        frame_data = []
        for sample_i in range(num_samples):
            for channel_i in range(n_channels):
                frame_data.append(
                    go.Scatter(
                        y=trajectories[step, sample_i, channel_i, :].numpy(),
                        # x=np.arange(num_timepoints),
                    )
                )
        frames.append(
            go.Frame(data=frame_data, traces=list(range(num_samples * n_channels)), name=f"Step {step}")
        )

    # Add play/pause buttons and slider
    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(frame=dict(duration=500, redraw=False), fromcurrent=True)],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                ),
            ],
        )
    ]

    sliders = [
        dict(
            steps=[
                dict(
                    method="animate",
                    args=[[f"Step {step}"],
                          dict(mode="immediate", frame=dict(duration=500, redraw=False))],
                    label=f"{step + 1}",
                ) for step in range(n_steps)
            ],
            active=0,
            transition=dict(duration=100),
            x=0,
            y=0,
            currentvalue=dict(font=dict(size=12), prefix="Step: ", visible=True, xanchor="left"),
            len=.8,
        )
    ]

    # Update layout
    fig.update_layout(
        title=title_base + (f"<br><sub>{subtitle}</sub>" if subtitle else ""),
        xaxis=dict(title="Timepoints", range=[0, num_timepoints - 1]),
        yaxis=dict(title="Value", range=[-1.1, 1.1]),
        updatemenus=updatemenus,
        sliders=sliders,
    )

    # Add frames to the figure
    fig.update(frames=frames)

    if wandb.run.disabled:  # type: ignore
        fig.show()

    return wandb.Html(plotly.io.to_html(fig))


def animated_window_set_plotly(
    meta: dict[str, torch.Tensor],
    trajectories: torch.Tensor,
    target_samples: torch.Tensor,
    generated_samples: torch.Tensor,
    conditioning_input: dict[str, torch.Tensor],
    repeat=4,
    n=None,
    title_base="",
    subtitle="",
    **kwargs  # catch-all for the other keys spread from model.evaluate output
):
    """Animate the integration flow for a curated window_set, with context overlays.

    Keeps the flow animation of animated_trajectory_plotly (the animation axis is the
    integration step) but adds the context features of multi_channel_lines_plotly:
    static ground-truth and history overlays, and shot/signal dropdown filters over the
    curated windows from config.window_set.

    Only the generated (predicted) traces animate, sweeping across trajectories[step]
    from prior noise to the final sample. History and target traces are static overlays
    that never appear in any frame, so they stay fixed while the animation plays.

    Each window w occupies collated rows [w*repeat : (w+1)*repeat] (see
    FusionShotDataset.window_set_batch); the `repeat` copies share conditioning but have
    different stochastic priors, giving the sample diversity we animate.

    Args:
        meta (dict): Metadata; shot_number and start are read per window.
        trajectories (torch.Tensor): [n_steps, num_samples, num_channels, num_timepoints].
        target_samples (torch.Tensor): [num_samples, num_channels, num_timepoints].
        generated_samples (torch.Tensor): Final samples (unused here; trajectories[-1]
            carries the same information). Accepted so **output spreads cleanly.
        conditioning_input (dict): Contains 'x_history' for the history overlay.
        repeat (int): Stochastic samples per window (rows per window in the batch).
        n: Accepted for signature parity with the other plotters; unused (all windows shown).
        title_base (str): Base title for the plot.
        subtitle (str): Subtitle for the plot.

    Isolation: every trace is its own legend entry with a unique name and no legendgroup,
    so a legend click toggles exactly one trace. The shot/signal dropdowns coarse-filter;
    the legend composes on top, letting the user drill down to a single (channel, sample,
    window) trace. The samples toggle uses restyle so it only touches the extra-sample
    traces and does not clobber the active dropdown selection.
    """
    C = get_current_config()
    CHANNEL_NAMES = C.data.cols.x
    n_channels = len(CHANNEL_NAMES)
    history_length = C.data.history_length
    seq_length = C.data.seq_length
    COLOR_SCALE = plt_colors.qualitative.Plotly

    n_steps = trajectories.size(0)
    num_samples = trajectories.size(1)
    num_windows = num_samples // repeat
    x_history = conditioning_input['x_history']
    shot_numbers = meta['shot_number']
    start_times = meta['start']

    fig = go.Figure()

    # Bookkeeping so filters and frames can address traces without relying on legendgroup.
    window_trace_indices: dict[int, list[int]] = {w: [] for w in range(num_windows)}
    channel_trace_indices: dict[int, list[int]] = {ch: [] for ch in range(n_channels)}
    animated_indices: list[int] = []   # global indices of the generated (animated) traces
    animated_rows: list[tuple[int, int]] = []  # parallel (batch_row, channel_i) for frame data
    extra_sample_indices: list[int] = []  # generated traces with s >= 1 (the samples toggle target)
    window_labels: list[str] = []

    def _register(idx, w, channel_i):
        window_trace_indices[w].append(idx)
        channel_trace_indices[channel_i].append(idx)

    for w in range(num_windows):
        base = w * repeat  # first collated row for this window
        shot_sample_id = f"{int(shot_numbers[base])}:{float(start_times[base]):.2f}s ({w})"
        window_labels.append(shot_sample_id)
        visible = (w == 0)  # default view: only the first window, keeps legend readable

        for channel_i in range(n_channels):
            channel_color = COLOR_SCALE[((n_channels * w) + channel_i) % len(COLOR_SCALE)]
            channel_name = CHANNEL_NAMES[channel_i]

            # History (static overlay, left of x=0)
            fig.add_trace(
                go.Scatter(
                    x=np.arange(-history_length, 0),
                    y=x_history[base, channel_i, :].numpy(),
                    mode='lines',
                    line=dict(color=channel_color, width=3),
                    opacity=0.8,
                    name=f'{channel_name} history [{shot_sample_id}]',
                    visible=visible,
                )
            )
            _register(len(fig.data) - 1, w, channel_i)

            # Target / ground truth (static overlay)
            fig.add_trace(
                go.Scatter(
                    x=np.arange(seq_length),
                    y=target_samples[base, channel_i, :].numpy(),
                    mode='lines',
                    line=dict(color=channel_color, width=3),
                    opacity=0.6,
                    name=f'{channel_name} target [{shot_sample_id}]',
                    visible=visible,
                )
            )
            _register(len(fig.data) - 1, w, channel_i)

            # Generated (animated): one dotted trace per stochastic sample
            for s in range(repeat):
                row = base + s
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(seq_length),
                        y=trajectories[0, row, channel_i, :].numpy(),  # the prior (step 0)
                        mode='lines',
                        line=dict(dash='dot', color=channel_color),
                        opacity=0.9,
                        name=f'{channel_name} pred s{s} [{shot_sample_id}]',
                        visible=visible,
                    )
                )
                idx = len(fig.data) - 1
                _register(idx, w, channel_i)
                animated_indices.append(idx)
                animated_rows.append((row, channel_i))
                if s >= 1:
                    extra_sample_indices.append(idx)

    n_traces = len(fig.data)

    # Static yellow history separator + shaded region (copied from flow_plots multi_channel_lines)
    fig.add_shape(
        type="line",
        x0=-0.5, x1=-0.5, y0=0, y1=1,
        line=dict(color="yellow", width=3, dash="solid"),
        xref="x", yref="paper", opacity=0.5,
    )
    fig.add_shape(
        type="rect",
        x0=-1, x1=0, y0=0, y1=1,
        fillcolor="yellow", opacity=0.3, line_width=0,
        xref="x", yref="paper",
    )

    # Frames: only the animated generated traces update; static overlays are never listed.
    frames = []
    for step in range(n_steps):
        frame_data = [
            go.Scatter(y=trajectories[step, row, channel_i, :].numpy())
            for (row, channel_i) in animated_rows
        ]
        frames.append(go.Frame(data=frame_data, traces=animated_indices, name=f"Step {step}"))

    # Play/Pause + slider (reused from animated_trajectory_plotly)
    play_pause = dict(
        type="buttons",
        showactive=False,
        x=0.0, xanchor="left", y=1.12, yanchor="top",
        direction="left",
        buttons=[
            dict(
                label="Play",
                method="animate",
                args=[None, dict(frame=dict(duration=500, redraw=False), fromcurrent=True)],
            ),
            dict(
                label="Pause",
                method="animate",
                args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
            ),
        ],
    )
    sliders = [
        dict(
            steps=[
                dict(
                    method="animate",
                    args=[[f"Step {step}"], dict(mode="immediate", frame=dict(duration=500, redraw=False))],
                    label=f"{step + 1}",
                ) for step in range(n_steps)
            ],
            active=0,
            transition=dict(duration=100),
            x=0, y=0,
            currentvalue=dict(font=dict(size=12), prefix="Step: ", visible=True, xanchor="left"),
            len=.8,
        )
    ]

    # Dropdown filters, driven by the Python-side index sets (not legendgroup strings).
    shot_buttons = [dict(label='All windows', method='update', args=[{'visible': [True] * n_traces}])]
    for w in range(num_windows):
        vis_set = set(window_trace_indices[w])
        shot_buttons.append(
            dict(
                label=window_labels[w],
                method='update',
                args=[{'visible': [i in vis_set for i in range(n_traces)]}],
            )
        )
    signal_buttons = [dict(label='All signals', method='update', args=[{'visible': [True] * n_traces}])]
    for channel_i, channel in enumerate(CHANNEL_NAMES):
        vis_set = set(channel_trace_indices[channel_i])
        signal_buttons.append(
            dict(
                label=channel,
                method='update',
                args=[{'visible': [i in vis_set for i in range(n_traces)]}],
            )
        )
    # Samples toggle: restyle only the extra-sample traces so it composes with the dropdowns.
    samples_buttons = [
        dict(label=f'{repeat} samples', method='restyle', args=[{'visible': True}, extra_sample_indices]),
        dict(label='1 sample', method='restyle', args=[{'visible': False}, extra_sample_indices]),
    ]

    def _dropdown(buttons, y):
        return dict(
            buttons=buttons, showactive=True, direction="down",
            x=1.02, xanchor="left", y=y, yanchor="top",
        )

    fig.update_layout(
        title=title_base + (f"<br><sub>{subtitle}</sub>" if subtitle else ""),
        template='ggplot2',
        xaxis=dict(title="Time steps (0.1ms/step)", range=[-history_length, seq_length]),
        yaxis=dict(title="Value", range=[-1.1, 1.1]),
        updatemenus=[
            play_pause,
            _dropdown(shot_buttons, 1.0),
            _dropdown(signal_buttons, 0.72),
            _dropdown(samples_buttons, 0.44),
        ],
        sliders=sliders,
    )
    fig.update(frames=frames)

    if wandb.run.disabled:  # type: ignore
        fig.show()

    return fig


def animated_trajectory_matplotlib(
    meta: dict[str, torch.Tensor],
    trajectories: torch.Tensor,
    n=5,
    title_base="",
    subtitle="",  # Subtitle for the plot
    **kwargs
):
    """Create an animated line plot showing trajectories over time using Matplotlib.

    Args:
        meta (dict): Dictionary containing metadata about the samples.
        trajectories (torch.Tensor): Trajectories, shape: [n_steps, num_samples, num_channels, num_timepoints].
        n (int, optional): Number of traces to visualize. Defaults to 5.
        title_base (str, optional): Base title for the plot. Defaults to "".
        subtitle (str, optional): Subtitle for the plot. Defaults to "".
    """
    C = get_current_config()
    n_steps, num_samples, n_channels, num_timepoints = trajectories.size()
    CHANNEL_NAMES = C.data.cols.x
    num_samples = min(n, num_samples)  # Limit the number of traces to visualize
    COLOR_SCALE = plt.cm.tab10.colors  # Use a colormap for colors

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, num_timepoints - 1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f"{title_base}\n{subtitle}", fontsize=14)
    ax.set_xlabel("Timepoints")
    ax.set_ylabel("Value")

    # Create line objects for each sample and channel
    lines = []
    for sample_i in range(num_samples):
        for channel_i in range(n_channels):
            line, = ax.plot(
                [], [],
                label=f"{CHANNEL_NAMES[channel_i]} (Sample {sample_i + 1})",
                color=COLOR_SCALE[(n_channels * sample_i + channel_i) % len(COLOR_SCALE)]
            )
            lines.append(line)

    ax.legend(loc="upper right")

    # Initialization function for the animation
    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    # Update function for each frame
    def update(frame):
        for sample_i in range(num_samples):
            for channel_i in range(n_channels):
                idx = sample_i * n_channels + channel_i
                lines[idx].set_data(
                    np.arange(num_timepoints), trajectories[frame, sample_i, channel_i, :].numpy()
                )
        return lines

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=n_steps, init_func=init, blit=True, interval=500)

    # Show the animation
    plt.show()
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('ani.mp4', writer=writer)
    plt.savefig()
    return ani