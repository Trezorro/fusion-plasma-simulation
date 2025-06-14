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