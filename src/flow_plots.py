# @title Utility code: styles, functions, generators, visualization
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import torch

from src.config import get_current_config

# for accessibility: Wong's color pallette: cf. https://davidmathlogic.com/colorblind
#wong_black = [0/255, 0/255, 0/255]          # #000000
wong_amber = [230 / 255, 159 / 255, 0 / 255]  # #E69F00
wong_cyan = [86 / 255, 180 / 255, 233 / 255]  # #56B4E9
wong_green = [0 / 255, 158 / 255, 115 / 255]  # #009E73
wong_yellow = [240 / 255, 228 / 255, 66 / 255]  # #F0E442
wong_navy = [0 / 255, 114 / 255, 178 / 255]  # #0072B2
wong_red = [213 / 255, 94 / 255, 0 / 255]  # #D55E00
wong_pink = [204 / 255, 121 / 255, 167 / 255]  # #CC79A7
wong_cmap = [wong_amber, wong_cyan, wong_green, wong_yellow, wong_navy, wong_red, wong_pink]

source_color = wong_navy
target_color = wong_red
pred_color = wong_green
line_color = wong_yellow
bg_theme = 'dark'  #  'black', 'white', 'dark', 'light'
if bg_theme in ['black', 'dark']:
    plt.style.use('dark_background')
else:
    plt.rcdefaults()


def plot_distributions(dist1, dist2, title1="Distribution 1", title2="Distribution 2", alpha=0.8):
    """Plot two distributions side by side

    By https://drscotthawley.github.io/blog/posts/FlowModels.html 
    """
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    ax1.scatter(dist1[:, 0], dist1[:, 1], alpha=alpha, s=10, color=source_color)
    ax2.scatter(dist2[:, 0], dist2[:, 1], alpha=alpha, s=10, color=target_color)

    ax1.set_title(title1)
    ax2.set_title(title2)

    # Set same scale for both plots
    max_range = max(abs(dist1).max().item(), abs(dist2).max().item())
    for ax in [ax1, ax2]:
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show(block=True)  # Explicitly show the plot
    plt.close()


def plot_flow(
    module,  # Trained lignting module to generate new samples
    batch,  # Initial points, shape: 2x[batch_size, num_features]
    title_base="",
    size=20,  # Size of scatter plot points
    alpha=0.5,  # Transparency of scatter plot points
    n_steps=100,  # Number of integration steps
    warp_fn=None,  # Optional function to warp time steps
):
    """Call the integrator to calculate the motion (probability path) given v field, generate new samples
       and visualize the results.

    Args:
        val_points (torch.Tensor): Initial points, shape: [batch_size, num_features].
        target_samples (torch.Tensor): Target samples, shape: [batch_size, num_features].
        trained_model (torch.nn.Module): Trained model to generate new samples.
        size (int, optional): Size of scatter plot points. Defaults to 20.
        alpha (float, optional): Transparency of scatter plot points. Defaults to 0.5.
        n_steps (int, optional): Number of integration steps. Defaults to 100.
        warp_fn (callable, optional): Optional function to warp time steps. Defaults to None.

    Returns:
        None
    """
    # Generate and visualize new samples
    device = module.device
    source_samples, target_samples = batch

    generated_samples, trajectories = module.integrate_path(
        source_samples.to(device), n_steps=n_steps, warp_fn=warp_fn, save_trajectories=True
    )

    n_viz = min(30, len(trajectories[0]))  # Number of trajectories to visualize

    fig, ax = plt.subplots(1, 4, figsize=(13, 3))
    plt.suptitle(title_base, fontsize=16, y=1.05)
    data_list = [source_samples.cpu(), generated_samples.cpu(), target_samples.cpu()]
    label_list = ['Initial Points', 'Generated Samples', 'Target Data', 'Trajectories']
    color_list = [source_color, pred_color, target_color]
    global_max = max(
        torch.max(torch.abs(torch.cat(data_list)), 0)[0][0],
        torch.max(torch.abs(torch.cat(data_list)), 0)[0][1]
    )
    for i in range(len(label_list)):
        ax[i].set_title(label_list[i])
        ax[i].set_xlim([-global_max, global_max])
        ax[i].set_ylim([-global_max, global_max])
        if i < 3:  # non-trajectory plots
            ax[i].scatter(
                data_list[i][:, 0],
                data_list[i][:, 1],
                s=size,
                alpha=alpha,
                label=label_list[i],
                color=color_list[i]
            )
        else:
            # Plot trajectory paths first
            for j in range(n_viz):
                path = trajectories[:, j]  # Shape: [n_steps, num_features]
                ax[3].plot(path[:, 0], path[:, 1], '-', color=line_color, alpha=1, linewidth=1)

            # Then plot start and end points for the SAME trajectories
            start_points = trajectories[0, :n_viz]  # Shape: [n_viz, num_features]
            end_points = trajectories[-1, :n_viz]  # Shape: [n_viz, num_features]
            ax[3].scatter(
                start_points[:, 0],
                start_points[:, 1],
                color=source_color,
                s=size,
                alpha=1,
                label='Source Points'
            )
            ax[3].scatter(
                end_points[:, 0],
                end_points[:, 1],
                color=pred_color,
                s=size,
                alpha=1,
                label='Current Endpoints'
            )
            ax[3].legend()

    plt.show()
    plt.close()
    return fig
