# @title Utility code: styles, functions, generators, visualization
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
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
