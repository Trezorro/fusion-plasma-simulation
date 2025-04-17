import plotly.graph_objects as go
import wandb
from PIL import Image


import io
from typing import Literal


def as_wandb_image(
    fig: go.Figure, format: Literal['png', 'webp', 'jpg', 'gif'] = "png", show=False
) -> wandb.Image:
    """
    Converts a Plotly figure into a Weights & Biases (wandb) Image object.

    Args:
        fig (go.Figure): The Plotly figure to be converted.
        format (Literal['png', 'webp', 'jpg', 'gif'], optional): 
            The format of the image to be generated. Defaults to "png".
        show (bool, optional): If True, displays the image using the default image viewer. Defaults to False.

    Usage:
        wandb_image = as_wandb_image(fig, format="png", show=wandb.run.disabled)
        return wandb_image

    Returns:
        wandb.Image: The converted image wrapped in a wandb.Image object.
    """
    image_bytes = fig.to_image(format=format, engine='auto')
    pil_image = Image.open(io.BytesIO(image_bytes))
    if show:
        pil_image.show()
    return wandb.Image(pil_image)
