import torch
import numpy as np


# A few different data distributions
def create_gaussian_data(n_points=1000, scale=1.0):
    """Create a 2D Gaussian distribution"""
    return torch.randn(n_points, 2) * scale


def create_square_data(n_points=1000, scale=3.0):  # 3 is set by the spread of the gaussian and spiral
    """Create points uniformly distributed in a square"""
    # Generate uniform points in a square
    points = (torch.rand(n_points, 2) * 2 - 1) * scale
    return points


def create_spiral_data(n_points=1000, scale=1):
    """Create a spiral distribution. i like this one more"""
    noise = 0.1 * scale
    #theta = torch.linspace(0, 6*np.pi, n_points) # preferred order? no way
    theta = 6 * np.pi * torch.rand(n_points)
    r = theta / (2 * np.pi) * scale
    x = r * torch.cos(theta) + noise * torch.randn(n_points)
    y = r * torch.sin(theta) + noise * torch.randn(n_points)
    return torch.stack([x, y], dim=1)


def create_heart_data(n_points=1000, scale=3.0):
    """Create a heart-shaped distribution of points"""
    square_points = create_square_data(n_points, scale=1.0)

    # Calculate the heart-shaped condition for each point
    x, y = square_points[:, 0], square_points[:, 1]
    heart_condition = x**2 + ((5 * (y + 0.25) / 4) - torch.sqrt(torch.abs(x)))**2 <= 1

    # Filter out points that don't satisfy the heart-shaped condition
    heart_points = square_points[heart_condition]

    # If we don't have enough points, generate more
    while len(heart_points) < n_points:
        new_points = create_square_data(n_points - len(heart_points), scale=1)
        x, y = new_points[:, 0], new_points[:, 1]
        new_heart_condition = x**2 + ((5 * (y + 0.25) / 4) - torch.sqrt(torch.abs(x)))**2 <= 1
        new_heart_points = new_points[new_heart_condition]
        heart_points = torch.cat([heart_points, new_heart_points], dim=0)

    heart_points *= scale
    return heart_points[:n_points]


def create_two_gaussians_data(n_points=1000, scale=1.0, shift=2.5):
    """Create a 2D Gaussian distribution"""
    g = torch.randn(n_points, 2) * scale
    g[:n_points // 2, 0] -= shift
    g[n_points // 2:, 0] += shift
    indices = torch.randperm(n_points)
    return g[indices]


def create_smiley_data(n_points=1000, scale=2.5):
    "make a smiley face"
    points = []
    # Face circle
    #angles = 2 * np.pi * torch.rand(n_points//2+20)
    #r = scale + (scale/10)*torch.sqrt(torch.rand(n_points//2+20))
    #points.append(torch.stack([r * torch.cos(angles), r * torch.sin(angles)], dim=1))

    # Eyes (small circles at fixed positions)
    for eye_pos in [[-1, 0.9], [1, 0.9]]:
        eye = torch.randn(n_points // 3 + 20, 2) * 0.2 + torch.tensor(eye_pos) * scale * 0.4
        points.append(eye)

    # Smile (arc in polar coordinates)
    theta = -np.pi / 6 - 2 * np.pi / 3 * torch.rand(n_points // 3 + 20)
    r_smile = scale * 0.6 + (scale / 4) * torch.rand_like(theta)
    points.append(torch.stack([r_smile * torch.cos(theta), r_smile * torch.sin(theta)], dim=1))

    points = torch.cat(points, dim=0)  # concatenate first
    points = points[torch.randperm(points.shape[0])]  # then shuffle
    return points[:n_points, :]
