#%% Imports
import numpy as np
import pandas as pd

print("Importing torch")
import torch

print("Importing torchmetrics")
import torchmetrics

print("Importing fourier")
from src.fourier import FourierMSLE
from rich.progress import Progress
import plotly.graph_objects as go

#%% Parameters
SAMPLE_RATE = 10000
PHASE_SHIFT = 0.5
DURATION = 0.01

BASE_METRIC = torchmetrics.MeanSquaredError()
COMPARE_METRIC = FourierMSLE()


#%% Functions
def generate_timepoints(duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return t


def get_sine_wave(t, frequency, phase=0.0):
    """Generate a sine wave with the given frequency and duration."""
    x = np.sin(2 * np.pi * (frequency * t + phase))
    return torch.tensor(x)


def get_gaussian_trace(t, mean, std):
    x = np.random.normal(mean, std, len(t))
    return torch.tensor(x)


def get_mixtures(t, means, stds, weights=None):
    if weights is None:
        weights = np.ones(len(means)) / len(means)  # Equal weights if not provided

    # Ensure weights sum to 1
    weights = np.array(weights)
    weights /= weights.sum()

    # Choose a Gaussian component for each sample
    components = np.random.choice(len(means), size=len(t), p=weights)

    # Generate samples from the chosen components
    x = np.array([np.random.normal(means[comp], stds[comp]) for comp in components])

    return torch.tensor(x)


def plot_signals(a: torch.Tensor, b: torch.Tensor, t: np.ndarray):
    abs_difference = torch.abs(a - b)
    cumulative_difference = torch.cumsum(abs_difference, dim=0)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=cumulative_difference,
            fill='tozeroy',
            mode="none",
            name="Cumulative Difference",
            yaxis="y2"
        )
    )
    fig.add_trace(go.Scatter(x=t, y=abs_difference, mode="lines", name="Absolute Difference", fill='tozeroy'))
    fig.add_trace(go.Scatter(x=t, y=a, mode="lines", name="Sine A"))
    fig.add_trace(go.Scatter(x=t, y=b, mode="lines", name="Sine B"))
    total_difference = torch.sum(abs_difference)
    error_a = BASE_METRIC(a, b)
    error_b = COMPARE_METRIC(a, b)
    fig.update_layout(
        title=f"Sine waves (difference sum {total_difference}, mse {error_a})",
        xaxis_title="Time",
        yaxis_title="Amplitude",
        yaxis2=dict(
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=True,
            showticklabels=True,
        )
    )
    print(f"Total difference: {total_difference}, MSE: {error_a}, Fourier MSE: {error_b}")
    fig.show()


#%% Mixtures test

# Example usage
t = np.linspace(0, 1, 4)
means = [0, 5, 10]
stds = [1, 1, 1]
weights = [0.2, 0.5, 0.3]
mixture_samples = get_mixtures(t, means, stds, weights)
mixture_samples

#%% Plot one experiment
frequency_a = 2
frequency_b = 3
phase_shift = .5
sample_rate = SAMPLE_RATE
duration = DURATION
t = generate_timepoints(duration, sample_rate)
# a = get_sine_wave(t, frequency_a, phase=0.0)
# b = get_sine_wave(t, frequency_b, phase=phase_shift)
# plot_signals(a, b, t)
# c = get_gaussian_trace(t, 1, 0.1)
# d = get_gaussian_trace(t, 0, 0.1)
# plot_signals(c, d, t)
e = get_mixtures(t, [0.1, 1, 10], [.1, .5, 2], [0.2, 0.5, 0.01])
f = get_mixtures(t, [0.1, 1, 5], [.1, .5, 2], [0.2, 0.5, 0.01])
plot_signals(e, f, t)
#%% Experiment loop


def experiment(frequency_a, frequency_b, phase_shift=0.5, duration=.1, sample_rate=20000):
    # Generate data
    t = generate_timepoints(duration, sample_rate)
    a = get_sine_wave(t, frequency_a, phase=0.0)
    b = get_sine_wave(t, frequency_b, phase=phase_shift)
    return BASE_METRIC(a, b).numpy(), COMPARE_METRIC(a, b).numpy()


def experiment_gaussian(mean_a, mean_b, std_a, std_b, weights=None, duration=1., sample_rate=20000):
    # Generate data
    t = generate_timepoints(duration, sample_rate)
    if weights is None or len(weights) == 0:
        assert type(mean_a) == type(mean_b) == type(std_a) == type(
            std_b
        ) == np.float64, "Weights must be provided for multiple components"
        a = get_gaussian_trace(t, mean_a, std_a)
        b = get_gaussian_trace(t, mean_b, std_b)
    else:
        a = get_mixtures(t, mean_a, std_a, weights)
        b = get_mixtures(t, mean_b, std_b, weights)
    return BASE_METRIC(a, b).numpy(), COMPARE_METRIC(a, b).numpy()


def generate_dataframe(n=100):
    frequencies_a, frequencies_b = np.random.randint(1, 5000, (2, n))
    phase_shifts = np.random.random_sample(n)
    data = []
    with Progress() as progress:
        task = progress.add_task("[green]Processing...", total=n)
        for frequency_a, frequency_b, phase_shift in zip(frequencies_a, frequencies_b, phase_shifts):
            metric_a, metric_b = experiment(
                frequency_a, frequency_b, phase_shift=phase_shift, sample_rate=SAMPLE_RATE, duration=DURATION
            )
            data.append(
                {
                    "frequency_a": frequency_a,
                    "frequency_b": frequency_b,
                    "phase_shift": phase_shift,
                    "duration": DURATION,
                    "sample_rate": SAMPLE_RATE,
                    "metric_a": metric_a,
                    "metric_b": metric_b,
                    "ratio": metric_a / metric_b,
                }
            )
            progress.advance(task)
    return pd.DataFrame(data)


def generate_gaussian_dataframe(n=100, components=1):
    if components > 1:
        means_a, means_b = np.random.random_sample((2, n, components)) * 10
        stds_a, stds_b = np.random.random_sample((2, n, components))
        weights = np.random.random_sample((n, components))
    else:
        means_a, means_b = np.random.random_sample((2, n)) * 2 - 1
        stds_a, stds_b = np.random.random_sample((2, n))
        weights = [None] * n
    data = []
    with Progress() as progress:
        task = progress.add_task("[green]Processing...", total=n)
        for mean_a, mean_b, std_a, std_b, weight in zip(means_a, means_b, stds_a, stds_b, weights):
            metric_a, metric_b = experiment_gaussian(
                mean_a, mean_b, std_a, std_b, weights=weight, sample_rate=SAMPLE_RATE, duration=DURATION
            )
            data.append(
                {
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "std_a": std_a,
                    "std_b": std_b,
                    "weights": weights,
                    "duration": DURATION,
                    "sample_rate": SAMPLE_RATE,
                    "metric_a": metric_a,
                    "metric_b": metric_b,
                    "ratio": metric_a / metric_b,
                }
            )
            progress.advance(task)
    return pd.DataFrame(data)


#%% Plot the experiments to find the ratio between MSE and fourtier MSE
import plotly.express as px


def plot_experiments(df):

    fig = px.scatter(
        df,
        x="metric_a",
        y="metric_b",
        color="ratio",
        height=720,
        width=800,
        labels={
            "metric_a": f"{type(BASE_METRIC).__name__}",
            "metric_b": type(COMPARE_METRIC).__name__,
            "ratio": "Ratio MSE/Fourier MSE"
        },
    )
    # Update layout to make scaling of x and y axes equal
    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1, tickmode='linear', dtick=0.25),
        yaxis=dict(scaleanchor="x", scaleratio=1, tickmode='linear', dtick=0.25),
    )
    return fig


#%%
BASE_METRIC = torchmetrics.MeanSquaredLogError()
# BASE_METRIC = torchmetrics.MeanSquaredLogError()
COMPARE_METRIC = FourierMSLE()
# %%
df = generate_dataframe()
fig = plot_experiments(df)
fig.show()

# %%
DURATION = .1
df = generate_gaussian_dataframe(100)
fig = plot_experiments(df)
fig.show()

#%% Test components
DURATION = .3
df = generate_gaussian_dataframe(1000, components=3)
fig = plot_experiments(df)
fig.show()

# %%
df
# %%
