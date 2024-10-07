"""Let's test how to translate a signal to fourier space and back."""
#%%
from matplotlib import legend
from matplotlib.pylab import f
from src import fourier
import numpy as np
import torch
import plotly.graph_objects as go
"""
We need to 

Plot the original signal
Plot the fourier transform of the signal
Plot the inverse transform

Optionally allow some manipulation of the signal in fourier space
And then plot the inverse transform of the manipulated signal
"""


#%%
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
    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()

    # Choose a Gaussian component for each sample
    components = np.random.choice(len(means), size=len(t), p=weights)

    # Generate samples from the chosen components
    x = np.array([np.random.normal(means[comp], stds[comp]) for comp in components])

    return torch.tensor(x)


#%%
t = generate_timepoints(duration=1, sample_rate=1000)
sine_wave = get_sine_wave(t, 10)
gaussian_wave = get_gaussian_trace(t, mean=0, std=1)
mixture_wave = get_mixtures(t, means=[0, 1], stds=[.2, 1], weights=[5, 1])

signals = [sine_wave, gaussian_wave, mixture_wave]
fourier_signals = [torch.fft.rfft(signal) for signal in signals]


#%% Plotting
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
    fig.update_layout(
        title=f"Sine waves (difference sum {total_difference})",
        xaxis_title="Time",
        yaxis_title="Amplitude",
        yaxis2=dict(
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=True,
            showticklabels=True,
        ),
        legend=dict(x=1.15, y=1, traceorder='normal', orientation='v')
    )
    print(f"Total difference: {total_difference}")
    fig.show()


def plot_fourier_space(fourier_signal):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(fourier_signal)), y=fourier_signal.abs(), mode="lines"))
    fig.update_layout(title="Fourier Transform")
    fig.show()


for signal, fourier_signal in zip(signals, fourier_signals):
    inverse_signal = torch.fft.irfft(fourier_signal)
    plot_signals(signal, inverse_signal, t)
    plot_fourier_space(fourier_signal)

# %% Test multi channel signal dimensions

signals = torch.stack([sine_wave, gaussian_wave, mixture_wave])
signals.size()
# %%
fourier_space = torch.fft.rfft(signals)
fourier_space.size()
# %%
torch.fft.irfft(fourier_space).size()
# %%
