import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torchaudio.transforms as audio_transforms
import torchmetrics

from src.config import get_current_config


class FourierMSLE(torchmetrics.MeanSquaredLogError):

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds_fft = torch.abs(torch.fft.fft(preds))
        target_fft = torch.abs(torch.fft.fft(target))
        assert preds_fft.shape == target_fft.shape
        super().update(preds_fft, target_fft)


class FrequencySpectrumMSESimple(torchmetrics.MeanSquaredError):

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        super().update(preds.real, target.real)
        super().update(preds.imag, target.imag)


class FrequencyPhaseAmpMSE(torchmetrics.MeanSquaredError):

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        super().update(preds.abs(), target.abs())
        super().update(preds.angle(), target.angle())


def get_sine_wave(frequency, duration, sample_rate=20000):
    """Generate a sine wave with the given frequency and duration."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    x = np.sin(2 * np.pi * frequency * t)
    return x


def test_specogram_plot():
    C = get_current_config()
    # Load the example file
    sine = get_sine_wave(500, 0.0191, sample_rate=10000)
    sine += get_sine_wave(1000, 0.0191, sample_rate=10000)
    sine += get_sine_wave(4000, 0.0191, sample_rate=10000)
    sine += get_sine_wave(4999, 1, sample_rate=10000)
    fig = spectogram_plot(sine, title="Spectogram of shot 0")
    return fig


def spectogram_plot(signal: np.ndarray, title="", hop_length=10, win_length=50):
    """Plot the spectogram of the data. Df should be a single shot."""
    C = get_current_config()
    signal = signal.squeeze()
    if len(signal) < win_length * 2:
        raise ValueError(
            f"Signal of length {len(signal)} is too short for the given window length {win_length}"
        )
    to_spectrogram = audio_transforms.Spectrogram(
        n_fft=win_length, win_length=win_length, hop_length=hop_length, power=1, pad=0
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    stft_spectogram = to_spectrogram(torch.tensor(signal)).numpy()
    fig.add_trace(
        go.Heatmap(
            z=np.log(stft_spectogram),
            dx=hop_length,
            colorscale='Viridis',
            colorbar=dict(title='Log Value'),
            name=f'STFT spectrogram (Win: {win_length}, Hop: {hop_length})',
            showlegend=True
        ),
        secondary_y=False
    )
    freq_domain = torch.fft.fft(torch.tensor(signal)).numpy()
    amplitudes = np.abs(freq_domain)
    fig.add_trace(
        px.line(
            x=np.arange(len(amplitudes)),  # t in sync with the hop windows
            y=np.log(amplitudes),
            labels={
                'x': 'Frequency',
                'y': 'Amplitude'
            },
            line_shape='linear',
            color_discrete_sequence=["rgb(255, 10, 10)"]
        ).data[0].update(name='Frequency Spectrum (log)', showlegend=True),
        secondary_y=False,
    )
    # Signal
    fig.add_trace(
        px.line(
            x=np.arange(len(signal)),  # t in sync with the hop windows
            y=signal,
            labels={
                'x': 'Time',
                'y': 'Value'
            },
            line_shape='linear',
        ).data[0].update(name='Signal', showlegend=True),
        secondary_y=True
    )

    fig.update_layout(
        title="Fourier plot: " + title,
        legend=dict(
            x=0.01,
            y=0.99,
            traceorder='normal',
            font=dict(family='sans-serif', size=12, color='black'),
            bgcolor='Azure',
            bordercolor='Black',
            borderwidth=2
        )
    )
    return fig


def signal_fourier_comparison_plot(true_signal: np.ndarray, predicted_signal: np.ndarray, title=""):
    """Plot the true and predicted signals and their corresponding amplitudes."""
    true_signal = true_signal.squeeze()
    predicted_signal = predicted_signal.squeeze()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # True Signal
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(true_signal)),
            y=true_signal,
            mode='lines+markers',
            name='True Signal',
            line=dict(color='blue',)
        ),
        secondary_y=True
    )

    # Predicted Signal
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(predicted_signal)),
            y=predicted_signal,
            mode='lines+markers',
            name='Predicted Signal',
            line=dict(color='blue', dash='dash')
        ),
        secondary_y=True
    )

    # True Signal Amplitudes
    true_freq_domain = torch.fft.fft(torch.tensor(true_signal)).numpy()
    true_amplitudes = np.abs(true_freq_domain)
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(true_amplitudes)),
            y=np.log(true_amplitudes),
            mode='lines',
            name='True Signal Amplitudes (log)',
            line=dict(color='red')
        ),
        secondary_y=False
    )

    # Predicted Signal Amplitudes
    predicted_freq_domain = torch.fft.fft(torch.tensor(predicted_signal)).numpy()
    predicted_amplitudes = np.abs(predicted_freq_domain)
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(predicted_amplitudes)),
            y=np.log(predicted_amplitudes),
            mode='lines',
            name='Predicted Signal Amplitudes (log)',
            line=dict(color='red', dash='dash')
        ),
        secondary_y=False
    )

    fig.update_layout(
        title="Signal Comparison: " + title,
        xaxis_title='Time / Frequency',
        yaxis_title='Value',
        legend=dict(
            x=0.01,
            y=0.99,
            traceorder='normal',
            font=dict(family='sans-serif', size=12, color='black'),
            bgcolor='Azure',
            bordercolor='Black',
            borderwidth=2
        )
    )
    return fig
