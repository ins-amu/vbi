import numpy as np
from scipy import signal
from numpy.fft import fft
import matplotlib.pyplot as plt


def fft_signal(x, t):
    dt = t[1] - t[0]
    if x.ndim == 1:
        x = x[None, :]
    N = x.shape[1]
    T = N * dt
    xf = fft(x - x.mean(axis=1, keepdims=True), axis=1)
    Sxx = 2 * dt**2 / T * (xf * xf.conj()).real
    Sxx = Sxx[:, :N//2]

    df = 1.0 / T
    fNQ = 1.0 / (2.0 * dt)
    faxis = np.arange(0, fNQ, df)
    return faxis, Sxx


def plot_ts_pxx(data, par, ax, method="welch", **kwargs):
    tspan = data['t']
    y = data['x']
    ax[0].plot(tspan, y.T, label='y1 - y2', **kwargs)

    if method == "welch":
        freq, pxx = signal.welch(y, 1000/par['dt'], nperseg=y.shape[1]//2)
    else:
        freq, pxx = fft_signal(y, tspan / 1000)
    ax[1].plot(freq, pxx.T, **kwargs)
    ax[1].set_xlim(0, 50)
    ax[1].set_xlabel("frequency [Hz]")
    ax[0].set_xlabel("time [ms]")
    ax[0].set_ylabel("y1-y2")
    ax[0].margins(x=0)

    plt.tight_layout()


def plot_ts(data, par, ax, **kwargs):
    tspan = data['t']
    y = data['x']
    ax[0].plot(tspan, y.T, label='y1 - y2', **kwargs)

    freq, pxx = signal.welch(y, 1000/par['dt'], nperseg=y.shape[1]//2)
    ax[1].plot(freq, pxx.T, **kwargs)
    ax[1].set_xlim(0, 50)
    ax[1].set_xlabel("frequency [Hz]")
    ax[1].set_ylabel("PSD")
    ax[0].set_xlabel("time [ms]")
    ax[0].set_ylabel("y1-y2")
    ax[0].margins(x=0)
    for i in range(2):
        ax[i].tick_params(labelsize=14)

    plt.tight_layout()
