import numpy as np
from scipy import signal
from numpy.fft import fft
import matplotlib.pyplot as plt
from vbi.feature_extraction.features_utils import km_order


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


def plot_ts_pxx_jr(data, par, ax, method="welch", **kwargs):
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


def plot_ts_jr(data, par, ax, **kwargs):
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

def plot_ts_pxx_sl(data, params, **kwargs):

    x = data['x']
    t = data['t']

    mosaic = """
    AAB
    """

    fs = 1/(params['dt']*params['record_step'])
    fig = plt.figure(constrained_layout=True, figsize=(12, 3))
    ax = fig.subplot_mosaic(mosaic)

    x_avg = np.mean(x, axis=0)
    ax['A'].plot(t, x_avg.T, label="x", **kwargs)
    ax['A'].set_ylabel(r"$\sum$ Real $Z$", fontsize=16)

    freq, pxx = signal.welch(x, fs=fs, nperseg=4096)
    # pxx /= np.max(pxx)
    pxx_avg = np.average(pxx, axis=0)
    ax['B'].plot(freq, pxx_avg, **kwargs)
    ax['B'].set_xlabel("Frequency [Hz]", fontsize=16)
    ax['B'].set_ylabel("Power", fontsize=16)
    ax['B'].set_xlim(0, 60)
    ti = params['t_transition']
    tf = params['t_end']
    ax['A'].set_xlim(tf-2, tf)
    ax['A'].set_xlabel("Time [s]", fontsize=16)

    idx = np.argmax(pxx_avg)
    print(f"fmax = {freq[idx]} Hz, Pxx = {pxx_avg[idx]}")


def plot_ts_pxx_km(data, params, ax, **kwargs):

    x = np.sin(data['x'])
    t = data['t']

    ax[0].plot(t, x.T, label="x", **kwargs)
    ax[0].set_ylabel(r"$\theta$", fontsize=16)

    freq, pxx = signal.welch(x, fs=1/params['dt'], nperseg=4096)
    # pxx /= np.max(pxx)
    ax[1].plot(freq, pxx.T, **kwargs)
    ax[1].set_xlabel("Frequency [Hz]", fontsize=16)
    ax[1].set_ylabel("Power", fontsize=16)
    ax[1].set_xlim(0, 1)

    # ti = params['t_transition']
    # tf = params['t_end']
    # ax[0].set_xlim(tf-2, tf)
    ax[0].set_xlabel("Time [s]", fontsize=16)

    pxx_avg = np.average(pxx, axis=0)
    idx = np.argmax(pxx_avg)
    print(f"fmax = {freq[idx]} Hz, Pxx = {pxx_avg[idx]}")


def plot_ts_pxx_km_cupy(data, params, ax, **kwargs):

    x = np.sin(data['x'])
    t = data['t']

    ax[0].plot(t, x[:, :, 0], label="x", **kwargs)
    ax[0].set_ylabel(r"$\theta$", fontsize=16)

    freq, pxx = signal.welch(x[:,:,0].T, fs=1/params['dt'], nperseg=4096)
    # pxx /= np.max(pxx)
    ax[1].plot(freq, pxx.T, **kwargs)
    ax[1].set_xlabel("Frequency [Hz]", fontsize=16)
    ax[1].set_ylabel("Power", fontsize=16)
    ax[1].set_xlim(0, 1)
    ax[0].set_xlabel("Time [s]", fontsize=16)

    ns = x.shape[2]
    R = np.zeros(ns)
    for i in range(ns):
        R[i] = km_order(x[:, :, i].T)

    ax[2].plot(params['G'], R, marker="o", c="k")
    ax[2].set_xlabel("G", fontsize=16)
    ax[2].set_ylabel(r"$\langle r \rangle_t$", fontsize=16)




def plot_mat(mat, ax, **kwargs):
    im = ax.imshow(mat, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
