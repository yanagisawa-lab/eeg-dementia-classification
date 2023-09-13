#!/usr/bin/env python3
# Time-stamp: "2021-11-30 17:09:22 (ylab)"

import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch
import torch.nn.functional as F
import torchaudio
from pyro.ops.stats import autocorrelation as pyro_acf
from scipy.signal import chirp, firwin
from functools import partial
from mngs.dsp.HilbertTransformation import HilbertTransformer


## Global definitions
BANDS_LIM_HZ_DICT = {
    "delta": [0.5, 4],
    "theta": [4, 8],
    "lalpha": [8, 10],
    "halpha": [10, 13],
    "beta": [13, 32],
    "gamma": [32, 75],
}


def demo_sig(batch_size=64, n_chs=19, samp_rate=1000, len_sec=10, freqs_hz=[2, 5, 10]):
    time = torch.arange(0, len_sec, 1 / samp_rate)
    sig = torch.vstack([torch.sin(f * 2 * torch.pi * time) for f in freqs_hz]).sum(
        dim=0
    )
    sig = sig.unsqueeze(0).unsqueeze(0).repeat(batch_size, n_chs, 1)
    return sig


def phase(x, axis=-1):
    analytical_x = hilbert(x, axis=-1)
    return analytical_x[..., 0]


def phase_band(x, samp_rate, band_str="delta", l=None, h=None):
    """
    Example:
        x = mngs.dsp.demo_sig(x)
        mngs.dsp.phase_band(x, samp_rate=1000, band_str="delta")

        x = mngs.dsp.demo_sig(x)
        mngs.dsp.phase_band(x, samp_rate=1000, band_str=None, l=0.5, h=4)

    Bands definitions:
        BANDS_LIM_HZ_DICT = {
        "delta": [0.5, 4],
        "theta": [4, 8],
        "lalpha": [8, 10],
        "halpha": [10, 13],
        "beta": [13, 32],
        "gamma": [32, 75],
    }
    """
    is_band_passed = (band_str is not None) & (l is None) & (h is None)
    is_lh_passed = (band_str is None) & (l is not None) & (h is not None)
    assert is_band_passed or is_lh_passed

    if is_band_passed:
        l, h = BANDS_LIM_HZ_DICT[band_str]

    x = bandpass(x, samp_rate, low_hz=l, high_hz=h)
    return phase(x)


def amp(x, axs=-1):
    analytical_x = hilbert(x, axis=-1)
    return analytical_x[..., 1]


def amp_band(x, samp_rate, band_str="delta", l=None, h=None):
    """
    Example:
        x = mngs.dsp.demo_sig(x)
        mngs.dsp.amp_band(x, samp_rate=1000, band_str="delta")

        x = mngs.dsp.demo_sig(x)
        mngs.dsp.amp_band(x, samp_rate=1000, band_str=None, l=0.5, h=4)

    Bands definitions:
        BANDS_LIM_HZ_DICT = {
        "delta": [0.5, 4],
        "theta": [4, 8],
        "lalpha": [8, 10],
        "halpha": [10, 13],
        "beta": [13, 32],
        "gamma": [32, 75],
    }
    """
    is_band_passed = (band_str is not None) & (l is None) & (h is None)
    is_lh_passed = (band_str is None) & (l is not None) & (h is not None)
    assert is_band_passed or is_lh_passed

    if is_band_passed:
        l, h = BANDS_LIM_HZ_DICT[band_str]

    x = bandpass(x, samp_rate, low_hz=l, high_hz=h)
    return amp(x)


def hilbert(x, axis=-1):
    if axis == -1:
        axis = x.ndim - 1
    return HilbertTransformer(axis=axis).to(x.device)(x)


def fft(x, samp_rate):
    fn = partial(_fft_1d, samp_rate=samp_rate, return_freq=False)
    fft_coef = _apply_to_the_time_dim(fn, x)
    freq = torch.fft.fftfreq(x.shape[-1], d=1 / samp_rate)
    return fft_coef, freq


def _fft_1d(x, samp_rate, return_freq=True):
    freq = torch.fft.fftfreq(x.shape[-1], d=1 / samp_rate)
    fft_coef = torch.fft.fft(x)  # [:, :, :nyq]

    if return_freq:
        return fft_coef, freq
    else:
        return fft_coef


def rfft_bands(
    x, samp_rate, bands_str=["delta", "theta", "lalpha", "halpha", "beta", "gamma"]
):
    """
    Returns mean absolute rfft coefficients of bands.
    Bands' definitions are as follows.

    BANDS_LIM_HZ_DICT = {
        "delta": [0.5, 4],
        "theta": [4, 8],
        "lalpha": [8, 10],
        "halpha": [10, 13],
        "beta": [13, 32],
        "gamma": [32, 75],
    }
    """
    coef, freq = rfft(x, samp_rate)

    coef_bands_abs_mean = []
    for band_str in bands_str:
        low, high = BANDS_LIM_HZ_DICT[band_str]
        indi_band = (low <= freq) & (freq <= high)
        coef_band_abs_mean = coef[..., indi_band].abs().mean(dim=-1)
        coef_bands_abs_mean.append(coef_band_abs_mean)

    return torch.stack(coef_bands_abs_mean, dim=-1)


def rfft(x, samp_rate):
    fn = partial(_rfft_1d, samp_rate=samp_rate, return_freq=False)
    fft_coef = _apply_to_the_time_dim(fn, x)
    freq = torch.fft.rfftfreq(x.shape[-1], d=1 / samp_rate)
    return fft_coef, freq


def _rfft_1d(x, samp_rate, return_freq=True):
    freq = torch.fft.rfftfreq(x.shape[-1], d=1 / samp_rate)
    fft_coef = torch.fft.rfft(x)  # [:, :, :nyq]

    if return_freq:
        return fft_coef, freq
    else:
        return fft_coef


def bandstop(x, samp_rate, low_hz=55, high_hz=65):
    fn = partial(_bandstop_1d, samp_rate=samp_rate, low_hz=low_hz, high_hz=high_hz)
    return _apply_to_the_time_dim(fn, x)


def _bandstop_1d(x, samp_rate, low_hz=55, high_hz=65):
    fft_coef, freq = _rfft_1d(x, samp_rate)
    indi_to_cut = (low_hz < freq) & (freq < high_hz)
    fft_coef[indi_to_cut] = 0
    return torch.fft.irfft(fft_coef)


def bandpass(x, samp_rate, low_hz=55, high_hz=65):
    fn = partial(_bandpass_1d, samp_rate=samp_rate, low_hz=low_hz, high_hz=high_hz)
    return _apply_to_the_time_dim(fn, x)


def _bandpass_1d(x, samp_rate, low_hz=55, high_hz=65):
    fft_coef, freq = _rfft_1d(x, samp_rate)
    indi_to_cut = (freq < low_hz) + (high_hz < freq)
    fft_coef[indi_to_cut] = 0
    return torch.fft.irfft(fft_coef)


def spectrogram(x, fft_size, device="cuda"):
    """
    Short-time FFT for signals.

    Input: [BS, N_CHS, SEQ_LEN]
    Output: [BS, N_CHS, fft_size//2 + 1, ?]

    spect = spectrogram(x, 50)
    print(spect.shape)
    """

    transform = torchaudio.transforms.Spectrogram(n_fft=fft_size).to(device)
    spectrogram = transform(x)
    return spectrogram


def mean(x):
    return x.mean(-1, keepdims=True)


def std(x):
    return x.std(-1, keepdims=True)


def zscore(x):
    _mean = mean(x)
    diffs = x - _mean
    var = torch.mean(torch.pow(diffs, 2.0), dim=-1, keepdims=True)
    std = torch.pow(var, 0.5)
    return diffs / std


def kurtosis(x):
    zscores = zscore(x)
    kurtoses = torch.mean(torch.pow(zscores, 4.0), dim=-1, keepdims=True) - 3.0
    return kurtoses


def skewness(x):
    zscores = zscore(x)
    return torch.mean(torch.pow(zscores, 3.0), dim=-1, keepdims=True)


def median(x):
    return torch.median(x, dim=-1, keepdims=True)[0]


def q25(x, q=0.25):
    return torch.quantile(x, q, dim=-1, keepdims=True)


def q75(x, q=0.75):
    return torch.quantile(x, q, dim=-1, keepdims=True)


def rms(x):
    return torch.square(x).sqrt().mean(dim=-1, keepdims=True)


def beyond_r_sigma_ratio(x, r=2.0):
    sigma = std(x)
    return (sigma < x).float().mean(dim=-1, keepdims=True)


def acf(x):
    return pyro_acf(x, dim=-1)


def _apply_to_the_time_dim(fn, x):
    """
    x: [BS, N_CHS, SEQ_LEN]
    When fn(x[0,0]) works, _apply_to_the_time_dim(fn, x) works.
    """
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    dim = 0
    applied = torch.stack([fn(x_i) for x_i in torch.unbind(x, dim=dim)], dim=dim)
    return applied.reshape(shape[0], shape[1], -1)


def _test_notch_filter_1d():
    time = torch.linspace(0, 1, 999)
    sig = (
        torch.cos(60 * 2 * torch.pi * time)
        + torch.cos(200 * 2 * torch.pi * time)
        + torch.cos(300 * 2 * torch.pi * time)
    )

    sig_filted = notch_filter_1d(sig, SAMP_RATE, cutoff_hz=60, width_hz=5)
    fig, axes = plt.subplots(4, 1)
    axes[0].plot(sig, label="sig")
    fft_coef_sig, freq_sig = _rfft_1d(sig, SAMP_RATE)
    axes[1].plot(freq_sig, fft_coef_sig.abs(), label="fft_coef_sig")
    axes[2].plot(sig_filted, label="sig_filted")

    fft_coef_sig_filted, freq_sig_filted = _rfft_1d(sig_filted, SAMP_RATE)
    axes[3].plot(
        freq_sig_filted, fft_coef_sig_filted.abs(), label="fft_coef_sig_filted"
    )
    for ax in axes:
        ax.legend()
    fig.show()


def test_phase_amp_bandpass():
    samp_rate = 1000
    len_seq = 10
    time = torch.arange(0, len_seq, 1 / samp_rate)
    freqs_hz = [2, 5, 10]
    sig = torch.vstack([torch.sin(f * 2 * torch.pi * time) for f in freqs_hz]).sum(
        dim=0
    )
    # sig = _bandstop_1d(sig, samp_rate, low_hz=4, high_hz=12)
    sig = sig.unsqueeze(0).unsqueeze(0).repeat(BS, N_CHS, 1)
    sig = bandpass(sig, samp_rate, low_hz=0, high_hz=3)

    phase = _phase_1d(sig)
    amp = _amp_1d(sig)

    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    axes[0].plot(sig[0, 0], label="sig")
    axes[0].legend()
    axes[1].plot(phase[0, 0], label="phase")
    axes[1].legend()
    axes[2].plot(amp[0, 0], label="amp")
    axes[2].legend()
    fig.show()


if __name__ == "__main__":
    SAMP_RATE = 1000
    x = demo_sig(freqs_hz=[2, 3, 5, 10], samp_rate=SAMP_RATE, len_sec=2)
    # x = torch.tensor(chirp(time, 3, 500, 100))
