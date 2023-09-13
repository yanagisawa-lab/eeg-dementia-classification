#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy import fftpack


def calc_fft_powers(signals_2d, samp_rate):
    """
    Example:
        sig_len = 1024
        n_sigs = 32
        signals_2d = np.random.rand(n_sigs, sig_len)
        samp_rate = 256
        fft_df = calc_fft_powers(signals_2d, samp_rate)
    """
    fft_powers = np.abs(fftpack.fft(signals_2d))
    fft_freqs = np.fft.fftfreq(signals_2d.shape[-1], d=1.0 / samp_rate)
    mask = fft_freqs >= 0
    fft_powers, fft_freqs = fft_powers[:, mask], np.round(fft_freqs[mask], 1)
    fft_df = pd.DataFrame(data=fft_powers, columns=fft_freqs.astype(str))
    return fft_df
