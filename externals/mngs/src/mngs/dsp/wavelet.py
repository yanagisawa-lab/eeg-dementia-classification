#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy.signal.tf_misfit import cwt


def wavelet(wave, samp_rate, f_min=100, f_max=None, plot=False):
    dt = 1.0 / samp_rate
    npts = len(wave)
    t = np.linspace(0, dt * npts, npts)
    if f_min == None:
        f_min = 0.1
    if f_max == None:
        f_max = int(samp_rate / 2)
    scalogram = cwt(wave, dt, 8, f_min, f_max)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x, y = np.meshgrid(
            t, np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0])
        )

        ax.pcolormesh(x, y, np.abs(scalogram), cmap=obspy_sequential)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_yscale("log")
        ax.set_ylim(f_min, f_max)
        plt.show()

    Hz = pd.Series(np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))
    df = pd.DataFrame(np.abs(scalogram))
    df["Hz"] = Hz
    df.set_index("Hz", inplace=True)

    return df
