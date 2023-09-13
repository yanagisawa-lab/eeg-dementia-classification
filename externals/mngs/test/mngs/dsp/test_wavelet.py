#!/usr/bin/env python3

from mngs.dsp import wavelet
import numpy as np


def test_wavelet():
    signal = np.random.rand(1024)
    samp_rate = 512
    Nyq = int(samp_rate / 2)
    try:
        wavelet(signal, samp_rate, f_min=10, f_max=Nyq, plot=False)
    except:
        assert False


test_wavelet()

"""
spy -m pytest ./test/mngs/dsp/test_wavelet.py
"""
