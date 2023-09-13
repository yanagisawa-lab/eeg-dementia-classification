#!/usr/bin/env python3

import numpy as np
from mngs.dsp import bandpassfilter


def test_bandpassfilter():
    data = np.random.rand(1024)
    lo_hz = 30
    hi_hz = 50
    fs = 512
    order = 5

    try:
        filted = bandpassfilter(data, lo_hz, hi_hz, fs, order=order)
    except:
        assert False


test_bandpassfilter()

## EOF
