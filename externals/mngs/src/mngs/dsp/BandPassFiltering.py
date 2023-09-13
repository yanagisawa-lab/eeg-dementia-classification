#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal


class BandPassFilter(nn.Module):
    # https://raw.githubusercontent.com/mravanelli/SincNet/master/dnn_models.py
    def __init__(self, order=None, low_hz=30, high_hz=60, fs=250, n_chs=19):
        super().__init__()
        self.fs = fs
        nyq = fs / 2
        self.order = fs if order is None else order
        self.numtaps = self.order + 1
        filter_npy = scipy.signal.firwin(
            self.numtaps, [low_hz, high_hz], pass_zero="bandpass", fs=fs,
        )

        self.register_buffer('filters',
                             torch.tensor(filter_npy).unsqueeze(0).unsqueeze(0)
        )
        

    def forward(self, x):
        dim = x.ndim
        sig_len_orig = x.shape[-1]

        if dim == 3:
            bs, n_chs, sig_len = x.shape
            x = x.reshape(bs*n_chs, 1, sig_len)
            
        filted = F.conv1d(x, self.filters.type_as(x), padding=int(self.numtaps/2))
        filted = filted.flip(dims=[-1]) # to backward
        filted = F.conv1d(filted, self.filters.type_as(x), padding=int(self.numtaps/2))
        filted = filted.flip(dims=[-1]) # reverse to the original order        

        filted = filted[..., 1:-1]
        # print(self.order, filted.shape[-1])        

        if dim == 3:
            filted = filted.reshape(bs, n_chs, -1)
        
        return filted


class BandPasser_CPU:
    def __init__(self, low_hz=100, high_hz=250, fs=1000):
        from scipy.signal import butter, sosfilt

        self.butter = butter
        self.sosfilt = sosfilt

        self.sos = self.mk_sos(low_hz, high_hz, fs)

    def __call__(self, raw_sig):
        filted = self.sosfilt(self.sos, raw_sig)
        return filted

    def mk_sos(self, low_hz, high_hz, fs, order=5):
        nyq = fs / 2.0
        sos = self.butter(
            order,
            [low_hz / nyq, high_hz / nyq],
            analog=False,
            btype="band",
            output="sos",
        )
        return sos

    
def wavelet(
    wave, samp_rate, f_min=100, f_max=None, plot=False, title=None
):  # for visual checking
    from obspy.signal.tf_misfit import cwt
    from obspy.imaging.cm import obspy_sequential

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
        ax.set_title(title)
        ax.set_ylim(f_min, f_max)
        fig.show()

    Hz = pd.Series(np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))
    df = pd.DataFrame(np.abs(scalogram))
    df["Hz"] = Hz
    df.set_index("Hz", inplace=True)

    return df

    
## from here, not important
def summarize_sinusoidal_waves(T_sec=50.0, fs=250, freqs_hz=[30, 60, 100, 200, 1000]):
    n = int(T_sec * fs)
    t = np.linspace(0, T_sec, n, endpoint=False)
    summed = np.array(
        [
            np.random.rand() * np.sin( (f_hz * t + np.random.rand() ) * (2 * np.pi) )            
            for f_hz in freqs_hz
        ]
    ).sum(axis=0)
    return summed

def prepair_demo_data(T_sec=50.0, fs=250, freqs_hz=[30, 60, 100, 200, 1000], n_chs=19):
    data = np.array([
        summarize_sinusoidal_waves(T_sec=T_sec, fs=fs, freqs_hz=freqs_hz)
        for _ in range(n_chs)
        ])
    return data


class TimeKeeper():
    def __init__(self,):
        self.start_time = time.time()
        
    def __call__(self, message=None):
        self.elapsed = time.time() - self.start_time
        print(message)        
        print('elapsed time: {:.5f} [sec]'.format(self.elapsed))

    def start(self, ):
        self.start_time = time.time()
    

if __name__ == "__main__":
    ## Parameters
    PASS_LOW_HZ = 40
    PASS_HIGH_HZ = 80
    T_sec = 50.0
    fs = 250
    freqs_hz = [30, 60, 100, 200, 1000]
    n_chs = 19
    bs = 64
    i_batch_show = 0
    i_ch_show = 0    
    sig_orig = np.array([
        prepair_demo_data(T_sec=T_sec, fs=fs, freqs_hz=freqs_hz, n_chs=n_chs)
        for _ in range(bs)
    ])
    print('sig_orig_shape: {}'.format(sig_orig.shape))
    tk = TimeKeeper()    
        
    
    ## Original
    title_orig = "Original, #{}, ch{} (Demo Freqs: {} Hz)"\
        .format(i_batch_show, i_ch_show, freqs_hz)
    wv_out_orig = wavelet(sig_orig[i_batch_show, i_ch_show], fs,
                          f_min=1, plot=True, title=title_orig)

    
    ## CPU Bandpass Filtering
    bp_cpu = BandPasser_CPU(low_hz=PASS_LOW_HZ, high_hz=PASS_HIGH_HZ, fs=fs)
    filted_cpu = sig_orig.copy()
    ### calculation start ###
    tk.start()
    for i_batch in range(bs):
        for i_ch in range(n_chs):
            filted_cpu[i_batch, i_ch] = bp_cpu(filted_cpu[i_batch, i_ch])
    tk(message='CPU')
    ### calculation end ###    
    title_filt_cpu = (
        "[CPU] Bandpass Filted, #{}, ch{} (Freqs: (Low_lim, High_lim) = ({}, {}) Hz) (time: {:.5f} [sec])"\
        .format(i_batch_show, i_ch_show, PASS_LOW_HZ, PASS_HIGH_HZ, tk.elapsed
        )
    )
    _wv_out_filted_cpu = wavelet(filted_cpu[i_batch_show, i_ch_show], fs,
                                 f_min=1, plot=True, title=title_filt_cpu)

    
    ## GPU Bandpass Filtering
    BandPassFilter_GPU = BandPassFilter
    bp_gpu = BandPassFilter_GPU(
        low_hz=PASS_LOW_HZ, high_hz=PASS_HIGH_HZ, fs=fs
    ).cuda()
    # sig_torch = torch.tensor(sig_orig).unsqueeze(0).unsqueeze(0).cuda()
    sig_torch = torch.tensor(sig_orig).cuda()
    ### calculation start ###
    tk.start()
    filted_gpu = bp_gpu(sig_torch)
    tk(message='GPU')
    ### calculation end ###    
    title_filt_gpu = (
        "[GPU] Bandpass Filted, #{}, ch{} (Freqs: (Low_lim, High_lim) = ({}, {}) Hz) (time: {:.5f} [sec])"\
        .format(i_batch_show, i_ch_show, PASS_LOW_HZ, PASS_HIGH_HZ, tk.elapsed
        )
    )
    _wv_out_filted_gpu = wavelet(filted_gpu[i_batch_show, i_ch_show].cpu(), fs,
                                 f_min=1, plot=True, title=title_filt_gpu)

    
    ## EOF
