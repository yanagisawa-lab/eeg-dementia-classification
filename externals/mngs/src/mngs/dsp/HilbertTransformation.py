#!/usr/bin/env python

import numpy as np
import torch  # 1.7.1
import torch.nn as nn
from torch.fft import fft, ifft


class BaseHilbertTransformer(nn.Module):
    # https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/simplelayers.py
    def __init__(self, axis=2, n=None):
        super().__init__()
        self.axis = axis
        self.n = n

    def transform(self, x):
        """
        Args:
            x: Tensor or array-like to transform.
               Must be real and in shape ``[Batch, chns, spatial1, spatial2, ...]``.
        Returns:
            torch.Tensor: Analytical signal of ``x``,
                          transformed along axis specified in ``self.axis`` using
                          FFT of size ``self.N``.
                          The absolute value of ``x_ht`` relates to the envelope of ``x``
                          along axis ``self.axis``.
        """

        n = x.shape[self.axis] if self.n is None else self.n

        # Create frequency axis
        f = torch.cat(
            [
                torch.true_divide(
                    torch.arange(0, (n - 1) // 2 + 1, device=x.device), float(n)
                ),
                torch.true_divide(
                    torch.arange(-(n // 2), 0, device=x.device), float(n)
                ),
            ]
        )

        xf = fft(x, n=n, dim=self.axis)

        # Create step functionb
        u = torch.heaviside(f, torch.tensor([0.5], device=f.device))
        u = torch.as_tensor(u, dtype=x.dtype, device=u.device)
        new_dims_before = self.axis
        new_dims_after = len(xf.shape) - self.axis - 1
        for _ in range(new_dims_before):
            u.unsqueeze_(0)
        for _ in range(new_dims_after):
            u.unsqueeze_(-1)

        transformed = ifft(xf * 2 * u, dim=self.axis)

        return transformed


class HilbertTransformer(BaseHilbertTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hilbert_transform = self.transform
        self.register_buffer("pi", torch.tensor(np.pi))

    def forward(self, sig):
        sig_comp = self.hilbert_transform(sig)

        pha = self._calc_Arg(sig_comp)
        amp = sig_comp.abs()

        out = torch.cat(
            [
                pha.unsqueeze(-1),
                amp.unsqueeze(-1),
            ],
            dim=-1,
        )

        return out

    def _calc_Arg(self, comp):
        """Calculates argument of complex numbers in (-pi, pi] space.
        Although torch.angle() does not have been implemented the derivative function,
        this function seems to work.
        """
        x, y = comp.real, comp.imag
        Arg = torch.zeros_like(x).type_as(x)

        c1 = x > 0  # condition #1
        Arg[c1] += torch.atan(y / x)[c1]

        c2 = (x < 0) * (y >= 0)
        Arg[c2] += torch.atan(y / x)[c2] + self.pi

        c3 = (x < 0) * (y < 0)
        Arg[c3] += torch.atan(y / x)[c3] - self.pi

        c4 = (x == 0) * (y > 0)
        Arg[c4] += self.pi / 2.0

        c5 = (x == 0) * (y < 0)
        Arg[c5] += -self.pi / 2.0

        c6 = (x == 0) * (y == 0)
        Arg[c6] += 0.0

        return Arg


def _unwrap(x):
    pi = torch.tensor(np.pi)
    y = x % (2 * pi)
    return torch.where(y > pi, 2 * pi - y, y)


def test_hilbert(sig, axis=1, test_layer=True):

    if test_layer:
        ## Hilbert Layer
        hilbert_layer = HilbertTransformer(axis=axis).cuda()
        sig_comp = hilbert_layer(sig)

    else:
        ## Hilbert Transformation
        hilbert = HilbertTransformer(axis=1)
        sig_comp = hilbert.transform(sig)

    print(sig.shape)
    print(sig_comp.shape)

    print(sig.dtype)
    print(sig_comp.dtype)

    print(sig_comp)

    # ## Extract Amplitude and Phase signals
    # amp = sig_comp.abs()
    # pha = _unwrap(sig_comp.angle())

    # instantaneous_freq = (np.diff(pha.cpu()) / (2.0*np.pi) * fs)

    # ## Plot
    # fig, ax = plt.subplots(3, 1)
    # ax[0].set_title('GPU Hilbert Transformation')

    # ax[0].plot(t, sig[0].cpu(), label='original signal')
    # ax[0].plot(t, amp[0].cpu(), label='envelope')
    # ax[0].set_xlabel("time in seconds")
    # ax[0].legend()

    # ax[1].plot(t, pha[0].cpu(), label='phase')
    # ax[1].set_xlabel("time in seconds")
    # ax[1].legend()

    # ax[2].plot(t[1:], instantaneous_freq[0], label='instantenous frequency')
    # ax[2].set_xlabel("time in seconds")
    # ax[2].set_ylim(0.0, 120.0)
    # ax[2].legend()

    # fig.show()

    pha, amp = sig_comp[..., 0], sig_comp[..., 1]
    return pha, amp


def mk_sig():
    from scipy.signal import chirp

    def _mk_sig():
        sig = chirp(t, 20.0, t[-1], 100.0)
        sig *= 1.0 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t)
        return sig

    sig = np.array([_mk_sig() for _ in range(batch_size)]).astype(np.float32)
    sig = torch.tensor(sig).cuda()
    return sig


if __name__ == "__main__":
    # import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.signal import chirp

    ## Parameters
    duration = 1.0
    fs = 400.0
    samples = int(fs * duration)
    t = np.arange(samples) / fs
    batch_size = 64

    ## Create demo signal
    sig = mk_sig()

    ## Test Code
    pha, amp = test_hilbert(sig, axis=1, test_layer=True)
    # ## EOF
