from typing import Union
import mne
from .filters_base import Filter
import numpy as np


class BandPassFilter(Filter):
    def __init__(
        self,
        fs: int = 500,
        low_freq: float = 0.53,
        high_freq: float = 100,
        fir_design: str = "firwin",
        fir_window: str = "hamming",
        validate: bool = True,
    ) -> None:
        super().__init__(fs, fir_design, fir_window, validate)

        self.low_freq = low_freq
        self.high_freq = high_freq

    def apply(self, x_in: np.ndarray, **kwargs) -> np.ndarray:

        x_out = mne.filter.filter_data(
            data=x_in,
            sfreq=self.fs,
            l_freq=self.low_freq,
            h_freq=self.high_freq,
            fir_design=self.fir_design,
            fir_window=self.fir_window,
            **kwargs
        )
        if self.validate:
            self._valid_signal(x_out)
        return x_out
