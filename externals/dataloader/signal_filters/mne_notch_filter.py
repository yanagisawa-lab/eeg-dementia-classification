import mne
import numpy as np
from typing import Union
from .filters_base import Filter


class NotchFilter(Filter):
    def __init__(
        self,
        fs: int = 500,
        freqs: np.ndarray = np.array([60]),
        fir_design: str = "firwin",
        fir_window: str = "hamming",
        validate: bool = True,
    ) -> None:
        super().__init__(fs, fir_design, fir_window, validate)

        self.freqs = freqs

    def apply(self, x_in: np.ndarray, **kwargs) -> np.ndarray:
        """
        # See the mne document :https://mne.tools/stable/generated/mne.filter.filter_data.html

        Args:
            x_in (np.ndarray): target signal
            fs (int): sampling frequency
            freqs (Union[float, np.ndarray, None]): frequencies to notch filter
            fir_design (str, optional): Defaults to 'firwin'.
            fir_window (str, optional): Defaults to 'hamming'.
            validate (bool, optional): If True, checks whether NaN or inf is contained in the array. Defaults to True.

        Returns:
            np.ndarray: filtered signal
        """

        try:
            x_out = mne.filter.notch_filter(
                x=x_in,
                Fs=self.fs,
                freqs=self.freqs,
                fir_design=self.fir_design,
                fir_window=self.fir_window,
                **kwargs
            )
            if self.validate:
                self._valid_signal(x_out)
            return x_out

        except Exception as e:
            print(e)
            print("\nSince notch filter was not applied appropriately, nan signal was returned.\n")
            return np.nan * x_in # None

