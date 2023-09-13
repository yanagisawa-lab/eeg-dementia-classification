import warnings
import mne
import numpy as np

class Filter(object):
    def __init__(
        self,
        fs: int,
        fir_design: str = 'firwin',
        fir_window: str = 'hamming',
        validate: bool = True) -> None:
        self.fs = fs
        self.fir_design = fir_design
        self.fir_window = fir_window
        self.validate = validate
        
    @classmethod
    def apply(self):
        pass


    @staticmethod
    def _valid_signal(x:np.ndarray) -> None:
        if np.isinf(x).any():
            warnings.warn('array contains inf')

        if np.isnan(x).any():
            warnings.warn('array contains NaN')
