"""
abr_toolbox.preprocess

Preprocessing utilities for ABR and audio signals.
"""
import numpy as np
import mne
from scipy.ndimage import gaussian_filter1d

def bandpass_filter(raw: mne.io.BaseRaw,
                    l_freq: float,
                    h_freq: float,
                    method: str = 'fir',
                    filter_length: str = 'auto') -> mne.io.BaseRaw:
    """
    Bandpass filter raw data in-place.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw data.
    l_freq : float
        Low cutoff frequency (Hz).
    h_freq : float
        High cutoff frequency (Hz).
    method : {'fir', 'iir'}
        Filter design method.
    filter_length : str
        Length for FIR filters.

    Returns
    -------
    raw : mne.io.BaseRaw
        Filtered data.
    """
    raw.filter(l_freq=l_freq, h_freq=h_freq,
               method=method,
               fir_design='firwin' if method == 'fir' else None,
               filter_length=filter_length)
    return raw


def apply_baseline(epochs: mne.Epochs,
                   baseline: tuple) -> mne.Epochs:
    """
    Apply baseline correction to epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoch data.
    baseline : tuple
        (tmin, tmax) in seconds.

    Returns
    -------
    epochs : mne.Epochs
        Baseline-corrected epochs.
    """
    epochs.apply_baseline(baseline)
    return epochs


from typing import Union

def smooth_waveform(data: Union[list, np.ndarray],
                    sigma: float) -> np.ndarray:
    """
    Apply Gaussian smoothing to 1D waveform.

    Parameters
    ----------
    data : array-like
        1D signal.
    sigma : float
        Gaussian kernel std.

    Returns
    -------
    np.ndarray
        Smoothed signal.
    """
    return gaussian_filter1d(data, sigma=sigma)
