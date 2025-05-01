import os
import numpy as np
import librosa
from sofa import SOFAFile
from mne_bids import BIDSPath, read_raw_bids
import mne

def load_eeg_epochs(bids_root: str,
                    subject_id: str,
                    task: str = 'rates',
                    tmin: float = -0.001,
                    tmax: float = 0.015,
                    l_freq: float = 100,
                    h_freq: float = 3000,
                    reject: dict = {'eeg': 30e-6}) -> mne.Epochs:
    """
    Load and preprocess EEG epochs from a BIDS dataset.

    Returns:
      epochs: mne.Epochs object with bandpass filter applied.
    """
    bids_path = BIDSPath(root=bids_root,
                         subject=subject_id,
                         task=task,
                         datatype='eeg')
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()
    raw.set_montage('standard_1020')

    # extract events
    events, _ = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(raw, events, event_id=None,
                        tmin=tmin, tmax=tmax,
                        baseline=(None, 0),
                        reject=reject,
                        preload=True,
                        verbose=False)
    # bandpass filter
    epochs.filter(l_freq, h_freq, method='fir')
    return epochs


def load_audio_file(path: str,
                    sr: float = None,
                    mono: bool = True) -> tuple[np.ndarray, float]:
    """
    Load a WAV/FLAC audio file.

    Returns:
      audio: ndarray (samples,)
      sr:    sampling rate
    """
    # prefer librosa for flexibility
    audio, rate = librosa.load(path, sr=sr, mono=mono)
    return audio, rate


def load_hrir(sofa_path: str,
              receiver: int = 0,
              channel: int = 0) -> tuple[np.ndarray, float]:
    """
    Load a time-domain HRIR from a SOFA file.

    Returns:
      hrir: 1D impulse response (samples)
      fs:   sampling rate (Hz)
    """
    sofa = SOFAFile(sofa_path, mode='r')
    hrirs = sofa.getDataIR()  # shape: (n_dirs, n_samples, n_channels)
    fs = float(sofa.getSamplingRate())
    return hrirs[receiver, :, channel], fs


def load_hrtf(sofa_path: str,
              receiver: int = 0,
              channel: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an HRTF (frequency response) from a SOFA file.

    Returns:
      H:    complex frequency response (n_bins,)
      freqs: frequency vector (Hz)
    """
    hrir, fs = load_hrir(sofa_path, receiver, channel)
    # FFT
    N = len(hrir)
    H = np.fft.rfft(hrir)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    return H, freqs
