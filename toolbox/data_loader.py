"""
abr_toolbox.data_loader

Functions to load EEG (ABR) and audio data into standardized NumPy/MNE formats.
"""
import os
from typing import List, Union, Optional
import numpy as np
from mne_bids import BIDSPath, read_raw_bids
import mne
import librosa

def load_eeg_bids(root: str,
                   subject: str,
                   task: str = 'rates',
                   session: Optional[str] = None,
                   run: Optional[str] = None,
                   datatype: str = 'eeg',
                   montage: str = 'standard_1020',
                   invert_channel: Optional[int] = None,
                   verbose: bool = False) -> mne.io.BaseRaw:
    """
    Load raw EEG data from a BIDS directory for a given subject and task.

    Parameters
    ----------
    root : str
        Path to BIDS root directory.
    subject : str
        Subject identifier (e.g., '01').
    task : str
        Task label in BIDS (default: 'rates').
    session : Optional[str]
        Session label if applicable.
    run : Optional[str]
        Run identifier if applicable.
    datatype : str
        BIDS datatype (default: 'eeg').
    montage : str
        MNE montage name for channel locations.
    invert_channel : Optional[int]
        If set, invert data on specified channel index.
    verbose : bool
        Verbosity flag passed to `read_raw_bids`.

    Returns
    -------
    raw : mne.io.BaseRaw
        Loaded and prepped raw EEG data.
    """
    bids_path = BIDSPath(root=root,
                         subject=subject,
                         session=session,
                         task=task,
                         run=run,
                         datatype=datatype)
    raw = read_raw_bids(bids_path=bids_path, verbose=verbose)
    raw.load_data()
    if invert_channel is not None:
        raw.apply_function(lambda x: -x, picks=[invert_channel])
    raw.set_montage(montage)
    return raw


def load_audio_file(filepath: str,
                    sr: Optional[int] = None,
                    mono: bool = True) -> tuple[np.ndarray, int]:
    """
    Load an audio file into a NumPy array using librosa.

    Parameters
    ----------
    filepath : str
        Path to audio file (wav, flac, etc.).
    sr : Optional[int]
        Sampling rate to resample audio. Default None (native).
    mono : bool
        Convert to mono if True.

    Returns
    -------
    y : np.ndarray
        Audio time series.
    sr_out : int
        Sampling rate.
    """
    y, sr_out = librosa.load(filepath, sr=sr, mono=mono)
    return y, sr_out
