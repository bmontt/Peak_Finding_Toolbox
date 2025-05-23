import numpy as np
import glob
import os
import mne
import soundfile as sf
import h5py
from typing import Tuple, Union

from mne_bids import BIDSPath, read_raw_bids
from mne import create_info, pick_types, read_events, events_from_annotations, Epochs
import zarr
    
def load_data(source: str,
              subject_id: str = None,
              task: str = 'rates',
              tmin: float = -0.001,
              tmax: float = 0.015,
              l_freq: float = 100,
              h_freq: float = 3000,
              reject: dict = None,
              sr: Union[int, None] = None,
              mono: bool = True,
              channel: str = 'left'
             ) -> Union[Epochs, Tuple]:
    """
    Detects source type by path/extension and routes to the correct loader.
    Returns either an mne.Epochs (for BIDS EEG) or (array, fs) for audio/HRIR/HRTF/Zarr.
    """
    ext = os.path.splitext(source)[1].lower()
    # directory → BIDS EEG or Zarr
    if os.path.isdir(source):
        if source.endswith('.zarr') or os.path.exists(os.path.join(source, '.zarray')):
            z = zarr.open(source, mode='r')
            data = z['abr_waveform'][:]
            fs   = z.attrs.get('sampling_rate', 48000)
            return data, fs
        elif os.path.exists(os.path.join(source, 'dataset_description.json')):
            return load_eeg_epochs(
                bids_root=source,
                subject_id=subject_id,
                task=task,
                tmin=tmin,
                tmax=tmax,
                l_freq=l_freq,
                h_freq=h_freq,
                reject=reject or {'eeg': 30e-6}
            )
    # file → audio, SOFA, etc.
    if ext in ('.wav', '.flac', '.mp3', '.ogg'):
        return load_audio_file(source, sr=sr, mono=mono)
    if ext == '.sofa':
        # decide IR vs HRTF by caller preference
        if 'frequency' in source or 'hrtf' in source:
            return load_hrtf_from_sofa(source, channel=channel)
        return load_hrir_from_sofa(source, channel=channel)

    raise ValueError(f"Unrecognized data source or type: {source}")


#####
"""helper functions for loading EEG and audio data"""
#####
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

    # extract events and epoch
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


def load_audio_file(path: str, sr: Union[int, None] = None, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load an audio file via soundfile, optionally resample and convert to mono.

    Returns:
      audio: ndarray of samples
      rate:  sampling rate in Hz
    """
    audio, rate = sf.read(path, always_2d=True)
    # convert to mono if requested
    if mono and audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    # resample if needed
    if sr is not None and sr != rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=rate, target_sr=sr)
        rate = sr
    return audio, rate


def list_sofa_files(sofa_dir: str) -> list[str]:
    """
    Return a sorted list of .sofa files in the given directory.
    """
    pattern = os.path.join(sofa_dir, '*.sofa')
    return sorted(glob.glob(pattern))


def load_hrir_from_sofa(sofa_path: str, channel: str = 'left') -> Tuple[np.ndarray, float]:
    """
    Load a time-domain HRIR from a SOFA file.

    Args:
      sofa_path: path to the SOFA file
      channel:   'left' or 'right'

    Returns:
      hrir: 1D impulse response (samples,)
      fs:   sampling rate (Hz)
    """
    with h5py.File(sofa_path, 'r') as f:
        fs = float(f['Data.SamplingRate'][()])
        ir = f['Data.IR'][()]
        ch_idx = 0 if channel.lower() == 'left' else 1
        # take first measurement and specified channel
        hrir = ir[0, ch_idx, :]
    return hrir, fs


def load_hrtf_from_sofa(sofa_path: str, channel: str = 'left') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an HRTF (frequency response) from a SOFA file.

    Args:
      sofa_path: path to the SOFA file
      channel:   'left' or 'right'

    Returns:
      H:     complex frequency response (n_bins,)
      freqs: frequency vector (Hz)
    """
    hrir, fs = load_hrir_from_sofa(sofa_path, channel)
    N = len(hrir)
    H = np.fft.rfft(hrir)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    return H, freqs


# Backward compatibility
# TODO: standardize
load_hrir = load_hrir_from_sofa
load_hrtf = load_hrtf_from_sofa
