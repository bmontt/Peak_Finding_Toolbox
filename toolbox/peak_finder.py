"""
abr_toolbox.peak_finder

Adaptive and classical peak detection for ABR/audio signals, following the original ABR pipeline:
    1) Predict anchor peak index (Wave V) using adaptive prominence
    2) Gaussian smoothing of the full waveform scaled by noise level
    3) Window from 0.5 ms before anchor to end for subsequent peaks
    4) Detect peaks with SNR-adaptive prominence and minimum distance
    5) Ensure inclusion of anchor and return first n_peaks sorted
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def compute_snr(data: np.ndarray,
                times_ms: np.ndarray,
                signal_window: tuple = (0, 10),
                noise_window: tuple = (-5, 0)) -> float:
    """
    Compute SNR in dB using RMS within signal vs. noise windows.
    """
    sig_mask = (times_ms >= signal_window[0]) & (times_ms <= signal_window[1])
    noi_mask = (times_ms >= noise_window[0]) & (times_ms <= noise_window[1])
    sig_rms = np.sqrt(np.mean(data[sig_mask] ** 2))
    noi_rms = np.sqrt(np.mean(data[noi_mask] ** 2))
    if noi_rms == 0:
        return np.inf
    return 20 * np.log10(sig_rms / noi_rms)


def normalize_snr(snr_db: float,
                    snr_min: float = -5,
                    snr_max: float = 10) -> float:
    """
    Clip and normalize SNR (dB) to [0,1].
    """
    snr_clipped = np.clip(snr_db, snr_min, snr_max)
    return (snr_clipped - snr_min) / (snr_max - snr_min)


def scale_prominence(snr: float,
                    base: float = 0.008,
                    high: float = 0.025,
                    curve: str = 'sqrt') -> float:
    """
    Map normalized SNR [0,1] to peak prominence threshold.
    """
    snr = np.clip(snr, 0, 1)
    if curve == 'sqrt':
        return base + (high - base) * (1 - np.sqrt(snr))
    if curve == 'linear':
        return base + (high - base) * (1 - snr)
    return base + (high - base) * (1 - snr ** 1.5)


def predict_anchor_latency(data: np.ndarray,
                            times_ms: np.ndarray,
                            snr_norm: float,
                            window: tuple = (5, 9)) -> int:
    """
    Predict anchor peak (Wave V) index using adaptive prominence.
    """
    prom = scale_prominence(snr_norm)
    mask = (times_ms >= window[0]) & (times_ms <= window[1])
    window_data = data[mask]
    peaks, _ = find_peaks(window_data, prominence=prom)
    if peaks.size:
        return np.where(mask)[0][peaks[0]]
    return np.where(mask)[0][np.argmax(window_data)]


def detect_peaks(
    data: np.ndarray,
    times_ms: np.ndarray,
    anchor_idx: int,
    n_peaks: int = 5,
    snr_norm: float = 0.5,
    base_sigma: float = 1.0
) -> np.ndarray:
    """
    Detect peaks following the original ABR pipeline.
    Args:
        data: 1D waveform in microvolts.
        times_ms: time vector in milliseconds.
        anchor_idx: index of Wave V anchor.
        n_peaks: number of peaks to return (I-V).
        snr_norm: normalized SNR [0,1].
        base_sigma: maximum Gaussian smoothing sigma.
    Returns:
        Sorted array of global peak indices (up to n_peaks).
    """
    # 1) Adaptive smoothing: more smoothing for noisier signals
    sigma = base_sigma * (1 - snr_norm)
    if sigma > 0:
        smoothed = gaussian_filter1d(data, sigma=sigma)
    else:
        smoothed = data.copy()

    # 2) Define analysis window: 0.5 ms before anchor to end
    dt = times_ms[1] - times_ms[0]
    half_samples = int(0.5 / dt)
    start_idx = max(anchor_idx - half_samples, 0)
    mask = np.zeros_like(data, dtype=bool)
    mask[start_idx:] = True
    window_data = smoothed[mask]

    # 3) Peak detection with adaptive prominence
    prom = scale_prominence(snr_norm)
    min_dist = half_samples
    peaks_rel, _ = find_peaks(window_data, prominence=prom, distance=min_dist)
    peaks_global = np.where(mask)[0][peaks_rel]

    # 4) Ensure anchor inclusion and sort
    if anchor_idx not in peaks_global:
        peaks_global = np.append(peaks_global, anchor_idx)
    peaks_global = np.unique(peaks_global)
    return np.sort(peaks_global)[:n_peaks]
