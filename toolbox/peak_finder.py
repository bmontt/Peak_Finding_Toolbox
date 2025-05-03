"""
toolbox.peak_finder

Adaptive and classical peak detection for ABR, HRIR, and general audio signals.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# === Parameter Definitions ===
# ABR-specific
ABR_SNR_SIGNAL_WINDOW = (0, 10)
ABR_SNR_NOISE_WINDOW = (-5, 0)
ABR_SNR_DB_RANGE = (-5, 10)
ABR_WAVE_V_WINDOW = (4.6, 9.2)
ABR_PEAK_WINDOW = (0, 15)
ABR_MIN_DISTANCE_MS = 0.7

# HRIR-specific
HRIR_PEAK_WINDOW_MS = 5.0
HRIR_MIN_DISTANCE_MS = 0.1

# Audio-specific: generic
AUDIO_MIN_DISTANCE_MS = 0.5

# Prominence settings
PROMINENCE_BASE = 0.01
PROMINENCE_HIGH = 0.1
PROMINENCE_EXPONENT = 3


def compute_snr_normalized(
    data: np.ndarray,
    times_ms: np.ndarray,
    signal_window: tuple = ABR_SNR_SIGNAL_WINDOW,
    noise_window: tuple = ABR_SNR_NOISE_WINDOW,
    db_range: tuple = ABR_SNR_DB_RANGE
) -> float:
    """
    Compute normalized SNR in [0,1] based on specified windows and dB range.
    """
    signal_mask = (times_ms >= signal_window[0]) & (times_ms <= signal_window[1])
    noise_mask = (times_ms >= noise_window[0]) & (times_ms <= noise_window[1])
    sig_rms = np.sqrt(np.mean(data[signal_mask]**2)) if signal_mask.any() else 0
    noise_rms = np.sqrt(np.mean(data[noise_mask]**2)) if noise_mask.any() else 0
    snr_db = 20*np.log10(sig_rms/noise_rms) if noise_rms>0 else np.inf
    snr_clip = np.clip(snr_db, *db_range)
    return (snr_clip - db_range[0])/(db_range[1]-db_range[0])


def scale_prominence(snr: float) -> float:
    """
    Map normalized SNR to a prominence threshold between base and high.
    """
    snr = np.clip(snr, 0, 1)
    return PROMINENCE_BASE + (PROMINENCE_HIGH - PROMINENCE_BASE)*(1 - snr**PROMINENCE_EXPONENT)


def _detect_peaks_abr(data, times_ms, n_peaks, base_sigma):
    # Anchor (Wave V) prediction
    snr = compute_snr_normalized(data, times_ms)
    adj_sig = base_sigma/(snr+0.1)
    adj_sig = np.clip(adj_sig, 0.3*base_sigma, 4*base_sigma)
    sm = gaussian_filter1d(data, sigma=adj_sig)

    mask_v = (times_ms>=ABR_WAVE_V_WINDOW[0]) & (times_ms<=ABR_WAVE_V_WINDOW[1])
    seg = sm[mask_v]
    prom = scale_prominence(snr)
    peaks_v, props = find_peaks(seg, prominence=prom)
    if peaks_v.size:
        best = peaks_v[np.argmax(props['prominences'])]
        anchor = np.where(mask_v)[0][best]
    else:
        anchor = np.where(mask_v)[0][np.argmax(np.abs(seg))] if seg.size else None

    # Full detection
    adj_full = base_sigma*(1 - snr)
    sm_full = gaussian_filter1d(data, sigma=adj_full) if adj_full>0 else data.copy()
    mask_full = (times_ms>=ABR_PEAK_WINDOW[0]) & (times_ms<=ABR_PEAK_WINDOW[1])
    seg_full = sm_full[mask_full]
    dt = times_ms[1]-times_ms[0]
    min_dist = int(ABR_MIN_DISTANCE_MS/dt)
    rel, _ = find_peaks(seg_full, prominence=prom, distance=min_dist)
    global_idx = np.where(mask_full)[0][rel]
    if anchor is not None and anchor not in global_idx:
        global_idx = np.sort(np.append(global_idx, anchor))
    else:
        global_idx = np.sort(global_idx)
    return global_idx[:n_peaks]


def _detect_peaks_hrir(data, times_ms, n_peaks, base_sigma):
    snr = compute_snr_normalized(data, times_ms)
    prom = scale_prominence(snr)
    max_i = np.searchsorted(times_ms, HRIR_PEAK_WINDOW_MS)
    seg = data[:max_i]
    dt = times_ms[1]-times_ms[0]
    min_dist = int(HRIR_MIN_DISTANCE_MS/dt)
    peaks, props = find_peaks(seg, prominence=prom, distance=min_dist)
    if len(peaks)>n_peaks:
        idx = np.argsort(props['prominences'])[::-1][:n_peaks]
        peaks = np.sort(peaks[idx])
    return peaks


def _detect_peaks_audio(data, times_ms, n_peaks, base_sigma):
    snr = compute_snr_normalized(data, times_ms)
    prom = scale_prominence(snr)
    dt = times_ms[1]-times_ms[0]
    min_dist = int(AUDIO_MIN_DISTANCE_MS/dt)
    peaks, props = find_peaks(data, prominence=prom, distance=min_dist)
    if len(peaks)>n_peaks:
        idx = np.argsort(data[peaks])[::-1][:n_peaks]
        peaks = np.sort(peaks[idx])
    return peaks


def detect_peaks(
    data: np.ndarray,
    times_ms: np.ndarray,
    n_peaks: int = 5,
    base_sigma: float = 1.0,
    mode: str = 'abr'
) -> np.ndarray:
    """
    Unified interface for peak detection.
    mode in {'abr','hrir','audio'}.
    """
    if mode == 'abr':
        return _detect_peaks_abr(data, times_ms, n_peaks, base_sigma)
    if mode == 'hrir':
        return _detect_peaks_hrir(data, times_ms, n_peaks, base_sigma)
    if mode == 'audio':
        return _detect_peaks_audio(data, times_ms, n_peaks, base_sigma)
    raise ValueError(f"Unknown mode '{mode}'")


def label_hrir_peaks(sofa_path, receiver=0, channel=0, n_peaks=5, base_sigma=1.0):
    # Backward compatibility
    from .data_loader import load_hrir
    hrir, fs = load_hrir(sofa_path, receiver, channel)
    times_ms = np.arange(len(hrir)) / fs * 1000
    peaks = detect_peaks(hrir, times_ms, n_peaks, base_sigma, mode='hrir')
    troughs = detect_peaks(-hrir, times_ms, n_peaks, base_sigma, mode='hrir')
    return {
        'peaks':   [(times_ms[i], hrir[i]) for i in peaks],
        'troughs': [(times_ms[i], -hrir[i]) for i in troughs]
    }
