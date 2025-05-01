"""
toolbox.peak_finder

Adaptive and classical peak detection for ABR/audio signals, following the original ABR pipeline:
    1) Predict anchor peak index (Wave V) using adaptive prominence
    2) Gaussian smoothing of the full waveform scaled by noise level
    3) Window from 0.5 ms before anchor to end for subsequent peaks
    4) Detect peaks with SNR-adaptive prominence and minimum distance
    5) Ensure inclusion of anchor and return first n_peaks sorted
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath, read_raw_bids
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm

# === Parameter Definitions ===

SNR_SIGNAL_WINDOW = (0, 10)
SNR_NOISE_WINDOW = (-5, 0)
SNR_DB_RANGE = (-5, 10)
WAVE_V_WINDOW = (4.6, 9.2)
PEAK_ANALYSIS_WINDOW = (0, 15)
MIN_PEAK_DISTANCE_MS = 0.7
PROMINENCE_BASE = 0.01
PROMINENCE_HIGH = 0.1
PROMINENCE_EXPONENT = 3

# === Toolbox Functions ===

def compute_snr_normalized(data: np.ndarray,
                times_ms: np.ndarray) -> float:
    signal_mask = (times_ms >= SNR_SIGNAL_WINDOW[0]) & (times_ms <= SNR_SIGNAL_WINDOW[1])
    noise_mask = (times_ms >= SNR_NOISE_WINDOW[0]) & (times_ms <= SNR_NOISE_WINDOW[1])
    signal_rms = np.sqrt(np.mean(data[signal_mask] ** 2))
    noise_rms = np.sqrt(np.mean(data[noise_mask] ** 2))
    snr_db = 20 * np.log10(signal_rms / noise_rms) if noise_rms > 0 else np.inf
    snr_clipped = np.clip(snr_db, *SNR_DB_RANGE)
    return (snr_clipped - SNR_DB_RANGE[0]) / (SNR_DB_RANGE[1] - SNR_DB_RANGE[0])

def scale_prominence(snr: float) -> float:
    snr = np.clip(snr, 0, 1)
    return PROMINENCE_BASE + (PROMINENCE_HIGH - PROMINENCE_BASE) * (1 - snr ** PROMINENCE_EXPONENT)

def predict_wave_V_latency(data: np.ndarray,
                            times_ms: np.ndarray,
                            snr_norm: float,
                            base_sigma: float) -> int:
    adj_sigma = base_sigma / (snr_norm + 0.1)
    adj_sigma = np.clip(adj_sigma, 0.3 * base_sigma, 4 * base_sigma)
    smoothed = gaussian_filter1d(data, sigma=adj_sigma)

    mask = (times_ms >= WAVE_V_WINDOW[0]) & (times_ms <= WAVE_V_WINDOW[1])
    segment = smoothed[mask]
    if segment.size == 0:
        return None

    prom = scale_prominence(snr_norm)
    peaks, props = find_peaks(segment, prominence=prom)
    if peaks.size:
        best = peaks[np.argmax(props['prominences'])]
        return np.where(mask)[0][best]
    else:
        rel = np.argmax(np.abs(segment))
        return np.where(mask)[0][rel]

def detect_peaks(data: np.ndarray,
                 times_ms: np.ndarray,
                 n_peaks: int = 5,
                 base_sigma: float = 1.0) -> np.ndarray:
    snr_norm = compute_snr_normalized(data, times_ms)
    anchor_idx = predict_wave_V_latency(data, times_ms, snr_norm, base_sigma)

    adj_sigma = base_sigma * (1 - snr_norm)
    smoothed = gaussian_filter1d(data, sigma=adj_sigma) if adj_sigma > 0 else data.copy()

    mask = (times_ms >= PEAK_ANALYSIS_WINDOW[0]) & (times_ms <= PEAK_ANALYSIS_WINDOW[1])
    windowed = smoothed[mask]
    dt = times_ms[1] - times_ms[0]
    min_dist = int(MIN_PEAK_DISTANCE_MS / dt)

    prom = scale_prominence(snr_norm)
    peaks_rel, _ = find_peaks(windowed, prominence=prom, distance=min_dist)
    peaks_global = np.where(mask)[0][peaks_rel]

    if anchor_idx is not None and anchor_idx not in peaks_global:
        peaks_global = np.sort(np.append(peaks_global, anchor_idx))
    else:
        peaks_global = np.sort(peaks_global)

    return peaks_global[:n_peaks]
