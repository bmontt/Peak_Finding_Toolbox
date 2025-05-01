import numpy as np
from CMSC499A.abr_spring.ABR_Toolbox.toolbox.abr_peak_finder import (
    compute_snr, normalize_snr, scale_prominence,
    predict_anchor_latency, detect_peaks
)

def test_compute_snr_inf_on_zero_noise():
    data = np.zeros(100)
    times_ms = np.linspace(-10, 50, 100)
    snr = compute_snr(data, times_ms, signal_window=(0, 10), noise_window=(-5, 0))
    assert snr == np.inf, "Zero noise floor should return infinite SNR"

def test_normalize_snr_bounds():
    assert normalize_snr(-5) == 0.0
    assert normalize_snr(10) == 1.0
    # clipped midpoint
    assert np.isclose(normalize_snr(2.5), (2.5 + 5) / 15)

def test_scale_prominence_extremes():
    # With SNR=1 → sqrt curve → base
    base = 0.008
    high = 0.025
    assert np.isclose(scale_prominence(1.0, base, high), base)
    # With SNR=0 → sqrt curve → high
    assert np.isclose(scale_prominence(0.0, base, high), high)

def test_predict_anchor_latency_simple_peak():
    # Create a triangular peak at ~7 ms
    times_ms = np.linspace(0, 20, 201)
    data = np.exp(-((times_ms - 7)/1.0)**2)
    # Low noise → high prominence → picks the obvious peak
    idx = predict_anchor_latency(data, times_ms, snr_norm=1.0, window=(5, 9))
    assert 5 <= times_ms[idx] <= 9

def test_detect_peaks_returns_n_peaks():
    # Synthetic: peaks at 6, 8, 10 ms
    times_ms = np.linspace(0, 20, 201)
    data = np.zeros_like(times_ms)
    for t in [6, 8, 10]:
        data[np.argmin(np.abs(times_ms - t))] = 1.0
    # Anchor at 8 ms, expect at least those three peaks
    anchor_idx = np.argmin(np.abs(times_ms - 8))
    peaks = detect_peaks(data, times_ms, anchor_idx, n_peaks=3, snr_norm=1.0)
    latencies = times_ms[peaks]
    assert set(np.round(latencies)) == {6, 8, 10}
