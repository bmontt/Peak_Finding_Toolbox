import numpy as np
import pytest
from toolbox.peak_finder import (
            compute_snr_normalized, scale_prominence, 
            predict_wave_V_latency, detect_peaks, label_hrir_peaks)

def test_compute_snr_normalized_signal_no_noise():
    # times from -5 to 10 ms
    times_ms = np.linspace(-5, 10, 3001)
    # noise window: [-5,0], signal window: [0,10]
    data = np.zeros_like(times_ms)
    data[times_ms >= 0] = 1.0

    snr = compute_snr_normalized(data, times_ms)
    # with infinite theoretical SNR (zero noise), normalized value should be 1.0
    assert pytest.approx(snr, rel=1e-3) == 1.0


def test_detect_peaks_two_gaussians():
    # synthetic waveform with two Gaussian peaks at ~5ms and ~10ms
    times_ms = np.arange(0, 15.1, 0.1)
    gauss1 = np.exp(-((times_ms - 5) ** 2) / (2 * 0.2 ** 2))
    gauss2 = 0.5 * np.exp(-((times_ms - 10) ** 2) / (2 * 0.2 ** 2))
    signal = gauss1 + gauss2

    peaks = detect_peaks(signal, times_ms, n_peaks=2, base_sigma=1.0)
    # verify that one peak is near 5 ms and another near 10 ms
    peak_times = times_ms[peaks]
    assert any(abs(pt - 5) < 0.5 for pt in peak_times)
    assert any(abs(pt - 10) < 0.5 for pt in peak_times)