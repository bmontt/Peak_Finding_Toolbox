import numpy as np
import pytest

from toolbox.peak_finder import (
    compute_snr_normalized,
    detect_peaks,
    scale_prominence,
)

def test_scale_prominence_bounds():
    # snr=0 → max base prominence
    p0 = scale_prominence(0.0)
    assert pytest.approx(p0, rel=1e-3) == 0.01 + (0.1 - 0.01) * (1 - 0**3)

    # snr=1 → min prominence = base
    p1 = scale_prominence(1.0)
    assert pytest.approx(p1, rel=1e-3) == 0.01

def test_compute_snr_normalized_simple():
    # data: flat noise=1 outside, signal=2 inside
    times = np.linspace(-5, 10, 16)  # -5…10 ms
    data = np.ones_like(times)
    data[(times >= 0) & (times <= 10)] = 2.0
    snr = compute_snr_normalized(data, times)
    # RMS_signal = 2, RMS_noise = 1 → 6.02 dB, clipped to [-5,10]
    assert 0 < snr < 1

def test_detect_peaks_handles_low_snr():
    # flat noise only → no real peaks, anchor at max abs location (middle)
    times = np.linspace(0, 15, 1501)
    data = 0.01 * np.random.randn(len(times))
    peaks = detect_peaks(data, times, n_peaks=2, base_sigma=1.0)
    # returns at most 2, but may be empty or same index
    assert len(peaks) <= 2
    assert all(0 <= p < len(times) for p in peaks)
