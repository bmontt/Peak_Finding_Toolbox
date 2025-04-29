import numpy as np
from toolbox.preprocess import smooth_waveform

def test_smooth_waveform_constant():
    # A constant signal should remain unchanged by smoothing
    data = np.ones(100)
    smoothed = smooth_waveform(data, sigma=2.0)
    assert np.allclose(smoothed, 1.0), "Constant signal should not change"

def test_smooth_waveform_reduces_noise():
    # Smoothing should reduce high-frequency noise around a step
    rng = np.random.RandomState(0)
    data = np.concatenate([np.zeros(50), np.ones(50)]) + 0.1 * rng.randn(100)
    smoothed = smooth_waveform(data, sigma=2.0)
    # After smoothing, the difference between adjacent samples should be smaller
    diffs = np.abs(np.diff(smoothed))
    assert diffs.max() < 0.5, "Smoothing did not sufficiently reduce noise"
