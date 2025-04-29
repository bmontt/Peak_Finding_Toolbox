import numpy as np
import pytest
from toolbox.detectors import BaseDetector, _registry

def test_registry_keys():
    expected = {"abr_adaptive", "scipy_peak", "onset_env", "hrir_xcorr"}
    assert set(_registry.keys()) >= expected

def test_create_known():
    d = BaseDetector.create("scipy_peak", prominence=0.1, distance=0.001)
    assert d.__class__.__name__ == "ScipyPeakDetector"

def test_scipy_peak_detect():
    # create a signal with peaks at samples 100 and 300
    sig = np.zeros(500)
    sig[100] = 1.0
    sig[300] = 1.0
    det = BaseDetector.create("scipy_peak", prominence=0.5, distance=0.1)
    peaks = det.detect(sig, sr=1000)
    assert set(peaks) == {100, 300}

@pytest.mark.xfail
def test_hrir_placeholder():
    det = BaseDetector.create("hrir_xcorr")
    with pytest.raises(NotImplementedError):
        det.detect(np.zeros(100), sr=44100)
