import numpy as np
import pytest
# For HRIR tests, use a small SOFA file from pyroomacoustics
from pyroomacoustics.datasets import SOFADatabase
from toolbox.data_loader import load_audio_file, load_hrir, load_hrtf


def test_load_audio_file(tmp_path):
    # generate a short sine wave and write to WAV
    sr = 8000
    t = np.linspace(0, 1, sr, endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t)
    path = tmp_path / 'test.wav'
    import soundfile as sf
    sf.write(str(path), sine, sr)

    audio, rate = load_audio_file(str(path), sr=None, mono=True)
    assert rate == sr
    assert isinstance(audio, np.ndarray)
    # first few samples match
    np.testing.assert_allclose(audio[:10], sine[:10], rtol=1e-3, atol=1e-4)

def test_load_hrir_and_hrtf():
    db = SOFADatabase()
    sofa_path = db.list()[0]

    hrir, fs = load_hrir(sofa_path)
    assert isinstance(hrir, np.ndarray)
    assert isinstance(fs, float)
    assert hrir.ndim == 1

    H, freqs = load_hrtf(sofa_path)
    assert isinstance(H, np.ndarray) and isinstance(freqs, np.ndarray)
    assert H.shape == freqs.shape