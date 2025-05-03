import numpy as np
import pytest
import os
import requests


from toolbox.data_loader import (
    load_audio_file,
    list_sofa_files,
    load_hrir_from_sofa,
    load_hrtf_from_sofa,
)

# -- Fixture to download a couple of headphone SOFAs once per test session --
@pytest.fixture(scope="session")
def sofa_dir(tmp_path_factory):
    urls = [
        "https://sofacoustics.org/data/headphones/ari/hpir_SennheiserHD650_nh830.sofa",
        "https://sofacoustics.org/data/headphones/ari/hpir_SennheiserHD650_nh831.sofa",
    ]
    out = tmp_path_factory.mktemp("sofa")
    for url in urls:
        fname = url.split("/")[-1]
        resp = requests.get(url)
        resp.raise_for_status()
        (out / fname).write_bytes(resp.content)
    return str(out)

def test_load_audio_file(tmp_path):
    # generate a short sine wave and write to WAV
    sr = 8000
    t = np.linspace(0, 1, sr, endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "test.wav"
    import soundfile as sf
    sf.write(str(path), sine, sr)

    audio, rate = load_audio_file(str(path), sr=None, mono=True)
    assert rate == sr
    assert isinstance(audio, np.ndarray)
    np.testing.assert_allclose(audio[:10], sine[:10], rtol=1e-3, atol=1e-4)

def test_load_hrir_and_hrtf(sofa_dir):
    # list the downloaded SOFAs
    sofa_files = list_sofa_files(sofa_dir)
    assert len(sofa_files) >= 2

    sofa_path = sofa_files[0]
    hrir, fs = load_hrir_from_sofa(sofa_path, channel="left")
    assert isinstance(hrir, np.ndarray)
    assert isinstance(fs, (int, float))
    assert hrir.ndim == 1

    H, freqs = load_hrtf_from_sofa(sofa_path, channel="left")
    assert isinstance(H, np.ndarray) and isinstance(freqs, np.ndarray)
    assert H.shape == freqs.shape
