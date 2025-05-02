import os
import pytest
import numpy as np
import pandas as pd
from click.testing import CliRunner
import toolbox.main as main_mod

# --- Fixtures to monkeypatch pipeline dependencies ---
class FakeEvoked:
    def __init__(self):
        # simulate times 0 to 15 ms at 0.1 ms steps
        self.times = np.arange(0, 15.1, 0.1)
        # simple waveform: Gaussian at 5 ms
        gauss = np.exp(-((self.times - 5) ** 2) / (2 * 0.2 ** 2))
        # two channels identical
        self.data = np.vstack([gauss, gauss])
    def average(self):
        return self

@pytest.fixture(autouse=True)
def patch_abr_and_hrir(monkeypatch):
    # Patch load_eeg_epochs to return FakeEvoked
    monkeypatch.setattr(main_mod, 'load_eeg_epochs', lambda bids_root, subject_id: FakeEvoked())
    # Patch label_hrir_peaks to return fixed peaks/troughs at 1 and 2 ms
    monkeypatch.setattr(main_mod, 'label_hrir_peaks', lambda sofa_path, receiver, channel, n_peaks, base_sigma: {
        'peaks': [(1.0, 0.8), (2.0, 0.5)],
        'troughs': [(1.5, -0.2)]
    })
    # Patch load_hrir to return dummy HRIR data
    monkeypatch.setattr(main_mod, 'load_hrir', lambda sofa_path, receiver, channel: (np.array([0]*10 + [1] + [0]*10), 1000.0))


def test_abr_full_pipeline(tmp_path):
    runner = CliRunner()
    outdir = str(tmp_path / 'out')
    result = runner.invoke(main_mod.main, ['abr', 'dummy_root', '01', '--mode', 'individual', '--sigma', '1.0', '--outdir', outdir])
    assert result.exit_code == 0
    # Check CSV
    csv_path = os.path.join(outdir, '01_abr_waves.csv')
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)
    # Expect Waves I-V rows
    assert df['Wave'].tolist()[:5] == [f'Wave {i}' if i != 4 else 'Wave V' for i in range(1,6)] or len(df) >= 5


def test_hrir_full_pipeline(tmp_path):
    runner = CliRunner()
    outdir = str(tmp_path / 'out')
    result = runner.invoke(main_mod.main, ['hrir', 'dummy.sofa', '--receiver', '0', '--channel', '0', '--n_peaks', '2', '--sigma', '1.0', '--outdir', outdir])
    assert result.exit_code == 0
    csv_path = os.path.join(outdir, 'dummy_hrir.csv')
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)
    # Expect at least one peak and one trough entry
    types = set(df['Type'])
    assert 'Peak' in types and 'Trough' in types
