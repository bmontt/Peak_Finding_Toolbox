import os
import click
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import toolbox.main as main_mod

@pytest.fixture(autouse=True)
def stub_pipeline(monkeypatch, tmp_path):
    """
    Stub out the heavy lifting inside abr and hrir commands so they create known outputs.
    """
    # Stub load_eeg_epochs → return a DummyEvoked with fixed data
    class DummyEvoked:
        def __init__(self):
            self.data = np.zeros((1, 100))
            self.times = np.linspace(0, .015, 100)
            self.ch_names = ['Ch1']
        def average(self):
            return self

    def fake_load_eeg_epochs(bids_root, subject_id):
        return DummyEvoked()

    monkeypatch.setattr(main_mod, 'load_eeg_epochs', fake_load_eeg_epochs)

    # Stub plot_abr to create a dummy file
    def stub_plot_abr(times_ms, data_arr, peaks_per_channel, ch_names, snr_list, subject_id, mode, outdir, **kwargs):
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"{subject_id}_abr_plot.png")
        open(path, 'wb').close()
        return path
    monkeypatch.setattr(main_mod, 'plot_abr', stub_plot_abr)

    # Stub load_hrir and label_hrir_peaks for HRIR CLI
    monkeypatch.setattr(main_mod, 'load_hrir', lambda sofa_path, receiver, channel: (np.zeros(50), 44100))
    monkeypatch.setattr(main_mod, 'label_hrir_peaks', lambda *args, **kwargs: {
        'peaks':   [(1.0, 0.5), (2.0, 0.4)],
        'troughs': [(1.5, -0.2), (2.5, -0.1)]
    })

    # Stub plot_hrir to create a dummy file
    def stub_plot_hrir(times_ms, hrir_data, peaks, troughs, base, outdir, **kwargs):
        os.makedirs(outdir, exist_ok=True)
        filename = f"{base}_hrir_plot.png"
        path = os.path.join(outdir, filename)
        open(path, 'wb').close()
        return path
    monkeypatch.setattr(main_mod, 'plot_hrir', stub_plot_hrir)

    yield


def test_abr_command_creates_csv(tmp_path):
    runner = CliRunner()
    # Create a dummy BIDS root directory so click.Path(exists=True) passes
    bids_root = tmp_path / "dummy_bids"
    bids_root.mkdir()
    outdir = str(tmp_path / "out_abr")
    result = runner.invoke(
        main_mod.main,
        ['abr', str(bids_root), 'subj1', '--mode', 'individual', '--sigma', '0.1', '--outdir', outdir]
    )
    assert result.exit_code == 0, result.output

    # Verify CSV
    csv = tmp_path / "out_abr" / "subj1_abr_waves.csv"
    assert csv.exists(), f"Missing CSV at {csv}"
    df = pd.read_csv(csv)
    assert list(df.columns) == ['Subject','Channel','Wave','Amplitude (µV)','Latency (ms)','SNR']
    assert (df.Subject == 'subj1').all()
    assert (df.Channel == 'Ch1').all()
    assert df.shape[0] == 5


def test_hrir_command_creates_csv_and_plot(tmp_path):
    runner = CliRunner()
    outdir = str(tmp_path / "out_hrir")
    sofa = tmp_path / "dummy.sofa"
    sofa.write_bytes(b"")  # dummy file

    result = runner.invoke(
        main_mod.main,
        ['hrir', str(sofa), '--receiver', '0', '--channel', '1', '--n_peaks', '2', '--sigma', '0.5', '--outdir', outdir]
    )
    assert result.exit_code == 0, result.output

    # Verify HRIR CSV
    csv = tmp_path / "out_hrir" / "dummy_hrir.csv"
    assert csv.exists(), f"Missing HRIR CSV at {csv}"
    df = pd.read_csv(csv)
    assert sorted(df.Type.unique()) == ['Peak','Trough']
    assert df.shape[0] == 4

    # Verify HRIR plot file
    png_files = list((tmp_path / "out_hrir").glob("*_hrir_plot.png"))
    assert png_files, "Expected HRIR plot file"
