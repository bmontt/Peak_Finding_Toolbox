# Peak-Finding and Analysis Toolkit for Auditory Brainstem Responses, Head-Related Transfer Functions, and General Acoustics

This repository is a **standalone Python package** providing tools to detect peaks in ABR waveforms, HRIR impulse responses, and general audio signals with adaptive algorithms, CLI support, and extensible APIs.

---

## 1. Project Structure
```text
Peak_Finding_Toolbox/
├── README.md                # This overview, installation, quickstart
├── pyproject.toml           # Packaging & build configuration
├── requirements-dev.txt     # Development and testing dependencies
├── toolbox/                 # Main Python package
│   ├── data_loader.py       # BIDS/MNE EEG and audio imports (WAV/FLAC, SOFA)
│   ├── peak_finder.py       # Classical & adaptive peak detection algorithms
│   ├── plotting.py          # Abstraction over matplotlib for ABR/HRIR plots
│   └── main.py              # CLI entry‑point for abr/hrir commands
├── examples/                # Jupyter notebooks demonstrating usage
│   ├── abr_analysis.ipynb   # End‑to‑end ABR workflow example
│   └── audio_onsets.ipynb   # General audio onset/peak detection example
├── tests/                   # pytest test suite
│   ├── test_data_loader.py  # Tests for audio/HRIR loaders
│   ├── test_peak_finder.py  # Algorithm correctness tests
│   └── test_cli.py          # CLI command tests (abr + hrir)
└── ci/                      # GitHub Actions workflows
    └── python-package.yml
```

---

## 2. Installation

From the project root, create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate        # or `venv\Scripts\activate` on Windows
pip install -e .                # installs package and core deps from pyproject.toml
pip install -r requirements-dev.txt  # installs pytest, h5py, soundfile, click, etc.
```

---

## 3. Quickstart

### Command-Line Usage

Run the CLI directly via module invocation:

```bash
# Show ABR command help
python -m toolbox.main abr --help

# Detect peaks in BIDS EEG data (ABR) for subject '01'
python -m toolbox.main abr /path/to/bids_dataset 01 --mode individual --sigma 0.06 --outdir results/abr

# Show HRIR command help
python -m toolbox.main hrir --help

# Label peaks & troughs in an HRIR SOFA file
python -m toolbox.main hrir data/sofa/CIPIC_subject001.sofa --receiver 0 --channel 1 --n_peaks 5 --outdir results/hrir
```

### Python API

Use the core loader and peak‑finder functions in your scripts:

```python
import numpy as np
from toolbox.data_loader import load_hrir_from_sofa
from toolbox.peak_finder import detect_peaks

# 1) Load a SOFA HRIR
hrir, fs = load_hrir_from_sofa('data/sofa/CIPIC_subject001.sofa', channel='left')

# 2) Build time vector in ms
times_ms = np.arange(len(hrir)) / fs * 1000

# 3) Detect the first 5 peaks with base sigma 1.0 ms
peaks = detect_peaks(hrir, times_ms, n_peaks=5, base_sigma=1.0)
print("Peak latencies (ms):", times_ms[peaks])
```

---

## 4. Testing

Run the full test suite locally with pytest:

```bash
pytest --maxfail=1 --disable-warnings -q
```

Coverage includes:

- **Data loaders**: WAV/FLAC audio, SOFA HRIR files via h5py
- **Peak‑finder**: SNR computation, adaptive smoothing, multi‑peak detection
- **CLI**: sanity checks that `abr` and `hrir` commands parse args and write CSV/plots

---

## 5. Core Functionality

- **Data Loading** (`toolbox/data_loader.py`):
  - `load_eeg_epochs`: BIDS/MNE EEG epochs for ABR analysis
  - `load_audio_file`: WAV/FLAC import with optional resampling & mono conversion
  - `load_hrir_from_sofa` / `load_hrtf_from_sofa`: Direct SOFA file parsing via h5py

- **Peak Detection** (`toolbox/peak_finder.py`):
  - `compute_snr_normalized`: SNR estimation over defined time windows
  - `predict_wave_V_latency`: Anchor peak (Wave V) prediction using adaptive smoothing
  - `detect_peaks`: Multi‑peak detection with SNR‑scaled prominence & distance constraints

- **Plotting** (`toolbox/plotting.py`):
  - `plot_abr`: ABR waveform + peak annotations + reference lines
  - `plot_hrir`: HRIR impulse response + peak/trough markers

- **CLI Entry‑point** (`toolbox/main.py`):
  - `abr` subcommand for batch ABR processing
  - `hrir` subcommand for HRIR peak/trough labeling

---

*For questions, contributions, or to report issues, please open an issue on GitHub.*
