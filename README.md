# Peak-Finding and Analysis Toolkit for Auditory Brainstem Responses, Head-Related Transfer Functions, and General Acoustics

This repository is a **standalone Python package** for ABR, HRTF/HRIR, and general acoustic analysis designed for extensibility and ease of use.

---

## 1. Project Structure
```
abr_toolbox/
├── README.md                # Overview, installation, quickstart
├── setup.py                 # Packaging metadata
├── requirements.txt         # Core dependencies
├── toolbox/                 # Main package
│   ├── __init__.py
│   ├── data_loader.py       # BIDS/MNE EEG and audio imports
│   ├── preprocess.py        # Filtering, baseline correction, smoothing
│   ├── peak_finder.py       # Classical and adaptive peak detection algorithms
│   ├── cnn_predictor.py     # Stub for CNN model integration
│   ├── visualizer.py        # Interactive waveform & stats plots
│   ├── utils.py             # Shared helpers (SNR, normalization)
│   └── cli.py               # Command‐line interface via click or argparse
        detectors.py
├── examples/                # Jupyter Notebooks and scripts
│   ├── abr_analysis.ipynb   # End‐to‐end ABR workflow example
│   ├── audio_onsets.ipynb   # Audio onset detection example
│   └── cnn_integration.ipynb# How to plug in a pretrained model
├── docs/                    # Sphinx documentation source
│   ├── conf.py
│   ├── index.rst
├── tests/                   # pytest test suite
│   ├── test_preprocess.py
│   ├── test_peak_finder.py
│   └── test_visualizer.py
├── ci/                      # GitHub Actions workflows
│   └── python-package.yml
└── Dockerfile               # Containerized environment for reproducibility
```

---

## 2. Core Functionality

- **Data loading** via `abr_toolbox.data_loader`:
  - Read EEG‐BIDS datasets with MNE-BIDS
  - Support raw audio (WAV/FLAC) via librosa

- **Preprocessing** (`preprocess.py`):
  - Bandpass filtering (FIR/IIR)
  - Baseline correction with flexible windows
  - Gaussian smoothing with adaptive sigma

- **Peak finding** (`peak_finder.py`):
  - Classical: `scipy.signal.find_peaks` with parameter tuning
  - Adaptive: SNR‐based prominence and distance
  - Anchor prediction via placeholder CNN stub

- **CNN integration** (`cnn_predictor.py`):
  - Load a pretrained TensorFlow/PyTorch model for wave I/V latency
  - Inference wrapper matching input/output API

- **Visualization** (`visualizer.py`):
  - Matplotlib & interactive (Plotly or ipywidgets) waveform viewer
  - Overlay detected peaks and annotations
  - SNR and latency vs. rate/frequency scatter/line charts

- **CLI** (`cli.py`):
  - Single‐command batch processing of subject folders
  - Options for mode (average/subtract), output formats (PNG, CSV)
