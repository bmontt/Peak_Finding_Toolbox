# ABR Toolbox Repository Deliverable Outline

This repository will be a **standalone Python package** for ABR analysis, built around your existing code, and designed for extensibility and ease of use by future PIRL researchers.

---

## 1. Project Structure
```
abr_toolbox/
├── README.md                # Overview, installation, quickstart
├── setup.py / pyproject.toml # Packaging metadata
├── requirements.txt         # Core dependencies
├── abr_toolbox/             # Main package
│   ├── __init__.py
│   ├── data_loader.py       # BIDS/MNE EEG and audio imports
│   ├── preprocess.py        # Filtering, baseline correction, smoothing
│   ├── peak_finder.py       # Classical and adaptive peak detection algorithms
│   ├── cnn_predictor.py     # Stub for CNN model integration
│   ├── visualizer.py        # Interactive waveform & stats plots
│   ├── utils.py             # Shared helpers (SNR, normalization)
│   └── cli.py               # Command‐line interface via click or argparse
├── examples/                # Jupyter Notebooks and scripts
│   ├── abr_analysis.ipynb   # End‐to‐end ABR workflow example
│   ├── audio_onsets.ipynb   # Audio onset detection example
│   └── cnn_integration.ipynb# How to plug in a pretrained model
├── docs/                    # Sphinx documentation source
│   ├── conf.py
│   ├── index.rst
│   └── tutorials/           # Step‐by‐step tutorials
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

---

## 3. Interactive Tools

- **Jupyter notebooks** in `examples/`:
  - Guide new users through ABR analysis steps
  - Show how to tune parameters and visualize results

- **Tutorials** in `docs/tutorials/`:
  - Build Sphinx‐hosted documentation with code snippets
  - Deploy ReadTheDocs site for quick reference

---

## 4. Extensibility & Maintenance

- **Plugin architecture**:
  - Allow third‐party peak detection algorithms via entry points
  - Register additional visualizations or data sources

- **Testing & CI**:
  - Unit tests with pytest covering synthetic and real ABR data
  - GitHub Actions for linting, testing on push/PR

- **Packaging & Distribution**:
  - PyPI publication for easy `pip install abr_toolbox`
  - Docker image for one‐step environment setup

---

## 5. Timeline & Milestones

1. **Week 1:** Repo scaffolding, core modules (`data_loader`, `preprocess`, `peak_finder`) and basic tests.
2. **Week 2:** CLI implementation and demonstration notebooks.
3. **Week 3:** Visualization module and Sphinx docs.
4. **Week 4:** CNN stub, plugin hooks, final testing, CI, Dockerfile.
5. **Week 5:** Documentation polish, PyPI release, lab handoff.

---

**Outcome:** A documented, tested, and Dockerized open‐source package that lab members can install, extend, and use for ABR research and beyond.

