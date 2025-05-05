# Peak\_Finding\_Toolbox

A Python toolkit for detecting and visualizing peaks in electrophysiological and audio signals.

**Supported Use Cases**

* **ABR (Auditory Brainstem Response)**: Detect clinical waves I–V in averaged EEG/ABR recordings, with QC checks and normative inter-peak windows.
* **HRIR/HRTF (Head-Related Impulse Response/Function)**: Load SOFA-format spatial audio files, detect arrival peaks and troughs, and annotate spatial measurement positions.
* **General Audio**: Lightweight peak detection on arbitrary waveforms (e.g., percussion loops, transient sounds) for onset detection, silence trimming, and event counting. Includes interactive scrolling plots.

---
## Quickstart

```bash
# Clone the repository and install
git clone https://github.com/bmontt/Peak_Finding_Toolbox.git
cd Peak_Finding_Toolbox
pip install -e .

# ABR example
python -m toolbox.main abr /path/to/bids subject01 \
    --mode average --sigma 0.06 --outdir results/abr/

# HRIR example
python -m toolbox.main hrir path/to/file.sofa \
    --receiver 0 --channel 0 --n_peaks 5 --sigma 1.0 \
    --outdir results/hrir/ --show

# General audio example
python -m toolbox.main audio \
    examples/data/audio_samples/percussion_loop.wav \
    --sr 44100 --n-peaks 15 --sigma 1.0 --window-width 100 \
    --outdir results/audio/ --show
```

---
## Installation

```bash
# Clone the repository
git clone https://github.com/bmontt/Peak_Finding_Toolbox.git
cd Peak_Finding_Toolbox

# Core install (editable mode)
pip install -e .

# Optional: development tools
tools/requirements-dev.txt contains Jupyter, testing, and linting dependencies:
# pip install -r requirements-dev.txt
```

---

## Project Structure

```text
Peak_Finding_Toolbox/
├─ toolbox/                # core package
│  ├─ data_loader.py       # audio, EEG (BIDS/MNE), SOFA loaders
│  ├─ peak_finder.py       # unified API: detect_peaks, detect_peaks_abr, label_hrir_peaks
│  ├─ plotting.py          # plot_abr, plot_hrir, plot_waveform, scroll_plot
│  └─ main.py              # CLI entry-point (abr, hrir, audio subcommands)
├─ examples/               # interactive Jupyter notebooks
│  ├─ abr_example.ipynb    # ABR end-to-end detection, QC, plotting
│  ├─ hrir_example.ipynb   # SOFA HRIR peak/trough labeling + scroll plotting
│  └─ audio_example.ipynb  # Raw waveform peak detection, trimming, event counting
├─ tests/                  # pytest unit tests for data loaders, peak-finder, CLI
├─ bench/                  # benchmarking scripts
├─ comparison_scripts/     # external-tool comparisons (e.g. ABRPresto)
├─ data/ & results/        # placeholders for sample data and output artifacts
├─ pyproject.toml          # packaging & runtime dependencies
├─ requirements-dev.txt    # development dependencies (Jupyter, pytest, flake8, etc.)
└─ README.md               # (this file)
```

---

## Key Algorithms

### `detect_peaks` (General Audio & HRIR)

* Computes a normalized SNR profile to set an adaptive prominence threshold for peak detection.
* Enforces a minimum inter-peak distance based on signal mode (`audio` vs. `hrir`).
* Finds the top‑N most salient peaks (or troughs when inverting the signal) using SciPy’s `find_peaks`.
* Returns integer indices of detected events for downstream analysis or plotting.

### `detect_peaks_abr` (ABR)

* Applies adaptive Gaussian smoothing informed by local SNR to emphasize genuine ABR components.
* Iteratively searches for candidate peaks, gradually relaxing prominence and separation constraints to handle variability across subjects.
* Checks resulting inter-peak latencies against clinical normative ranges (I–III, III–V, I–V) to assign Waves I–V.
* Outputs peak indices for each wave and a QC flag indicating whether all latency criteria are met.

### `label_hrir_peaks`

* Loads HRIR data from SOFA format and prepares a time vector in milliseconds.
* Uses `detect_peaks` on the impulse response (and its inverse) to find arrival peaks and troughs.
* Returns lists of (latency, amplitude) tuples for both peaks and troughs for convenient downstream use.

---

*Maintained by Brody Montag. Feel free to open issues or PRs on GitHub.*
