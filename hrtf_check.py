import numpy as np
from toolbox.data_loader import load_hrir_from_sofa
from toolbox.peak_finder import detect_peaks
from toolbox.plotting import scroll_plot, plot_abr, plot_hrir

# 1) Load a SOFA HRIR
hrir, fs = load_hrir_from_sofa('tests/hrtf_data/sofa/hpir_SennheiserHD650_nh830.sofa', channel='left')

# 2) Build time vector in ms
times_ms = np.arange(len(hrir)) / fs * 1000

# 3) Detect the first 5 peaks with base sigma 1.0 ms
peaks = detect_peaks(hrir, times_ms, n_peaks=5, base_sigma=1.0, mode='hrir')
troughs = detect_peaks(-hrir, times_ms, n_peaks=5, base_sigma=1.0, mode='hrir')
# Plot
path = plot_hrir(
    times_ms,
    hrir,
    peaks,
    troughs,
    base='hpir_SennheiserHD650_nh830_2',
    outdir='results')

print("Peak latencies (ms):", times_ms[peaks])
print("Trough latencies (ms):", times_ms[troughs])