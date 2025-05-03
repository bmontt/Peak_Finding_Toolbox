import numpy as np
from toolbox.data_loader import load_hrir_from_sofa
from toolbox.peak_finder import detect_peaks
from toolbox.plotting import scroll_plot

# 1) Load a SOFA HRIR
hrir, fs = load_hrir_from_sofa('tests/fixtures/sofa/hpir_SennheiserHD650_nh830.sofa', channel='left')

# 2) Build time vector in ms
times_ms = np.arange(len(hrir)) / fs * 1000

# 3) Detect the first 5 peaks with base sigma 1.0 ms
peaks = detect_peaks(hrir, times_ms, n_peaks=5, base_sigma=1.0, mode='hrir')

# Plot
fig, ax, slider = scroll_plot(times_ms, hrir, window_width_ms=5, peaks=peaks)

print("Peak latencies (ms):", times_ms[peaks])