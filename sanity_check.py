import numpy as np
from toolbox.peak_finder import detect_peaks

fs = 48000
t = np.linspace(0, 1, fs)
signal = np.exp(-((t - 0.5)**2) / (2 * (0.01**2))) + 0.02 * np.random.randn(fs)
peaks = detect_peaks(signal, t * 1000, n_peaks=3)
print("Detected peaks (ms):", peaks)
