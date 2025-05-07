#!/usr/bin/env python3
"""
benchmark_abr_pipeline.py

Benchmark your ABR peak detection (without manual GUI steps),
based on a single average file from your earndb_data set.

Ensure that 'fname' below points to a .dat/.hea pair in earndb_data.
"""

import os
import time
import platform
import sys
import numpy as np
import wfdb

from toolbox.peak_finder import detect_peaks  # your adaptive peak finder

# === Configuration ===
FNAME = "N1_evoked_ave20_F1_R1"   # basename without .dat/.hea
DATA_DIR = "data/earndb_data/average"          # path to your .dat/.hea files
N_REPS = 100                      # how many times to repeat for timing
SIGMA = 0.06                      # smoothing parameter for peak detection

def get_hardware_info():
    uname = platform.uname()
    cpu = uname.processor or uname.machine
    os_info = f"{uname.system} {uname.release}"
    py_version = sys.version.split()[0]
    return f"CPU: {cpu} on {os_info}, Python: {py_version}"

def load_abr(file_base):
    hdr = wfdb.rdheader(file_base)
    rec = wfdb.rdrecord(file_base)
    fs = hdr.fs
    gain_uv = hdr.adc_gain[0]
    data_uv = rec.p_signal[:, 0] * gain_uv
    times_ms = np.arange(len(data_uv)) / fs * 1000
    return data_uv, times_ms

def main():
    print("Hardware Info:", get_hardware_info())
    file_base = os.path.join(DATA_DIR, os.path.splitext(FNAME)[0])
    print(f"Loading ABR data from: {file_base}")

    data_uv, times_ms = load_abr(file_base)

    # Warm-up to exclude import/init time
    _ = detect_peaks(data_uv, times_ms, n_peaks=5, base_sigma=SIGMA)

    print(f"Benchmarking detect_peaks over {N_REPS} runs...")
    start = time.perf_counter()
    for _ in range(N_REPS):
        _ = detect_peaks(data_uv, times_ms, n_peaks=5, base_sigma=SIGMA)
    elapsed = time.perf_counter() - start
    avg_time_ms = (elapsed / N_REPS) * 1000
    print(f"Average runtime: {avg_time_ms:.3f} ms over {N_REPS} runs")

if __name__ == "__main__":
    main()
