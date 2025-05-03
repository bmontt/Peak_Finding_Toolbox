# tests/conftest.py
import scipy
import scipy.signal
import scipy.signal.windows as _win

# Monkey‑patch scipy.signal to expose hann for older pyroomacoustics
scipy.signal.hann = _win.hann
