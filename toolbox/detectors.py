"""
abr_toolbox/detectors.py

Pluggable peak/onset detector registry for ABR, HRIR, and general audio.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Type

# Registry for detectors
_registry: Dict[str, Type['BaseDetector']] = {}

def register_detector(name: str):
    """
    Decorator to register a detector class under a given name.
    """
    def decorator(cls):
        _registry[name] = cls
        return cls
    return decorator

class BaseDetector(ABC):
    """
    Abstract base class for all peak/onset detectors.
    """
    @abstractmethod
    def detect(self, data: np.ndarray, sr: float, **kwargs) -> np.ndarray:
        """
        Return sample indices of detected peaks/onsets.
        """
        pass

    @classmethod
    def create(cls, name: str, **params) -> 'BaseDetector':
        """
        Factory method: instantiate a registered detector by name.
        """
        if name not in _registry:
            raise ValueError(f"Detector '{name}' not registered")
        return _registry[name](**params)

# --- ABR Adaptive Detector (integrates existing ABR pipeline) ---
from toolbox.abr_peak_finder import compute_snr_normalized, predict_wave_V_latency
from toolbox.abr_peak_finder import detect_peaks as abr_detect

@register_detector('abr_adaptive')
class ABRAdaptiveDetector(BaseDetector):
    def __init__(self, n_peaks: int = 5, base_sigma: float = 1.0):
        self.n_peaks = n_peaks
        self.base_sigma = base_sigma

    def detect(self, data: np.ndarray, sr: float, **kwargs) -> np.ndarray:
        # Convert samples to time in milliseconds
        times_ms = np.arange(len(data)) / sr * 1000.0
        # Compute SNR and normalization
        snr_norm = compute_snr_normalized(data, times_ms)
        # Predict anchor (Wave V)
        anchor_idx = predict_wave_V_latency(data, times_ms, snr_norm)
        # Delegate to existing ABR peak finder
        return abr_detect(
            data,
            times_ms,
            anchor_idx,
            n_peaks=self.n_peaks,
            snr_norm=snr_norm,
            base_sigma=self.base_sigma)

# --- Classical SciPy Peak Detector ---
from scipy.signal import find_peaks

@register_detector('scipy_peak')
class ScipyPeakDetector(BaseDetector):
    def __init__(self, prominence: float = 0.01, distance: float = 0.001):
        # prominence: required prominence height in amplitude
        # distance: minimum distance between peaks in seconds
        self.prominence = prominence
        self.distance = distance

    def detect(self, data: np.ndarray, sr: float, **kwargs) -> np.ndarray:
        dist_samples = int(self.distance * sr)
        peaks, _ = find_peaks(
            data,
            prominence=self.prominence,
            distance=dist_samples)
        return peaks

# --- Onset Envelope Detector (speech/music) ---
@register_detector('onset_env')
class OnsetEnvelopeDetector(BaseDetector):
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def detect(self, data: np.ndarray, sr: float, **kwargs) -> np.ndarray:
        import librosa
        env = librosa.onset.onset_strength(y=data, sr=sr)
        peaks = librosa.onset.onset_detect(
            onset_envelope=env,
            backtrack=False,
            units='samples',
            energy_threshold=self.threshold)
        return peaks

# --- Placeholder: HRIR Cross-Correlation Detector ---
@register_detector('hrir_xcorr')
class HRIRXcorrDetector(BaseDetector):
    def __init__(self, template: np.ndarray = None):
        """
        HRIR cross-correlation detector placeholder.
        TODO: Load or pass an HRIR template for matching.
        """
        self.template = template

    def detect(self, data: np.ndarray, sr: float, **kwargs) -> np.ndarray:
        """
        TODO: implement HRIR impulse detection via cross-correlation.
        """
        raise NotImplementedError("HRIRXcorrDetector not implemented yet.")
