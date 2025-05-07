"""
toolbox.peak_finder

Adaptive and classical peak detection for ABR, HRIR, and general audio signals.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import itertools

# === Parameter Definitions ===
# ABR-specific
ABR_SNR_SIGNAL_WINDOW = (0, 10)
ABR_SNR_NOISE_WINDOW = (-5, 0)
ABR_SNR_DB_RANGE = (-5, 10)
ABR_WAVE_V_WINDOW = (5.6, 9.6)
ABR_PEAK_WINDOW = (0, 15)
ABR_MIN_DISTANCE_MS = 0.5

# HRIR-specific
HRIR_PEAK_WINDOW_MS = 5.0
HRIR_MIN_DISTANCE_MS = 0.1

# Audio-specific: generic
AUDIO_MIN_DISTANCE_MS = 0.5

# Prominence settings
PROMINENCE_BASE = 0.01
PROMINENCE_HIGH = 0.1
PROMINENCE_EXPONENT = 3


def compute_snr_normalized(
    data: np.ndarray,
    times_ms: np.ndarray,
    signal_window: tuple = ABR_SNR_SIGNAL_WINDOW,
    noise_window: tuple = ABR_SNR_NOISE_WINDOW,
    db_range: tuple = ABR_SNR_DB_RANGE
) -> float:
    """
    Compute normalized SNR in [0,1] based on specified windows and dB range.
    """
    signal_mask = (times_ms >= signal_window[0]) & (times_ms <= signal_window[1])
    noise_mask = (times_ms >= noise_window[0]) & (times_ms <= noise_window[1])
    sig_rms = np.sqrt(np.mean(data[signal_mask]**2)) if signal_mask.any() else 0
    noise_rms = np.sqrt(np.mean(data[noise_mask]**2)) if noise_mask.any() else 0
    snr_db = 20*np.log10(sig_rms/noise_rms) if noise_rms>0 else np.inf
    snr_clip = np.clip(snr_db, *db_range)
    return (snr_clip - db_range[0])/(db_range[1]-db_range[0])


def scale_prominence(snr: float) -> float:
    """
    Map normalized SNR to a prominence threshold between base and high.
    """
    snr = np.clip(snr, 0, 1)
    return PROMINENCE_BASE + (PROMINENCE_HIGH - PROMINENCE_BASE)*(1 - snr**PROMINENCE_EXPONENT)


# normative inter‑peak windows (ms)
IP_I_III = (1.80, 2.30)
IP_III_V = (1.40, 2.30)
IP_I_V   = (3.76, 4.70)

def _detect_peaks_abr(data_uv, times_ms, n_peaks=5, base_sigma=1.0):
    """
    Returns:
      selected: np.ndarray of up to n_peaks indices into times_ms for waves I–V (–1 if missing)
      qc:       bool indicating whether I–III, III–V, I–V latencies all fell within normative windows
    """
    import itertools
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal  import find_peaks

    T = times_ms
    dt = T[1] - T[0]
    min_dist = int(ABR_MIN_DISTANCE_MS / dt)

    # ——— 0) Baseline correction (only if a pre‑stimulus window exists) ———
    noise_mask = (T >= ABR_SNR_NOISE_WINDOW[0]) & (T < 0)
    if noise_mask.any():
        baseline = data_uv[noise_mask].mean()
        data_uv = data_uv - baseline
    # else: skip baseline if no pre‑stim samples

    # ——— 1) Precompute local SNR stats ———
    sig_mask = (T >= ABR_SNR_SIGNAL_WINDOW[0]) & (T <= ABR_SNR_SIGNAL_WINDOW[1])
    if sig_mask.any() and noise_mask.any():
        signal_std = data_uv[sig_mask].std()
        noise_std  = data_uv[noise_mask].std()
    else:
        # fallback to global std
        signal_std = data_uv.std()
        noise_std  = signal_std
    # we’ll use signal_std as our prom baseline

    best_tri = None
    prominences = np.array([])  # for fallback

    # ——— 2) Candidate detection + smart triplet scoring ———
    for attempt in range(3):
        # adaptive, SNR‐weighted smoothing
        snr = compute_snr_normalized(data_uv, T)
        sigma = base_sigma * (1 + 0.2*attempt)
        adj_sigma = np.clip(sigma/(snr+0.1), 0.3*sigma, 4*sigma)
        smooth = gaussian_filter1d(data_uv, sigma=adj_sigma)

        # local‑SNR‐based prominence
        prom = signal_std * (1 - 0.2*attempt)

        # find all peaks in the full window
        full_mask = (T >= ABR_PEAK_WINDOW[0]) & (T <= ABR_PEAK_WINDOW[1])
        rel, props = find_peaks(smooth[full_mask], prominence=prom, distance=min_dist)
        if rel.size < 3:
            rel, props = find_peaks(
                smooth[full_mask],
                prominence=prom*0.5,
                distance=min_dist
            )

        cands = np.where(full_mask)[0][rel]
        prominences = props.get('prominences', np.zeros_like(rel))

        # prune beyond max V latency
        valid_mask = T[cands] <= ABR_WAVE_V_WINDOW[1]
        if valid_mask.sum() >= 3:
            cands = cands[valid_mask]
            prominences = prominences[valid_mask]

        # score every I–III–V triplet
        best_score = np.inf
        for i, j, k in itertools.combinations(range(len(cands)), 3):
            tI, tIII, tV = T[cands[i]], T[cands[j]], T[cands[k]]
            d13, d35, d15 = tIII-tI, tV-tIII, tV-tI

            # enforce normative windows
            if not (IP_I_III[0] <= d13 <= IP_I_III[1]
                    and IP_III_V[0] <= d35 <= IP_III_V[1]
                    and IP_I_V[0]   <= d15 <= IP_I_V[1]):
                continue

            # weighted deviation: weight I–V interval more
            score = abs(d13-2.0) + abs(d35-2.0) + 2*abs(d15-4.0)
            if score < best_score:
                best_score, best_tri = score, (cands[i], cands[j], cands[k])

        if best_tri is not None:
            break

    # ——— 3) Fallback if we never found a valid triplet ———
    if best_tri is None:
        if len(cands) >= 3:
            order = np.argsort(prominences)[::-1]
            top3 = np.sort(cands[order[:3]])
            best_tri = tuple(top3)
        else:
            # not enough candidates: mark all three waves missing
            best_tri = (-1, -1, -1)

    wave_I, wave_III, wave_V = best_tri

    # ——— 4) Dynamic search for II & IV between the anchors ———
    buf = 0.2  # ms margin
    def pick_between(t_lo, t_hi):
        mask = (T > t_lo+buf) & (T < t_hi-buf)
        if not mask.any():
            return -1
        pks, pps = find_peaks(smooth[mask], prominence=prom, distance=min_dist)
        if pks.size:
            rel_idx = pks[np.argmax(pps['prominences'])]
            return np.where(mask)[0][rel_idx]
        return -1

    wave_II  = pick_between(T[wave_I],   T[wave_III])
    wave_IV  = pick_between(T[wave_III], T[wave_V])

    # ——— 5) Assemble & QC ———
    peaks = np.array([wave_I, wave_II, wave_III, wave_IV, wave_V], dtype=int)
    lat13 = T[wave_III] - T[wave_I] if wave_I>=0 and wave_III>=0 else np.nan
    lat35 = T[wave_V]   - T[wave_III] if wave_III>=0 and wave_V>=0 else np.nan
    lat15 = T[wave_V]   - T[wave_I]   if wave_I>=0 and wave_V>=0 else np.nan

    qc = (
        IP_I_III[0] <= lat13 <= IP_I_III[1] and
        IP_III_V[0] <= lat35 <= IP_III_V[1] and
        IP_I_V[0]   <= lat15 <= IP_I_V[1]
    )

    # ——— 6) Return top‑n in time order ———
    priority = [4, 0, 2, 1, 3]  # V, I, III, II, IV
    sel_n    = min(n_peaks, peaks.size)
    selected = np.sort(peaks[priority[:sel_n]])

    return selected, bool(qc)



def _detect_peaks_hrir(data, times_ms, n_peaks, base_sigma):
    snr = compute_snr_normalized(data, times_ms)
    prom = scale_prominence(snr)
    max_i = np.searchsorted(times_ms, HRIR_PEAK_WINDOW_MS)
    seg = data[:max_i]
    dt = times_ms[1]-times_ms[0]
    min_dist = int(HRIR_MIN_DISTANCE_MS/dt)
    peaks, props = find_peaks(seg, prominence=prom, distance=min_dist)
    if len(peaks)>n_peaks:
        idx = np.argsort(props['prominences'])[::-1][:n_peaks]
        peaks = np.sort(peaks[idx])
    return peaks


def _detect_peaks_audio(data, times_ms, n_peaks, base_sigma):
    snr = compute_snr_normalized(data, times_ms)
    prom = scale_prominence(snr)
    dt = times_ms[1]-times_ms[0]
    min_dist = int(AUDIO_MIN_DISTANCE_MS/dt)
    peaks, props = find_peaks(data, prominence=prom, distance=min_dist)
    if len(peaks)>n_peaks:
        idx = np.argsort(data[peaks])[::-1][:n_peaks]
        peaks = np.sort(peaks[idx])
    return peaks


def detect_peaks(
    data: np.ndarray,
    times_ms: np.ndarray,
    n_peaks: int = 5,
    base_sigma: float = 1.0,
    mode: str = 'abr'
) -> np.ndarray:
    """
    Unified interface for peak detection.
    mode in {'abr','hrir','audio'}.
    """
    if mode == 'abr':
        peaks, qc = _detect_peaks_abr(data, times_ms, n_peaks, base_sigma)
        return peaks
    if mode == 'hrir':
        return _detect_peaks_hrir(data, times_ms, n_peaks, base_sigma)
    if mode == 'audio':
        return _detect_peaks_audio(data, times_ms, n_peaks, base_sigma)
    raise ValueError(f"Unknown mode '{mode}'")


def label_hrir_peaks(sofa_path, channel=0, n_peaks=5, base_sigma=1.0):
    # back compat patch
    from .data_loader import load_hrir
    hrir, fs = load_hrir(sofa_path, channel)
    times_ms = np.arange(len(hrir)) / fs * 1000
    peaks = detect_peaks(hrir, times_ms, n_peaks, base_sigma, mode='hrir')
    troughs = detect_peaks(-hrir, times_ms, n_peaks, base_sigma, mode='hrir')
    return {
        'peaks':   [(times_ms[i], hrir[i]) for i in peaks],
        'troughs': [(times_ms[i], -hrir[i]) for i in troughs]
    }
