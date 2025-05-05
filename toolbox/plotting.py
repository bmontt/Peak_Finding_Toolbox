import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def plot_abr(
    times_ms: np.ndarray,
    data_arr: np.ndarray,
    peaks_per_channel: list[np.ndarray],
    ch_names: list[str],
    snr_list: list[float],
    subject_id: str,
    mode: str,
    outdir: str,
    ref_lines=(1.5, 3.5, 6.0),
    auto_zoom: bool = False,
    window_margin_ms: float = 1.0,
):
    """
    Plot ABR waveforms with detected peaks, auto‑zoomed around the peaks.

    Args:
        auto_zoom:        If True, limits x‑axis to [min_peak - margin, max_peak + margin].
        window_margin_ms: Margin in ms around the outermost peaks.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each channel + annotate peaks
    all_peak_idxs = []
    for idx, ch in enumerate(ch_names):
        data = data_arr[idx]
        ax.plot(times_ms, data, label=f"{ch} (SNR {snr_list[idx]*100:.1f}%)")
        for p in peaks_per_channel[idx]:
            lat, amp = times_ms[p], data[p]
            all_peak_idxs.append(p)
            ax.plot(lat, amp, 'o')
            ax.annotate(f"{lat:.1f} ms",
                        xy=(lat, amp),
                        xytext=(lat, amp * 1.05),
                        arrowprops=dict(arrowstyle="->"),
                        ha='center', va='bottom', fontsize=8)

    # Reference lines
    for x in ref_lines:
        ax.axvline(x, linestyle='--', alpha=0.4)

    # Auto‑zoom
    if auto_zoom and all_peak_idxs:
        ts = times_ms[np.array(all_peak_idxs)]
        xmin = max(0, ts.min() - window_margin_ms)
        xmax = ts.max() + window_margin_ms
        ax.set_xlim(xmin, xmax)

    ax.set_title(f"Subject {subject_id} ABR ({mode})")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{subject_id}_abr_plot.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_hrir(
    times_ms: np.ndarray,
    hrir: np.ndarray,
    peaks: list[int],
    troughs: list[int],
    base: str,
    outdir: str,
    annotate: bool = True,
    auto_zoom: bool = True,
    window_margin_ms: float = 0.5,
):
    """
    Plot HRIR impulse response with peaks & troughs, auto‑zoomed around detections.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times_ms, hrir, label="HRIR")

    # Plot peaks
    for idx in peaks:
        lat, amp = times_ms[idx], hrir[idx]
        ax.scatter(lat, amp, marker='^', label='Peaks' if idx == peaks[0] else "")
        if annotate:
            ax.annotate(f"{lat:.1f} ms",
                        xy=(lat, amp),
                        xytext=(lat, amp * 1.05),
                        arrowprops=dict(arrowstyle="->"),
                        ha='center', va='bottom', fontsize=8)

    # Plot troughs
    for idx in troughs:
        lat, amp = times_ms[idx], hrir[idx]
        ax.scatter(lat, amp, marker='v', label='Troughs' if idx == troughs[0] else "")
        if annotate:
            ax.annotate(f"{lat:.1f} ms",
                        xy=(lat, amp),
                        xytext=(lat, amp * 0.95),
                        arrowprops=dict(arrowstyle="->"),
                        ha='center', va='top', fontsize=8)

    # Auto‑zoom
    if auto_zoom and (len(peaks)>0 or len(troughs) > 0):
        idxs = np.array(peaks + troughs)
        ts = times_ms[idxs]
        xmin = max(0, ts.min() - window_margin_ms)
        xmax = ts.max() + window_margin_ms
        ax.set_xlim(xmin, xmax)

    ax.set_title(f"{base} HRIR Peaks & Troughs")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{base}_hrir_plot.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# —————————————————————————————————————————————
def scroll_plot(
    times: np.ndarray,
    signal: np.ndarray,
    window_width_ms: float,
    peaks: np.ndarray = None,
):
    """
    Interactive scrollable plot: use a slider to pan through a long waveform.

    Args:
        window_width_ms: width of the visible window in milliseconds.
        peaks:          optional peak indices to overlay.
    """
    # Convert window width from ms to time units (assumes uniform spacing)
    dt = times[1] - times[0]
    win_samps = int(window_width_ms / dt)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, signal, lw=1)
    if peaks is not None:
        ax.scatter(times[peaks], signal[peaks], color='r', marker='x')

    # initial view
    ax.set_xlim(times[0], times[0] + window_width_ms)

    # Slider axis
    axcolor = 'lightgoldenrodyellow'
    slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03], facecolor=axcolor)
    slider = Slider(slider_ax, 'Start (ms)', times[0], times[-1] - window_width_ms, valinit=times[0])

    # Update function
    def update(val):
        start = slider.val
        ax.set_xlim(start, start + window_width_ms)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
    return fig, ax, slider
