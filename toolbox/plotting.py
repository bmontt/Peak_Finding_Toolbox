import os
import matplotlib.pyplot as plt


def plot_abr(
    times_ms,
    data_arr,
    peaks_per_channel,
    ch_names,
    snr_list,
    subject_id,
    mode,
    outdir,
    ref_lines=(1.5, 3.5, 6.0)
):
    """
    Plot ABR waveforms with detected peaks and reference latency lines.

    Args:
        times_ms (np.ndarray): Time vector in milliseconds.
        data_arr (np.ndarray): Array of shape (n_channels, n_samples).
        peaks_per_channel (list of np.ndarray): Peak indices per channel.
        ch_names (list of str): Channel names.
        snr_list (list of float): Normalized SNR per channel.
        subject_id (str): Identifier for the subject.
        mode (str): Processing mode ('average', 'subtract', 'individual').
        outdir (str): Directory to save the plot.
        ref_lines (tuple): Latencies (ms) at which to draw vertical reference lines.

    Returns:
        str: Filepath of saved plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each channel with peak annotations
    for idx, ch in enumerate(ch_names):
        data = data_arr[idx]
        snr = snr_list[idx]
        ax.plot(times_ms, data, label=f"{ch} (SNR {snr*100:.1f}%)")
        peaks = peaks_per_channel[idx]
        for p in peaks:
            lat = times_ms[p]
            amp = data[p]
            ax.plot(lat, amp, 'o')
            ax.annotate(
                f"{lat:.1f} ms",
                xy=(lat, amp),
                xytext=(lat, amp * 1.05),
                arrowprops=dict(arrowstyle="->", lw=1),
                ha='center',
                va='bottom',
                fontsize=8
            )

    # Add reference latency lines
    for x in ref_lines:
        ax.axvline(x, linestyle='--', linewidth=1, alpha=0.7)

    ax.set_title(f"Subject {subject_id} ABR ({mode})")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    os.makedirs(outdir, exist_ok=True)
    filename = f"{subject_id}_abr_plot.png"
    filepath = os.path.join(outdir, filename)
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath


def plot_hrir(
    times_ms,
    hrir,
    peaks,
    troughs,
    base,
    outdir,
    annotate=True
):
    """
    Plot HRIR impulse response with detected peaks and troughs.

    Args:
        times_ms (np.ndarray): Time vector in milliseconds.
        hrir (np.ndarray): HRIR samples.
        peaks (list of int): Peak indices.
        troughs (list of int): Trough indices.
        base (str): Base filename identifier.
        outdir (str): Directory to save the plot.
        annotate (bool): Whether to annotate peak/trough latencies.

    Returns:
        str: Filepath of saved plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times_ms, hrir, label="HRIR")

    # Plot peaks
    if peaks:
        ax.scatter([times_ms[p] for p in peaks], [hrir[p] for p in peaks],
                   marker='^', label='Peaks')
        if annotate:
            for p in peaks:
                lat = times_ms[p]
                amp = hrir[p]
                ax.annotate(f"{lat:.1f} ms",
                            xy=(lat, amp),
                            xytext=(lat, amp * 1.05),
                            arrowprops=dict(arrowstyle="->", lw=1),
                            ha='center', va='bottom', fontsize=8)

    # Plot troughs
    if troughs:
        ax.scatter([times_ms[t] for t in troughs], [hrir[t] for t in troughs],
                   marker='v', label='Troughs')
        if annotate:
            for t in troughs:
                lat = times_ms[t]
                amp = hrir[t]
                ax.annotate(f"{lat:.1f} ms",
                            xy=(lat, amp),
                            xytext=(lat, amp * 0.95),
                            arrowprops=dict(arrowstyle="->", lw=1),
                            ha='center', va='top', fontsize=8)

    ax.set_title(f"{base} HRIR Peaks & Troughs")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    os.makedirs(outdir, exist_ok=True)
    filename = f"{base}_hrir_plot.png"
    filepath = os.path.join(outdir, filename)
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath
