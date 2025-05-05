"""
Main CLI pipeline for ABR and HRIR peak detection using the shared peak_finder module.
All plotting is delegated to toolbox/plotting.py.
"""
import os
import click
import numpy as np
import pandas as pd

from .data_loader import load_eeg_epochs, load_hrir, load_audio_file
from .peak_finder import detect_peaks, compute_snr_normalized, label_hrir_peaks
from .plotting import plot_abr, plot_hrir, scroll_plot

# Default parameters
DEFAULT_SIGMA_ABR = 0.06
DEFAULT_SIGMA_HRIR = 1.0
DEFAULT_OUTDIR = 'results'
WAVE_LABELS = ["Wave I", "Wave II", "Wave III", "Wave IV", "Wave V"]

@click.group()
def main():
    """Command-line interface for ABR and HRIR analysis"""
    pass

@main.command()
@click.argument('bids_root', type=click.Path(exists=True))
@click.argument('subject_id')
@click.option('--mode', default='average', type=click.Choice(['average', 'subtract', 'individual']),
              help='Processing mode for ABR: average, subtract, or individual')
@click.option('--sigma', default=DEFAULT_SIGMA_ABR, show_default=True,
              help='Base sigma for adaptive smoothing in ABR')
@click.option('--outdir', default=DEFAULT_OUTDIR, show_default=True,
              help='Directory to save ABR outputs')
def abr(bids_root, subject_id, mode, sigma, outdir):
    """
    Detect ABR Waves I–V for a BIDS subject, save CSV and (optionally) plot.
    """
    # Load and preprocess ABR epochs
    epochs = load_eeg_epochs(bids_root, subject_id)
    evoked = epochs.average()

    # Prepare data array & channel names
    if mode == 'average':
        data_arr = np.mean(evoked.data, axis=0, keepdims=True) * 1e6
        ch_names = ['Averaged']
    elif mode == 'subtract':
        diff = (evoked.data[0] - evoked.data[1]) * 1e6
        data_arr = diff.reshape(1, -1)
        ch_names = ['R_minus_L']
    else:
        data_arr = evoked.data * 1e6
        ch_names = evoked.ch_names

    times_ms = evoked.times * 1000
    records = []
    snr_list = []
    peaks_list = []

    # Detect peaks and record results
    for idx, ch in enumerate(ch_names):
        data = data_arr[idx]
        snr = compute_snr_normalized(data, times_ms)
        peaks = detect_peaks(data, times_ms, n_peaks=5, base_sigma=sigma)
        snr_list.append(snr)
        peaks_list.append(peaks)
        for i, label in enumerate(WAVE_LABELS):
            amp = data[peaks[i]] if i < len(peaks) else None
            lat = times_ms[peaks[i]] if i < len(peaks) else None
            records.append({
                'Subject': subject_id,
                'Channel': ch,
                'Wave': label,
                'Amplitude (µV)': amp,
                'Latency (ms)': lat,
                'SNR': snr
            })

    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)
    # Save CSV
    csv_path = os.path.join(outdir, f"{subject_id}_abr_waves.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    click.echo(f"ABR results saved to {csv_path}")

    # Plot if individual mode
    if mode == 'individual':
        plot_path = plot_abr(times_ms, data_arr, peaks_list, ch_names, snr_list,
                              subject_id, mode, outdir)
        click.echo(f"ABR plot saved to {plot_path}")

@main.command()
@click.argument('sofa_path', type=click.Path(exists=True))
@click.option('--channel', default='left', show_default=True,
              help='Either left or right channel for HRIR')
@click.option('--n_peaks', default=5, show_default=True,
              help='Number of peaks/troughs to detect')
@click.option('--sigma', default=DEFAULT_SIGMA_HRIR, show_default=True,
              help='Base sigma for adaptive smoothing in HRIR')
@click.option('--outdir', default=DEFAULT_OUTDIR, show_default=True,
              help='Directory to save HRIR outputs')
def hrir(sofa_path, channel, n_peaks, sigma, outdir):
    """
    Detect peaks and troughs in an HRIR SOFA file and save CSV and plot.
    """
    # Load and label HRIR
    info = label_hrir_peaks(sofa_path, channel,
                            n_peaks=n_peaks, base_sigma=sigma)
    hrir_data, fs = load_hrir(sofa_path, channel)
    times_ms = np.arange(len(hrir_data)) / fs * 1000
    peaks = [int(t/((times_ms[1]-times_ms[0])) ) for t, _ in info['peaks']] 
    troughs = [int(t/((times_ms[1]-times_ms[0])) ) for t, _ in info['troughs']]

    # Prepare records
    records = []
    for t, a in info['peaks']:
        records.append({'Type': 'Peak', 'Time (ms)': t, 'Amplitude': a})
    for t, a in info['troughs']:
        records.append({'Type': 'Trough', 'Time (ms)': t, 'Amplitude': a})

    # Save CSV
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(sofa_path))[0]
    csv_path = os.path.join(outdir, f"{base}_hrir.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    click.echo(f"HRIR peaks/troughs saved to {csv_path}")

    # Plot HRIR
    plot_path = plot_hrir(times_ms, hrir_data, peaks, troughs, base, outdir)
    click.echo(f"HRIR plot saved to {plot_path}")



@main.command()
@click.argument('audio_path', type=click.Path(exists=True))
@click.option('--sr',        default=None,     type=int,   help='Resample rate (Hz)')
@click.option('--n_peaks',   default=15,       type=int,   help='Number of peaks to detect')
@click.option('--sigma',     default=1.0,      type=float, help='Base sigma (ms) for peak detection')
@click.option('--outdir',    default=DEFAULT_OUTDIR,         help='Directory to save outputs')
@click.option('--window_width', default=1000, type=int,    help='Window width for scrolling plot (ms)')
@click.option('--show/--no_show', default=False,               help='Display plot interactively')
def audio(audio_path, sr, n_peaks, sigma, outdir, window_width, show):
    """
    Detect prominent peaks in a general audio file, save CSV, and (optionally) plot.
    """
    # 1) Load audio (mono)
    data, fs = load_audio_file(audio_path, sr)
    times_ms = np.arange(len(data)) / fs * 1000

    # 2) Run peak detection on raw waveform
    peaks = detect_peaks(data, times_ms, n_peaks=n_peaks, base_sigma=sigma, mode='audio')

    # 3) Prepare results table
    records = []
    for idx in peaks:
        records.append({
            'Time (ms)': float(times_ms[idx]),
            'Amplitude': float(data[idx])
        })

    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)

    # 4) Save CSV
    base = os.path.splitext(os.path.basename(audio_path))[0]
    csv_path = os.path.join(outdir, f'{base}_audio_peaks.csv')
    pd.DataFrame(records).to_csv(csv_path, index=False)
    click.echo(f"Audio peaks saved to {csv_path}")

    # 5) Plot if requested
    if show:
        fig, ax, _ = scroll_plot(times_ms, data, window_width_ms=window_width, peaks=peaks)
        ax.set_title(f'Detected Peaks – {base}')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        fig_path = os.path.join(outdir, f'{base}_audio_peaks.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        click.echo(f"Audio waveform plot saved to {fig_path}")


if __name__ == '__main__':
    main()