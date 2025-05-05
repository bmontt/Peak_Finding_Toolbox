import os
import glob
import csv
import re
import numpy as np
import wfdb
import matplotlib.pyplot as plt

from toolbox.peak_finder import detect_peaks  # automatic peak detection

# Configuration
data_root = '/mnt/s/Datasets/data'
DATASETS = {
    'normal': os.path.join(data_root, 'earndb'),
    'impaired': os.path.join(data_root, 'earhdb'),
}
OUTPUT_CSV = 'bench/peak_comparison.csv'
N_PEAKS = 5
# Stimuli to benchmark
SELECTED_STIMULI = ['70_F1', '90_F4', '1kHz_70', '4kHz_90']


def find_replicate_groups(root_dir):
    """Group .dat files by (subject, stimulus)"""
    files = glob.glob(os.path.join(root_dir, '*.dat'))
    groups = {}
    for path in files:
        name = os.path.basename(path)
        subj = name.split('_', 1)[0]
        stim = name.rsplit('_R', 1)[0] if '_R' in name else name.rsplit('_', 1)[0]
        groups.setdefault((subj, stim), []).append(path)
    return groups


def parse_trial_length(header):
    """Extract trial length (samples) from WFDB header comments"""
    for comment in getattr(header, 'comments', []):
        m = re.search(r'<Trial Length \(samples\)>:\s*(\d+)', comment)
        if m:
            return int(m.group(1))
    return None


def average_group(paths):
    """Align replicates by trigger and compute mean ABR waveform"""
    signals, onsets, fs_list = [], [], []
    for path in sorted(paths):
        base = path[:-4]
        hdr = wfdb.rdheader(base)
        rec = wfdb.rdrecord(base)
        # safe channel names
        raw_names = getattr(hdr, 'sig_name', [])
        names = [s.lower() if isinstance(s, str) else '' for s in raw_names]
        # find channels
        trig_idx = names.index('trigger') if 'trigger' in names else 0
        abr_idx  = names.index('abr') if 'abr' in names else (1 if len(names) > 1 else 0)
        trig = rec.p_signal[:, trig_idx]
        abr  = rec.p_signal[:, abr_idx]
        onset = int(np.argmax(trig > 0))
        signals.append(abr)
        onsets.append(onset)
        fs_list.append(getattr(hdr, 'fs', rec.fs))

    if not signals:
        return None, None, None, None

    if len(set(fs_list)) > 1:
        print(f"Warning: inconsistent sample rates in {paths[0]}")
    fs = fs_list[0]
    # truncate to shortest aligned segment
    min_len = min(len(sig) - o for sig, o in zip(signals, onsets))
    aligned = [sig[o:o+min_len] for sig, o in zip(signals, onsets)]
    mean_wave = np.mean(np.stack(aligned, axis=1), axis=1)
    onset_min = min(onsets)
    trial_len = parse_trial_length(hdr)
    return mean_wave, fs, onset_min, trial_len


def manual_peak_pick(times, wave, n_peaks=N_PEAKS, window=None, marker=None):
    """Interactive manual peak picking"""
    picks, lines = [], []
    fig, ax = plt.subplots()
    ax.plot(times, wave)
    ax.set_title('Manual Peaks: left-click=pick, right-click=skip, Ctrl+Z=undo')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')

    if marker is not None:
        ax.axvline(marker, color='b', linestyle='-', linewidth=2, label='Onset')
        ax.legend()
    if window:
        ax.set_xlim(window)

    def onclick(event):
        if event.inaxes != ax or len(picks) >= n_peaks:
            return
        x = event.xdata
        linestyle = '--' if event.button == 1 else ':'
        color = 'k' if event.button == 1 else 'r'
        picks.append(x)
        lines.append(ax.axvline(x, color=color, linestyle=linestyle))
        fig.canvas.draw()
        if len(picks) >= n_peaks:
            plt.close(fig)

    def onkey(event):
        if event.key in ('ctrl+z', 'control+z') and picks:
            picks.pop()
            ln = lines.pop()
            ln.remove()
            fig.canvas.draw()

    cid1 = fig.canvas.mpl_connect('button_press_event', onclick)
    cid2 = fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()
    fig.canvas.mpl_disconnect(cid1)
    fig.canvas.mpl_disconnect(cid2)
    return picks


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as fout:
        writer = csv.writer(fout)
        header = ['dataset', 'subject', 'stimulus']
        header += [f'Manual_Wave{i}' for i in range(1, N_PEAKS+1)]
        header += [f'Auto_Wave{i}'   for i in range(1, N_PEAKS+1)]
        writer.writerow(header)

        for label, root in DATASETS.items():
            groups = find_replicate_groups(root)
            sel = {k: v for k, v in groups.items() if any(s in k[1] for s in SELECTED_STIMULI)}
            for (subj, stim), paths in sorted(sel.items()):
                wave, fs, onset, tlen = average_group(paths)
                if wave is None:
                    continue
                times = ((np.arange(len(wave)) + onset) / fs) * 1000
                window = (times[0], times[-1])
                marker = onset / fs * 1000

                manual = manual_peak_pick(times, wave, N_PEAKS, window, marker)
                auto = detect_peaks(wave, times, N_PEAKS, base_sigma=0.06, mode='abr')
                # pad lists
                manual += [np.nan] * (N_PEAKS - len(manual))
                auto   += [np.nan] * (N_PEAKS - len(auto))

                writer.writerow([label, subj, stim] + manual + auto)
                print(f"{label},{subj},{stim}: Manual={manual} Auto={auto}")

    print(f"Results saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
