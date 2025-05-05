import os
import glob
import csv
import numpy as np
import wfdb
import matplotlib.pyplot as plt

from toolbox.peak_finder import detect_peaks  # optional: to overlay

# Configuration
DATASETS = {
    'normal':   '/mnt/s/Datasets/data/earndb',
    'impaired': '/mnt/s/Datasets/data/earhdb',
}
OUTPUT_CSV = 'bench/manual_peak_picks.csv'
N_PEAKS = 5
# Filter stimuli by substring match (choose the two most relevant)
SELECTED_STIMULI = [
    '70_F1', '90_F4', '1kHz_70', '4kHz_90'
]
# Time window for manual picking relative to stimulus onset (ms)
TIME_WINDOW = (0, 50)


def find_replicate_groups(root_dir):
    paths = glob.glob(os.path.join(root_dir, '*.dat'))
    groups = {}
    for p in paths:
        fname = os.path.basename(p)
        subj = fname.split('_', 1)[0]
        if '_R' in fname:
            stimulus = fname.rsplit('_R', 1)[0]
        else:
            stimulus = fname.rsplit('_', 1)[0]
        groups.setdefault((subj, stimulus), []).append(p)
    return groups


def average_group(paths):
    signals, onsets, fs_vals = [], [], []
    for p in sorted(paths):
        base = p[:-4]
        hdr = wfdb.rdheader(base)
        rec = wfdb.rdrecord(base)
        names = [n.lower() if isinstance(n, str) else '' for n in hdr.sig_name]
        trig_idx = names.index('trigger') if 'trigger' in names else 0
        abr_idx = names.index('abr') if 'abr' in names else (1 if len(names) > 1 else 0)
        trig = rec.p_signal[:, trig_idx]
        abr = rec.p_signal[:, abr_idx]
        onsets.append(int(np.argmax(trig > 0)))
        signals.append(abr)
        fs_vals.append(hdr.fs)
    if not signals:
        return None, None, None
    if len(set(fs_vals)) > 1:
        print(f"Warning: fs mismatch in {paths[0]}")
    fs = fs_vals[0]
    min_len = min(len(sig) - o for sig, o in zip(signals, onsets))
    aligned = [sig[o:o + min_len] for sig, o in zip(signals, onsets)]
    data = np.mean(np.stack(aligned, axis=1), axis=1)
    onset = min(onsets)
    return data, fs, onset


def manual_peak_pick(times, signal, n_peaks=N_PEAKS):
    picks = []
    lines = []
    fig, ax = plt.subplots()
    ax.plot(times, signal, label='ABR')
    ax.set_title("Click Waves I–V (left-click to pick, right-click to skip, Ctrl+Z to undo)")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(TIME_WINDOW)

    def onclick(event):
        if event.inaxes != ax or len(picks) >= n_peaks:
            return
        if event.button == 1:
            x = event.xdata
            picks.append(x)
            line = ax.axvline(x, color='k', linestyle='--')
            lines.append(line)
        elif event.button == 3:
            picks.append(np.nan)
            line = ax.axvline(event.xdata, color='r', linestyle=':')
            lines.append(line)
        fig.canvas.draw()
        if len(picks) >= n_peaks:
            plt.close(fig)

    def onkey(event):
        if event.key in ('ctrl+z', 'control+z') and picks:
            picks.pop()
            ln = lines.pop()
            ln.remove()
            fig.canvas.draw()

    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_key = fig.canvas.mpl_connect('key_press_event', onkey)

    plt.show()
    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)
    return picks


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'subject', 'stimulus'] + [f'Wave{i}_ms' for i in range(1, N_PEAKS+1)])

        for label, root in DATASETS.items():
            print(f"Dataset: {label}")
            all_groups = find_replicate_groups(root)
            selected = {k: v for k, v in all_groups.items() if any(substr in k[1] for substr in SELECTED_STIMULI)}

            for (subj, stimulus), paths in sorted(selected.items()):
                # Print header info for debugging
                hdr0 = wfdb.rdheader(paths[0][:-4])
                print(f"-- Header for {label}, {subj}, {stimulus} --")
                print("Signal names:", hdr0.sig_name)
                print("Units:", getattr(hdr0, 'units', None))
                print("Comments:", getattr(hdr0, 'comments', None))

                data, fs, onset = average_group(paths)
                if data is None:
                    continue

                times = (np.arange(data.size) / fs) * 1000
                mask = (times >= TIME_WINDOW[0]) & (times <= TIME_WINDOW[1])
                times_win = times[mask]
                data_win = data[mask]

                latencies = manual_peak_pick(times_win, data_win, N_PEAKS)
                writer.writerow([label, subj, stimulus] + latencies)
                print(f"Picked for {label}, {subj}, {stimulus}: {latencies}")

    print(f"Saved manual picks to {OUTPUT_CSV}")


if __name__ == '__main__':
    main()