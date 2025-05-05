import os
import glob
import csv
import numpy as np
import wfdb
import matplotlib.pyplot as plt

from toolbox.peak_finder import detect_peaks  # algorithmic peak finder
from toolbox.plotting import plot_abr           # ABR plotting function

# Configuration
AVERAGE_DIR        = 'data/earndb_data/average'    # local directory for average dataset
AUTO_OUTPUT_CSV    = 'bench/auto_peak_picks.csv'   # automatic-only results
MANUAL_OUTPUT_CSV  = 'bench/manual_peak_picks.csv' # manual + automatic results
PERFORM_MANUAL     = False                          # toggle manual picking stage
PLOT_AUTO          = False                         # toggle overlay of auto picks in manual stage
PLOT_ABR_GRAPHS    = True                          # toggle generation of ABR plots
PLOT_OUTDIR        = 'bench/abr_plots'             # output directory for ABR plots
N_PEAKS            = 5
TIME_WINDOW        = (0, 15)                       # ms window for peak extraction
# Optional: filter stimuli by substring match
SELECTED_STIMULI   = ['70_F1', '90_F4', '1kHz_70', '4kHz_90']


def manual_peak_pick(times, signal, n_peaks=N_PEAKS):
    """
    Interactive picker: left-click picks, right-click skips (NaN).
    Returns sorted picks along x-axis.
    Overlays auto-detected peaks if PLOT_AUTO is True.
    """
    picks, lines = [], []
    fig, ax = plt.subplots()
    ax.plot(times, signal, label='ABR')

    # overlay auto picks
    if hasattr(manual_peak_pick, 'auto_latencies') and PLOT_AUTO:
        for x in manual_peak_pick.auto_latencies:
            ax.axvline(x, linestyle='--', linewidth=1, alpha=0.7)
        ax.legend(['ABR', 'Auto peaks'])

    ax.set_title("Click Peaks (left-click to pick, right-click to skip, Ctrl+Z undo)")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(TIME_WINDOW)

    def onclick(event):
        if event.inaxes != ax or len(picks) >= n_peaks:
            return
        x = event.xdata
        if event.button == 1:
            picks.append(x)
            lines.append(ax.axvline(x, color='k', linestyle='--'))
        elif event.button == 3:
            picks.append(np.nan)
            lines.append(ax.axvline(x, color='r', linestyle=':'))
        fig.canvas.draw()
        if len(picks) >= n_peaks:
            plt.close(fig)

    def onkey(event):
        if event.key.lower() in ('ctrl+z', 'control+z') and picks:
            picks.pop()
            ln = lines.pop()
            ln.remove()
            fig.canvas.draw()

    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_key   = fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()
    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)

    # sort picks by latency, NaNs at end
    non_nans = sorted([p for p in picks if not np.isnan(p)])
    return non_nans + [np.nan] * (n_peaks - len(non_nans))


def main():
    # ensure plot directory exists
    if PLOT_ABR_GRAPHS:
        os.makedirs(PLOT_OUTDIR, exist_ok=True)

    # Stage 1: Automatic peak detection
    os.makedirs(os.path.dirname(AUTO_OUTPUT_CSV), exist_ok=True)
    print(f"Processing automatic peak detection for files in {AVERAGE_DIR!r}...")
    dat_files = glob.glob(os.path.join(AVERAGE_DIR, '*.dat'))
    print(f"Found {len(dat_files)} .dat files")
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in {AVERAGE_DIR}")

    auto_results = {}
    with open(AUTO_OUTPUT_CSV, 'w', newline='') as fa:
        writer_auto = csv.writer(fa)
        writer_auto.writerow(
            ['subject', 'stimulus']
            + [f'Auto_Wave{{i}}_ms'.format(i=i) for i in range(1, N_PEAKS+1)]
        )
        for p in sorted(dat_files):
            base = os.path.splitext(p)[0]
            fname = os.path.basename(base)
            subj, stimulus = (fname.split('_',1) if '_' in fname else (fname, ''))
            # filter stimuli
            if SELECTED_STIMULI and not any(s in stimulus for s in SELECTED_STIMULI):
                continue

            # load ABR data
            hdr = wfdb.rdheader(base)
            rec = wfdb.rdrecord(base)
            fs = hdr.fs
            abr = rec.p_signal[:, 0]

            # windowed times & data
            times = np.arange(len(abr)) / fs * 1000
            mask = (times >= TIME_WINDOW[0]) & (times <= TIME_WINDOW[1])
            times_win, data_win = times[mask], abr[mask]

            # auto detect peaks
            peak_idxs = detect_peaks(data_win, times_win, n_peaks=N_PEAKS)
            peak_idxs = np.array(peak_idxs, dtype=int)
            auto_latencies = times_win[peak_idxs].tolist()
            auto_latencies = (auto_latencies + [np.nan]*N_PEAKS)[:N_PEAKS]

            writer_auto.writerow([subj, stimulus] + auto_latencies)
            auto_results[base] = auto_latencies

            # generate ABR auto plot
            if PLOT_ABR_GRAPHS:
                plot_abr(
                    times_win,
                    np.array([data_win]),           # single channel
                    [peak_idxs],                    # list of index arrays
                    [stimulus],                     # channel names
                    [0.],                           # dummy SNR
                    subj,
                    'auto',
                    PLOT_OUTDIR
                )
    print(f"Saved automatic peak picks to {AUTO_OUTPUT_CSV}")

    if not PERFORM_MANUAL:
        print("Manual picking disabled (PERFORM_MANUAL=False); exiting.")
        return

    # Stage 2: Manual peak picking
    os.makedirs(os.path.dirname(MANUAL_OUTPUT_CSV), exist_ok=True)
    with open(MANUAL_OUTPUT_CSV, 'w', newline='') as fm:
        writer = csv.writer(fm)
        header = (
            ['subject', 'stimulus']
            + [f'Manual_Wave{{i}}_ms'.format(i=i) for i in range(1, N_PEAKS+1)]
            + [f'Auto_Wave{{i}}_ms'.format(i=i)   for i in range(1, N_PEAKS+1)]
        )
        writer.writerow(header)

        for p in sorted(dat_files):
            base = os.path.splitext(p)[0]
            fname = os.path.basename(base)
            subj, stimulus = (fname.split('_',1) if '_' in fname else (fname, ''))
            if SELECTED_STIMULI and not any(s in stimulus for s in SELECTED_STIMULI):
                continue

            auto_latencies = auto_results.get(base, [np.nan]*N_PEAKS)
            manual_peak_pick.auto_latencies = auto_latencies
            manual_peak_pick.plot_auto      = PLOT_AUTO

            # reload signal
            hdr = wfdb.rdheader(base)
            rec = wfdb.rdrecord(base)
            fs = hdr.fs
            abr = rec.p_signal[:, 0]
            times = np.arange(len(abr)) / fs * 1000
            mask = (times >= TIME_WINDOW[0]) & (times <= TIME_WINDOW[1])
            times_win, data_win = times[mask], abr[mask]

            manual_latencies = manual_peak_pick(times_win, data_win, N_PEAKS)
            writer.writerow([subj, stimulus] + manual_latencies + auto_latencies)
            print(f"{fname}: manual={manual_latencies}, auto={auto_latencies}")

            # generate ABR manual plot
            if PLOT_ABR_GRAPHS:
                manual_idxs = [int(np.argmin(np.abs(times_win - m)))
                               for m in manual_latencies if not np.isnan(m)]
                plot_abr(
                    times_win,
                    np.array([data_win]),
                    [manual_idxs],
                    [stimulus],
                    [0.],
                    subj,
                    'manual',
                    PLOT_OUTDIR
                )
    print(f"Saved manual+auto peak picks to {MANUAL_OUTPUT_CSV}")


if __name__ == '__main__':
    main()
