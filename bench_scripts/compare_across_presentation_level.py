import os
import glob
import csv
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from collections import defaultdict

from toolbox.peak_finder import detect_peaks  # algorithmic peak finder

# ————— Configuration —————
AVERAGE_DIR        = 'data/earndb_data/average'
AUTO_OUTPUT_CSV    = 'bench/auto_peak_picks.csv'
MANUAL_OUTPUT_CSV  = 'bench/manual_peak_picks_fake.csv'
PLOT_OUTDIR        = 'bench/abr_plots/auto'

PERFORM_MANUAL = False   # toggle interactive manual picking
PLOT_OVERLAY   = True    # toggle static overlay PNGs

N_PEAKS     = 1
TIME_WINDOW = (0, 15)    # ms window for peak extraction
SHIFT_MS    = 5          # shift amount for special subjects
SHIFT_SUBJS = {3, 8}     # subject numbers to shift

# patterns: “_F4” for normal, “4kHz_” for abnormal 4 kHz stimuli
SELECTED_STIMULI    = ['_F4', '4kHz_']
# only these presentation levels (denoted by "ave{n}_")
PRESENTATION_LEVELS = [20, 40, 60, 80, 100]


def manual_overlay_pick(entries, tw):
    """
    entries: list of (db_level, times, data, auto_peak_idxs)
    tw: tuple (t0, t1) for x-axis window
    Plots all curves + auto‑peaks, lets user click N_PEAKS times per curve.
    Returns dict: db_level → sorted list of N_PEAKS latencies.
    """
    picks = {db: [] for db, *_ in entries}
    fig, ax = plt.subplots(figsize=(8, 5))

    for db, times, data, peak_idxs in entries:
        ax.plot(times, data, label=f'{db} dB')
        ax.scatter(times[peak_idxs], data[peak_idxs], marker='x', s=50)

    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Overlay — click {N_PEAKS}× per curve')
    ax.set_xlim(tw)
    ax.legend()

    def onclick(event):
        if event.inaxes != ax:
            return
        x, y = event.xdata, event.ydata
        dists = []
        for db, times, data, _ in entries:
            idx = np.argmin(np.abs(times - x))
            dists.append(abs(data[idx] - y))
        i = int(np.argmin(dists))
        db, times, data, _ = entries[i]
        if len(picks[db]) < N_PEAKS:
            picks[db].append(x)
            ax.axvline(x, color='k', linestyle='--', linewidth=1)
            fig.canvas.draw()
            if all(len(picks[d]) >= N_PEAKS for d in picks):
                plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    for db in picks:
        vals = sorted(picks[db])
        picks[db] = vals + [np.nan] * (N_PEAKS - len(vals))

    return picks


def main():
    os.makedirs(PLOT_OUTDIR, exist_ok=True)
    os.makedirs(os.path.dirname(AUTO_OUTPUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(MANUAL_OUTPUT_CSV), exist_ok=True)

    dat_files = glob.glob(os.path.join(AVERAGE_DIR, '*.dat'))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files in {AVERAGE_DIR!r}")

    auto_results = {}
    overlay_data = {
        'normal':   defaultdict(lambda: defaultdict(list)),
        'abnormal': defaultdict(lambda: defaultdict(list)),
    }

    # ——— Stage 1: Automatic detection ———
    with open(AUTO_OUTPUT_CSV, 'w', newline='') as fa:
        writer = csv.writer(fa)
        writer.writerow(['subject', 'stimulus'] + ['Auto_WaveV_ms'])

        for p in sorted(dat_files):
            base  = os.path.splitext(p)[0]
            hea_f = base + '.hea'
            dat_f = base + '.dat'
            if not os.path.exists(hea_f) or not os.path.exists(dat_f):
                print(f"⚠️ Missing header/data for {base!r}; skipping.")
                continue

            try:
                hdr = wfdb.rdheader(base)
                rec = wfdb.rdrecord(base)
            except FileNotFoundError as e:
                print(f"⚠️ Could not load {base!r}: {e}; skipping.")
                continue

            fs      = hdr.fs
            gain_uv = hdr.adc_gain[0]      # µV per count
            abr_uv  = rec.p_signal[:, 0] * gain_uv
            times   = np.arange(len(abr_uv)) / fs * 1000

            fname = os.path.basename(base)
            subj, stimulus = (fname.split('_', 1) if '_' in fname else (fname, ''))
            # ensure subject names start with 'N'
            if subj.isdigit():
                subj = f'N{subj}'
            try:
                subj_num = int(subj.lstrip('N'))
            except ValueError:
                subj_num = None

            # filter by stimulus & presentation level
            if not any(pat in stimulus for pat in SELECTED_STIMULI):
                continue
            if not any(f'ave{lvl}_' in stimulus for lvl in PRESENTATION_LEVELS):
                continue

            # determine window for this subject
            if subj_num in SHIFT_SUBJS:
                tw = (TIME_WINDOW[0] + SHIFT_MS, TIME_WINDOW[1] + SHIFT_MS)
            else:
                tw = TIME_WINDOW

            # window the data
            mask      = (times >= tw[0]) & (times <= tw[1])
            times_win = times[mask]
            data_win  = abr_uv[mask]

            # auto peaks
            peak_idxs = np.array(
                detect_peaks(data_win, times_win, n_peaks=N_PEAKS, base_sigma=0.5),
                dtype=int
            )
            auto_lat  = times_win[peak_idxs].tolist()
            auto_lat  = (auto_lat + [np.nan] * N_PEAKS)[:N_PEAKS]

            # stash for overlay
            parts = stimulus.split('_')
            rep   = parts[-1]
            overlay_data['normal' if '_F4' in stimulus else 'abnormal'][subj][rep].append(
                (int(next(x.replace('ave','') for x in parts if x.startswith('ave'))),
                 times_win, data_win, peak_idxs)
            )

            # write auto results
            stim_name = (
                f"{int(next(x.replace('ave','') for x in parts if x.startswith('ave')))}_F4_{rep}"
                if '_F4' in stimulus else
                f"4kHz_{int(next(x.replace('ave','') for x in parts if x.startswith('ave')))}_{rep}"
            )
            writer.writerow([subj, stim_name] + auto_lat)
            auto_results[base] = auto_lat

    print(f"Saved automatic picks → {AUTO_OUTPUT_CSV}")

    # ——— Stage 1b: Static overlays ———
    if PLOT_OVERLAY:
        for group, subjects in overlay_data.items():
            for subj, reps in subjects.items():
                try:
                    subj_num = int(subj.lstrip('N'))
                except ValueError:
                    subj_num = None
                if subj_num in SHIFT_SUBJS:
                    tw = (TIME_WINDOW[0] + SHIFT_MS, TIME_WINDOW[1] + SHIFT_MS)
                else:
                    tw = TIME_WINDOW

                for rep, entries in reps.items():
                    # compute offset for stacking
                    ptps     = [np.ptp(data) for _, _, data, _ in entries]
                    ptp_max  = max(ptps)
                    n_levels = len(entries)
                    offset_uv = (ptp_max * 1.1) / n_levels

                    plt.figure(figsize=(8, 5))
                    for i, (db, times, data, peak_idxs) in enumerate(sorted(entries, key=lambda x: x[0])):
                        data_off = data + i * offset_uv
                        plt.plot(times, data_off, label=f'{db} dB')
                        plt.scatter(times[peak_idxs], data_off[peak_idxs], marker='x', s=50)

                    plt.xlabel('Latency (ms)')
                    plt.ylabel('Amplitude + offset (µV)')
                    plt.title(f'{subj} — 4 kHz {group} overlay ({rep})')
                    plt.xlim(tw)
                    plt.legend(title='Level', loc='upper right')
                    plt.tight_layout()

                    out = os.path.join(PLOT_OUTDIR, f'{subj}_{group}_{rep}_overlay.png')
                    plt.savefig(out)
                    plt.close()
                    print(f"Saved overlay → {out}")

    # ——— Stage 2: Interactive manual picks ———
    if PERFORM_MANUAL:
        with open(MANUAL_OUTPUT_CSV, 'w', newline='') as fm:
            writer = csv.writer(fm)
            writer.writerow(
                ['subject', 'stimulus']
                + [f'Manual_Wave{i}_ms' for i in range(1, N_PEAKS + 1)]
                + [f'Auto_Wave{i}_ms'   for i in range(1, N_PEAKS + 1)]
            )

            for group, subjects in overlay_data.items():
                for subj, reps in subjects.items():
                    try:
                        subj_num = int(subj.lstrip('N'))
                    except ValueError:
                        subj_num = None
                    if subj_num in SHIFT_SUBJS:
                        tw = (TIME_WINDOW[0] + SHIFT_MS, TIME_WINDOW[1] + SHIFT_MS)
                    else:
                        tw = TIME_WINDOW

                    for rep, entries in reps.items():
                        print(f"\n— Manual picking for {subj} ({group}, {rep}) —")
                        manual_picks = manual_overlay_pick(entries, tw)

                        for db, times, data, peak_idxs in entries:
                            stim_name = (
                                f"{db}_F4_{rep}" if group == 'normal'
                                else f"4kHz_{db}_{rep}"
                            )
                            auto_lat = times[peak_idxs].tolist()
                            auto_lat = (auto_lat + [np.nan]*N_PEAKS)[:N_PEAKS]
                            man_lat  = manual_picks[db]
                            writer.writerow([subj, stim_name] + man_lat + auto_lat)

        print(f"Saved manual + auto picks → {MANUAL_OUTPUT_CSV}")
    else:
        print("PERFORM_MANUAL=False → skipped manual picking.")


if __name__ == '__main__':
    main()
