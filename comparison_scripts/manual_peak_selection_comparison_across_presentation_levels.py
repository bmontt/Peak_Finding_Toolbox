import os
import glob
import csv
import sys
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from collections import defaultdict
from tqdm import tqdm

# ————— Configuration —————
AVERAGE_DIR         = 'data/earndb_data/average'
MANUAL_OUTPUT_CSV   = 'bench/manual_peak_picks.csv'
PLOT_OUTDIR         = 'bench/abr_plots/manual'

PERFORM_MANUAL      = True
TIME_WINDOW         = (0, 15)   # ms window
SELECTED_STIMULI    = ['_F1', '1kHz_']
PRESENTATION_LEVELS = [20, 40, 60, 80, 100]


def load_existing_picks(csv_path):
    picks = {}
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                picks[(row['subject'], row['stimulus'])] = float(row['WaveV_ms'])
    return picks


def update_csv(csv_path, updates):
    data = load_existing_picks(csv_path)
    data.update(updates)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject', 'stimulus', 'WaveV_ms'])
        for (subj, stim), lat in sorted(data.items()):
            writer.writerow([subj, stim, lat])


def manual_overlay_pick(entries, prior_picks, out_png_path):
    entries_sorted = sorted(entries, key=lambda x: x[0])
    n = len(entries_sorted)
    ptps = [np.ptp(data) for _, _, data in entries_sorted]
    max_ptp = max(ptps) if ptps else 1.0
    offsets = [i*(max_ptp*1.1/n) for i in range(n)]

    picks = dict(prior_picks)
    line_data = {db:(times,data,off) for (db,times,data),off in zip(entries_sorted, offsets)}
    scatters = {}
    dragging = {'db': None}

    fig = plt.figure(figsize=(8,5))
    ax = fig.add_axes([0.05,0.15,0.9,0.8])
    btn_next_ax = fig.add_axes([0.45,0.02,0.1,0.05])
    btn_quit_ax = fig.add_axes([0.57,0.02,0.1,0.05])
    btn_next = Button(btn_next_ax, 'Next')
    btn_quit = Button(btn_quit_ax, 'Quit')

    for (db, times, data), off in zip(entries_sorted, offsets):
        ax.plot(times, data+off, label=f'{db} dB')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Amplitude + offset (µV)')
    ax.set_title('Click or drag ONE Wave V per trace, then press Next or Quit')
    ax.set_xlim(TIME_WINDOW)
    ax.legend(title='Level', loc='upper right')

    for db, latency in prior_picks.items():
        times, data, off = line_data[db]
        idx = np.argmin(np.abs(times-latency))
        scatters[db] = ax.scatter(latency, data[idx]+off, s=80, marker='o', color='k', picker=5)

    def on_click(event):
        if event.inaxes != ax: return
        x, y = event.xdata, event.ydata
        _, db_sel = min(
            (abs(data[np.argmin(np.abs(times-x))]+off - y), db)
            for db,(times,data,off) in line_data.items()
        )
        times_sel, data_sel, off_sel = line_data[db_sel]
        idx = np.argmin(np.abs(times_sel - x))
        latency = times_sel[idx]
        picks[db_sel] = latency
        y_plot = data_sel[idx] + off_sel
        if db_sel in scatters:
            scatters[db_sel].set_offsets((latency, y_plot))
        else:
            scatters[db_sel] = ax.scatter(latency, y_plot, s=80, marker='o', color='k', picker=5)
        fig.canvas.draw()

    def on_pick(event):
        for db, sc in scatters.items():
            if sc == event.artist:
                dragging['db'] = db
                break

    def on_motion(event):
        db = dragging['db']
        if db is None or event.inaxes != ax: return
        x = event.xdata
        times, data, off = line_data[db]
        idx = np.argmin(np.abs(times - x))
        latency = times[idx]
        picks[db] = latency
        scatters[db].set_offsets((latency, data[idx]+off))
        fig.canvas.draw()

    def on_release(event):
        dragging['db'] = None

    def on_next(event):
        if len(picks) < n:
            print(f"⚠️ Please pick all {n} points before proceeding.")
            return
        fig.savefig(out_png_path)
        plt.close(fig)

    def on_quit(event):
        plt.close(fig)
        sys.exit(0)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('pick_event',         on_pick)
    fig.canvas.mpl_connect('motion_notify_event',on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    btn_next.on_clicked(on_next)
    btn_quit.on_clicked(on_quit)

    plt.show()
    return picks


def main():
    os.makedirs(PLOT_OUTDIR, exist_ok=True)
    os.makedirs(os.path.dirname(MANUAL_OUTPUT_CSV), exist_ok=True)

    existing = load_existing_picks(MANUAL_OUTPUT_CSV)

    overlay_data = {
        'normal':   defaultdict(lambda: defaultdict(list)),
        'abnormal': defaultdict(lambda: defaultdict(list)),
    }

    for p in sorted(glob.glob(os.path.join(AVERAGE_DIR,'*.dat'))):
        base = os.path.splitext(p)[0]
        hea, dat = base+'.hea', base+'.dat'
        if not os.path.exists(hea) or not os.path.exists(dat): continue
        try:
            hdr = wfdb.rdheader(base); rec = wfdb.rdrecord(base)
        except FileNotFoundError:
            continue
        fs, gain_uv = hdr.fs, hdr.adc_gain[0]
        abr_uv = rec.p_signal[:,0] * gain_uv
        times  = np.arange(len(abr_uv)) / fs * 1000

        fname = os.path.basename(base)
        subj, stim = (fname.split('_',1) if '_' in fname else (fname,''))

        if not any(p in stim for p in SELECTED_STIMULI): continue
        if not any(f'ave{lvl}_' in stim for lvl in PRESENTATION_LEVELS): continue

        parts = stim.split('_')
        rep   = parts[-1]
        ave   = next(x for x in parts if x.startswith('ave'))
        db    = int(ave.replace('ave',''))
        group = 'normal' if '_F1' in stim else 'abnormal'

        mask = (times>=TIME_WINDOW[0]) & (times<=TIME_WINDOW[1])
        overlay_data[group][subj][rep].append((db, times[mask], abr_uv[mask]))

    tasks = [
        (group, subj, rep, entries)
        for group, subs in overlay_data.items()
        for subj, reps in subs.items()
        for rep, entries in reps.items()
        if entries
    ]

    for group, subj, rep, entries in tqdm(tasks, desc='Annotating', unit='plot'):
        print(f"\n=== Annotating {subj} ({group}, {rep}) ===")
        out_png = os.path.join(PLOT_OUTDIR, f'{subj}_{group}_{rep}_overlay.png')
        prior = {
            db: existing[(subj, f'{db}_F1_{rep}' if group=='normal' else f'1kHz_{db}_{rep}')]
            for db,_,_ in entries
            if (subj, f'{db}_F1_{rep}' if group=='normal' else f'1kHz_{db}_{rep}') in existing
        }
        picks = manual_overlay_pick(entries, prior, out_png)
        updates = {}
        for db, latency in picks.items():
            stim = f'{db}_F1_{rep}' if group=='normal' else f'1kHz_{db}_{rep}'
            updates[(subj, stim)] = latency
        update_csv(MANUAL_OUTPUT_CSV, updates)

    print(f"\nAll done. Human annotations saved to {MANUAL_OUTPUT_CSV}")

if __name__ == '__main__':
    main()
