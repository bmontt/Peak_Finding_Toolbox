import os
import time
import pandas as pd
from pathlib import Path

# ABRPresto imports
from ABRpresto.main  import run_fit
from ABRpresto.utils import load_fits, compare_thresholds

# Your toolbox imports
from toolbox.data_loader import load_eeg_epochs
from toolbox.peak_finder    import detect_peaks

def run_abrpresto(csv_root: str, outdir: str) -> pd.DataFrame:
    """
    Run ABRPresto on the example CSV data (recursive=True).
    Returns a DataFrame with a 'threshold' column.
    """
    os.makedirs(outdir, exist_ok=True)
    # --loader csv tells ABRPresto to ingest CSV files
    run_fit(
        csv_root,
        'csv',
        outdir
    )
    return load_fits(outdir)

def run_mine(bids_root: str, subject_ids: list[str]) -> pd.DataFrame:
    """
    Run your peak-based pipeline on the same levels (subject_ids),
    inferring threshold = subject_id (dB), and capturing Wave V latency.
    """
    rows = []
    for sid in subject_ids:
        epochs = load_eeg_epochs(bids_root, sid)
        ev = epochs.average()
        data = ev.data[0] * 1e6      # µV
        times = ev.times * 1000      # ms

        peaks = detect_peaks(data, times,
                             n_peaks=5,
                             base_sigma=0.06,
                             mode='abr')
        # If no peaks, mark threshold as NaN
        if len(peaks) == 0:
            thresh = float('nan')
            lat     = float('nan')
        else:
            # we assume sid is the dB level (e.g. "70")
            thresh = float(sid)
            lat     = times[peaks[-1]]   # Wave V is last of 5

        rows.append({'subject': sid,
                     'threshold_my': thresh,
                     'time_my_ms':  lat})
    return pd.DataFrame(rows)

def main():
    # Paths in the ABRPresto repo
    csv_root   = 'src/abrpresto/example_data'  # adjust if needed
    presto_out = 'bench/abrpresto_out'
    # We’ll reuse the same ids that ABRPresto uses: the CSV filenames are like “70.csv”, “75.csv”, etc.
    # So subject_ids = [basename without extension for each CSV]
    subject_ids = [Path(f.path).stem for f in os.scandir(csv_root) if f.name.endswith('.csv')]
    
    # 1) ABRPresto
    t0 = time.perf_counter()
    df_pre = run_abrpresto(csv_root, presto_out)
    dt_pre = time.perf_counter() - t0

    # 2) Your method
    # If you converted the CSVs to BIDS, point bids_root there. 
    # Or just reuse csv_root and hack load_eeg_epochs to accept CSV.
    bids_root = 'src/abrpresto/example_data'  
    t1 = time.perf_counter()
    df_my = run_mine(bids_root, subject_ids)
    dt_my = time.perf_counter() - t1

    # 3) Compare
    df_cmp = compare_thresholds(df_pre,
                                df_my,
                                col1='threshold',
                                col2='threshold_my')
    df_cmp['time_my_ms'] = df_my.set_index('subject').loc[df_cmp.subject, 'time_my_ms']
    df_cmp['runtime_presto_s'] = dt_pre
    df_cmp['runtime_my_s']     = dt_my

    # 4) Save & print
    os.makedirs('bench', exist_ok=True)
    df_cmp.to_csv('bench/threshold_comparison.csv', index=False)
    print(df_cmp)
    print(f"\nRuntimes — ABRPresto: {dt_pre:.2f}s, Mine: {dt_my:.2f}s")

if __name__ == '__main__':
    main()