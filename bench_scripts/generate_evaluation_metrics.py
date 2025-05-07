"""
generate_evaluation_metrics.py

Compute Wave V comparison metrics between auto and manual picks,
ignoring rows with 0 manual labels. Assumes row-by-row alignment.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def compute_metrics(manual_csv, auto_csv, tolerances=(0.5, 1.0, 1.5, 2.0)):
    """
    Compares aligned CSVs row-by-row:
      manual: subject, stimulus, WaveV_ms
      auto:   subject, stimulus, Auto_WaveV_ms

    Returns:
      metrics: dict of various comparison metrics
    """
    df_m = pd.read_csv(manual_csv)
    df_a = pd.read_csv(auto_csv)

    if len(df_m) != len(df_a):
        raise ValueError("Manual and Auto CSVs must have the same number of rows")

    # ensure same ordering
    if not ((df_m['subject'] == df_a['subject']) & (df_m['stimulus'] == df_a['stimulus'])).all():
        raise ValueError("Mismatch in subject/stimulus ordering between files")

    manual = df_m['WaveV_ms'].values
    auto   = df_a['Auto_WaveV_ms'].values

    # exclude zeros
    mask = manual != 0
    manual = manual[mask]
    auto   = auto[mask]

    # raw differences
    diffs_signed = auto - manual
    diffs        = np.abs(diffs_signed)

    # basic
    mean_abs    = np.mean(diffs)
    median_abs  = np.median(diffs)
    iqr_abs     = np.percentile(diffs, 75) - np.percentile(diffs, 25)
    rmse        = np.sqrt(np.mean(diffs**2))
    bias        = np.mean(diffs_signed)
    sd_diff     = np.std(diffs_signed, ddof=1)
    loa_low     = bias - 1.96 * sd_diff
    loa_high    = bias + 1.96 * sd_diff
    r, pval     = pearsonr(manual, auto)

    # accuracy at multiple tolerances
    acc = {tol: (diffs <= tol).mean() * 100.0 for tol in tolerances}

    return {
        'N_pairs':          len(diffs),
        'Mean |Δt| (ms)':   mean_abs,
        'Median |Δt| (ms)': median_abs,
        'IQR |Δt| (ms)':    iqr_abs,
        'RMSE (ms)':        rmse,
        'Bias (ms)':        bias,
        'LoA lower (ms)':   loa_low,
        'LoA upper (ms)':   loa_high,
        'Pearson r':        r,
        'Pearson p':        pval,
        'Accuracy':         acc
    }

if __name__ == '__main__':
    manual_csv = "bench/manual_peak_picks.csv"
    auto_csv   = "bench/auto_peak_picks.csv"
    tolerances = [0.5, 1.0, 1.5, 2.0]

    m = compute_metrics(manual_csv, auto_csv, tolerances)

    print(f"N          : {m['N_pairs']}")
    print(f"Mean |Δt|  : {m['Mean |Δt| (ms)']:.3f} ms")
    print(f"Median |Δt|: {m['Median |Δt| (ms)']:.3f} ms (IQR {m['IQR |Δt| (ms)']:.3f})")
    print(f"RMSE       : {m['RMSE (ms)']:.3f} ms")
    print(f"Bias       : {m['Bias (ms)']:+.3f} ms")
    print(f"95% LoA    : [{m['LoA lower (ms)']:+.3f}, {m['LoA upper (ms)']:+.3f}] ms")
    print(f"Pearson r  : {m['Pearson r']:.3f} (p ={m['Pearson p']:.3g})")
    for tol, acc in m['Accuracy'].items():
        print(f"Acc ≤ {tol:.1f} ms: {acc:5.2f}%")
