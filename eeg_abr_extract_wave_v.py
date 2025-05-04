import os
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from mne_bids import BIDSPath, read_raw_bids


# empty list to store peak data for logging
log = []
max_subject_id = 20  # Change as needed
mode = "individual"     # Options: average, subtract, individual
# Gaussian smoothing parameter (sigma) 
sigma = 0.06   # Tune based on expected noise level
wave_labels = ["Wave I","Wave II","Wave III","Wave IV","Wave V"]

def scale_prominence(snr, base=0.01, high=0.1, exponent=3):
    """
    Prominence threshold goes from `high` at snr=0
    down to `base` at snr=1, following a sharp power curve.
    """
    snr = np.clip(snr, 0, 1)
    return base + (high - base) * (1 - snr**exponent)

# def plot_wave_latency(df):
#     sns.set_theme(style="whitegrid")
#     intensities = sorted(df['Ammplitude (µV)'].unique())
#     stim_rates = sorted(df['Stimulus'].unique())
    
#     fig, axes = plt.subplots(len(amplitudes), len(stim_rates), figsize=(14, 6), sharey=True)
    
#     for i, intensity in enumerate(intensities):
#         ax = axes[i]
#         intensity_df = df[df['Intensity'] == intensity]
        
#         for freq in frequencies:
#             freq_df = intensity_df[intensity_df['Frequency'] == freq]
            
#             # Plot left ear
#             sns.lineplot(
#                 data=freq_df[freq_df['Ear'] == 'Left'],
#                 x='Stimulus Rate', y='Latency (ms)',
#                 marker='X', linestyle='--',
#                 label=f'{freq} Hz - Left', ax=ax
#             )

#             # Plot right ear
#             sns.lineplot(
#                 data=freq_df[freq_df['Ear'] == 'Right'],
#                 x='Stimulus Rate', y='Latency (ms)',
#                 marker='o', linestyle='-',
#                 label=f'{freq} Hz - Right', ax=ax
#             )

#         ax.set_title(f'{intensity} dB peSPL')
#         ax.set_xlabel('Stimulus rate (Hz)')
#         if i == 0:
#             ax.set_ylabel('Wave V latency (ms)')
#         else:
#             ax.set_ylabel('')

#     plt.tight_layout()
#     plt.legend()
#     print("Wave V Latency Plot generated")
#     plt.show()
            
# =============================================================================
# Based on ABRA approach, hypbrid deep learning + rule based detection
# once CNN predicts first peak (I), anchor point is determined 
# and classical Gaussian smoothing and conventional peak finding is applied 
# to find next n waves
# 
# (data driven + deterministic combo to adapt to both normal and abnormal responses)
def predict_wave_V_latency(waveform, times, signal_to_noise_ratio, sigma):
    # blow up smoothing at low‐SNR, collapse to near‑nominal at high‐SNR
    adj_sigma = sigma / (signal_to_noise_ratio + 0.1)
    # but don’t let it go below 0.3× or above 4× your base
    adj_sigma = np.clip(adj_sigma, 0.3*sigma, 4*sigma)
    smooth = gaussian_filter1d(waveform, sigma=adj_sigma)

    # define mask
    times_ms = times * 1000
    mask = (times_ms >= 4.6) & (times_ms <= 9.2)
    segment = smooth[mask]
    if segment.size == 0:
        return None
    prom = scale_prominence(signal_to_noise_ratio)
    peaks, props = find_peaks(segment, prominence=prom)

    if peaks.size:
        # Pick the peak with the largest prominence
        best = peaks[np.argmax(props['prominences'])]

        global_idx = np.where(mask)[0][best]
        return global_idx
    else:
        # otherwise take the absolute max in the window
        rel = np.argmax(np.abs(segment))
        return np.where(mask)[0][rel]


def compute_snr(data, times_ms, signal_window=(0, 10), noise_window=(-5, 0)):
    """Compute SNR (in dB) based on RMS in signal vs. noise window."""
    signal_mask = (times_ms >= signal_window[0]) & (times_ms <= signal_window[1])
    noise_mask = (times_ms >= noise_window[0]) & (times_ms <= noise_window[1])
    
    signal_rms = np.sqrt(np.mean(data[signal_mask] ** 2))
    noise_rms = np.sqrt(np.mean(data[noise_mask] ** 2))
    
    # avoid division by zero (compute dB)
    snr_db = 20 * np.log10(signal_rms / noise_rms) if noise_rms > 0 else np.inf

    snr_min, snr_max = -5, 10  # dB range for normalization
    # Clip and normalize SNR (in dB) to [0, 1].
    snr_clipped = np.clip(snr_db, snr_min, snr_max)
    return (snr_clipped - snr_min)/(snr_max - snr_min)


def load_valid_event_onsets(bids_path, intensity=81, token=2, rates=[40]):
    """ Locate onsets for events matching criteria from events.tsv file """
    events_tsv = bids_path.copy().update(extension='.tsv', suffix='events')
    
    try:
        df = pd.read_csv(events_tsv, sep='\t')
    except FileNotFoundError:
        print(f"[{bids_path.subject}] No events.tsv found.")
        return np.array([])

    valid_events = df[
        (df['intensity'] == intensity) &
        (df['token'] == token) &
        (df['rate'].isin(rates))
    ]

    if valid_events.empty:
        print(f"[{bids_path.subject}] No matching stimulus conditions (80dB, 2kHz, 40Hz).")
        return np.array([])

    return valid_events['onset'].values

def process_subject(bids_root, subject_id, mode, sigma):
    bids_path = BIDSPath(root=bids_root,
                        subject=subject_id,
                        session=None, 
                        task="rates",  # pABR
                        run=None,
                        datatype="eeg")
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()
    if subject_id == "01":  # right channel is inverted for subject 01
        raw.apply_function(lambda x: -x, picks=[0])
    if subject_id == "16":
        print(f"[Subject {subject_id}] Skipping since subject fell asleep.")
        return
    # match events from annotation to onset
    raw.set_montage("standard_1020")  # Set montage since channel locs aren't embedded
    valid_onsets = load_valid_event_onsets(bids_path)
    all_events, _ = mne.events_from_annotations(raw)
    sfreq = raw.info['sfreq']
    events_onset_sec = all_events[:, 0] / sfreq
    mask = np.isclose(events_onset_sec[:, None], valid_onsets[None, :], atol=1e-3).any(axis=1)
    filtered_events = all_events[mask]
    
    if len(filtered_events) == 0:
        print(f"[Subject {subject_id}] No valid events found.")
        return
    
    tmin, tmax = -0.001, 0.015  # Window: 3 ms before to 15 ms after stimulus onset
    epochs = mne.Epochs(raw, filtered_events, event_id=None, tmin=tmin, tmax=tmax, 
                        baseline=(None, 0), preload=True, reject=dict(eeg=30e-6), verbose=False)
    epochs.filter(l_freq=100, h_freq=3000, method='fir')  # 100-3000 Hz bandpass
    evoked = epochs.average()  # Averaging epochs
    
    if mode == "average":
        combined_data = np.mean(evoked.data, axis=0, keepdims=True).reshape(1, -1)
        info_new = mne.create_info(ch_names=["Averaged_ABR"], sfreq=evoked.info["sfreq"], ch_types=["eeg"])
        evoked_final = mne.EvokedArray(combined_data, info_new, tmin=evoked.tmin)
        plt_title = "Averaged pABR Evoked Response"
    elif mode == "subtract":
        combined_data = (evoked.data[0, :] - evoked.data[1, :]).reshape(1, -1)
        info_new = mne.create_info(ch_names=["Right - Left Difference"], sfreq=evoked.info["sfreq"], ch_types=["eeg"])
        evoked_final = mne.EvokedArray(combined_data, info_new, tmin=evoked.tmin)
        plt_title = "pABR Difference (Right - Left)"
    else:
        # INDIVIDUAL MODE: plot both ears on one figure
        fig, ax = plt.subplots(figsize=(10, 6))
        times_ms = evoked.times * 1000

        for ch_idx, _ in enumerate(evoked.ch_names):
            data_uv = evoked.data[ch_idx] * 1e6  # µV
            snr = compute_snr(data_uv, times_ms)
            waveV_idx = predict_wave_V_latency(data_uv, evoked.times, snr, sigma)

            # smoothing + find all peaks in 0–15 ms
            # blow up smoothing at low‐SNR, collapse to near‑nominal at high‐SNR
            adj_sigma = sigma / (snr + 0.1)
            # but don’t let it go below 0.3× or above 4× your base
            adj_sigma = np.clip(adj_sigma, 0.3*sigma, 4*sigma)
            smooth = gaussian_filter1d(data_uv, sigma=adj_sigma)
            mask15 = (times_ms >= 0) & (times_ms <= 15)
            dt = times_ms[1] - times_ms[0]
            min_dist = int(0.7 / dt)
            prom = scale_prominence(snr)
            peaks, _ = find_peaks(smooth[mask15], prominence=prom, distance=min_dist)
            global_peaks = np.where(mask15)[0][peaks]
            if waveV_idx not in global_peaks:
                global_peaks = np.sort(np.append(global_peaks, waveV_idx))

            ear = "Left" if ch_idx == 0 else "Right"
            for i, label in enumerate(wave_labels):
                if i < len(global_peaks):
                    idx = global_peaks[i]
                    amp = data_uv[idx]
                    lat = times_ms[idx]
                    log.append([subject_id, ear, label, amp, lat, snr])
                else:
                    log.append([subject_id, ear, label, "No Peak", "No Peak", snr])

            # plot the waveform
            ax.plot(times_ms, data_uv, linewidth=1.5, label=f"{ear} (SNR {snr*100:.1f}%)")

            # now annotate every detected peak
            for j, label in enumerate(wave_labels):
                if j < len(global_peaks):
                    idx = global_peaks[j]
                    amp = data_uv[idx]
                    lat = times_ms[idx]
                    ax.plot(lat, amp, 'o', ms=6, color='red')
                    ax.annotate(f"{label}\n{lat:.1f} ms",
                                xy=(lat, amp),
                                xytext=(lat, amp + 0.1*(ax.get_ylim()[1]-ax.get_ylim()[0])),
                                arrowprops=dict(facecolor='red', arrowstyle='->', lw=1.2),
                                ha='center', fontsize=9, color='red')

        ax.set_title(f"Subject {subject_id} — Individual Ears")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()

        outdir = os.path.join(bids_root, 'per_ear_one_stim_wave_V_adj_snr')
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, f"subject_{subject_id}_ears.png"), dpi=300)
        plt.close(fig)
        return

    # for averaged or subtracted data
    # convert data to µV for consistency
    evoked_final._data *= 1e6  # now in µV
    data = evoked_final.data[0] # single channel data
    times = evoked_final.times      # seconds
    times_ms = times * 1000         # in ms
    # based on ABRA approach
    
    snr = compute_snr(data, times_ms)
    print(f"[Subject {subject_id}] SNR = {snr*100:.2f}%")
    waveV_idx = predict_wave_V_latency(data, times, snr, sigma)
    # Gaussian smoothing waveform --> more robust peak detection
    adj_sigma = sigma * (1 - snr)  # adjust sigma based on SNR
    smoothed_data = gaussian_filter1d(data, sigma=adj_sigma)
    # rule-based for peaks II–V
    # analysis window a bit before I to end of response (e.g., 15 ms).
    analysis_mask = (times_ms >= 0) & (times_ms <= 15)
    dt_ms = times_ms[1] - times_ms[0]
    min_distance_samples = int(0.7 / dt_ms)  # peaks must be >= 0.7 ms apart (retool for 4/5 complex)

    adj_prominence = scale_prominence(snr)
    detected_peaks, _ = find_peaks(smoothed_data[analysis_mask],
                            prominence=adj_prominence,
                            distance=min_distance_samples)

    # map detected peaks back to global indices.
    global_peaks = np.where(analysis_mask)[0][detected_peaks]
    if waveV_idx not in global_peaks:
        global_peaks = np.sort(np.append(global_peaks, [waveV_idx]))
    else:
        global_peaks = np.sort(global_peaks)
    
    # Label the first five peaks as I-V
    for i, label in enumerate(wave_labels):
        if i < len(global_peaks):
            idx = global_peaks[i]
            amp = data[idx]
            lat = times_ms[idx]
            log.append([subject_id, "N/A", label, amp, lat, snr])
        else:
            log.append([subject_id, "N/A", label, "No Peak", "No Peak", snr])

    # plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times_ms, data, color='#1f77b4', linewidth=2)
    ax.set_title(f"{plt_title} - Subject {subject_id}", fontsize=12)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Amplitude (µV)', fontsize=10)
    
    # drawing vertical reference lines (e.g., at typical ABR latencies)
    for ref_line in [1.5, 3.5, 6.0]:
        ax.axvline(x=ref_line, color='k', linestyle='--', alpha=0.7)
    
    # annotate detected peaks
    for label, amp_lat in zip(wave_labels, global_peaks):
        lat = times_ms[amp_lat]
        amp = data[amp_lat]
        ax.plot(lat, amp, 'ro', ms=6)
        ax.annotate(f"{label}\n{lat:.1f} ms", xy=(lat, amp),
                    xytext=(lat, amp + 0.1*(ax.get_ylim()[1]-ax.get_ylim()[0])),
                    arrowprops=dict(facecolor='red', arrowstyle="->", lw=1.5),
                    fontsize=10, color='red', ha='center')
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    output_dir = os.path.join(bids_root, 'one_stim_wave_V_adj_snr')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"subject_{subject_id}_evoked.png"), dpi=300)
    plt.close(fig)

def main(max_subject_id, mode, sigma):
    bids_root = 'abr_spring/dryad_dataset/pABR-rates-eeg-dataset'
    
    for i in tqdm(range(1, max_subject_id + 1), desc="Processing Subjects", unit="subject"):
        subject_id = f"{i:02d}"
        process_subject(bids_root, subject_id, mode, sigma)

    # log data to df and save to csv
    log_df = pd.DataFrame(log, columns=["Subject ID", "Ear", "Wave", "Amplitude (µV)", "Latency (ms)", "SNR (%)"])
    csv_path = os.path.join(bids_root, 'output', 'waveV_data_log.csv')
    log_df.to_csv(csv_path, index=False)
    wave_V_df = log_df[log_df['Wave'] == 'Wave V'].copy()
    print(wave_V_df.head())
    # plot_wave_latency(wave_V_df)

main(max_subject_id, mode, sigma)