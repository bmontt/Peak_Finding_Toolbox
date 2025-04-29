# abr_toolbox/cli.py
import argparse
import numpy as np
from toolbox.data_loader import load_audio_file, load_eeg_bids
from toolbox.detectors import BaseDetector
from toolbox.visualizer import plot_peaks  # assuming you have this

def main():
    p = argparse.ArgumentParser(prog="abrtoolbox")
    p.add_argument("--mode", choices=["abr", "audio", "hrir"], required=True,
                    help="Type of data to process")
    p.add_argument("--input", required=True, help="Path to data file or BIDS root")
    p.add_argument("--detector", default="abr_adaptive",
                    help="Which detector to use (abr_adaptive, scipy_peak, onset_env, hrir_xcorr)")
    # Generic detector params:
    p.add_argument("--prominence", type=float, default=0.01,
                    help="Prominence for scipy_peak detector")
    p.add_argument("--distance", type=float, default=0.001,
                    help="Min distance (s) for scipy_peak detector")
    p.add_argument("--threshold", type=float, default=0.1,
                    help="Energy threshold for onset_env detector")
    p.add_argument("--n_peaks", type=int, default=5,
                    help="Number of peaks for abr_adaptive detector")
    p.add_argument("--base_sigma", type=float, default=1.0,
                    help="Base sigma for abr_adaptive detector")
    args = p.parse_args()

    # Load signal
    if args.mode == "abr":
        raw = load_eeg_bids(args.input, subject="01")  # adjust subject logic as needed
        data = raw.get_data(picks=[0])[0]
        sr = raw.info["sfreq"]  # in Hz
    elif args.mode == "audio":
        data, sr = load_audio_file(args.input)
    elif args.mode == "hrir":
        # placeholder until you implement load_hrir
        from toolbox.data_loader import load_hrir
        data, sr = load_hrir(args.input)
    else:
        raise ValueError("Unsupported mode")

    # Instantiate detector
    det_params = {}
    if args.detector == "scipy_peak":
        det_params = {"prominence": args.prominence, "distance": args.distance}
    elif args.detector == "onset_env":
        det_params = {"threshold": args.threshold}
    elif args.detector == "abr_adaptive":
        det_params = {"n_peaks": args.n_peaks, "base_sigma": args.base_sigma}
    # for hrir_xcorr, you'll later pass template via kwargs

    detector = BaseDetector.create(args.detector, **det_params)
    peaks = detector.detect(data, sr)

    # Visualize or save
    plot_peaks(data, sr, peaks, output="peaks.png")
    print(f"Detected {len(peaks)} peaks: {peaks}")

if __name__ == "__main__":
    main()
