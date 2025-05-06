# run_abr_annotator.py
import os, sys
from manual_peak_selection_comparison_across_presentation_levels import main   # assume you refactored your big script into abr_annotator.py

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print("Fatal error:", e, file=sys.stderr)
        sys.exit(1)
