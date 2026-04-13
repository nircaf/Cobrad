"""
filter_ecg_edfs.py

Scans a directory recursively for EDF files and prints (one per line) only
those that contain at least one ECG / EKG channel.  The output is consumed by
init_zellij_sessions.sh to populate the worker queue.

Usage:
    python filter_ecg_edfs.py <edf_root_dir>
"""

import sys
import os
import glob
import warnings

import mne

mne.set_log_level("ERROR")


def has_ecg_channel(edf_path: str) -> bool:
    """Return True if the EDF file contains at least one ECG/EKG channel."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        ch_lower = [ch.lower() for ch in raw.ch_names]
        return any("ecg" in ch or "ekg" in ch for ch in ch_lower)
    except Exception:
        return False


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <edf_root_dir>", file=sys.stderr)
        sys.exit(1)

    edf_root = sys.argv[1]
    if not os.path.isdir(edf_root):
        print(f"Directory not found: {edf_root}", file=sys.stderr)
        sys.exit(1)

    edf_files = sorted(
        glob.glob(os.path.join(edf_root, "**", "*.edf"), recursive=True)
        + glob.glob(os.path.join(edf_root, "**", "*.EDF"), recursive=True),
        reverse=True,
    )

    total = len(edf_files)
    kept = 0
    for path in edf_files:
        if has_ecg_channel(path):
            print(path)
            print(f"[ok]   ECG found:       {path}", file=sys.stderr)
            kept += 1
        else:
            print(f"[skip] no ECG channel: {path}", file=sys.stderr)

    print(f"[filter_ecg_edfs] {kept}/{total} files have an ECG channel.", file=sys.stderr)


if __name__ == "__main__":
    main()
