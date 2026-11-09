#!/usr/bin/env python3
"""HEP spectrum per sleep stage, to set beside stage_band_cleaning.py's raw / ICA / CFAR / ps-CFAR spectra.

Per patient already in stage_band_cache/ (same nights, filter and artifact-epoch rule): beats inside the
clean 30-s epochs of each expert-scored stage are averaged into one HEP (raw EEG, -0.3..0.4 s), and the
HEP's periodogram (Hann, zero-padded to 4 s so it lands on the same FREQS grid) gets the same features.
The HEP lasts 0.7 s, so its spectrum has ~1.4-Hz resolution and nothing below ~1.4 Hz is real.

Writes stage_band_cache/hep/<pid>.parquet, same columns as stage_band_cache/<pid>.parquet (condition "hep").
Run: venv/bin/python "paper utils/Paper CFA/stage_band_hep.py" --limit 500 --workers 48
"""
import argparse, glob, os, traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import periodogram

import make_cleaning_demo as cd
import stage_band_cleaning as sb
from ica_ecg_component_variance import detect_r_peaks, epoch_around_r_peaks

OUT = os.path.join(sb.CACHE, "hep")


def process(job):
    pid, path = job
    out = os.path.join(OUT, f"{pid}.parquet")
    if os.path.exists(out):
        return pid, "cached"
    try:
        expert = sb.expert_stages(path)
        if expert is None:
            return pid, "no_expert"
        night = sb.load_night(path, stage=False)
        if isinstance(night, str):
            return pid, night
        eeg, ecg, sf, _ = night
        peaks = detect_r_peaks(ecg, sf)
        n = int(sb.EPOCH_SEC * sf)
        n_ep = min(len(expert), eeg.shape[1] // n)
        ep = eeg[:, :n_ep * n].reshape(6, n_ep, n)
        ok = (ep.max(2) - ep.min(2)).max(0) < sb.PTP_UV  # same rule as stage_rows
        beat_ep = peaks // n
        rows = []
        for st in sb.STAGES:
            sel = (expert[:n_ep] == st) & ok
            pk = peaks[(beat_ep < n_ep) & sel[np.minimum(beat_ep, n_ep - 1)]]
            if sel.sum() < 4 or len(pk) < 100:
                continue
            hep = epoch_around_r_peaks(eeg, pk, sf).mean(0)
            f, p = periodogram(hep, sf, window="hann", nfft=int(4 * sf), axis=1)
            psd = np.stack([np.interp(sb.FREQS, f, c) for c in p])
            for ch, spec in [("mean", psd.mean(0)), *zip(cd.SITES, psd)]:
                rows.append({"patient_id": pid, "staging": "expert", "stage": st, "condition": "hep", "channel": ch,
                             "n_epochs": int(sel.sum()), "n_beats": len(pk), "psd": spec.astype(np.float32),
                             **sb.features(spec)})
        pd.DataFrame(rows).to_parquet(out)
        return pid, f"ok {len(rows)}"
    except Exception as exc:  # noqa: BLE001 -- odd EDFs skipped, reason logged
        return pid, f"error {type(exc).__name__}: {exc} | {traceback.format_exc(limit=1).splitlines()[-1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    done = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(sb.CACHE, "*.parquet"))}
    cfa = pd.read_parquet(os.path.join(cd.HERE, "cfa_combined.parquet"), columns=["patient_id", "edf_path"]).drop_duplicates("patient_id")
    cfa = cfa[cfa.patient_id.isin(done)].sample(frac=1, random_state=cd.SEED).head(a.limit)
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (pid, status) in enumerate(ex.map(process, zip(cfa.patient_id, cfa.edf_path)), 1):
            print(f"[{i}/{len(cfa)}] {pid} {status}", flush=True)


if __name__ == "__main__":
    main()
