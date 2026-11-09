#!/usr/bin/env python3
"""HEP per expert-scored sleep stage, raw vs. CFAR (six-channel HSP nights): does cleaning change the known
wake/sleep modulation of the HEP?

Same nights, filter and clean-epoch rule as stage_band_cleaning.py. Per stage: R-peak-locked average (-0.2..0.6 s)
of raw EEG and of EEG after CFAR (cohort pattern), and the ECG average. Writes stage_hep_amp.parquet.
Run: venv/bin/python "paper utils/Paper CFA/stage_hep_amp.py" --limit 300 --workers 40
"""
import argparse, glob, os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import make_cleaning_demo as cd
import stage_band_cleaning as sb
from ica_ecg_component_variance import detect_r_peaks, epoch_around_r_peaks


def process(job):
    pid, path = job
    try:
        expert = sb.expert_stages(path)
        if expert is None:
            return []
        night = sb.load_night(path, stage=False)
        if isinstance(night, str):
            return []
        eeg, ecg, sf, _ = night
        peaks = detect_r_peaks(ecg, sf)
        clean = cd.project_out(eeg, sb.U_POP)
        n = int(sb.EPOCH_SEC * sf)
        n_ep = min(len(expert), eeg.shape[1] // n)
        ep = eeg[:, :n_ep * n].reshape(6, n_ep, n)
        ok = (ep.max(2) - ep.min(2)).max(0) < sb.PTP_UV
        be = peaks // n
        rows = []
        for st in sb.STAGES:
            sel = (expert[:n_ep] == st) & ok
            pk = peaks[(be < n_ep) & sel[np.minimum(be, n_ep - 1)]]
            if len(pk) < 200:
                continue
            row = {"patient_id": pid, "stage": st, "n_beats": int(len(pk)), "sfreq": sf}
            for m, x in [("raw", eeg), ("cfar", clean)]:
                row[f"hep_{m}"] = epoch_around_r_peaks(x, pk, sf, -0.2, 0.6).mean(0).astype(np.float32).ravel().tolist()
            row["ecg"] = epoch_around_r_peaks(ecg[None], pk, sf, -0.2, 0.6).mean(0)[0].astype(np.float32).tolist()
            rows.append(row)
        return rows
    except Exception:  # noqa: BLE001 -- odd nights skipped
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--workers", type=int, default=40)
    a = ap.parse_args()
    done = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(sb.CACHE, "*.parquet"))}
    c = pd.read_parquet(os.path.join(cd.HERE, "cfa_combined.parquet"), columns=["patient_id", "edf_path"]).drop_duplicates("patient_id")
    c = c[c.patient_id.isin(done)].sample(frac=1, random_state=cd.SEED + 3).head(a.limit)
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(process, zip(c.patient_id, c.edf_path)):
            rows += r
    pd.DataFrame(rows).to_parquet(os.path.join(cd.HERE, "stage_hep_amp.parquet"))
    print("rows", len(rows), "patients", len({r["patient_id"] for r in rows}))


if __name__ == "__main__":
    main()
