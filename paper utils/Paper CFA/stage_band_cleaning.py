#!/usr/bin/env python3
"""Dominant EEG frequency band per sleep stage, before vs. after cardiac cleaning.

Per patient (same cohort + 6-site mastoid montage as make_cleaning_demo.py):
  1. Stage the whole night with YASA (30-s epochs, as window_stage_sensitivity.py).
  2. Clean the whole night four ways:
       raw   1-40 Hz band-pass only
       ica   ECG-informed ICA (make_cleaning_demo.ica_clean, Section 2.4)
       cfar  CFAR: project out the population pattern u_pop (paper_stats.json, make_cleaning_demo.py); ECG-free
       own   patient-specific CFAR: project out the patient's own R-peak-locked pattern
  3. For every 30-s epoch of W / N1 / N2 / N3 / R (epochs with raw ptp > PTP_UV dropped),
     Welch PSD per channel, averaged over the stage's epochs. Done twice: staging = "yasa" (YASA on raw C4)
     and "expert" (technician hypnogram, *_task-psg_sleep_annotations.csv; epoch 1 = recording start).
  3b. Per-epoch file stage_band_cache/epochs/<pid>.parquet: expert stage and YASA re-run on each cleaned
     signal's C4 (yasa_<cond>), so the dashboard can score staging accuracy per cleaning method.
  4. Band powers 1-30 Hz (abs and relative), dominant band = band with most relative power, and
     peak frequency = max of the PSD after removing a log-log linear (1/f) fit, with its band.

One parquet per patient in stage_band_cache/ (resumable; the dashboard reads it while running).
Validation by artifact injection: stage_band_injection.py (reuses load_night/stage_rows below).

Run: venv/bin/python "paper utils/Paper CFA/stage_band_cleaning.py" --limit 8 --workers 8   # sample
     venv/bin/python "paper utils/Paper CFA/stage_band_cleaning.py" --workers 48              # all
Dashboard: bash "paper utils/Paper CFA/stage_band_dashboard.sh"
"""
import argparse
import glob
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor

import mne
import numpy as np
import pandas as pd
from scipy.signal import medfilt, welch

import make_cleaning_demo as cd
from channel_utils import canonicalize
from ica_ecg_component_variance import channel_names, detect_r_peaks

sys.path.insert(0, os.path.dirname(os.path.dirname(cd.HERE)))
from HEP_parquet_generation import prep_for_yasa  # noqa: E402

CACHE = os.path.join(cd.HERE, "stage_band_cache")
EPOCH_CACHE = os.path.join(CACHE, "epochs")
EPOCH_SEC, L_FREQ, H_FREQ, PTP_UV = 30, 1.0, 40.0, 500.0
STAGES = ["W", "N1", "N2", "N3", "R"]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 16), "beta": (16, 30)}
# ponytail: whole-cohort u_pop (each patient is ~1/4000 of its training set); cross-fit if leakage ever matters
U_POP = np.array([json.load(open(os.path.join(cd.HERE, "paper_stats.json")))["cleaning_demo"]["u_pop"][s] for s in cd.SITES])
CONDITIONS = ["raw", "ica", "cfar", "own"]
FREQS = np.arange(L_FREQ, H_FREQ + 0.01, 0.25)  # common grid so spectra average across sfreqs


def band_of(f):
    return next((b for b, (lo, hi) in BANDS.items() if lo <= f < hi), "beta")


def features(psd):
    """psd on FREQS -> abs/rel band powers (1-30 Hz), dominant band, 1/f-corrected peak frequency.
    1.25-Hz median filter first: some recordings carry narrow 20/40 Hz device lines."""
    df = FREQS[1] - FREQS[0]
    psd = medfilt(psd, 5)
    absb = {b: float(psd[(FREQS >= lo) & (FREQS < hi)].sum() * df) for b, (lo, hi) in BANDS.items()}
    tot = sum(absb.values())
    m = (FREQS < 30) & ((FREQS < 19) | (FREQS > 21))  # 20-Hz device line survives the median filter in some nights
    lf, lp = np.log10(FREQS[m]), np.log10(psd[m])
    resid = lp - np.polyval(np.polyfit(lf, lp, 1), lf)
    peak = float(FREQS[m][np.argmax(resid)])
    return {**{f"abs_{b}": v for b, v in absb.items()}, **{f"rel_{b}": v / tot for b, v in absb.items()},
            "dominant_band": max(absb, key=absb.get), "peak_hz": peak, "peak_band": band_of(peak)}


def expert_stages(edf_path):
    """Technician hypnogram (30-s epochs from recording start) or None. Unscored ('L' etc.) -> ''."""
    f = glob.glob(os.path.join(os.path.dirname(edf_path), "*sleep_annotations.csv"))
    if not f:
        return None
    st = pd.read_csv(f[0], usecols=["Epoch", "Stage"]).drop_duplicates("Epoch").set_index("Epoch").Stage
    st = st.reindex(range(1, st.index.max() + 1)).astype(str)
    return np.where(st.isin(STAGES), st, "").astype("<U2")


def yasa_stages(raw_or_array, ch, sf=None):
    import yasa
    if sf is not None:  # (n,) uV array -> one-channel Raw
        raw_or_array = mne.io.RawArray(raw_or_array[None] * 1e-6, mne.create_info([ch], sf, "eeg"), verbose="ERROR")
    return np.asarray(yasa.SleepStaging(raw_or_array, eeg_name=ch).predict())


def load_night(path, stage=True):
    """Whole night, 6-site montage only -> (eeg uV in cd.SITES order, ecg uV, sf, YASA stages on raw C4 or None)
    or a reason str."""
    raw = mne.io.read_raw_edf(path, preload=False, encoding="latin1", verbose="ERROR")
    eeg_names, ecg_name = channel_names(raw)
    canon = canonicalize(pd.Series(eeg_names)).tolist()
    if len(eeg_names) != 6 or sorted(canon) != sorted(cd.SITES):
        return "montage"
    raw = raw.pick(eeg_names + [ecg_name]).load_data(verbose="ERROR")
    raw.set_channel_types({ecg_name: "ecg"}, on_unit_change="ignore", verbose="ERROR")
    sf = float(raw.info["sfreq"])
    stage_ch = eeg_names[canon.index("C4")]
    stages = yasa_stages(prep_for_yasa(raw.copy(), stage_ch), stage_ch) if stage else None
    raw.filter(L_FREQ, min(H_FREQ, sf / 2 - 0.5), picks="all", verbose="ERROR")
    data = raw.get_data() * 1e6
    return data[:-1][[canon.index(s) for s in cd.SITES]], data[-1], sf, stages


def stage_rows(sigs, stagings, ok_from, sf, **meta):
    """Per-stage mean PSD + features for each signal in sigs and each staging in stagings ({name: labels})."""
    n = int(EPOCH_SEC * sf)
    n_ep = min(min(len(v) for v in stagings.values()), ok_from.shape[1] // n)
    ep = ok_from[:, :n_ep * n].reshape(6, n_ep, n)
    ok = (ep.max(2) - ep.min(2)).max(0) < PTP_UV  # artifact epochs judged on one signal, same epochs for all
    rows = []
    for cond, x in sigs.items():
        f, p = welch(x[:, :n_ep * n].reshape(6, n_ep, n), sf, nperseg=int(4 * sf), axis=-1)
        p = np.stack([[np.interp(FREQS, f, p[c, e]) for e in range(n_ep)] for c in range(6)])  # (ch, ep, freq)
        for staging, labels in stagings.items():
            for st in STAGES:
                sel = (labels[:n_ep] == st) & ok
                if sel.sum() < 4:  # < 2 min of clean stage data
                    continue
                psd = p[:, sel].mean(1)
                for ch, spec in [("mean", psd.mean(0)), *zip(cd.SITES, psd)]:
                    rows.append({**meta, "staging": staging, "stage": st, "condition": cond, "channel": ch,
                                 "n_epochs": int(sel.sum()), "psd": spec.astype(np.float32), **features(spec)})
    return rows


def clean_all(eeg, ecg, peaks, sf):
    return {"raw": eeg, "ica": cd.ica_clean(eeg, ecg, sf), "cfar": cd.project_out(eeg, U_POP),
            "own": cd.project_out(eeg, cd.cfa_pattern(eeg, peaks, sf))}


def is_cached(pid):
    out = os.path.join(CACHE, f"{pid}.parquet")
    if not os.path.exists(out) or not os.path.exists(os.path.join(EPOCH_CACHE, f"{pid}.parquet")):
        return False
    d = pd.read_parquet(out, columns=["condition", "staging"])
    return set(d.condition) >= set(CONDITIONS)


def process(job):
    pid, path = job
    out = os.path.join(CACHE, f"{pid}.parquet")
    if is_cached(pid):
        return pid, "cached"
    try:
        night = load_night(path)
        if isinstance(night, str):
            return pid, night
        eeg, ecg, sf, stages = night
        peaks = detect_r_peaks(ecg, sf)
        if len(peaks) < 100:
            return pid, "few_peaks"
        sigs = clean_all(eeg, ecg, peaks, sf)
        expert = expert_stages(path)
        stagings = {"yasa": stages, **({"expert": expert} if expert is not None else {})}
        rows = stage_rows(sigs, stagings, eeg, sf, patient_id=pid)
        if expert is not None:  # staging accuracy per cleaning method (YASA re-run on each cleaned C4)
            c4 = cd.SITES.index("C4")
            ys = {f"yasa_{c}": yasa_stages(x[c4], "C4", sf) for c, x in sigs.items()}
            n = min(len(expert), *map(len, ys.values()))
            pd.DataFrame({"patient_id": pid, "epoch": np.arange(n), "expert": expert[:n], "yasa_orig": stages[:n],
                          **{k: v[:n] for k, v in ys.items()}}).to_parquet(os.path.join(EPOCH_CACHE, f"{pid}.parquet"))
        else:
            pd.DataFrame(columns=["patient_id"]).to_parquet(os.path.join(EPOCH_CACHE, f"{pid}.parquet"))
        pd.DataFrame(rows).to_parquet(out)
        return pid, f"ok {len(rows)}{'' if expert is not None else ' no_expert'}"
    except Exception as exc:  # noqa: BLE001 -- unreadable/odd EDFs skipped, reason logged
        return pid, f"error {type(exc).__name__}: {exc} | {traceback.format_exc(limit=1).splitlines()[-1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="random sample of patients (sample run)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(EPOCH_CACHE, exist_ok=True)
    cfa = pd.read_parquet(os.path.join(cd.HERE, "cfa_combined.parquet"), columns=["patient_id", "edf_path"])
    cfa = cfa.drop_duplicates("patient_id")
    cfa = cfa[cfa.patient_id.str.startswith("I0")].sample(frac=1, random_state=cd.SEED)
    if args.limit:
        cfa = cfa.head(args.limit)
    print(f"{len(cfa)} patients, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (pid, status) in enumerate(ex.map(process, zip(cfa.patient_id, cfa.edf_path)), 1):
            print(f"[{i}/{len(cfa)}] {pid} {status}", flush=True)


if __name__ == "__main__":
    main()
