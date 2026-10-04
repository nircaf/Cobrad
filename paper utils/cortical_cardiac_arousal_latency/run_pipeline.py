#!/usr/bin/env python3
"""Cortical-cardiac arousal latency pilot: extract EEG-scored arousal events,
detect matching HR-acceleration onset from the EKG channel, compute
Delta t = t_HR - t_EEG per event.

  source venv/bin/activate && python3 "paper utils/cortical_cardiac_arousal_latency/run_pipeline.py"

Data-schema note (found during this pilot, not assumed beforehand):
Only Harvard study I0002 exports the structured
`Epoch,Stage,Type,Time,Length,Description` events_annotations.csv with the
Apnea/Hypopnea/MA taxonomy and bracketed MA linkage tags
(e.g. "Microarousal [Hypopnea]") that this method depends on. I0003, I0004,
I0006 use different, incompatible per-study event-log schemas (free-text
annotation streams / clinical-system exports) and I0003 has no directly
readable EDF (only .h5). This pilot therefore draws its subjects from I0002
only; see the paper's Limitations section.
"""
from __future__ import annotations

import multiprocessing
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

HERE = Path(__file__).resolve().parent
BIDS_ROOT = HERE.parent.parent / "EDF_Format" / "Harvard_Electroencephalography" / "bids"
STUDY = "I0002"

N_SAMPLE_CANDIDATES = 900  # sample size across I0002; scaled up from the 60-session pilot
MAX_EVENTS_PER_SESSION = 60  # cap runtime on sessions with unusually many events
RANDOM_SEED = 42
# Pickle-cache note: pickles_sleep_stage/<study>/<stage>/*.pkl (used by the HEP
# pipeline) was investigated as a shortcut and rejected -- inspection showed
# each file is a stage-purified, concatenated (capped at ~1200 s), resampled
# (256 Hz) synthetic mne.RawArray built for heartbeat-evoked-potential
# averaging (see 6_hep_group_comparison.py::process_file_data), with empty
# .annotations and no mapping from its samples back to absolute Record Time
# in the source EDF. R-peaks in it can't be aligned to t_EEG from
# events_annotations.csv, and windows spanning stage transitions are simply
# missing. So this pipeline still reads the raw EDF directly, just in
# parallel across sessions (ProcessPoolExecutor, same pattern as this repo's
# parallel_patient_processing.py) to cover far more subjects.
DEFAULT_WORKERS = max(1, multiprocessing.cpu_count() // 3)

# ---- HR-onset detection parameters (documented here + in the paper Methods) ----
BASELINE_START = -30.0   # s relative to t_EEG: baseline window start
BASELINE_END = -5.0      # s relative to t_EEG: baseline window end
SEARCH_START = -5.0      # s relative to t_EEG: HR-onset search window start
SEARCH_END = 15.0        # s relative to t_EEG: HR-onset search window end
THRESHOLD_K = 1.0        # threshold = baseline_mean + K * baseline_SD
SUSTAIN_BEATS = 3        # consecutive beats at/above threshold to count as onset
MIN_BASELINE_BEATS = 5   # minimum R-peaks in baseline window to trust it
PLAUSIBLE_IBI = (0.3, 2.0)  # seconds; beats outside this are dropped as artifacts
EXTRACT_PAD = 15.0       # extra seconds padded on both ends of extraction window
                          # to avoid R-peak edge effects


def bandpass_ecg(sig, sfreq, lowcut=0.5, highcut=40, order=4):
    nyq = 0.5 * sfreq
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, sig)


def to_seconds(hms: str) -> float:
    h, m, s = hms.split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)


MA_TAG_RE = re.compile(r"\[([^\]]+)\]")


def classify_ma(description: str, prior_rows: pd.DataFrame) -> str:
    """Classify a Microarousal (MA) row's linked event type from its bracket
    tag(s), resolving Apnea into obstructive/central via the nearest
    preceding Apnea row's free-text Description."""
    tags = set(MA_TAG_RE.findall(description))
    if "PLMS" in tags:
        return "PLM"
    if "RERA" in tags:
        return "RERA"
    if "Apnea" in tags:
        if len(prior_rows):
            desc = str(prior_rows.iloc[-1]["Description"])
            if "Central" in desc:
                return "central_apnea"
            if "Obstructive" in desc:
                return "obstructive_apnea"
        return "apnea_unspecified"
    if "Hypopnea" in tags:
        return "hypopnea"
    if "LM" in tags:
        return "limb_movement"
    if "Spon" in tags:
        return "spontaneous"
    return "other"


def parse_events(events_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(events_csv)
    df["t_sec"] = df["Record Time"].map(to_seconds)
    ma = df[df["Type"] == "MA"].copy()
    apnea = df[df["Type"] == "Apnea"]

    rows = []
    for i, r in ma.iterrows():
        prior = apnea[(apnea["t_sec"] < r["t_sec"]) & (apnea["t_sec"] > r["t_sec"] - 30)]
        event_type = classify_ma(str(r["Description"]), prior)
        rows.append(dict(
            t_eeg=r["t_sec"], stage=r["Stage"], event_type=event_type,
            description=r["Description"],
        ))
    out = pd.DataFrame(rows)
    if len(out):
        out = out[out["stage"].isin(["N1", "N2", "N3", "R"])].reset_index(drop=True)
    return out


def r_peak_times(seg, sfreq):
    """Bandpass -> neurokit2 R-peak detection, with a scipy fallback."""
    try:
        seg_clean = bandpass_ecg(seg, sfreq)
        _, info = nk.ecg_peaks(seg_clean, sampling_rate=sfreq)
        peaks = np.asarray(info["ECG_R_Peaks"], dtype=float)
    except Exception:
        from scipy.signal import find_peaks
        seg_clean = bandpass_ecg(seg, sfreq)
        peaks, _ = find_peaks(seg_clean, distance=int(sfreq * 0.3))
        peaks = peaks.astype(float)
    return peaks / sfreq  # seconds relative to segment start


def detect_hr_onset(ecg_full, sfreq, t_eeg, edf_dur):
    """Returns (t_hr, flag). t_hr is None if unusable/no crossing."""
    win_start = t_eeg + BASELINE_START - EXTRACT_PAD
    win_end = t_eeg + SEARCH_END + EXTRACT_PAD
    if win_start < 0 or win_end > edf_dur:
        return None, "window_out_of_bounds"

    i0, i1 = int(win_start * sfreq), int(win_end * sfreq)
    seg = ecg_full[i0:i1]
    if not np.all(np.isfinite(seg)):
        seg = np.nan_to_num(seg, nan=np.nanmedian(seg))
    r_t = r_peak_times(seg, sfreq) + win_start  # absolute seconds (recording-relative)

    if len(r_t) < 2:
        return None, "too_few_beats"
    ibi = np.diff(r_t)
    beat_t = r_t[1:]  # HR sample assigned at the second peak of each interval
    beat_hr = 60.0 / ibi
    valid = (ibi >= PLAUSIBLE_IBI[0]) & (ibi <= PLAUSIBLE_IBI[1])
    beat_t, beat_hr = beat_t[valid], beat_hr[valid]
    if len(beat_t) < 2:
        return None, "too_few_valid_beats"

    base_mask = (beat_t >= t_eeg + BASELINE_START) & (beat_t <= t_eeg + BASELINE_END)
    if base_mask.sum() < MIN_BASELINE_BEATS:
        return None, "insufficient_baseline"
    base_hr = beat_hr[base_mask]
    threshold = base_hr.mean() + THRESHOLD_K * base_hr.std()

    search_mask = (beat_t >= t_eeg + SEARCH_START) & (beat_t <= t_eeg + SEARCH_END)
    s_t, s_hr = beat_t[search_mask], beat_hr[search_mask]
    if len(s_t) < SUSTAIN_BEATS:
        return None, "too_few_search_beats"

    above = s_hr >= threshold
    for i in range(len(s_t) - SUSTAIN_BEATS + 1):
        if above[i:i + SUSTAIN_BEATS].all():
            return float(s_t[i]), "ok"
    return None, "no_sustained_crossing"


def caisr_arousal_agreement(caisr_csv: Path, t_eeg: float, sfreq: float, tol=3.0) -> bool | None:
    """CAISR annotation rows are sampled at the same rate as the EDF (verified:
    row count / EDF duration == EDF sfreq for this dataset), not ~1 Hz as the
    file's row-per-second look might suggest -- so row index must be scaled by
    sfreq, not treated as whole seconds."""
    try:
        col = pd.read_csv(caisr_csv, usecols=["arousal_caisr"])["arousal_caisr"]
    except Exception:
        return None
    lo, hi = max(0, int((t_eeg - tol) * sfreq)), int((t_eeg + tol) * sfreq)
    if hi >= len(col):
        return None
    return bool((col.iloc[lo:hi + 1] == 1).any())


def find_candidate_sessions():
    root = BIDS_ROOT / STUDY
    dirs = sorted(root.glob("sub-*/ses-*/eeg"))
    cand = [d for d in dirs
            if list(d.glob("*_task-PSG_eeg.edf"))
            and list(d.glob("*events_annotations.csv"))
            and list(d.glob("*caisr_annotations.csv"))]
    rng = random.Random(RANDOM_SEED)
    return rng.sample(cand, min(N_SAMPLE_CANDIDATES, len(cand)))


def process_session(d: Path):
    """Worker: parse events + extract EKG + compute Delta t for one session.
    Returns (sub_id, ses_id, rows, n_candidate, n_ok, error_str_or_None).
    Must be top-level (not a closure) so ProcessPoolExecutor can pickle it."""
    sub_id, ses_id = d.parts[-3], d.parts[-2]
    edf = list(d.glob("*_task-PSG_eeg.edf"))[0]
    ev_csv = list(d.glob("*events_annotations.csv"))[0]
    caisr_csv = list(d.glob("*caisr_annotations.csv"))[0]

    try:
        events = parse_events(ev_csv)
    except Exception as e:
        return sub_id, ses_id, [], 0, 0, f"events parse failed: {e}"
    if len(events) == 0:
        return sub_id, ses_id, [], 0, 0, None

    try:
        raw = mne.io.read_raw_edf(edf, preload=False, verbose=False)
    except Exception as e:
        return sub_id, ses_id, [], len(events), 0, f"EDF read failed: {e}"
    ch_lower = [c.lower() for c in raw.ch_names]
    ekg_idx = [i for i, c in enumerate(ch_lower) if "ekg" in c or "ecg" in c]
    if not ekg_idx:
        return sub_id, ses_id, [], len(events), 0, "no EKG channel"
    sfreq = raw.info["sfreq"]
    dur = raw.n_times / sfreq
    try:
        raw.pick([raw.ch_names[ekg_idx[0]]])  # only load the EKG channel (speed)
        raw.load_data(verbose=False)
    except Exception as e:
        return sub_id, ses_id, [], len(events), 0, f"EDF load failed: {e}"
    ecg_full = raw.get_data()[0]

    rows, n_ok = [], 0
    for _, ev in events.iloc[:MAX_EVENTS_PER_SESSION].iterrows():
        t_hr, flag = detect_hr_onset(ecg_full, sfreq, ev["t_eeg"], dur)
        agree = caisr_arousal_agreement(caisr_csv, ev["t_eeg"], sfreq) if flag == "ok" else None
        rows.append(dict(subject=sub_id, session=ses_id, t_eeg=ev["t_eeg"], stage=ev["stage"],
                          event_type=ev["event_type"], t_hr=t_hr,
                          dt=(t_hr - ev["t_eeg"]) if t_hr is not None else np.nan,
                          flag=flag, caisr_arousal_agree=agree))
        n_ok += flag == "ok"
    return sub_id, ses_id, rows, len(events), n_ok, None


def main():
    sessions = find_candidate_sessions()
    workers = min(DEFAULT_WORKERS, len(sessions))
    print(f"Processing {len(sessions)} candidate sessions with {workers} workers", flush=True)

    all_events, used_subjects, n_done = [], set(), 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_session, d): d for d in sessions}
        for fut in as_completed(futures):
            n_done += 1
            try:
                sub_id, ses_id, rows, n_cand, n_ok, err = fut.result()
            except Exception as e:
                print(f"[worker crashed] {futures[fut]}: {e}", flush=True)
                continue
            if err:
                print(f"[{sub_id}/{ses_id}] {err}", flush=True)
                continue
            all_events.extend(rows)
            if n_ok > 0:
                used_subjects.add(sub_id)
            if n_cand:
                print(f"[{sub_id}/{ses_id}] {n_cand} candidate events, {n_ok} usable Delta t "
                      f"({n_done}/{len(sessions)} sessions done)", flush=True)

    out = pd.DataFrame(all_events)
    out_path = HERE / "events_dt.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} candidate events ({out['flag'].eq('ok').sum() if len(out) else 0} usable) "
          f"from {len(used_subjects)} subjects -> {out_path}")


if __name__ == "__main__":
    main()
