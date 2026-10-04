#!/usr/bin/env python3
"""Full-cohort real-data 5-minute brain-heart coupling analysis.

Every eligible subject contributes the earliest qualifying PSG to the primary
analysis; subjects with repeats contribute their second qualifying PSG to the
long-term analysis. Per-recording caches make the multi-terabyte run resumable.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import butter, coherence, detrend, hilbert, sosfiltfilt, find_peaks, resample_poly
from scipy.stats import pearsonr, spearmanr
import statsmodels.formula.api as smf


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "EDF_Format" / "Harvard_Electroencephalography" / "bids"
CACHE = HERE / "real_psg_windows.parquet"
STATS = HERE / "real_psg_stats.json"
MANIFEST = HERE / "real_psg_manifest.csv"
RECORD_CACHE = HERE / "real_psg_record_cache_50hz_fullread"
EEG_CHANNELS = ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1"]
SEED = 20260820
WINDOW_SECONDS = 300
MAX_WORKERS = 12


def annotation_path(edf: Path) -> Path | None:
    hits = list(edf.parent.glob("*_sleep_annotations.csv"))
    return hits[0] if hits else None


def inspect_metadata(path: Path):
    """Screen BIDS sidecars without opening the much larger EDF."""
    try:
        channel_files = list(path.parent.glob("*_channels.tsv"))
        ann = annotation_path(path)
        if not channel_files or ann is None:
            return None
        channels = pd.read_csv(channel_files[0], sep="\t")
        labels = channels["name"].astype(str).tolist()
        rates = dict(zip(channels["name"].astype(str), pd.to_numeric(channels["sampling_frequency"], errors="coerce")))
        ecg = next((x for x in ["EKG", "ECG", "ECG1-ECG2"] if x in labels), None)
        if not ecg or not all(x in labels for x in EEG_CHANNELS):
            return None
        if min(rates[x] for x in EEG_CHANNELS + [ecg]) < 100:
            return None
        # Header plus at least 480 30-second epochs = four hours.
        with ann.open("rb") as handle:
            epoch_rows = sum(1 for _ in handle) - 1
        if epoch_rows < 480:
            return None
        subject = next(x for x in path.parts if x.lower().startswith("sub-")).lower()
        session = next(x for x in path.parts if x.lower().startswith("ses-")).lower()
        return {"subject": subject, "session": session, "path": str(path), "annotation": str(ann),
                "annotation_epochs": epoch_rows, "ecg": ecg}
    except Exception:
        return None


def build_manifest() -> pd.DataFrame:
    if MANIFEST.exists():
        return pd.read_csv(MANIFEST)
    candidates = sorted(ROOT.glob("*/sub-*/ses-*/eeg/*_eeg.edf"))
    rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, result in enumerate(ex.map(inspect_metadata, candidates, chunksize=16), 1):
            if result:
                rows.append(result)
            if i % 500 == 0:
                print(f"screened {i}/{len(candidates)} EDF sidecars", flush=True)
    frame = pd.DataFrame(rows)
    # Session numbers are chronological in this BIDS archive. Retain at most two
    # qualifying sessions per subject: first for primary, second for repeat.
    frame["session_number"] = pd.to_numeric(frame.session.str.extract(r"(\d+)")[0], errors="coerce")
    frame = frame.sort_values(["subject", "session_number", "session"])
    frame = frame.groupby("subject", group_keys=False).head(2)
    frame.to_csv(MANIFEST, index=False)
    return frame


def stage_label(value) -> str:
    s = str(value).strip().upper().replace("STAGE", "").replace(" ", "")
    mapping = {"1": "N1", "N1": "N1", "2": "N2", "N2": "N2", "3": "N3", "4": "N3",
               "N3": "N3", "N4": "N3", "R": "REM", "REM": "REM", "W": "Wake", "WAKE": "Wake"}
    return mapping.get(s, "Other")


def safe_filter(x, fs, lo, hi):
    sos = butter(4, [lo, min(hi, fs / 2 - .5)], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def detect_rpeaks(ecg, fs):
    x = safe_filter(np.asarray(ecg, float), fs, 5, 25)
    x = detrend(x)
    scale = np.median(np.abs(x - np.median(x))) * 1.4826
    if not np.isfinite(scale) or scale <= 0:
        return np.array([], dtype=int)
    z = x / scale
    pos, _ = find_peaks(z, distance=int(.30 * fs), prominence=2.5)
    neg, _ = find_peaks(-z, distance=int(.30 * fs), prominence=2.5)
    peaks = pos if (np.median(z[pos]) if len(pos) else 0) >= (np.median(-z[neg]) if len(neg) else 0) else neg
    if len(peaks) < 3:
        return peaks
    rr = np.diff(peaks) / fs
    keep = np.r_[True, (rr > .33) & (rr < 1.72)]
    return peaks[keep]


def coupling_for_channel(eeg, fs, peaks):
    eeg = np.asarray(eeg, float)
    if not np.all(np.isfinite(eeg)) or np.ptp(eeg) == 0:
        return np.nan
    robust_sd = np.median(np.abs(eeg - np.median(eeg))) * 1.4826
    if robust_sd <= 0 or np.max(np.abs(eeg - np.median(eeg))) > 30 * robust_sd:
        return np.nan
    delta = safe_filter(eeg, fs, .5, 4)
    envelope = np.log(np.abs(hilbert(delta)) + np.finfo(float).eps)
    grid = np.arange(0, len(eeg) / fs, .25)
    env4 = np.interp(grid, np.arange(len(eeg)) / fs, envelope)
    beat_t = peaks / fs
    if len(beat_t) < 120:
        return np.nan
    rr = np.diff(beat_t)
    hr_t = (beat_t[1:] + beat_t[:-1]) / 2
    hr = 60 / rr
    good = (hr >= 35) & (hr <= 180)
    if good.sum() < 100 or (hr_t[good].max() - hr_t[good].min()) < 240:
        return np.nan
    hr4 = np.interp(grid, hr_t[good], hr[good])
    f, cxy = coherence(detrend(hr4), detrend(env4), fs=4, nperseg=256, noverlap=128)
    band = (f >= .04) & (f <= .40)
    return float(np.nanmean(cxy[band])) if band.any() else np.nan


def process_record(row: dict) -> pd.DataFrame:
    path = Path(row["path"])
    ann = pd.read_csv(row["annotation"])
    stages = [stage_label(x) for x in ann["Stage"]]
    out = []
    with pyedflib.EdfReader(str(path)) as f:
        start_date = f.getStartdatetime().isoformat()
        labels = f.getSignalLabels()
        idx = {x: labels.index(x) for x in EEG_CHANNELS + [row["ecg"]]}
        fs = float(f.getSampleFrequency(idx[row["ecg"]]))
        if any(abs(float(f.getSampleFrequency(idx[x])) - fs) > .01 for x in EEG_CHANNELS):
            return pd.DataFrame()
        nwin = min(int(f.getFileDuration() // WINDOW_SECONDS), len(stages) // 10)
        nsamp = int(WINDOW_SECONDS * fs)
        down = max(1, int(round(fs / 50.0)))
        fs_50 = fs / down
        nsamp_50 = int(round(WINDOW_SECONDS * fs_50))
        total_samples = min(int(nwin * nsamp), f.getNSamples()[idx[row["ecg"]]])
        eeg_50_by_channel = {}
        for channel in EEG_CHANNELS:
            channel_samples = min(total_samples, f.getNSamples()[idx[channel]])
            eeg_native = f.readSignal(idx[channel], start=0, n=channel_samples, digital=False)
            eeg_50_by_channel[channel] = resample_poly(eeg_native, up=1, down=down)
        for w in range(nwin):
            start = w * nsamp
            ecg = f.readSignal(idx[row["ecg"]], start=start, n=nsamp, digital=False)
            peaks = detect_rpeaks(ecg, fs)
            if not (150 <= len(peaks) <= 900):
                continue
            window_stages = stages[w * 10:(w + 1) * 10]
            counts = Counter(x for x in window_stages if x != "Other")
            stage = counts.most_common(1)[0][0] if counts else "Other"
            purity = counts.get(stage, 0) / 10
            values = {}
            for channel in EEG_CHANNELS:
                start_50 = w * nsamp_50
                eeg_50 = eeg_50_by_channel[channel][start_50:start_50 + nsamp_50]
                if len(eeg_50) < nsamp_50:
                    values[channel] = np.nan
                    continue
                peaks_50 = np.rint(peaks * fs_50 / fs).astype(int)
                values[channel] = coupling_for_channel(eeg_50, fs_50, peaks_50)
            valid = [v for v in values.values() if np.isfinite(v)]
            if len(valid) < 4:
                continue
            result = {"subject": row["subject"], "session": row["session"], "start_date": start_date,
                      "window": w, "hour": (w + .5) / 12, "stage": stage, "stage_purity": purity,
                      "n_beats": len(peaks), "coupling": float(np.median(valid))}
            result.update({f"coupling_{k}": v for k, v in values.items()})
            out.append(result)
    return pd.DataFrame(out)


def variance_model(data, adjusted=False):
    d = data.copy()
    if adjusted:
        fit = smf.ols("coupling ~ C(stage) + hour + I(hour ** 2) + stage_purity + n_beats", d).fit()
        d["outcome"] = fit.resid + d.coupling.mean()
    else:
        d["outcome"] = d.coupling
    grouped = d.groupby("subject")["outcome"]
    sizes = grouped.size().astype(float)
    means = grouped.mean()
    n, k = len(d), len(sizes)
    grand = float(d.outcome.mean())
    ss_between = float(np.sum(sizes * (means - grand) ** 2))
    ss_within = float(grouped.apply(lambda x: np.sum((x - x.mean()) ** 2)).sum())
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    n0 = (n - float(np.sum(sizes ** 2)) / n) / (k - 1)
    between = max((ms_between - ms_within) / n0, 0.0)
    within = ms_within
    return {"between": between, "within": within, "icc": between / (between + within),
            "n_windows": len(d), "n_subjects": d.subject.nunique()}


def bootstrap_icc(data, adjusted=False, nboot=500):
    rng = np.random.default_rng(SEED + int(adjusted))
    subjects = data.subject.unique(); vals = []
    for b in range(nboot):
        picked = rng.choice(subjects, len(subjects), replace=True)
        chunks = []
        for j, s in enumerate(picked):
            x = data[data.subject == s].copy(); x["subject"] = f"b{j}"; chunks.append(x)
        vals.append(variance_model(pd.concat(chunks, ignore_index=True), adjusted)["icc"])
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])]


def absolute_icc(a, b):
    x = np.c_[a, b]; n, k = x.shape
    gm = x.mean(); row = x.mean(1); col = x.mean(0)
    msr = k * np.sum((row - gm) ** 2) / (n - 1)
    msc = n * np.sum((col - gm) ** 2) / (k - 1)
    mse = np.sum((x - row[:, None] - col[None, :] + gm) ** 2) / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def summarize(data):
    primary = data[(data.stage != "Other") & (data.stage_purity >= .8)].copy()
    unadjusted = variance_model(primary, False); adjusted = variance_model(primary, True)
    unadjusted["ci"] = bootstrap_icc(primary, False); adjusted["ci"] = bootstrap_icc(primary, True)
    stage_results = {}
    for stage, d in primary.groupby("stage"):
        if d.subject.nunique() >= 20:
            r = variance_model(d, False); r["ci"] = bootstrap_icc(d, False, 150); stage_results[stage] = r
    visits = primary.groupby(["subject", "session"], as_index=False).agg(coupling=("coupling", "mean"), start_date=("start_date", "first"))
    visits = visits.sort_values(["subject", "start_date"]).groupby("subject").head(2)
    wide = visits.assign(visit=visits.groupby("subject").cumcount()).pivot(index="subject", columns="visit", values="coupling").dropna()
    dates = visits.assign(visit=visits.groupby("subject").cumcount()).pivot(index="subject", columns="visit", values="start_date").loc[wide.index]
    intervals = (pd.to_datetime(dates[1]) - pd.to_datetime(dates[0])).dt.days.abs() / 365.25
    repeat = {"n": len(wide), "icc_a1": absolute_icc(wide[0].values, wide[1].values),
              "pearson_r": float(pearsonr(wide[0], wide[1]).statistic),
              "spearman_r": float(spearmanr(wide[0], wide[1]).statistic),
              "median_interval_years": float(intervals.median()), "max_interval_years": float(intervals.max())}
    # Channel x stage fingerprints and blinded identification.
    channel_cols = [f"coupling_{x}" for x in EEG_CHANNELS]
    fp = primary.groupby(["subject", "session", "stage"])[channel_cols].mean().unstack("stage")
    fp = fp.sort_index(axis=1)
    first, second, subjects = [], [], []
    for s, g in fp.groupby(level=0):
        if len(g) >= 2:
            a, b = g.iloc[0].values.astype(float), g.iloc[1].values.astype(float)
            good = np.isfinite(a) & np.isfinite(b)
            if good.sum() >= 12:
                first.append(a); second.append(b); subjects.append(s)
    # Pairwise correlations use pairwise-complete features.
    sim = np.full((len(subjects), len(subjects)), np.nan)
    for i, a in enumerate(second):
        for j, b in enumerate(first):
            good = np.isfinite(a) & np.isfinite(b)
            if good.sum() >= 12: sim[i, j] = np.corrcoef(a[good], b[good])[0, 1]
    identify = float(np.mean(np.nanargmax(sim, axis=1) == np.arange(len(subjects)))) if len(subjects) else np.nan
    repeat.update({"fingerprint_n": len(subjects), "identification_accuracy": identify,
                   "within_similarity_median": float(np.nanmedian(np.diag(sim))),
                   "between_similarity_median": float(np.nanmedian(sim[~np.eye(len(sim), dtype=bool)]))})
    return {"cohort": {"subjects": int(primary.subject.nunique()), "sessions": int(primary.groupby(["subject", "session"]).ngroups),
                        "windows": len(primary), "median_windows_per_session": float(primary.groupby(["subject", "session"]).size().median())},
            "unadjusted": unadjusted, "adjusted": adjusted, "stages": stage_results, "repeat": repeat}


def main():
    manifest = build_manifest()
    if CACHE.exists():
        data = pd.read_parquet(CACHE)
    else:
        RECORD_CACHE.mkdir(exist_ok=True)
        chunks = []
        rows = manifest.to_dict("records")
        missing = []
        for row in rows:
            cache_path = RECORD_CACHE / f"{row['subject']}_{row['session']}.parquet"
            failure_path = RECORD_CACHE / f"{row['subject']}_{row['session']}.failed.txt"
            if failure_path.exists():
                continue
            if cache_path.exists():
                try:
                    d = pd.read_parquet(cache_path)
                    if not d.empty:
                        chunks.append(d)
                    continue
                except Exception:
                    pass
            missing.append(row)
        print(f"manifest recordings={len(rows)} cached={len(rows)-len(missing)} missing={len(missing)}", flush=True)
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(process_record, row): row for row in missing}
            for i, future in enumerate(as_completed(futures), 1):
                row = futures[future]
                cache_path = RECORD_CACHE / f"{row['subject']}_{row['session']}.parquet"
                failure_path = RECORD_CACHE / f"{row['subject']}_{row['session']}.failed.txt"
                try:
                    d = future.result()
                except Exception as exc:
                    failure_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
                    print(f"failed {row['subject']} {row['session']}: {exc}", flush=True)
                    continue
                d.to_parquet(cache_path, index=False)
                if not d.empty:
                    chunks.append(d)
                print(f"processed {i}/{len(missing)}", flush=True)
        data = pd.concat(chunks, ignore_index=True)
        data.to_parquet(CACHE, index=False)
    stats = summarize(data)
    STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
