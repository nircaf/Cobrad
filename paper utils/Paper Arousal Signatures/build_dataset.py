"""Extract per-arousal-event cardiocortical features from BDSP/Harvard BIDS PSG recordings.

For every scored microarousal ("MA" rows in *_task-psg_events_annotations.csv,
tagged by the tech with a bracketed cause: [Spon], [Apnea], [Hypopnea], [RERA],
[Leg]) this pulls the aligned per-sample h5 signals (ecg-derived instantaneous
HR + 6 EEG derivations) and computes a baseline-vs-post feature vector:
delta HR, HR-peak latency, EEG delta/alpha/beta band-power change, spectral
entropy change, and cross-channel spatial spread of the band-power change.

Central vs obstructive apnea is disambiguated by looking back at the nearest
preceding Type=="Apnea" row's free-text Description (e.g. "Central Apnea...").
Generic "Apnea" with no such qualifier nearby is dropped rather than guessed.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT = ROOT / "EDF_Format" / "Harvard_Electroencephalography" / "bids"

EEG_CHANNELS = ["f3-m2", "f4-m1", "c3-m2", "c4-m1", "o1-m2", "o2-m1"]
BANDS = {"delta": (0.5, 4.0), "alpha": (8.0, 12.0), "beta": (13.0, 30.0)}

BASELINE = (-30.0, -10.0)  # seconds relative to arousal onset
POST = (0.0, 15.0)

CAUSE_MAP = {
    "spon": "spontaneous",
    "rera": "RERA",
    "lm": "PLM",  # limb movement
    # "apnea" resolved to obstructive/central below; "hypopnea" kept as-is
}


def _rt_to_sec(t: str) -> int:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _resolve_apnea_cause(events: pd.DataFrame, onset_sec: float, lookback_s: float = 30.0) -> str | None:
    prior = events[(events["Type"] == "Apnea") & (events["sec"] <= onset_sec) & (events["sec"] >= onset_sec - lookback_s)]
    if prior.empty:
        return None
    desc = str(prior.iloc[-1]["Description"]).lower()
    if "central" in desc:
        return "central apnea"
    if "obstructive" in desc:
        return "obstructive apnea"
    if "mixed" in desc:
        return None  # excluded: not one of the five requested classes
    return None


def _spectral_entropy(psd: np.ndarray) -> float:
    p = psd / (psd.sum() + 1e-24)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / np.log2(len(p)))


def _band_features(sig: np.ndarray, fs: float) -> tuple[dict[str, float], float]:
    freqs, psd = welch(sig, fs=fs, nperseg=min(len(sig), int(4 * fs)))
    bp = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        bp[name] = float(np.trapz(psd[mask], freqs[mask])) if mask.any() else np.nan
    return bp, _spectral_entropy(psd)


def _extract_event_features(h5f: h5py.File, fs: float, onset_idx: int) -> dict | None:
    n = h5f["signals/hr"].shape[0]
    b0, b1 = [onset_idx + int(t * fs) for t in BASELINE]
    p0, p1 = [onset_idx + int(t * fs) for t in POST]
    if b0 < 0 or p1 >= n:
        return None

    hr_base = h5f["signals/hr"][b0:b1, 0]
    hr_post = h5f["signals/hr"][p0:p1, 0]
    if len(hr_base) == 0 or len(hr_post) == 0 or np.all(np.isnan(hr_base)) or np.all(np.isnan(hr_post)):
        return None
    delta_hr = float(np.nanmean(hr_post) - np.nanmean(hr_base))
    peak_idx = int(np.nanargmax(hr_post)) if not np.all(np.isnan(hr_post)) else 0
    hr_latency = peak_idx / fs

    band_deltas = {f"eeg_{b}": [] for b in BANDS}
    entropy_deltas = []
    for ch in EEG_CHANNELS:
        base_bp, base_ent = _band_features(h5f[f"signals/{ch}"][b0:b1, 0], fs)
        post_bp, post_ent = _band_features(h5f[f"signals/{ch}"][p0:p1, 0], fs)
        for b in BANDS:
            ratio = np.log2((post_bp[b] + 1e-12) / (base_bp[b] + 1e-12))
            band_deltas[f"eeg_{b}"].append(ratio)
        entropy_deltas.append(post_ent - base_ent)

    feat = {f"{k}_mean": float(np.nanmean(v)) for k, v in band_deltas.items()}
    feat.update({f"{k}_spread": float(np.nanstd(v)) for k, v in band_deltas.items()})
    feat["eeg_entropy_delta"] = float(np.nanmean(entropy_deltas))
    feat["delta_hr"] = delta_hr
    feat["hr_latency_s"] = hr_latency
    return feat


def extract_subject(eeg_dir: Path) -> pd.DataFrame:
    ev_paths = list(eeg_dir.glob("*_task-psg_events_annotations.csv"))
    sl_paths = list(eeg_dir.glob("*_task-psg_sleep_annotations.csv"))
    h5_paths = list(eeg_dir.glob("*.h5"))
    if not (ev_paths and sl_paths and h5_paths):
        return pd.DataFrame()

    events = pd.read_csv(ev_paths[0])
    events = events[events["Record Time"].notna()].copy()
    events["sec"] = events["Record Time"].apply(_rt_to_sec)
    n_epochs = pd.read_csv(sl_paths[0])["Epoch"].max()

    rows = []
    with h5py.File(h5_paths[0], "r") as h5f:
        n_samples = h5f["signals/hr"].shape[0]
        fs = round(n_samples / (n_epochs * 30))

        ma = events[events["Type"] == "MA"]
        for i, r in ma.iterrows():
            desc = str(r["Description"])
            if "[" not in desc:
                continue
            tag = desc.split("[")[-1].rstrip("]").strip().lower()
            if tag == "apnea":
                cause = _resolve_apnea_cause(events, r["sec"])
                if cause is None:
                    continue
            elif tag == "hypopnea":
                cause = "hypopnea"
            elif tag in CAUSE_MAP:
                cause = CAUSE_MAP[tag]
            else:
                continue

            onset_idx = int(r["sec"] * fs)
            feat = _extract_event_features(h5f, fs, onset_idx)
            if feat is None:
                continue
            feat["cause"] = cause
            feat["subject"] = eeg_dir.parents[1].name
            feat["session"] = eeg_dir.parents[0].name
            feat["event_row"] = i
            rows.append(feat)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=25, help="number of subject-sessions to scan (pilot run)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "events_features.csv")
    args = ap.parse_args()

    eeg_dirs = sorted(Path(p).parent for p in glob.glob(str(BIDS_ROOT / "*" / "sub-*" / "ses-*" / "eeg" / "*.h5")))
    frames = []
    for d in eeg_dirs[: args.n_subjects]:
        try:
            df = extract_subject(d)
        except Exception as e:  # pilot run: skip malformed subjects, keep going
            print(f"skip {d}: {e}")
            continue
        if not df.empty:
            frames.append(df)
        print(f"{d.parents[1].name}/{d.parents[0].name}: {len(df)} events")

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} events -> {args.out}")
    if len(out):
        print(out["cause"].value_counts())


if __name__ == "__main__":
    main()
