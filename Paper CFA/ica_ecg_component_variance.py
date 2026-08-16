#!/usr/bin/env python3

"""Parallel ECG-informed ICA variance analysis using quality-controlled EDF windows.

ICA is fit on the continuous window as usual, but each component's reported
variance is measured on its R-peak-locked HEP evoked average (epoch window
matching this repo's own HEP analysis, see HEP_TMIN/HEP_TMAX below) -- i.e.
how much of the heartbeat-evoked EEG response this component explains, not
its share of generic continuous-signal variance."""
from __future__ import annotations

import argparse, hashlib, os, re, sys, tempfile, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
ECG_RE = re.compile(r"ecg|ekg", re.I)
# Whitelist against the actual 10-20/10-10 site names, not a blacklist of
# known-junk keywords: this corpus mixes scalp EEG with iEEG depth-electrode
# recordings (e.g. 'SFA1', 'MFA3'), DC/aux inputs ('X1'-'X7', 'DC01'-'DC08'),
# non-10-20 extra electrodes ('T1', 'T2', 'PG1', 'elA22'), and non-English
# aux-channel names ('TERMISTORE', 'TORACE') that a keyword blacklist misses.
_MONTAGE_1020_SITES = {name.upper() for name in mne.channels.make_standard_montage("standard_1020").ch_names}
# HEP epoch window and cardiac-field-artifact exclusion window, matching this
# repo's own HEP analysis (6_hep_group_comparison.py: pipeline minmax=(-0.3,
# 0.4)s; permutation_cluster_jitter_test's qrs_exclude_sec=0.05) -- so "how
# much of the EEG is explained by ECG" is measured on the same epochs the
# HEP science actually uses, not an arbitrary continuous window.
HEP_TMIN, HEP_TMAX = -0.3, 0.4
QRS_EXCLUDE_SEC = 0.05
COLUMNS = ["patient_id", "recording_id", "edf_path", "eeg_channel", "component", "component_name", "n_eeg_channels", "n_components", "sfreq_hz", "recording_duration_s", "window_start_s", "window_duration_s", "window_attempt", "qc_good_eeg_fraction", "qc_ecg_beats", "qc_ecg_bpm", "qc_plausible_rr_fraction", "samples_used", "n_hep_epochs", "source_variance", "source_variance_fraction", "mixing_weight", "channel_component_variance", "channel_component_variance_fraction", "channel_hep_variance_pre_ica", "channel_hep_variance_post_ica", "channel_hep_variance_pct_drop", "ecg_score", "ecg_artifact_component", "ecg_channel", "ica_method"]
ERROR_COLUMNS = ["patient_id", "recording_id", "edf_path", "error_type", "error", "traceback"]
CACHE_VERSION = 4  # bumped: added channel_hep_variance_pre/post_ica and pct_drop (actual cleaning effect, not component share)


def discover(root: Path, limit: int | None) -> list[Path]:
    result = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if name.lower().endswith(".edf") and not name.startswith("._"):
                result.append((Path(directory) / name).absolute())
                if limit and len(result) >= limit:
                    return result
    return result


def identifiers(path: Path) -> tuple[str, str]:
    stem = path.stem
    bids = re.search(r"(?:^|_)sub-([^_]+)", stem, re.I)
    legacy = re.search(r"(?:^|\D)(\d{4}-\d{3})(?:\D|$)", stem)
    parent = next((p.name for p in path.parents if re.fullmatch(r"I\d+", p.name, re.I)), None)
    patient = bids.group(1) if bids else parent or (legacy.group(1) if legacy else stem.split("_")[0])
    return patient.upper(), stem.upper()


def _is_1020_channel(label: str) -> bool:
    """True only if every '-'-separated part (one referential site, or two
    for a bipolar derivation like 'F3-C3') names an actual 10-20/10-10
    electrode."""
    label = re.sub(r"^EEG\s+", "", label, flags=re.I).strip()
    parts = label.split("-")
    return bool(parts) and all(part.strip().upper() in _MONTAGE_1020_SITES for part in parts)


def channel_names(raw: mne.io.BaseRaw) -> tuple[list[str], str]:
    ecg = next((ch for ch in raw.ch_names if ECG_RE.search(ch)), None)
    if ecg is None:
        raise ValueError("no ECG/EKG channel found")
    eeg = [ch for ch in raw.ch_names if ch != ecg and _is_1020_channel(ch)]
    if len(eeg) < 2:
        raise ValueError(f"only {len(eeg)} plausible EEG channel(s) found")
    return eeg, ecg


def quality(segment: mne.io.BaseRaw, eeg: list[str], ecg: str) -> tuple[bool, dict]:
    data_uv = segment.get_data(picks=eeg) * 1e6
    finite = np.isfinite(data_uv).mean(axis=1) >= .999
    std = np.nanstd(data_uv, axis=1)
    ptp = np.nanmax(data_uv, axis=1) - np.nanmin(data_uv, axis=1)
    flat = np.mean(np.abs(np.diff(data_uv, axis=1)) < 1e-9, axis=1)
    good = finite & (std >= .1) & (std <= 500) & (ptp <= 5000) & (flat < .20)
    good_fraction = float(good.mean())

    signal = segment.get_data(picks=[ecg])[0]
    sfreq = float(segment.info["sfreq"])
    peaks = np.array([], dtype=int)
    bpm = rr_fraction = 0.0
    if np.isfinite(signal).mean() >= .999 and np.nanstd(signal) > 1e-9:
        centered = np.nan_to_num(signal - np.nanmedian(signal))
        upper = min(25.0, sfreq / 2 - .5)
        if upper > 5:
            filtered = sosfiltfilt(butter(3, [5, upper], btype="bandpass", fs=sfreq, output="sos"), centered)
            envelope = np.abs(filtered)
            mad = np.median(np.abs(envelope - np.median(envelope)))
            peaks, _ = find_peaks(envelope, distance=max(1, int(.30 * sfreq)), prominence=max(3 * mad, np.finfo(float).eps))
            bpm = float(len(peaks) / (len(signal) / sfreq / 60))
            if len(peaks) > 1:
                rr = np.diff(peaks) / sfreq
                rr_fraction = float(np.mean((rr >= .33) & (rr <= 2)))
    metrics = {"good_eeg_fraction": good_fraction, "ecg_beats": int(len(peaks)), "ecg_bpm": bpm, "plausible_rr_fraction": rr_fraction, "r_peak_samples": peaks}
    return good_fraction >= .70 and 35 <= bpm <= 180 and rr_fraction >= .70, metrics


def epoch_around_r_peaks(data: np.ndarray, r_peak_samples, sfreq: float, tmin: float = HEP_TMIN, tmax: float = HEP_TMAX) -> np.ndarray:
    """Slice (..., n_samples) data into (n_epochs, ..., n_epoch_samples) around each R-peak.

    Epochs that would run off either edge of `data` are dropped. Works on a
    single ECG trace (1D) or all EEG channels / ICA sources at once (2D,
    channels-first) -- same slicing either way.
    """
    pre = int(round(-tmin * sfreq))
    post = int(round(tmax * sfreq))
    n_samples = data.shape[-1]
    valid = [int(p) for p in r_peak_samples if p - pre >= 0 and p + post <= n_samples]
    if not valid:
        return np.empty((0,) + data.shape[:-1] + (pre + post,))
    return np.stack([data[..., p - pre:p + post] for p in valid], axis=0)


def select_quality_window(raw, eeg, ecg, duration, minutes, seed, path, max_attempts):
    """Pick a QC-passing window with a path+seed-derived RNG.

    Factored out so every script that needs a comparable window (e.g. a
    direct cardiac-field-artifact regression alongside this ICA run) reuses
    the exact same seeded search instead of a parallel reimplementation that
    could silently drift out of sync.
    """
    window_s = minutes * 60
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{seed}:{path}".encode()).digest()[:8], "little"))
    selected = qc = start = None
    attempts = max_attempts if duration > window_s else 1
    for attempt in range(1, attempts + 1):
        start = float(rng.uniform(0, duration - window_s)) if duration > window_s else 0.0
        candidate = raw.copy().pick(eeg + [ecg])
        if duration > window_s:
            candidate.crop(start, start + window_s, include_tmax=False)
        candidate.load_data(verbose="ERROR")
        passed, qc = quality(candidate, eeg, ecg)
        if passed:
            selected = candidate
            break
        del candidate
    if selected is None:
        raise ValueError(f"no good-quality {minutes:g}-minute window in {attempts} attempts; last QC={qc}")
    return selected, qc, start, attempt


def process(job):
    path_text, minutes, seed, max_attempts, max_components, low, high = job
    path = Path(path_text)
    patient, recording = identifiers(path)
    try:
        raw = mne.io.read_raw_edf(path, preload=False, encoding="latin1", verbose="ERROR")
        eeg, ecg = channel_names(raw)
        raw.set_channel_types({ch: "eeg" for ch in eeg}, on_unit_change="ignore", verbose="ERROR")
        raw.set_channel_types({ecg: "ecg"}, on_unit_change="ignore", verbose="ERROR")
        duration = raw.n_times / float(raw.info["sfreq"])
        selected, qc, start, attempt = select_quality_window(raw, eeg, ecg, duration, minutes, seed, path, max_attempts)

        upper = min(high, float(selected.info["sfreq"]) / 2 - .5)
        if upper <= low:
            raise ValueError(f"sampling rate {selected.info['sfreq']} Hz is too low")
        selected.filter(low, upper, picks=eeg, verbose="ERROR")
        count = min(max_components, len(eeg) - 1)
        try:
            import picard  # noqa: F401
            method, params = "picard", {"ortho": False, "extended": True}
        except ImportError:
            method, params = "infomax", {"extended": True}
        ica = mne.preprocessing.ICA(n_components=count, method=method, fit_params=params, random_state=42, max_iter=500)
        ica.fit(selected, picks=eeg, verbose="ERROR")
        bad, scores = ica.find_bads_ecg(selected, ch_name=ecg, method="correlation", verbose="ERROR")
        scores = np.asarray(scores, float)
        if not bad and scores.size:
            bad = [int(np.nanargmax(np.abs(scores)))]

        # ICA still needs to be *fit* on the continuous recording for a
        # stable unmixing matrix, but each component's reported "variance"
        # is the HEP evoked average -- the R-peak-locked, epoch-averaged
        # source waveform -- not raw continuous-window variance. That's what
        # actually answers "how much of the heartbeat-evoked EEG does this
        # component explain".
        sfreq = float(selected.info["sfreq"])
        sources = ica.get_sources(selected).get_data()
        source_epochs = epoch_around_r_peaks(sources, qc["r_peak_samples"], sfreq)
        n_hep_epochs = source_epochs.shape[0]
        if n_hep_epochs < 20:
            raise ValueError(f"only {n_hep_epochs} R-peak epochs fit in this window; too few for a HEP evoked estimate")
        evoked_sources = source_epochs.mean(axis=0)  # (n_components, n_epoch_samples)
        source_var = np.var(evoked_sources, axis=1)
        mixing = np.asarray(ica.get_components())
        contribution = mixing ** 2 * source_var[None, :]

        # The actual cleaning effect: each channel's HEP evoked variance
        # before vs. after removing the flagged ECG component(s) -- distinct
        # from source_variance above, which measures the component's own
        # share, not what excluding it actually did to the channel signal.
        cleaned = selected.copy()
        ica.apply(cleaned, exclude=bad, verbose="ERROR")
        eeg_pre = selected.get_data(picks=eeg) * 1e6
        eeg_post = cleaned.get_data(picks=eeg) * 1e6
        evoked_pre = epoch_around_r_peaks(eeg_pre, qc["r_peak_samples"], sfreq).mean(axis=0)
        evoked_post = epoch_around_r_peaks(eeg_post, qc["r_peak_samples"], sfreq).mean(axis=0)
        channel_hep_var_pre = np.var(evoked_pre, axis=1)
        channel_hep_var_post = np.var(evoked_post, axis=1)
        channel_hep_pct_drop = np.where(
            channel_hep_var_pre > 0,
            (channel_hep_var_pre - channel_hep_var_post) / channel_hep_var_pre,
            np.nan,
        )

        rows = []
        for channel_index, channel in enumerate(eeg):
            for component in range(count):
                rows.append({"patient_id": patient, "recording_id": recording, "edf_path": str(path), "eeg_channel": channel, "component": component, "component_name": f"ICA{component:03d}", "n_eeg_channels": len(eeg), "n_components": count, "sfreq_hz": sfreq, "recording_duration_s": duration, "window_start_s": start, "window_duration_s": sources.shape[1] / sfreq, "window_attempt": attempt, "qc_good_eeg_fraction": qc["good_eeg_fraction"], "qc_ecg_beats": qc["ecg_beats"], "qc_ecg_bpm": qc["ecg_bpm"], "qc_plausible_rr_fraction": qc["plausible_rr_fraction"], "samples_used": sources.shape[1], "n_hep_epochs": n_hep_epochs, "source_variance": float(source_var[component]), "source_variance_fraction": float(source_var[component] / source_var.sum()), "mixing_weight": float(mixing[channel_index, component]), "channel_component_variance": float(contribution[channel_index, component]), "channel_component_variance_fraction": float(contribution[channel_index, component] / contribution[channel_index].sum()), "channel_hep_variance_pre_ica": float(channel_hep_var_pre[channel_index]), "channel_hep_variance_post_ica": float(channel_hep_var_post[channel_index]), "channel_hep_variance_pct_drop": float(channel_hep_pct_drop[channel_index]), "ecg_score": float(scores[component]) if component < len(scores) else np.nan, "ecg_artifact_component": component in bad, "ecg_channel": ecg, "ica_method": method})
        return rows, None
    except Exception as exc:
        return [], {"patient_id": patient, "recording_id": recording, "edf_path": str(path), "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}


def atomic_parquet(frame: pd.DataFrame, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def cache_paths(path: Path, args) -> tuple[Path, Path]:
    """Return parameter-specific success/error cache paths for one EDF."""
    identity = "|".join(map(str, (
        CACHE_VERSION, path.absolute(), args.window_minutes, args.random_seed,
        args.max_window_attempts, args.max_components, args.l_freq, args.h_freq,
    )))
    key = hashlib.sha256(identity.encode()).hexdigest()
    return args.cache_dir / "results" / f"{key}.parquet", args.cache_dir / "errors" / f"{key}.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf-root", type=Path, default=HERE.parent / "EDF_Format")
    parser.add_argument("--output", type=Path, default=HERE / "ica_ecg_component_variance.parquet")
    parser.add_argument("--cache-dir", type=Path, default=HERE / "ica_ecg_component_variance_cache")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--window-minutes", type=float, default=10)
    parser.add_argument("--max-window-attempts", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-components", type=int, default=15)
    parser.add_argument("--l-freq", type=float, default=1)
    parser.add_argument("--h-freq", type=float, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-errors", action="store_true", help="Retry EDFs already stored in the error cache")
    args = parser.parse_args()
    paths = discover(args.edf_root, args.limit)
    if not paths:
        parser.error(f"no EDF files found under {args.edf_root}")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    success_caches, error_caches = [], []
    for path in paths:
        success_cache, error_cache = cache_paths(path, args)
        if success_cache.exists():
            success_caches.append(success_cache)
            continue
        if error_cache.exists() and not args.retry_errors:
            error_caches.append(error_cache)
            continue
        job = (str(path), args.window_minutes, args.random_seed, args.max_window_attempts,
               args.max_components, args.l_freq, args.h_freq)
        jobs.append((job, success_cache, error_cache))
    cached = len(success_caches) + len(error_caches)
    print(
        f"Found {len(paths)} EDF files; cached={cached}; remaining={len(jobs)}; "
        f"{args.window_minutes:g}-minute QC windows; {args.workers} workers",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process, job): (job[0], success_cache, error_cache)
            for job, success_cache, error_cache in jobs
        }
        bar = tqdm(as_completed(futures), total=len(futures), unit="EDF", desc="ICA variance", dynamic_ncols=True)
        for future in bar:
            result, error = future.result()
            edf_path, success_cache, error_cache = futures[future]
            if error:
                atomic_parquet(pd.DataFrame([error], columns=ERROR_COLUMNS), error_cache)
                error_caches.append(error_cache)
                tqdm.write(f"ERROR {edf_path}: {error['error']}")
            else:
                atomic_parquet(pd.DataFrame(result, columns=COLUMNS), success_cache)
                success_caches.append(success_cache)
                error_cache.unlink(missing_ok=True)
            bar.set_postfix(cached=len(success_caches), errors=len(error_caches), refresh=False)

    # Combine the durable per-EDF cache only after all requested EDFs finish.
    result_frames = [pd.read_parquet(path) for path in success_caches]
    error_frames = [pd.read_parquet(path) for path in error_caches]
    result_frame = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame(columns=COLUMNS)
    error_frame = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame(columns=ERROR_COLUMNS)
    atomic_parquet(result_frame, args.output)
    error_output = args.output.with_name(args.output.stem + ".errors.parquet")
    atomic_parquet(error_frame, error_output)
    print(f"Wrote {len(result_frame):,} rows to {args.output}; {len(error_frame):,} errors to {error_output}")
    return int(bool(len(error_frame) and not len(result_frame)))


if __name__ == "__main__":
    sys.exit(main())
