#!/usr/bin/env python3
"""Build diagnosed-subject and eligible local Harvard BIDS PSG manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HARVARD_ROOT = (
    SCRIPT_DIR / "EDF_Format" / "Harvard_Electroencephalography"
)
DEFAULT_EHR_ROOT = DEFAULT_HARVARD_ROOT / "EHR"
DEFAULT_BIDS_ROOT = DEFAULT_HARVARD_ROOT / "bids"
DEFAULT_STUDIES = ("I0002", "I0003", "I0004", "I0006")

DIAGNOSIS_SOURCES = {
    "I0002": (
        Path("I0002-EHR/I0002_structured/icd10_codes_nax_2024_parquet"),
        "BDSPPatientID",
        "LongDescription",
    ),
    "I0003": (
        Path("I0003-EHR/data_Structured/diagnosis"),
        "BDSPPatientID",
        "Diagnosis",
    ),
    "I0004": (
        Path("I0004-EHR/data_Structured/bdsp_diagnoses_i0004"),
        "bdsp_patient_id",
        "description",
    ),
    "I0006": (
        Path(
            "I0006-EHR/data_Structured/Parquet/Parquet/"
            "bdsp_i0006_diagnosis"
        ),
        "bdsp_patient_id",
        "dx_name",
    ),
}

CANONICAL_1020 = {
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "T3", "T7", "C3", "CZ", "C4", "T4", "T8",
    "T5", "P7", "P3", "PZ", "P4", "T6", "P8",
    "O1", "OZ", "O2",
}
INVALID_DIAGNOSIS_VALUES = {
    "", "-", "--", "n/a", "na", "nan", "none", "not recorded", "null",
}


def _atomic_write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        "".join(f"{value}\n" for value in sorted(set(values))),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    fieldnames = list(rows[0]) if rows else [
        "study", "subject", "edf_path", "duration_hours",
        "eeg_electrodes", "eligible", "reason",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _parquet_files(root: Path) -> list[Path]:
    return sorted(root.glob("*.parquet")) or sorted(root.rglob("*.parquet"))


def _valid_diagnosis_mask(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip().str.lower()
    return values.notna() & ~normalized.isin(INVALID_DIAGNOSIS_VALUES)


def diagnosed_subjects_for_study(ehr_root: Path, study: str) -> set[str]:
    if study not in DIAGNOSIS_SOURCES:
        raise ValueError(f"No diagnosis source configured for {study}")
    relative_root, patient_column, diagnosis_column = DIAGNOSIS_SOURCES[study]
    source_root = ehr_root / relative_root
    parquet_files = _parquet_files(source_root)
    if not parquet_files:
        raise FileNotFoundError(
            f"No diagnosis parquet files found for {study}: {source_root}"
        )

    patient_ids: set[int] = set()
    failed_files: list[str] = []
    for parquet_path in parquet_files:
        try:
            frame = pd.read_parquet(
                parquet_path,
                columns=[patient_column, diagnosis_column],
            )
        except Exception as exc:
            failed_files.append(f"{parquet_path}: {exc}")
            continue
        numeric_ids = pd.to_numeric(frame[patient_column], errors="coerce")
        valid = numeric_ids.notna() & _valid_diagnosis_mask(
            frame[diagnosis_column]
        )
        patient_ids.update(numeric_ids.loc[valid].astype(int).tolist())

    if failed_files:
        preview = "\n".join(failed_files[:5])
        raise RuntimeError(
            f"Failed to read {len(failed_files)} diagnosis parquet file(s) "
            f"for {study}. First failures:\n{preview}"
        )
    return {
        f"sub-{study}{patient_id:09d}"
        for patient_id in patient_ids
        if 0 <= patient_id <= 999_999_999
    }


def write_diagnosis_manifests(
    ehr_root: Path,
    output_dir: Path,
    studies: Iterable[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}
    for study in studies:
        subjects = diagnosed_subjects_for_study(ehr_root, study)
        _atomic_write_lines(output_dir / f"{study}.txt", subjects)
        summary[study] = len(subjects)
        print(
            f"{study}: {len(subjects)} subjects with at least one diagnosis",
            flush=True,
        )
    summary_path = output_dir / "summary.json"
    temporary = summary_path.with_name(
        f".{summary_path.name}.tmp.{os.getpid()}"
    )
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary, summary_path)


def edf_duration_seconds(edf_path: Path) -> float:
    """Read EDF fixed-header duration without loading signal samples."""
    with edf_path.open("rb") as edf_file:
        header = edf_file.read(256)
    if len(header) < 252:
        raise ValueError("EDF fixed header is shorter than 252 bytes")
    try:
        n_records = float(header[236:244].decode("ascii").strip())
        record_seconds = float(header[244:252].decode("ascii").strip())
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("EDF duration fields are invalid") from exc
    duration = n_records * record_seconds
    if n_records < 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError(
            f"EDF duration is unavailable (records={n_records}, "
            f"record_seconds={record_seconds})"
        )
    return duration


def _normalized_electrode(channel_name: object) -> str | None:
    raw = str(channel_name).strip()
    raw = re.sub(r"^EEG\s+", "", raw, flags=re.IGNORECASE)
    candidates = [raw]
    for separator in ("-", "_", " "):
        if separator in raw:
            candidates.append(raw.split(separator, 1)[0])
    for candidate in candidates:
        normalized = re.sub(r"[^A-Za-z0-9]", "", candidate).upper()
        if normalized in CANONICAL_1020:
            return normalized
    return None


def count_good_eeg_electrodes(channels_path: Path) -> int:
    channels = pd.read_csv(channels_path, sep="\t", dtype=str)
    lowered_columns = {
        str(column).strip().lower(): column for column in channels.columns
    }
    name_column = lowered_columns.get("name", channels.columns[0])
    status_column = lowered_columns.get("status")
    if status_column is not None:
        good = (
            channels[status_column]
            .fillna("good")
            .astype(str)
            .str.strip()
            .str.lower()
            .ne("bad")
        )
        channels = channels.loc[good]
    return len({
        electrode
        for value in channels[name_column]
        if (electrode := _normalized_electrode(value)) is not None
    })


def _subject_from_path(path: Path) -> tuple[str | None, str | None]:
    for part in path.parts:
        match = re.fullmatch(r"sub-(I\d{13})", part, flags=re.IGNORECASE)
        if match:
            subject = f"sub-{match.group(1).upper()}"
            return match.group(1).upper()[:5], subject
    return None, None


def _channels_sidecar_for_edf(edf_path: Path) -> Path | None:
    expected_name = re.sub(
        r"_eeg\.edf$",
        "_channels.tsv",
        edf_path.name,
        flags=re.IGNORECASE,
    )
    expected = edf_path.with_name(expected_name)
    if expected.exists():
        return expected
    expected_lower = expected_name.lower()
    for candidate in edf_path.parent.iterdir():
        if candidate.is_file() and candidate.name.lower() == expected_lower:
            return candidate
    return None


def build_local_eligible_manifest(
    bids_root: Path,
    diagnosis_dir: Path,
    output_path: Path,
    report_path: Path,
    studies: Iterable[str],
    min_duration_hours: float,
    min_eeg_electrodes: int,
) -> int:
    diagnosed_by_study: dict[str, set[str]] = {}
    for study in studies:
        diagnosis_path = diagnosis_dir / f"{study}.txt"
        if not diagnosis_path.exists():
            raise FileNotFoundError(
                f"Missing diagnosed-subject manifest: {diagnosis_path}"
            )
        diagnosed_by_study[study] = set(
            diagnosis_path.read_text(encoding="utf-8").splitlines()
        )

    rows: list[dict] = []
    eligible_paths: list[str] = []
    for study in studies:
        study_root = bids_root / study
        if not study_root.exists():
            continue
        discovered_edfs = sorted(
            path
            for eeg_dir in study_root.glob("sub-*/ses-*/eeg")
            for path in eeg_dir.iterdir()
            if path.is_file()
            and re.search(r"_task-PSG_eeg\.edf$", path.name, re.IGNORECASE)
        )
        # Older processing code could uppercase BIDS EDF filenames. Treat
        # case-only duplicates as the same recording.
        edf_paths = list({
            str(path).lower(): path for path in discovered_edfs
        }.values())
        for edf_path in edf_paths:
            path_study, subject = _subject_from_path(edf_path)
            reason = ""
            duration_hours = math.nan
            electrode_count = 0
            if path_study != study or subject is None:
                reason = "unrecognized_subject"
            elif subject not in diagnosed_by_study[study]:
                reason = "no_diagnosis"
            else:
                try:
                    duration_hours = edf_duration_seconds(edf_path) / 3600.0
                except (OSError, ValueError):
                    reason = "invalid_edf_duration"
                if not reason and duration_hours < min_duration_hours:
                    reason = "shorter_than_minimum"
                channels_path = _channels_sidecar_for_edf(edf_path)
                if not reason and channels_path is None:
                    reason = "missing_channels_tsv"
                if not reason:
                    try:
                        electrode_count = count_good_eeg_electrodes(
                            channels_path
                        )
                    except Exception:
                        reason = "invalid_channels_tsv"
                if not reason and electrode_count < min_eeg_electrodes:
                    reason = "too_few_eeg_electrodes"

            eligible = not reason
            if eligible:
                eligible_paths.append(str(edf_path.resolve()))
                reason = "eligible"
            rows.append({
                "study": study,
                "subject": subject or "",
                "edf_path": str(edf_path.resolve()),
                "duration_hours": (
                    f"{duration_hours:.6f}"
                    if math.isfinite(duration_hours) else ""
                ),
                "eeg_electrodes": electrode_count,
                "eligible": eligible,
                "reason": reason,
            })

    _atomic_write_lines(output_path, eligible_paths)
    _atomic_write_csv(report_path, rows)
    print(
        f"Eligible local PSG recordings: {len(eligible_paths)} / {len(rows)}; "
        f"manifest: {output_path}",
        flush=True,
    )
    return len(eligible_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnosis_parser = subparsers.add_parser("diagnosis-manifests")
    diagnosis_parser.add_argument("--ehr-root", type=Path, default=DEFAULT_EHR_ROOT)
    diagnosis_parser.add_argument("--output-dir", type=Path, required=True)
    diagnosis_parser.add_argument(
        "--studies", nargs="+", default=list(DEFAULT_STUDIES)
    )

    local_parser = subparsers.add_parser("eligible-local")
    local_parser.add_argument("--bids-root", type=Path, default=DEFAULT_BIDS_ROOT)
    local_parser.add_argument("--diagnosis-dir", type=Path, required=True)
    local_parser.add_argument("--output", type=Path, required=True)
    local_parser.add_argument("--report", type=Path, required=True)
    local_parser.add_argument(
        "--studies", nargs="+", default=list(DEFAULT_STUDIES)
    )
    local_parser.add_argument("--min-duration-hours", type=float, default=7.0)
    local_parser.add_argument("--min-eeg-electrodes", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "diagnosis-manifests":
            write_diagnosis_manifests(
                args.ehr_root,
                args.output_dir,
                args.studies,
            )
        else:
            build_local_eligible_manifest(
                args.bids_root,
                args.diagnosis_dir,
                args.output,
                args.report,
                args.studies,
                args.min_duration_hours,
                args.min_eeg_electrodes,
            )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
