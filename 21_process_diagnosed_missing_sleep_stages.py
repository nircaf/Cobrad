#!/usr/bin/env python3
"""Incrementally generate missing sleep-stage pickles for diagnosed EDF patients.

The default ``dashboard`` cohort is the set of distinct Harvard subjects present
in dashboard 16's min-10-channel HEP caches and having at least one EHR diagnosis.
``all-local`` expands discovery to every locally available diagnosed Harvard EDF.

For every selected stage, only subjects without a complete patient pickle are
written to a manifest and passed to HEP_parquet_generation.py.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EDF_ROOT = SCRIPT_DIR / "EDF_Format" / "Harvard_Electroencephalography"
DEFAULT_PICKLE_ROOT = (
    SCRIPT_DIR / "pickles_sleep_stage" / "Harvard_Electroencephalography"
)
DEFAULT_PARQUET_ROOT = SCRIPT_DIR / "parquets_HEP"
DEFAULT_STAGES = ("light_sleep", "N3", "R")
VALID_STAGES = ("light_sleep", "N3", "R", "W")
MIN_COMPLETE_PICKLE_BYTES = 64 * 1024
PERMANENT_SKIP_REASONS = {
    "no_stage",
    "no_valid_segments",
    "no_segments",
    "too_short",
    "no_continuous",
}
EHR_I0006_DIAG = (
    DEFAULT_EDF_ROOT
    / "EHR"
    / "data_Structured"
    / "Parquet"
    / "Parquet"
    / "bdsp_i0006_diagnosis"
)
EHR_I0002_DIAG = (
    DEFAULT_EDF_ROOT
    / "EHR"
    / "I0002-EHR"
    / "I0002_structured"
    / "icd10_codes_nax_2024_parquet"
)


def canonical_subject_id(value: object) -> str | None:
    match = re.search(r"(I\d{13})", str(value), flags=re.IGNORECASE)
    return f"SUB-{match.group(1).upper()}" if match else None


def bdsp_id(subject_id: str) -> int:
    match = re.search(r"I\d{4}(\d{9})", subject_id)
    if not match:
        raise ValueError(f"Cannot derive BDSPPatientID from {subject_id}")
    return int(match.group(1))


def discover_local_edfs(edf_root: Path) -> dict[str, list[str]]:
    subjects: dict[str, list[str]] = {}
    for root, _dirs, files in os.walk(edf_root):
        for name in files:
            if not name.lower().endswith(".edf") or name.startswith("._"):
                continue
            path = os.path.join(root, name)
            subject_id = canonical_subject_id(name) or canonical_subject_id(root)
            if subject_id is not None:
                subjects.setdefault(subject_id, []).append(path)
    return subjects


def dashboard_cache_subjects(pickle_root: Path, stages: Iterable[str]) -> set[str]:
    subjects: set[str] = set()
    for stage in stages:
        cache_path = (
            pickle_root
            / stage
            / "individuals_cache_min10_1020eeg_rpeak_shape_v1.pkl"
        )
        if not cache_path.exists():
            continue
        with cache_path.open("rb") as cache_file:
            individuals = pickle.load(cache_file)
        for individual in individuals:
            if individual:
                subject_id = canonical_subject_id(individual[0])
                if subject_id is not None:
                    subjects.add(subject_id)
        del individuals
    return subjects


def _add_diagnosed_ids(
    output: set[int],
    paths: Iterable[str],
    patient_column: str,
    diagnosis_column: str,
    target_ids: set[int],
) -> None:
    for path in paths:
        try:
            frame = pd.read_parquet(
                path,
                columns=[patient_column, diagnosis_column],
                filters=[(patient_column, "in", list(target_ids))],
            )
        except Exception:
            # Some parquet engines cannot push down a large IN filter. Reading
            # only two columns is the bounded compatibility fallback.
            frame = pd.read_parquet(
                path, columns=[patient_column, diagnosis_column]
            )
            frame[patient_column] = pd.to_numeric(
                frame[patient_column], errors="coerce"
            )
            frame = frame[frame[patient_column].isin(target_ids)]
        if frame.empty:
            continue
        frame[patient_column] = pd.to_numeric(
            frame[patient_column], errors="coerce"
        )
        diagnoses = frame[diagnosis_column].astype(str).str.strip()
        valid = (
            frame[patient_column].notna()
            & frame[diagnosis_column].notna()
            & diagnoses.ne("")
            & diagnoses.str.lower().ne("not recorded")
        )
        output.update(frame.loc[valid, patient_column].astype(int).tolist())


def diagnosed_subjects(subjects: set[str]) -> set[str]:
    target_ids = {bdsp_id(subject) for subject in subjects}
    diagnosed_ids: set[int] = set()
    _add_diagnosed_ids(
        diagnosed_ids,
        sorted(glob.glob(str(EHR_I0006_DIAG / "*.parquet"))),
        "bdsp_patient_id",
        "dx_name",
        target_ids,
    )
    _add_diagnosed_ids(
        diagnosed_ids,
        sorted(glob.glob(str(EHR_I0002_DIAG / "*.parquet"))),
        "BDSPPatientID",
        "LongDescription",
        target_ids,
    )
    return {subject for subject in subjects if bdsp_id(subject) in diagnosed_ids}


def complete_stage_subjects(pickle_root: Path, stage: str) -> set[str]:
    subjects: set[str] = set()
    for path in (pickle_root / stage).glob("SUB-*.pkl"):
        try:
            complete = path.stat().st_size >= MIN_COMPLETE_PICKLE_BYTES
        except OSError:
            continue
        if complete:
            subject_id = canonical_subject_id(path.name)
            if subject_id is not None:
                subjects.add(subject_id)
    return subjects


def status_entries(parquet_root: Path, stage: str) -> tuple[Path, dict]:
    path = (
        parquet_root
        / f"Harvard_Electroencephalography_{stage}"
        / "run_status.json"
    )
    if not path.exists():
        return path, {}
    try:
        with path.open() as status_file:
            return path, json.load(status_file)
    except (OSError, json.JSONDecodeError):
        return path, {}


def write_manifest(path: Path, subjects: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{subject}\n" for subject in sorted(subjects)),
        encoding="utf-8",
    )


def write_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def remove_incomplete_target_pickles(
    pickle_root: Path, stage: str, targets: set[str]
) -> int:
    removed = 0
    for path in (pickle_root / stage).glob("SUB-*.pkl"):
        subject_id = canonical_subject_id(path.name)
        if subject_id not in targets:
            continue
        try:
            if path.stat().st_size >= MIN_COMPLETE_PICKLE_BYTES:
                continue
            path.unlink()
            removed += 1
        except FileNotFoundError:
            continue
    return removed


def clear_stale_successes(
    status_path: Path, status: dict, missing_subjects: set[str]
) -> int:
    stale_keys = [
        patient_id
        for patient_id, entry in status.items()
        if canonical_subject_id(patient_id) in missing_subjects
        and entry.get("status") == "success"
    ]
    if not stale_keys:
        return 0
    for patient_id in stale_keys:
        del status[patient_id]
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = status_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(temporary_path, status_path)
    return len(stale_keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort",
        choices=("dashboard", "all-local"),
        default="dashboard",
        help=(
            "dashboard: diagnosed subjects in dashboard 16's min-10-channel "
            "caches; all-local: every local EDF subject with an EHR diagnosis"
        ),
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=VALID_STAGES,
        default=list(DEFAULT_STAGES),
    )
    parser.add_argument("--edf-root", type=Path, default=DEFAULT_EDF_ROOT)
    parser.add_argument("--pickle-root", type=Path, default=DEFAULT_PICKLE_ROOT)
    parser.add_argument("--parquet-root", type=Path, default=DEFAULT_PARQUET_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--retry-permanent",
        action="store_true",
        help="Retry patients previously marked as structurally lacking a stage",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        SCRIPT_DIR / "logs" / "21_diagnosed_missing_stages" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning local EDF files under {args.edf_root} ...", flush=True)
    local_edfs = discover_local_edfs(args.edf_root)
    local_subjects = set(local_edfs)
    if args.cohort == "dashboard":
        candidates = dashboard_cache_subjects(args.pickle_root, args.stages)
        candidates &= local_subjects
    else:
        candidates = local_subjects

    cohort = diagnosed_subjects(candidates)
    print(
        f"Cohort source: {args.cohort}; local EDF subjects: {len(local_subjects)}; "
        f"candidate subjects: {len(candidates)}; diagnosed subjects: {len(cohort)}",
        flush=True,
    )

    manifests: dict[str, Path] = {}
    report_rows: list[dict] = []
    total_to_run = 0
    for stage in args.stages:
        complete = complete_stage_subjects(args.pickle_root, stage)
        missing = cohort - complete
        status_path, status = status_entries(args.parquet_root, stage)
        permanent = {
            subject
            for patient_id, entry in status.items()
            if (subject := canonical_subject_id(patient_id)) in missing
            and entry.get("status") == "failed"
            and entry.get("reason") in PERMANENT_SKIP_REASONS
        }
        to_run = missing if args.retry_permanent else missing - permanent
        manifest = output_dir / "manifests" / f"{stage}.txt"
        write_manifest(manifest, to_run)
        manifests[stage] = manifest
        total_to_run += len(to_run)
        report_rows.append(
            {
                "stage": stage,
                "cohort_subjects": len(cohort),
                "complete_pickle": len(cohort & complete),
                "missing_pickle": len(missing),
                "permanent_unavailable": len(permanent),
                "scheduled": len(to_run),
                "manifest": str(manifest),
            }
        )
        print(
            f"{stage}: complete={len(cohort & complete)}, missing={len(missing)}, "
            f"permanent={len(permanent)}, scheduled={len(to_run)}",
            flush=True,
        )

    write_report(output_dir / "summary.csv", report_rows)
    write_manifest(output_dir / "diagnosed_subjects.txt", cohort)
    print(f"Discovery report: {output_dir / 'summary.csv'}", flush=True)

    if not args.execute:
        print(f"Dry run complete; {total_to_run} patient-stage jobs would run.")
        return 0

    for row in report_rows:
        stage = row["stage"]
        scheduled = int(row["scheduled"])
        if scheduled == 0:
            continue
        manifest = manifests[stage]
        missing_targets = set(manifest.read_text(encoding="utf-8").splitlines())
        removed = remove_incomplete_target_pickles(
            args.pickle_root, stage, missing_targets
        )
        status_path, status = status_entries(args.parquet_root, stage)
        cleared = clear_stale_successes(status_path, status, missing_targets)
        if removed or cleared:
            print(
                f"{stage}: removed {removed} incomplete pickle(s), cleared "
                f"{cleared} stale success status(es).",
                flush=True,
            )

        command = [
            sys.executable,
            str(SCRIPT_DIR / "HEP_parquet_generation.py"),
            "--edf_root",
            str(args.edf_root),
            "--mode",
            "stage",
            "--stage",
            stage,
            "--patient-ids-file",
            str(manifest),
        ]
        if args.retry_permanent:
            command.append("--rerun-failed")
        print(f"Running {stage}: {' '.join(command)}", flush=True)
        completed = subprocess.run(command, cwd=SCRIPT_DIR)
        if completed.returncode != 0:
            print(
                f"{stage} processing failed with exit code {completed.returncode}.",
                file=sys.stderr,
            )
            return completed.returncode

    print("All scheduled stage runs finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
