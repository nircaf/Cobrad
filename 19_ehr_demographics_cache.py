"""Scan EHR parquet files for age & sex per patient, cache to one parquet.

Source: EDF_Format/Harvard_Electroencephalography/EHR/I000{1,2,3,4,6,9}-EHR
(I0007 has no structured demographics — notes only, skipped).

Run: python 19_ehr_demographics_cache.py [--force]
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

_EHR_BASE = "/storage/pblab_shared_data2/Nir/Cobrad/EDF_Format/Harvard_Electroencephalography/EHR"
CACHE_PATH = "/storage/pblab_shared_data2/Nir/Cobrad/cache/ehr_demographics.parquet"

# (glob pattern, pid_col, dob_col, sex_col, reference_date_col-or-None)
_SOURCES = [
    (f"{_EHR_BASE}/I0001-EHR/data_Structured/old_backup/I0001_ParquetFiles_batch1/demographics_I0001/*.parquet",
     "BDSPPatientID", "DateOfBirth", "SexDSC", "BDSPLastModifiedDTS"),
    (f"{_EHR_BASE}/I0002-EHR/I0002_structured/patient_nax_2024_parquet/*.parquet",
     "BDSPPatientID", "DateOfBirth", "SexDSC", "BDSPLastModifiedDTS"),
    (f"{_EHR_BASE}/I0003-EHR/data_Structured/patient/*.parquet",
     "BDSPPatientID", "BirthDate", "Gender", "BDSPLastModifiedDTS"),
    (f"{_EHR_BASE}/I0004-EHR/data_Structured/I0004_patient_demographics_2ndApril2026.parquet",
     "BDSPPatientID", "DateOfBirth", "Gender", "RecentEncounterDate"),
    (f"{_EHR_BASE}/I0006-EHR/data_Structured/Parquet/Parquet/bdsp_i0006_demographics/*.parquet",
     "bdsp_patient_id", "shifted_date_of_birth", "gender", None),
    (f"{_EHR_BASE}/I0009-EHR/I0009_patient_28thApril2026.parquet",
     "BDSPPatientID", "DateOfBirth", "Gender", "BDSPLastModifiedDTS"),
]

_SEX_MAP = {"male": "Male", "m": "Male", "female": "Female", "f": "Female"}


def _normalize_sex(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(_SEX_MAP)


def _extract_one_source(pattern: str, pid_col: str, dob_col: str, sex_col: str, ref_col: str | None) -> pd.DataFrame:
    columns = [pid_col, dob_col, sex_col] + ([ref_col] if ref_col else [])
    frames = []
    for path in sorted(glob.glob(pattern)):
        try:
            frames.append(pd.read_parquet(path, columns=columns))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["patient_id", "date_of_birth", "sex", "reference_date"])
    df = pd.concat(frames, ignore_index=True)

    out = pd.DataFrame()
    out["patient_id"] = pd.to_numeric(df[pid_col], errors="coerce")
    out["date_of_birth"] = pd.to_datetime(df[dob_col], errors="coerce")
    out["sex"] = _normalize_sex(df[sex_col])
    out["reference_date"] = pd.to_datetime(df[ref_col], errors="coerce") if ref_col else pd.NaT
    out = out.dropna(subset=["patient_id"])
    out["patient_id"] = out["patient_id"].astype("int64")
    return out


def build_ehr_demographics_cache(force: bool = False) -> pd.DataFrame:
    """Return {patient_id, age, sex} — cached to CACHE_PATH after the first scan."""
    if not force and os.path.exists(CACHE_PATH):
        return pd.read_parquet(CACHE_PATH)

    parts = [_extract_one_source(*source) for source in _SOURCES]
    combined = pd.concat(parts, ignore_index=True)

    # BDSPLastModifiedDTS is the closest available proxy for "as of" date when a
    # source has no dedicated encounter date; age is only ever approximate.
    fallback_ref = combined["reference_date"].dropna()
    global_fallback = fallback_ref.median() if len(fallback_ref) else pd.Timestamp.today()
    ref = combined["reference_date"].fillna(global_fallback)
    combined["age"] = ((ref - combined["date_of_birth"]).dt.days / 365.25).round(1)
    combined.loc[(combined["age"] < 0) | (combined["age"] > 110), "age"] = pd.NA

    # One row per patient: first non-null age, first non-null sex across sources
    # (GroupBy.first skips NaN by default).
    result = (
        combined.sort_values("patient_id")
        .groupby("patient_id", as_index=False)
        .agg(age=("age", "first"), sex=("sex", "first"))
    )

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    result.to_parquet(CACHE_PATH, index=False)
    return result


def _demo():
    df = build_ehr_demographics_cache(force=False)
    assert {"patient_id", "age", "sex"} <= set(df.columns)
    assert df["patient_id"].is_unique
    assert df["age"].dropna().between(0, 110).all()
    print(f"cached {len(df)} patients -> {CACHE_PATH}")
    print(df.describe(include="all"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rescan EHR parquets, ignore existing cache")
    args = parser.parse_args()
    build_ehr_demographics_cache(force=args.force)
    _demo()
