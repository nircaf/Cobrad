#!/usr/bin/env python3
"""Consolidate the CFA/ICA per-EDF cache shards into patient-level tables and
join demographics, so the paper-writing scripts (make_figures.py, make_pdf.py)
read one small parquet instead of re-globbing ~27k cache files.

This intentionally reads from *_cache/results/*.parquet directly rather than
the batch scripts' own combined .parquet output: as of this run the ICA batch
(ica_ecg_component_variance.py) is still executing in the background (see
ica_ecg_component_variance.log) and will not write its final combined output
until the full EDF corpus finishes, which takes days. The cache shards are
each a complete, atomically-written per-EDF result, so combining them here
gives a valid interim snapshot of however many EDFs have completed so far
without waiting on or disturbing that job.

Run:
  venv/bin/python "Paper CFA/build_dataset.py"
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

CLINICAL_FILE = REPO / "COBRAD_clinical_24022025.xlsx"
EHR_DEMOGRAPHICS_CACHE = REPO / "cache" / "ehr_demographics.parquet"
_EHR_DIAG_BASE = REPO / "EDF_Format" / "Harvard_Electroencephalography" / "EHR" / "data_Structured" / "Parquet" / "Parquet" / "bdsp_i0006_diagnosis"
_EHR_I0002_ICD10_BASE = REPO / "EDF_Format" / "Harvard_Electroencephalography" / "EHR" / "I0002-EHR" / "I0002_structured" / "icd10_codes_nax_2024_parquet"

# Diagnosis categories, copied verbatim from 16_diagnosis_sleep_stage_comparison_dashboard.py
# so the paper's diagnosis grouping matches the rest of the repo's dashboards.
DIAG_CATEGORIES = {
    "Obstructive Sleep Apnea": lambda n: "sleep apnea" in n.lower(),
    "Hypertension": lambda n: "hypertension" in n.lower(),
    "Diabetes": lambda n: "diabetes" in n.lower(),
    "Heart Failure": lambda n: "heart failure" in n.lower(),
    "Coronary Artery Disease": lambda n: "coronary" in n.lower() or "atherosclerotic heart" in n.lower(),
    "Heart Transplant": lambda n: "heart transplant" in n.lower() or "transplant status" in n.lower(),
    "Atrial Fibrillation": lambda n: "atrial fibrillation" in n.lower() or "atrial flutter" in n.lower(),
    "COPD": lambda n: "chronic obstructive" in n.lower(),
    "Obesity": lambda n: "obesity" in n.lower() or "obese" in n.lower(),
    "Depression": lambda n: "depressive" in n.lower() or "depression" in n.lower(),
    "Anxiety": lambda n: "anxiety" in n.lower(),
    "Cognitive Impairment / Dementia": lambda n: "cognitive" in n.lower() or "alzheimer" in n.lower() or "dementia" in n.lower(),
    "Stroke / Cerebrovascular": lambda n: "infarction" in n.lower() or "cerebrovascular" in n.lower() or "stroke" in n.lower() or "hemorrhage" in n.lower(),
    "Kidney Disease": lambda n: "kidney" in n.lower() or "renal" in n.lower(),
    "Anemia": lambda n: "anemia" in n.lower(),
}


def combine_cache(cache_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(cache_dir / "results" / "*.parquet")))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def pid_to_bdsp(pid: str) -> int | None:
    m = re.search(r"I\d{4}(\d{9})", str(pid))
    return int(m.group(1)) if m else None


def load_ehr_diagnosis_map(target_ids: set[int]) -> dict[int, list[str]]:
    patient_map: dict[int, set[str]] = {}

    def add(df, pid_col, dx_col):
        if df is None or df.empty:
            return
        work = df[[pid_col, dx_col]].dropna()
        work[pid_col] = pd.to_numeric(work[pid_col], errors="coerce")
        work = work.dropna(subset=[pid_col])
        work[pid_col] = work[pid_col].astype(int)
        work = work[work[pid_col].isin(target_ids)]
        work[dx_col] = work[dx_col].astype(str).str.strip()
        work = work[(work[dx_col] != "") & (work[dx_col].str.lower() != "not recorded")]
        for pid_val, grp in work.groupby(pid_col):
            patient_map.setdefault(int(pid_val), set()).update(grp[dx_col].unique().tolist())

    for ch in sorted(glob.glob(str(_EHR_DIAG_BASE / "*.parquet"))):
        try:
            add(pd.read_parquet(ch, columns=["bdsp_patient_id", "dx_name"], filters=[("bdsp_patient_id", "in", list(target_ids))]),
                "bdsp_patient_id", "dx_name")
        except Exception:
            pass
    for ch in sorted(glob.glob(str(_EHR_I0002_ICD10_BASE / "*.parquet"))):
        try:
            add(pd.read_parquet(ch, columns=["BDSPPatientID", "LongDescription"], filters=[("BDSPPatientID", "in", list(target_ids))]),
                "BDSPPatientID", "LongDescription")
        except Exception:
            pass
    return {pid: sorted(dx) for pid, dx in patient_map.items()}


def assign_categories(dx_names: list[str]) -> list[str]:
    cats = set()
    for dx in dx_names:
        for cat, test in DIAG_CATEGORIES.items():
            if test(dx):
                cats.add(cat)
    return sorted(cats)


def main():
    print("Combining CFA cache...")
    cfa = combine_cache(HERE / "cfa_variance_explained_cache")
    print(f"  {len(cfa):,} rows, {cfa.patient_id.nunique():,} patients")
    cfa.to_parquet(HERE / "cfa_combined.parquet", index=False)

    print("Combining ICA cache (job still running in background -- interim snapshot)...")
    ica = combine_cache(HERE / "ica_ecg_component_variance_cache")
    print(f"  {len(ica):,} rows, {ica.patient_id.nunique():,} patients")
    ica.to_parquet(HERE / "ica_combined.parquet", index=False)

    # ---- Demographics -------------------------------------------------
    # Harvard_Electroencephalography patient IDs (e.g. "I0006179000017")
    # convert to a 9-digit BDSPPatientID and join the age/sex cache built by
    # 19_ehr_demographics_cache.py. This is the dominant source group (>90%
    # of rows) so it is the analyzable subset for age/sex/diagnosis
    # stratification; other source cohorts (CAP Sleep Database, Berkeley,
    # legacy COBRAD "EDF" folder) lack a matching age/sex table and are
    # included only in the pooled/overall numbers.
    all_patients = pd.Index(cfa.patient_id.unique()).union(ica.patient_id.unique())
    bdsp_map = {p: pid_to_bdsp(p) for p in all_patients}
    bdsp_ids = {v for v in bdsp_map.values() if v is not None}
    print(f"Harvard-format patient IDs: {len(bdsp_ids):,} / {len(all_patients):,}")

    ehr = pd.read_parquet(EHR_DEMOGRAPHICS_CACHE)
    ehr["patient_id"] = pd.to_numeric(ehr["patient_id"], errors="coerce")
    ehr_dedup = (
        ehr.dropna(subset=["patient_id"])
        .assign(patient_id=lambda d: d.patient_id.astype("int64"))
        .groupby("patient_id")
        .agg(age=("age", "median"), sex=("sex", lambda s: s.dropna().mode().iat[0] if s.notna().any() else None))
        .reset_index()
    )
    ehr_dedup = ehr_dedup[ehr_dedup.patient_id.isin(bdsp_ids)]
    print(f"  matched demographics for {len(ehr_dedup):,} / {len(bdsp_ids):,} Harvard patients")

    print("Loading EHR diagnosis map (this scans the diagnosis parquet corpus)...")
    dx_map = load_ehr_diagnosis_map(bdsp_ids)
    print(f"  diagnosis records for {len(dx_map):,} patients")

    demo_rows = []
    for p in all_patients:
        bdsp = bdsp_map.get(p)
        if bdsp is None:
            continue
        row = ehr_dedup.loc[ehr_dedup.patient_id == bdsp]
        age = float(row.age.iat[0]) if len(row) and pd.notna(row.age.iat[0]) else None
        sex = row.sex.iat[0] if len(row) and pd.notna(row.sex.iat[0]) else None
        dx_names = dx_map.get(bdsp, [])
        demo_rows.append({
            "patient_id": p, "bdsp_patient_id": bdsp, "age": age, "sex": sex,
            "diagnosis_categories": assign_categories(dx_names), "n_diagnoses": len(dx_names),
        })
    demo = pd.DataFrame(demo_rows)
    demo.to_parquet(HERE / "demographics_combined.parquet", index=False)
    print(f"Wrote demographics_combined.parquet: {len(demo):,} rows, "
          f"{demo.age.notna().sum():,} with age, {demo.sex.notna().sum():,} with sex")


if __name__ == "__main__":
    main()
