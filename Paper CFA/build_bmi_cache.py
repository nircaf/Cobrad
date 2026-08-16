#!/usr/bin/env python3
"""Cache per-patient BMI across every linked EHR site that has it.

- I0002 (BIDMC): derive BMI from both the longitudinal vitals flowsheet
  (height in cm, weight in kg) and the admission table (height in inches,
  weight in kg). The admission table substantially improves coverage.
- I0004: flowsheet has BMI computed directly (Measure == "BMI").
- I0006: raw height_cm/daily_weight_kg vitals table -> derive BMI.
- I0003, I0007, I0009: no vitals/flowsheet table exists at all (checked).

All three keyed by the 9-digit bdsp_patient_id, so this stays a plain
concat once resolved to one row per patient per site. cfa_combined.parquet
(the CFA R^2 cache used by make_figures.py) currently only contains I0002/
I0003 patients -- the I0004/I0006 CFA batch hasn't reached those EDFs yet
(see cfa_variance_explained.log, ~35% through, alphabetical glob order) --
so I0004/I0006 rows here won't join to any R^2 yet, but will automatically
once that job catches up and cfa_combined.parquet is rebuilt.

Run: venv/bin/python "Paper CFA/build_bmi_cache.py"
"""
import glob
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EHR = "/storage/pblab_shared_data2/Nir/Cobrad/EDF_Format/Harvard_Electroencephalography/EHR"
OUT = os.path.join(HERE, "bmi_combined.parquet")

PLAUSIBLE_LO, PLAUSIBLE_HI = 10, 80


def i0002_bmi() -> pd.DataFrame:
    vitals_dir = f"{EHR}/I0002-EHR/I0002_structured/vitals_nax_2024_parquet"
    wanted = {"Height (cm)", "Weight in kg"}
    chunks = []
    for f in sorted(glob.glob(os.path.join(vitals_dir, "*.parquet"))):
        df = pd.read_parquet(f, columns=["BDSPPatientID", "FieldNM", "FieldValue"])
        chunks.append(df[df.FieldNM.isin(wanted)])
    raw = pd.concat(chunks, ignore_index=True)
    raw["FieldValue"] = pd.to_numeric(raw["FieldValue"], errors="coerce")
    raw = raw.dropna(subset=["FieldValue"])
    ht = raw[raw.FieldNM == "Height (cm)"].groupby("BDSPPatientID")["FieldValue"].median().rename("height_cm")
    wt = raw[raw.FieldNM == "Weight in kg"].groupby("BDSPPatientID")["FieldValue"].median().rename("weight_kg")
    bmi_vitals = pd.concat([ht, wt], axis=1).dropna()
    bmi_vitals["bmi"] = bmi_vitals["weight_kg"] / (bmi_vitals["height_cm"] / 100) ** 2
    bmi_vitals = bmi_vitals[["bmi"]]

    admission_file = f"{EHR}/I0002-EHR/I0002_structured/adt_admission_nax_2024.parquet"
    admission = pd.read_parquet(admission_file, columns=["BDSPPatientID", "Height", "Weight"])
    admission["Height"] = pd.to_numeric(admission["Height"], errors="coerce")
    admission["Weight"] = pd.to_numeric(admission["Weight"], errors="coerce")
    # The admission table stores adult height in inches and weight in kilograms.
    admission = admission[
        admission["Height"].between(36, 90) & admission["Weight"].between(20, 400)
    ]
    admission_ht = admission.groupby("BDSPPatientID")["Height"].median()
    admission_wt = admission.groupby("BDSPPatientID")["Weight"].median()
    bmi_admission = pd.concat([admission_ht, admission_wt], axis=1).dropna()
    bmi_admission["bmi"] = bmi_admission["Weight"] / (bmi_admission["Height"] * 0.0254) ** 2
    bmi_admission = bmi_admission[["bmi"]]

    # Collapse the two independently recorded estimates without privileging a source.
    bmi = pd.concat([bmi_vitals, bmi_admission]).groupby(level=0)["bmi"].median().reset_index()
    return bmi.rename(columns={"BDSPPatientID": "bdsp_patient_id"})[["bdsp_patient_id", "bmi"]]


def i0004_bmi() -> pd.DataFrame:
    flow_dir = f"{EHR}/I0004-EHR/data_Structured/bdsp_flowsheets_I0004"
    # "BMI (CALCULATED) (KG/M2)" carries most of the volume; the others are
    # the same quantity under differently-templated flowsheet rows.
    bmi_measures = {"BMI", "BMI (CALCULATED)", "BMI (CALCULATED) (KG/M2)", "EXTERNAL BMI"}
    chunks = []
    for f in sorted(glob.glob(os.path.join(flow_dir, "*.parquet"))):
        df = pd.read_parquet(f, columns=["BDSPPatientID", "Measure", "Value"])
        chunks.append(df[df.Measure.isin(bmi_measures)])
    raw = pd.concat(chunks, ignore_index=True)
    raw["Value"] = pd.to_numeric(raw["Value"], errors="coerce")
    raw = raw.dropna(subset=["Value"])
    bmi = raw.groupby("BDSPPatientID")["Value"].median().rename("bmi").reset_index()
    return bmi.rename(columns={"BDSPPatientID": "bdsp_patient_id"})[["bdsp_patient_id", "bmi"]]


def i0006_bmi() -> pd.DataFrame:
    vitals_dir = f"{EHR}/data_Structured/Parquet/Parquet/bdsp_i0006_vitals"
    chunks = []
    for f in sorted(glob.glob(os.path.join(vitals_dir, "*.parquet"))):
        df = pd.read_parquet(f, columns=["bdsp_patient_id", "height_cm", "daily_weight_kg"])
        chunks.append(df.dropna(subset=["height_cm", "daily_weight_kg"], how="all"))
    raw = pd.concat(chunks, ignore_index=True)
    raw["height_cm"] = pd.to_numeric(raw["height_cm"], errors="coerce")
    raw["daily_weight_kg"] = pd.to_numeric(raw["daily_weight_kg"], errors="coerce")
    ht = raw.dropna(subset=["height_cm"]).groupby("bdsp_patient_id")["height_cm"].median()
    wt = raw.dropna(subset=["daily_weight_kg"]).groupby("bdsp_patient_id")["daily_weight_kg"].median()
    bmi = pd.concat([ht.rename("height_cm"), wt.rename("weight_kg")], axis=1).dropna()
    bmi["bmi"] = bmi["weight_kg"] / (bmi["height_cm"] / 100) ** 2
    return bmi.reset_index()[["bdsp_patient_id", "bmi"]]


parts = {"I0002": i0002_bmi(), "I0004": i0004_bmi(), "I0006": i0006_bmi()}
for site, df in parts.items():
    print(f"{site}: {len(df):,} patients with raw BMI")

combined = pd.concat(parts.values(), ignore_index=True)
combined = combined[(combined.bmi >= PLAUSIBLE_LO) & (combined.bmi <= PLAUSIBLE_HI)]
# a patient could in principle appear at >1 site under the same bdsp id;
# median across sites collapses that rather than double-counting.
combined = combined.groupby("bdsp_patient_id", as_index=False)["bmi"].median()

combined.to_parquet(OUT)
print(f"wrote {len(combined):,} patients total to {OUT}")
