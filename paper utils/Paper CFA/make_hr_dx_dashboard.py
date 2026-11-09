#!/usr/bin/env python3
"""Self-contained HTML dashboard: heart rate (HR) vs HEP R^2 (raw vs CFAR-cleaned), by diagnosis
(alone, overlapping, A-vs-B explorer). HEP R^2 = cfa R^2 of ECG on R-peak-locked EEG (benchmark_all.parquet:
r2_raw, r2_cfar; 6,711 held-out patients of the paper). HR = mean detected bpm over the patient's windows.

Run: venv/bin/python "paper utils/Paper CFA/make_hr_dx_dashboard.py"  -> hr_dx_dashboard.html
"""
import json, os
import numpy as np
import pandas as pd
import plotly

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda f: os.path.join(HERE, f)

b = pd.read_parquet(P("benchmark_all.parquet")).set_index("patient_id")
dem = pd.read_parquet(P("demographics_combined.parquet")).set_index("patient_id")
hr = pd.read_parquet(P("cfa_combined.parquet"), columns=["patient_id", "qc_ecg_bpm"]).groupby("patient_id").qc_ecg_bpm.mean()
d = b[["r2_raw", "r2_cfar", "r2_own", "pow_raw", "pow_cfar"]].join(dem[["age", "sex", "diagnosis_categories"]]).join(hr.rename("hr")).dropna(subset=["hr"])
DX = sorted({x for l in d.diagnosis_categories for x in l})
mask = d.diagnosis_categories.map(lambda l: sum(1 << DX.index(x) for x in l)).astype(int)
data = dict(
    dx=DX, hr=d.hr.round(2).tolist(), raw=d.r2_raw.round(4).tolist(), cf=d.r2_cfar.round(4).tolist(),
    own=d.r2_own.round(4).tolist(), praw=d.pow_raw.round(4).tolist(), pcf=d.pow_cfar.round(4).tolist(),
    mask=mask.tolist(), k=d.diagnosis_categories.map(len).tolist(),
    age=d.age.round(1).where(d.age.notna(), None).tolist(),
)
html = open(P("hr_dx_dashboard.template.html")).read()
plotly_js = open(os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")).read()
html = html.replace("/*PLOTLY*/", plotly_js).replace("/*DATA*/", "const D=" + json.dumps(data) + ";")
open(P("hr_dx_dashboard.html"), "w").write(html)
print("wrote hr_dx_dashboard.html", len(d), "patients,", len(DX), "dx")
