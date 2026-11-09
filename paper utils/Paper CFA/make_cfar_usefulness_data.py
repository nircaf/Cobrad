#!/usr/bin/env python3
"""Where is CFAR most useful? Patient-level ICA vs. CFAR (and ps-CFAR) by subgroup, bootstrapped over sample size.

Per patient (benchmark_all.parquet, n = 6,711, 6-site montage): ECG-explained EEG variance left after cleaning,
% of raw (as Figure 2d). Advantage = ICA left − method left (percentage points; > 0 = cleaner than ICA).
For each subgroup (sex, BMI class, age band, raw-CFA quartile, diagnosis) and each sample size n, 400 bootstrap
draws of n patients give the median advantage (95% interval) and power = share of draws where a Wilcoxon test
finds the method cleaner than ICA at p < 0.05.

Writes cfar_usefulness.json (read by the dashboard page). Run:
  venv/bin/python "paper utils/Paper CFA/make_cfar_usefulness_data.py"
"""
import json, os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
NS = [5, 10, 20, 30, 50, 100, 200, 500, 1000]
N_BOOT = 400
METHODS = {"cfar": "CFAR (no ECG)", "own": "Patient-specific CFAR"}


def data():
    b = pd.read_parquet(os.path.join(HERE, "benchmark_all.parquet"))
    for m in ["ica", "cfar", "own"]:
        b[f"left_{m}"] = b[f"pow_{m}"] / b.pow_raw * 100
    cd = pd.read_parquet(os.path.join(HERE, "cleaning_demo.parquet"))[["patient_id", "r2_pre", "bmi", "sex", "age"]]
    demo = pd.read_parquet(os.path.join(HERE, "demographics_combined.parquet"))[["patient_id", "diagnosis_categories"]]
    d = b.merge(cd, on="patient_id", how="left").merge(demo, on="patient_id", how="left")
    return d.dropna(subset=["left_ica", "left_cfar", "left_own"])


def groups(d):
    g = {"All patients": {"All": np.ones(len(d), bool)}}
    g["Sex"] = {s: (d.sex == s).values for s in ["Female", "Male"]}
    g["BMI"] = {"< 25": (d.bmi < 25).values, "25–30": d.bmi.between(25, 30, inclusive="left").values,
                "30–35": d.bmi.between(30, 35, inclusive="left").values, "≥ 35": (d.bmi >= 35).values}
    g["Age"] = {"< 40": (d.age < 40).values, "40–60": d.age.between(40, 60, inclusive="left").values,
                "60–75": d.age.between(60, 75, inclusive="left").values, "≥ 75": (d.age >= 75).values}
    q = d.r2_pre.quantile([0.25, 0.5, 0.75]).values
    g["Raw CFA (R² quartile)"] = {f"Q1 (< {q[0]:.2f})": (d.r2_pre < q[0]).values,
                                  f"Q2": ((d.r2_pre >= q[0]) & (d.r2_pre < q[1])).values,
                                  f"Q3": ((d.r2_pre >= q[1]) & (d.r2_pre < q[2])).values,
                                  f"Q4 (≥ {q[2]:.2f})": (d.r2_pre >= q[2]).values}
    cats = pd.read_parquet(os.path.join(HERE, "fig2_dx_forest.parquet")).category.tolist()
    g["Diagnosis"] = {c: d.diagnosis_categories.apply(lambda v: v is not None and c in list(v)).values for c in cats}
    return g


def boot(args):
    adv, seed = args
    rng = np.random.default_rng(seed)
    out = {"n": [], "median": [], "lo": [], "hi": [], "power": []}
    for n in NS:
        if n > len(adv):
            continue
        meds, sig = [], 0
        for _ in range(N_BOOT):
            s = adv[rng.integers(0, len(adv), n)]
            meds.append(np.median(s))
            if np.any(s != 0) and stats.wilcoxon(s, alternative="greater").pvalue < 0.05:
                sig += 1
        out["n"].append(n)
        out["median"].append(float(np.median(meds)))
        lo, hi = np.percentile(meds, [2.5, 97.5])
        out["lo"].append(float(lo))
        out["hi"].append(float(hi))
        out["power"].append(sig / N_BOOT)
    return out


def main():
    d = data()
    G = groups(d)
    jobs, keys = [], []
    for dim, subs in G.items():
        for name, mask in subs.items():
            if mask.sum() < 20:
                continue
            for m in METHODS:
                adv = (d.left_ica - d[f"left_{m}"]).values[mask]
                jobs.append((adv, hash((dim, name, m)) % 2**32))
                keys.append((dim, name, m, mask))
    with ProcessPoolExecutor(32) as ex:
        res = list(ex.map(boot, jobs))
    rows = []
    for (dim, name, m, mask), r in zip(keys, res):
        sub = d[mask]
        adv = (sub.left_ica - sub[f"left_{m}"]).values
        rows.append({"dimension": dim, "group": name, "method": m, "n_patients": int(mask.sum()),
                     "ica_left": float(sub.left_ica.median()), "method_left": float(sub[f"left_{m}"].median()),
                     "advantage": float(np.median(adv)), "pct_better": float((adv > 0).mean() * 100),
                     "raw_r2": float(sub.r2_pre.mean()), "boot": r})
    out = {"methods": METHODS, "ns": NS, "n_boot": N_BOOT, "n_total": int(len(d)), "rows": rows,
           "dimensions": list(G)}
    json.dump(out, open(os.path.join(HERE, "cfar_usefulness.json"), "w"))
    t = pd.DataFrame([{k: v for k, v in r.items() if k != "boot"} for r in rows])
    print(t[t.method == "cfar"].round(1).to_string())


if __name__ == "__main__":
    main()
