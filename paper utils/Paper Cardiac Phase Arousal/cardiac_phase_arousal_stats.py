#!/usr/bin/env python3
"""Circular statistics on the cardiac phase of cortical-arousal onsets.

Reads events.parquet (build_dataset.py), writes paper_stats.json and
cardiac_phase_arousal.parquet (one row per stratum).

  source venv/bin/activate && python3 "paper utils/Paper Cardiac Phase Arousal/cardiac_phase_arousal_stats.py"
  ... --selftest    run the Rayleigh implementation self-check only
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
EVENTS = HERE / "events.parquet"
SESSIONS = HERE / "sessions.parquet"
OUT_PARQUET = HERE / "cardiac_phase_arousal.parquet"
OUT_JSON = HERE / "paper_stats.json"
LOG = HERE / "cardiac_phase_arousal.log"

N_BINS = 12
MAIN_STAGES = ["N1", "N2", "N3", "REM", "W"]


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def rayleigh(phi: np.ndarray) -> dict:
    """Rayleigh test of uniformity on the circle (Zar, Biostatistical Analysis)."""
    phi = np.asarray(phi, dtype=float)
    n = phi.size
    if n < 2:
        return dict(n=n, R=np.nan, Z=np.nan, p=np.nan, mean_phase=np.nan)
    C, S = np.cos(phi).mean(), np.sin(phi).mean()
    R = float(np.hypot(C, S))
    Z = n * R * R
    p = float(np.exp(-Z) * (1 + (2 * Z - Z**2) / (4 * n)
                            - (24 * Z - 132 * Z**2 + 76 * Z**3 - 9 * Z**4) / (288 * n**2)))
    p = float(min(max(p, 0.0), 1.0))
    mean_phase = float(np.arctan2(S, C) % (2 * np.pi))
    return dict(n=int(n), R=R, Z=float(Z), p=p, mean_phase=mean_phase)


def rayleigh_permutation(phi: np.ndarray, n_perm: int = 10000, seed: int = 0) -> float:
    """Sanity check on the analytic p-value: R under random phases is
    distribution-free, so resample uniform phases of the same n."""
    rng = np.random.default_rng(seed)
    obs = rayleigh(phi)["Z"]
    null = np.array([rayleigh(rng.uniform(0, 2 * np.pi, phi.size))["Z"] for _ in range(n_perm)])
    return float((null >= obs).mean())


def chi2_uniform(phi: np.ndarray, n_bins: int = N_BINS) -> dict:
    """Exposure-time-normalised chi-square GOF. Phase bins are equal-proportion by
    construction, so every cardiac cycle of duration T contributes exactly T/n_bins
    seconds of time-at-risk to each bin: expected counts are uniform."""
    counts, _ = np.histogram(phi % (2 * np.pi), bins=n_bins, range=(0, 2 * np.pi))
    chi2, p = stats.chisquare(counts)
    return dict(counts=counts.astype(int).tolist(), chi2=float(chi2), p=float(p),
                dof=int(n_bins - 1), n=int(counts.sum()))


def stratum_row(name: str, group: str, phi: np.ndarray, n_subjects: int) -> dict:
    r = rayleigh(phi)
    c = chi2_uniform(phi)
    return dict(stratum=name, group=group, n_events=r["n"], n_subjects=n_subjects,
                R=r["R"], Z=r["Z"], p_rayleigh=r["p"], mean_phase_rad=r["mean_phase"],
                mean_phase_deg=np.degrees(r["mean_phase"]),
                mean_phase_frac_rr=r["mean_phase"] / (2 * np.pi),
                chi2=c["chi2"], p_chi2=c["p"], bin_counts=json.dumps(c["counts"]))


def selftest() -> None:
    rng = np.random.default_rng(1)
    u = rayleigh(rng.uniform(0, 2 * np.pi, 20000))
    assert u["p"] > 0.01 and u["R"] < 0.05, u
    conc = rayleigh(rng.vonmises(1.0, 0.3, 20000))          # weak clustering at 1.0 rad
    assert conc["p"] < 1e-10 and abs(conc["mean_phase"] - 1.0) < 0.1, conc
    assert chi2_uniform(rng.uniform(0, 2 * np.pi, 20000))["p"] > 0.01
    assert chi2_uniform(rng.vonmises(1.0, 0.3, 20000))["p"] < 1e-10
    assert rayleigh_permutation(rng.uniform(0, 2 * np.pi, 500), n_perm=500) > 0.01
    print("selftest OK")


def main() -> None:
    ev = pd.read_parquet(EVENTS)
    ses = pd.read_parquet(SESSIONS)
    log(f"=== cardiac phase arousal stats: {len(ev):,} events, "
        f"{ev['subject'].nunique():,} subjects, {len(ses):,} sessions ===")

    rows = [stratum_row("pooled", "all", ev["phi"].to_numpy(), ev["subject"].nunique())]

    for st in MAIN_STAGES:
        g = ev[ev["stage"] == st]
        if len(g) >= 100:
            rows.append(stratum_row("stage", st, g["phi"].to_numpy(), g["subject"].nunique()))

    for sx in ["Female", "Male"]:
        g = ev[ev["sex"] == sx]
        if len(g) >= 100:
            rows.append(stratum_row("sex", sx, g["phi"].to_numpy(), g["subject"].nunique()))

    age_med = float(ev["age_years"].median())
    for lab, g in [("age <= median", ev[ev["age_years"] <= age_med]),
                   ("age > median", ev[ev["age_years"] > age_med])]:
        if len(g) >= 100:
            rows.append(stratum_row("age", lab, g["phi"].to_numpy(), g["subject"].nunique()))

    res = pd.DataFrame(rows)
    res["p_bonferroni"] = np.minimum(res["p_rayleigh"] * len(res), 1.0)
    res.to_parquet(OUT_PARQUET, index=False)
    for _, r in res.iterrows():
        log(f"{r['stratum']:>7} {r['group']:>14}  n={r['n_events']:>7,}  R={r['R']:.4f}  "
            f"Z={r['Z']:.2f}  p={r['p_rayleigh']:.3g}  phase={r['mean_phase_deg']:.1f} deg  "
            f"chi2={r['chi2']:.1f} p={r['p_chi2']:.3g}")

    # per-subject mean resultant vectors -> second-level test that is immune to
    # a few high-event-count subjects dominating the pooled estimate
    per_sub = (ev.groupby("subject")["phi"]
                 .apply(lambda x: np.exp(1j * x.to_numpy()).mean() if len(x) >= 20 else np.nan)
                 .dropna())
    sub_level = rayleigh(np.angle(per_sub.to_numpy()))

    pooled = rows[0]
    perm_p = rayleigh_permutation(
        ev["phi"].sample(min(len(ev), 5000), random_state=0).to_numpy(), n_perm=2000)

    stats_json = dict(
        cohort=dict(
            n_sessions=int(len(ses)), n_subjects=int(ev["subject"].nunique()),
            n_sessions_available=9231,
            n_events=int(len(ev)),
            n_rr_intervals=int(ses["n_rr_valid"].sum()),
            n_rr_intervals_total=int(ses["n_rr_total"].sum()),
            recording_hours=float(ses["recording_hours"].sum()),
            age_median=age_med, age_min=float(ev["age_years"].min()),
            age_max=float(ev["age_years"].max()),
            sex_counts={k: int(v) for k, v in
                        ses["sex"].value_counts(dropna=True).items()},
            n_sessions_sex_missing=int(ses["sex"].isna().sum()),
            stage_counts={k: int(v) for k, v in ev["stage"].value_counts().items()},
            median_rr=float(ev["rr"].median()),
            n_bins=N_BINS,
        ),
        pooled={k: v for k, v in pooled.items() if k != "bin_counts"},
        pooled_bin_counts=json.loads(pooled["bin_counts"]),
        pooled_permutation_p=perm_p,
        subject_level=dict(sub_level, n_subjects_included=int(len(per_sub))),
        strata=[{k: v for k, v in r.items() if k != "bin_counts"}
                | {"p_bonferroni": float(min(r["p_rayleigh"] * len(rows), 1.0))}
                for r in rows],
        n_strata_tested=len(rows),
    )
    with open(OUT_JSON, "w") as f:
        json.dump(stats_json, f, indent=2)
    log(f"WROTE {OUT_JSON.name} and {OUT_PARQUET.name}")
    log(f"pooled permutation p (5k subsample, 2k perms) = {perm_p:.4g}; "
        f"subject-level Rayleigh p = {sub_level['p']:.3g} (n={len(per_sub)} subjects)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        selftest()
        main()
