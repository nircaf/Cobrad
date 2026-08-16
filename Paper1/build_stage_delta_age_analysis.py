"""
Reproduce, headlessly, the two dashboard-16 analyses requested for the new
paper focus:
  1. Susp. Epilepsy pairwise within-subject stage-delta contrasts (paired
     cluster-permutation), matched N=2090.
  2. Younger vs Older (age median-split) independent-samples contrasts per
     sleep stage, on the full selected cohort (~N=2661).

Calls dash16 / hep_mod functions directly (no re-implementation of stats or
plotting). Mock-streamlit import pattern copied from Paper1/analysis_age_by_stage.py.

Run with the project venv: source venv/bin/activate && python3 Paper1/build_stage_delta_age_analysis.py
"""
from __future__ import annotations

import os
import sys
import types
import pickle
import importlib.util
import time

import numpy as np
import pandas as pd

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
OUT_DIR = os.path.join(REPO, "Paper1")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
RESULTS_PKL = os.path.join(OUT_DIR, "stage_delta_age_results.pkl")

GROUP = "Harvard_Electroencephalography"
STAGES = ["light_sleep", "N3", "R"]
N_PERM = 200

TARGETS_PAIRWISE = {
    ("light_sleep", "N3"): 0.0050,
    ("N3", "R"): 0.0050,
    ("light_sleep", "R"): 0.0199,
}
TARGETS_AGE = {"light_sleep": 0.0050, "N3": 0.0100, "R": 0.0050}


# ---------------------------------------------------------------------------
# Mock streamlit + module loaders (copied pattern from analysis_age_by_stage.py)
# ---------------------------------------------------------------------------
def _make_mock_streamlit():
    class _NoOp:
        def __init__(self, *a, **kw):
            pass
        def __call__(self, *a, **kw):
            return _NoOp()
        def progress(self, *a, **kw):
            pass
        def empty(self, *a, **kw):
            pass
        def text(self, *a, **kw):
            pass
        def warning(self, *a, **kw):
            pass
        def error(self, *a, **kw):
            pass
        def info(self, *a, **kw):
            pass
        def __getattr__(self, name):
            return _NoOp()

    def _cache_data(func=None, **kwargs):
        if func is not None:
            return func
        return lambda f: f

    def _cache_resource(func=None, **kwargs):
        if func is not None:
            return func
        return lambda f: f

    st = types.ModuleType("streamlit")
    st.cache_data = _cache_data
    st.cache_resource = _cache_resource
    st.session_state = {}
    st.progress = lambda value=0, text=None, **kw: _NoOp()
    st.empty = lambda *a, **kw: _NoOp()
    st.warning = lambda msg, *a, **kw: print(f"[WARN] {msg}", file=sys.stderr)
    st.error = lambda msg, *a, **kw: print(f"[ERROR] {msg}", file=sys.stderr)
    st.info = lambda msg, *a, **kw: None
    st.write = lambda *a, **kw: None
    st.title = lambda *a, **kw: None
    st.header = lambda *a, **kw: None
    st.subheader = lambda *a, **kw: None
    st.markdown = lambda *a, **kw: None
    st.text = lambda *a, **kw: None
    st.pyplot = lambda *a, **kw: None
    st.sidebar = _NoOp()
    st.columns = lambda *a, **kw: [_NoOp() for _ in range(a[0] if a else 2)]
    st.tabs = lambda labels: [_NoOp() for _ in labels]
    st.expander = lambda *a, **kw: _NoOp()
    st.spinner = lambda *a, **kw: _NoOp()
    st.set_page_config = lambda *a, **kw: None
    st.stop = lambda: None
    st.experimental_rerun = lambda: None
    st.rerun = lambda: None
    st.__enter__ = lambda s: s
    st.__exit__ = lambda s, *a: False
    components = types.ModuleType("streamlit.components")
    components.v1 = _NoOp()
    st.components = components
    return st


def load_hep_module():
    hep_path = os.path.join(REPO, "6_hep_group_comparison.py")
    spec = importlib.util.spec_from_file_location("hep_module", hep_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hep_module"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_dash16_module():
    dash_path = os.path.join(REPO, "16_diagnosis_sleep_stage_comparison_dashboard.py")
    spec = importlib.util.spec_from_file_location("dash16_module", dash_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dash16_module"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    t0 = time.time()
    mock_st = _make_mock_streamlit()
    sys.modules["streamlit"] = mock_st
    for submod in ["streamlit.components", "streamlit.components.v1"]:
        if submod not in sys.modules:
            sys.modules[submod] = types.ModuleType(submod)

    print("Loading hep_module (6_hep_group_comparison.py) via mock streamlit...")
    hep_mod = load_hep_module()
    print("Loading dash16 (16_diagnosis_sleep_stage_comparison_dashboard.py)...")
    dash16 = load_dash16_module()

    # -----------------------------------------------------------------
    # Cohort construction
    # -----------------------------------------------------------------
    print("Loading patient data (min_eeg_channels=10, apply_ica=False)...")
    raw_df = dash16.load_patient_data(
        hep_mod, [GROUP], STAGES,
        force_rebuild=False, apply_ica=False, min_eeg_channels=10,
    )
    n_unique = raw_df["patient_id"].nunique()
    print(f"raw_df: {len(raw_df)} rows, {n_unique} unique patients")
    if n_unique < 500:
        print("!!! Cohort n is wildly smaller than target ~2661. STOP and diagnose.")
        sys.exit(1)

    # Channel selection: every MONTAGE_1020_CHANNEL_ORDER channel present
    # (case-insensitively) in any loaded individual's ch_names.
    montage_1020_channel_order = dash16.MONTAGE_1020_CHANNEL_ORDER
    montage_1020_by_lower = {c.lower(): c for c in montage_1020_channel_order}
    selected_channels = sorted({
        montage_1020_by_lower[str(ch).lower()]
        for individual in raw_df["individual"]
        if individual is not None and len(individual) > 3 and individual[3] is not None
        for ch in individual[3]
        if str(ch).lower() in montage_1020_by_lower
    })
    preferred_rank = {c.lower(): i for i, c in enumerate(montage_1020_channel_order)}
    selected_channels.sort(key=lambda c: (preferred_rank.get(c.lower(), len(preferred_rank)), c.lower()))
    print(f"selected_channels ({len(selected_channels)}): {selected_channels}")

    # -----------------------------------------------------------------
    # Diagnosis split
    # -----------------------------------------------------------------
    print("Loading EHR data / splitting Susp. Epilepsy cohort...")
    ehr_df = dash16.load_ehr_data(tuple(raw_df["patient_id"].unique()))
    healthy = dash16.select_non_diagnosis_cohort(raw_df, ehr_df)
    print(f"Susp. Epilepsy cohort: {healthy['patient_id'].nunique()} unique patients")

    # -----------------------------------------------------------------
    # 1) Pairwise Susp. Epilepsy stage-delta contrasts
    # -----------------------------------------------------------------
    print("\n=== Pairwise Susp. Epilepsy stage-delta contrasts (n_permutations=200) ===")
    healthy_pairs = dash16._rank_stage_pairs(
        healthy, "Susp. Epilepsy", STAGES, selected_channels, n_permutations=N_PERM,
    )
    print(f"Got {len(healthy_pairs)} pairwise results")

    pairwise_results = []
    for result in healthy_pairs:
        stage_a, stage_b = result["stage_a"], result["stage_b"]
        n = len(result["patient_ids"])
        p = result["p_value"]
        target = TARGETS_PAIRWISE.get((stage_a, stage_b)) or TARGETS_PAIRWISE.get((stage_b, stage_a))
        print(f"  {stage_a} - {stage_b}: N={n}, p={dash16.format_p(p)} (target {target})")

        waveform_figure = dash16._plot_delta_summary_and_tstat(
            result, "Susp. Epilepsy", zscore=False, hep_mod=hep_mod,
            n_permutations=N_PERM, show_patient_lines=False,
        )
        channel_results = dash16.prepare_electrode_contrasts(
            healthy, "Susp. Epilepsy", stage_a, "Susp. Epilepsy", stage_b,
            selected_channels, False, N_PERM, paired=True,
        )
        stage_label = f"{dash16.STAGE_LABELS[stage_a]} vs {dash16.STAGE_LABELS[stage_b]}"
        waveform_figure = dash16.add_topomap_inset_to_waveform(
            waveform_figure, channel_results, "Susp. Epilepsy", "Susp. Epilepsy", stage_label,
        )
        fig_name = f"pairwise_{stage_a}_vs_{stage_b}.png"
        fig_path = os.path.join(FIG_DIR, fig_name)
        waveform_figure.write_image(fig_path, scale=3)
        print(f"    wrote {fig_path} ({os.path.getsize(fig_path)} bytes)")

        topo_fig = dash16.make_red_significance_topomap(channel_results, "Susp. Epilepsy", "Susp. Epilepsy", stage_label)
        topo_path = os.path.join(FIG_DIR, f"pairwise_{stage_a}_vs_{stage_b}_topomap.png")
        topo_fig.savefig(topo_path, dpi=150, bbox_inches="tight", facecolor="white")
        import matplotlib.pyplot as plt
        plt.close(topo_fig)

        n_sig = sum(1 for r in channel_results.values() if r.get("cluster_significant"))
        n_fdr_sig = sum(1 for r in channel_results.values() if r.get("fdr_significant"))
        sig_channels = sorted(ch for ch, r in channel_results.items() if r.get("cluster_significant"))
        print(f"    electrodes tested={len(channel_results)}, cluster-significant={n_sig}, "
              f"FDR-significant={n_fdr_sig}, sig channels={sig_channels}")

        pairwise_results.append({
            "stage_a": stage_a, "stage_b": stage_b, "n": n, "p_value": p,
            "p_formatted": dash16.format_p(p), "target_p": target,
            "fig_path": fig_path, "topo_path": topo_path,
            "n_electrodes_tested": len(channel_results),
            "n_cluster_significant": n_sig, "n_fdr_significant": n_fdr_sig,
            "sig_channels": sig_channels,
            "channel_results_summary": {
                ch: {"p_value": r["contrast_p"], "q_value": r.get("q_value"),
                     "cluster_significant": r.get("cluster_significant"),
                     "fdr_significant": r.get("fdr_significant")}
                for ch, r in channel_results.items()
            },
        })

    # -----------------------------------------------------------------
    # 2) Age median-split, per stage
    # -----------------------------------------------------------------
    print("\n=== Age median-split per stage (independent samples, n_permutations=200) ===")
    sick = raw_df[~raw_df["patient_id"].isin(healthy["patient_id"])].copy()
    sick["diagnosis_group"] = "Diagnosed (any)"
    combined = pd.concat([healthy, sick], ignore_index=True)
    print(f"combined cohort: {combined['patient_id'].nunique()} unique patients "
          f"(healthy={healthy['patient_id'].nunique()}, sick={sick['patient_id'].nunique()})")

    clinical_df = dash16.load_cobrad_clinical()
    demo = dash16._patient_demographics(combined["patient_id"].unique(), clinical_df)
    valid_ages = demo.set_index("patient_id")["age"].dropna()
    age_median = float(valid_ages.median())
    print(f"age_median={age_median}, n with valid age={len(valid_ages)}")
    age_group_map = {pid: ("Older" if age >= age_median else "Younger") for pid, age in valid_ages.items()}

    grouped = combined.copy()
    grouped["diagnosis_group"] = grouped["patient_id"].astype(str).map(age_group_map)
    grouped = grouped[grouped["diagnosis_group"].isin(["Younger", "Older"])].copy()

    age_split_results = []
    for stage in STAGES:
        result = dash16.prepare_waveform_contrast(
            grouped, "Younger", stage, "Older", stage, selected_channels, False, N_PERM, paired=False,
        )
        p = result["contrast_p"]
        n_y, n_o = len(result["patient_ids_a"]), len(result["patient_ids_b"])
        target = TARGETS_AGE[stage]
        print(f"  {stage}: N_younger={n_y}, N_older={n_o}, p={dash16.format_p(p)} (target {target})")

        waveform_figure = dash16.plot_waveform_contrast(
            result, "Younger", "Older", zscore_subjects=False, stage_label=dash16.STAGE_LABELS[stage],
        )
        channel_results = dash16.prepare_electrode_contrasts(
            grouped, "Younger", stage, "Older", stage, selected_channels, False, N_PERM, paired=False,
        )
        waveform_figure = dash16.add_topomap_inset_to_waveform(
            waveform_figure, channel_results, "Younger", "Older", dash16.STAGE_LABELS[stage],
        )
        fig_path = os.path.join(FIG_DIR, f"agesplit_{stage}.png")
        waveform_figure.write_image(fig_path, scale=3)
        print(f"    wrote {fig_path} ({os.path.getsize(fig_path)} bytes)")

        n_sig = sum(1 for r in channel_results.values() if r.get("cluster_significant"))
        n_fdr_sig = sum(1 for r in channel_results.values() if r.get("fdr_significant"))
        sig_channels = sorted(ch for ch, r in channel_results.items() if r.get("cluster_significant"))
        print(f"    electrodes tested={len(channel_results)}, cluster-significant={n_sig}, "
              f"FDR-significant={n_fdr_sig}, sig channels={sig_channels}")

        age_split_results.append({
            "stage": stage, "n_younger": n_y, "n_older": n_o, "p_value": p,
            "p_formatted": dash16.format_p(p), "target_p": target,
            "fig_path": fig_path,
            "n_electrodes_tested": len(channel_results),
            "n_cluster_significant": n_sig, "n_fdr_significant": n_fdr_sig,
            "sig_channels": sig_channels,
        })

    # -----------------------------------------------------------------
    # Table 1: cohort demographics
    # -----------------------------------------------------------------
    def summarize_demo(patient_ids, label):
        d = dash16._patient_demographics(list(patient_ids), clinical_df)
        ages = d["age"].dropna()
        n_male = int((d["sex"] == "Male").sum())
        n_female = int((d["sex"] == "Female").sum())
        n_unknown = int((d["sex"] == "Unknown").sum())
        return {
            "label": label, "n": len(patient_ids), "n_with_age": len(ages),
            "age_mean": float(ages.mean()) if len(ages) else np.nan,
            "age_median": float(ages.median()) if len(ages) else np.nan,
            "age_sd": float(ages.std()) if len(ages) else np.nan,
            "age_min": float(ages.min()) if len(ages) else np.nan,
            "age_max": float(ages.max()) if len(ages) else np.nan,
            "n_male": n_male, "n_female": n_female, "n_unknown": n_unknown,
        }

    matched_3stage_ids = healthy_pairs[0]["patient_ids"] if healthy_pairs else []
    table1 = [
        summarize_demo(combined["patient_id"].unique(), "Selected cohort (light/N3/REM, >=10 EEG ch)"),
        summarize_demo(matched_3stage_ids, "Susp. Epilepsy matched 3-stage subset"),
    ]
    print("\n=== Table 1 ===")
    for row in table1:
        print(f"  {row['label']}: n={row['n']} age_mean={row['age_mean']:.1f} "
              f"age_median={row['age_median']:.1f} age_sd={row['age_sd']:.1f} "
              f"range=[{row['age_min']:.0f},{row['age_max']:.0f}] "
              f"M={row['n_male']} F={row['n_female']} Unk={row['n_unknown']}")

    # -----------------------------------------------------------------
    # Verification summary
    # -----------------------------------------------------------------
    print("\n=== VERIFICATION: reproduced vs target ===")
    for r in pairwise_results:
        match = "OK" if r["target_p"] is not None and r["p_formatted"] == f"{r['target_p']:.4f}" else "MISMATCH"
        print(f"  Pairwise {r['stage_a']}-{r['stage_b']}: N={r['n']} p={r['p_formatted']} "
              f"target={r['target_p']} [{match}]")
    for r in age_split_results:
        match = "OK" if f"{r['p_value']:.4f}" == f"{r['target_p']:.4f}" else "MISMATCH"
        print(f"  AgeSplit {r['stage']}: N_y={r['n_younger']} N_o={r['n_older']} p={r['p_formatted']} "
              f"target={r['target_p']} [{match}]")

    results = dict(
        n_cohort=n_unique, selected_channels=selected_channels,
        pairwise_results=pairwise_results, age_split_results=age_split_results,
        table1=table1, age_median=age_median,
    )
    with open(RESULTS_PKL, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {RESULTS_PKL}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
