"""
Recompute the authoritative current N/p/age_median/Table1 numbers (no figure
writing, no risk of clobbering the already-fixed agesplit_*.png files), since
concurrent cache rebuilds on this machine mean the old stage_delta_age_results.pkl
may be stale. Prints a JSON blob to stdout; also dumps to
Paper1/stage_delta_age_results_v2.json.

Run with the project venv: source venv/bin/activate && python3 Paper1/recompute_current_numbers.py
"""
import os
import sys
import types
import json
import importlib.util

import numpy as np
import pandas as pd

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
OUT_DIR = os.path.join(REPO, "Paper1")
N_PERM = 200
STAGES = ["light_sleep", "N3", "R"]
CHECKPOINT_PATH = os.path.join(OUT_DIR, "recompute_checkpoint.json")


def _load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"pairwise": {}, "age_split": {}}


def _save_checkpoint(ckpt):
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f, indent=2)
    os.replace(tmp, CHECKPOINT_PATH)


def _make_mock_streamlit():
    class _NoOp:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return _NoOp()
        def __getattr__(self, name): return _NoOp()

    def _cache_data(func=None, **kwargs):
        return func if func is not None else (lambda f: f)

    st = types.ModuleType("streamlit")
    st.cache_data = _cache_data
    st.cache_resource = _cache_data
    st.session_state = {}
    st.sidebar = _NoOp()
    st.__enter__ = lambda s: s
    st.__exit__ = lambda s, *a: False
    for name in ["warning", "error", "info", "write", "title", "header", "subheader",
                 "markdown", "text", "pyplot", "spinner", "set_page_config", "stop",
                 "experimental_rerun", "rerun"]:
        setattr(st, name, (lambda *a, **kw: None))
    st.empty = lambda *a, **kw: _NoOp()
    st.progress = lambda *a, **kw: _NoOp()
    st.columns = lambda *a, **kw: [_NoOp() for _ in range(a[0] if a else 2)]
    st.tabs = lambda labels: [_NoOp() for _ in labels]
    st.expander = lambda *a, **kw: _NoOp()
    components = types.ModuleType("streamlit.components")
    components.v1 = _NoOp()
    st.components = components
    return st


def main():
    mock_st = _make_mock_streamlit()
    sys.modules["streamlit"] = mock_st
    for submod in ["streamlit.components", "streamlit.components.v1"]:
        sys.modules.setdefault(submod, types.ModuleType(submod))

    hep_path = os.path.join(REPO, "6_hep_group_comparison.py")
    spec = importlib.util.spec_from_file_location("hep_module", hep_path)
    hep_mod = importlib.util.module_from_spec(spec)
    sys.modules["hep_module"] = hep_mod
    spec.loader.exec_module(hep_mod)

    dash_path = os.path.join(REPO, "16_diagnosis_sleep_stage_comparison_dashboard.py")
    spec2 = importlib.util.spec_from_file_location("dash16_module", dash_path)
    dash16 = importlib.util.module_from_spec(spec2)
    sys.modules["dash16_module"] = dash16
    spec2.loader.exec_module(dash16)

    raw_df = dash16.load_patient_data(
        hep_mod, ["Harvard_Electroencephalography"], STAGES,
        force_rebuild=False, apply_ica=False, min_eeg_channels=10,
    )
    n_cohort = int(raw_df["patient_id"].nunique())
    print(f"n_cohort={n_cohort}", flush=True)

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

    ehr_df = dash16.load_ehr_data(tuple(raw_df["patient_id"].unique()))
    healthy = dash16.select_non_diagnosis_cohort(raw_df, ehr_df)
    print(f"susp_epilepsy_n={healthy['patient_id'].nunique()}", flush=True)

    ckpt = _load_checkpoint()

    healthy_pairs = dash16._rank_stage_pairs(
        healthy, "Susp. Epilepsy", STAGES, selected_channels, n_permutations=N_PERM,
    )
    pairwise = []
    for result in healthy_pairs:
        stage_a, stage_b = result["stage_a"], result["stage_b"]
        key = f"{stage_a}-{stage_b}"
        if key in ckpt["pairwise"]:
            row = ckpt["pairwise"][key]
            print(f"PAIRWISE {key}: [from checkpoint] N={row['n']} p={row['p_formatted']} "
                  f"sig={row['n_cluster_significant']}/{row['n_electrodes_tested']} "
                  f"fdr={row['n_fdr_significant']} chans={row['sig_channels']}", flush=True)
            pairwise.append(row)
            continue
        channel_results = dash16.prepare_electrode_contrasts(
            healthy, "Susp. Epilepsy", stage_a, "Susp. Epilepsy", stage_b,
            selected_channels, False, N_PERM, paired=True,
        )
        sig_channels = sorted(ch for ch, r in channel_results.items() if r.get("cluster_significant"))
        n_fdr = sum(1 for r in channel_results.values() if r.get("fdr_significant"))
        row = {
            "stage_a": stage_a, "stage_b": stage_b,
            "n": len(result["patient_ids"]), "p_formatted": dash16.format_p(result["p_value"]),
            "n_electrodes_tested": len(channel_results), "n_cluster_significant": len(sig_channels),
            "n_fdr_significant": n_fdr, "sig_channels": sig_channels,
        }
        pairwise.append(row)
        ckpt["pairwise"][key] = row
        _save_checkpoint(ckpt)
        print(f"PAIRWISE {stage_a}-{stage_b}: N={row['n']} p={row['p_formatted']} "
              f"sig={len(sig_channels)}/{len(channel_results)} fdr={n_fdr} chans={sig_channels}", flush=True)

    sick = raw_df[~raw_df["patient_id"].isin(healthy["patient_id"])].copy()
    sick["diagnosis_group"] = "Diagnosed (any)"
    combined = pd.concat([healthy, sick], ignore_index=True)

    clinical_df = dash16.load_cobrad_clinical()
    demo = dash16._patient_demographics(combined["patient_id"].unique(), clinical_df)
    valid_ages = demo.set_index("patient_id")["age"].dropna()
    age_median = float(valid_ages.median())
    print(f"age_median={age_median}", flush=True)
    age_group_map = {pid: ("Older" if age >= age_median else "Younger") for pid, age in valid_ages.items()}
    grouped = combined.copy()
    grouped["diagnosis_group"] = grouped["patient_id"].astype(str).map(age_group_map)
    grouped = grouped[grouped["diagnosis_group"].isin(["Younger", "Older"])].copy()

    age_split = []
    for stage in STAGES:
        if stage in ckpt["age_split"]:
            row = ckpt["age_split"][stage]
            print(f"AGESPLIT {stage}: [from checkpoint] Ny={row['n_younger']} No={row['n_older']} "
                  f"p={row['p_formatted']} sig={row['n_cluster_significant']}/{row['n_electrodes_tested']} "
                  f"fdr={row['n_fdr_significant']} chans={row['sig_channels']}", flush=True)
            age_split.append(row)
            continue
        result = dash16.prepare_waveform_contrast(
            grouped, "Younger", stage, "Older", stage, selected_channels, False, N_PERM, paired=False,
        )
        channel_results = dash16.prepare_electrode_contrasts(
            grouped, "Younger", stage, "Older", stage, selected_channels, False, N_PERM, paired=False,
        )
        sig_channels = sorted(ch for ch, r in channel_results.items() if r.get("cluster_significant"))
        n_fdr = sum(1 for r in channel_results.values() if r.get("fdr_significant"))
        row = {
            "stage": stage, "n_younger": len(result["patient_ids_a"]),
            "n_older": len(result["patient_ids_b"]), "p_formatted": dash16.format_p(result["contrast_p"]),
            "n_electrodes_tested": len(channel_results), "n_cluster_significant": len(sig_channels),
            "n_fdr_significant": n_fdr, "sig_channels": sig_channels,
        }
        age_split.append(row)
        ckpt["age_split"][stage] = row
        _save_checkpoint(ckpt)
        print(f"AGESPLIT {stage}: Ny={row['n_younger']} No={row['n_older']} p={row['p_formatted']} "
              f"sig={len(sig_channels)}/{len(channel_results)} fdr={n_fdr} chans={sig_channels}", flush=True)

    def summarize_demo(patient_ids, label):
        d = dash16._patient_demographics(list(patient_ids), clinical_df)
        ages = d["age"].dropna()
        return {
            "label": label, "n": len(patient_ids), "n_with_age": len(ages),
            "age_mean": float(ages.mean()) if len(ages) else float("nan"),
            "age_median": float(ages.median()) if len(ages) else float("nan"),
            "age_sd": float(ages.std()) if len(ages) else float("nan"),
            "age_min": float(ages.min()) if len(ages) else float("nan"),
            "age_max": float(ages.max()) if len(ages) else float("nan"),
            "n_male": int((d["sex"] == "Male").sum()), "n_female": int((d["sex"] == "Female").sum()),
            "n_unknown": int((d["sex"] == "Unknown").sum()),
        }

    matched_ids = healthy_pairs[0]["patient_ids"] if healthy_pairs else []
    table1 = [
        summarize_demo(combined["patient_id"].unique(), "Selected cohort (light/N3/REM, >=10 EEG ch)"),
        summarize_demo(matched_ids, "Susp. Epilepsy matched 3-stage subset"),
    ]
    for row in table1:
        print(f"TABLE1 {row['label']}: n={row['n']} age_mean={row['age_mean']:.1f} "
              f"age_median={row['age_median']:.1f} age_sd={row['age_sd']:.1f} "
              f"range=[{row['age_min']:.0f},{row['age_max']:.0f}] "
              f"M={row['n_male']} F={row['n_female']} Unk={row['n_unknown']}", flush=True)

    out = {"n_cohort": n_cohort, "selected_channels": selected_channels,
           "pairwise": pairwise, "age_split": age_split, "age_median": age_median,
           "table1": table1}
    with open(os.path.join(OUT_DIR, "stage_delta_age_results_v2.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
