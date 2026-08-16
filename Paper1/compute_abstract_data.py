"""
Slow step: compute the 3 contrasts (REM-Deep Sleep, REM-Light Sleep,
Older-vs-Younger REM) -- both the whole-scalp waveform contrast and the
per-electrode topomap contrast -- and cache everything needed to plot them
to Paper1/abstract_data.pkl. Content (N, p-values, clusters, per-electrode
significance) never changes just because a figure's font sizes or layout
change, so this is split out from plot_abstract_figure.py: rerun this only
when the underlying data/cohort changes; rerun the plot script (seconds,
not minutes) for any purely cosmetic iteration.

Run: source venv/bin/activate && python3 Paper1/compute_abstract_data.py
"""
import os
import sys
import types
import pickle
import importlib.util

import numpy as np
import pandas as pd

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
OUT_DIR = os.path.join(REPO, "Paper1")
DATA_PKL = os.path.join(OUT_DIR, "abstract_data.pkl")
N_PERM = 200
STAGES = ["light_sleep", "N3", "R"]


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


def load_modules():
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
    return hep_mod, dash16


def _slim_result(result):
    """Keep only what plotting needs -- drop nothing plot-relevant, but this
    is already just plain arrays/dicts/lists, so pickle directly."""
    return {
        "matrix_a": np.asarray(result["matrix_a"]),
        "matrix_b": np.asarray(result["matrix_b"]),
        "times": np.asarray(result["times"]),
        "clusters": result["clusters"],
        "contrast_p": result["contrast_p"],
        "paired": result["paired"],
    }


def main():
    hep_mod, dash16 = load_modules()

    raw_df = dash16.load_patient_data(
        hep_mod, ["Harvard_Electroencephalography"], STAGES,
        force_rebuild=False, apply_ica=False, min_eeg_channels=10,
    )
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

    def pairwise_result(stage_a, stage_b):
        return dash16.prepare_waveform_contrast(
            healthy, "Susp. Epilepsy", stage_a, "Susp. Epilepsy", stage_b,
            selected_channels, False, N_PERM, paired=True,
        )

    rem_n3 = pairwise_result("R", "N3")
    rem_light = pairwise_result("R", "light_sleep")

    sick = raw_df[~raw_df["patient_id"].isin(healthy["patient_id"])].copy()
    sick["diagnosis_group"] = "Diagnosed (any)"
    combined = pd.concat([healthy, sick], ignore_index=True)
    clinical_df = dash16.load_cobrad_clinical()
    demo = dash16._patient_demographics(combined["patient_id"].unique(), clinical_df)
    valid_ages = demo.set_index("patient_id")["age"].dropna()
    age_median = float(valid_ages.median())
    age_group_map = {pid: ("Older" if age >= age_median else "Younger") for pid, age in valid_ages.items()}
    grouped = combined.copy()
    grouped["diagnosis_group"] = grouped["patient_id"].astype(str).map(age_group_map)
    grouped = grouped[grouped["diagnosis_group"].isin(["Younger", "Older"])].copy()

    old_young_rem = dash16.prepare_waveform_contrast(
        grouped, "Older", "R", "Younger", "R", selected_channels, False, N_PERM, paired=False,
    )

    rem_n3_channels = dash16.prepare_electrode_contrasts(
        healthy, "Susp. Epilepsy", "R", "Susp. Epilepsy", "N3", selected_channels, False, N_PERM, paired=True,
    )
    rem_light_channels = dash16.prepare_electrode_contrasts(
        healthy, "Susp. Epilepsy", "R", "Susp. Epilepsy", "light_sleep", selected_channels, False, N_PERM, paired=True,
    )
    old_young_rem_channels = dash16.prepare_electrode_contrasts(
        grouped, "Older", "R", "Younger", "R", selected_channels, False, N_PERM, paired=False,
    )

    data = {
        "hep_artifact_exclude_s": getattr(hep_mod, "HEP_ARTIFACT_EXCLUDE_S", (-0.05, 0.05)),
        "panels": {
            "A": {
                "result": _slim_result(rem_n3), "title": "REM − Deep Sleep (N3)",
                "n_label": f"N = {rem_n3['matrix_a'].shape[0]}",
                "channel_results": rem_n3_channels, "label_a": "Susp. Epilepsy", "label_b": "Susp. Epilepsy",
            },
            "B": {
                "result": _slim_result(rem_light), "title": "REM − Light Sleep",
                "n_label": f"N = {rem_light['matrix_a'].shape[0]}",
                "channel_results": rem_light_channels, "label_a": "Susp. Epilepsy", "label_b": "Susp. Epilepsy",
            },
            "C": {
                "result": _slim_result(old_young_rem), "title": "Older − Younger, REM",
                "n_label": f"N-older = {old_young_rem['matrix_a'].shape[0]}, "
                            f"N-younger = {old_young_rem['matrix_b'].shape[0]}",
                "channel_results": old_young_rem_channels, "label_a": "Older", "label_b": "Younger",
            },
        },
    }
    with open(DATA_PKL, "wb") as f:
        pickle.dump(data, f)
    print(f"wrote {DATA_PKL}")
    for key, p in data["panels"].items():
        n_sig = sum(1 for r in p["channel_results"].values() if r.get("cluster_significant"))
        print(f"  {key}: {p['title']}  N={p['n_label']}  "
              f"p={p['result']['contrast_p']:.4f}  sig_electrodes={n_sig}/{len(p['channel_results'])}")


if __name__ == "__main__":
    main()
