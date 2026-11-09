"""Recompute the conference-abstract contrasts after population CFAR.

This uses the validated six-channel CFAR direction (F3/F4/C3/C4/O1/O2)
from Paper CFA/paper_stats.json.  The cached raw per-electrode matrices from
the original abstract are aligned by patient, stacked into six-channel HEPs,
and cleaned by the same signal-space projection used in the CFAR paper:

    X_clean = X - u (u.T X)

The resulting cache is consumed by plot_abstract_figure_cfar.py.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
import types

import numpy as np

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
HERE = os.path.join(REPO, "paper utils", "Paper1")
RAW_CACHE = os.path.join(HERE, "abstract_data.pkl")
OUT_CACHE = os.path.join(HERE, "abstract_data_cfar.pkl")
STATS = os.path.join(REPO, "paper utils", "Paper CFA", "paper_stats.json")
SITES = ["F3", "F4", "C3", "C4", "O1", "O2"]
N_PERM = 200


def _mock_streamlit():
    class NoOp:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return NoOp()
        def __getattr__(self, name): return NoOp()
    st = types.ModuleType("streamlit")
    st.cache_data = lambda func=None, **kwargs: func if func is not None else (lambda f: f)
    st.cache_resource = st.cache_data
    st.session_state = {}
    st.sidebar = NoOp()
    for name in ("warning", "error", "info", "write", "title", "header", "subheader",
                 "markdown", "text", "pyplot", "spinner", "set_page_config", "stop",
                 "experimental_rerun", "rerun"):
        setattr(st, name, lambda *args, **kwargs: None)
    st.empty = lambda *args, **kwargs: NoOp()
    st.progress = lambda *args, **kwargs: NoOp()
    st.columns = lambda *args, **kwargs: [NoOp(), NoOp()]
    st.tabs = lambda labels: [NoOp() for _ in labels]
    st.expander = lambda *args, **kwargs: NoOp()
    return st


def load_dash16():
    sys.modules["streamlit"] = _mock_streamlit()
    for submod in ("streamlit.components", "streamlit.components.v1"):
        sys.modules.setdefault(submod, types.ModuleType(submod))
    path = os.path.join(REPO, "16_diagnosis_sleep_stage_comparison_dashboard.py")
    spec = importlib.util.spec_from_file_location("dash16_cfar", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dash16_cfar"] = mod
    spec.loader.exec_module(mod)
    return mod


def side_map(channel_result, side):
    ids = channel_result[f"patient_ids_{side}"]
    mat = np.asarray(channel_result[f"matrix_{side}"], dtype=float)
    return {str(pid): mat[i] for i, pid in enumerate(ids)}


def aligned_six_channel(panel, side, patient_ids):
    maps = {site: side_map(panel["channel_results"][site], side) for site in SITES}
    return np.stack([[maps[site][pid] for site in SITES] for pid in patient_ids], axis=0)


def project_out(x, u):
    # x: patients x channels x time; u is unit length.
    return x - u[None, :, None] * np.einsum("c,nct->nt", u, x)[:, None, :]


def test_result(dash, a, b, times, paired):
    clusters, t_stat, threshold, p_value = dash.cluster_permutation_contrast(
        a, b, times, paired=paired, n_permutations=N_PERM
    )
    return {"matrix_a": a, "matrix_b": b, "times": times, "clusters": clusters,
            "t_stat": t_stat, "t_threshold": threshold, "contrast_p": p_value,
            "paired": paired}


def main():
    dash = load_dash16()
    with open(RAW_CACHE, "rb") as f:
        source = pickle.load(f)
    with open(STATS) as f:
        paper_stats = json.load(f)
    u = np.array([paper_stats["cleaning_demo"]["u_pop"][s] for s in SITES], dtype=float)
    u /= np.linalg.norm(u)

    output = {
        "hep_artifact_exclude_s": source["hep_artifact_exclude_s"],
        "cfar_sites": SITES,
        "cfar_u": u,
        "panels": {},
    }
    for key, panel in source["panels"].items():
        channel_maps_a = {s: side_map(panel["channel_results"][s], "a") for s in SITES}
        channel_maps_b = {s: side_map(panel["channel_results"][s], "b") for s in SITES}
        ids_a = set.intersection(*(set(m) for m in channel_maps_a.values()))
        ids_b = set.intersection(*(set(m) for m in channel_maps_b.values()))
        paired = bool(panel["result"]["paired"])
        if paired:
            ids_a = ids_b = sorted(ids_a & ids_b)
        else:
            shared = ids_a & ids_b
            ids_a, ids_b = sorted(ids_a - shared), sorted(ids_b - shared)

        raw_a = aligned_six_channel(panel, "a", ids_a)
        raw_b = aligned_six_channel(panel, "b", ids_b)
        clean_a, clean_b = project_out(raw_a, u), project_out(raw_b, u)
        times = np.asarray(panel["result"]["times"], dtype=float)
        global_result = test_result(dash, clean_a.mean(axis=1), clean_b.mean(axis=1), times, paired)

        channel_results = {}
        p_values = []
        for idx, site in enumerate(SITES):
            result = test_result(dash, clean_a[:, idx], clean_b[:, idx], times, paired)
            result.update({"patient_ids_a": ids_a, "patient_ids_b": ids_b})
            channel_results[site] = result
            p_values.append(result["contrast_p"])
        q_values = dash.benjamini_hochberg(np.asarray(p_values))
        for site, q in zip(SITES, q_values):
            result = channel_results[site]
            result["q_value"] = float(q)
            result["cluster_significant"] = any(c["significant"] for c in result["clusters"])
            result["fdr_significant"] = bool(result["cluster_significant"] and q < 0.05)

        n_label = (f"N = {len(ids_a)}" if paired else
                   f"N-older = {len(ids_a)}, N-younger = {len(ids_b)}")
        output["panels"][key] = {
            "result": global_result, "title": panel["title"], "n_label": n_label,
            "channel_results": channel_results, "label_a": panel["label_a"],
            "label_b": panel["label_b"],
        }
        n_sig = sum(r["fdr_significant"] for r in channel_results.values())
        print(f"{key}: {n_label}; cluster p={global_result['contrast_p']:.4f}; "
              f"FDR-significant={n_sig}/{len(SITES)}")

    with open(OUT_CACHE, "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {OUT_CACHE}")


if __name__ == "__main__":
    main()
