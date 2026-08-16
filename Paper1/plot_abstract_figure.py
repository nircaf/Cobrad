"""
Fast step: render the A/B/C conference-abstract figure from the cached data
in Paper1/abstract_data.pkl (produced by compute_abstract_data.py). Nothing
here recomputes statistics -- only plotting/layout/font-size choices live in
this file, so it reruns in seconds and is the file to edit for any purely
cosmetic change (font size, spacing, colors, panel arrangement).

Layout: A (REM-Deep Sleep) and B (REM-Light Sleep) stacked on the left,
each row split into trace | topomap; C (Older-vs-Younger REM) spans both
rows on the right, trace | topomap.

Run: source venv/bin/activate && python3 Paper1/plot_abstract_figure.py
"""
import os
import sys
import types
import pickle
import importlib.util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
OUT_DIR = os.path.join(REPO, "Paper1")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
DATA_PKL = os.path.join(OUT_DIR, "abstract_data.pkl")

COLOR_NS = "#4C72B0"      # non-significant trace segments
COLOR_SIG = "#C0392B"     # significant trace segments
COLOR_QRS = "#BBBBBB"     # QRS/CFA exclusion shading
COLOR_SIG_BG = "#F5D000"  # background highlight behind significant stretches

# ---- cosmetic knobs -- edit these freely, then just rerun this file ----
# One shared base size so every text element (trace ticks/labels, topomap
# ticks/labels, colorbar) reads as the same size across the whole figure.
FONT_BASE = 17
FONT_TITLE = 19          # panel titles (A/B/C trace + cluster-p line)
FONT_NLABEL = FONT_BASE
FONT_PANEL_LETTER = 24
FONT_AXIS_LABEL = FONT_BASE
FONT_TICK = FONT_BASE
FONT_TOPO_TITLE = FONT_TITLE
FONT_LEGEND = FONT_BASE
FIGSIZE = (15.5, 8.6)


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


def load_dash16():
    """Only needed for format_p() and make_red_significance_topomap() --
    both pure functions of already-computed data, no patient loading here."""
    mock_st = _make_mock_streamlit()
    sys.modules["streamlit"] = mock_st
    for submod in ["streamlit.components", "streamlit.components.v1"]:
        sys.modules.setdefault(submod, types.ModuleType(submod))
    dash_path = os.path.join(REPO, "16_diagnosis_sleep_stage_comparison_dashboard.py")
    spec = importlib.util.spec_from_file_location("dash16_module", dash_path)
    dash16 = importlib.util.module_from_spec(spec)
    sys.modules["dash16_module"] = dash16
    spec.loader.exec_module(dash16)
    return dash16


def delta_and_sem(result):
    a, b = result["matrix_a"], result["matrix_b"]
    if result["paired"]:
        delta = a - b
        return np.nanmean(delta, axis=0), np.nanstd(delta, axis=0) / np.sqrt(delta.shape[0])
    mean_a, mean_b = np.nanmean(a, axis=0), np.nanmean(b, axis=0)
    sem_a = np.nanstd(a, axis=0) / np.sqrt(a.shape[0])
    sem_b = np.nanstd(b, axis=0) / np.sqrt(b.shape[0])
    return mean_a - mean_b, np.sqrt(sem_a ** 2 + sem_b ** 2)


def plot_trace_panel(ax, times, delta_mean, delta_sem, clusters, qrs_window,
                      title, n_label, y_label, panel_label, n_label_ha="left"):
    t0, t1 = qrs_window
    ax.axvspan(t0, t1, color=COLOR_QRS, alpha=0.5, zorder=0, linewidth=0)
    ax.axhline(0, color="#999999", lw=0.8, zorder=1)
    ax.axvline(0, color="black", lw=0.9, linestyle="--", zorder=1)

    ax.fill_between(times, delta_mean - delta_sem, delta_mean + delta_sem,
                     color=COLOR_NS, alpha=0.20, zorder=2, linewidth=0)
    # Explicit boundary lines on top of the fill -- the SEM band in panels A/C
    # is only a sliver relative to their larger y-range, so the fill alone
    # can be invisible at those scales; a thin outline stays visible regardless.
    ax.plot(times, delta_mean - delta_sem, color=COLOR_NS, lw=0.6, alpha=0.55, zorder=2)
    ax.plot(times, delta_mean + delta_sem, color=COLOR_NS, lw=0.6, alpha=0.55, zorder=2)
    ax.plot(times, delta_mean, color=COLOR_NS, lw=1.6, zorder=3)

    sig_mask = np.zeros_like(times, dtype=bool)
    for c in clusters:
        if c["significant"]:
            sig_mask |= (times >= c["start"]) & (times <= c["end"])
    if sig_mask.any():
        idx = np.flatnonzero(sig_mask)
        runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for run in runs:
            if len(run) < 2:
                continue
            # Yellow band behind the red segment -- makes the significant
            # stretch pop against the QRS grey / SEM blue at a glance.
            ax.axvspan(times[run[0]], times[run[-1]], color=COLOR_SIG_BG,
                       alpha=0.25, zorder=1, linewidth=0)
            ax.plot(times[run], delta_mean[run], color=COLOR_SIG, lw=2.4,
                    zorder=4, solid_capstyle="round")

    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=9)
    n_label_x = 0.98 if n_label_ha == "right" else 0.02
    ax.text(n_label_x, 0.97, n_label, transform=ax.transAxes, fontsize=FONT_NLABEL,
            va="top", ha=n_label_ha, color="#333333")
    if panel_label:
        ax.text(-0.16, 1.2, panel_label, transform=ax.transAxes, fontsize=FONT_PANEL_LETTER,
                fontweight="bold", va="top", ha="left")
    ax.set_xlabel("Time from R-peak (s)", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel(y_label, fontsize=FONT_AXIS_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xlim(times.min(), times.max())


DISPLAY_P_THRESHOLD = 0.05
MONTAGE_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def _montage_info_and_values(channel_results):
    """FDR q-value per electrode, restricted to the standard 10-20 montage
    -- same channel selection/aliasing dash16 uses for its own topomaps."""
    import mne

    montage = mne.channels.make_standard_montage("standard_1020")
    montage_lookup = {ch.upper(): ch for ch in montage.ch_names}
    montage_names, values = [], []
    for channel, result in channel_results.items():
        lookup_name = MONTAGE_ALIASES.get(channel.upper(), channel.upper())
        montage_name = montage_lookup.get(lookup_name)
        if montage_name is None or montage_name in montage_names:
            continue
        p_value = float(result.get("q_value", 1.0))
        is_significant = bool(result.get("fdr_significant", False) and np.isfinite(p_value))
        montage_names.append(montage_name)
        values.append(max(0.0, p_value) if is_significant else DISPLAY_P_THRESHOLD)

    info = mne.create_info(montage_names, sfreq=250.0, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore")
    valid = np.array([
        np.all(np.isfinite(ch["loc"][:3])) and np.any(ch["loc"][:3] != 0)
        for ch in info["chs"]
    ])
    info = mne.pick_info(info, np.flatnonzero(valid))
    data = np.asarray(values, dtype=float)[valid]
    return info, data


def plot_topomap_panel(ax, channel_results, show_colorbar):
    import mne

    info, data = _montage_info_and_values(channel_results)
    im, _ = mne.viz.plot_topomap(
        data, info, axes=ax, show=False, cmap="Reds_r",
        vlim=(0.0, DISPLAY_P_THRESHOLD), sensors=False,
        outlines="head", contours=0, extrapolate="head",
    )
    # mne's own sensor markers (sensors=True) render at a fixed, near-invisible
    # size -- draw our own on top instead, at the same positions.
    pos, _ = mne.viz.topomap._get_pos_outlines(info, np.arange(len(info.ch_names)), None)
    ax.scatter(pos[:, 0], pos[:, 1], s=28, facecolor="white", edgecolor="black",
               linewidth=0.9, zorder=5)
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.06)
        cbar.set_ticks(np.linspace(0.0, DISPLAY_P_THRESHOLD, 6))
        cbar.set_label("FDR corr. p-value", fontsize=FONT_TOPO_TITLE)
        cbar.ax.tick_params(labelsize=FONT_TICK)
    ax.axis("off")
    n_sig = sum(1 for r in channel_results.values() if r.get("fdr_significant"))
    pct_sig = 100.0 * n_sig / len(channel_results)
    ax.set_title(f"{pct_sig:.0f}% electrodes significant",
                 fontsize=FONT_TOPO_TITLE, pad=6)


def main():
    with open(DATA_PKL, "rb") as f:
        data = pickle.load(f)
    dash16 = load_dash16()
    panels = data["panels"]
    qrs_window = data["hep_artifact_exclude_s"]

    # Display titles override whatever compute_abstract_data.py cached --
    # abbreviate Deep Sleep / Light Sleep so DS/LS don't need N3 repeated.
    display_title = {"A": "REM − DS", "B": "REM − LS", "C": "Older − Younger, REM"}

    # 3 columns (A, B, C), 2 rows: trace on top, topomap below.
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1],
                           height_ratios=[1.0, 1.3], hspace=0.4, wspace=0.15,
                           left=0.055, right=0.98, top=0.88, bottom=0.13)

    for col, key in zip(range(3), ["A", "B", "C"]):
        p = panels[key]
        ax_trace = fig.add_subplot(gs[0, col])
        ax_topo = fig.add_subplot(gs[1, col])
        delta_mean, delta_sem = delta_and_sem(p["result"])
        p_str = dash16.format_p(p["result"]["contrast_p"])
        # Panel C's "N-older = ..., N-younger = ..." is too wide as one line
        # right-aligned -- its left edge reaches past the trace peak at t=0
        # and overlaps it. Stacking the two counts keeps it clear of the peak.
        n_label = p["n_label"].replace(", ", "\n") if key == "C" else p["n_label"]
        plot_trace_panel(
            ax_trace, p["result"]["times"], delta_mean, delta_sem, p["result"]["clusters"], qrs_window,
            f"{display_title[key]}\ncluster p = {p_str}", n_label,
            "Δ HEP amplitude (μV)" if col == 0 else "", key,
            n_label_ha="right" if key == "C" else "left",
        )
        plot_topomap_panel(ax_topo, p["channel_results"], show_colorbar=(key == "C"))

    legend_elems = [
        Line2D([0], [0], color=COLOR_NS, lw=1.6, label="Mean Δ HEP ± SEM"),
        Line2D([0], [0], color=COLOR_SIG, lw=2.4, label="Cluster-significant (p < 0.01)"),
        Patch(facecolor=COLOR_SIG_BG, alpha=0.25, label="Significant window"),
        Patch(facecolor=COLOR_QRS, alpha=0.5, label="QRS window (CFA excluded)"),
        Line2D([0], [0], color="black", lw=0.9, linestyle="--", label="R-peak (t = 0)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=5, fontsize=FONT_LEGEND,
               frameon=False, bbox_to_anchor=(0.5, 0.02))

    out_png = os.path.join(FIG_DIR, "abstract_3panel.png")
    out_pdf = os.path.join(FIG_DIR, "abstract_3panel.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
