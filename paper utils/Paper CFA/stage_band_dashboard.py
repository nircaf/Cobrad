"""Dominant frequency band per sleep stage, raw vs. ECG-informed ICA vs. CFAR vs. patient-specific CFAR.

Reads stage_band_cache/*.parquet from stage_band_cleaning.py (works while the batch run is in
progress; press "Reload" for new patients). All error bars / bands are mean +- SD across patients.
Validation tabs: (1) artifact injection with known truth (stage_band_injection.py), (2) technician
hypnograms instead of YASA, incl. YASA staging accuracy per cleaning method, (3) high- vs low-CFA patients.

Run: bash "paper utils/Paper CFA/stage_band_dashboard.sh"
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.dataset as ds
import streamlit as st
from plotly.subplots import make_subplots

from sklearn.metrics import cohen_kappa_score

from stage_band_cleaning import BANDS, FREQS, STAGES

HERE = Path(__file__).resolve().parent
CACHE = HERE / "stage_band_cache"
INJ_CACHE = HERE / "stage_band_injection_cache"
STAGINGS = {"expert": "Expert (technician hypnogram)", "yasa": "YASA (automatic, raw C4)"}
CONDS = {"raw": "Raw", "ica": "ECG-informed ICA", "cfar": "CFAR", "own": "Patient-specific CFAR"}
COLORS = {"raw": "#666666", "ica": "#0072B2", "cfar": "#D55E00", "own": "#009E73"}
STAGE_COLORS = {"W": "#E69F00", "N1": "#56B4E9", "N2": "#0072B2", "N3": "#000000", "R": "#CC79A7"}
BAND_NAMES = list(BANDS)
# Stage contrasts and the band changes sleep physiology predicts (+1 increase, -1 decrease), relative power.
# Rechtschaffen & Kales 1968; AASM manual (Iber 2007); Carskadon & Dement, Principles & Practice of Sleep Medicine.
CONTRASTS = [("N1", "W"), ("N2", "N1"), ("N3", "N2"), ("R", "N2"), ("R", "W"), ("N3", "W")]
EXPECTED = {
    ("N1", "W"): {"alpha": -1, "theta": +1},               # alpha dropout, theta onset at sleep entry
    ("N2", "N1"): {"sigma": +1, "delta": +1},              # spindles (12-16 Hz) + K-complexes
    ("N3", "N2"): {"delta": +1, "sigma": -1, "beta": -1},  # slow-wave sleep: delta dominates
    ("R", "N2"): {"delta": -1, "sigma": -1, "beta": +1},   # REM: desynchronised, no spindles/slow waves
    ("R", "W"): {"alpha": -1, "theta": +1},                # REM: low alpha, sawtooth theta
    ("N3", "W"): {"delta": +1, "alpha": -1, "beta": -1},
}
EXPECTED_BAND = {"W": "alpha", "N1": "theta", "N2": "sigma", "N3": "delta", "R": "theta"}  # hallmark band per stage
EXPECTED_TEXT = """
**Expected spectra by stage** (relative power, 1-30 Hz; 6-site mastoid montage):
- **W** – most alpha (8-12 Hz, posterior O1/O2, eyes closed) and beta; least delta. Peak ≈ alpha.
- **N1** – alpha drops out, theta (4-8 Hz) rises. Peak ≈ theta / low alpha.
- **N2** – sleep spindles → **sigma (12-16 Hz) peak**, central C3/C4 strongest; delta rises (K-complexes).
- **N3** – slow waves → **delta dominant** (largest of all stages), frontal F3/F4 strongest; sigma/beta lowest.
- **R** – desynchronised like wake but low alpha; theta (sawtooth waves) and beta up, delta and sigma down vs N2.

**What good cardiac cleaning should do:** the heartbeat artifact sits at the heart rate (~1-1.5 Hz) and its harmonics,
i.e. mostly in **delta**, and is relatively largest when the EEG is small (W, N1, R). Good cleaning should therefore
(1) leave every stage's dominant band/peak unchanged, (2) change spectra mainly in delta, most in W/N1/R and least in N3
(where real slow waves dwarf the artifact), and (3) keep or slightly sharpen the stage contrasts above — not shrink them.

**Score:** for each expected item, paired Cohen's d = mean(Δ)/SD(Δ) across patients, signed so positive = expected
direction, then averaged; plus % of patient-contrasts in the expected direction. Higher = stages more physiologically
separated after that cleaning.

**Caveats:** (a) with YASA staging, stages come from *raw* C4, so raw EEG has a built-in advantage — switch the sidebar
to expert staging to remove it. (b) Projection-based methods (CFAR, patient-specific CFAR) remove one of 6
spatial dimensions, which can also remove brain signal sharing that direction; a method can lower the score by removing
neural signal, not only artifact. (c) This is plausibility, not ground truth — no recording here has an artifact-free reference.
"""

st.set_page_config(page_title="Sleep-stage dominant band × cleaning", layout="wide")


@st.cache_data(max_entries=400)
def load_chunk(files, channel, staging):
    """files: ((path, mtime), ...) -- mtime in the key so a recomputed patient file is re-read."""
    filt = (ds.field("channel") == channel) & (ds.field("staging") == staging)
    return ds.dataset([f for f, _ in files], format="parquet").to_table(filter=filt).to_pandas()


def load(channel, staging, cache=CACHE, chunk=250, what="patients"):
    """Read a cache dir in chunks behind a progress bar; finished chunks stay cached across reruns."""
    files = sorted((str(p), p.stat().st_mtime) for p in cache.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    bar = st.progress(0.0, f"Loading {what}…")
    parts = []
    for i in range(0, len(files), chunk):
        parts.append(load_chunk(tuple(files[i:i + chunk]), channel, staging))
        done = min(i + chunk, len(files))
        bar.progress(done / len(files), f"Loading {what}… {done}/{len(files)}")
    bar.empty()
    return pd.concat(parts, ignore_index=True)


def run_progress():
    """(done, total) of the batch run, from the last '[i/N]' line of stage_band_cleaning.log."""
    try:
        lines = [l for l in open(CACHE.parent / "stage_band_cleaning.log") if l.startswith("[")]
        i, n = lines[-1][1:].split("]")[0].split("/")
        return int(i), int(n)
    except (OSError, IndexError, ValueError):
        return None


def melt_rel(df):
    rel = df.melt(id_vars=["patient_id", "stage", "condition"], value_vars=[f"rel_{b}" for b in BAND_NAMES],
                  var_name="band", value_name="rel")
    rel["band"] = rel.band.str[4:]
    return rel


def stage_diffs(rel):
    """Long table of paired stage contrasts: patient, condition, band, diff, contrast."""
    wide_s = rel.pivot_table(index=["patient_id", "condition", "band"], columns="stage", values="rel", observed=True)
    diffs = [(wide_s[a] - wide_s[b]).dropna().rename("diff").reset_index().assign(contrast=f"{a} − {b}")
             for a, b in CONTRASTS if a in wide_s and b in wide_s]
    return pd.concat(diffs) if diffs else pd.DataFrame(columns=["patient_id", "condition", "band", "diff", "contrast"])


def contrast_score(diffs):
    """Per (method, expected item): paired Cohen's d signed to the expected direction, % patients in that direction."""
    rows = []
    for (a, b), exp in EXPECTED.items():
        for band, sign in exp.items():
            sub = diffs[(diffs.contrast == f"{a} − {b}") & (diffs.band == band)]
            for cond in CONDS:
                x = sub[sub.condition == cond]["diff"].to_numpy()
                if len(x) >= 3:
                    rows.append({"condition": cond, "item": f"{a}−{b} {band} {'▲' if sign > 0 else '▼'}",
                                 "signed_d": sign * x.mean() / x.std(ddof=1), "pct_expected": 100 * np.mean(sign * x > 0)})
    return pd.DataFrame(rows)


def score_summary(score):
    summ = score.groupby("condition").agg(effect=("signed_d", "mean"), agree=("pct_expected", "mean"),
                                          n_items=("item", "count")).reindex(list(CONDS)).dropna()
    summ.index = summ.index.map(CONDS)
    return summ.sort_values("effect", ascending=False)


def winner(series, lower_better=True, nd=1):
    """Names tied at the best rounded value, joined with ' = '."""
    v = series.round(nd)
    best = v.min() if lower_better else v.max()
    return " = ".join(v[v == best].index)


def err_bars(fig, df, x, y, row=1, col=1, offset=True):
    """One trace per condition, mean +- SD across patients."""
    for cond in CONDS:
        g = df[df.condition == cond].groupby(x, sort=False)[y].agg(["mean", "std", "count"]).reindex(df[x].unique())
        fig.add_trace(go.Bar(x=g.index, y=g["mean"], error_y=dict(type="data", array=g["std"]), name=CONDS[cond],
                             marker_color=COLORS[cond], legendgroup=cond, showlegend=(row, col) == (1, 1),
                             customdata=g["count"], hovertemplate="%{x}: %{y:.3f} ± %{error_y.array:.3f} (n=%{customdata})"),
                      row=row, col=col)


with st.sidebar:
    st.header("Options")
    staging = st.radio("Sleep staging", list(STAGINGS), format_func=STAGINGS.get,
                       help="Expert = technician-scored hypnogram shipped with the recording (no bias toward raw EEG).")
    channel = st.selectbox("Channel", ["mean", "F3", "F4", "C3", "C4", "O1", "O2"], help="'mean' = average PSD of the 6 sites")
    dom_def = st.radio("Dominant band definition", ["dominant_band", "peak_band"],
                       format_func={"dominant_band": "Max relative power", "peak_band": "1/f-corrected spectral peak"}.get)
    paired = st.checkbox("Only patients with all 4 cleaning methods in a stage", True)
    st.button("Reload")  # rerun picks up new cache files; already-read chunks are not re-read
    if prog := run_progress():
        st.progress(prog[0] / prog[1], f"Batch run: {prog[0]}/{prog[1]} patients processed")

df = load(channel, staging)
if df.empty:
    st.warning(f"No results yet in {CACHE} for {STAGINGS[staging]} staging. Run stage_band_cleaning.py.")
    st.stop()
df["stage"] = pd.Categorical(df.stage, STAGES, ordered=True)
if paired:
    df = df[df.groupby(["patient_id", "stage"], observed=True).condition.transform("nunique") == len(CONDS)]
if df.empty:
    st.warning("No patient has all 4 cleaning methods yet (batch run in progress). Untick the sidebar filter.")
    st.stop()
df = df.sort_values(["stage", "condition"])

st.title("Sleep-stage dominant frequency band: raw vs. ICA vs. CFAR vs. patient-specific CFAR")
n_stage = df[df.condition == "raw"].groupby("stage", observed=True).patient_id.nunique()
c = st.columns(len(STAGES) + 1)
c[0].metric("Patients", df.patient_id.nunique())
for i, s in enumerate(STAGES, 1):
    c[i].metric(f"{s} patients", int(n_stage.get(s, 0)))

(tab_meth, tab_v1, tab_v2, tab_v3, tab_grp, tab_delta, tab_psd, tab_pt) = st.tabs([
    "Stage differences & method ranking", "✔1 Injection test (ground truth)", "✔2 Expert staging",
    "✔3 Low vs high CFA patients", "Group: bands per stage", "Cleaning effect (Δ vs raw)", "PSD per stage", "Patient-specific"])

# long format of relative band powers
rel = melt_rel(df)

with tab_meth:
    st.subheader("Stage contrasts per cleaning method (paired per patient: stage A − stage B, mean ± SD)")
    st.caption("▲/▼ above a band = the change expected from sleep physiology (see 'What should the results be').")
    diffs = stage_diffs(rel)
    names = [f"{a_} − {b_}" for a_, b_ in CONTRASTS]
    fig = make_subplots(2, 3, subplot_titles=names, shared_yaxes=False, vertical_spacing=0.14)
    for k, (name, (a_, b_)) in enumerate(zip(names, CONTRASTS)):
        r, c = divmod(k, 3)
        sub = diffs[diffs.contrast == name]
        for cond in CONDS:
            g = sub[sub.condition == cond].groupby("band")["diff"].agg(["mean", "std", "count"]).reindex(BAND_NAMES)
            fig.add_trace(go.Bar(x=g.index, y=g["mean"], error_y=dict(type="data", array=g["std"]), name=CONDS[cond],
                                 marker_color=COLORS[cond], legendgroup=cond, showlegend=k == 0, customdata=g["count"],
                                 hovertemplate=f"{name} %{{x}}: %{{y:+.3f}} ± %{{error_y.array:.3f}} (n=%{{customdata}})"),
                          r + 1, c + 1)
        for band, sign in EXPECTED[(a_, b_)].items():
            fig.add_annotation(x=band, y=1.0, yref=f"y{k + 1 if k else ''} domain", text="▲" if sign > 0 else "▼",
                               showarrow=False, font=dict(size=14, color="#333"), xref=f"x{k + 1 if k else ''}")
        fig.add_hline(0, line_color="#999", line_width=1, row=r + 1, col=c + 1)
    fig.update_layout(barmode="group", height=720, legend_orientation="h", yaxis_title="Δ relative power")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Which method is more right? Agreement with expected stage physiology")
    score = contrast_score(diffs)
    if score.empty:
        st.info("Not enough paired patients yet.")
    else:
        summ = score_summary(score)
        c1, c2 = st.columns([1, 2])
        c1.dataframe(summ.rename(columns={"effect": "mean signed Cohen's d (↑ better)",
                                          "agree": "% patient-contrasts in expected direction"}).round(2),
                     use_container_width=True)
        c1.success(f"Largest physiological stage separation ({STAGINGS[staging]}): **{winner(summ.effect, False, 2)}**")
        piv = score.pivot_table(index="item", columns="condition", values="signed_d")[[c for c in CONDS if c in set(score.condition)]]
        fig = go.Figure(go.Heatmap(z=piv.to_numpy(), x=[CONDS[c] for c in piv.columns], y=piv.index, colorscale="RdBu",
                                   zmid=0, text=np.round(piv.to_numpy(), 2), texttemplate="%{text}",
                                   colorbar_title="signed d"))
        fig.update_layout(height=40 + 26 * len(piv), margin=dict(l=10, r=10, t=10, b=10))
        c2.plotly_chart(fig, use_container_width=True)

    with st.expander("What should the results be (expected physiology, and how to read the score)", expanded=True):
        st.markdown(EXPECTED_TEXT)

with tab_grp:
    st.subheader("Relative band power per stage (mean ± SD across patients)")
    fig = make_subplots(1, len(STAGES), subplot_titles=STAGES, shared_yaxes=True)
    for i, s in enumerate(STAGES, 1):
        err_bars(fig, rel[rel.stage == s], "band", "rel", 1, i)
    fig.update_layout(barmode="group", height=420, yaxis_title="Relative power (1-30 Hz)", legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dominant band: % of patients per stage")
    fig = make_subplots(1, len(CONDS), subplot_titles=list(CONDS.values()), shared_yaxes=True)
    band_colors = dict(zip(BAND_NAMES, ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00"]))
    for j, cond in enumerate(CONDS, 1):
        ct = pd.crosstab(df[df.condition == cond].stage, df[df.condition == cond][dom_def], normalize="index") * 100
        for b in BAND_NAMES:
            if b in ct:
                fig.add_trace(go.Bar(x=ct.index.astype(str), y=ct[b], name=b, marker_color=band_colors[b], legendgroup=b,
                                     showlegend=j == 1, hovertemplate=f"{b}: %{{y:.1f}}%"), 1, j)
    fig.update_layout(barmode="stack", height=380, yaxis_title="% patients", legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("1/f-corrected peak frequency per stage (mean ± SD)")
    fig = make_subplots(1, 1)
    err_bars(fig, df.assign(stage=df.stage.astype(str)), "stage", "peak_hz")
    fig.update_layout(barmode="group", height=360, yaxis_title="Peak frequency (Hz)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Summary table")
    tbl = rel.groupby(["stage", "condition", "band"], observed=True).rel.agg(["mean", "std"]).unstack("band")
    tbl.columns = [f"{b} {s}" for s, b in tbl.columns]
    st.dataframe(tbl.round(3), use_container_width=True)

with tab_delta:
    st.subheader("Change in relative band power after cleaning (cleaned − raw, paired, mean ± SD)")
    wide = rel.pivot_table(index=["patient_id", "stage", "band"], columns="condition", values="rel", observed=True).reset_index()
    fig = make_subplots(1, len(STAGES), subplot_titles=STAGES, shared_yaxes=True)
    for i, s in enumerate(STAGES, 1):
        w = wide[wide.stage == s]
        for cond in ["ica", "cfar", "own"]:
            if cond not in w:
                continue
            g = (w[cond] - w["raw"]).groupby(w.band).agg(["mean", "std", "count"]).reindex(BAND_NAMES)
            fig.add_trace(go.Bar(x=g.index, y=g["mean"], error_y=dict(type="data", array=g["std"]), name=CONDS[cond],
                                 marker_color=COLORS[cond], legendgroup=cond, showlegend=i == 1,
                                 hovertemplate="%{x}: %{y:+.4f} ± %{error_y.array:.4f}"), 1, i)
    fig.add_hline(0, line_color="#999", line_width=1)
    fig.update_layout(barmode="group", height=420, yaxis_title="Δ relative power", legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Patients whose dominant band changed vs raw")
    dom = df.pivot_table(index=["patient_id", "stage"], columns="condition", values=dom_def, aggfunc="first", observed=True)
    chg = pd.DataFrame({CONDS[c]: (dom[c] != dom["raw"]).groupby(level="stage", observed=True).mean() * 100
                        for c in ["ica", "cfar", "own"] if c in dom})
    st.dataframe(chg.round(1).rename(columns=lambda c: f"{c} (% changed)"), use_container_width=True)

    st.header("Verdict: which method cleans better and shows the expected band more prominently")
    # 1. cleaning: heartbeat-locked CFA power left after each method (benchmark_ica_cfar.py --all, same 10-min windows)
    bench = pd.read_parquet(CACHE.parent / "benchmark_all.parquet",
                            columns=["patient_id"] + [f"{p}_{c}" for p in ("pow", "hep") for c in CONDS])
    shared = bench[bench.patient_id.isin(df.patient_id)]
    bench = shared if len(shared) >= 20 else bench
    left = pd.DataFrame({CONDS[c]: {"CFA power left (% of raw, median)": 100 * (bench[f"pow_{c}"] / bench.pow_raw).median(),
                                    "synthetic HEP kept (median)": bench[f"hep_{c}"].median()} for c in CONDS if c != "raw"}).T
    best_clean = left.iloc[:, 0].idxmin()

    # 2. prominence of the stage's expected band: % patients whose spectrum peaks there (N3: most power, since the
    # 1/f-corrected peak misplaces N3's steep delta), and paired change of that band's relative power vs raw
    prom = []
    for s, band in EXPECTED_BAND.items():
        sub = df[df.stage == s]
        col = "dominant_band" if s == "N3" else "peak_band"
        hit = sub.assign(hit=sub[col] == band).pivot_table(index="patient_id", columns="condition", values="hit")
        val = sub.pivot_table(index="patient_id", columns="condition", values=f"rel_{band}")
        for c in CONDS:
            if c in hit:
                dlt = val[c] - val["raw"]
                prom.append({"stage": s, "expected band": band, "method": CONDS[c], "% patients hit": 100 * hit[c].mean(),
                             "Δ rel power vs raw": dlt.mean(), "% patients band ↑": 100 * (dlt > 0).mean()})
    prom = pd.DataFrame(prom)
    c1, c2 = st.columns(2)
    c1.markdown("**Artifact removal** (lower power left = cleaner)")
    c1.dataframe(left.round(2), use_container_width=True)
    c1.caption(f"n = {len(bench)} patients, heartbeat-locked CFA power in each patient's 10-min window.")
    c2.markdown("**Expected band per stage** (higher % hit = band more prominent)")
    c2.dataframe(prom.pivot_table(index=["stage", "expected band"], columns="method", values="% patients hit",
                                  sort=False)[list(CONDS.values())].round(1), use_container_width=True)
    if not prom.empty:
        clean_only = prom[prom.method != "Raw"]
        wins = clean_only.loc[clean_only.groupby("stage")["% patients hit"].idxmax()]
        raw_hit = prom[prom.method == "Raw"].set_index("stage")["% patients hit"]
        mean_hit = prom.groupby("method")["% patients hit"].mean()
        cleaned = mean_hit.drop("Raw").round(1)
        best_prom = " = ".join(cleaned[cleaned == cleaned.max()].index)
        if mean_hit["Raw"].round(1) >= cleaned.max():
            best_prom += f" (but no cleaning beats raw, {mean_hit['Raw']:.1f}%: cleaning does not make the expected band more prominent)"
        lines = [f"- **{r.stage}** (expect {r['expected band']}): best = **{r.method}**, {r['% patients hit']:.1f}% of patients "
                 f"vs {raw_hit[r.stage]:.1f}% raw; that band's share {r['Δ rel power vs raw']:+.3f} "
                 f"(↑ in {r['% patients band ↑']:.0f}% of patients)." for _, r in wins.iterrows()]
        st.markdown(
            f"**Cleans the signal best: {best_clean}** — leaves {left.loc[best_clean].iloc[0]:.1f}% of the raw CFA power "
            f"(" + ", ".join(f"{m} {v:.1f}%" for m, v in left.iloc[:, 0].items()) + ").\n\n"
            f"**Most prominent expected band across stages: {best_prom}** — mean hit rate {cleaned.max():.1f}% "
            f"(" + ", ".join(f"{m} {v:.1f}%" for m, v in mean_hit.items()) + ").\n\nPer stage:\n" + "\n".join(lines) +
            "\n\n*Reading it:* the CFA numbers are the direct cleaning measure. Hit-rate differences of a few % between "
            "methods are within patient-to-patient noise until n is large; stages were scored on raw EEG, which favours raw. "
            "HEP kept < 1 means the projection also removes part of a brain signal aligned with the cardiac pattern.")

with tab_psd:
    st.subheader("Mean PSD ± SD across patients (log10 µV²/Hz)")
    fig = make_subplots(1, len(STAGES), subplot_titles=STAGES, shared_yaxes=True)
    for i, s in enumerate(STAGES, 1):
        for cond in CONDS:
            sub = df[(df.stage == s) & (df.condition == cond)]
            if sub.empty:
                continue
            P = np.log10(np.stack(sub.psd.to_numpy()))
            m, sd = P.mean(0), P.std(0)
            fig.add_trace(go.Scatter(x=np.r_[FREQS, FREQS[::-1]], y=np.r_[m + sd, (m - sd)[::-1]], fill="toself",
                                     fillcolor=COLORS[cond], opacity=0.15, line_width=0, hoverinfo="skip",
                                     showlegend=False, legendgroup=cond), 1, i)
            fig.add_trace(go.Scatter(x=FREQS, y=m, name=CONDS[cond], line=dict(color=COLORS[cond], width=2),
                                     legendgroup=cond, showlegend=i == 1, hovertemplate="%{x:.2f} Hz: %{y:.2f}"), 1, i)
    for lo, _ in list(BANDS.values())[1:]:
        fig.add_vline(lo, line_dash="dot", line_color="#bbb", line_width=1)
    fig.update_xaxes(range=[1, 30], title="Hz")
    fig.update_layout(height=420, yaxis_title="log10 PSD", legend_orientation="h", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab_pt:
    pid = st.selectbox("Patient", sorted(df.patient_id.unique()))
    p = df[df.patient_id == pid]
    st.subheader(f"{pid}: dominant band per stage")
    st.dataframe(p.pivot_table(index="stage", columns="condition", values=[dom_def, "peak_hz", "n_epochs"],
                               aggfunc="first", observed=True), use_container_width=True)
    pr = rel[rel.patient_id == pid]
    fig = make_subplots(1, len(STAGES), subplot_titles=STAGES, shared_yaxes=True)
    for i, s in enumerate(STAGES, 1):
        for cond in CONDS:
            q = pr[(pr.stage == s) & (pr.condition == cond)]
            fig.add_trace(go.Bar(x=q.band, y=q.rel, name=CONDS[cond], marker_color=COLORS[cond], legendgroup=cond,
                                 showlegend=i == 1), 1, i)
    fig.update_layout(barmode="group", height=380, yaxis_title="Relative power", legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)
    fig = make_subplots(1, len(STAGES), subplot_titles=STAGES, shared_yaxes=True)
    for i, s in enumerate(STAGES, 1):
        for cond in CONDS:
            q = p[(p.stage == s) & (p.condition == cond)]
            if len(q):
                fig.add_trace(go.Scatter(x=FREQS, y=np.log10(q.psd.iloc[0]), name=CONDS[cond], legendgroup=cond,
                                         line=dict(color=COLORS[cond], width=2), showlegend=i == 1), 1, i)
    fig.update_xaxes(range=[1, 30], title="Hz")
    fig.update_layout(height=380, yaxis_title="log10 PSD", legend_orientation="h", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- validation 1: artifact injection, known truth
with tab_v1:
    st.subheader("Injection test: a real high-CFA heartbeat artifact added to a low-CFA night; truth = the original night")
    st.caption("stage_band_injection.py. Raw = contaminated and uncleaned (shows how much the artifact distorts the bands). "
               "Error = 10·log10(band power after method / true band power), per pair and stage.")
    inj = load(channel, staging, INJ_CACHE, what="injection pairs")
    if inj.empty:
        st.info(f"No injection results yet for {STAGINGS[staging]} staging. Run "
                "`venv/bin/python \"paper utils/Paper CFA/stage_band_injection.py\" --pairs 150`.")
    else:
        key = ["patient_id", "donor_id", "stage"]
        truth = inj[inj.condition == "truth"].set_index(key)
        err = []
        for cond in CONDS:
            m = inj[inj.condition == cond].set_index(key)
            idx = m.index.intersection(truth.index)
            for b in BAND_NAMES:
                err.append(pd.DataFrame({"condition": cond, "band": b, "stage": idx.get_level_values("stage"),
                                         "err_db": 10 * np.log10(m.loc[idx, f"abs_{b}"] / truth.loc[idx, f"abs_{b}"]).to_numpy(),
                                         "err_rel_pp": 100 * (m.loc[idx, f"rel_{b}"] - truth.loc[idx, f"rel_{b}"]).to_numpy()}))
            match = (m.loc[idx, dom_def] == truth.loc[idx, dom_def]).groupby(level="stage").mean() * 100
            err.append(pd.DataFrame({"condition": cond, "band": "_match", "stage": match.index, "match": match.to_numpy()}))
        err = pd.concat(err, ignore_index=True)
        e = err[err.band != "_match"]
        n_pairs = inj.groupby(["patient_id", "donor_id"]).ngroups
        st.markdown(f"**{n_pairs} pairs**; injected artifact = {inj.groupby(['patient_id', 'donor_id']).artifact_pct_var.first().median():.1f}% "
                    f"of the recipient's EEG variance (median).")
        fig = make_subplots(1, len(STAGES), subplot_titles=STAGES, shared_yaxes=True)
        for i, s in enumerate(STAGES, 1):
            for cond in CONDS:
                g = e[(e.stage == s) & (e.condition == cond)].groupby("band").err_db.agg(["mean", "std"]).reindex(BAND_NAMES)
                fig.add_trace(go.Bar(x=g.index, y=g["mean"], error_y=dict(type="data", array=g["std"]), name=CONDS[cond],
                                     marker_color=COLORS[cond], legendgroup=cond, showlegend=i == 1,
                                     hovertemplate="%{x}: %{y:+.2f} ± %{error_y.array:.2f} dB"), 1, i)
        fig.add_hline(0, line_color="#999", line_width=1)
        fig.update_layout(barmode="group", height=420, yaxis_title="Band-power error vs truth (dB; 0 = perfect)",
                          legend_orientation="h")
        st.plotly_chart(fig, use_container_width=True)

        tab = e.assign(abs_db=e.err_db.abs(), abs_pp=e.err_rel_pp.abs()).groupby("condition").agg(
            abs_db=("abs_db", "mean"), abs_pp=("abs_pp", "mean")).reindex(list(CONDS))
        tab["match"] = err[err.band == "_match"].groupby("condition").match.mean().reindex(list(CONDS))
        tab.index = tab.index.map(CONDS)
        tab.columns = ["mean |band-power error| (dB)", "mean |relative-power error| (pp)",
                       f"% stages with the true {'dominant' if dom_def == 'dominant_band' else 'peak'} band"]
        per_stage = e.assign(a=e.err_db.abs()).pivot_table(index="stage", columns="condition", values="a").reindex(STAGES)
        per_stage = per_stage[[c for c in CONDS if c in per_stage]].rename(columns=CONDS)
        c1, c2 = st.columns(2)
        c1.markdown("**Overall (lower error = more accurate)**")
        c1.dataframe(tab.round(3), use_container_width=True)
        c2.markdown("**Mean |error| (dB) per stage**")
        c2.dataframe(per_stage.round(3), use_container_width=True)
        cleaned = tab.drop("Raw")
        st.success(f"**Most accurate band powers: {winner(cleaned.iloc[:, 0], True, 3)}** "
                   f"({cleaned.iloc[:, 0].min():.3f} dB mean error vs {tab.iloc[0, 0]:.3f} dB if left uncleaned). "
                   f"Best at keeping the true band: {winner(cleaned.iloc[:, 2], False, 1)} ({cleaned.iloc[:, 2].max():.1f}%).")
        st.caption("Negative bars = power removed (over-cleaning, incl. brain signal); positive = artifact left. "
                   "Caveat: one fixed template per donor; the recipient's own small CFA remains in 'truth'.")

# ---------------------------------------------------------------- validation 2: technician hypnograms
@st.cache_data(max_entries=50)
def load_epochs(files):
    parts = [pd.read_parquet(f) for f, _ in files]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


with tab_v2:
    st.subheader("A. Does cleaning make automatic staging agree better with the technician?")
    st.caption("YASA re-run on each method's cleaned C4 (all 1-40 Hz), compared epoch by epoch with the expert hypnogram. "
               "Higher Cohen's kappa = the cleaned EEG carries the stage information more faithfully.")
    ep = load_epochs(tuple(sorted((str(p), p.stat().st_mtime) for p in (CACHE / "epochs").glob("*.parquet"))))
    if ep.empty:
        st.info("No expert-hypnogram epochs yet.")
    else:
        ep = ep[ep.expert != ""]
        cols = [f"yasa_{c}" for c in CONDS if f"yasa_{c}" in ep]
        kap = ep.groupby("patient_id").apply(
            lambda g: pd.Series({c: cohen_kappa_score(g.expert, g[c]) for c in cols}), include_groups=False)
        fig = go.Figure()
        for c in cols:
            cond = c[5:]
            fig.add_trace(go.Bar(x=[CONDS[cond]], y=[kap[c].mean()], error_y=dict(type="data", array=[kap[c].std()]),
                                 marker_color=COLORS[cond], name=CONDS[cond],
                                 hovertemplate="kappa %{y:.3f} ± %{error_y.array:.3f}"))
        fig.update_layout(height=340, yaxis_title="Cohen's kappa vs expert (mean ± SD across patients)", showlegend=False)
        c1, c2 = st.columns([1, 1])
        c1.plotly_chart(fig, use_container_width=True)
        recall = pd.DataFrame({CONDS[c[5:]]: (ep[c] == ep.expert).groupby(ep.expert).mean() * 100
                               for c in cols}).reindex(STAGES)
        summ_k = pd.DataFrame({"kappa mean": kap.mean(), "kappa SD": kap.std(),
                               "Δ kappa vs raw (mean)": kap.sub(kap["yasa_raw"], axis=0).mean(),
                               "% patients better than raw": (kap.gt(kap["yasa_raw"], axis=0)).mean() * 100})
        summ_k.index = summ_k.index.map(lambda c: CONDS[c[5:]])
        c2.markdown(f"**{len(kap)} patients, {len(ep):,} scored epochs**")
        c2.dataframe(summ_k.round(3), use_container_width=True)
        c2.markdown("**% epochs staged correctly, per expert stage**")
        c2.dataframe(recall.round(1), use_container_width=True)
        orig = ep.groupby("patient_id").apply(lambda g: cohen_kappa_score(g.expert, g.yasa_orig), include_groups=False)
        st.success(f"**Best staging agreement with the technician: {winner(summ_k['kappa mean'], False, 3)}** "
                   f"(kappa {summ_k['kappa mean'].max():.3f}; raw {summ_k.loc['Raw', 'kappa mean']:.3f}). "
                   f"Reference: YASA on unfiltered raw C4 = {orig.mean():.3f}.")

    st.subheader("B. Stage-contrast score with expert vs YASA staging")
    st.caption("Same score as the first tab (signed paired Cohen's d of the expected stage changes). "
               "With expert staging raw EEG has no built-in advantage.")
    other = "yasa" if staging == "expert" else "expert"
    df_o = load(channel, other)
    tabs_b = {}
    for name, d in {staging: df, other: df_o}.items():
        if len(d):
            if paired:
                d = d[d.groupby(["patient_id", "stage"]).condition.transform("nunique") == len(CONDS)]
            sc = contrast_score(stage_diffs(melt_rel(d.assign(stage=pd.Categorical(d.stage, STAGES, ordered=True)))))
            if len(sc):
                tabs_b[STAGINGS[name]] = score_summary(sc).effect
    if tabs_b:
        tb = pd.DataFrame(tabs_b)
        st.dataframe(tb.round(3), use_container_width=True)
        if STAGINGS["expert"] in tb:
            st.success(f"**With expert staging, largest physiological separation: {winner(tb[STAGINGS['expert']], False, 2)}**")

# ---------------------------------------------------------------- validation 3: high- vs low-CFA patients
@st.cache_data
def cfa_strength():
    c = pd.read_parquet(HERE / "cfa_combined.parquet", columns=["patient_id", "cfa_r2_excl_qrs"])
    d = pd.read_parquet(HERE / "demographics_combined.parquet", columns=["patient_id", "age", "sex"])
    return c.groupby("patient_id").cfa_r2_excl_qrs.mean().rename("cfa_r2").reset_index().merge(d, on="patient_id")


with tab_v3:
    st.subheader("Do high-CFA patients still differ from low-CFA patients after cleaning?")
    st.caption("Patients split into CFA tertiles (paper's per-patient CFA R², cfa_combined.parquet). For each stage, band and "
               "method: relative power ~ high-CFA + age + sex (OLS), comparing top vs bottom tertile. The high-CFA "
               "coefficient is the band-power gap the artifact causes; an accurate method shrinks it toward 0.")
    meta = cfa_strength()
    meta = meta[meta.patient_id.isin(df.patient_id)].dropna(subset=["age"])
    lo_t, hi_t = meta.cfa_r2.quantile([1 / 3, 2 / 3])
    meta = meta[(meta.cfa_r2 <= lo_t) | (meta.cfa_r2 >= hi_t)].assign(high=lambda m: (m.cfa_r2 >= hi_t).astype(float),
                                                                       male=lambda m: (m.sex == "Male").astype(float))
    r3 = rel.merge(meta[["patient_id", "high", "age", "male"]], on="patient_id")
    out = []
    for (s, b, cond), g in r3.groupby(["stage", "band", "condition"], observed=True):
        if g.high.nunique() < 2 or len(g) < 10:
            continue
        X = np.column_stack([np.ones(len(g)), g.high, g.age, g.male])
        beta, res, *_ = np.linalg.lstsq(X, g.rel.to_numpy(), rcond=None)
        sigma2 = np.sum((g.rel.to_numpy() - X @ beta) ** 2) / (len(g) - X.shape[1])
        se = np.sqrt(sigma2 * np.linalg.pinv(X.T @ X)[1, 1])
        out.append({"stage": s, "band": b, "condition": cond, "beta_pp": 100 * beta[1], "t": beta[1] / se})
    out = pd.DataFrame(out)
    if out.empty:
        st.info("Not enough patients with CFA strength and demographics yet.")
    else:
        st.markdown(f"**{int(meta.high.sum())} high-CFA vs {int((1 - meta.high).sum())} low-CFA patients** "
                    f"(R² ≥ {hi_t:.3f} vs ≤ {lo_t:.3f}).")
        out["item"] = out.stage.astype(str) + " " + out.band
        piv = out.pivot_table(index="item", columns="condition", values="beta_pp", sort=False)[[c for c in CONDS if c in set(out.condition)]]
        fig = go.Figure(go.Heatmap(z=piv.to_numpy(), x=[CONDS[c] for c in piv.columns], y=piv.index, colorscale="RdBu",
                                   zmid=0, text=np.round(piv.to_numpy(), 2), texttemplate="%{text}",
                                   colorbar_title="gap (pp)"))
        fig.update_layout(height=60 + 22 * len(piv), margin=dict(l=10, r=10, t=10, b=10))
        c1, c2 = st.columns([2, 1])
        c1.markdown("**High − low CFA gap in relative power (percentage points), adjusted for age and sex**")
        c1.plotly_chart(fig, use_container_width=True)
        summ3 = out.assign(abs_b=out.beta_pp.abs(), abs_t=out.t.abs()).groupby("condition").agg(
            gap=("abs_b", "mean"), t=("abs_t", "mean"), sig=("abs_t", lambda v: 100 * np.mean(v > 1.96))).reindex(list(CONDS))
        summ3.index = summ3.index.map(CONDS)
        summ3.columns = ["mean |gap| (pp, ↓ better)", "mean |t|", "% stage-bands with p<0.05 gap"]
        c2.dataframe(summ3.round(3), use_container_width=True)
        cl = summ3.drop("Raw").iloc[:, 0]
        c2.success(f"**Smallest artifact-related gap: {winner(cl, True, 2)}** ({cl.min():.2f} pp vs raw "
                   f"{summ3.iloc[0, 0]:.2f} pp).")
        c2.caption("Caveat: high-CFA patients also differ physiologically (heavier, more cardiac disease), so a residual "
                   "gap is not all artifact. All methods are compared on the same patients, so the ranking is what counts.")
