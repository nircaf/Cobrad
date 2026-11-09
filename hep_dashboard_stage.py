"""Sleep-stage deltas page: REM, DEEP and LIGHT compared inside the same patient.

Every contrast is paired within patient, so one participant contributes only the
stage pairs that were actually recorded. The page covers the three cleaning
methods, per-electrode and overall effects, scalp maps, demographics, method
agreement, the statistics behind every table, and CSV export.
"""
import hep_dashboard_common as _common

globals().update({_name: _value for _name, _value in vars(_common).items()
                  if not _name.startswith("__")})

WINDOW_PRESETS = {
    "0-400 ms after R-peak (default)": (0, 8),
    "0-600 ms (whole window)": (0, 12),
    "50-350 ms (systolic window)": (1, 7),
}
AGE_EDGES = [0, 25, 45, 60, 75, 200]
AGE_LABELS = ["<25", "25-44", "45-59", "60-74", "75+"]
MIN_GROUP = 5

STAGE_METADATA = {
    "The_Human_Sleep_Project":
        ROOT / "EDF_Format" / "The_Human_Sleep_Project" / "psg_metadata.csv",
    "Young_The_Human_Sleep_Project":
        ROOT / "EDF_Format" / "Young_The_Human_Sleep_Project" / "psg_metadata.csv",
}
EDF_CLINICAL = ROOT / "EDF_Format" / "EDF" / "COBRAD_clinical_24022025.xlsx"
EDF_CLASSES = [("Class_bvftd", "bvFTD"), ("Class_LBD", "LBD"),
               ("Class_NA", "Unclassified")]


def _meta_key(value) -> str:
    """Normalise an id so the stage cache and the metadata files agree."""
    return re.sub(r"[^A-Z0-9]+", "", str(value).strip().upper()).lstrip("0")


def _clean_sex(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text in {"1", "1.0", "m", "male"}:
        return "Male"
    if text in {"2", "2.0", "f", "female"}:
        return "Female"
    return str(value).strip()


@st.cache_data(show_spinner=False)
def stage_demographics() -> pd.DataFrame:
    """Age, sex, diagnosis and BMI (when it exists) for staged patients.

    HSP cohorts carry age and sex in ``psg_metadata.csv`` (matched on the BIDS
    folder). COBRAD / EDF carries age, sex and the bvFTD/LBD class flags in the
    clinical sheet (matched on the record id). None of these sources records
    BMI, so the BMI column stays empty and the page says so instead of guessing.
    """
    records = []
    for dataset, path in STAGE_METADATA.items():
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        bmi_col = next((c for c in frame.columns if "bmi" in str(c).lower()), None)
        for _, row in frame.iterrows():
            if pd.isna(row.get("BIDSFolder")):
                continue
            records.append({
                "dataset": dataset,
                "key": _meta_key(row.get("BIDSFolder")),
                "age": pd.to_numeric(row.get("AgeAtVisit"), errors="coerce"),
                "sex": _clean_sex(row.get("SexDSC")),
                "bmi": pd.to_numeric(row.get(bmi_col), errors="coerce") if bmi_col else np.nan,
                "diagnosis": "",
                "source": path.name,
            })
    if EDF_CLINICAL.exists():
        clinical = pd.read_excel(EDF_CLINICAL, sheet_name="clinical")
        bmi_col = next((c for c in clinical.columns if "bmi" in str(c).lower()), None)
        for _, row in clinical.iterrows():
            labels = [name for column, name in EDF_CLASSES
                      if str(row.get(column)).strip() in {"1", "1.0"}]
            records.append({
                "dataset": "EDF",
                "key": _meta_key(row.get("record_id")),
                "age": pd.to_numeric(row.get("age"), errors="coerce"),
                "sex": _clean_sex(row.get("sex")),
                "bmi": pd.to_numeric(row.get(bmi_col), errors="coerce") if bmi_col else np.nan,
                "diagnosis": ", ".join(labels),
                "source": "COBRAD_clinical_24022025.xlsx",
            })
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["patient_key", "age", "sex", "bmi", "diagnosis"])
    frame["patient_key"] = frame["dataset"] + "|" + frame["key"]
    return frame.drop_duplicates("patient_key")


def _patient_key(patient: str) -> str:
    dataset, _, bare = str(patient).partition(":")
    return f"{dataset}|{_meta_key(bare)}"


@st.cache_data(show_spinner="Loading staged HEP caches...")
def stage_table(cohorts: tuple[str, ...], methods: tuple[str, ...]) -> pd.DataFrame:
    table = stage_delta_table(cohorts)
    if table.empty:
        return table
    return table[table.method.isin(methods)].reset_index(drop=True)


def _window_site_means(delta: np.ndarray, lo: int, hi: int) -> dict[str, float]:
    values = {}
    for index, site in enumerate(SITES):
        block = delta[index * 12 + lo: index * 12 + hi]
        values[site] = float(np.nanmean(block)) if np.isfinite(block).any() else np.nan
    return values


@st.cache_data(show_spinner="Building patient-level sleep-stage deltas...")
def delta_summary(cohorts: tuple[str, ...], methods: tuple[str, ...],
                  lo: int, hi: int) -> pd.DataFrame:
    """One row per patient, method and contrast, with per-site window means."""
    table = stage_table(cohorts, methods)
    if table.empty:
        return pd.DataFrame()
    rows = []
    for record in table.itertuples(index=False):
        sites = _window_site_means(np.asarray(record.delta, dtype=float), lo, hi)
        usable = [value for value in sites.values() if np.isfinite(value)]
        row = {"patient": record.cobrad_id, "cohort": record.dataset,
               "method": record.method, "contrast": record.contrast}
        row.update(sites)
        row["overall"] = float(np.mean(usable)) if usable else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    demo = stage_demographics()
    out["patient_key"] = out["patient"].map(_patient_key)
    out = out.merge(demo[["patient_key", "age", "sex", "bmi", "diagnosis"]],
                    on="patient_key", how="left").drop(columns="patient_key")
    out["age_bin"] = out["age"].map(_age_bin)
    out["sex"] = out["sex"].fillna("")
    return out


def _age_bin(age) -> str:
    if pd.isna(age):
        return "unknown"
    for low, high, label in zip(AGE_EDGES[:-1], AGE_EDGES[1:], AGE_LABELS):
        if low <= float(age) < high:
            return label
    return AGE_LABELS[-1]


def _effects(values) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return {}
    deviation = float(np.nanstd(values, ddof=1))
    return {
        "N": int(len(values)),
        "Mean": float(values.mean()),
        "Median": float(np.median(values)),
        "SD": deviation,
        "d_z": float(values.mean() / deviation) if deviation > 0 else np.nan,
        "% > 0": float((values > 0).mean()),
    }


def _signed_rank_p(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 5 or np.allclose(values, 0):
        return np.nan
    try:
        return float(stats.wilcoxon(values, zero_method="wilcox",
                                    alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def _add_fdr(frame: pd.DataFrame, p_col: str = "p") -> pd.DataFrame:
    frame = frame.copy()
    if not len(frame):
        return frame
    frame["q"] = np.nan
    mask = frame[p_col].notna().to_numpy()
    if mask.any():
        frame.loc[mask, "q"] = bh(frame.loc[mask, p_col].tolist())
    frame["Significant"] = frame["q"] < 0.05
    return frame


@st.cache_data(show_spinner=False)
def overall_stats(cohorts: tuple[str, ...], methods: tuple[str, ...],
                  lo: int, hi: int) -> pd.DataFrame:
    """Is the patient-level window mean different from zero, per cell?"""
    summary = delta_summary(cohorts, methods, lo, hi)
    rows = []
    for (method, contrast), group in summary.groupby(["method", "contrast"], sort=False):
        effects = _effects(group["overall"])
        if not effects:
            continue
        rows.append({"method": method, "contrast": contrast, **effects,
                     "p": _signed_rank_p(group["overall"])})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["method", "contrast"]).reset_index(drop=True)
        out = _add_fdr(out)
    return out


@st.cache_data(show_spinner="Testing every electrode x window delta against zero...")
def feature_stats(cohorts: tuple[str, ...], methods: tuple[str, ...]) -> pd.DataFrame:
    table = stage_table(cohorts, methods)
    names = feature_names()
    frames = []
    for method in methods:
        for contrast in STAGE_CONTRAST_NAMES:
            frame = stage_delta_stats(table, method, contrast, names)
            if len(frame):
                frames.append(frame.assign(method=method, contrast=contrast))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(show_spinner="Comparing cleaning methods inside patient...")
def method_stats(cohorts: tuple[str, ...], methods: tuple[str, ...]) -> pd.DataFrame:
    table = stage_table(cohorts, methods)
    names = feature_names()
    frames = []
    for contrast in STAGE_CONTRAST_NAMES:
        frame = stage_delta_method_comparison(table, contrast, names)
        if len(frame):
            frames.append(frame.assign(contrast=contrast))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def method_site_differences(summary: pd.DataFrame, contrast: str,
                            first: str, second: str) -> tuple[dict[str, float], int]:
    """Mean paired difference of the site window means between two methods."""
    a = summary[(summary.method == first) & (summary.contrast == contrast)].set_index("patient")
    b = summary[(summary.method == second) & (summary.contrast == contrast)].set_index("patient")
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return {}, 0
    values = {}
    for site in SITES:
        difference = (a.loc[common, site] - b.loc[common, site]).dropna()
        values[site] = float(difference.mean()) if len(difference) else np.nan
    return values, len(common)


def _group_test(groups: dict[str, np.ndarray]) -> tuple[str, float]:
    usable = {name: values[np.isfinite(values)]
              for name, values in groups.items() if np.isfinite(values).sum() >= MIN_GROUP}
    if len(usable) < 2:
        return "", np.nan
    if len(usable) == 2:
        first, second = list(usable.values())
        return "Mann-Whitney U", float(stats.mannwhitneyu(first, second, alternative="two-sided").pvalue)
    return "Kruskal-Wallis", float(stats.kruskal(*usable.values()).pvalue)


def subgroup_table(summary: pd.DataFrame, method: str, contrast: str,
                   column: str, order: list[str] | None = None) -> tuple[pd.DataFrame, str, float]:
    subset = summary[(summary.method == method) & (summary.contrast == contrast)]
    groups = {}
    rows = []
    labels = order or sorted(str(value) for value in subset[column].dropna().unique())
    for label in labels:
        values = subset.loc[subset[column].astype(str) == label, "overall"].to_numpy()
        values = values[np.isfinite(values)]
        groups[label] = values
        rows.append({"Group": label, "N patients": int(len(values)),
                     "Median Δ (µV)": float(np.median(values)) if len(values) else np.nan,
                     "Mean Δ (µV)": float(values.mean()) if len(values) else np.nan,
                     "dB": float(values.mean() / np.std(values, ddof=1))
                     if len(values) > 2 and np.std(values, ddof=1) > 0 else np.nan,
                     "% > 0": float((values > 0).mean()) if len(values) else np.nan})
    test_name, p_value = _group_test(groups)
    return pd.DataFrame(rows), test_name, p_value


def _icc21(first: np.ndarray, second: np.ndarray) -> float:
    """Two-way random-effects ICC(2,1) for two methods over the same patients."""
    matrix = np.column_stack([first, second])
    n, k = matrix.shape
    if n < 3:
        return np.nan
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    ss_rows = k * float(((row_means - grand) ** 2).sum())
    ss_cols = n * float(((col_means - grand) ** 2).sum())
    ss_total = float(((matrix - grand) ** 2).sum())
    ss_error = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1)) if n > 1 and k > 1 else np.nan
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator > 0 else np.nan


def agreement_table(summary: pd.DataFrame, contrast: str) -> pd.DataFrame:
    wide = {}
    for method in summary.method.unique():
        subset = summary[(summary.method == method) & (summary.contrast == contrast)]
        wide[method] = subset.set_index("patient")["overall"]
    rows = []
    methods = [m for m in METHODS if m in wide]
    for first, second in combinations(methods, 2):
        common = wide[first].index.intersection(wide[second].index)
        paired = pd.DataFrame({"a": wide[first].loc[common], "b": wide[second].loc[common]}).dropna()
        if len(paired) < MIN_GROUP:
            continue
        difference = paired["a"] - paired["b"]
        deviation = float(difference.std(ddof=1))
        rows.append({
            "Pair": f"{first} vs {second}",
            "N patients": int(len(paired)),
            "Pearson r": float(stats.pearsonr(paired["a"], paired["b"]).statistic),
            "Spearman rho": float(stats.spearmanr(paired["a"], paired["b"]).statistic),
            "ICC(2,1)": _icc21(paired["a"].to_numpy(), paired["b"].to_numpy()),
            "Mean difference (µV)": float(difference.mean()),
            "95% LoA low": float(difference.mean() - 1.96 * deviation),
            "95% LoA high": float(difference.mean() + 1.96 * deviation),
            "Same sign": float((np.sign(paired["a"]) == np.sign(paired["b"])).mean()),
            "Wilcoxon p": _signed_rank_p(difference.to_numpy()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = _add_fdr(out, "Wilcoxon p")
    return out


# --------------------------------------------------------------------------- #
# Page body
# --------------------------------------------------------------------------- #
st.title("Sleep-stage deltas inside the same patient")
st.caption("REM, DEEP and LIGHT compared within patient, for the three HEP cleaning methods. "
           "Each patient contributes one HEP vector per stage on the six shared electrodes "
           "(72 bins, 0-600 ms); the delta is the stage difference of those vectors, so every "
           "test is paired within patient rather than between cohorts.")

with st.sidebar:
    st.header("Sleep-stage analysis")
    chosen_cohorts = st.multiselect(
        "Cohorts with staged caches", list(STAGE_DATASETS),
        default=list(STAGE_DATASETS),
        format_func=lambda key: STAGE_DATASETS[key], key="stage_page_cohorts")
    chosen_methods = st.multiselect(
        "Cleaning methods", list(METHODS), default=list(METHODS),
        key="stage_page_methods")
    window_name = st.selectbox("Averaging window", list(WINDOW_PRESETS),
                               index=0, key="stage_page_window")
    window_lo, window_hi = WINDOW_PRESETS[window_name]

if not chosen_cohorts or not chosen_methods:
    st.warning("Pick at least one cohort and one cleaning method in the sidebar.")
    st.stop()

summary = delta_summary(tuple(chosen_cohorts), tuple(chosen_methods),
                        window_lo, window_hi)
if summary.empty:
    st.error("No patient has usable HEP data in more than one sleep stage "
             "for the selected cohorts and methods.")
    st.stop()

zero_stats = overall_stats(tuple(chosen_cohorts), tuple(chosen_methods),
                           window_lo, window_hi)
feature_frame = feature_stats(tuple(chosen_cohorts), tuple(chosen_methods))
cross_frame = method_stats(tuple(chosen_cohorts), tuple(chosen_methods))
methods_here = [m for m in METHODS if m in set(summary.method)]

tabs = st.tabs([
    "Overview", "Scalp maps", "Per electrode", "Overall & distributions",
    "Age & sex", "Diagnosis & BMI", "Method agreement", "Stats & math",
    "Tables & export",
])

# --------------------------------------------------------------------------- #
with tabs[0]:
    pairs = {contrast: int(summary.loc[summary.contrast == contrast, "patient"].nunique())
             for contrast in STAGE_CONTRAST_NAMES}
    columns = st.columns(5)
    columns[0].metric("Patients with stage deltas", summary.patient.nunique())
    columns[1].metric("REM − DEEP pairs", pairs["REM − DEEP"])
    columns[2].metric("REM − LIGHT pairs", pairs["REM − LIGHT"])
    columns[3].metric("DEEP − LIGHT pairs", pairs["DEEP − LIGHT"])
    columns[4].metric("Methods", len(methods_here))
    composition = (summary[summary.method == methods_here[0]]
                   .groupby("cohort")["patient"].nunique().sort_values(ascending=False))
    st.caption("Patients per cohort: "
               + ", ".join(f"{name} {count}" for name, count in composition.items())
               + f". ICA is limited to {summary.loc[summary.method == 'ICA', 'patient'].nunique()} "
                 "patients because several external-cohort ICA caches are zero or degenerate "
                 "and are rejected by the scale check.")

    if len(zero_stats) and zero_stats.Significant.any():
        best = zero_stats[zero_stats.Significant].sort_values("q").iloc[0]
        st.success(f"Strongest overall stage effect: **{best.contrast}** under **{best.method}** — "
                   f"median {best.Median:+.3f} µV, mean {best.Mean:+.3f} µV, "
                   f"d$_z$={best.d_z:+.2f}, q={best.q:.2g}, N={int(best.N)} patients. "
                   "Overall = mean of the six electrode window means.")
    elif len(zero_stats):
        st.info("No overall window mean differs from zero after FDR correction.")

    if len(feature_frame) and feature_frame.Significant.any():
        top = (feature_frame[feature_frame.Significant]
               .sort_values("q_value").head(12)
               [["method", "contrast", "Feature", "N patients", "Median Δ (µV)",
                 "Mean Δ (µV)", "Patients with Δ > 0", "q_value"]])
        st.markdown("#### Strongest electrode × window effects")
        st.dataframe(top, hide_index=True, use_container_width=True, column_config={
            "Median Δ (µV)": st.column_config.NumberColumn(format="%+.3f"),
            "Mean Δ (µV)": st.column_config.NumberColumn(format="%+.3f"),
            "Patients with Δ > 0": st.column_config.NumberColumn(format="%.2f"),
            "q_value": st.column_config.NumberColumn("q (BH)", format="%.3g")})
        st.caption("Each row is a paired Wilcoxon signed-rank test of one electrode window "
                   "against zero, FDR-corrected within its method × contrast family.")

    st.markdown("#### Overall effect per method and contrast")
    display = zero_stats.rename(columns={"method": "Method", "contrast": "Contrast"})
    st.dataframe(display[["Method", "Contrast", "N", "Mean", "Median", "SD", "d_z", "% > 0", "p", "q"]],
                 hide_index=True, use_container_width=True, column_config={
                     "Mean": st.column_config.NumberColumn("Mean Δ (µV)", format="%+.3f"),
                     "Median": st.column_config.NumberColumn("Median Δ (µV)", format="%+.3f"),
                     "SD": st.column_config.NumberColumn("SD (µV)", format="%.3f"),
                     "d_z": st.column_config.NumberColumn(format="%+.2f"),
                     "% > 0": st.column_config.NumberColumn(format="%.2f"),
                     "p": st.column_config.NumberColumn(format="%.3g"),
                     "q": st.column_config.NumberColumn("q (BH)", format="%.3g")})

    st.markdown("#### Patients per cohort and contrast")
    counts = (summary.groupby(["cohort", "contrast"])["patient"].nunique()
              .unstack(fill_value=0).reindex(columns=STAGE_CONTRAST_NAMES, fill_value=0))
    st.dataframe(counts, use_container_width=True)
    st.caption("A patient appears in a contrast only when both stages were recorded with a "
               "usable HEP, which is why the per-contrast counts differ.")

# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("Scalp maps of the stage delta")
    st.caption(f"Mean delta over {window_name}. MNE topomaps interpolate the six measured "
               "electrodes inside the head outline, so the field between them is a guide rather "
               "than an independent measurement. Each map carries its own colour bar; the range "
               "is fixed per method across its three contrasts so the rows stay comparable, but "
               "not across methods, whose delta sizes differ by several fold.")

    scales = {}
    for method in methods_here:
        magnitudes = [abs(value) for contrast in STAGE_CONTRAST_NAMES
                      for value in summary[(summary.method == method)
                                           & (summary.contrast == contrast)][SITES]
                      .mean().to_numpy() if np.isfinite(value)]
        scales[method] = max([max(magnitudes, default=0.0), 1e-3])

    header = st.columns([1.0] + [1.0] * len(methods_here))
    for column, method in zip(header[1:], methods_here):
        column.markdown(f"**{method}**")
    for contrast in STAGE_CONTRAST_NAMES:
        row = st.columns([1.0] + [1.0] * len(methods_here))
        row[0].markdown(f"**{contrast}**")
        for column, method in zip(row[1:], methods_here):
            subset = summary[(summary.method == method) & (summary.contrast == contrast)]
            values = subset[SITES].mean().to_dict()
            with column:
                figure = brain_map_figure(values, f"{method} · {contrast}",
                                          scales[method], int(subset.patient.nunique()))
                st.pyplot(figure, use_container_width=True)
                plt.close(figure)

    if len(methods_here) > 1:
        st.divider()
        st.subheader("Paired map of the method difference")
        st.caption("Difference of the same patient's site delta between two cleaning methods, "
                   "computed on the paired subset only. Each map has its own colour bar because "
                   "the differences are much smaller than the deltas themselves.")
        contrast_for_map = st.selectbox("Contrast", STAGE_CONTRAST_NAMES,
                                        key="stage_map_diff_contrast")
        pair_labels = [f"{first} − {second}"
                       for first, second in combinations(methods_here, 2)]
        map_columns = st.columns(len(pair_labels))
        rows = []
        for column, (first, second), label in zip(map_columns, combinations(methods_here, 2),
                                                  pair_labels):
            values, n_pairs = method_site_differences(summary, contrast_for_map, first, second)
            if not values:
                column.info("Too few paired patients.")
                continue
            finite = [abs(v) for v in values.values() if np.isfinite(v)]
            with column:
                figure = brain_map_figure(values, label, max(max(finite, default=0.0), 1e-3), n_pairs)
                st.pyplot(figure, use_container_width=True)
                plt.close(figure)
            rows.append({"Comparison": label, **{site: values[site] for site in SITES},
                         "N paired": n_pairs})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                         column_config={site: st.column_config.NumberColumn(format="%+.3f")
                                        for site in SITES})

# --------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader("Per-electrode stage deltas")
    st.caption("Electrode values are the mean over the chosen window, averaged across patients. "
               "Tests below use the full 0-600 ms feature vector, one test per electrode window.")

    per_site = summary.groupby(["method", "contrast"])[SITES].mean()
    table = per_site.unstack("method").reindex(STAGE_CONTRAST_NAMES)
    table.columns = [f"{site} · {method}" for site, method in table.columns]
    st.dataframe(table.style.format("{:+.4f}"), use_container_width=True)

    first, second = st.columns(2)
    bar_contrast = first.selectbox("Contrast for the electrode bars", STAGE_CONTRAST_NAMES,
                                   key="stage_electrode_bars_contrast")
    heat_method = second.selectbox("Method for the electrode heatmap", methods_here,
                                   key="stage_electrode_heat_method")
    bars = (summary[summary.contrast == bar_contrast]
            .groupby("method")[SITES].mean()
            .reindex(methods_here).reset_index()
            .melt(id_vars="method", var_name="Electrode", value_name="Δ (µV)"))
    figure = px.bar(bars, x="Electrode", y="Δ (µV)", color="method", barmode="group",
                    color_discrete_map=METHOD_COLORS,
                    title=f"{bar_contrast}: window mean by electrode and method")
    figure.add_hline(y=0, line_color="grey", line_width=1)
    figure.update_layout(template="plotly_white", height=400,
                         legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
    st.plotly_chart(figure, use_container_width=True)

    heat = (summary[summary.method == heat_method]
            .groupby("contrast")[SITES].mean().reindex(STAGE_CONTRAST_NAMES))
    limit = float(np.nanmax(np.abs(heat.to_numpy()))) or 1e-3
    st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="RdBu_r",
                              zmin=-limit, zmax=limit, text_auto=".3f",
                              labels={"x": "Electrode", "y": "Contrast", "color": "Δ µV"},
                              title=f"{heat_method}: electrode × contrast window mean"),
                    use_container_width=True)

    if len(feature_frame):
        st.markdown("#### Significant electrode × window tests")
        tally = (feature_frame.groupby(["method", "contrast", "Electrode"], as_index=False)
                 .agg(Tests=("q_value", "size"), Significant=("Significant", "sum"),
                      Best_q=("q_value", "min"),
                      Median=("Median Δ (µV)", "median")))
        chosen_method = st.selectbox("Method", methods_here, key="stage_electrode_tally_method")
        st.dataframe(tally[tally.method == chosen_method].sort_values(["contrast", "Best_q"]),
                     hide_index=True, use_container_width=True, column_config={
                         "Best_q": st.column_config.NumberColumn("best q (BH)", format="%.3g"),
                         "Median": st.column_config.NumberColumn("median Δ (µV)", format="%+.3f")})
        st.caption("12 windows per electrode (50 ms bins, 0-600 ms); q is corrected within "
                   "method × contrast across all 72 features.")

# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Overall delta: distribution and paired changes")
    st.caption("Overall delta = mean of the six electrode window means, so every patient has one "
               "value per method and contrast. Boxes show the median and quartiles; points are "
               "outliers beyond 1.5 IQR.")

    figure = px.box(summary, x="contrast", y="overall", color="method", points="outliers",
                    color_discrete_map=METHOD_COLORS, category_orders={"contrast": STAGE_CONTRAST_NAMES},
                    labels={"contrast": "Contrast", "overall": "Overall Δ (µV)", "method": "Method"},
                    title=f"Overall stage delta, {window_name}")
    figure.add_hline(y=0, line_color="grey", line_width=1)
    figure.update_layout(template="plotly_white", height=430, boxmode="group",
                         legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
    st.plotly_chart(figure, use_container_width=True)

    selectors = st.columns(3)
    contrast = selectors[0].selectbox("Contrast", STAGE_CONTRAST_NAMES, key="stage_overall_contrast")
    method = selectors[1].selectbox("Method", methods_here, key="stage_overall_method")
    histogram = px.histogram(
        summary[(summary.contrast == contrast) & (summary.method == method)],
        x="overall", nbins=40, color_discrete_sequence=[METHOD_COLORS[method]],
        labels={"overall": "Overall Δ (µV)"},
        title=f"{method} · {contrast}: patient distribution")
    histogram.add_vline(x=0, line_color="grey", line_width=1)
    histogram.update_layout(template="plotly_white", height=340, showlegend=False,
                            yaxis_title="Patients")
    selectors[2].plotly_chart(histogram, use_container_width=True)

    if len(methods_here) > 1:
        st.markdown("#### Same patients under two methods")
        pair = st.selectbox("Method pair", [f"{a} vs {b}" for a, b in combinations(methods_here, 2)],
                            key="stage_overall_pair")
        first, second = pair.split(" vs ")
        pivot = (summary[summary.contrast == contrast]
                 .pivot_table(index="patient", columns="method", values="overall"))
        if {first, second}.issubset(pivot.columns):
            paired = pivot[[first, second]].dropna()
            figure = px.scatter(paired, x=first, y=second, opacity=0.55,
                                labels={first: f"{first} Δ (µV)", second: f"{second} Δ (µV)"},
                                title=f"{contrast}: {first} vs {second} per patient (N={len(paired)})")
            limit = float(np.nanmax(np.abs(paired.to_numpy()))) * 1.05 or 1e-3
            figure.add_shape(type="line", x0=-limit, y0=-limit, x1=limit, y1=limit,
                             line=dict(color="grey", dash="dash", width=1))
            figure.add_hline(y=0, line_color="lightgrey", line_width=1)
            figure.add_vline(x=0, line_color="lightgrey", line_width=1)
            figure.update_layout(template="plotly_white", height=430)
            st.plotly_chart(figure, use_container_width=True)
            rho = stats.spearmanr(paired[first], paired[second])
            st.caption(f"Spearman ρ={rho.statistic:.2f} (p={rho.pvalue:.2g}); the dashed line is "
                       "identity. Points off the diagonal are patients whose delta depends on the "
                       "cleaning method.")

    st.markdown("#### Effect size and paired test per cell")
    st.dataframe(zero_stats.rename(columns={"method": "Method", "contrast": "Contrast"}),
                 hide_index=True, use_container_width=True, column_config={
                     "Mean": st.column_config.NumberColumn("Mean Δ (µV)", format="%+.3f"),
                     "Median": st.column_config.NumberColumn("Median Δ (µV)", format="%+.3f"),
                     "SD": st.column_config.NumberColumn("SD (µV)", format="%.3f"),
                     "d_z": st.column_config.NumberColumn(format="%+.2f"),
                     "% > 0": st.column_config.NumberColumn(format="%.2f"),
                     "p": st.column_config.NumberColumn(format="%.3g"),
                     "q": st.column_config.NumberColumn("q (BH)", format="%.3g")})

# --------------------------------------------------------------------------- #
with tabs[4]:
    st.subheader("Age and sex")
    coverage = summary.drop_duplicates("patient")
    with_age = int(coverage.age.notna().sum())
    st.caption(f"Age and sex come from the HSP ``psg_metadata.csv`` files (BIDS folder) and the "
               f"COBRAD clinical sheet (record id). {with_age} of {len(coverage)} staged patients "
               "have an age; the rest stay out of these comparisons rather than being imputed.")

    selectors = st.columns(2)
    demo_method = selectors[0].selectbox("Method", methods_here, key="stage_demo_method")
    demo_contrast = selectors[1].selectbox("Contrast", STAGE_CONTRAST_NAMES,
                                           key="stage_demo_contrast")
    subset = summary[(summary.method == demo_method) & (summary.contrast == demo_contrast)]
    aged = subset.dropna(subset=["age", "overall"])

    if len(aged) >= MIN_GROUP:
        rho = stats.spearmanr(aged["age"], aged["overall"])
        slope, intercept = np.polyfit(aged["age"].to_numpy(float),
                                      aged["overall"].to_numpy(float), 1)
        figure = px.scatter(aged, x="age", y="overall", color="cohort", opacity=0.6,
                            labels={"age": "Age (years)", "overall": "Overall Δ (µV)",
                                    "cohort": "Cohort"},
                            title=f"{demo_method} · {demo_contrast}: age vs stage delta "
                                  f"(N={len(aged)})")
        grid = np.linspace(aged["age"].min(), aged["age"].max(), 50)
        figure.add_trace(go.Scatter(x=grid, y=slope * grid + intercept, mode="lines",
                                    name="linear fit", line=dict(color="black", width=2)))
        figure.add_hline(y=0, line_color="grey", line_width=1)
        figure.update_layout(template="plotly_white", height=420)
        st.plotly_chart(figure, use_container_width=True)
        st.caption(f"Spearman ρ={rho.statistic:+.2f}, p={rho.pvalue:.2g}; slope "
                   f"{slope:+.4f} µV/year. Correlation, not a within-patient change.")

        age_table, age_test, age_p = subgroup_table(subset, demo_method, demo_contrast,
                                                   "age_bin", AGE_LABELS + ["unknown"])
        st.markdown("#### Age bins")
        st.dataframe(age_table, hide_index=True, use_container_width=True, column_config={
            "Median Δ (µV)": st.column_config.NumberColumn(format="%+.3f"),
            "Mean Δ (µV)": st.column_config.NumberColumn(format="%+.3f"),
            "dB": st.column_config.NumberColumn("Cohen d", format="%+.2f"),
            "% > 0": st.column_config.NumberColumn(format="%.2f")})
        if age_test:
            st.caption(f"{age_test} across bins with at least {MIN_GROUP} patients: p={age_p:.3g}.")
        else:
            st.caption(f"Not enough patients per bin (minimum {MIN_GROUP}) to test across ages.")

    st.markdown("#### Sex")
    sex_table, sex_test, sex_p = subgroup_table(subset, demo_method, demo_contrast, "sex")
    if sex_table["N patients"].sum() >= 2 * MIN_GROUP:
        st.dataframe(sex_table, hide_index=True, use_container_width=True, column_config={
            "Median Δ (µV)": st.column_config.NumberColumn(format="%+.3f"),
            "Mean Δ (µV)": st.column_config.NumberColumn(format="%+.3f"),
            "dB": st.column_config.NumberColumn("Cohen d", format="%+.2f"),
            "% > 0": st.column_config.NumberColumn(format="%.2f")})
        figure = px.box(subset[subset.sex.isin(["Male", "Female"])], x="sex", y="overall",
                        color="sex", points="outliers",
                        labels={"sex": "Sex", "overall": "Overall Δ (µV)"},
                        title=f"{demo_method} · {demo_contrast}: stage delta by sex")
        figure.add_hline(y=0, line_color="grey", line_width=1)
        figure.update_layout(template="plotly_white", height=380, showlegend=False)
        st.plotly_chart(figure, use_container_width=True)
        st.caption(f"{sex_test}: p={sex_p:.3g}." if sex_test
                   else f"Both sexes need at least {MIN_GROUP} patients for a test.")
    else:
        st.info("Too few patients with a recorded sex in this cell.")

    st.markdown("#### Age and sex coverage by cohort")
    cohort_coverage = (coverage.assign(has_age=coverage.age.notna(),
                                       has_sex=coverage.sex.ne(""))
                       .groupby("cohort")
                       .agg(Patients=("patient", "nunique"),
                            With_age=("has_age", "sum"), With_sex=("has_sex", "sum")))
    st.dataframe(cohort_coverage, use_container_width=True)

# --------------------------------------------------------------------------- #
with tabs[5]:
    st.subheader("Diagnosis and BMI")
    coverage = summary.drop_duplicates("patient")
    coverage_table = pd.DataFrame([
        {"Variable": "Age", "Source": "HSP psg_metadata.csv, COBRAD clinical sheet",
         "Patients matched": int(coverage.age.notna().sum())},
        {"Variable": "Sex", "Source": "HSP psg_metadata.csv, COBRAD clinical sheet",
         "Patients matched": int(coverage.sex.fillna("").ne("").sum())},
        {"Variable": "Diagnosis", "Source": "COBRAD clinical sheet (bvFTD / LBD class flags)",
         "Patients matched": int(coverage.diagnosis.fillna("").ne("").sum())},
        {"Variable": "BMI", "Source": "no source records BMI for the staged cohorts",
         "Patients matched": int(coverage.bmi.notna().sum())},
    ])
    coverage_table["Staged patients"] = len(coverage)
    st.dataframe(coverage_table, hide_index=True, use_container_width=True)
    st.caption("The HSP cohorts are sleep studies with no diagnostic field: their metadata carries "
               "age, sex, study type and timing only. Diagnosis therefore exists only for the EDF "
               "subset that appears in the COBRAD clinical sheet.")

    st.markdown("#### Diagnosis groups")
    diagnosed = coverage[coverage.diagnosis.fillna("").ne("")]
    if diagnosed.empty:
        st.warning("No staged patient has a diagnosis record, so the diagnosis comparison cannot "
                   "run. The coverage table above shows exactly which variables are missing.")
    else:
        st.write(f"**{len(diagnosed)} diagnosed patients**, all from COBRAD / EDF. "
                 f"Classes: {diagnosed.diagnosis.value_counts().to_dict()}")
        diagnosis_selector = st.selectbox("Method for the diagnosis table", methods_here,
                                          key="stage_diagnosis_method")
        diagnosis_contrast = st.selectbox("Contrast", STAGE_CONTRAST_NAMES,
                                          key="stage_diagnosis_contrast")
        subset = summary[(summary.method == diagnosis_selector)
                         & (summary.contrast == diagnosis_contrast)]
        subset = subset[subset.diagnosis.fillna("").ne("")]
        table, test_name, p_value = subgroup_table(subset, diagnosis_selector,
                                                   diagnosis_contrast, "diagnosis")
        st.dataframe(table, hide_index=True, use_container_width=True, column_config={
            "Median Δ (µV)": st.column_config.NumberColumn(format="%+.3f"),
            "Mean Δ (µV)": st.column_config.NumberColumn(format="%+.3f"),
            "dB": st.column_config.NumberColumn("Cohen d", format="%+.2f"),
            "% > 0": st.column_config.NumberColumn(format="%.2f")})
        if test_name:
            st.caption(f"{test_name}: p={p_value:.3g}. Groups below {MIN_GROUP} patients are left "
                       "out of the test, which is why such a small cohort rarely supports one.")
        else:
            st.caption(f"No group reaches {MIN_GROUP} patients, so only descriptives are shown. "
                       "Read these as case descriptions, not as a diagnosis effect.")
        if subset.patient.nunique() < 30:
            st.warning("Diagnosis inference is unpowered here: fewer than 30 diagnosed staged "
                       "patients exist in every metadata source this dashboard can join.")

    st.markdown("#### BMI")
    if coverage.bmi.notna().any():
        bmi_selector = st.selectbox("Method for the BMI table", methods_here, key="stage_bmi_method")
        bmi_contrast = st.selectbox("Contrast", STAGE_CONTRAST_NAMES, key="stage_bmi_contrast")
        bmi_subset = summary[(summary.method == bmi_selector) & (summary.contrast == bmi_contrast)]
        bmi_subset = bmi_subset.dropna(subset=["bmi", "overall"])
        rho = stats.spearmanr(bmi_subset["bmi"], bmi_subset["overall"])
        figure = px.scatter(bmi_subset, x="bmi", y="overall", color="cohort",
                            labels={"bmi": "BMI", "overall": "Overall Δ (µV)"},
                            title=f"{bmi_selector} · {bmi_contrast}: BMI vs stage delta "
                                  f"(N={len(bmi_subset)})")
        figure.add_hline(y=0, line_color="grey", line_width=1)
        figure.update_layout(template="plotly_white", height=400)
        st.plotly_chart(figure, use_container_width=True)
        st.caption(f"Spearman ρ={rho.statistic:+.2f}, p={rho.pvalue:.2g}.")
    else:
        st.warning("BMI cannot be analysed: none of the metadata files that join to the staged "
                   "patients records height or weight (HSP `psg_metadata.csv` has 19 columns, the "
                   "COBRAD clinical sheet records age, sex, education and clinical scales). "
                   "The section is wired to the same name-detection logic as age and sex, so it "
                   "starts working unchanged if a `bmi` column is added to either source.")

# --------------------------------------------------------------------------- #
with tabs[6]:
    st.subheader("Do the cleaning methods agree?")
    st.caption("Agreement is computed on the patient-level window mean, on the paired subset only. "
               "ICC(2,1) treats patients as random and methods as fixed raters; the limits of "
               "agreement are mean difference ± 1.96 SD of the difference.")

    contrast = st.selectbox("Contrast", STAGE_CONTRAST_NAMES, key="stage_agree_contrast")
    agreement = agreement_table(summary, contrast)
    if len(agreement):
        st.dataframe(agreement, hide_index=True, use_container_width=True, column_config={
            "Pearson r": st.column_config.NumberColumn(format="%.2f"),
            "Spearman rho": st.column_config.NumberColumn(format="%.2f"),
            "ICC(2,1)": st.column_config.NumberColumn(format="%.2f"),
            "Mean difference (µV)": st.column_config.NumberColumn(format="%+.4f"),
            "95% LoA low": st.column_config.NumberColumn(format="%+.4f"),
            "95% LoA high": st.column_config.NumberColumn(format="%+.4f"),
            "Same sign": st.column_config.NumberColumn(format="%.2f"),
            "Wilcoxon p": st.column_config.NumberColumn(format="%.3g"),
            "q": st.column_config.NumberColumn("q (BH)", format="%.3g")})

        chosen_pair = st.selectbox("Bland-Altman pair", list(agreement.Pair),
                                   key="stage_agree_pair")
        first, second = chosen_pair.split(" vs ")
        pivot = (summary[summary.contrast == contrast]
                 .pivot_table(index="patient", columns="method", values="overall"))
        if {first, second}.issubset(pivot.columns):
            paired = pivot[[first, second]].dropna()
            difference = paired[first] - paired[second]
            mean_value = (paired[first] + paired[second]) / 2
            deviation = float(difference.std(ddof=1))
            figure = px.scatter(x=mean_value, y=difference, opacity=0.6,
                                labels={"x": "Mean of the two methods (µV)",
                                        "y": f"{first} − {second} (µV)"},
                                title=f"Bland-Altman, {contrast}, N={len(paired)}")
            figure.add_hline(y=float(difference.mean()), line_color="black", line_width=1,
                             annotation_text="mean")
            for sign in (-1, 1):
                figure.add_hline(y=float(difference.mean()) + sign * 1.96 * deviation,
                                 line_color="grey", line_dash="dash", line_width=1,
                                 annotation_text="±1.96 SD")
            figure.add_hline(y=0, line_color="lightgrey", line_width=1)
            figure.update_layout(template="plotly_white", height=430)
            st.plotly_chart(figure, use_container_width=True)
    else:
        st.info("Not enough paired patients for an agreement estimate.")

    if len(cross_frame):
        st.markdown("#### Feature-level method differences")
        summary_table = (cross_frame.groupby(["Comparison", "contrast"], as_index=False)
                         .agg(Tests=("q_value", "size"), Significant=("Significant", "sum"),
                              Best_q=("q_value", "min"),
                              Median_difference=("Median Δ difference (µV)", "median")))
        st.dataframe(summary_table, hide_index=True, use_container_width=True, column_config={
            "Best_q": st.column_config.NumberColumn("best q (BH)", format="%.3g"),
            "Median_difference": st.column_config.NumberColumn("median difference (µV)",
                                                               format="%+.3f")})
        with st.expander("All significant method differences"):
            significant = cross_frame[cross_frame.Significant]
            if len(significant):
                st.dataframe(significant.sort_values("q_value").head(80),
                             hide_index=True, use_container_width=True)
            else:
                st.write("No electrode window differs between methods after FDR.")
        st.download_button("Download method comparison CSV",
                           cross_frame.to_csv(index=False),
                           "stage_delta_method_comparison.csv", "text/csv")

# --------------------------------------------------------------------------- #
with tabs[7]:
    st.subheader("The maths behind the tables")
    st.caption("Definitions match the code in `hep_dashboard_common.py`; nothing here is computed "
               "differently from the tables above.")

    st.markdown("**1. Within-patient stage delta** — for patient *i*, method *m*, electrode *s*, "
                "50 ms window *w*:")
    st.latex(r"\Delta_{i,m,s,w} \;=\; \mathrm{HEP}^{first}_{i,m,s,w} \;-\; \mathrm{HEP}^{second}_{i,m,s,w}")
    st.markdown("**2. Electrode window mean** (what the maps show):")
    st.latex(r"\bar{\Delta}_{m,s} \;=\; \frac{1}{N}\sum_{i=1}^{N}\frac{1}{|W|}\sum_{w \in W}\Delta_{i,m,s,w}")
    st.markdown("**3. Overall delta** — mean of the six electrode window means, so each patient "
                "contributes one number:")
    st.latex(r"\bar{\Delta}_{i,m} \;=\; \frac{1}{6}\sum_{s=1}^{6}\frac{1}{|W|}\sum_{w \in W}\Delta_{i,m,s,w}")
    st.markdown("**4. Paired test against zero.** Wilcoxon signed-rank on the within-patient "
                "values, normal approximation with tie correction; the zero method drops exact "
                "zeros:")
    st.latex(r"W^{+} = \sum_{i:\,\bar{\Delta}_i > 0} R_i,\qquad z = \frac{W^{+} - \frac{n(n+1)}{4}}"
             r"{\sqrt{\frac{n(n+1)(2n+1)}{24}}},\qquad p = 2\,(1-\Phi(|z|))")
    st.markdown("**5. Multiple comparisons.** Benjamini-Hochberg within each method × contrast "
                "family of electrode windows (72 tests), and within each method-comparison family:")
    st.latex(r"q_{(k)} = \min_{j \ge k}\; \frac{n\,p_{(j)}}{j},\qquad \text{reject } H_0 \text{ when } q_{(k)} \le 0.05")
    st.markdown("**6. Effect sizes.** Standardised mean change and the share of patients with a "
                "positive delta:")
    st.latex(r"d_z = \frac{\bar{\Delta}}{s_{\Delta}},\qquad \pi^{+} = \frac{1}{N}\sum_i \mathbb{1}[\bar{\Delta}_i > 0]")
    st.markdown("**7. Group comparisons** (age bins, sex, diagnosis, cohort): Kruskal-Wallis for "
                "three or more groups, Mann-Whitney U for two; groups with fewer than "
                f"{MIN_GROUP} patients are excluded from the test but still shown.")
    st.markdown("**8. Agreement.** Two-way random-effects ICC(2,1) on the paired subset, with "
                "$MS_R$ between patients, $MS_C$ between methods and $MS_E$ residual:")
    st.latex(r"\mathrm{ICC}(2,1) = \frac{MS_R - MS_E}{MS_R + (k-1)MS_E + k\,(MS_C - MS_E)/n}")

    st.markdown("#### What each test controls")
    st.dataframe(pd.DataFrame([
        {"Analysis": "Electrode × window vs zero", "Test": "Wilcoxon signed-rank",
         "Unit": "patient", "Correction": "BH within method × contrast (72 tests)"},
        {"Analysis": "Overall vs zero", "Test": "Wilcoxon signed-rank", "Unit": "patient",
         "Correction": "BH across the method × contrast cells (up to 9)"},
        {"Analysis": "Method vs method, per feature", "Test": "Wilcoxon signed-rank on the paired difference",
         "Unit": "patient (both methods)", "Correction": "BH within comparison × contrast"},
        {"Analysis": "Age bins / diagnosis / cohort", "Test": "Kruskal-Wallis; Mann-Whitney U for two groups",
         "Unit": "patient", "Correction": "none, exploratory"},
        {"Analysis": "Sex", "Test": "Mann-Whitney U", "Unit": "patient", "Correction": "none, exploratory"},
        {"Analysis": "Method agreement", "Test": "ICC(2,1), Pearson, Spearman, Bland-Altman",
         "Unit": "patient (paired)", "Correction": "BH across method pairs"},
    ]), hide_index=True, use_container_width=True)
    st.caption("All tests are exploratory: cohorts are observational, diagnoses overlap, and the "
               "ICA contrasts rest on far fewer patients than HEP and CFAR.")

# --------------------------------------------------------------------------- #
with tabs[8]:
    st.subheader("Tables and export")
    st.caption("Everything below is computed from the cohorts, methods and window selected in the "
               "sidebar.")

    st.markdown("**Patient-level window means**")
    patient_table = summary[["patient", "cohort", "method", "contrast", "overall", "age", "sex",
                             "bmi", "diagnosis"] + SITES].copy()
    st.dataframe(patient_table.head(300), hide_index=True, use_container_width=True)
    st.download_button("Download patient-level deltas CSV",
                       patient_table.to_csv(index=False),
                       "sleep_stage_patient_deltas.csv", "text/csv")

    if len(feature_frame):
        st.markdown("**Electrode × window tests**")
        st.download_button("Download all electrode window tests CSV",
                           feature_frame.to_csv(index=False),
                           "sleep_stage_feature_tests.csv", "text/csv")
        st.dataframe(feature_frame.head(50), hide_index=True, use_container_width=True)

    st.markdown("**Overall tests**")
    st.download_button("Download overall tests CSV", zero_stats.to_csv(index=False),
                       "sleep_stage_overall_tests.csv", "text/csv")

    if len(cross_frame):
        st.download_button("Download method comparison CSV", cross_frame.to_csv(index=False),
                           "sleep_stage_method_tests.csv", "text/csv")

    st.markdown("**Site × method × contrast matrix**")
    matrix = (summary.groupby(["method", "contrast"])[SITES].mean()
              .round(6).reset_index())
    st.dataframe(matrix, hide_index=True, use_container_width=True)
    st.download_button("Download site matrix CSV", matrix.to_csv(index=False),
                       "sleep_stage_site_matrix.csv", "text/csv")

    st.markdown("**Metadata coverage**")
    st.download_button("Download metadata coverage CSV",
                       stage_demographics().to_csv(index=False),
                       "sleep_stage_metadata.csv", "text/csv")

st.divider()
st.caption("Deltas are paired within patient, so a patient contributes only to the contrasts for "
           "which both stages were recorded. Explanations and effect sizes describe this sample; "
           "they are not external validation.")
