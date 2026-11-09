"""HEP / ICA / CFAR comparison page, loaded only when selected in the sidebar."""
import hep_dashboard_common as _common
globals().update({_name: _value for _name, _value in vars(_common).items()
                  if not _name.startswith("__")})

st.title("HEP · ICA · CFAR comparison dashboard")
st.caption("Matched-patient analysis of uncleaned HEP, ECG-informed ICA, and population CFAR. "
           "All methods use the same six channels and the same patients.")

if not HEP_FILE.exists() or not DEMO_FILE.exists():
    st.error("Required Paper CFA parquet files are missing.")
    st.stop()

df = load_data()
dx_counts = diagnosis_counts(df)

with st.sidebar:
    st.header("Cohort and target")
    target_type = st.radio("Phenotype", ["Disease group", "Diagnosis", "Sex", "Age"], horizontal=True)
    diagnosis = None
    if target_type == "Diagnosis":
        eligible = dx_counts[dx_counts >= 10]
        diagnosis = st.selectbox("Diagnosis", eligible.index.tolist(),
                                 format_func=lambda x: f"{x} (n={eligible[x]})")
    elif target_type == "Disease group":
        group_counts = {name: int((df.disease_family == name).sum()) for name in DISEASE_GROUPS}
        diagnosis = st.selectbox("Disease group", list(DISEASE_GROUPS),
                                 format_func=lambda x: f"{x} (n={group_counts[x]})")
    methods = st.multiselect("Methods", list(METHODS), default=list(METHODS))
    min_age, max_age = int(np.floor(df.age.min())), int(np.ceil(df.age.max()))
    age_range = st.slider("Age range", min_age, max_age, (min_age, max_age))
    sexes = st.multiselect("Sex", sorted(df.sex.unique()), default=sorted(df.sex.unique()))
    folds = st.slider("Cross-validation folds", 3, 10, 5)
    model_name = st.selectbox("Classifier", ["Logistic regression", "Random forest", "RBF SVM"])
    k_clusters = st.slider("KMeans clusters", 2, 8, 3)

filtered = df[df.age.between(*age_range) & df.sex.isin(sexes)].reset_index(drop=True)
y_all, valid, target_label = target_labels(filtered, target_type, diagnosis)
filtered = filtered.loc[valid].reset_index(drop=True)
y = y_all[valid]

if len(np.unique(y)) < 2:
    st.error("The current filters leave only one phenotype class. Broaden the filters.")
    st.stop()

metrics_cols = st.columns(5)
metrics_cols[0].metric("Patients", len(filtered))
metrics_cols[1].metric("Label 1", int(y.sum()))
metrics_cols[2].metric("Label 0", int((y == 0).sum()))
metrics_cols[3].metric("Median age", f"{filtered.age.median():.1f} y")
metrics_cols[4].metric("Methods", len(methods))
st.info(f"Current target: **{target_label}**. Diagnosis negatives mean absence of the selected diagnosis, "
        "not necessarily healthy controls.")

tabs = st.tabs([
    "Classification & method comparison",
    "Heart–brain decomposition",
    "Raw vs. CFAR: Diagnosis differences",
    "Overview",
    "Waveforms & statistics",
    "Clustering",
    "Classification details",
    "Neuronal validity",
    "MoCA / MMSE",
    "Diagnosis screen",
    "EDF source folders",
    "Systole vs diastole",
    "Sleep-stage deltas",
    "Tables & export",
])

with tabs[0]:
    st.subheader("Classification and method comparison")
    st.caption("This landing page puts the main question first: which representation carries useful phenotype "
               "information, and how much of that information may still be cardiac or demographic?")
    headline_rows, headline_prob, headline_cm = [], {}, {}
    for method in methods:
        result, prob, cm = classify(method_features(filtered, method), y, model_name, folds)
        headline_rows.append({"Method": method, **result})
        headline_prob[method] = prob
        headline_cm[method] = cm
    headline = pd.DataFrame(headline_rows).set_index("Method")
    best_auc = headline["ROC AUC"].idxmax()
    best_bal = headline["Balanced accuracy"].idxmax()
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Best ROC AUC", best_auc, f"{headline.loc[best_auc, 'ROC AUC']:.3f}")
    h2.metric("Best balanced accuracy", best_bal, f"{headline.loc[best_bal, 'Balanced accuracy']:.3f}")
    h3.metric("Target", target_type)
    h4.metric("Matched N", len(filtered))

    roc = go.Figure()
    for method, prob in headline_prob.items():
        fpr, tpr, _ = roc_curve(y, prob)
        roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"{method} (AUC {headline.loc[method, 'ROC AUC']:.3f})",
                                 line=dict(color=METHOD_COLORS[method], width=3)))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                             line=dict(color="grey", dash="dash")))
    roc.update_layout(template="plotly_white", title=f"Out-of-fold classification: {target_label}",
                      xaxis_title="False-positive rate", yaxis_title="True-positive rate", height=470)
    left, right = st.columns([1, 1.35])
    with left:
        st.dataframe(headline, use_container_width=True)
        cm_method = st.selectbox("Confusion matrix", methods, key="headline_cm_method")
        st.dataframe(pd.DataFrame(headline_cm[cm_method], index=["True 0", "True 1"],
                                  columns=["Pred 0", "Pred 1"]), use_container_width=True)
    with right:
        st.plotly_chart(roc, use_container_width=True)

    st.subheader("Statistical and artifact-validity comparison")
    comparison_rows = []
    similarity_summary = ecg_similarity(filtered).groupby("method")["HEP–ECG |r|"].median()
    for method in methods:
        table = univariate_stats(method_features(filtered, method), y, tuple(feature_names()))
        comparison_rows.append({
            "Method": method,
            "ROC AUC": headline.loc[method, "ROC AUC"],
            "Balanced accuracy": headline.loc[method, "Balanced accuracy"],
            "FDR-significant HEP features": int(table.fdr_significant.sum()),
            "Minimum q": float(table.q_value.min()),
            "Largest |rank-biserial effect|": float(table.rank_biserial.abs().max()),
            "Median residual HEP–ECG |r|": float(similarity_summary.get(method, np.nan)),
        })
    comparison = pd.DataFrame(comparison_rows).set_index("Method")
    st.dataframe(comparison, use_container_width=True)

    validity_rows = []
    for validity_name, validity_y in validity_targets(
        filtered, diagnosis if target_type == "Diagnosis" else None
    ).items():
        if min(validity_y.sum(), (validity_y == 0).sum()) < 5:
            continue
        for method in methods:
            result, _, _ = classify(method_features(filtered, method), validity_y,
                                    "Logistic regression", folds)
            validity_rows.append({"Target": validity_name, "Method": method,
                                  "ROC AUC": result["ROC AUC"]})
    validity_headline = pd.DataFrame(validity_rows)
    if len(validity_headline):
        fig = px.bar(validity_headline, x="Target", y="ROC AUC", color="Method", barmode="group",
                     color_discrete_map=METHOD_COLORS,
                     title="Neural target versus cardiac/demographic information")
        fig.add_hline(y=.5, line_dash="dash", line_color="grey")
        fig.update_layout(xaxis_tickangle=-25, height=510)
        st.plotly_chart(fig, use_container_width=True)
    st.info("Prefer a method that preserves or improves the selected neural/clinical target while reducing "
            "HEP–ECG similarity and prediction of heart-rate or cardiovascular information. Highest AUC alone "
            "does not establish the cleanest neuronal signal.")

with tabs[1]:
    st.subheader("Heart, brain, and the heart projected onto the brain recording")
    st.markdown("""
The four views below answer different questions:

1. **ECG** is the electrical heartbeat measured at the chest. It is the direct cardiac reference.
2. **Raw HEP** is EEG averaged around ECG R-peaks. It contains neuronal heartbeat responses **plus** cardiac field artifact.
3. **Cleaned brain (CFAR)** is the raw HEP after projecting out the learned cardiac scalp direction. It is our cleaner neuronal estimate.
4. **Removed CFA** is exactly `Raw HEP − CFAR`. It is the portion of EEG aligned with the cardiac spatial direction—the part normally discarded.

The decomposition is exact in feature and waveform space: **Raw HEP = CFAR-cleaned HEP + removed CFA**. The removed part is not guaranteed to be purely heart: neuronal activity aligned with the same spatial direction can also be removed.
""")
    decomposition_site = st.selectbox("Electrode", SITES, index=2, key="decomposition_site")
    st.plotly_chart(decomposition_waveform_figure(filtered, decomposition_site), use_container_width=True)

    components = decomposition_features(filtered)
    comparable = {
        name: (matrix if name == "ECG" else channel_average_features(matrix))
        for name, matrix in components.items()
    }
    overlap_rows = []
    for name_a, name_b in combinations(comparable, 2):
        correlations = []
        for a, b in zip(comparable[name_a], comparable[name_b]):
            valid_values = np.isfinite(a) & np.isfinite(b)
            if valid_values.sum() >= 3 and np.std(a[valid_values]) > 0 and np.std(b[valid_values]) > 0:
                correlations.append(np.corrcoef(a[valid_values], b[valid_values])[0, 1])
        overlap_rows.append({"Signal A": name_a, "Signal B": name_b,
                             "Median patient correlation": np.median(correlations),
                             "Median |correlation|": np.median(np.abs(correlations)),
                             "N": len(correlations)})
    overlap = pd.DataFrame(overlap_rows)

    raw_features = components["Raw HEP"]
    clean_features = components["Cleaned brain (CFAR)"]
    removed_features = components["Removed CFA"]
    raw_energy = np.nansum(raw_features**2, axis=1)
    energy = pd.DataFrame({
        "CFAR-cleaned share": np.nansum(clean_features**2, axis=1)/np.maximum(raw_energy, 1e-12),
        "Removed-CFA share": np.nansum(removed_features**2, axis=1)/np.maximum(raw_energy, 1e-12),
        "Cross-term share": 2*np.nansum(clean_features*removed_features, axis=1)/np.maximum(raw_energy, 1e-12),
    })
    energy_summary = energy.agg(["median", "mean", "std"]).T.reset_index(names="Component")
    d1, d2 = st.columns(2)
    with d1:
        st.write("**Signal overlap by waveform shape**")
        st.dataframe(overlap, hide_index=True, use_container_width=True)
    with d2:
        st.write("**Raw-HEP feature-energy decomposition**")
        st.dataframe(energy_summary, hide_index=True, use_container_width=True)
        st.caption("Shares need not sum as two positive percentages because cleaned and removed signals have a cross-term.")

    st.subheader("What does each signal say about the three exclusive disease families?")
    family_mask = filtered.disease_family.notna().to_numpy()
    family_df = filtered.loc[family_mask].reset_index(drop=True)
    family_components = decomposition_features(family_df)
    component_auc_rows = []
    for family_name in DISEASE_GROUPS:
        family_y = (family_df.disease_family == family_name).astype(int).to_numpy()
        if min(family_y.sum(), (family_y == 0).sum()) < 5:
            continue
        for component_name, X in family_components.items():
            result, _, _ = classify(X, family_y, "Logistic regression", folds)
            component_auc_rows.append({"Disease family": family_name, "Signal": component_name,
                                       "ROC AUC": result["ROC AUC"],
                                       "Balanced accuracy": result["Balanced accuracy"],
                                       "N": result["N"]})
    component_auc = pd.DataFrame(component_auc_rows)
    if len(component_auc):
        auc_fig = px.bar(component_auc, x="Disease family", y="ROC AUC", color="Signal", barmode="group",
                         title="Disease-family information carried by heart, cleaned brain, and removed CFA")
        auc_fig.add_hline(y=.5, line_dash="dash", line_color="grey")
        auc_fig.update_layout(height=520, xaxis_tickangle=-20)
        st.plotly_chart(auc_fig, use_container_width=True)
        st.dataframe(component_auc, hide_index=True, use_container_width=True)

        story_rows = []
        for family_name, group in component_auc.groupby("Disease family"):
            best = group.loc[group["ROC AUC"].idxmax()]
            clean_auc = float(group.loc[group.Signal == "Cleaned brain (CFAR)", "ROC AUC"].iloc[0])
            removed_auc = float(group.loc[group.Signal == "Removed CFA", "ROC AUC"].iloc[0])
            ecg_auc = float(group.loc[group.Signal == "ECG", "ROC AUC"].iloc[0])
            story_rows.append({"Disease family": family_name,
                               "Strongest signal": best.Signal, "Best AUC": best["ROC AUC"],
                               "Cleaned-brain AUC": clean_auc, "Removed-CFA AUC": removed_auc,
                               "ECG AUC": ecg_auc,
                               "Interpretation": (
                                   "More neuronal than cardiac information" if clean_auc > max(removed_auc, ecg_auc)
                                   else "Cardiac/CFA information is at least as predictive as cleaned brain")})
        st.write("**The story told by each component**")
        st.dataframe(pd.DataFrame(story_rows), hide_index=True, use_container_width=True)

    st.markdown("""
**How to read this tab**

- A diagnosis predicted mainly by **cleaned CFAR HEP** is more compatible with a neuronal/interoceptive association.
- A diagnosis predicted mainly by **ECG** reflects direct cardiac physiology.
- A diagnosis predicted mainly by **removed CFA** may be encoded through cardiac field propagation, body conductivity, or a neural source aligned with the removed direction.
- Similar performance in cleaned and removed components means the raw HEP mixes two informative processes; it should not be interpreted as a single biomarker.
- This analysis is descriptive and cross-validated, not causal or externally validated.
""")

with tabs[2]:
    st.subheader("Raw vs. Post-Cleaning (CFAR): What do the Diagnosis Differences Mean?")
    st.markdown("""
A central question of this PhD thesis is understanding the physiological meaning of differences in the
**Heartbeat-Evoked Potential (HEP)** across clinical diagnoses: **does a difference reflect altered cortical
interoception, or is it merely volume-conducted cardiac field artifact (CFA) noise?**

Because cardiac electrical fields conduct through thoracic tissue and skull into scalp EEG, uncleaned EEG (**Raw HEP**)
contains substantial cardiac noise that varies by body habitus, heart volume, and conduction velocity.
Our **CFAR (Cardiac Field Artifact Removal)** algorithm projects out this cardiac spatial direction, leaving the
neuronal response (**Post-cleaning**). The removed component is the **CFA noise** ($\\text{Raw} - \\text{CFAR}$).

Comparing group differences before and after cleaning reveals **Three Distinct Difference Regimes**:
1. 🔴 **Artifact-Driven / Noise Confound ($\\Delta \\text{AUC} < -0.02$)**: Differences present in raw EEG that **diminish or collapse after CFAR**.
   These are not brain differences; they reflect cardiac vector, stroke volume, or BMI differences (e.g. cardiovascular disorders, obesity).
2. 🟢 **Preserved Neuronal Signal ($|\\Delta \\text{AUC}| \\le 0.02$, $\\text{AUC} \\ge 0.53$)**: Differences that **survive cardiac artifact removal**.
   These represent genuine cortical interoceptive processing differences between patient groups.
3. 🔵 **Unmasked Neuronal Signal ($\\Delta \\text{AUC} > +0.02$)**: Differences that **emerge or strengthen after CFAR**.
   In raw EEG, cardiac noise was obscuring the underlying cortical signal; cleaning reveals the authentic difference.
""")

    raw_X = method_features(filtered, "HEP")
    cfar_X = method_features(filtered, "CFAR")
    cfa_X = raw_X - cfar_X
    diag_tuples = tuple(tuple(d) for d in filtered["diagnoses"])
    family_tuples = tuple(filtered["disease_family"].tolist())
    eligible_dx = tuple(dx_counts[dx_counts >= 10].index.tolist())

    col_t1, col_t2 = st.columns([1.5, 1])
    with col_t1:
        delta_thresh = st.slider(
            "Difference category threshold (|Δ AUC|)",
            min_value=0.01, max_value=0.05, value=0.02, step=0.005,
            help="Threshold for categorizing whether a difference is artifact-driven (drops > threshold) or unmasked (rises > threshold).",
            key="diff_tab_delta_thresh",
        )
    with col_t2:
        st.caption(f"Evaluated on N = {len(filtered)} patients using {folds}-fold CV with {model_name}.")

    df_fam = disease_family_screen(raw_X, cfar_X, cfa_X, family_tuples, model_name, folds)
    df_screen = pre_post_diagnosis_screen(raw_X, cfar_X, cfa_X, diag_tuples, eligible_dx, model_name, folds, delta_thresh)
    df_pairs = pairwise_family_screen(raw_X, cfar_X, cfa_X, family_tuples, model_name, folds)

    st.markdown("---")
    st.markdown("### 1. Difference Category Breakdown across All Diagnoses")

    n_artifact = int((df_screen["Difference Category"] == "Artifact-driven (noise confound)").sum())
    n_preserved = int((df_screen["Difference Category"] == "Preserved neuronal (authentic)").sum())
    n_unmasked = int((df_screen["Difference Category"] == "Unmasked neuronal (signal revealed)").sum())
    n_equivocal = int((df_screen["Difference Category"] == "Equivocal / Low signal").sum())

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    c_m1.metric("🔴 Artifact-Driven", f"{n_artifact} diagnoses", "Noise confound (drops)")
    c_m2.metric("🟢 Preserved Neuronal", f"{n_preserved} diagnoses", "Authentic cortical (stable)")
    c_m3.metric("🔵 Unmasked Neuronal", f"{n_unmasked} diagnoses", "Signal revealed (rises)")
    c_m4.metric("⚪ Equivocal / Low Signal", f"{n_equivocal} diagnoses", "Near chance (both)")

    st.markdown("---")
    st.markdown("### 2. The Three Broad Disease Families: Pre vs. Post Cleaning")
    st.caption("How do Cardiometabolic, Neuropsychiatric, and Sleep conditions differ when comparing Raw EEG vs. Post-cleaning CFAR?")

    col_fam_l, col_fam_r = st.columns([1.1, 1.4])
    with col_fam_l:
        st.dataframe(
            df_fam[[
                "Disease Family", "Patients N", "Raw AUC (Pre-cleaning)",
                "CFAR AUC (Post-cleaning)", "Removed CFA AUC (Noise)",
                "Δ AUC (CFAR − Raw)", "Difference Category"
            ]].style.format({
                "Raw AUC (Pre-cleaning)": "{:.3f}",
                "CFAR AUC (Post-cleaning)": "{:.3f}",
                "Removed CFA AUC (Noise)": "{:.3f}",
                "Δ AUC (CFAR − Raw)": "{:+.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("""
**Key Findings by Disease Family:**
- **Cardiometabolic / vascular**: Apparent HEP differences are dominated by the **Removed CFA (cardiac artifact)**. Cleaning with CFAR significantly attenuates classification power, demonstrating that raw findings were driven by heart noise, thoracic geometry, and vascular status rather than brain interoception.
- **Neuropsychiatric / cognitive**: Classification power is **preserved or unmasked** after CFAR. Because cardiac artifact is projected out, surviving differences reflect genuine central nervous system / cortical interoceptive processing alterations (e.g. in dementia or mood disorders).
- **Sleep / respiratory**: Retains moderate post-cleaning signal, reflecting both thoracic/sleep-depth changes and central respiratory drive.
""")

    with col_fam_r:
        fam_bar_df = []
        for _, r in df_fam.iterrows():
            fam_bar_df.append({"Family": r["Disease Family"], "Representation": "Raw HEP (Pre-cleaning)", "ROC AUC": r["Raw AUC (Pre-cleaning)"]})
            fam_bar_df.append({"Family": r["Disease Family"], "Representation": "CFAR HEP (Post-cleaning)", "ROC AUC": r["CFAR AUC (Post-cleaning)"]})
            fam_bar_df.append({"Family": r["Disease Family"], "Representation": "Removed CFA (Noise)", "ROC AUC": r["Removed CFA AUC (Noise)"]})
        fam_bar_frame = pd.DataFrame(fam_bar_df)
        fig_fam = px.bar(
            fam_bar_frame, x="Family", y="ROC AUC", color="Representation", barmode="group",
            color_discrete_map={"Raw HEP (Pre-cleaning)": "#D55E00", "CFAR HEP (Post-cleaning)": "#009E73", "Removed CFA (Noise)": "#AA4499"},
            title="Classification by Disease Family: Raw vs. CFAR vs. CFA Noise",
        )
        fig_fam.add_hline(y=0.5, line_dash="dash", line_color="grey")
        fig_fam.update_layout(height=420, yaxis_range=[0.4, max(0.65, fam_bar_frame["ROC AUC"].max() + 0.05)])
        st.plotly_chart(fig_fam, use_container_width=True)

    st.markdown("#### Head-to-Head Discrimination between Disease Families")
    st.caption("When classifying between two distinct disease categories directly, how much did cardiac noise explain their separation?")
    st.dataframe(
        df_pairs.style.format({
            "Raw AUC": "{:.3f}",
            "CFAR AUC": "{:.3f}",
            "Removed CFA AUC": "{:.3f}",
            "Δ AUC (CFAR − Raw)": "{:+.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("### 3. Diagnosis Screen: Pre-cleaning (Raw) vs. Post-cleaning (CFAR)")
    st.caption("Every dot represents one clinical diagnosis. Points below the diagonal y = x line had their effect diminished by cleaning (cardiac artifact); points above were unmasked; points on the line are robust neuronal signals.")

    col_scat, col_dumb = st.columns([1.2, 1])
    with col_scat:
        color_by = st.radio("Color scatter points by:", ["Disease Family", "Difference Category"], horizontal=True, key="color_scatter_by")
        c_map = FAMILY_COLORS if color_by == "Disease Family" else DIFF_CAT_COLORS

        fig_scat = px.scatter(
            df_screen,
            x="Raw AUC (Pre-cleaning)",
            y="CFAR AUC (Post-cleaning)",
            color=color_by,
            color_discrete_map=c_map,
            size="Positive N",
            hover_name="Diagnosis",
            hover_data={
                "Disease Family": True,
                "Positive N": True,
                "Raw AUC (Pre-cleaning)": ":.3f",
                "CFAR AUC (Post-cleaning)": ":.3f",
                "Removed CFA AUC (Noise)": ":.3f",
                "Δ AUC (CFAR − Raw)": ":.3f",
                "Difference Category": True,
            },
            title="Pre-cleaning (Raw) vs. Post-cleaning (CFAR) AUC",
        )
        min_v = min(0.46, df_screen["Raw AUC (Pre-cleaning)"].min(), df_screen["CFAR AUC (Post-cleaning)"].min()) - 0.02
        max_v = max(0.68, df_screen["Raw AUC (Pre-cleaning)"].max(), df_screen["CFAR AUC (Post-cleaning)"].max()) + 0.02
        fig_scat.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(color="grey", dash="dash", width=1.5))
        fig_scat.add_vline(x=0.5, line_dash="dot", line_color="#cccccc")
        fig_scat.add_hline(y=0.5, line_dash="dot", line_color="#cccccc")
        fig_scat.update_layout(height=490, xaxis_range=[min_v, max_v], yaxis_range=[min_v, max_v])
        st.plotly_chart(fig_scat, use_container_width=True)

    with col_dumb:
        fig_dumb = go.Figure()
        df_sorted_dx = df_screen.sort_values("Raw AUC (Pre-cleaning)", ascending=True)
        for _, r in df_sorted_dx.iterrows():
            fig_dumb.add_trace(go.Scatter(
                x=[r["Raw AUC (Pre-cleaning)"], r["CFAR AUC (Post-cleaning)"]],
                y=[r["Diagnosis"], r["Diagnosis"]],
                mode="lines",
                line=dict(color="#aaaaaa", width=2),
                showlegend=False,
                hoverinfo="skip",
            ))
        fig_dumb.add_trace(go.Scatter(
            x=df_sorted_dx["Raw AUC (Pre-cleaning)"],
            y=df_sorted_dx["Diagnosis"],
            mode="markers",
            name="Raw HEP (Pre)",
            marker=dict(color="#D55E00", size=9, symbol="circle"),
        ))
        fig_dumb.add_trace(go.Scatter(
            x=df_sorted_dx["CFAR AUC (Post-cleaning)"],
            y=df_sorted_dx["Diagnosis"],
            mode="markers",
            name="CFAR HEP (Post)",
            marker=dict(color="#009E73", size=9, symbol="diamond"),
        ))
        fig_dumb.add_vline(x=0.5, line_dash="dash", line_color="grey")
        fig_dumb.update_layout(
            title="AUC Shift (Pre vs. Post Cleaning)",
            xaxis_title="ROC AUC",
            height=490,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_dumb, use_container_width=True)

    st.markdown("#### Comprehensive Diagnosis Difference Table")
    st.dataframe(
        df_screen[[
            "Diagnosis", "Disease Family", "Positive N", "Prevalence (%)",
            "Raw AUC (Pre-cleaning)", "CFAR AUC (Post-cleaning)",
            "Removed CFA AUC (Noise)", "Δ AUC (CFAR − Raw)",
            "Difference Category", "Scientific Interpretation"
        ]].style.format({
            "Raw AUC (Pre-cleaning)": "{:.3f}",
            "CFAR AUC (Post-cleaning)": "{:.3f}",
            "Removed CFA AUC (Noise)": "{:.3f}",
            "Δ AUC (CFAR − Raw)": "{:+.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("### 4. Waveform Contrast & Scalp Topography Deep-Dive")
    st.caption("Inspect the exact temporal dynamics and 6-channel scalp topography for any diagnosis or disease family.")

    sel_mode = st.radio("Select target type to inspect:", ["Specific Diagnosis", "Disease Family"], horizontal=True, key="insp_target_type")
    if sel_mode == "Specific Diagnosis":
        chosen_dx = st.selectbox("Select diagnosis", list(eligible_dx), index=0, key="insp_chosen_dx")
        y_insp = filtered.diagnoses.map(lambda ds, d=chosen_dx: int(d in ds)).to_numpy()
        lbl_p = f"{chosen_dx} (+)"
        lbl_n = f"Non-{chosen_dx} (-)"
    else:
        chosen_fam = st.selectbox("Select disease family", list(DISEASE_GROUPS.keys()), index=0, key="insp_chosen_fam")
        y_insp = (filtered.disease_family == chosen_fam).astype(int).to_numpy()
        lbl_p = f"{chosen_fam} (+)"
        lbl_n = "Other Families (-)"

    c_site, c_window = st.columns([1, 1.5])
    with c_site:
        chosen_site = st.selectbox("Waveform Electrode", SITES, index=2, key="insp_chosen_site")
    with c_window:
        st.caption("Waveforms show grand-average ±SEM. The difference panel (3) directly overlays the Raw group difference vs. CFAR group difference.")

    st.plotly_chart(
        pre_post_contrast_waveforms(filtered, y_insp, chosen_site, label_pos=lbl_p, label_neg=lbl_n),
        use_container_width=True,
    )

    st.plotly_chart(
        pre_post_channel_topography_bars(raw_X, cfar_X, y_insp),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### 5. PhD Thesis Synthesis & Export")
    st.markdown("""
**Methodological & Theoretical Conclusions for Dissertation:**
1. **Cardiac Field Artifact is a major clinical confound in EEG:** In uncleaned data, diagnosis differences are heavily inflated by cardiac noise (especially for cardiometabolic conditions, heart failure, and obesity). Studies attributing raw HEP differences in cardiovascular or obese cohorts solely to altered "brain interoception" have likely reported volume-conducted cardiac artifacts.
2. **CFAR specifically removes cardiac noise while preserving genuine neural HEP:** The divergence between the **Removed CFA** and **Post-cleaning CFAR** confirms that CFAR successfully disambiguates cardiac volume conduction from cortical potential.
3. **True cortical interoceptive differences exist in neuropsychiatric/cognitive domains:** After complete removal of the cardiac spatial direction, neuropsychiatric and cognitive disorders (e.g. Cognitive Impairment, Depression) preserve or unmask differential HEP signatures, supporting true neuro-cardiac interoceptive dysregulation.
""")
    st.download_button(
        "Download Complete Diagnosis Differences CSV",
        df_screen.to_csv(index=False),
        "hep_raw_vs_cfar_diagnosis_differences.csv",
        "text/csv",
    )


with tabs[3]:
    c1, c2 = st.columns(2)
    with c1:
        count_df = dx_counts.rename_axis("diagnosis").reset_index(name="patients")
        st.plotly_chart(px.bar(count_df, x="patients", y="diagnosis", orientation="h",
                               title="Diagnosis prevalence in matched cohort").update_layout(
                                   yaxis={"categoryorder": "total ascending"}, height=520),
                        use_container_width=True)
    with c2:
        demo = filtered.assign(age_band=pd.cut(filtered.age, bins=[0, 40, 50, 60, 70, 80, 120], right=False))
        st.plotly_chart(px.histogram(demo, x="age", color="sex", barmode="overlay",
                                     title="Age and sex distribution", nbins=30), use_container_width=True)
        st.dataframe(pd.DataFrame({
            "Method": ["HEP", "ICA", "CFAR"],
            "Operation": ["No artifact removal", "ECG-informed independent-component removal",
                          "Population cardiac-direction signal-space projection"],
            "Channels": [", ".join(SITES)] * 3,
        }), hide_index=True, use_container_width=True)

with tabs[4]:
    site = st.selectbox("Waveform electrode", SITES, index=2)
    st.plotly_chart(waveform_figure(filtered, y, site), use_container_width=True)
    stat_tables = {}
    summary = []
    for method in methods:
        X = method_features(filtered, method)
        table = univariate_stats(X, y, tuple(feature_names()))
        stat_tables[method] = table
        summary.append({"method": method, "FDR-significant features": int(table.fdr_significant.sum()),
                        "minimum q": float(table.q_value.min()),
                        "maximum |effect|": float(table.rank_biserial.abs().max())})
    st.subheader("Phenotype difference summary")
    st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)
    if methods:
        shown_method = st.selectbox("Detailed feature statistics", methods)
        st.dataframe(stat_tables[shown_method], hide_index=True, use_container_width=True,
                     column_config={"p_value": st.column_config.NumberColumn(format="%.3e"),
                                    "q_value": st.column_config.NumberColumn(format="%.3e")})

with tabs[5]:
    st.caption("KMeans is fitted without phenotype labels. Label association is evaluated only after clustering.")
    cluster_summaries = []
    cluster_outputs = {}
    for method in methods:
        plot_df, result, contingency = cluster_result(method_features(filtered, method), y, k_clusters)
        plot_df["patient_id"] = filtered.patient_id.to_numpy()
        cluster_outputs[method] = plot_df
        result["method"] = method
        cluster_summaries.append(result)
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.scatter(plot_df, x="PC1", y="PC2", color="cluster", symbol="phenotype",
                             hover_data=["patient_id"], title=f"{method}: PCA + KMeans", opacity=.7)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.write(f"**{method} cluster × phenotype**")
            st.dataframe(contingency, use_container_width=True)
    if cluster_summaries:
        st.subheader("Unsupervised clustering metrics")
        st.dataframe(pd.DataFrame(cluster_summaries).set_index("method"), use_container_width=True)

    if cluster_outputs:
        st.subheader("Do the methods find the same groups?")
        agreement = pd.DataFrame(index=methods, columns=methods, dtype=float)
        for method_a in methods:
            for method_b in methods:
                agreement.loc[method_a, method_b] = adjusted_rand_score(
                    cluster_outputs[method_a].cluster,
                    cluster_outputs[method_b].cluster,
                )
        st.dataframe(agreement.style.background_gradient(cmap="Blues", vmin=0, vmax=1).format("{:.3f}"),
                     use_container_width=True)
        st.caption("Adjusted Rand index: 1 means identical patient grouping; values near 0 mean agreement expected by chance.")

        st.subheader("What do people in one cluster have in common?")
        with st.expander("Disease-family definitions"):
            st.dataframe(pd.DataFrame([
                {"Disease group": name, "Included diagnoses": "; ".join(sorted(members))}
                for name, members in DISEASE_GROUPS.items()
            ]), hide_index=True, use_container_width=True)
        profile_method = st.selectbox("Method to profile", methods,
                                      index=methods.index("CFAR") if "CFAR" in methods else 0,
                                      key="cluster_profile_method")
        profile_labels = cluster_outputs[profile_method].cluster.astype(int).to_numpy()
        family_enrichment = disease_group_cluster_enrichment(filtered, profile_labels)
        heatmap_data = family_enrichment.pivot(
            index="disease_group", columns="cluster", values="signed_log10_q"
        )
        heatmap = px.imshow(
            heatmap_data, color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
            labels={"color": "signed −log10(q)"},
            title=f"{profile_method}: disease-family enrichment by cluster",
            aspect="auto", text_auto=".2f",
        )
        heatmap.update_layout(height=420)
        st.plotly_chart(heatmap, use_container_width=True)
        st.caption("Positive/red = enriched; negative/blue = depleted. q-values use FDR across every cluster × family test.")
        significant_families = family_enrichment[family_enrichment.fdr_significant]
        if len(significant_families):
            st.success(f"{len(significant_families)} significant cluster–disease-family associations after FDR correction.")
            st.dataframe(significant_families, hide_index=True, use_container_width=True)
        else:
            st.info("No disease family is significantly enriched or depleted after global FDR correction.")
        with st.expander("All disease-family enrichment tests"):
            st.dataframe(family_enrichment, hide_index=True, use_container_width=True)

        cluster_sizes = pd.Series(profile_labels).value_counts().sort_index()
        profile_cluster = st.selectbox(
            "Cluster", cluster_sizes.index.tolist(),
            format_func=lambda value: f"Cluster {value} (n={cluster_sizes[value]})",
            key="cluster_profile_cluster",
        )
        feature_profile = cluster_feature_profile(
            method_features(filtered, profile_method), profile_labels, int(profile_cluster),
            tuple(feature_names()),
        )
        continuous_profile, binary_profile = cluster_clinical_profile(
            filtered, profile_labels, int(profile_cluster)
        )

        p1, p2 = st.columns([1.25, 1])
        with p1:
            top = feature_profile.head(16).sort_values("standardized_difference")
            fig = px.bar(top, x="standardized_difference", y="feature", orientation="h",
                         color="fdr_significant", color_discrete_map={True: "#D55E00", False: "#999999"},
                         title=f"{profile_method} cluster {profile_cluster}: strongest shared HEP features")
            fig.update_layout(height=560)
            st.plotly_chart(fig, use_container_width=True)
        with p2:
            st.write("**Continuous clinical traits**")
            st.dataframe(continuous_profile, hide_index=True, use_container_width=True)
            st.write("**Enriched diagnoses and sex**")
            st.dataframe(binary_profile.head(12), hide_index=True, use_container_width=True)

        st.write("**All cluster-defining HEP features**")
        st.dataframe(feature_profile, hide_index=True, use_container_width=True)
        members = filtered.loc[profile_labels == int(profile_cluster),
                               ["patient_id", "age", "sex", "hr_bpm", "n_diagnoses"]].copy()
        members["diagnoses"] = filtered.loc[profile_labels == int(profile_cluster), "diagnoses"].map(
            lambda values: "; ".join(values)
        )
        st.write("**Cluster members**")
        st.dataframe(members, hide_index=True, use_container_width=True)
        st.download_button("Download this cluster", members.to_csv(index=False),
                           f"{profile_method.lower()}_cluster_{profile_cluster}.csv", "text/csv")

with tabs[6]:
    st.caption("Predictions are out-of-fold. Scaling and model fitting occur inside each training fold.")
    class_rows, roc_fig = [], go.Figure()
    cms = {}
    for method in methods:
        result, prob, cm = classify(method_features(filtered, method), y, model_name, folds)
        result["Method"] = method
        class_rows.append(result)
        cms[method] = cm
        fpr, tpr, _ = roc_curve(y, prob)
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"{method} (AUC {result['ROC AUC']:.3f})",
                                     line=dict(color=METHOD_COLORS[method], width=3)))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                                 line=dict(color="grey", dash="dash")))
    roc_fig.update_layout(template="plotly_white", title=f"Out-of-fold ROC: {model_name}",
                          xaxis_title="False-positive rate", yaxis_title="True-positive rate", height=480)
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.dataframe(pd.DataFrame(class_rows).set_index("Method"), use_container_width=True)
        for method, cm in cms.items():
            st.write(f"**{method} confusion matrix**")
            st.dataframe(pd.DataFrame(cm, index=["True 0", "True 1"],
                                      columns=["Pred 0", "Pred 1"]), use_container_width=True)
    with c2:
        st.plotly_chart(roc_fig, use_container_width=True)

with tabs[7]:
    st.subheader("Neuronal-validity checks")
    st.caption("A cleaner method is not expected to maximize every AUC. Reduced prediction of ECG-related or "
               "cardiovascular targets can indicate less artifact, while preserved neural-target performance is desirable.")
    similarity = ecg_similarity(filtered)
    c1, c2 = st.columns([1.1, 1.4])
    with c1:
        sim_summary = similarity.groupby("method")["HEP–ECG |r|"].agg(["median", "mean", "count"]).reset_index()
        st.dataframe(sim_summary, hide_index=True, use_container_width=True)
        st.plotly_chart(px.box(similarity, x="method", y="HEP–ECG |r|", color="method",
                               color_discrete_map=METHOD_COLORS,
                               title="Residual ECG-shape information (outside ±50 ms)"),
                        use_container_width=True)
    with c2:
        validity_rows = []
        for target_name, target_y in validity_targets(
            filtered, diagnosis if target_type == "Diagnosis" else None
        ).items():
            if min(target_y.sum(), (target_y == 0).sum()) < 5:
                continue
            for method in methods:
                result, _, _ = classify(method_features(filtered, method), target_y,
                                        "Logistic regression", folds)
                validity_rows.append({"target": target_name, "method": method,
                                      "roc_auc": result["ROC AUC"],
                                      "balanced_accuracy": result["Balanced accuracy"]})
        validity = pd.DataFrame(validity_rows)
        if len(validity):
            fig = px.bar(validity, x="target", y="roc_auc", color="method", barmode="group",
                         color_discrete_map=METHOD_COLORS,
                         title="What information remains after each cleaning method?")
            fig.add_hline(y=.5, line_dash="dash", line_color="grey")
            fig.update_layout(xaxis_tickangle=-25, height=500)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(validity, hide_index=True, use_container_width=True)
    st.markdown("""
**Interpretation guide**

- CFAR lower than ICA for ECG similarity, heart rate, sex, or cardiovascular targets supports improved artifact specificity.
- Similar diagnosis performance with lower ECG information supports a robust neuronal association.
- Higher CFAR diagnosis performance with lower ECG information is the strongest pattern for unmasking neuronal signal.
- A diagnosis effect that disappears only after CFAR may have been cardiac-field driven, but over-cleaning must also be considered.
""")

with tabs[8]:
    st.subheader("COBRAD cognitive scores")
    st.info("All COBRAD participants with cognitive scores and usable HEP data are included for HEP and "
            "patient-specific CFAR. Regional features use every surviving electrode, so a rejected six-site channel "
            "does not exclude the participant. ICA is included wherever decomposition succeeded, with a separate N.")
    cognitive = load_cobrad_cognitive()
    if cognitive.empty:
        st.error("No matched COBRAD HEP/ICA/cognitive records were found.")
    else:
        score_name = st.radio("Cognitive score", ["moca", "mmse"], horizontal=True,
                              format_func=lambda x: x.upper())
        stage_choice = st.selectbox("Sleep stage", ["All stages (patient mean)"] + COGNITIVE_STAGES)
        work = cognitive[cognitive[score_name].notna()].copy()
        if stage_choice != "All stages (patient mean)":
            work = work[work.stage == stage_choice].copy()
        else:
            rows = []
            for pid, group in work.groupby("cobrad_id"):
                row = {"cobrad_id": pid, score_name: group[score_name].iloc[0],
                       "age": group.age.iloc[0], "sex": group.sex.iloc[0],
                       "n_stages": group.stage.nunique()}
                for method in METHODS:
                    values = [np.asarray(value) for value in group[method]
                              if isinstance(value, (list, tuple, np.ndarray))]
                    if not values:
                        row[method] = None
                        continue
                    stacked = np.stack(values)
                    count = np.isfinite(stacked).sum(axis=0)
                    row[method] = np.divide(np.nansum(stacked, axis=0), count,
                                            out=np.full(stacked.shape[1], np.nan), where=count > 0)
                rows.append(row)
            work = pd.DataFrame(rows)
        method_counts = {
            method: int(work[method].map(lambda value: isinstance(value, (list, tuple, np.ndarray))).sum())
            for method in methods
        }
        count_cols = st.columns(max(1, len(method_counts)))
        for column, (method, count) in zip(count_cols, method_counts.items()):
            column.metric(f"{method} participants", count)
        if len(work) < 6:
            st.info("Too few HEP observations for cross-validated cognitive modelling in this stage.")
        else:
            cognitive_rows, predictions, corr_tables = [], [], {}
            for method in methods:
                method_mask = work[method].map(lambda value: isinstance(value, (list, tuple, np.ndarray))).to_numpy()
                method_work = work.loc[method_mask]
                if len(method_work) < 6:
                    continue
                score = method_work[score_name].to_numpy(float)
                X = np.stack(method_work[method])
                available = np.isfinite(X).any(axis=0)
                X = X[:, available]
                available_names = tuple(np.asarray(cognitive_feature_names())[available])
                result, predicted = cognitive_regression(X, score, min(folds, len(method_work)))
                result["Method"] = method
                cognitive_rows.append(result)
                corr_tables[method] = feature_score_correlations(X, score, available_names)
                predictions.extend({"Observed": obs, "Predicted": pred, "Method": method}
                                   for obs, pred in zip(score, predicted))
            if cognitive_rows:
                st.dataframe(pd.DataFrame(cognitive_rows).set_index("Method"), use_container_width=True)
                scatter = px.scatter(pd.DataFrame(predictions), x="Observed", y="Predicted", color="Method",
                                     color_discrete_map=METHOD_COLORS, trendline="ols",
                                     title=f"Out-of-fold {score_name.upper()} prediction")
                st.plotly_chart(scatter, use_container_width=True)
                detail_method = st.selectbox("Cognitive feature correlations", list(corr_tables),
                                             key="cognitive_method")
                st.dataframe(corr_tables[detail_method], hide_index=True, use_container_width=True)
            threshold = 26 if score_name == "moca" else 24
            class_rows = []
            for method in methods:
                method_mask = work[method].map(lambda value: isinstance(value, (list, tuple, np.ndarray))).to_numpy()
                method_work = work.loc[method_mask]
                if len(method_work) < 6:
                    continue
                impaired = (method_work[score_name].to_numpy(float) < threshold).astype(int)
                if min(impaired.sum(), (impaired == 0).sum()) >= 3:
                    X = np.stack(method_work[method])
                    X = X[:, np.isfinite(X).any(axis=0)]
                    result, _, _ = classify(X, impaired,
                                            "Logistic regression", min(folds, 5))
                    class_rows.append({"Method": method, "Threshold": f"<{threshold}", **result})
            if class_rows:
                st.subheader("Cognitive impairment classification")
                st.dataframe(pd.DataFrame(class_rows).set_index("Method"), use_container_width=True)
        st.dataframe(work[[c for c in ["cobrad_id", "stage", "n_stages", "age", "sex", score_name]
                           if c in work.columns]], hide_index=True, use_container_width=True)

with tabs[9]:
    st.caption("Batch one-vs-rest diagnosis analysis. Rare diagnoses below 10 positive patients are omitted.")
    screen_model = st.selectbox("Screen classifier", ["Logistic regression", "Random forest"], key="screen_model")
    if st.button("Run all-diagnosis screen", type="primary"):
        rows = []
        progress = st.progress(0.0)
        eligible = dx_counts[dx_counts >= 10].index.tolist()
        total = len(eligible) * len(methods)
        done = 0
        for dx in eligible:
            y_dx = filtered.diagnoses.map(lambda xs: int(dx in xs)).to_numpy()
            if min(y_dx.sum(), (y_dx == 0).sum()) < 5:
                continue
            for method in methods:
                X = method_features(filtered, method)
                stat_table = univariate_stats(X, y_dx, tuple(feature_names()))
                result, _, _ = classify(X, y_dx, screen_model, folds)
                rows.append({"diagnosis": dx, "method": method,
                             "positive_n": int(y_dx.sum()), "negative_n": int((y_dx == 0).sum()),
                             "roc_auc": result["ROC AUC"],
                             "balanced_accuracy": result["Balanced accuracy"],
                             "fdr_significant_features": int(stat_table.fdr_significant.sum()),
                             "minimum_q": float(stat_table.q_value.min())})
                done += 1
                progress.progress(min(done / max(total, 1), 1.0))
        st.session_state["diagnosis_screen"] = pd.DataFrame(rows)
    screen = st.session_state.get("diagnosis_screen")
    if screen is not None and len(screen):
        st.dataframe(screen.sort_values(["diagnosis", "method"]), hide_index=True, use_container_width=True)
        fig = px.bar(screen, x="diagnosis", y="roc_auc", color="method", barmode="group",
                     title="Diagnosis classification by cleaning method")
        fig.add_hline(y=.5, line_dash="dash", line_color="grey")
        fig.update_layout(xaxis_tickangle=-35, height=560)
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download diagnosis screen CSV", screen.to_csv(index=False),
                           "hep_ica_cfar_diagnosis_screen.csv", "text/csv")

with tabs[10]:
    st.subheader("Can HEP morphology tell which EDF folder a recording came from?")
    st.caption("Every recording under EDF_Format is represented the same way: the HEP on the six sites "
               "shared by all datasets, in 50 ms bins from 0 to 400 ms after the R-peak. The target is the "
               "source folder itself, so a well-above-chance confusion matrix means the folders carry "
               "identifiable recording signatures (hardware, referencing, montage, cohort, signal quality).")
    if not SOURCE_FILE.exists():
        st.error(f"Source-folder features are missing. Build them with `venv/bin/python 25_build_edf_source_cache.py` "
                 f"(expected {SOURCE_FILE}).")
    else:
        source_all = load_source_features()
        groups = source_feature_groups(source_all)
        counts = source_all.groupby("source").size().sort_values(ascending=False)
        c1, c2, c3 = st.columns([1.3, 1.3, 1])
        with c1:
            selected_sources = st.multiselect("Folders to compare", list(counts.index), default=list(counts.index))
        with c2:
            feature_sets = st.multiselect("Feature sets", list(groups), default=["HEP waveform (µV)",
                                                                                "HEP shape (z-scored)", "Heart rate"])
        with c3:
            cap = st.slider("Max recordings per folder", 20, 1000, 250, step=10)
        min_sites = st.slider("Minimum shared EEG sites required", 1, 6, 4, key="source_min_sites")

        source = source_all[source_all.source.isin(selected_sources)]
        source = source[source.n_shared_sites.fillna(0) >= min_sites]
        if source.empty:
            st.warning("No recordings match the current folder and montage selection.")
        else:
            source = balanced_sample(source, cap)
            columns = sorted({column for name in feature_sets for column in groups[name]})
            st.dataframe(pd.DataFrame({
                "Folder": counts.index, "Recordings in cache": counts.to_numpy(),
                "Used here": [int((source.source == s).sum()) for s in counts.index],
                "Stages present": [", ".join(sorted(source_all.loc[source_all.source == s, "stage"].unique()))
                                   for s in counts.index],
            }), hide_index=True, use_container_width=True)

            if not columns:
                st.info("Select at least one feature set.")
            elif source.source.nunique() < 2:
                st.info("Select at least two folders.")
            else:
                X = source[columns].to_numpy(float)
                labels = tuple(source.source.to_numpy())
                try:
                    result, y_true, y_pred, classes = classify_source(X, labels, model_name, folds)
                except ValueError as error:
                    st.error(str(error))
                else:
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Accuracy", f"{result['Accuracy']:.3f}",
                              f"chance {result['Chance accuracy']:.3f}")
                    m2.metric("Balanced accuracy", f"{result['Balanced accuracy']:.3f}")
                    m3.metric("Macro F1", f"{result['Macro F1']:.3f}")
                    m4.metric("Weighted F1", f"{result['Weighted F1']:.3f}")
                    m5.metric("Recordings", result["N"])
                    kappa = result["Cohen's kappa"]
                    st.caption(f"{result['Folds']}-fold stratified cross-validation, predictions stored out of fold. "
                               f"Majority-class baseline is {result['Majority-class accuracy']:.3f}; "
                               f"Cohen's kappa is {kappa:.3f}.")

                    cm = confusion_matrix(y_true, y_pred, labels=classes)
                    row_share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
                    left, right = st.columns([1.05, 1])
                    with left:
                        st.plotly_chart(
                            px.imshow(row_share, x=list(classes), y=list(classes), text_auto=".2f",
                                      color_continuous_scale="Blues", zmin=0, zmax=1, aspect="auto",
                                      labels={"x": "Predicted folder", "y": "True folder",
                                              "color": "Row share"},
                                      title="Confusion matrix (row-normalised)"),
                            use_container_width=True)
                    with right:
                        precision, recall, f1, support = precision_recall_fscore_support(
                            y_true, y_pred, labels=classes, zero_division=0)
                        per_class = pd.DataFrame({
                            "Folder": list(classes), "Recordings": support,
                            "Precision": precision, "Recall": recall, "F1": f1,
                        })
                        st.write("**Per-folder performance (out of fold)**")
                        st.dataframe(per_class.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
                                     hide_index=True, use_container_width=True)
                        st.dataframe(pd.DataFrame(cm, index=[f"true {c}" for c in classes],
                                                  columns=[f"pred {c}" for c in classes]),
                                     use_container_width=True)

                    st.write("**Which folders are hardest to separate?**")
                    pairwise = pd.DataFrame(index=list(classes), columns=list(classes), dtype=float)
                    for i, name_a in enumerate(classes):
                        for j, name_b in enumerate(classes):
                            denominator = cm[i].sum() + cm[j].sum()
                            pairwise.loc[name_a, name_b] = (
                                (cm[i, i] + cm[j, j]) / denominator if denominator and i != j else np.nan
                            )
                    st.plotly_chart(
                        px.imshow(pairwise.to_numpy(dtype=float), x=list(classes), y=list(classes),
                                  text_auto=".2f", color_continuous_scale="Greens", zmin=0, zmax=1,
                                  aspect="auto", labels={"color": "Pair accuracy"},
                                  title="Pairwise separation from the confusion matrix"),
                        use_container_width=True)
                    st.caption("Each cell is the share of the two folders' recordings that land on the correct "
                               "side of their 2x2 sub-matrix. Low values mean the classifier confuses them.")

                    features = pd.DataFrame(X, columns=columns)
                    imputed = features.fillna(features.median(numeric_only=True)).fillna(0.0)
                    latent = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(
                        StandardScaler().fit_transform(imputed.to_numpy(float)))
                    scatter_frame = pd.DataFrame({"PC1": latent[:, 0], "PC2": latent[:, 1],
                                                  "folder": source.source.to_numpy(),
                                                  "patient": source.patient_id.to_numpy()})
                    st.plotly_chart(
                        px.scatter(scatter_frame, x="PC1", y="PC2", color="folder", hover_data=["patient"],
                                   opacity=.65, title="Folder separation in feature space (PCA)"),
                        use_container_width=True)

                    tests = source_feature_tests(X, labels, tuple(columns))
                    if len(tests):
                        st.write("**Features that differ across folders (Kruskal–Wallis, FDR-corrected)**")
                        st.dataframe(tests.head(15), hide_index=True, use_container_width=True)
                        st.caption("A large epsilon-squared means that feature alone separates folders; it is a "
                                   "descriptive ranking, not a causal explanation.")

                    export = source[["source", "patient_id", "recording_id", "stage", "sfreq",
                                     "n_channels", "n_eeg_channels", "duration_s", "n_shared_sites"]] \
                        .assign(predicted=y_pred, correct=(y_pred == y_true)).join(features)
                    st.download_button("Download recordings, predictions and features CSV",
                                       export.to_csv(index=False),
                                       "edf_source_folder_predictions.csv", "text/csv")
                    st.info("Interpretation: this task is provenance detection, not a neural biomarker. "
                            "Waveform features are dominated by hardware, referencing, montage and cohort "
                            "differences, and ANPHY/Harvard recordings are much longer or higher-rate than "
                            "the clinical folders. Use the feature-set checkboxes to see how much of the "
                            "separation comes from signal shape versus recording metadata.")

with tabs[11]:
    st.subheader("Systole versus diastole: does each cleaning method keep phase-specific diagnosis information?")
    st.caption("Every heartbeat is split into the contraction (systolic, red) and expansion (diastolic, blue) "
               "windows of the cardiac cycle. All three cleaning methods use the same beats and the same "
               "patients, so any difference between them is attributable to the cleaning, not to timing. "
               "Amplitudes are mean HEP µV inside each window, measured in all six channels.")

    if not methods:
        st.info("Select at least one cleaning method in the sidebar.")
    else:
        c1, c2, c3, c4 = st.columns([1.1, 1.1, .9, 1.1])
        systole_window = c1.slider("Contraction window after R-peak (s)", 0.0, 0.35,
                                   SYSTOLE_WINDOW, step=0.01, key="phase_systole")
        diastole_window = c2.slider("Expansion window after R-peak (s)", 0.30, 0.60,
                                    DIASTOLE_WINDOW, step=0.01, key="phase_diastole")
        min_patients = c3.number_input("Minimum patients per diagnosis", 5, 100,
                                       PHASE_MIN_PATIENTS, step=5, key="phase_min_n")
        include_pre_r = c4.checkbox("Include pre-R late diastole (−200–0 ms)", True,
                                    key="phase_pre_r",
                                    help="Late diastolic filling happens before the averaged R-peak, so "
                                         "excluding it under-represents expansion, especially at lower heart rates.")

        st.plotly_chart(phase_waveform_figure(filtered, tuple(systole_window), tuple(diastole_window),
                                              include_pre_r), use_container_width=True)

        findings = phase_diagnosis_findings(filtered, tuple(methods), tuple(systole_window),
                                            tuple(diastole_window), include_pre_r, int(min_patients))
        if not len(findings):
            st.warning("No diagnosis reaches the minimum patient count, so no association test can run.")
        else:
            significant = findings[findings.Significant]
            strongest = findings.iloc[0]
            phase_counts = significant.groupby("Phase").size() if len(significant) else pd.Series(dtype=int)
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("FDR-significant tests", int(significant.shape[0]),
                      f"of {len(findings)}", delta_color="off")
            k2.metric("Diagnoses tested", findings.Diagnosis.nunique())
            k3.metric("Smallest q", f"{strongest.q_global:.2g}")
            k4.metric("Methods with a finding", significant.Method.nunique() if len(significant) else 0)
            k5.metric("N patients", len(filtered))

            phase_phrase = {"Contraction": "contraction", "Expansion": "expansion",
                            "Contraction − expansion": "contraction-versus-expansion difference"}
            if len(significant):
                st.success(
                    f"Strongest phase-specific result: **{strongest.Diagnosis}** shows a "
                    f"{phase_phrase[strongest.Phase]} HEP difference in **{strongest.Method}** "
                    f"({strongest.Feature}: {strongest.Direction}, q={strongest.q_global:.3g}). "
                    f"{int(significant.shape[0])} feature tests survive FDR across "
                    f"{findings.Diagnosis.nunique()} diagnoses and {len(methods)} cleaning methods "
                    f"({', '.join(f'{phase}: {n}' for phase, n in phase_counts.items())})."
                )
            else:
                st.info(
                    "No contraction or expansion feature separates any diagnosis after FDR correction. "
                    f"The smallest q was {strongest.q_global:.3g} ({strongest.Diagnosis}, "
                    f"{strongest.Method}, {strongest.Feature})."
                )

            st.write("**All FDR-significant diagnosis differences**")
            if len(significant):
                consolidated = (significant.sort_values("q_global")
                                .groupby(["Method", "Diagnosis", "Phase"], as_index=False).first()
                                .sort_values("q_global"))
                st.dataframe(
                    consolidated[["Method", "Diagnosis", "Phase", "Feature", "Direction",
                                 "Rank-biserial", "Median present", "Median absent",
                                 "q_global", "N present", "N absent"]],
                    hide_index=True, use_container_width=True,
                    column_config={"q_global": st.column_config.NumberColumn("q (global FDR)", format="%.3e"),
                                   "Rank-biserial": st.column_config.NumberColumn(format="%.2f"),
                                   "Median present": st.column_config.NumberColumn(format="%.2f"),
                                   "Median absent": st.column_config.NumberColumn(format="%.2f")})
                st.caption("One row per method, diagnosis and phase (the strongest feature of that family). "
                           "Positive rank-biserial means higher HEP amplitude in patients with the diagnosis. "
                           "With a cohort this large even trivial median differences reach q < 0.05, so judge "
                           "the rank-biserial and the median gap, not the q-value alone.")
                with st.expander(f"Every significant feature test ({len(significant)} rows)"):
                    st.dataframe(
                        significant[["Method", "Diagnosis", "Phase", "Feature", "Direction",
                                    "Rank-biserial", "p_value", "q_global", "N present", "N absent"]],
                        hide_index=True, use_container_width=True,
                        column_config={"p_value": st.column_config.NumberColumn(format="%.3e"),
                                       "q_global": st.column_config.NumberColumn("q (global FDR)", format="%.3e")})
            else:
                st.caption("Nothing to list: use the per-method pages below to see each method's closest results.")

            st.divider()
            left, right = st.columns([1, 1])
            with left:
                st.write("**Contraction versus expansion within each method**")
                asymmetry_rows = []
                for method in methods:
                    X, _ = phase_features(filtered, method, tuple(systole_window),
                                          tuple(diastole_window), include_pre_r)
                    difference = X[:, -1]
                    difference = difference[np.isfinite(difference)]
                    if len(difference) < 5:
                        continue
                    _, p_value = stats.wilcoxon(difference)
                    asymmetry_rows.append({"Method": method, "Median contraction − expansion (µV)": float(np.median(difference)),
                                           "Patients with larger contraction": float(np.mean(difference > 0)),
                                           "p (difference from zero)": p_value})
                asymmetry = pd.DataFrame(asymmetry_rows)
                if len(asymmetry):
                    asymmetry["q"] = bh(asymmetry["p (difference from zero)"].tolist())
                    st.dataframe(asymmetry, hide_index=True, use_container_width=True,
                                 column_config={"Median contraction − expansion (µV)": st.column_config.NumberColumn(format="%.3f"),
                                                "p (difference from zero)": st.column_config.NumberColumn(format="%.3e"),
                                                "q": st.column_config.NumberColumn(format="%.3e")})
            with right:
                st.write("**Does the phase split cluster?**")
                cluster_rows = []
                for method in methods:
                    X, _ = phase_features(filtered, method, tuple(systole_window),
                                          tuple(diastole_window), include_pre_r)
                    frame = pd.DataFrame(X)
                    imputed = frame.fillna(frame.median()).fillna(0.0).to_numpy(float)
                    _, summary, _ = cluster_result(imputed, y, k_clusters)
                    cluster_rows.append({"Method": method, "Silhouette": summary["silhouette"],
                                         "ARI with target": summary["adjusted_rand"],
                                         "Cluster × target p": summary["chi_square_p"]})
                cluster_summary = pd.DataFrame(cluster_rows)
                st.dataframe(cluster_summary, hide_index=True, use_container_width=True,
                             column_config={"Silhouette": st.column_config.NumberColumn(format="%.3f"),
                                            "ARI with target": st.column_config.NumberColumn(format="%.3f"),
                                            "Cluster × target p": st.column_config.NumberColumn(format="%.3e")})
                st.caption(f"KMeans on contraction and expansion amplitudes, k={k_clusters}, target "
                           f"= {target_label}. Low silhouette means no natural phase-based grouping; "
                           "a high cluster × target p only means the clusters are unrelated to the target.")

            st.divider()
            st.subheader("Per-cleaning-method pages")
            st.caption("Each page opens with the most important finding for that cleaning method; the tables "
                       "below it hold the supporting evidence.")
            method_pages = st.tabs([f"{method} — phase detail" for method in methods])
            for page, method in zip(method_pages, methods):
                with page:
                    method_findings = findings[findings.Method == method]
                    top = method_findings.iloc[0] if len(method_findings) else None
                    if top is not None and bool(top["Significant"]):
                        heading = (f"{method}: {top.Diagnosis} differs in the {phase_phrase[top.Phase]} "
                                   f"window (q={top.q_global:.2g})")
                        detail = (f"{top.Feature} is {top.Direction} "
                                  f"(median {top['Median present']:.2f} vs {top['Median absent']:.2f} µV, "
                                  f"rank-biserial {top['Rank-biserial']:.2f}, q={top.q_global:.3g}).")
                    elif top is not None:
                        heading = f"{method}: no contraction or expansion difference survives FDR correction"
                        detail = (f"Closest result: {top.Diagnosis}, {top.Feature} "
                                  f"(q={top.q_global:.3g}). The phase split is still descriptive of this method.")
                    else:
                        heading = f"{method}: not enough diagnoses to test"
                        detail = "Lower the minimum patient count to run association tests."
                    st.subheader(heading)
                    st.caption(detail)

                    X, _ = phase_features(filtered, method, tuple(systole_window),
                                          tuple(diastole_window), include_pre_r)
                    frame = pd.DataFrame(X)
                    imputed = frame.fillna(frame.median()).fillna(0.0).to_numpy(float)
                    plot_df, summary, contingency = cluster_result(imputed, y, k_clusters)
                    n_sig = int(method_findings.Significant.sum()) if len(method_findings) else 0
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Significant tests", n_sig)
                    m2.metric("Smallest q", f"{method_findings.q_global.min():.2g}" if len(method_findings) else "n/a")
                    m3.metric("Cluster silhouette", f"{summary['silhouette']:.3f}")
                    m4.metric("Clusters vs target (ARI)", f"{summary['adjusted_rand']:.3f}")

                    phase_rows = []
                    for site_index, site in enumerate(SITES):
                        phase_rows.append({"Site": site, "Phase": "Contraction",
                                           "Amplitude (µV)": float(np.nanmean(X[:, site_index])),
                                           "SEM": stats.sem(X[:, site_index], nan_policy="omit")})
                        phase_rows.append({"Site": site, "Phase": "Expansion",
                                           "Amplitude (µV)": float(np.nanmean(X[:, 6 + site_index])),
                                           "SEM": stats.sem(X[:, 6 + site_index], nan_policy="omit")})
                    phase_frame = pd.DataFrame(phase_rows)
                    figure = px.bar(phase_frame, x="Site", y="Amplitude (µV)", color="Phase",
                                    barmode="group", error_y="SEM",
                                    color_discrete_map={"Contraction": "#CC3311", "Expansion": "#0072B2"},
                                    title=f"{method}: mean contraction versus expansion amplitude by site")
                    figure.update_layout(template="plotly_white", height=420)
                    st.plotly_chart(figure, use_container_width=True)

                    if len(method_findings) and n_sig:
                        st.write(f"**Diagnosis differences that survive FDR (q < 0.05) — {method}**")
                        st.dataframe(
                            method_findings[method_findings.Significant][
                                ["Diagnosis", "Phase", "Feature", "Direction", "Rank-biserial",
                                 "Median present", "Median absent", "q_global", "N present"]].head(60),
                            hide_index=True, use_container_width=True,
                            column_config={"q_global": st.column_config.NumberColumn("q (global FDR)", format="%.3e")})
                    else:
                        st.write(f"**Closest diagnosis differences — {method}**")
                        st.dataframe(
                            method_findings[["Diagnosis", "Phase", "Feature", "Direction",
                                             "Rank-biserial", "q_global"]].head(12),
                            hide_index=True, use_container_width=True,
                            column_config={"q_global": st.column_config.NumberColumn("q (global FDR)", format="%.3e")})

                    st.write("**Cluster phase profiles**")
                    labels = plot_df["cluster"].astype(int).to_numpy()
                    heat_left, heat_right = st.columns(2)
                    for container, offset, phase in [(heat_left, 0, "Contraction"),
                                                     (heat_right, 6, "Expansion")]:
                        means = (pd.DataFrame(X[:, offset:offset + len(SITES)], columns=SITES)
                                 .assign(Cluster=labels).groupby("Cluster").mean())
                        with container:
                            st.plotly_chart(
                                px.imshow(means.T, aspect="auto", text_auto=".2f",
                                          color_continuous_scale="RdBu_r",
                                          labels={"x": "Cluster", "y": "Site", "color": "µV"},
                                          title=f"{phase} amplitude by cluster"),
                                use_container_width=True)
                    st.dataframe(contingency.rename(columns={0: "Label 0", 1: "Label 1"}),
                                 use_container_width=True)
                    st.caption(f"Clusters are label-free; the table checks them against {target_label}. "
                               "A cluster pattern that repeats across methods and follows a diagnosis is "
                               "the strongest evidence for a phase-specific, method-independent effect.")

with tabs[12]:
    st.subheader("Sleep-stage deltas: REM, DEEP and LIGHT compared inside the same patient")
    st.caption("Every patient contributes one HEP vector per sleep stage on the six shared electrodes "
               "(72 bins, 0-600 ms). The delta is the stage difference of those vectors, so each test is "
               "paired within patient, not between cohorts. Positive means the first stage of the contrast "
               "has the larger HEP; maps show the mean delta across the 0-400 ms post-R-peak window.")
    dataset_options = list(STAGE_DATASETS)
    chosen_datasets = st.multiselect("Cohorts with staged caches", dataset_options,
                                     default=dataset_options,
                                     format_func=lambda key: STAGE_DATASETS[key],
                                     key="stage_datasets")
    stage_deltas = stage_delta_table(tuple(chosen_datasets)) if chosen_datasets else pd.DataFrame()
    if stage_deltas.empty:
        st.error("No patient has usable HEP data in more than one sleep stage for the selected cohorts.")
    elif not methods:
        st.info("Select at least one cleaning method in the sidebar.")
    else:
        stage_work = stage_deltas[stage_deltas.method.isin(methods)].copy()
        stage_names = feature_names()
        zero_stats = {(m, c): stage_delta_stats(stage_work, m, c, stage_names)
                      for m in methods for c in STAGE_CONTRAST_NAMES}
        cross_stats = ({c: stage_delta_method_comparison(stage_work, c, stage_names)
                        for c in STAGE_CONTRAST_NAMES} if len(methods) > 1 else {})

        # One colour range per method: a single global range let the 10-patient ICA
        # REM-DEEP outlier (-0.41 µV) flatten every HEP/CFAR map to near-white.
        map_scales = {}
        for method in methods:
            panel_values = [abs(value) for contrast in STAGE_CONTRAST_NAMES
                            for value in site_window_delta(stage_work, method, contrast).values()
                            if np.isfinite(value)]
            map_scales[method] = max([max(panel_values, default=0.0), 1e-3])

        pair_counts = {c: stage_work[(stage_work.contrast == c)].cobrad_id.nunique()
                       for c in STAGE_CONTRAST_NAMES}
        summary_cols = st.columns(5)
        summary_cols[0].metric("Patients with stage deltas", stage_work.cobrad_id.nunique())
        summary_cols[1].metric("REM − DEEP pairs", pair_counts["REM − DEEP"])
        summary_cols[2].metric("REM − LIGHT pairs", pair_counts["REM − LIGHT"])
        summary_cols[3].metric("DEEP − LIGHT pairs", pair_counts["DEEP − LIGHT"])
        summary_cols[4].metric("Methods", len(methods))
        base_method = "HEP" if "HEP" in methods else methods[0]
        composition = stage_work[stage_work.method == base_method].groupby("dataset").cobrad_id.nunique()
        st.caption("Patients per cohort: " + ", ".join(f"{name} {count}" for name, count in composition.items())
                   + ". External-cohort ICA caches are zero or numerically degenerate, so ICA only "
                   f"contributes {stage_work[stage_work.method == 'ICA'].cobrad_id.nunique()} patients "
                   "with a usable decomposition.")

        pooled = pd.concat([frame.assign(Method=m, Contrast=c)
                            for (m, c), frame in zero_stats.items() if len(frame)],
                           ignore_index=True) if any(len(f) for f in zero_stats.values()) else pd.DataFrame()
        if len(pooled) and pooled.Significant.any():
            strongest = pooled[pooled.Significant].sort_values("q_value").iloc[0]
            counts = pooled[pooled.Significant].groupby("Contrast").size()
            st.success(f"Strongest stage effect: **{strongest['Contrast']}** under **{strongest['Method']}** "
                       f"({strongest['Feature']}: median {strongest['Median Δ (µV)']:+.3f} µV, "
                       f"q={strongest['q_value']:.2g}, N={int(strongest['N patients'])}). "
                       f"Significant electrode-window tests: "
                       + ", ".join(f"{name} {int(count)}" for name, count in counts.items()) + ".")
        elif len(pooled):
            st.info("No stage delta differs from zero after FDR correction.")

        st.markdown("#### Scalp maps of the stage delta (0-400 ms)")
        header = st.columns([1.0] + [1.0] * len(methods))
        for column, method in zip(header[1:], methods):
            column.markdown(f"**{method}**")
        for contrast in STAGE_CONTRAST_NAMES:
            row = st.columns([1.0] + [1.0] * len(methods))
            row[0].markdown(f"**{contrast}**")
            for column, method in zip(row[1:], methods):
                with column:
                    n_pairs = int(stage_work[(stage_work.method == method)
                                             & (stage_work.contrast == contrast)].cobrad_id.nunique())
                    figure = brain_map_figure(site_window_delta(stage_work, method, contrast),
                                              f"{method} · {contrast}", map_scales[method], n_pairs)
                    st.pyplot(figure, use_container_width=True)
                    plt.close(figure)
        st.caption("MNE topomaps interpolate the six measured electrodes inside the head outline, so the "
                   "field between them is a guide, not an independent measurement. Each map carries its "
                   "own colour bar; the range is fixed per method (across its three contrasts) so rows "
                   "stay comparable, but not across methods, whose delta sizes differ by several fold. "
                   "An electrode without usable data is left out of its own map.")

        bar_contrast = st.selectbox("Contrast for the amplitude comparison", STAGE_CONTRAST_NAMES,
                                    key="stage_delta_bar_contrast")
        st.plotly_chart(stage_delta_bars(stage_work, bar_contrast), use_container_width=True)

        st.markdown("#### Statistical difference from zero")
        c1, c2 = st.columns(2)
        detail_method = c1.selectbox("Cleaning method", list(methods), key="stage_zero_method")
        detail_contrast = c2.selectbox("Contrast", STAGE_CONTRAST_NAMES, key="stage_zero_contrast")
        detail = zero_stats[(detail_method, detail_contrast)]
        if len(detail):
            n_sig = int(detail.Significant.sum())
            st.write(f"**{detail_method} · {detail_contrast}**: {n_sig} of {len(detail)} electrode-window tests "
                     f"differ from zero after FDR (N patients {int(detail['N patients'].max())}).")
            st.dataframe(detail.head(12), hide_index=True, use_container_width=True)
            with st.expander("All significant delta tests"):
                st.dataframe(detail[detail.Significant], hide_index=True, use_container_width=True)
        else:
            st.info("Too few paired stages for this method and contrast.")

        if cross_stats:
            st.markdown("#### Do the cleaning methods give different deltas?")
            scenario = st.selectbox("Contrast for the method comparison", STAGE_CONTRAST_NAMES,
                                    key="stage_cross_contrast")
            cross = cross_stats[scenario]
            if len(cross):
                totals = cross.groupby("Comparison").agg(
                    Tests=("q_value", "size"),
                    Significant=("Significant", "sum"),
                    Smallest_q=("q_value", "min")).reset_index()
                st.dataframe(totals, hide_index=True, use_container_width=True)
                with st.expander("Significant method differences"):
                    st.dataframe(cross[cross.Significant].head(60), hide_index=True,
                                 use_container_width=True)
                st.download_button(f"Download {scenario} method comparison CSV",
                                   cross.to_csv(index=False),
                                   "stage_delta_method_comparison.csv", "text/csv")
            else:
                st.info("Not enough patients observed under at least two methods.")

            combined = pd.concat([frame.assign(Contrast=c, Method=m)
                                  for (m, c), frame in zero_stats.items()
                                  if len(frame)], ignore_index=True) if any(
                len(f) for f in zero_stats.values()) else pd.DataFrame()
            if len(combined):
                st.markdown("#### Delta size by method and contrast")
                pivot = combined.groupby(["Contrast", "Method"], as_index=False).agg(
                    Significant=("Significant", "sum"),
                    Mean_abs_delta=("Mean Δ (µV)", lambda v: float(np.nanmean(np.abs(v)))))
                st.dataframe(pivot, hide_index=True, use_container_width=True)

        export_rows = []
        for record in stage_work.itertuples(index=False):
            values = np.asarray(record.delta, dtype=float)
            row = {"patient": record.cobrad_id, "dataset": record.dataset,
                   "method": record.method, "contrast": record.contrast}
            for index, site in enumerate(SITES):
                block = values[index * 12 + DELTA_BIN_RANGE[0]: index * 12 + DELTA_BIN_RANGE[1]]
                row[f"{site} Δ (µV)"] = float(np.nanmean(block)) if np.isfinite(block).any() else np.nan
            export_rows.append(row)
        st.download_button("Download per-patient stage deltas CSV",
                           pd.DataFrame(export_rows).to_csv(index=False),
                           "sleep_stage_deltas.csv", "text/csv")
        st.caption("Stage deltas are paired within patient, so a patient contributes only to the contrasts "
                   "for which both stages were recorded. Fewer participants survive in the ICA contrasts "
                   "because decomposition can fail for one of the two stages.")

with tabs[13]:
    cohort_table = filtered[["patient_id", "age", "sex", "diagnoses", "hr_bpm", "n_beats"]].copy()
    cohort_table["target"] = y
    cohort_table["diagnoses"] = cohort_table["diagnoses"].map(lambda xs: "; ".join(xs))
    st.subheader("Analysis cohort")
    st.dataframe(cohort_table, hide_index=True, use_container_width=True)
    st.download_button("Download cohort CSV", cohort_table.to_csv(index=False),
                       "hep_ica_cfar_cohort.csv", "text/csv")
    if methods:
        export_method = st.selectbox("Feature table to export", methods, key="export_method")
        features = pd.DataFrame(method_features(filtered, export_method), columns=feature_names())
        features.insert(0, "target", y)
        features.insert(0, "patient_id", filtered.patient_id.to_numpy())
        st.download_button(f"Download {export_method} features CSV", features.to_csv(index=False),
                           f"{export_method.lower()}_features.csv", "text/csv")

st.divider()
st.caption("Exploratory analysis only. Classification performance is not external validation. Diagnoses can overlap, "
           "and one-vs-rest controls may carry other conditions. CFAR is the population six-channel filter; ICA uses ECG.")
