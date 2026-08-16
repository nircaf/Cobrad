# Methodological Overview of `23_brain_body_synchrony_dysrhythmia_dashboard.py`

This document summarizes the hypotheses, statistical tests, and reported outputs implemented in `23_brain_body_synchrony_dysrhythmia_dashboard.py`, a Streamlit dashboard testing predictions from Young A, et al. (2026), "I sync, therefore I am: brain-body synchrony in typical and disordered consciousness," *Neuroscience of Consciousness*, 2026(1), niag028 (https://doi.org/10.1093/nc/niag028), against this repo's cached HEP/sleep-stage/diagnosis data.

The script does not recompute HEP traces or diagnosis mappings — it reuses `6_hep_group_comparison.get_group_individuals` (via `individuals_cache*.pkl`, built by `17_generate_hep_cache.py`) and the loading/diagnosis machinery of `16_diagnosis_sleep_stage_comparison_dashboard.py`, imported as a module at runtime.

---

## 1. Data assembly

- `build_grouped_df` loads raw patient/stage/individual rows via `mod16.load_patient_data` (→ `get_group_individuals`), then builds a non-exclusive "By Diagnosis Category" grouping: each patient is duplicated into every one of the four target diagnosis groups (`DIAG_GROUPS`: Atrial Fibrillation, Heart Failure, Stroke / Cerebrovascular, Cognitive Impairment / Dementia) they carry, plus a reference cohort (`REF_GROUP = "Susp. Epilepsy"`, the same no-diagnosis cohort as script 16's `select_non_diagnosis_cohort`).
- `compute_amplitudes` reduces each (diagnosis_group, stage, patient) row to one scalar: the mean of the channel-averaged HEP trace (`mod16._patient_hep_trace`) inside the canonical post-R-peak window `AMP_WINDOW = (0.15, 0.5)` s, matching script 16's default `t_window`.
- `add_zscores` computes, per sleep stage, `z_i = (A_i − mean(A_ref)) / SD(A_ref)` from the reference group's own within-stage distribution, plus `abs_z_vs_ref = |z_i|`.

## 2. Hypothesis → test → output map

### H1 — signed HEP amplitude differs from reference (direction not assumed)

- **Test:** `mannwhitney_with_effect(group_amplitude, ref_amplitude, alternative="two-sided")` — two-sided Mann-Whitney U on raw `amplitude`, per (diagnosis group, stage). Also reports rank-biserial effect size `r = Z / sqrt(N)` via the normal approximation to U.
- **Where reported:** "Results: signed comparison (H1) vs |z|-deviation comparison (H2)" table — columns `Signed_p`, `Signed_r`, `Signed_mean_diff`, `Signed_q` (BH-FDR corrected). Also visually as the amplitude violin plots under "HEP amplitude distributions" (`plot_group_distributions`, group vs reference, per stage).
- **Reading it:** `Signed_q < 0.05` with a consistent-sign `Signed_mean_diff` across stages = a simple unidirectional deficit/excess in HEP amplitude for that group; `Signed_q ≥ 0.05` (null) means no naive shift in mean amplitude — but per the paper's bidirectional-coupling claim, this null does *not* rule out disturbed coupling if H2 is significant (opposite-signed individual deviations can cancel in the signed mean).

### H2 — |z|-deviation from reference is elevated even when the signed effect is null (paper's core claim)

- **Test:** `mannwhitney_with_effect(group_abs_z, ref_abs_z, alternative="greater")` — one-sided Mann-Whitney U (alternative = "greater") on `abs_z_vs_ref`, per (diagnosis group, stage).
- **Where reported:** Same results table, columns `AbsZ_p`, `AbsZ_r`, `AbsZ_q` (BH-FDR corrected, same family as H1 — see §3). Also the "H3: state-dependence heatmap" (`plot_absz_heatmap`) plots the `AbsZ_r` effect size (with `AbsZ_q` annotated) as a group × stage heatmap.
- **Reading it:** `AbsZ_q < 0.05` together with `Signed_q ≥ 0.05` in the same row is the paper's signature — atypically large deviation from the reference distribution in *either* direction ("optimal window" / bidirectional-atypical-coupling), invisible to a naive signed test. `AbsZ_q ≥ 0.05` = no detectable excess dispersion around the reference; combined with a null H1, the group's coupling is indistinguishable from reference at that window/stage.

### H3 — the group's coupling atypicality is state-dependent (light sleep / N3 / REM)

- **Test:** `compute_h3_table` runs `scipy.stats.kruskal(*stage_vals)` — Kruskal-Wallis — on `abs_z_vs_ref` across the selected sleep stages, separately per diagnosis group (requires ≥2 stages with ≥2 valid values each).
- **Where reported:** "H3: state-dependence heatmap" section — the `h3_table`/`h3_display` dataframe, columns `Group`, `H_stat`, `p_value`, `N_stages`, `q_value` (BH-FDR corrected as its own family, separate from H1/H2 — see §3). The heatmap above it (`plot_absz_heatmap`) is the per-stage breakdown that motivates the test but is not itself the H3 statistic.
- **Reading it:** `q_value < 0.05` = the magnitude of atypical coupling differs significantly across sleep stages for that group (state-dependent effect, e.g. worse in N3 than REM); `q_value ≥ 0.05` = no evidence the |z|-deviation differs by stage for that group.

## 3. Multiple-comparisons correction

`compute_results_table` pools every `Signed_p` and `AbsZ_p` across all selected (group × stage) combinations into one array and BH-FDR corrects them together (`mod16.benjamini_hochberg`) into `Signed_q`/`AbsZ_q` — i.e., H1 and H2 share one correction family. `compute_h3_table` BH-FDR corrects its per-group Kruskal-Wallis p-values (`p_value` → `q_value`) as an independent, separate family. All significance calls in the dashboard (including the auto-generated "Conclusions" section, `build_conclusions`) use `q < alpha` (default `alpha = 0.05`), not raw p-values.

## 4. Other reported elements (not separate hypotheses)

- **Math panel** (expander in the UI): displays the z-score formula, the Mann-Whitney U statistic, and the rank-biserial `r` formula used above.
- **Conclusions section:** `build_conclusions` programmatically classifies each diagnosis group from its `results_table`/`h3_table` rows into one of: "supports H1+H2 (atypical, bidirectional coupling)", "consistent unidirectional [increase/decrease]", "signed effect significant but sign varies by stage (H1+H2 bidirectionality)", or "no significant coupling effect" — each appended with the H3 state-dependence verdict. This is a summary of the tests above, not an independent statistical test.
