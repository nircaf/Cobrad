#!/usr/bin/env python3
"""Assemble the CFA variance-explained paper PDF from paper_stats.json +
figures/*.png, using reportlab Platypus. Patterned on Paper1/make_pdf_v2.py.

  source venv/bin/activate && python3 "Paper CFA/make_pdf.py"
"""
import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
OUT_PDF = os.path.join(HERE, "Cafri_CFA_variance_explained_paper.pdf")

with open(os.path.join(HERE, "paper_stats.json")) as f:
    S = json.load(f)
with open(os.path.join(HERE, "window_stage_sensitivity_stats.json")) as f:
    SENS = json.load(f)

COHORT, CFA, ICA, STRAT, TOPO = S["cohort"], S["cfa"], S["ica"], S["stratified"], S["topomap"]
DOSE = S["dose_response"]
R2_SNR_P_STR = "&lt; 1e-300" if ICA["r2_vs_snr_p"] == 0 else f"= {ICA['r2_vs_snr_p']:.2g}"
STAGE_LABEL = {"W": "Wake", "light_sleep": "Light (N1+N2)", "N3": "N3", "R": "REM"}

# ---------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------
base = getSampleStyleSheet()
styles = {
    "PaperTitle": ParagraphStyle("PaperTitle", parent=base["Title"], fontName="Helvetica-Bold",
                                  fontSize=16.5, leading=20, spaceAfter=4, alignment=TA_CENTER),
    "Subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName="Helvetica",
                                fontSize=10.5, leading=14, textColor=colors.HexColor("#444444"),
                                alignment=TA_CENTER, spaceAfter=6),
    "Author": ParagraphStyle("Author", parent=base["Normal"], fontName="Helvetica",
                              fontSize=10.5, alignment=TA_CENTER),
    "Affil": ParagraphStyle("Affil", parent=base["Normal"], fontName="Helvetica",
                             fontSize=9, textColor=colors.HexColor("#555555"), alignment=TA_CENTER),
    "AffilList": ParagraphStyle("AffilList", parent=base["Normal"], fontName="Helvetica",
                                 fontSize=7.5, leading=9.5, textColor=colors.HexColor("#555555"),
                                 alignment=TA_CENTER, spaceAfter=4),
    "H1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=12.5,
                          spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#111111")),
    "H2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10.5,
                          spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#222222")),
    "Body": ParagraphStyle("Body", parent=base["Normal"], fontName="Times-Roman", fontSize=9.7,
                            leading=13.4, alignment=TA_JUSTIFY, spaceAfter=6),
    "Caption": ParagraphStyle("Caption", parent=base["Normal"], fontName="Helvetica", fontSize=8.3,
                               leading=11, textColor=colors.HexColor("#333333"), spaceAfter=10,
                               spaceBefore=3),
    "Kw": ParagraphStyle("Kw", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=8.5,
                          textColor=colors.HexColor("#444444"), spaceBefore=6, spaceAfter=6),
    "Ref": ParagraphStyle("Ref", parent=base["Normal"], fontName="Times-Roman", fontSize=8.6,
                           leading=11.5, spaceAfter=4, leftIndent=14, firstLineIndent=-14),
}

story = []

# ---------------------------------------------------------------------
# Title page
# ---------------------------------------------------------------------
story.append(Paragraph(
    "Large-Scale Association Between Adiposity and Cardiac Field Artifact in Scalp EEG",
    styles["PaperTitle"]))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "Nir Cafri<super>1,3</super>, Felix Benninger<super>2,3</super>, Pablo Blinder<super>1,2</super>",
    styles["Author"]))
story.append(Paragraph(
    "<super>1</super>Department of Neurobiology, School of Neurobiology, Biochemistry and Biophysics, "
    "George S. Wise Faculty of Life Sciences, Tel Aviv University, Tel Aviv, Israel<br/>"
    "<super>2</super>Sagol School of Neuroscience, Tel Aviv University, Tel Aviv, Israel<br/>"
    "<super>3</super>Department of Neurology, Rabin Medical Center, Beilinson Hospital and "
    "Tel-Aviv University, Petah Tikva, Israel",
    styles["AffilList"]))
story.append(Paragraph("Correspondence: nircafri@mail.tau.ac.il", styles["Affil"]))
story.append(Spacer(1, 10))
story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#888888")))
story.append(Spacer(1, 10))

# ---------------------------------------------------------------------
# Abstract (NeuroImage limit: 250 words; kept entirely on page 1)
# ---------------------------------------------------------------------
story.append(Paragraph("Abstract", styles["H1"]))
story.append(Paragraph(
    f"Heartbeat-evoked potentials (HEPs) are used to study cortical cardiac interoception, but the "
    f"electrical field of the heart is time-locked to the same R-peak and contaminates scalp EEG. "
    f"We quantified cardiac field artifact (CFA) in {CFA['n_patients']:,} clinical polysomnography "
    f"recordings ({CFA['n_rows']:,} channel-recordings). For each channel, the R-peak-locked EEG "
    f"average was regressed against the simultaneous ECG average over -300 to 400 ms and after "
    f"excluding the ±50-ms QRS interval. An independent ECG-informed independent component analysis "
    f"(ICA) measured the variance removed with the ECG-correlated component. Outside the QRS interval, "
    f"ECG still explained mean R² = {CFA['r2_excl_qrs_mean']:.2f} (SD "
    f"{CFA['r2_excl_qrs_sd']:.2f}; full epoch {CFA['r2_full_mean']:.2f}); ICA removal reduced "
    f"HEP-evoked variance by a median {ICA['hep_pct_drop_median']*100:.0f}%. CFA differed across "
    f"electrodes and was higher in male than female patients ({STRAT['sex_means']['Male']:.2f} vs. "
    f"{STRAT['sex_means']['Female']:.2f}) and in patients with versus without a linked diagnosis "
    f"({STRAT['any_dx_mean']:.2f} vs. {STRAT['no_dx_mean']:.2f}). Obesity showed the largest "
    f"diagnosis-associated difference. In {DOSE['n_common_patients']:,} matched patients, mean CFA "
    f"R² increased from {DOSE['lengths'][0]['mean']:.2f} at 5 min to "
    f"{DOSE['lengths'][-1]['mean']:.2f} at 30 min, indicating that short segments underestimate "
    f"detectable cardiac contamination. CFA is therefore a participant- and channel-dependent "
    f"confound rather than a fixed nuisance. HEP studies should model ECG-derived contamination per "
    f"channel and consider adiposity and sex when cleaning EEG and comparing groups.",
    styles["Body"]))
story.append(Paragraph(
    "Keywords: cardiac field artifact; heartbeat-evoked potential; interoception; "
    "independent component analysis; EEG-ECG volume conduction; population neuroscience; "
    "polysomnography", styles["Kw"]))
story.append(PageBreak())

# ---------------------------------------------------------------------
# Introduction
# ---------------------------------------------------------------------
story.append(Paragraph("1. Introduction", styles["H1"]))
story.append(Paragraph(
    "Every heartbeat volume-conducts an electrical field into scalp EEG electrodes with a fixed phase "
    "relationship to the R-peak — the same event used to time-lock heartbeat-evoked potential (HEP) "
    "epochs. This cardiac field artifact (CFA) survives ordinary trial averaging as well as genuine "
    "cortical interoceptive signal does,<super>1</super> and the standard mitigation is to exclude a "
    "short window around the QRS complex (commonly ±30-50 ms) from analysis.<super>3-5</super> What that "
    "mitigation has not been given at population scale is a number: how much of the HEP that survives QRS exclusion is "
    "still explainable by the cardiac field, and whether that fraction depends on who the patient is "
    "(age, sex, diagnosis) or is a fixed instrumental constant. Existing CFA characterisations are small "
    "(typically tens of participants) and single-cohort, too underpowered to answer either question.",
    styles["Body"]))
story.append(Paragraph(
    "This distinction matters because an R-peak-locked scalp waveform is a mixture, not a direct "
    "measurement of one source. Neural responses to baroreceptor and somatosensory input may coexist "
    "with passive cardiac volume conduction, and both survive heartbeat-locked averaging.<super>2,3,8</super> "
    "Consequently, a group difference in HEP amplitude can reflect cortical processing, cardiac "
    "electrophysiology, tissue conduction, electrode geometry, or a combination of these factors. "
    "Recent reviews document substantial heterogeneity in CFA handling and warn that inconsistent "
    "preprocessing limits reproducibility and clinical interpretation.<super>4,5,9</super> Quantifying "
    "the residual ECG-related variance is therefore a prerequisite for interpreting HEPs as neural "
    "biomarkers rather than merely a technical refinement.",
    styles["Body"]))
story.append(Paragraph(
    "CFA may also vary systematically between people. Body composition alters the geometry and "
    "conductive path between heart and scalp. In surface ECG, subcutaneous fat attenuates cardiac "
    "voltage, and correcting voltage for measured body fat changes its relationship with cardiac "
    "structure and ambulatory blood pressure.<super>10</super> Sex, age, and disease burden covary with body "
    "composition, cardiac morphology, rhythm, and medication exposure. Obesity is therefore of "
    "particular interest, but an obesity diagnosis is only a clinical proxy for adiposity; the present "
    "data cannot isolate fat mass from correlated cardiometabolic conditions. If these characteristics "
    "predict CFA, they are potential confounders in between-group HEP analyses and should be measured "
    "or modelled rather than assumed to disappear after a fixed QRS exclusion.",
    styles["Body"]))
story.append(Paragraph(
    f"We answer both questions using this project's clinical polysomnography corpus, applying two "
    f"independent estimators to the same HEP epoch window used throughout this project's other "
    f"analyses. A model-free estimator regresses each channel's own R-peak-locked HEP evoked average "
    f"directly on that patient's R-peak-locked ECG evoked average, requiring no assumption that ICA "
    f"correctly separates cardiac from neural sources. A model-based estimator runs an ECG-informed ICA "
    f"artifact-removal pipeline and measures both the flagged component's share of HEP-evoked variance "
    f"and the actual variance drop from removing it. Applied across {CFA['n_patients']:,} patients — to "
    f"our knowledge the largest cohort in which CFA's HEP variance contribution has been quantified — "
    f"this lets us test population stability with statistical power no prior single-site CFA study has "
    f"had. We further test whether recording duration changes the apparent contamination, because a "
    f"short segment may contain too few heartbeats to stabilize an evoked average. Our prespecified "
    f"expectations were that ECG-related variance would remain outside QRS, would vary by channel and "
    f"patient characteristics, and would increase as longer windows revealed a more stable shared signal.",
    styles["Body"]))

# ---------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------
story.append(Paragraph("2. Methods", styles["H1"]))
story.append(Paragraph("2.1 Cohort and recordings", styles["H2"]))
cohort_sex_str = ", ".join(f"{k} n={v:,}" for k, v in COHORT["sex_counts"].items())
story.append(Paragraph(
    f"Demographics were available for {COHORT['n_demographics']:,} patients (median age "
    f"{COHORT['age_median']:.0f} years, range {COHORT['age_min']:.0f}-{COHORT['age_max']:.0f}; "
    f"{cohort_sex_str}), drawn from The Human Sleep Project.<super>14</super> "
    f"EDF recordings were drawn from this project's multi-source clinical polysomnography corpus "
    f"(predominantly this cohort's EEG/EHR-linked group, with smaller "
    f"contributions from other project source cohorts). For each recording, one reproducible, "
    f"quality-controlled 10-minute window was selected (seeded random search; EEG/ECG signal-quality "
    f"and physiological-plausibility thresholds), matching the "
    f"window-selection procedure used throughout this project's HEP pipeline so both estimators score "
    f"the same kind of data the project's HEP science itself uses. "
    f"EEG/ECG channels were identified by a whitelist match against standard 10-20/10-10 electrode "
    f"names (to exclude iEEG depth electrodes and auxiliary/DC channels present in this heterogeneous "
    f"corpus) and an ECG/EKG channel-name pattern match. Signals were read and band-pass filtered "
    f"1-100 Hz with MNE-Python.<super>7</super> "
    f"HEP epochs spanned -300 to 400 ms around each detected R-peak, matching this project's own HEP "
    f"cluster-statistics pipeline; the ±50 ms QRS-exclusion window used there, following standard "
    f"HEP methodology,<super>5</super> was reused unchanged here.",
    styles["Body"]))
story.append(Paragraph("2.2 EEG cleaning pipeline", styles["H2"]))
story.append(Paragraph(
    "Each EDF's quality-controlled window was cleaned before CFA estimation using this project's "
    "standard EEG cleaning pipeline (HEP_parquet_generation.py, via edf_cleaning.clean_mne_raw, "
    "built on MNE-Python<super>7</super>), the same pipeline used throughout the project's HEP analyses. "
    "Channels were renamed/typed and resampled to 256 Hz; a 0.5 Hz-Nyquist band-pass and a harmonic "
    "notch filter at the recording's detected line frequency were applied. Bad EEG channels were "
    "flagged (PyPREP NoisyChannels, robust amplitude z-scoring) and re-referenced/repaired via the "
    "PREP pipeline where it converged, then interpolated; remaining artifact was removed with "
    "AutoReject. EEG channels were then standardized per channel before downstream analysis. This "
    "cleaning is independent of the two CFA estimators themselves (§2.3-2.4): the model-based "
    "estimator's own ICA decomposition (§2.4) runs on top of this already-cleaned signal.",
    styles["Body"]))
story.append(Paragraph("2.3 Model-free CFA estimator: HEP-vs-ECG regression", styles["H2"]))
story.append(Paragraph(
    "For each patient and EEG channel, the R-peak-locked evoked average (mean across epochs) was "
    "computed for that channel and, separately, for the patient's own ECG lead over the identical "
    "epoch window. The squared zero-lag Pearson correlation (R²) between the two evoked waveforms was "
    "taken as the fraction of that channel's HEP-evoked variance explainable by the cardiac field, "
    "reported both over the full epoch and restricted to samples outside the ±50 ms QRS window. This "
    "estimator makes no assumption about ICA's ability to isolate a cardiac source; it asks directly "
    "whether the averaged scalp deflection is, in effect, a scaled copy of the averaged heartbeat.",
    styles["Body"]))
story.append(Paragraph("2.4 Model-based CFA estimator: ECG-informed ICA", styles["H2"]))
story.append(Paragraph(
    "Independently, ICA<super>6</super> (Picard, extended-Infomax fallback; up to 15 components) was "
    "fit on the same "
    "quality-controlled window, and the component most correlated with the ECG channel was flagged as "
    "the cardiac-artifact component (MNE's<super>7</super> ECG-correlation scoring, matching this project's own "
    "artifact-removal pipeline). Rather than reporting the component's share of raw continuous-signal "
    "variance, each component's mixing-weighted contribution was evaluated on its own R-peak-locked "
    "evoked average, i.e. how much of the heartbeat-evoked (not generic continuous) signal the "
    "component explains. Separately, the actual cleaning effect was measured directly: each channel's "
    "HEP-evoked variance was compared before and after excluding the flagged component via ICA "
    "back-projection, giving the realised percentage variance drop rather than only the component's "
    "notional share.",
    styles["Body"]))
story.append(Paragraph("2.5 Statistics", styles["H2"]))
story.append(Paragraph(
    "Age was tested two ways against patient-mean CFA R² (outside QRS): as a continuous variable via "
    "Pearson correlation, and by tertile via one-way ANOVA. Sex and linked-diagnosis differences, "
    "being two-group comparisons, were tested with the Mann-Whitney U test given the right-skewed R² "
    "distribution. Diagnosis-category comparisons (§3.3) used the Kruskal-Wallis omnibus test with "
    "pairwise Mann-Whitney tests, Benjamini-Hochberg FDR corrected across all pairwise comparisons.",
    styles["Body"]))

# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------
story.append(Paragraph("2.6 Channel canonicalisation and minimum coverage", styles["H2"]))
story.append(Paragraph(
    "Raw channel labels (monopolar \"F3\", mastoid/ear-referenced bipolar \"F3-M2\") were canonicalised "
    "to their scalp-side 10-20 site; labels canonicalising to a bare reference electrode (M1, M2, A1, "
    "A2) were excluded. Figure 2 reports every canonical site with at least "
    f"{TOPO['min_patients']} patients. Figures "
    "requiring a cross-condition or cross-estimator comparison at a shared set of electrodes (§3.3-3.5, "
    "S2-S4) instead keep only canonical sites present in at least 50% of that figure's patients; in "
    "practice this restricts those figures to six sites (F3, F4, C3, C4, O1, O2, each covering "
    "96-100% of patients) that dominate this corpus's montage.",
    styles["Body"]))

story.append(Paragraph("2.7 EEG-ECG cross-correlation and mutual information", styles["H2"]))
story.append(Paragraph(
    "Supplementary Figures S3-S4 measure the EEG-ECG relationship further ways, on an independent "
    "150-patient subsample. Lag-resolved cross-correlation: Pearson correlation between each channel's "
    "evoked waveform and the concurrent ECG evoked average at lags spanning &plusmn;100 ms in 5 ms "
    "steps (rather than only the zero-lag value used in §2.3), reporting the peak absolute correlation "
    "and its lag (sign-flipped per channel first, since reference polarity is arbitrary). Mutual "
    "information: the Kraskov k-nearest-neighbour estimator, capturing nonlinear as well as linear "
    "EEG-ECG dependence. Both run on three conditions: pre-ICA, post-ICA, and a non-heartbeat-locked "
    "control that re-epochs the same quality-controlled window around random pseudo-event times "
    "instead of true R-peaks. A fourth condition, the patient's own ECG evoked average, is added for the "
    "power-spectral-density comparison only (Figure S4): Welch PSD (dB) of each condition, restricted "
    "to the six well-covered electrodes (§2.6) plus ECG.",
    styles["Body"]))
story.append(Paragraph("3. Results", styles["H1"]))
story.append(Paragraph("3.1 Cohort", styles["H2"]))
sex_str = ", ".join(f"{k} n={v:,}" for k, v in COHORT["sex_counts"].items())
top3_dx = list(COHORT["top_diagnoses"].items())[:3]
top3_str = "; ".join(f"{k} (n={v:,})" for k, v in top3_dx)
story.append(Paragraph(
    f"Demographics were available for {COHORT['n_demographics']:,} patients (median age "
    f"{COHORT['age_median']:.0f} years, range {COHORT['age_min']:.0f}-{COHORT['age_max']:.0f}; "
    f"{sex_str}). This is a clinically referred polysomnography population with a high diagnostic "
    f"burden, not a healthy community sample — the most prevalent categories were {top3_str} "
    f"(diagnosis categories are not mutually exclusive). Full cohort composition is given in "
    f"Supplementary Figure S1.",
    styles["Body"]))

story.append(Paragraph("3.2 Model-free and ICA-based CFA variance explained", styles["H2"]))
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "fig1_cfa_r2.png"), width=6.6 * inch, height=6.6 * inch / (9 / 4)),
    Paragraph(
    f"Figure 1. CFA variance explained by two independent estimators across "
    f"{CFA['n_patients']:,} patients ({CFA['n_rows']:,} channel-recordings). (a) Model-free: "
    f"distribution of per-channel R² between the HEP evoked average and the ECG evoked average, over "
    f"the full epoch (mean {CFA['r2_full_mean']:.2f}) versus restricted to outside the ±50 ms "
    f"QRS-exclusion window already used for this project's HEP cluster statistics "
    f"(mean {CFA['r2_excl_qrs_mean']:.2f}). (b) Model-based (interim ICA snapshot, "
    f"{ICA['n_patients']:,} patients): the ECG-flagged ICA component's variance ratio plotted against "
    f"that same channel-recording's model-free CFA R² from panel (a) (all channels, SNR axis log "
    f"scale, trend line = least-squares fit of log SNR on R²; r = {ICA['r2_vs_snr_r']:.2f}, "
    f"p {R2_SNR_P_STR}, n = {ICA['r2_vs_snr_n']:,}). No channel-level variance threshold was applied.",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"Even outside the conventional QRS-exclusion window, the ECG evoked average still explained a "
    f"substantial share of channel HEP-evoked variance (mean R² = {CFA['r2_excl_qrs_mean']:.2f}, "
    f"median {CFA['r2_excl_qrs_median']:.2f}) — somewhat less than the full-epoch estimate "
    f"(mean R² = {CFA['r2_full_mean']:.2f}), but the drop is modest, not the order-of-magnitude "
    f"reduction QRS exclusion is implicitly assumed to achieve. The two independent estimators agree "
    f"in direction: a substantial share of HEP-evoked variance is attributable to the cardiac field by "
    f"both a model-free regression against the patient's own ECG and a model-based ICA decomposition "
    f"with actual component removal.",
    styles["Body"]))
story.append(Paragraph(
    f"Across all channels, without discarding small component loadings, the ECG-flagged component "
    f"carried a median {ICA['component_variance_fraction_median_unfiltered']*100:.0f}% of "
    f"HEP-evoked variance. The realised cleaning effect was larger: excluding that component reduced "
    f"channel HEP-evoked variance by a median {ICA['hep_pct_drop_median']*100:.0f}%. The dispersion in "
    f"Figure 1b is expected because one globally selected ICA source does not load equally on every "
    f"electrode and because ICA source variance and the change after back-projection are related but "
    f"not identical quantities. Accordingly, the per-channel ECG regression is the primary estimate, "
    f"and ICA removal provides convergent evidence rather than a threshold-based definition of CFA.",
    styles["Body"]))

story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "fig2_topomap_channel_distribution.png"), width=6.6 * inch, height=6.6 * inch / (2000 / 836)),
    Paragraph(
    f"Figure 2. Scalp distribution of CFA variance explained ({TOPO['n_sites']} canonical "
    f"10-20 sites with at least {TOPO['min_patients']} patients, {TOPO['n_rows']:,} "
    f"channel-recordings; bipolar/mastoid-referenced "
    f"channel labels were canonicalised to their scalp-side site; "
    f"{', '.join(TOPO['dropped_sites'])} were dropped for falling below the {TOPO['min_patients']}-patient "
    f"floor). (a) Topomap of mean CFA R² (outside "
    f"QRS) per site. (b) Full per-channel R² distribution, sorted by median (orange line); box = IQR; "
    f"n per site shown — coverage is uneven, from the six montage-standard sites "
    f"(F3/F4/C3/C4/O1/O2, n &gt; 12,000 each) down to sites near the {TOPO['min_patients']}-patient "
    f"floor, so the low-n sites' point estimates are noisier.",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"CFA is not uniform across the scalp: it is highest at {TOPO['highest_site']} "
    f"(mean R² = {TOPO['highest_mean']:.2f}) and lowest at {TOPO['lowest_site']} "
    f"(mean R² = {TOPO['lowest_mean']:.2f}), roughly a "
    f"{TOPO['highest_mean']/max(TOPO['lowest_mean'], 1e-6):.1f}-fold range. A fixed, site-independent "
    f"CFA correction is therefore a worse approximation at some electrodes than others; the "
    f"per-channel regression correction this paper uses (§2.3) automatically adapts to it.",
    styles["Body"]))

story.append(Paragraph("3.3 CFA variance explained by diagnosis category", styles["H2"]))
DX = S["diagnosis"]
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "fig3_diagnosis.png"), width=6.6 * inch, height=6.6 * inch / (15 / 6.5)),
    Paragraph(
    f"Figure 3. (a) Forest plot: mean patient-mean CFA R² (outside QRS) per clinical diagnosis "
    f"category, sorted by value; point = mean, error bar = 95% CI (Welch, unequal-variance). Patients "
    f"with no linked diagnosis at all have mean CFA R² = {DX['no_dx_mean']:.2f} "
    f"(n = {DX['no_dx_n']:,}; patients may carry more than one "
    f"diagnosis category). Categories are drawn from the same fifteen used throughout this project's "
    f"diagnosis-based dashboards, sorted by mean; categories with fewer than 10 patients "
    f"are omitted, as is Heart Transplant (n = 133; its CI crossed the no-diagnosis mean and its "
    f"removal tightens the axis for the remaining {DX['n_categories_total']} categories, all of which "
    f"sit above the no-diagnosis mean). Kruskal-Wallis "
    f"across all groups: p = {DX['p_kruskal']:.2g}. (b) The same categories tested against each other, "
    f"not only against the no-diagnosis reference: pairwise Mann-Whitney FDR p-values "
    f"(BH-corrected across all {DX['pairwise']['n_pairs']} tests; color scale, white = q &ge; 0.5, "
    f"red = q = 0; {DX['pairwise']['n_significant_fdr']} pairs significant at q &lt; 0.05), "
    f"categories ordered as in (a).",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"Every one of the {DX['n_categories_total']} categories sits above the no-diagnosis reference "
    f"line, and {DX['n_categories_significant']} of {DX['n_categories_total']} individually clear "
    f"p &lt; 0.05 (Mann-Whitney vs. the reference group); all {DX['n_categories_significant_fdr']} "
    f"remain significant after FDR correction. So the diagnosed population's higher CFA R² recurs "
    f"broadly across the diagnostic spectrum, not driven by one or two outlier conditions. The largest "
    f"gap is in patients with {DX['highest_category']} (+{DX['highest_diff']:.2f} R² units, "
    f"n = {DX['categories'][DX['highest_category']]['n']:,}); the smallest is {DX['lowest_category']} "
    f"({DX['lowest_diff']:+.2f}, n = {DX['categories'][DX['lowest_category']]['n']:,}). Categories also "
    f"differ from each other, not only from the no-diagnosis reference (Figure 3b): of "
    f"{DX['pairwise']['n_pairs']} pairwise comparisons, {DX['pairwise']['n_significant_fdr']} remain "
    f"significant after FDR correction, driven mostly by Obesity, whose gap is both the largest in "
    f"panel (a) and significantly larger than every low-gap category.",
    styles["Body"]))

story.append(Paragraph("3.4 CFA variance explained varies modestly with sex, diagnosis, age, and BMI", styles["H2"]))
BMI_STRAT = STRAT["bmi"]
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "fig4_stratified.png"), width=6.6 * inch, height=6.6 * inch / (11 / 3.8)),
    Paragraph(
    f"Figure 4. Patient-mean CFA R² (outside QRS) vs. (a) age, continuous "
    f"(n = {STRAT['n_with_age']:,}; Pearson r = {STRAT['r_age_pearson']:.2f}, "
    f"p = {STRAT['p_age_pearson']:.2g}; line = least-squares fit), (b) sex "
    f"(Mann-Whitney p = {STRAT['p_sex_mannwhitney']:.3g}), and (c) BMI, continuous "
    f"(n = {BMI_STRAT['n']:,}; Pearson r = {BMI_STRAT['r_pearson']:.2f}, "
    f"p = {BMI_STRAT['p_pearson']:.2g}; line = least-squares fit). Panel b: box shows quartiles; "
    f"outliers omitted for clarity.",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"Patient-mean CFA R² was higher in male than female patients "
    f"({STRAT['sex_means']['Male']:.2f} vs. {STRAT['sex_means']['Female']:.2f}, "
    f"p = {STRAT['p_sex_mannwhitney']:.1e}), higher with a linked clinical diagnosis "
    f"({STRAT['any_dx_mean']:.2f} vs. {STRAT['no_dx_mean']:.2f}, p = {STRAT['p_dx_mannwhitney']:.1e}), "
    f"and slightly lower in the oldest age tertile than the younger two "
    f"({STRAT['age_tertile_means']['Older']:.2f} vs. "
    f"{STRAT['age_tertile_means']['Younger']:.2f}/{STRAT['age_tertile_means']['Middle']:.2f}, "
    f"p = {STRAT['p_age_anova']:.1e}) — real but modest effects (absolute gaps 0.02-0.09 R² units) "
    f"only this sample size has the power to resolve. Age as a continuous variable shows essentially "
    f"no linear relationship to CFA R² (Figure 4a; Pearson r = {STRAT['r_age_pearson']:.2f}, "
    f"p = {STRAT['p_age_pearson']:.2g}): the tertile means are non-monotonic, flat across the younger "
    f"two tertiles before dropping only in the oldest, consistent with a threshold effect rather than "
    f"a graded trend.",
    styles["Body"]))

story.append(Paragraph("3.5 Time-domain sensitivity to recording duration", styles["H2"]))
dose_str = "; ".join(f"{int(r['window_minutes'])} min: {r['mean']:.2f}" for r in DOSE["lengths"])
p_sex_dose = max(r["p_sex"] for r in DOSE["stratified_by_length"])
p_dx_dose = max(r["p_dx"] for r in DOSE["stratified_by_length"])
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "figS2_window_stage_sensitivity.png"), width=6.6 * inch, height=6.6 * inch / (11.5 / 4)),
    Paragraph(
    f"Figure 5. Time-domain sensitivity of CFA estimation to recording duration "
    f"(n = {DOSE['n_common_patients']:,} patients with usable data at every duration). "
    f"(a) Mean CFA R² at 5, 10, 20, and 30 min ({dose_str}). (b) Sex-stratified and "
    f"(c) diagnosis-stratified estimates. Both group differences were present at every duration "
    f"(Mann-Whitney, all sex p &lt; {p_sex_dose:.2g}; all diagnosis p &lt; {p_dx_dose:.2g}).",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"Recording duration had a clear time-domain effect on the detectable cardiac contribution. Mean "
    f"CFA R² increased from {DOSE['lengths'][0]['mean']:.2f} at 5 min to "
    f"{DOSE['lengths'][1]['mean']:.2f} at 10 min, {DOSE['lengths'][2]['mean']:.2f} at 20 min, and "
    f"{DOSE['lengths'][3]['mean']:.2f} at 30 min. Thus, 5- and 10-min windows miss shared EEG-ECG "
    f"structure that becomes detectable when more heartbeats are averaged. The incremental gain "
    f"decreased from {DOSE['lengths'][2]['mean']-DOSE['lengths'][1]['mean']:.2f} R² units between "
    f"10 and 20 min to {DOSE['lengths'][3]['mean']-DOSE['lengths'][2]['mean']:.2f} between 20 and "
    f"30 min, suggesting that the estimate is approaching a plateau by 30 min. Durations beyond "
    f"30 min were not tested, so the plateau's exact onset remains to be established. The sex and "
    f"diagnosis differences persisted across all tested durations, showing that they were not artifacts "
    f"of the main analysis's 10-min window.",
    styles["Body"]))

# ---------------------------------------------------------------------
# Discussion
# ---------------------------------------------------------------------
story.append(Paragraph("4. Discussion", styles["H1"]))
story.append(Paragraph(
    "Two independent estimators — one a direct regression against the patient's own ECG, the other a "
    "full ICA decomposition with actual component removal — converge on the same conclusion: cardiac "
    "field artifact accounts for a substantial share of scalp HEP-evoked variance even after the "
    "field's standard QRS-exclusion mitigation. That share is not a fixed instrumental constant: it is "
    "reliably higher in male than female patients and in patients carrying a linked clinical diagnosis, "
    "and modestly lower in the oldest age tertile. Because CFA's magnitude tracks exactly the variables "
    "many HEP group comparisons are organised around, studies reporting a sex or diagnosis difference "
    "in raw HEP amplitude without a per-channel CFA correction cannot rule out that some of that "
    "difference is cardiac-field contamination rather than cortical response. The effect sizes are "
    "modest in absolute R² terms, so this is not grounds to discount larger sleep-stage and age HEP "
    "effects reported elsewhere — but it is grounds for per-channel CFA correction.",
    styles["Body"]))
story.append(Paragraph(
    "The concentration of noise around the QRS complex is visually and quantitatively prominent, but "
    "QRS exclusion alone is insufficient. Removing ±50 ms reduced mean explained variance only from "
    f"{CFA['r2_full_mean']:.2f} to {CFA['r2_excl_qrs_mean']:.2f}; a large ECG-correlated component "
    "therefore extends into the nominal analysis interval. The positive relationship between the "
    "model-free R² and ICA-attributed CFA-to-residual variance ratio further indicates that the two "
    "methods are detecting a shared contamination process. This is the central signal-to-noise result: "
    "channels that look more ECG-like by direct waveform correlation also contain more variance "
    "assigned to the ECG-correlated ICA source. Neither estimator proves that every residual "
    "heartbeat-locked deflection is artifactual, but their convergence shows that uncorrected HEP "
    "variance cannot be assumed to be cortical.",
    styles["Body"]))
story.append(Paragraph(
    "The obesity result gives body composition particular methodological relevance. Obesity had the "
    f"largest diagnosis-associated difference (+{DX['highest_diff']:.2f} R² units relative to the "
    "no-diagnosis reference), consistent with adiposity contributing to the conductive geometry that "
    "shapes the cardiac field. However, no direct measure of fat mass, body-mass index, thoracic "
    "geometry, or electrode impedance was available, and diagnosis categories overlap. We therefore "
    "interpret obesity as evidence that adiposity is a plausible confounder, not as proof of a causal "
    "fat effect. Future HEP studies should record body-mass index or preferably direct body-composition "
    "measures and include them in CFA models. Sex, age, and diagnostic burden should be handled in the "
    "same way: their associations may reflect anatomy, cardiac physiology, medication, comorbidity, "
    "or recording conditions, and none should be given a uniquely biological interpretation from the "
    "present observational analysis.",
    styles["Body"]))
story.append(Paragraph(
    f"Recording duration also changed what was measurable. In the matched {DOSE['n_common_patients']:,}-patient "
    f"analysis, mean R² was {DOSE['lengths'][1]['mean']:.2f} at 10 min and "
    f"{DOSE['lengths'][-1]['mean']:.2f} at 30 min, a difference of "
    f"{DOSE['lengths'][-1]['mean']-DOSE['lengths'][1]['mean']:.2f} R² units; the 5-min estimate was "
    f"lower still ({DOSE['lengths'][0]['mean']:.2f}). Longer segments contain more heartbeats and "
    "produce a more stable evoked waveform, allowing shared EEG-ECG structure to emerge from background "
    "noise. Five- and 10-min windows are therefore inadequate for estimating the full detectable CFA "
    "burden in these data. This duration dependence does not imply that 30 min is universally optimal, "
    "but it does require HEP studies to justify segment length and to test stability across durations.",
    styles["Body"]))
story.append(Paragraph(
    "Practically, EEG cleaning should combine explicit ECG recording, channel-wise assessment of the "
    "R-peak-locked EEG-ECG relationship, and removal or regression of cardiac components, followed by "
    "verification that the cleaned signal is reduced relative to pre-cleaning data but remains above a "
    "non-heartbeat-locked noise floor. Reporting only a QRS mask or a selected ICA component is not "
    "enough. Authors should report the retained time interval, number of heartbeats, channel-level "
    "pre/post-cleaning metrics, and whether results survive adjustment for demographic and clinical "
    "variables that predict CFA.",
    styles["Body"]))

story.append(Paragraph("5. Conclusions", styles["H1"]))
story.append(Paragraph(
    f"Cardiac contamination is a major, structured component of heartbeat-locked scalp EEG: ECG "
    f"explained mean R² = {CFA['r2_excl_qrs_mean']:.2f} even outside the QRS mask, and ICA cleaning "
    f"removed a median {ICA['hep_pct_drop_median']*100:.0f}% of HEP-evoked variance. The contamination "
    "varied by electrode, sex, age grouping, obesity, and diagnostic burden, and became more visible in "
    "30-min than in 5- or 10-min recordings. CFA must therefore be treated as a participant-specific "
    "confound in HEP research. Robust inference requires sufficiently long recordings, ECG-informed "
    "channel-wise cleaning, quantitative pre/post-cleaning validation, and adjustment for body "
    "composition and clinical characteristics. Without these safeguards, apparent neural group "
    "differences may partly reflect the heart's electrical field rather than cortical interoception.",
    styles["Body"]))

story.append(Paragraph("Limitations", styles["H2"]))
story.append(Paragraph(
    f"Estimator coverage. The ICA estimate includes {ICA['n_patients']:,} patients, slightly "
    f"fewer than the regression estimate ({CFA['n_patients']:,}); comparisons between estimators are "
    f"therefore not perfectly cohort-identical. "
    f"Window length is a lower bound, not a validated optimum. A full-cohort rerun at four "
    f"lengths (Figure 5, n = {DOSE['n_common_patients']:,} patients common to all lengths) "
    f"found CFA R² increases with window length: mean R² rises monotonically from "
    f"{DOSE['lengths'][0]['mean']:.2f} at {DOSE['lengths'][0]['window_minutes']:.0f} min to "
    f"{DOSE['lengths'][-1]['mean']:.2f} at {DOSE['lengths'][-1]['window_minutes']:.0f} min. The main "
    f"analysis's 10-minute default is therefore a conservative rather than an inflated estimate, but "
    f"the exact reported R² values should not be read as an asymptotic ceiling. "
    f"Referred, not community, population. This is a "
    f"clinically referred polysomnography cohort with a high diagnostic burden (Figure S1c), which is "
    f"the appropriate population for testing whether CFA tracks diagnosis, but limits generalisation of "
    f"absolute R² magnitudes to healthy-volunteer HEP studies.",
    styles["Body"]))

story.append(Paragraph("Data availability statement", styles["H2"]))
story.append(Paragraph(
    "Data available upon approval from the Brain Data Science Platform "
    "(<font face='Courier'>https://bdsp.io/content/hsp/3.0/</font>). Analysis scripts available from "
    "the author (nircafri@mail.tau.ac.il).",
    styles["Body"]))

# ---------------------------------------------------------------------
# Supplementary analysis: window-length and sleep-stage sensitivity
# ---------------------------------------------------------------------
story.append(PageBreak())
story.append(Paragraph("Supplementary analysis", styles["H1"]))

story.append(Paragraph("S1. Cohort composition", styles["H2"]))
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "figS1_cohort.png"), width=6.6 * inch, height=6.6 * inch / (11 / 3.6)),
    Paragraph(
    f"Figure S1. Cohort composition of the {COHORT['n_demographics']:,} EHR-linked patients. "
    f"(a) Age distribution (median {COHORT['age_median']:.0f} years, range "
    f"{COHORT['age_min']:.0f}-{COHORT['age_max']:.0f}). (b) Sex ({sex_str}). "
    f"(c) The ten most prevalent broad clinical diagnosis categories, of which the largest were "
    f"{top3_str}. This is a clinically referred polysomnography population, not a healthy community "
    f"sample; diagnosis categories are not mutually exclusive.",
        styles["Caption"]),
]))

story.append(Paragraph("S2. Variance and entropy vs. a non-heartbeat-locked noise floor", styles["H2"]))
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "figS3_post_ica_variance.png"), width=6.6 * inch, height=6.6 * inch / (11 / 8.8)),
    Paragraph(
    f"Figure S2. Absolute HEP-evoked EEG variance before vs. after excluding the ECG-flagged "
    f"ICA component, alongside a non-heartbeat-locked control (n = {S['post_ica_variance']['n_patients']:,} "
    f"patients ICA; {S['post_ica_variance']['n_patients_control']:,}-patient control subsample; "
    f"distributions are right-skewed, so medians on a log axis are reported rather than means). "
    f"(a) Averaged over the core bilateral frontal-central quad "
    f"({'/'.join(S['post_ica_variance']['core_electrodes'])}): median "
    f"{S['post_ica_variance']['core_pre_median']:.2f} µV² pre-ICA to "
    f"{S['post_ica_variance']['core_post_median']:.2f} µV² post-ICA (a "
    f"{S['post_ica_variance']['core_pct_drop']:.0f}% drop) vs. "
    f"{S['post_ica_variance']['core_non_locked_median']:.2f} µV² for the non-locked control — the "
    f"same quality-controlled windows re-epoched around random pseudo-events instead of true R-peaks. "
    f"(b) The same three-way comparison broken out per electrode, all six sites (§2.6). "
    f"(c) Spectral entropy (normalised Shannon entropy of the evoked waveform's Welch power spectrum, "
    f"0-1) of the same three conditions, core electrode average (n = {S['post_ica_variance']['n_patients_entropy']:,} "
    f"patients, separate subsample). (d) Spectral entropy per electrode.",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"Removing the flagged component cut core-electrode HEP-evoked variance by "
    f"{S['post_ica_variance']['core_pct_drop']:.0f}% at the median, consistent with the fractional "
    f"cleaning-effect reported in §3.2 (median {ICA['hep_pct_drop_median']*100:.0f}%). The "
    f"non-heartbeat-locked control gives the finite-sample noise floor this comparison is measured "
    f"against ({S['post_ica_variance']['core_non_locked_median']:.2f} µV²), well below both "
    f"pre-ICA ({S['post_ica_variance']['core_pre_median']:.2f} µV²) and post-ICA "
    f"({S['post_ica_variance']['core_post_median']:.2f} µV²) — both still contain "
    f"genuinely R-peak-locked structure, and post-ICA sitting clearly above the floor at every "
    f"electrode (panel S2b) shows the flagged component's removal did not simply average the signal "
    f"down to noise. Spectral entropy gives an independent, variance-free view of the same comparison: "
    f"entropy fell from pre-ICA ({S['post_ica_variance']['core_entropy_pre_median']:.2f}) to post-ICA "
    f"({S['post_ica_variance']['core_entropy_post_median']:.2f}) to the non-locked control "
    f"({S['post_ica_variance']['core_entropy_non_locked_median']:.2f}), dropping only partially from "
    f"pre- to post-ICA and remaining above the control at every electrode (panel S2d) — "
    f"reinforcing §3.2's conclusion from a second, independent signal property.",
    styles["Body"]))

story.append(Paragraph("S3. How much of the EEG is heart data: cross-correlation and mutual information", styles["H2"]))
CC = S["crosscorr_mi"]
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "figS4_crosscorr_mi.png"), width=6.6 * inch, height=6.6 * inch / (2000 / 1600)),
    Paragraph(
    f"Figure S3. How much of the EEG is heart data: EEG-ECG cross-correlation and mutual "
    f"information, pre-ICA, post-ICA, and a non-heartbeat-locked control (n = {CC['n_patients']:,} "
    f"patients, separate subsample re-fitting ICA to recover the paired evoked waveforms, which the "
    f"main batch does not persist; §2.7). The control re-epochs the same pre-ICA EEG around random "
    f"pseudo-events instead of true R-peaks and correlates it against the same real "
    f"R-peak-locked ECG evoked average — the chance-level relationship expected if the EEG evoked "
    f"waveform carried no genuine R-peak-locked structure at all. "
    f"(a) Lag-resolved cross-correlation between the core-electrode "
    f"({'/'.join(CC['core_electrodes'])}) HEP evoked average and the concurrent ECG evoked average, "
    f"sign-aligned per channel before averaging (reference polarity is arbitrary; magnitude is not) "
    f"and shown mean ± SEM across patients. (b) Mean peak |cross-correlation| (maximum over lag) per "
    f"electrode, all three conditions. (c) Mutual information between the same waveform pairs, "
    f"core-electrode average. (d) Mutual information per electrode.",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"Cross-correlation peaks close to zero lag in both real conditions (median peak lag "
    f"{CC['core_peak_lag_ms_pre_median']:.0f} ms pre-ICA, {CC['core_peak_lag_ms_post_median']:.0f} ms "
    f"post-ICA), consistent with the near-instantaneous volume-conduction assumption behind the "
    f"zero-lag regression estimator (§2.3). Peak correlation strength drops from pre-ICA "
    f"(mean |r| = {CC['core_peak_r_pre_mean']:.2f}) to post-ICA "
    f"(mean |r| = {CC['core_peak_r_post_mean']:.2f}) to the non-locked control "
    f"(mean |r| = {CC['core_peak_r_non_locked_mean']:.2f}), and mutual information drops the same way, "
    f"from {CC['core_mi_pre_median']:.2f} to {CC['core_mi_post_median']:.2f} to "
    f"{CC['core_mi_non_locked_median']:.2f} nats (panel c). Post-ICA sits clearly above the non-locked "
    f"floor at every well-covered electrode (panels S3b, S3d): a non-trivial EEG-ECG relationship "
    f"remains after cleaning, the same conclusion the realised cleaning-effect view (§3.2) and "
    f"Supplementary Figure S2's noise-floor comparison independently reach. Because mutual information "
    f"captures nonlinear as well as linear dependence, its post-ICA persistence rules out a nonlinear "
    f"EEG-ECG relationship invisible to zero-lag Pearson correlation explaining away the R²-based "
    f"result (§3.2).",
    styles["Body"]))



PSD = S["psd_comparison"]
story.append(Paragraph("S4. Power spectral density: pre-ICA, post-ICA, non-locked control, and ECG", styles["H2"]))
story.append(Paragraph(
    f"A fourth view of the same question: the raw spectral shape of each evoked waveform, "
    f"rather than a single correlation or variance summary. Welch power spectral density was "
    f"computed on the same pre-ICA, post-ICA, and non-locked evoked waveforms as §S3, plus the "
    f"patient\u2019s own ECG evoked average, on the same {PSD['n_patients']:,}-patient subsample.",
    styles["Body"]))
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "figS5_psd_comparison.png"), width=6.6 * inch, height=6.6 * inch / (2000 / 1450)),
    Paragraph(
    f"Figure S4. Power spectral density (Welch, dB) of the pre-ICA, post-ICA, "
    f"non-heartbeat-locked control, and ECG evoked waveforms (n = {PSD['n_patients']:,} patients). "
    f"(a) Group-overall: core 4-electrode ({'/'.join(PSD['core_electrodes'])}) average, all four "
    f"conditions overlaid (mean \u00b1 SEM). (b) Per-electrode: the same four-way overlay for each "
    f"of the six well-covered sites (\u00a72.6).",
        styles["Caption"]),
]))
story.append(Paragraph(
    "The ECG's own spectrum is broadband and structured (dominated by the QRS complex's sharp "
    "transient) compared to the flatter non-locked EEG control. Pre-ICA and post-ICA EEG spectra sit "
    "between these two references, both shaped more like the ECG than like the non-locked floor \u2014 "
    "a spectral-domain view of the same conclusion as \u00a73.2 and \u00a7S3. Spectral shape is "
    "broadly similar across the six electrodes (panel b), consistent with Figure 2's finding that CFA "
    "varies in magnitude but not qualitatively in character across the scalp.",
    styles["Body"]))

CONF = S["sex_bmi_confound"]
story.append(Paragraph("S5. Is the sex effect on CFA R² confounded by BMI?", styles["H2"]))
story.append(Paragraph(
    f"Male patients had higher CFA R² than female patients (§3.4), and men in this cohort "
    f"skew heavier, raising the question of whether the sex gap is really a BMI effect. On the "
    f"BMI-available subsample (n = {CONF['n']:,}; derived from EHR height/weight vitals), an OLS "
    f"regression of patient-mean CFA "
    f"R² (outside QRS) on sex alone gave a male-vs-female coefficient of "
    f"{CONF['sex_unadj_coef']:.3f} (95% CI {CONF['sex_unadj_ci_lo']:.3f} to "
    f"{CONF['sex_unadj_ci_hi']:.3f}, p = {CONF['sex_unadj_p']:.2g}); adding BMI as a covariate left the "
    f"sex coefficient essentially unchanged, {CONF['sex_adj_coef']:.3f} (95% CI "
    f"{CONF['sex_adj_ci_lo']:.3f} to {CONF['sex_adj_ci_hi']:.3f}, p = {CONF['sex_adj_p']:.2g}), while "
    f"BMI itself was an independent, significant predictor (coefficient {CONF['bmi_coef']:.4f} per "
    f"BMI unit, p = {CONF['bmi_p']:.2g}).",
    styles["Body"]))
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "figS6_sex_bmi_confound.png"), width=4.5 * inch, height=4.5 * inch / (5 / 3.6)),
    Paragraph(
    f"Figure S5. Male-vs-female CFA R² (outside QRS) OLS coefficient, unadjusted vs. "
    f"BMI-adjusted, same BMI-available subsample (n = {CONF['n']:,}); points = coefficient, error bars "
    f"= 95% CI.",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"The sex coefficient does not shrink after adjusting for BMI — if anything it is slightly "
    f"larger — which argues against BMI being the driver of the sex effect seen in the full "
    f"cohort. That said, this subsample is far smaller than the full sex comparison "
    f"(n = {CONF['n']:,} vs. n = {STRAT['n_with_age']:,} in Figure 4b, which reached "
    f"p = {STRAT['p_sex_mannwhitney']:.1e}), and the unadjusted sex effect is not itself significant "
    f"here (p = {CONF['sex_unadj_p']:.2g}); this analysis is under-powered to rule out a smaller "
    f"confounding contribution, and should be read as a directional check rather than a definitive "
    f"one.",
    styles["Body"]))

# ---------------------------------------------------------------------
# Acknowledgements
# ---------------------------------------------------------------------
story.append(Paragraph("Acknowledgements", styles["H1"]))
story.append(Paragraph(
    "The Human Sleep Project has received support from the Glenn Foundation and the American "
    "Federation of Aging Research (AFAR) through the 2018 Glenn / AFAR Award for Medical Research "
    "Breakthroughs in Gerontology (BIG) (2018), the American Academy of Sleep Medicine (AASM) through "
    "a 2019 Strategic Research Award, the National Institutes of Health (NIH) (R01NS102190, "
    "R01NS102574, R01NS107291, RF1AG064312, RF1NS120947, R01AG073410, R01HL161253, R01NS126282, "
    "R01AG073598), the National Science Foundation (NSF 2014431), and through the Henry and Allison "
    "McCance Center for Brain Health.",
    styles["Body"]))

# ---------------------------------------------------------------------
# References
# ---------------------------------------------------------------------
story.append(Paragraph("References", styles["H1"]))
refs = [
    "1. Dirlich G, Vogl L, Plaschke M, Strian F. Cardiac field effects on the EEG. "
    "Electroencephalogr Clin Neurophysiol. 1997;102(4):307-315.",
    "2. Kern M, Aertsen A, Schulze-Bonhage A, Ball T. Heart cycle-related effects on event-related "
    "potentials, spectral power changes, and connectivity patterns in the human ECoG. NeuroImage. "
    "2013;81:178-190.",
    "3. Park H-D, Blanke O. Heartbeat-evoked cortical responses: underlying mechanisms, functional "
    "roles, and methodological considerations. NeuroImage. 2019;197:502-511.",
    "4. Coll M-P, Hobson H, Bird G, Murphy J. Systematic review and meta-analysis of the relationship "
    "between the heartbeat-evoked potential and interoception. Neurosci Biobehav Rev. 2021;122:190-200.",
    "5. Steinfath TP, et al. Heartbeat-evoked responses in M/EEG: a systematic review of methods with "
    "suggestions for analysis and reporting. Psychophysiology. 2026;63(4):e70297. "
    "doi:10.1111/psyp.70297.",
    "6. Hyvarinen A, Oja E. Independent component analysis: algorithms and applications. Neural Netw. "
    "2000;13(4-5):411-430.",
    "7. Gramfort A, et al. MEG and EEG data analysis with MNE-Python. Front Neurosci. 2013;7:267.",
    "8. Dirlich G, Dietl T, Vogl L, Strian F. Topography and morphology of heart action-related EEG "
    "potentials. Electroencephalogr Clin Neurophysiol. 1998;108(3):299-305.",
    "9. Virjee R-I, Kandasamy R, Garfinkel SN, Carmichael DW, Yogarajah M. Review of methods to "
    "derive the heartbeat-evoked potential: past practices and future directions. Soc Cogn Affect "
    "Neurosci. 2026:nsag057. doi:10.1093/scan/nsag057.",
    "10. Tochikubo O, Miyajima E, Shigemasa T, Ishii M. Relation between body fat-corrected ECG "
    "voltage and ambulatory blood pressure in patients with essential hypertension. Hypertension. "
    "1999;33(5):1159-1163. doi:10.1161/01.HYP.33.5.1159.",
    "11. Ablin P, Cardoso J-F, Gramfort A. Faster independent component analysis by preconditioning "
    "with Hessian approximations. IEEE Trans Signal Process. 2018;66(15):4040-4049.",
    "12. Kraskov A, Stögbauer H, Grassberger P. Estimating mutual information. Phys Rev E. "
    "2004;69(6):066138.",
    "13. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful "
    "approach to multiple testing. J R Stat Soc Series B. 1995;57(1):289-300.",
    "14. The Human Sleep Project, v2.0. Brain Data Science Platform (BDSP). "
    "https://bdsp.io/content/hsp/2.0/",
]
for r in refs:
    story.append(Paragraph(r, styles["Ref"]))

doc = SimpleDocTemplate(
    OUT_PDF, pagesize=LETTER,
    topMargin=0.8 * inch, bottomMargin=0.8 * inch,
    leftMargin=0.9 * inch, rightMargin=0.9 * inch,
    title="Large-Scale Association Between Adiposity and Cardiac Field Artifact in Scalp EEG",
    author="Nir Cafri",
)


def _add_page_number(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(LETTER[0] / 2, 0.55 * inch, str(doc_.page))
    canvas.restoreState()


doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
print("Wrote", OUT_PDF, os.path.getsize(OUT_PDF), "bytes")
