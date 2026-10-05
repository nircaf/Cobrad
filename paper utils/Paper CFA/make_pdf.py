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
PAPERS_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "papers")
os.makedirs(PAPERS_DIR, exist_ok=True)
OUT_PDF = os.path.join(PAPERS_DIR, "Cafri_CFA_variance_explained_paper.pdf")

with open(os.path.join(HERE, "paper_stats.json")) as f:
    S = json.load(f)
with open(os.path.join(HERE, "window_stage_sensitivity_stats.json")) as f:
    SENS = json.load(f)

COHORT, CFA, ICA, STRAT, TOPO = S["cohort"], S["cfa"], S["ica"], S["stratified"], S["topomap"]
DOSE = S["dose_response"]
SRC = S["cohort"]["source_counts"]
DXS = S["stratified"]["dx_by_site"]
ZL = S["crosscorr_mi"]["zero_lag_r2_full_epoch"]
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
    "Large-Scale Associations of BMI and Clinical Obesity With Cardiac Field Artifact in Scalp EEG",
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
    f"Heartbeat-evoked potentials (HEPs) are used to study cardiac interoception; however, "
    f"the cardiac electrical field is time-locked to the same R-peak and contaminates scalp EEG. "
    f"Previous studies have characterised this cardiac field artifact (CFA) in small, single-cohort "
    f"samples and implicitly treated residual contamination after QRS exclusion as constant; yet "
    f"the extent to which CFA varies with BMI, medical conditions, sex, and age, and whether short "
    f"segments underestimate it, have not been examined. To address this gap, we quantified CFA in "
    f"{CFA['n_patients']:,} polysomnography patients ({CFA['n_rows']:,} "
    f"channel-recordings), predominantly from the Human Sleep Project, a publicly available, curated "
    f"dataset. Per channel, the R-peak-locked EEG average was regressed on the ECG "
    f"average (-300 to 400 ms), excluding a ±50 ms QRS interval. Outside "
    f"the QRS interval, the ECG explained a mean R² of {CFA['r2_excl_qrs_mean']:.2f} (full epoch: "
    f"{CFA['r2_full_mean']:.2f}), and removing the ECG-related independent component reduced HEP "
    f"variance by a median of "
    f"{ICA['hep_pct_drop_median']*100:.0f}%. CFA varied across electrodes and was greater in male "
    f"than in female patients ({STRAT['sex_means']['Male']:.2f} vs. "
    f"{STRAT['sex_means']['Female']:.2f}) and in patients with than without a linked diagnosis "
    f"({STRAT['any_dx_mean']:.2f} vs. {STRAT['no_dx_mean']:.2f}; confounded by recording site); among diagnoses, it was highest for "
    f"obesity. BMI correlated weakly with CFA (exploratory; r = {STRAT['bmi']['r_pearson']:.2f}); "
    f"age showed no linear trend. In {DOSE['n_common_patients']:,} duration-matched "
    f"patients, mean R² increased from {DOSE['lengths'][0]['mean']:.2f} at 5 min to "
    f"{DOSE['lengths'][-1]['mean']:.2f} at {DOSE['lengths'][-1]['window_minutes']:.0f} min, "
    f"suggesting that short segments underestimate contamination. CFA is a participant- and "
    f"channel-dependent confound. HEP studies should therefore model ECG-derived contamination per "
    f"channel and account for BMI, clinical obesity, and sex when comparing groups.",
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
    "The electrical field generated by each heartbeat is volume-conducted to scalp EEG electrodes "
    "with a fixed phase relationship to the R-peak, the same event used to time-lock "
    "heartbeat-evoked potential (HEP) epochs. This cardiac field artifact (CFA) is therefore "
    "preserved by trial averaging to the same degree as genuine cortical interoceptive "
    "activity.<super>1</super> The standard mitigation is to exclude a short interval around the QRS "
    "complex (typically ±30–50 ms) from analysis.<super>2-4</super> Previous studies have "
    "characterised the topography and morphology of CFA in single cohorts of a few tens of "
    "participants,<super>1,5</super> samples underpowered to detect modest between-participant "
    "differences; yet, to our knowledge, no study has quantified the proportion of HEP variance that "
    "remains shared with the ECG after QRS exclusion at the population level, nor examined whether "
    "this proportion varies with BMI, medical conditions, sex, and age, or with the length of the "
    "analysed segment. Instead, residual contamination after QRS exclusion has largely been treated "
    "as an approximately constant property of the recording.",
    styles["Body"]))
story.append(Paragraph(
    "The assumption of constant contamination matters because the R-peak-locked scalp waveform is a mixture of sources "
    "rather than a measurement of a single generator. Neural responses to baroreceptor and "
    "somatosensory input coexist with passive cardiac volume conduction, and both are preserved by "
    "heartbeat-locked averaging.<super>2,5,6</super> A group difference in HEP amplitude may "
    "therefore reflect cortical processing, cardiac electrophysiology, tissue conductivity, "
    "electrode geometry, or a combination of these factors. Recent reviews have documented "
    "considerable heterogeneity in how CFA is handled and have noted that inconsistent preprocessing "
    "limits reproducibility and clinical interpretation.<super>3,4,7</super> Quantifying the residual "
    "ECG-related variance is therefore a prerequisite for interpreting HEPs as neural biomarkers, "
    "not merely a technical refinement.",
    styles["Body"]))
story.append(Paragraph(
    "CFA may also differ systematically between individuals. Body composition alters the geometry "
    "and conductivity of the path between the heart and the scalp. In surface ECG, subcutaneous fat "
    "attenuates cardiac voltage, and correcting voltage for measured body fat alters its "
    "relationship with cardiac structure and ambulatory blood pressure.<super>8</super> Clinical "
    "obesity and BMI are therefore of particular interest, although neither directly measures "
    "body-fat distribution. Sex, age, and disease burden also covary with body composition, cardiac "
    "morphology, rhythm, and medication exposure. If body composition or these characteristics predict CFA, they constitute "
    "potential confounders in between-group HEP analyses and should be measured or modelled rather "
    "than assumed to be eliminated by a fixed QRS exclusion.",
    styles["Body"]))
story.append(Paragraph(
    f"Here, we addressed these questions in clinical polysomnography recordings drawn predominantly from "
    f"the Human Sleep Project, a publicly available, curated dataset,<super>9</super> using two "
    f"complementary estimators applied to a common HEP epoch window. A model-free estimator regresses "
    f"each channel's R-peak-locked evoked average on the same patient's R-peak-locked ECG evoked "
    f"average and therefore requires no assumption that ICA correctly separates cardiac from neural "
    f"sources. A model-based estimator applies ECG-informed ICA and quantifies both the share of "
    f"HEP variance carried by the ECG-related component and the variance reduction obtained by "
    f"removing it. To our knowledge, this {CFA['n_patients']:,}-patient cohort is the largest in which "
    f"the contribution of CFA to HEP variance has been quantified. We additionally examined whether "
    f"segment duration affects the apparent contamination, as short segments may contain too few "
    f"heartbeats to yield a stable evoked average. We hypothesised that ECG-related variance would "
    f"persist outside the QRS interval, would vary across channels and patient characteristics, and "
    f"would increase with segment length as the shared signal became more stable.",
    styles["Body"]))

# ---------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------
story.append(Paragraph("2. Methods", styles["H1"]))
story.append(Paragraph("2.1 Cohort and recordings", styles["H2"]))
story.append(Paragraph(
    f"This study is a secondary analysis of existing polysomnography recordings; no recordings were "
    f"acquired for this study. Most patients ({SRC['Harvard_Electroencephalography']:,}) were "
    f"obtained from the Human Sleep Project,<super>9</super> a publicly available, curated dataset "
    f"of clinical polysomnography recordings with linked electronic health records (EHR), "
    f"distributed through the Brain Data Science Platform. The remaining patients came from the CAP "
    f"Sleep Database<super>10,11</super> (n = {SRC['CAP_Sleep_Database']}), a sleep dataset from "
    f"[AUTHOR: Berkeley dataset source and citation] (n = {SRC['Berkeley_data']}), and clinical "
    f"polysomnography recordings from Rabin Medical Center (n = {SRC['EDF']}); because these "
    f"cohorts lack EHR linkage, they contributed only to the cohort-wide CFA estimates and not to "
    f"demographic, diagnosis, or BMI analyses. BMI was "
    f"computed from each patient's median height and weight recorded in the EHR vitals and admission "
    f"tables; values outside 10–80 kg/m² were excluded as implausible. EEG channels were identified by matching channel labels against standard "
    f"10-20/10-10 electrode names, thereby excluding intracranial depth electrodes and auxiliary/DC "
    f"channels; at least two EEG channels were required. The ECG channel was identified by label "
    f"pattern matching (ECG/EKG), and the first matching lead was used. Cohort characteristics are "
    f"reported in §3.1.",
    styles["Body"]))
story.append(Paragraph("2.2 Window selection, R-peak detection, and preprocessing", styles["H2"]))
story.append(Paragraph(
    "For each recording, a single 10-min analysis window was selected by a seeded, reproducible "
    "random search (uniformly distributed start times; up to 10 attempts). R-peaks were detected on "
    "the ECG of each candidate window before the analysis filter described below: the "
    "median-subtracted signal was band-pass "
    "filtered at 5–25 Hz (third-order zero-phase Butterworth), rectified, and peaks were identified "
    "with a minimum inter-peak distance of 300 ms and a prominence threshold of three times the "
    "median absolute deviation of the rectified signal; rectification made detection insensitive to "
    "ECG polarity. A window was accepted if (i) at least 70% of EEG "
    "channels passed signal-quality criteria (≥99.9% finite samples; standard deviation 0.1–500 µV; "
    "peak-to-peak amplitude ≤5,000 µV; &lt;20% flat samples), (ii) the mean heart rate was 35–180 "
    "beats/min, and (iii) at least 70% of R-R intervals lay between 0.33 and 2.0 s. The accepted "
    "window was then band-pass filtered between 1 Hz and min(100 Hz, Nyquist − 0.5 Hz) using a "
    "zero-phase FIR filter in MNE-Python;<super>12</super> because 86% of recordings were sampled at "
    "200 Hz, the upper edge lay just below the Nyquist frequency in most recordings. The same filter "
    "was applied to the EEG and ECG channels, and signals were analysed in µV at their native "
    "sampling rate without re-referencing, channel interpolation, or amplitude-based rejection, so "
    "that CFA was estimated in minimally processed data.",
    styles["Body"]))
story.append(Paragraph("2.3 Model-free CFA estimator: HEP-vs-ECG regression", styles["H2"]))
story.append(Paragraph(
    "HEP epochs spanned -300 to 400 ms relative to each detected R-peak; epochs extending beyond the "
    "window edges were discarded, no baseline correction was applied, and recordings with fewer than "
    "20 epochs were excluded (median 940 epochs per 10-min window). "
    "For each patient and EEG channel, the R-peak-locked evoked average (mean across epochs) was "
    "computed, together with the evoked average of the patient's ECG lead over the same epoch "
    "window. The squared zero-lag Pearson correlation (R²) between the two evoked waveforms was "
    "taken as the fraction of HEP variance in that channel shared with the ECG and used as an index "
    "of CFA. R² "
    "was computed over the full epoch and over the samples outside a ±50 ms QRS-exclusion window, as "
    "is standard in HEP analysis.<super>4</super> This estimator "
    "does not depend on the ability of ICA to isolate a cardiac source; rather, it tests directly "
    "the extent to which the averaged scalp waveform is a linearly scaled (and offset) copy of the "
    "averaged ECG. For patient-level analyses (§2.7), channel-level R² values were averaged across "
    "each patient's channels at the six well-covered sites (§2.5) and, for patients with more than "
    "one recording, across recordings (patient-mean CFA R²).",
    styles["Body"]))
story.append(Paragraph("2.4 Model-based CFA estimator: ECG-informed ICA", styles["H2"]))
story.append(Paragraph(
    "In parallel, ICA<super>13</super> (extended Picard algorithm<super>14</super>; "
    "min(15, number of EEG channels − 1) components; 500 iterations; fixed "
    "random seed) was fitted to the filtered EEG of the same 10-min window used for the regression "
    "estimator. "
    "Components whose correlation with the ECG channel exceeded the default threshold were identified "
    "as ECG-related using the correlation-based scoring implemented in "
    "MNE-Python;<super>12</super> if none did, the component with "
    "the highest absolute score was used (two components were flagged in 49 of 13,624 recordings). Rather than each component's share "
    "of continuous-signal variance, its mixing-weighted contribution to each channel was evaluated on "
    "the R-peak-locked evoked average, thereby quantifying the fraction of heartbeat-evoked signal "
    "that the component explains. In addition, the effect of artifact removal was measured directly "
    "by comparing the HEP variance of each channel before and after excluding the ECG-related "
    "component through ICA back-projection, yielding the realised percentage reduction in variance, "
    "100 × (pre - post)/pre, rather than the nominal share of the component. For Figure 1b, the "
    "ratio of ECG-related component variance to residual variance (SNR) was also computed for each "
    "channel-recording.",
    styles["Body"]))
story.append(Paragraph("2.5 Channel canonicalisation and minimum coverage", styles["H2"]))
story.append(Paragraph(
    "Channel labels were mapped to their scalp-side 10-20 site, and labels corresponding only to a "
    "reference electrode were excluded. Figure 2 includes all canonical sites recorded in at least "
    f"{TOPO['min_patients']} patients. Analyses requiring comparison across conditions or estimators "
    "at a common set of electrodes (§3.3–3.4, S2–S4) were restricted to sites present in at least "
    "50% of the patients included in that analysis. In practice, this criterion retained six sites "
    "(F3, F4, C3, C4, O1, O2; each present in 96–100% of patients), which constitute the predominant "
    "montage of the dataset.",
    styles["Body"]))

story.append(Paragraph("2.6 EEG-ECG cross-correlation and mutual information", styles["H2"]))
story.append(Paragraph(
    "The EEG-ECG relationship was further characterised in a separate subsample of 150 patients "
    f"(Supplementary Figures S3–S4; {S['crosscorr_mi']['n_patients']} with usable cross-correlation "
    "and mutual-information data). Lag-resolved cross-correlation was computed as the Pearson "
    "correlation between each channel's evoked waveform and the concurrent ECG evoked average at "
    "lags of &plusmn;100 ms in 5 ms steps, extending the zero-lag analysis of §2.3; the peak absolute "
    "correlation and its lag were reported after per-channel sign alignment, as reference polarity "
    "is arbitrary. Mutual information was estimated with the Kraskov k-nearest-neighbour "
    "estimator<super>15</super> to capture both linear and nonlinear EEG-ECG dependence. Both measures "
    "were computed for three conditions: pre-ICA, post-ICA, and a non-heartbeat-locked control in "
    "which the same window was re-epoched around an equal number of pseudo-events placed uniformly at "
    "random instead of R-peaks; in this subsample, ICA was refitted and R-peaks were re-detected on "
    "the band-pass-filtered ECG. For the "
    "power spectral density (PSD) comparison (Figure S4), the patient's ECG evoked average was added "
    "as a fourth condition, and Welch PSD (dB) was computed for each condition at the six "
    "well-covered electrodes (§2.5). Two further random subsamples supported Figure S2: in one "
    f"({S['post_ica_variance']['n_patients_control']:,} patients), the ICA windows were re-epoched "
    "around the same number of pseudo-events, without ICA refitting, to estimate a variance noise "
    f"floor; in the other ({S['post_ica_variance']['n_patients_entropy']:,} patients), ICA was "
    "refitted and the spectral entropy of each evoked waveform (Shannon entropy of its Welch power "
    "spectrum, normalised to 0–1) was computed for the pre-ICA, post-ICA, and pseudo-event conditions.",
    styles["Body"]))
story.append(Paragraph("2.7 Statistics", styles["H2"]))
story.append(Paragraph(
    "Diagnoses were assigned to 15 predefined categories by case-insensitive keyword matching on EHR "
    "diagnosis descriptions (Supplementary Table S1); a patient could belong to several categories. "
    "Patients were classified as having a linked diagnosis if any diagnosis was recorded in the EHR, "
    "and as having no linked diagnosis if they were EHR-linked but had no recorded diagnosis (a "
    "group that, owing to incomplete diagnosis linkage, included patients from a site without "
    "diagnosis tables; see Limitations); "
    "patients without an EHR link were excluded from diagnosis analyses. "
    "The association between age and patient-mean CFA R² (outside QRS) was assessed as a continuous "
    "variable using Pearson correlation and by tertile using one-way ANOVA. Because the R² "
    "distribution was right-skewed, differences by sex and by presence of a linked diagnosis were "
    "assessed with the Mann-Whitney U test. Differences among diagnosis categories (§3.3) were "
    "assessed with the Kruskal-Wallis test followed by pairwise Mann-Whitney tests, with "
    "Benjamini-Hochberg false discovery rate (FDR) correction<super>16</super> across all pairwise "
    "comparisons. The association with continuous BMI, an exploratory (not prespecified) analysis, was "
    "assessed using Pearson correlation. To retain one "
    "observation per patient, the BMI analysis used the longest available CFA window for each "
    "patient (30 min for 98% of patients), so absolute R² values in the BMI and BMI-adjusted sex "
    "analyses are not directly comparable with the 10-min estimates. BMI correlations were also "
    "estimated separately by sex, and a BMI-by-sex interaction "
    "term in an ordinary least-squares (OLS) model was used to test whether the slopes differed. A "
    "supplementary OLS model compared the sex coefficient before and after adjustment for BMI in the "
    "same subset. The relationship between model-free R² and the log-transformed ICA SNR (§2.4) "
    "was assessed with Pearson correlation across channel-recordings (Figure 1b). Sensitivity to "
    "segment duration (Supplementary §S6) was assessed in patients with usable data at all six "
    "durations (5–60 min), using all EEG channels rather than only the six well-covered sites "
    "(Figure S6a reports channel-level means), comparing sexes and diagnosis groups with "
    "Mann-Whitney tests at each "
    "duration. Windows of each length were drawn with the same seeded procedure, so that shorter "
    "windows were usually nested within longer ones; the 45- and 60-min analyses used one recording "
    "per patient. Sample sizes differ between analyses because each requires different data: the "
    f"model-free estimator included all patients with a valid window ({CFA['n_patients']:,}); the ICA "
    f"estimator, patients with a converged decomposition ({ICA['n_patients']:,}); age, sex, and "
    f"diagnosis analyses, EHR-linked patients with known age and at least one well-covered site "
    f"({STRAT['n_with_age']:,}); BMI analyses, patients with a plausible BMI "
    f"({STRAT['bmi']['n']:,}); and the duration analysis, patients with data at all six durations "
    f"({DOSE['n_common_patients']:,}).",
    styles["Body"]))
story.append(Paragraph("3. Results", styles["H1"]))
story.append(Paragraph("3.1 Cohort", styles["H2"]))
sex_str = ", ".join(f"{k} n={v:,}" for k, v in COHORT["sex_counts"].items())
top3_dx = list(COHORT["top_diagnoses"].items())[:3]
top3_str = "; ".join(f"{k} (n={v:,})" for k, v in top3_dx)
story.append(Paragraph(
    f"Of the {CFA['n_patients']:,} analysed patients, {SRC['Harvard_Electroencephalography']:,} came "
    f"from the Human Sleep Project (§2.1). Demographic data were available for {COHORT['n_age']:,} "
    f"of them (median age "
    f"{COHORT['age_median']:.0f} years, range {COHORT['age_min']:.0f}–{COHORT['age_max']:.0f}; "
    f"{sex_str}). This EHR-linked cohort consisted of clinically referred polysomnography patients with a high "
    f"diagnostic burden rather than a healthy community sample; the most prevalent diagnosis "
    f"categories were {top3_str} (categories are not mutually exclusive). Cohort composition is "
    f"shown in Supplementary Figure S1.",
    styles["Body"]))

story.append(Paragraph("3.2 Model-free and ICA-based CFA estimates", styles["H2"]))
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "fig1_cfa_r2.png"), width=6.6 * inch, height=6.6 * inch / (9 / 4)),
    Paragraph(
    f"Figure 1. Cardiac field artifact (CFA) estimates from two complementary estimators across "
    f"{CFA['n_patients']:,} patients ({CFA['n_rows']:,} channel-recordings). (a) Model-free: "
    f"distribution of per-channel R² between the HEP evoked average and the ECG evoked average, over "
    f"the full epoch (mean {CFA['r2_full_mean']:.2f}) and outside the ±50 ms QRS-exclusion window "
    f"(mean {CFA['r2_excl_qrs_mean']:.2f}). (b) Model-based estimator ({ICA['n_patients']:,} "
    f"patients): ICA SNR (ECG-related component variance relative to residual variance; §2.4) plotted against the model-free CFA "
    f"R² of the same channel-recording (all channels; logarithmic SNR axis; line, least-squares fit "
    f"of log SNR on R²; r = {ICA['r2_vs_snr_r']:.2f}, p {R2_SNR_P_STR}, "
    f"n = {ICA['r2_vs_snr_n']:,}). No channel-level variance threshold was applied.",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"Outside the conventional QRS-exclusion window, the ECG evoked average explained a substantial "
    f"proportion of HEP variance (mean R² = {CFA['r2_excl_qrs_mean']:.2f}, median "
    f"{CFA['r2_excl_qrs_median']:.2f}). This value was modestly lower than the full-epoch "
    f"estimate (mean R² = {CFA['r2_full_mean']:.2f}), indicating that QRS exclusion removed only approximately "
    f"{(1 - CFA['r2_excl_qrs_mean']/CFA['r2_full_mean'])*100:.0f}% of the ECG-explained variance. In the {ZL['non_locked']['n']}-patient "
    f"subsample with a chance-level reference (§2.6), full-epoch R² at F3/F4/C3/C4 was "
    f"{ZL['pre_ica']['mean']:.2f} for R-peak-locked averages but {ZL['non_locked']['mean']:.2f} "
    f"(median {ZL['non_locked']['median']:.2f}) for averages around random pseudo-events, so the "
    f"observed R² far exceeded chance similarity between finite-length evoked waveforms.",
    styles["Body"]))
story.append(Paragraph(
    f"Across all channels (no loading threshold applied), the ECG-related component "
    f"accounted for a median of {ICA['component_variance_fraction_median_unfiltered']*100:.0f}% of "
    f"HEP variance. The realised effect of removal was larger: excluding this component "
    f"reduced HEP variance by a median of {ICA['hep_pct_drop_median']*100:.0f}%. The "
    f"dispersion in Figure 1b is expected, because a single, globally selected ICA source does not "
    f"load equally on all electrodes, and because source variance and the change after "
    f"back-projection are related but distinct quantities. We therefore regard the per-channel ECG "
    f"regression as the primary estimate and the ICA-based removal as convergent evidence.",
    styles["Body"]))

story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "fig2_topomap_channel_distribution.png"), width=6.6 * inch, height=6.6 * inch / (2000 / 836)),
    Paragraph(
    f"Figure 2. Scalp distribution of CFA R² ({TOPO['n_sites']} canonical "
    f"10-20 sites with at least {TOPO['min_patients']} patients, {TOPO['n_rows']:,} "
    f"channel-recordings; bipolar/mastoid-referenced "
    f"channel labels were canonicalised to their scalp-side site; "
    f"{', '.join(TOPO['dropped_sites'])} were dropped for falling below the {TOPO['min_patients']}-patient "
    f"floor). (a) Topographic map of mean CFA R² (outside QRS) per site. (b) Distribution of "
    f"per-channel R² per site, sorted by median (orange line); box, interquartile range; n per site "
    f"is indicated. Coverage is uneven, ranging from the six standard montage sites "
    f"(F3/F4/C3/C4/O1/O2; n &gt; 12,000 each) to sites near the {TOPO['min_patients']}-patient "
    f"threshold, for which estimates are less precise.",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"CFA was not uniformly distributed across the scalp; it was highest at {TOPO['highest_site']} "
    f"(mean R² = {TOPO['highest_mean']:.2f}) and lowest at {TOPO['lowest_site']} "
    f"(mean R² = {TOPO['lowest_mean']:.2f}), an approximately "
    f"{TOPO['highest_mean']/max(TOPO['lowest_mean'], 1e-6):.1f}-fold range. Because sites outside the standard montage came from a minority of recordings "
    f"with different montages and references, part of this range reflects montage and referencing "
    f"rather than electrode position; within the six standard, mastoid-referenced sites, mean R² "
    f"ranged from {TOPO['site_means']['F3']:.2f} (F3) to {TOPO['site_means']['F4']:.2f} (F4). A site-independent CFA "
    f"correction would therefore be expected to perform unevenly across electrodes, which argues "
    f"for channel-level assessment.",
    styles["Body"]))

story.append(Paragraph("3.3 CFA R² by diagnosis category", styles["H2"]))
DX = S["diagnosis"]
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "fig3_diagnosis.png"), width=6.6 * inch, height=6.6 * inch / (15 / 6.5)),
    Paragraph(
    f"Figure 3. (a) Forest plot of patient-mean cardiac field artifact (CFA) R² (outside the QRS-exclusion window) per clinical diagnosis "
    f"category; point = mean, error bar = 95% CI (Welch, unequal-variance). EHR-linked patients "
    f"with no recorded diagnosis (reference; confounded by recording site, see Limitations) had "
    f"mean CFA R² = {DX['no_dx_mean']:.2f} "
    f"(n = {DX['no_dx_n']:,}); patients may belong to more than one diagnosis category. Categories "
    f"are drawn from fifteen predefined diagnosis groups and sorted by mean; categories with fewer "
    f"than 10 patients are omitted. Heart Transplant (n = 133; mean 0.31, 95% CI 0.27–0.35; "
    f"Mann-Whitney p = 0.053 vs. the reference) is also omitted to improve axis resolution. "
    f"Kruskal-Wallis test across all groups: p = {DX['p_kruskal']:.2g}. (b) Pairwise comparisons "
    f"among categories: Mann-Whitney p-values, Benjamini-Hochberg corrected across all "
    f"{DX['pairwise']['n_pairs']} tests (colour scale: white, q &ge; 0.5; red, q = 0; "
    f"{DX['pairwise']['n_significant_fdr']} pairs significant at q &lt; 0.05); categories ordered "
    f"as in (a).",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"All {DX['n_categories_total']} displayed categories lay above the no-diagnosis reference, "
    f"and all of these differences remained significant (Mann-Whitney) "
    f"after FDR correction. Because these comparisons were unadjusted for age, sex, and BMI, and the "
    f"reference group is confounded by recording site (see Limitations), they should not be "
    f"interpreted as disease effects; the category-versus-category comparisons below are not "
    f"affected by the site confound. Relative to the reference, the largest "
    f"difference was observed in patients with {DX['highest_category']} "
    f"(+{DX['highest_diff']:.2f} R² units, n = {DX['categories'][DX['highest_category']]['n']:,}) "
    f"and the smallest in {DX['lowest_category']} ({DX['lowest_diff']:+.2f}, "
    f"n = {DX['categories'][DX['lowest_category']]['n']:,}). Categories also differed from one "
    f"another (Figure 3b): of {DX['pairwise']['n_pairs']} pairwise comparisons, "
    f"{DX['pairwise']['n_significant_fdr']} remained significant after FDR correction, of which "
    f"{sum('Obesity' in (q['a'], q['b']) for q in DX['pairwise']['significant_pairs'])} involved "
    f"Obesity, which differed significantly from every other category; the remainder involved "
    f"Obstructive Sleep Apnea.",
    styles["Body"]))

story.append(Paragraph("3.4 CFA R² varies modestly with sex, age, and BMI", styles["H2"]))
BMI_STRAT = STRAT["bmi"]
BMI_WINDOW_COUNTS = ", ".join(
    f"{minutes} min: n={count:,}" for minutes, count in sorted(
        ((int(k), v) for k, v in BMI_STRAT["window_counts"].items())
    )
)
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "fig4_stratified.png"), width=6.6 * inch, height=6.6 * inch / (11 / 3.8)),
    Paragraph(
    f"Figure 4. Patient-mean CFA R² (outside QRS) vs. (a) age, continuous "
    f"(n = {STRAT['n_with_age']:,}; Pearson r = {STRAT['r_age_pearson']:.2f}, "
    f"p = {STRAT['p_age_pearson']:.2g}; line = least-squares fit), (b) sex "
    f"(Mann-Whitney p = {STRAT['p_sex_mannwhitney']:.1e}), and (c) BMI, continuous "
    f"(n = {BMI_STRAT['n']:,}; Pearson r = {BMI_STRAT['r_pearson']:.2f}, "
    f"p = {BMI_STRAT['p_pearson']:.2g}; one observation per patient, longest available window; "
    f"{BMI_WINDOW_COUNTS}; line = least-squares fit). Panel b: box shows quartiles; "
    f"outliers omitted for clarity. Panel c shows sex-specific fits: female n = "
    f"{BMI_STRAT['by_sex']['Female']['n']:,}, r = {BMI_STRAT['by_sex']['Female']['r_pearson']:.2f}, "
    f"p = {BMI_STRAT['by_sex']['Female']['p_pearson']:.1e}; male n = "
    f"{BMI_STRAT['by_sex']['Male']['n']:,}, r = {BMI_STRAT['by_sex']['Male']['r_pearson']:.2f}, "
    f"p = {BMI_STRAT['by_sex']['Male']['p_pearson']:.1e}; BMI-by-sex interaction "
    f"p = {BMI_STRAT['sex_interaction_p']:.2g}.",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"Patient-mean CFA R² was higher in male than female patients "
    f"({STRAT['sex_means']['Male']:.2f} vs. {STRAT['sex_means']['Female']:.2f}, "
    f"p = {STRAT['p_sex_mannwhitney']:.1e}), higher with a linked clinical diagnosis "
    f"({STRAT['any_dx_mean']:.2f} vs. {STRAT['no_dx_mean']:.2f}, p = {STRAT['p_dx_mannwhitney']:.1e}; confounded by recording site, see Limitations), "
    f"and slightly lower in the oldest age tertile than the younger two "
    f"({STRAT['age_tertile_means']['Older']:.2f} vs. "
    f"{STRAT['age_tertile_means']['Younger']:.2f}/{STRAT['age_tertile_means']['Middle']:.2f}, "
    f"p = {STRAT['p_age_anova']:.1e}). These effects were statistically significant but modest in "
    f"magnitude (absolute differences of 0.02–0.05 R² units for sex and age) and were detectable owing to the large "
    f"sample size. Age as a continuous variable showed no appreciable linear relationship with CFA "
    f"R² (Figure 4a; Pearson r = {STRAT['r_age_pearson']:.2f}, p = {STRAT['p_age_pearson']:.2g}); "
    f"tertile means were similar in the two younger tertiles and lower only in the oldest, a pattern "
    f"that does not indicate a graded trend. In an exploratory analysis, BMI was positively "
    f"associated with CFA R² (Figure 4c; n = "
    f"{BMI_STRAT['n']:,}, Pearson r = {BMI_STRAT['r_pearson']:.2f}, "
    f"p = {BMI_STRAT['p_pearson']:.1e}; mean BMI {BMI_STRAT['bmi_mean']:.1f} kg/m²). This association, although weak "
    f"(r² ≈ {BMI_STRAT['r_pearson']**2:.2f}), is consistent with the obesity-diagnosis comparison (§3.3). The association was positive in both female "
    f"patients (r = {BMI_STRAT['by_sex']['Female']['r_pearson']:.2f}, "
    f"p = {BMI_STRAT['by_sex']['Female']['p_pearson']:.1e}) and male patients "
    f"(r = {BMI_STRAT['by_sex']['Male']['r_pearson']:.2f}, "
    f"p = {BMI_STRAT['by_sex']['Male']['p_pearson']:.1e}). The BMI-by-sex interaction was not "
    f"significant (p = {BMI_STRAT['sex_interaction_p']:.2g}), providing no evidence that the BMI "
    f"slope differs by sex.",
    styles["Body"]))

# ---------------------------------------------------------------------
# Discussion
# ---------------------------------------------------------------------
story.append(Paragraph("4. Discussion", styles["H1"]))
story.append(Paragraph(
    "Consistent with our first hypothesis, two complementary estimators, a direct regression on the patient's ECG and an ICA decomposition "
    "with component removal, converged on the same conclusion: a substantial "
    "proportion of scalp HEP variance is shared with the ECG even after standard QRS exclusion. "
    "In line with the second hypothesis, this proportion was not a fixed property of the recording. "
    f"It varied approximately {TOPO['highest_mean']/max(TOPO['lowest_mean'], 1e-6):.1f}-fold across "
    "scalp sites (partly reflecting montage and reference differences; §3.2), extending earlier "
    "topographic descriptions in small samples<super>1,5</super> to the "
    "population level; it was higher in male than in female patients at every segment duration; and "
    "it was modestly lower in the oldest age tertile. The main-analysis difference between patients "
    "with and without a linked clinical "
    "diagnosis reversed within the site that had diagnosis records and was not "
    "reproduced in the duration-matched subset (Supplementary §S6; Limitations), so we do not "
    "interpret it further. "
    "Because the magnitude of CFA covaries with the variables around which "
    "many HEP group comparisons are organised, studies reporting sex-, BMI-, or diagnosis-related "
    "differences in HEP amplitude without channel-level CFA correction cannot exclude the "
    "possibility that part of the difference reflects cardiac-field contamination rather than "
    "cortical activity. The effect sizes are modest in absolute R² terms and do not by themselves "
    "invalidate larger HEP effects reported previously; they do, however, "
    "support routine channel-level CFA correction.",
    styles["Body"]))
story.append(Paragraph(
    "Although the artifact is most prominent around the QRS complex, the conventional ±30–50 ms "
    "exclusion<super>2-4</super> did not remove it: excluding ±50 ms reduced mean ECG-shared variance (R²) "
    f"only from {CFA['r2_full_mean']:.2f} to {CFA['r2_excl_qrs_mean']:.2f}, indicating that a "
    "substantial ECG-correlated component extends into the nominal analysis interval. "
    "The positive relationship "
    "between the model-free R² and the ICA-derived SNR suggests "
    "that both methods detect a common contamination process; channels whose waveforms "
    "more closely resemble the ECG also contain more variance assigned to the ECG-related ICA "
    "source. Neither estimator establishes that every residual heartbeat-locked deflection is "
    "artifactual, and genuine cortical heartbeat-evoked activity is well documented.<super>2,6</super> "
    "Nevertheless, the convergence of the two estimators indicates that uncorrected HEP variance "
    "cannot be assumed to be of cortical origin.",
    styles["Body"]))
story.append(Paragraph(
    "Clinically obese patients showed the "
    f"highest mean CFA R² of all diagnosis categories and differed significantly from every other "
    "category (§3.3), consistent with a contribution of BMI or obesity-related "
    "characteristics to the conductive geometry that shapes the cardiac field. The "
    f"exploratory continuous-BMI analysis points in the same direction ({BMI_STRAT['n']:,} patients; "
    f"r = {BMI_STRAT['r_pearson']:.2f}). Because R² indexes waveform similarity rather than "
    "amplitude, these findings do not conflict with the attenuation of surface ECG voltage by "
    "subcutaneous fat;<super>8</super> the underlying mechanism cannot be determined from the present data. The BMI–CFA association was numerically stronger in women "
    f"than in men (r = {BMI_STRAT['by_sex']['Female']['r_pearson']:.2f} vs. "
    f"{BMI_STRAT['by_sex']['Male']['r_pearson']:.2f}); however, the BMI-by-sex interaction was not "
    f"significant (p = {BMI_STRAT['sex_interaction_p']:.2g}), and this difference should not be "
    "interpreted as a sex-specific effect without replication. The sex difference itself persisted "
    "after adjustment for BMI (Supplementary §S5) and is therefore unlikely to be explained by BMI.",
    styles["Body"]))
story.append(Paragraph(
    "However, BMI is an anthropometric index rather "
    "than a direct measure of adiposity; no direct measures of fat mass, thoracic geometry, or "
    "electrode impedance were available, and diagnosis categories overlap. We therefore interpret "
    "these findings as evidence that BMI and clinical obesity are plausible confounders, not as "
    "evidence of a causal effect of body fat. Likewise, the associations of sex, age, and diagnosis category "
    "with CFA may reflect anatomy, cardiac "
    "physiology, medication, comorbidity, or recording conditions, and none can be assigned a "
    "specific biological interpretation on the basis of the present observational analysis.",
    styles["Body"]))
story.append(Paragraph(
    f"Segment duration also affected the detectable contamination. In the duration-matched {DOSE['n_common_patients']:,}-patient "
    f"analysis, mean R² rose from {DOSE['lengths'][0]['mean']:.2f} at 5 min and "
    f"{DOSE['lengths'][1]['mean']:.2f} at 10 min to {DOSE['lengths'][-1]['mean']:.2f} at "
    f"{DOSE['lengths'][-1]['window_minutes']:.0f} min; the increase was already present between 5 and "
    "30 min, where channel composition was nearly constant (§S6). Longer segments contain more heartbeats and yield a "
    "more stable evoked waveform, which presumably allows shared EEG-ECG structure to emerge from "
    "background noise; this pattern is consistent with our third hypothesis. "
    "Windows of 5 and 10 min, including the 10-min windows of the main analysis, therefore probably "
    "underestimate the detectable CFA burden in these data (see Limitations). This duration "
    "dependence does not identify a universally optimal segment length, but it indicates that HEP "
    "studies should justify the chosen segment length and assess the stability of their results "
    "across durations. In addition, because R² depends on the number of averaged heartbeats, "
    "between-group differences in heart rate could contribute to between-group differences in R²; "
    "this was not examined in the present analysis.",
    styles["Body"]))
story.append(Paragraph(
    "In practice, EEG preprocessing for HEP analysis should combine concurrent ECG recording, "
    "channel-level assessment of the R-peak-locked EEG-ECG relationship, and removal or regression "
    "of ECG-related components, followed by verification that heartbeat-locked variance in the "
    "cleaned signal is reduced relative to the uncleaned data while remaining above a non-heartbeat-locked noise floor. Reporting only the "
    "QRS mask or the selected ICA component is insufficient. We recommend that studies report the "
    "analysis interval, the number of heartbeats, channel-level metrics before and after cleaning, "
    "and whether results are robust to adjustment for demographic and clinical variables that "
    "predict CFA, including BMI or, preferably, direct body-composition measures. These "
    "recommendations extend recent methodological reviews of HEP analysis and reporting.<super>4,7</super>",
    styles["Body"]))

story.append(Paragraph("5. Conclusions", styles["H1"]))
story.append(Paragraph(
    f"Cardiac contamination is a substantial and systematic component of heartbeat-locked scalp EEG: the ECG "
    f"explained a mean R² of {CFA['r2_excl_qrs_mean']:.2f} outside the QRS-exclusion window, and ICA-based "
    f"removal of the ECG-related component reduced HEP variance by a median of {ICA['hep_pct_drop_median']*100:.0f}%. The "
    "contamination varied with electrode, sex, obesity, and BMI, but only weakly with age; its "
    "apparent association with diagnostic burden was confounded by recording site. Estimated CFA "
    "also increased with segment duration up to the longest window examined (60 min). CFA should therefore be treated as a "
    "participant- and channel-dependent confound in HEP research. Robust inference requires "
    "ECG-informed channel-level cleaning with quantitative before-and-after validation, justified "
    "segment lengths checked for stability across durations, and adjustment for BMI (ideally direct body-composition measures) and clinical characteristics. Without these "
    "safeguards, apparent group differences in neural responses may partly reflect the electrical "
    "field of the heart rather than cortical interoceptive processing.",
    styles["Body"]))

story.append(Paragraph("Limitations", styles["H2"]))
story.append(Paragraph(
    f"Several limitations should be noted. First, diagnosis records were available for only part "
    f"of the Human Sleep Project, so the no-diagnosis reference group consisted largely of patients "
    f"from a site without linked diagnosis tables (mean R² {DXS['I0003_no_dx']['mean']:.2f}, "
    f"n = {DXS['I0003_no_dx']['n']}). Within the site with diagnosis records, patients without a "
    f"recorded diagnosis had higher R² than those with one ({DXS['I0002_no_dx']['mean']:.2f}, "
    f"n = {DXS['I0002_no_dx']['n']}, vs. {DXS['I0002_any_dx']['mean']:.2f}, "
    f"n = {DXS['I0002_any_dx']['n']:,}). The comparison of patients with and without a linked "
    f"diagnosis is therefore confounded by recording site and should not be interpreted as a disease "
    f"effect; comparisons among diagnosis categories are not affected by this confound. Second, the "
    f"detected heart rate was high for sleep recordings (median "
    f"{CFA['bpm_quartiles']['0.5']:.0f} beats/min, interquartile range "
    f"{CFA['bpm_quartiles']['0.25']:.0f}–{CFA['bpm_quartiles']['0.75']:.0f}); the custom R-peak "
    f"detector was not validated against an established algorithm, and occasional detection of T "
    f"waves would mix T-wave-locked epochs into the averages. Third, R² does not distinguish "
    f"volume-conducted CFA from neural activity whose time course resembles the ECG, and epochs "
    f"ended at 400 ms after the R-peak, so the estimates do not cover later HEP intervals that "
    f"overlap the T wave. Fourth, CFA was estimated in the native recording references, "
    f"predominantly contralateral-mastoid derivations (e.g., F3-M2, F4-M1); because the cardiac "
    f"field projects differently onto each reference electrode, site-level values and hemispheric "
    f"asymmetries may not generalise to average-referenced HEP data.",
    styles["Body"]))
story.append(Paragraph(
    f"Fifth, 10-min estimates are likely lower bounds with respect to segment duration, because mean "
    f"CFA R² increased monotonically with duration, including from 5 to 30 min, where channel "
    f"composition was nearly constant (§4; Supplementary §S6). Sixth, the chance-level "
    f"(pseudo-event) R² was estimated only in a subsample of {ZL['non_locked']['n']} patients and "
    f"only over the full epoch (§3.2); a null for the outside-QRS estimate was not computed. "
    f"Seventh, windows were not selected by sleep stage, so vigilance state was not controlled and "
    f"may differ between groups and segment durations. Eighth, the cohort was predominantly a "
    f"clinically referred population with a high diagnostic burden (Figure S1c), which limits the "
    f"generalisability of absolute R² values to healthy volunteers; underweight patients were also "
    f"under-represented ({BMI_STRAT['underweight_n']:,} of {BMI_STRAT['n']:,} patients with "
    f"measured BMI had a BMI &lt;18.5 kg/m²). Finally, because this was a secondary analysis of "
    f"existing datasets, recording parameters and clinical annotations could not be controlled by "
    f"the authors.",
    styles["Body"]))

story.append(Paragraph("Ethics statement", styles["H2"]))
story.append(Paragraph(
    "Human Sleep Project data were de-identified under the HIPAA Safe Harbor standard and made "
    "available under an approved institutional review board protocol (IRB protocol #2022P000417) "
    "with a waiver of informed consent; they were accessed under the Brain Data Science Platform "
    "data use agreement. CAP Sleep Database recordings are publicly available, de-identified data "
    "distributed through PhysioNet. [AUTHOR: ethics approval for the Berkeley dataset.] Use of the "
    "Rabin Medical Center recordings was approved by the institutional Helsinki committee "
    "[AUTHOR: approval number]. No attempt was made to re-identify participants.",
    styles["Body"]))

story.append(Paragraph("Data availability statement", styles["H2"]))
story.append(Paragraph(
    "The Human Sleep Project dataset is publicly available, subject to a data use agreement, from "
    "the Brain Data Science Platform (<font face='Courier'>https://bdsp.io/content/hsp/</font>), "
    "and the CAP Sleep Database is publicly available from PhysioNet. The Rabin Medical Center and "
    "Berkeley recordings are not publicly available because of privacy restrictions. Analysis code is available from "
    "the corresponding author upon reasonable request (nircafri@mail.tau.ac.il).",
    styles["Body"]))

story.append(Paragraph("Funding", styles["H2"]))
story.append(Paragraph(
    "This research received no specific grant from any funding agency in the public, commercial, "
    "or not-for-profit sectors.",
    styles["Body"]))

story.append(Paragraph("Declaration of competing interests", styles["H2"]))
story.append(Paragraph(
    "The authors declare no competing interests.",
    styles["Body"]))

story.append(Paragraph("Author contributions (CRediT)", styles["H2"]))
story.append(Paragraph(
    "Nir Cafri: Conceptualization, Methodology, Software, Formal analysis, Data curation, "
    "Visualization, Writing – original draft. Felix Benninger: Conceptualization, Supervision, "
    "Writing – review &amp; editing. Pablo Blinder: Conceptualization, Supervision, "
    "Writing – review &amp; editing.",
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
    f"Figure S1. Cohort composition of the {COHORT['n_demographics']:,} Human Sleep Project patients. "
    f"(a) Age distribution (median {COHORT['age_median']:.0f} years, range "
    f"{COHORT['age_min']:.0f}–{COHORT['age_max']:.0f}). (b) Sex ({sex_str}). "
    f"(c) The ten most prevalent clinical diagnosis categories, the largest of which were "
    f"{top3_str}. The cohort is a clinically referred polysomnography population rather than a "
    f"healthy community sample; diagnosis categories are not mutually exclusive.",
        styles["Caption"]),
]))
DX_RULES = [  # mirrors DIAG_CATEGORIES in build_dataset.py
    ("Obstructive Sleep Apnea", "sleep apnea"), ("Hypertension", "hypertension"),
    ("Diabetes", "diabetes"), ("Heart Failure", "heart failure"),
    ("Coronary Artery Disease", "coronary; atherosclerotic heart"),
    ("Heart Transplant", "heart transplant; transplant status"),
    ("Atrial Fibrillation", "atrial fibrillation; atrial flutter"), ("COPD", "chronic obstructive"),
    ("Obesity", "obesity; obese"), ("Depression", "depressive; depression"), ("Anxiety", "anxiety"),
    ("Cognitive Impairment / Dementia", "cognitive; alzheimer; dementia"),
    ("Stroke / Cerebrovascular", "infarction; cerebrovascular; stroke; hemorrhage"),
    ("Kidney Disease", "kidney; renal"), ("Anemia", "anemia"),
]
dx_table = Table(
    [[Paragraph("Category", styles["Caption"]), Paragraph("Keywords (any match)", styles["Caption"])]]
    + [[Paragraph(c, styles["Caption"]), Paragraph(k, styles["Caption"])] for c, k in DX_RULES],
    colWidths=[2.4 * inch, 4.2 * inch])
dx_table.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                              ("VALIGN", (0, 0), (-1, -1), "TOP")]))
story.append(KeepTogether([
    Paragraph(
        "Table S1. Diagnosis categories and the case-insensitive keywords matched against EHR "
        "diagnosis descriptions. Keyword matching is inclusive; for example, \"infarction\" also "
        "matches myocardial infarction.",
        styles["Caption"]),
    dx_table,
]))

story.append(Paragraph("S2. Variance and entropy vs. a non-heartbeat-locked noise floor", styles["H2"]))
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "figS3_post_ica_variance.png"), width=6.6 * inch, height=6.6 * inch / (11 / 8.8)),
    Paragraph(
    f"Figure S2. Absolute HEP variance before vs. after excluding the ECG-related "
    f"ICA component, alongside a non-heartbeat-locked control (n = {S['post_ica_variance']['n_patients']:,} "
    f"ICA patients with at least one well-covered site; "
    f"{S['post_ica_variance']['n_patients_control']:,}-patient control subsample, §2.6; "
    f"distributions are right-skewed, so medians on a log axis are reported rather than means). "
    f"(a) Pooled over channels at the four core frontal-central electrodes "
    f"({'/'.join(S['post_ica_variance']['core_electrodes'])}): median "
    f"{S['post_ica_variance']['core_pre_median']:.2f} µV² pre-ICA to "
    f"{S['post_ica_variance']['core_post_median']:.2f} µV² post-ICA (a "
    f"{S['post_ica_variance']['core_pct_drop']:.0f}% drop) vs. "
    f"{S['post_ica_variance']['core_non_locked_median']:.2f} µV² for the non-locked control (the "
    f"same windows re-epoched around random pseudo-events instead of R-peaks). "
    f"(b) The same three-way comparison broken out per electrode, all six sites (§2.5). "
    f"(c) Spectral entropy (normalised Shannon entropy of the evoked waveform's Welch power spectrum, "
    f"0–1) of the same three conditions, core electrode average (n = {S['post_ica_variance']['n_patients_entropy']:,} "
    f"patients, separate subsample). (d) Spectral entropy per electrode.",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"Removal of the ECG-related component reduced median core-electrode HEP variance by "
    f"{S['post_ica_variance']['core_pct_drop']:.0f}%; this ratio of group medians is larger than, and not "
    f"directly comparable to, the median per-channel reduction reported in §3.2 "
    f"({ICA['hep_pct_drop_median']*100:.0f}%). The non-heartbeat-locked control defines the "
    f"finite-sample noise floor ({S['post_ica_variance']['core_non_locked_median']:.2f} µV²), which "
    f"was well below both the pre-ICA ({S['post_ica_variance']['core_pre_median']:.2f} µV²) and "
    f"post-ICA ({S['post_ica_variance']['core_post_median']:.2f} µV²) values. Both conditions "
    f"therefore retained R-peak-locked structure; post-ICA variance remained above the noise floor "
    f"at every electrode (panel S2b), indicating that component removal did not reduce the "
    f"heartbeat-locked signal to noise level. Spectral entropy provided a complementary, variance-independent measure: "
    f"entropy decreased from pre-ICA ({S['post_ica_variance']['core_entropy_pre_median']:.2f}) to "
    f"post-ICA ({S['post_ica_variance']['core_entropy_post_median']:.2f}) to the non-locked control "
    f"({S['post_ica_variance']['core_entropy_non_locked_median']:.2f}); the post-ICA decrease was "
    f"small, and post-ICA values exceeded the control at every electrode (panel S2d).",
    styles["Body"]))

story.append(Paragraph("S3. EEG-ECG cross-correlation and mutual information", styles["H2"]))
CC = S["crosscorr_mi"]
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "figS4_crosscorr_mi.png"), width=6.6 * inch, height=6.6 * inch / (2000 / 1600)),
    Paragraph(
    f"Figure S3. EEG-ECG cross-correlation and mutual information for pre-ICA, post-ICA, and "
    f"non-heartbeat-locked control conditions (n = {CC['n_patients']:,} patients; separate "
    f"subsample in which ICA was refitted to obtain paired evoked waveforms; §2.6). In the control "
    f"condition, the same pre-ICA EEG was re-epoched around random pseudo-events and correlated with "
    f"the R-peak-locked ECG evoked average, providing the chance-level relationship expected in the "
    f"absence of R-peak-locked structure. (a) Lag-resolved cross-correlation between the "
    f"core-electrode ({'/'.join(CC['core_electrodes'])}) HEP evoked average and the concurrent ECG "
    f"evoked average, sign-aligned per channel before averaging; mean ± SEM across patients. "
    f"(b) Mean peak |cross-correlation| (maximum over lags) per electrode for all three conditions. "
    f"(c) Mutual information between the same waveform pairs, core-electrode average. (d) Mutual "
    f"information per electrode.",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"Cross-correlation peaked near zero lag in both heartbeat-locked conditions (median peak lag "
    f"{CC['core_peak_lag_ms_pre_median']:.0f} ms pre-ICA and {CC['core_peak_lag_ms_post_median']:.0f} "
    f"ms post-ICA), consistent with the near-instantaneous volume conduction assumed by the zero-lag "
    f"regression estimator (§2.3). Peak correlation decreased from pre-ICA "
    f"(mean |r| = {CC['core_peak_r_pre_mean']:.2f}) to post-ICA "
    f"(mean |r| = {CC['core_peak_r_post_mean']:.2f}) to the non-locked control "
    f"(mean |r| = {CC['core_peak_r_non_locked_mean']:.2f}), and mutual information decreased "
    f"correspondingly, from {CC['core_mi_pre_median']:.2f} to {CC['core_mi_post_median']:.2f} to "
    f"{CC['core_mi_non_locked_median']:.2f} nats (panel c). Post-ICA values remained above the "
    f"non-locked floor at every well-covered electrode (panels S3b, S3d), indicating that a "
    f"non-trivial EEG-ECG relationship persists after cleaning, in agreement with §3.2 and "
    f"Supplementary Figure S2. Mutual information, which is also sensitive to nonlinear dependence, "
    f"followed the same ordering as peak correlation, giving no indication of substantial EEG-ECG "
    f"dependence beyond that captured by the linear measures of §3.2.",
    styles["Body"]))



PSD = S["psd_comparison"]
story.append(Paragraph("S4. Power spectral density: pre-ICA, post-ICA, non-locked control, and ECG", styles["H2"]))
story.append(Paragraph(
    f"To complement the correlation- and variance-based summaries, we compared the spectral content "
    f"of each evoked waveform. Welch power spectral density was computed for the pre-ICA, post-ICA, "
    f"and non-locked evoked waveforms described in §S3 and for the patient’s ECG evoked "
    f"average, in the same {PSD['n_patients']:,}-patient subsample.",
    styles["Body"]))
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "figS5_psd_comparison.png"), width=6.6 * inch, height=6.6 * inch / (2000 / 1450)),
    Paragraph(
    f"Figure S4. Power spectral density (Welch, dB) of the pre-ICA, post-ICA, "
    f"non-heartbeat-locked control, and ECG evoked waveforms (n = {PSD['n_patients']:,} patients). "
    f"(a) Group average: core four-electrode ({'/'.join(PSD['core_electrodes'])}) average, all four "
    f"conditions overlaid (mean \u00b1 SEM). (b) Per-electrode: the same four-way overlay for each "
    f"of the six well-covered sites (\u00a72.5).",
        styles["Caption"]),
]))
story.append(Paragraph(
    "The ECG spectrum was broadband and structured, dominated by the sharp QRS transient, whereas "
    "the non-locked EEG control lacked this QRS-dominated structure. Pre-ICA and post-ICA EEG spectra lay between these two "
    "references and appeared visually closer to the ECG than the non-locked floor did, providing "
    "spectral-domain support for the conclusions of \u00a73.2 and \u00a7S3. Spectral shape was "
    "visually similar across the six electrodes (panel b), suggesting that CFA varies across the "
    "scalp mainly in magnitude (Figure 2) rather than in spectral content.",
    styles["Body"]))

CONF = S["sex_bmi_confound"]
story.append(Paragraph("S5. Adjustment of the sex effect for BMI", styles["H2"]))
story.append(Paragraph(
    f"Male patients had higher CFA R² than female patients (§3.4), whereas mean BMI was higher in "
    f"female patients ({BMI_STRAT['by_sex']['Female']['bmi_mean']:.1f} vs. "
    f"{BMI_STRAT['by_sex']['Male']['bmi_mean']:.1f} kg/m²); because BMI was itself associated with "
    f"CFA, the sex comparison could be confounded by BMI. In the "
    f"subsample with available BMI (n = {CONF['n']:,}), an OLS regression of patient-mean CFA R² "
    f"(outside QRS) on sex alone yielded a male-versus-female coefficient of "
    f"{CONF['sex_unadj_coef']:.3f} (95% CI {CONF['sex_unadj_ci_lo']:.3f} to "
    f"{CONF['sex_unadj_ci_hi']:.3f}, p = {CONF['sex_unadj_p']:.2g}). Inclusion of BMI as a covariate "
    f"did not attenuate the sex coefficient ({CONF['sex_adj_coef']:.3f}; 95% CI "
    f"{CONF['sex_adj_ci_lo']:.3f} to {CONF['sex_adj_ci_hi']:.3f}, p = {CONF['sex_adj_p']:.2g}), while "
    f"BMI itself was an independent, significant predictor (coefficient {CONF['bmi_coef']:.4f} per "
    f"BMI unit, p = {CONF['bmi_p']:.2g}).",
    styles["Body"]))
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "figS6_sex_bmi_confound.png"), width=4.5 * inch, height=4.5 * inch / (5 / 3.6)),
    Paragraph(
    f"Figure S5. Male-versus-female CFA R² (outside QRS) OLS coefficient, unadjusted vs. "
    f"BMI-adjusted, same BMI-available subsample (n = {CONF['n']:,}); points = coefficient, error bars "
    f"= 95% CI.",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"The sex coefficient did not decrease after adjustment for BMI and, if anything, increased "
    f"slightly, as expected given the higher BMI of female patients, arguing against BMI as the explanation for the sex effect observed in the full "
    f"cohort. In this observational model, sex and BMI were thus "
    f"independently associated with CFA.",
    styles["Body"]))

story.append(Paragraph("S6. Sensitivity to segment duration", styles["H2"]))
dose_str = "; ".join(f"{int(r['window_minutes'])} min: {r['mean']:.2f}" for r in DOSE["lengths"])
dose_minutes = [int(r["window_minutes"]) for r in DOSE["lengths"]]
dose_minutes_text = ", ".join(map(str, dose_minutes[:-1])) + f", and {dose_minutes[-1]}"
dose_by_min = {int(r["window_minutes"]): r for r in DOSE["lengths"]}
p_sex_dose = max(r["p_sex"] for r in DOSE["stratified_by_length"])
p_dx_dose_min = min(r["p_dx"] for r in DOSE["stratified_by_length"])
p_dx_dose_max = max(r["p_dx"] for r in DOSE["stratified_by_length"])
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "figS2_window_stage_sensitivity.png"), width=6.6 * inch, height=6.6 * inch / (11.5 / 4)),
    Paragraph(
    f"Figure S6. Sensitivity of CFA estimates to segment duration "
    f"(n = {DOSE['n_common_patients']:,} patients with usable data at every duration). "
    f"(a) Mean channel-level CFA R² (all EEG channels) at {dose_minutes_text} min ({dose_str}). "
    f"(b) Sex-stratified and "
    f"(c) diagnosis-stratified estimates. The sex difference was present at every available duration "
    f"(Mann-Whitney, all p &le; {p_sex_dose:.3g}); diagnosis differences were not significant "
    f"(p = {p_dx_dose_min:.2g}–{p_dx_dose_max:.2g}).",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"Segment duration had a clear effect on the detectable cardiac contribution. Mean "
    f"CFA R² increased from {dose_by_min[5]['mean']:.2f} at 5 min to "
    f"{dose_by_min[10]['mean']:.2f} at 10 min, {dose_by_min[20]['mean']:.2f} at 20 min, "
    f"{dose_by_min[30]['mean']:.2f} at 30 min, {dose_by_min[45]['mean']:.2f} at 45 min, and "
    f"{dose_by_min[60]['mean']:.2f} at 60 min, consistent with incomplete "
    f"averaging of heartbeat-locked structure over fewer beats in shorter windows. The "
    f"incremental gain "
    f"decreased from {dose_by_min[20]['mean']-dose_by_min[10]['mean']:.2f} R² units between "
    f"10 and 20 min to {dose_by_min[30]['mean']-dose_by_min[20]['mean']:.2f} between 20 and "
    f"30 min and {dose_by_min[60]['mean']-dose_by_min[45]['mean']:.2f} between 45 and 60 min, "
    f"indicating diminishing returns with longer windows, although no plateau was reached within 60 min. "
    f"Fewer channel-recordings contributed at 45 and 60 min ({dose_by_min[45]['n_rows']:,} vs. "
    f"{dose_by_min[30]['n_rows']:,} at 30 min), so the longest-window means may partly reflect a "
    f"different channel composition; between 5 and 30 min, however, channel composition was nearly "
    f"constant ({dose_by_min[5]['n_rows']:,} to {dose_by_min[30]['n_rows']:,} channel-recordings), "
    f"and R² still increased monotonically. The sex difference persisted "
    f"across all tested durations, whereas "
    f"diagnosis differences were not significant in this matched subset. Unlike in the main analysis, "
    f"the no-diagnosis mean was numerically higher than the any-diagnosis mean at every duration "
    f"(e.g. {DOSE['stratified_by_length'][0]['no_dx_mean']:.2f} vs. "
    f"{DOSE['stratified_by_length'][0]['any_dx_mean']:.2f} at 5 min), consistent with the within-site reversal described in the Limitations; the "
    f"diagnosis association should therefore not be interpreted as a disease effect.",
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
    "2. Park H-D, Blanke O. Heartbeat-evoked cortical responses: underlying mechanisms, functional "
    "roles, and methodological considerations. NeuroImage. 2019;197:502-511.",
    "3. Coll M-P, Hobson H, Bird G, Murphy J. Systematic review and meta-analysis of the relationship "
    "between the heartbeat-evoked potential and interoception. Neurosci Biobehav Rev. 2021;122:190-200.",
    "4. Steinfath TP, et al. Heartbeat-evoked responses in M/EEG: a systematic review of methods with "
    "suggestions for analysis and reporting. Psychophysiology. 2026;63(4):e70297. "
    "doi:10.1111/psyp.70297.",
    "5. Dirlich G, Dietl T, Vogl L, Strian F. Topography and morphology of heart action-related EEG "
    "potentials. Electroencephalogr Clin Neurophysiol. 1998;108(3):299-305.",
    "6. Kern M, Aertsen A, Schulze-Bonhage A, Ball T. Heart cycle-related effects on event-related "
    "potentials, spectral power changes, and connectivity patterns in the human ECoG. NeuroImage. "
    "2013;81:178-190.",
    "7. Virjee R-I, Kandasamy R, Garfinkel SN, Carmichael DW, Yogarajah M. Review of methods to "
    "derive the heartbeat-evoked potential: past practices and future directions. Soc Cogn Affect "
    "Neurosci. 2026:nsag057. doi:10.1093/scan/nsag057.",
    "8. Tochikubo O, Miyajima E, Shigemasa T, Ishii M. Relation between body fat-corrected ECG "
    "voltage and ambulatory blood pressure in patients with essential hypertension. Hypertension. "
    "1999;33(5):1159-1163. doi:10.1161/01.HYP.33.5.1159.",
    "9. Li Q, Wen S, Sun H, Ganglberger W, Tripathi A, Turley N, et al.; Westover MB. The Human "
    "Sleep Project (HSP). Brain Data Science Platform. 2026. doi:10.60508/m3sw-rz13.",
    "10. Terzano MG, Parrino L, Smerieri A, Chervin R, Chokroverty S, Guilleminault C, et al. Atlas, "
    "rules, and recording techniques for the scoring of cyclic alternating pattern (CAP) in human "
    "sleep. Sleep Med. 2001;2(6):537-553.",
    "11. Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, et al. PhysioBank, "
    "PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic "
    "signals. Circulation. 2000;101(23):e215-e220.",
    "12. Gramfort A, et al. MEG and EEG data analysis with MNE-Python. Front Neurosci. 2013;7:267.",
    "13. Hyvarinen A, Oja E. Independent component analysis: algorithms and applications. Neural Netw. "
    "2000;13(4-5):411-430.",
    "14. Ablin P, Cardoso J-F, Gramfort A. Faster independent component analysis by preconditioning "
    "with Hessian approximations. IEEE Trans Signal Process. 2018;66(15):4040-4049.",
    "15. Kraskov A, Stögbauer H, Grassberger P. Estimating mutual information. Phys Rev E. "
    "2004;69(6):066138.",
    "16. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful "
    "approach to multiple testing. J R Stat Soc Series B. 1995;57(1):289-300.",
]
for r in refs:
    story.append(Paragraph(r, styles["Ref"]))

doc = SimpleDocTemplate(
    OUT_PDF, pagesize=LETTER,
    topMargin=0.8 * inch, bottomMargin=0.8 * inch,
    leftMargin=0.9 * inch, rightMargin=0.9 * inch,
    title="Large-Scale Associations of BMI and Clinical Obesity With Cardiac Field Artifact in Scalp EEG",
    author="Nir Cafri",
)


def _add_page_number(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(LETTER[0] / 2, 0.55 * inch, str(doc_.page))
    canvas.restoreState()


STORY_FOR_EXPORT = list(story)  # snapshot for make_docx.py; doc.build() consumes `story` in place
doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
print("Wrote", OUT_PDF, os.path.getsize(OUT_PDF), "bytes")
