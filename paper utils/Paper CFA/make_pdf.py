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
OUT_PDF = os.path.join(PAPERS_DIR, "Cafri_CFA_EEG_paper_v2.pdf")
NOTITLE_DIR = os.path.join(FIG_DIR, "_notitle")
os.makedirs(NOTITLE_DIR, exist_ok=True)


def fig(name, width=6.6 * inch, crop=True):
    """Image flowable for figures/<name> with its baked-in matplotlib
    suptitle cropped off, so figure numbers come only from the captions."""
    import numpy as np
    from PIL import Image as PILImage
    im = PILImage.open(os.path.join(FIG_DIR, name)).convert("RGB")
    ink = (np.asarray(im.convert("L")) < 245).any(axis=1)
    min_gap = int(0.015 * len(ink))
    top, r = 0, int(np.argmax(ink))  # first title row
    while crop and r < len(ink):
        gap_end = r
        while gap_end < len(ink) and not ink[gap_end]:
            gap_end += 1
        if gap_end - r >= min_gap and r > int(np.argmax(ink)):
            top = max(gap_end - 8, 0)
            break
        r = gap_end + 1
    # ponytail: gap heuristic; a figure without a suptitle loses only top whitespace
    out = os.path.join(NOTITLE_DIR, name)
    im.crop((0, top, im.width, im.height)).save(out)
    w, h = im.width, im.height - top
    return Image(out, width=width, height=width * h / w)

with open(os.path.join(HERE, "paper_stats.json")) as f:
    S = json.load(f)
with open(os.path.join(HERE, "window_stage_sensitivity_stats.json")) as f:
    SENS = json.load(f)

COHORT, CFA, ICA, STRAT, TOPO = S["cohort"], S["cfa"], S["ica"], S["stratified"], S["topomap"]
DX = S["diagnosis"]
EX, WT = S["v2"]["examples"], S["v2"]["window_test"]
WT_PAIR = {(q["a"], q["b"]): q for q in WT["pairs"]}
BMIHR = S["v2"]["bmi_hr"]
CD = S["cleaning_demo"]
CD_INJ_UV = 2.0  # matches INJECT_UV in make_cleaning_demo.py
CD_GAP = (CD["r2_pre_mean"] - CD["r2_clean_mean"]) / (CD["r2_pre_mean"] - CD["r2_own_mean"]) * 100
WT_P_STR = "&lt; 1e-300" if WT["friedman_p"] == 0 else f"= {WT['friedman_p']:.2g}"
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
    "Cleaning the Heart's Noise from Brain Signals: A Large-Cohort Study of Cardiac Field Artifact in EEG",
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
story.append(Paragraph("Correspondence: pb@tauex.tau.ac.il", styles["Affil"]))
story.append(Spacer(1, 10))
story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#888888")))
story.append(Spacer(1, 10))

# ---------------------------------------------------------------------
# Abstract (NeuroImage limit: 250 words; kept entirely on page 1)
# ---------------------------------------------------------------------
story.append(Paragraph("Abstract", styles["H1"]))
story.append(Paragraph(
    f"The electrical field of the heart is volume-conducted to the scalp and adds a cardiac field "
    f"artifact (CFA) to every EEG recording. CFA is a source of noise for any EEG analysis and a "
    f"direct confound for heartbeat-locked measures such as heartbeat-evoked potentials (HEPs), yet "
    f"it has been characterised only in small samples and is usually assumed to be constant across "
    f"people and to be removed by excluding the QRS complex. We quantified CFA in "
    f"{CFA['n_patients']:,} polysomnography patients ({CFA['n_rows']:,} channel-recordings) from "
    f"the Human Sleep Project, the largest sample in which CFA has been measured. Per channel, the "
    f"R-peak-locked EEG average was regressed on the ECG average, excluding a "
    f"±50 ms QRS interval. Outside the QRS interval, the ECG explained a mean R² of "
    f"{CFA['r2_excl_qrs_mean']:.2f} of heartbeat-locked EEG variance, and removing the ECG-related "
    f"independent component reduced this variance by a median of "
    f"{ICA['hep_pct_drop_median']*100:.0f}%. CFA ranged from negligible to near-complete between "
    f"individual patients and varied across electrodes. It increased with BMI "
    f"(r = {STRAT['bmi']['r_pearson']:.2f}, p = {STRAT['bmi']['p_pearson']:.1e}; "
    f"n = {STRAT['bmi']['n']:,}) and was higher in men than in women "
    f"({STRAT['sex_means']['Male']:.2f} vs. {STRAT['sex_means']['Female']:.2f}, "
    f"p = {STRAT['p_sex_mannwhitney']:.1e}), independently of BMI and at every segment duration. "
    f"Segment duration had a significant effect: in {WT['n']:,} duration-matched patients, mean R² "
    f"rose from {DOSE['lengths'][0]['mean']:.2f} at 5 min to {DOSE['lengths'][-1]['mean']:.2f} at "
    f"{DOSE['lengths'][-1]['window_minutes']:.0f} min (Friedman p {WT_P_STR}). CFA is therefore a participant-, channel-, and duration-dependent noise source; "
    f"EEG studies should estimate it per channel, report and justify segment length, and account "
    f"for BMI and sex when comparing groups.",
    styles["Body"]))
story.append(Paragraph("Highlights", styles["H2"]))
for h in [
    f"In the largest study of cardiac field artifact (CFA) to date ({CFA['n_patients']:,} patients, "
    f"{CFA['n_rows']:,} EEG channel-recordings), the ECG explained a mean "
    f"{CFA['r2_excl_qrs_mean']*100:.0f}% of heartbeat-locked EEG variance even after QRS exclusion.",
    f"CFA increased with BMI (r = {STRAT['bmi']['r_pearson']:.2f}, "
    f"p = {STRAT['bmi']['p_pearson']:.1e}) and was higher in men than in women "
    f"({STRAT['sex_means']['Male']:.2f} vs. {STRAT['sex_means']['Female']:.2f}, "
    f"p = {STRAT['p_sex_mannwhitney']:.1e}); the sex difference was independent of BMI "
    f"(BMI-adjusted p = {S['sex_bmi_confound']['sex_adj_p']:.1e}).",
    f"Segment duration significantly changed CFA estimates (mean R² {DOSE['lengths'][0]['mean']:.2f} "
    f"at 5 min vs. {DOSE['lengths'][-1]['mean']:.2f} at 60 min), so the analysis window must be "
    f"reported and justified.",
]:
    story.append(Paragraph("• " + h, styles["Body"]))
story.append(Paragraph(
    "Keywords: cardiac field artifact; EEG artifact; ECG; body mass index; sex differences; "
    "segment duration; heartbeat-evoked potential; independent component analysis; "
    "polysomnography", styles["Kw"]))
story.append(PageBreak())

# ---------------------------------------------------------------------
# Introduction
# ---------------------------------------------------------------------
story.append(Paragraph("1. Introduction", styles["H1"]))
story.append(Paragraph(
    "The heart is one of the strongest bioelectrical generators in the body, and its electrical field is "
    "volume-conducted through the tissues of the chest, neck, and head to the scalp. Every EEG "
    "recording therefore contains a cardiac field artifact (CFA): a heartbeat-locked deflection "
    "that is not of cortical origin<super>1-4</super>. Because CFA repeats with each heartbeat, it "
    "adds structured noise to any EEG measure computed over time<super>3,4</super>, and it is preserved, rather than "
    "averaged out, in analyses that are time-locked to the R-peak. The heartbeat-evoked potential "
    "(HEP), the R-peak-locked EEG response used to study cardiac interoception<super>5-7</super>, is the clearest "
    "example: CFA and genuine cortical responses to the heartbeat share the same time-locking "
    "event<super>1,6</super>. The standard mitigation is to exclude a short interval around the QRS "
    "complex (typically ±30–50 ms) and, in some studies, to remove an ECG-related independent "
    "component<super>6-8</super>.",
    styles["Body"]))
story.append(Paragraph(
    "Despite its ubiquity, CFA has been characterised only in single cohorts of a few tens of "
    "participants<super>1,2</super>, samples that cannot detect modest differences between "
    "people. As a result, CFA is usually treated as an approximately constant property of the "
    "recording, which a fixed QRS exclusion is assumed to remove. Recent reviews have documented "
    "considerable heterogeneity in how CFA is handled and have noted that inconsistent "
    "preprocessing limits reproducibility and clinical interpretation<super>7-9</super>. If the "
    "amount of CFA differs systematically between people or between analysis choices, then a group "
    "difference in heartbeat-locked EEG may reflect cardiac electrophysiology, tissue "
    "conductivity, or electrode geometry rather than brain activity<super>2,6,10,11</super>.",
    styles["Body"]))
story.append(Paragraph(
    "Three factors are of particular interest. First, body composition alters the geometry and "
    "conductivity of the path between the heart and the scalp; in surface ECG, subcutaneous fat "
    "attenuates cardiac voltage, and correcting voltage for body fat changes its relationship with "
    "cardiac structure<super>12</super>, and obesity is associated with low QRS voltage and shifts of the "
    "cardiac electrical axis<super>13</super>. BMI, although an indirect measure of adiposity, is "
    "available in most clinical datasets. Second, men and women differ in thoracic anatomy, heart "
    "size, and body-fat distribution, so CFA may differ by sex, a common grouping variable in EEG "
    "research. Third, the analysis itself matters: CFA is estimated from averages over "
    "heartbeats, and averages over fewer events are less reliable<super>14</super>, so the apparent amount of contamination "
    "may depend on the chosen segment duration. To our knowledge, none of these factors has been "
    "examined at the population level.",
    styles["Body"]))
story.append(Paragraph(
    f"Here, we quantified CFA in clinical polysomnography recordings from {CFA['n_patients']:,} "
    f"patients in the Human Sleep Project, a publicly available, curated dataset<super>15</super>, "
    f"the largest sample in which CFA has been measured. We used two complementary estimators: a "
    f"model-free estimator that regresses each channel's R-peak-locked EEG average on the same "
    f"patient's R-peak-locked ECG average, and a model-based estimator that applies ECG-informed "
    f"ICA. We illustrate the range of CFA in individual patients, describe its scalp distribution, "
    f"and test its dependence on BMI, sex, age, and segment duration (5–60 min). We hypothesised "
    f"that ECG-related variance would persist outside the QRS interval, that it would be greater "
    f"with higher BMI and differ by sex, and that it would increase with segment duration as the "
    f"heartbeat-locked average became more stable.",
    styles["Body"]))

# ---------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------
story.append(Paragraph("2. Methods", styles["H1"]))
story.append(Paragraph("2.1 Cohort and recordings", styles["H2"]))
story.append(Paragraph(
    f"This study is a secondary analysis of existing polysomnography recordings; no recordings were "
    f"acquired for this study. Patient data were "
    f"taken from the Human Sleep Project<super>15</super>, a publicly available, curated dataset "
    f"of clinical polysomnography recordings with linked electronic health records (EHR), "
    f"distributed through the Brain Data Science Platform. BMI was "
    f"computed from each patient's median height and weight recorded in the EHR vitals and admission "
    f"tables; values outside 10–80 kg/m² were excluded as implausible. EEG channels were identified by matching channel labels against standard "
    f"10-20/10-10 electrode names, thereby excluding intracranial depth electrodes and auxiliary/DC "
    f"channels; at least two EEG channels were required. The ECG channel was identified by label "
    f"pattern matching (ECG/EKG), and the first matching lead was used. Cohort characteristics are "
    f"reported in Section 3.1.",
    styles["Body"]))
story.append(Paragraph("2.2 Window selection, R-peak detection, and preprocessing", styles["H2"]))
story.append(Paragraph(
    "For the main analysis, a single 10-min analysis window was selected for each recording by a "
    "seeded, reproducible random search (uniformly distributed start times; up to 10 attempts). R-peaks were detected on "
    "the ECG of each candidate window before the analysis filter (see below and Table 1): the "
    "median-subtracted signal was band-pass "
    "filtered at 5–25 Hz (third-order zero-phase Butterworth), rectified, and peaks were identified "
    "with a minimum inter-peak distance of 300 ms and a prominence threshold of three times the "
    "median absolute deviation of the rectified signal; rectification made detection insensitive to "
    "ECG polarity. A window was accepted if (i) at least 70% of EEG "
    "channels passed signal-quality criteria (≥99.9% finite samples; standard deviation 0.1–500 µV; "
    "peak-to-peak amplitude ≤5,000 µV; &lt;20% flat samples), (ii) the mean heart rate was 35–180 "
    "beats/min, and (iii) at least 70% of R-R intervals lay between 0.33 and 2.0 s. The accepted "
    "window was then band-pass filtered between 1 Hz and min(100 Hz, Nyquist − 0.5 Hz) using a "
    "zero-phase FIR filter in MNE-Python<super>16</super>; because 86% of recordings were sampled at "
    "200 Hz, the upper edge lay just below the Nyquist frequency in most recordings. The same filter "
    "was applied to the EEG and ECG channels, and signals were analysed in µV at their native "
    "sampling rate without re-referencing, channel interpolation, or amplitude-based rejection, so "
    "that CFA was estimated in minimally processed data. To test the effect of segment duration, "
    "the same procedure was repeated with windows of 5, 20, 30, 45, and 60 min (see Section 2.7). "
    "All window-selection, quality-control, and preprocessing parameters are summarised in Table 1.",
    styles["Body"]))
PREPROC = [
    ("Window selection", "Seeded random search, uniformly distributed start times, up to 10 attempts; "
     "10 min (main analysis), 5–60 min (duration analysis, see Section 2.7)"),
    ("R-peak detection", "Median subtraction; 5–25 Hz third-order zero-phase Butterworth band-pass; "
     "rectification; minimum inter-peak distance 300 ms; prominence ≥3 × MAD"),
    ("EEG channel quality", "≥99.9% finite samples; SD 0.1–500 µV; peak-to-peak ≤5,000 µV; "
     "&lt;20% flat samples"),
    ("Window acceptance", "≥70% of EEG channels pass quality criteria; mean heart rate 35–180 "
     "beats/min; ≥70% of R-R intervals 0.33–2.0 s"),
    ("Analysis filter", "Zero-phase FIR band-pass, 1 Hz to min(100 Hz, Nyquist − 0.5 Hz), "
     "MNE-Python; applied to EEG and ECG"),
    ("Signal handling", "Native sampling rate, µV; no re-referencing, channel interpolation, or "
     "amplitude-based rejection"),
    ("HEP epochs", "−300 to 400 ms around each R-peak; no baseline correction; ≥20 epochs; "
     "±50 ms QRS exclusion for outside-QRS R² (see Section 2.3)"),
]
preproc_table = Table(
    [[Paragraph("Step", styles["Caption"]), Paragraph("Parameters", styles["Caption"])]]
    + [[Paragraph(a, styles["Caption"]), Paragraph(b, styles["Caption"])] for a, b in PREPROC],
    colWidths=[1.6 * inch, 5.0 * inch])
preproc_table.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP")]))
story.append(KeepTogether([
    Paragraph("<b>Table 1.</b> Window selection, quality control, and preprocessing parameters.",
              styles["Caption"]),
    preproc_table,
]))
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
    "is standard in HEP analysis<super>8</super>. This estimator "
    "does not depend on the ability of ICA to isolate a cardiac source; rather, it tests directly "
    "the extent to which the averaged scalp waveform is a linearly scaled (and offset) copy of the "
    "averaged ECG. For patient-level analyses (see Section 2.7), channel-level R² values were averaged across "
    "each patient's channels at the six well-covered sites (see Section 2.5) and, for patients with more than "
    "one recording, across recordings (patient-mean CFA R²).",
    styles["Body"]))
story.append(Paragraph("2.4 Model-based CFA estimator: ECG-informed ICA", styles["H2"]))
story.append(Paragraph(
    "In parallel, ICA<super>17</super> (extended Picard algorithm<super>18</super>; "
    "min(15, number of EEG channels − 1) components; 500 iterations; fixed "
    "random seed) was fitted to the filtered EEG of the same 10-min window used for the regression "
    "estimator. "
    "Components whose correlation with the ECG channel exceeded the default threshold were identified "
    "as ECG-related using the correlation-based scoring implemented in "
    "MNE-Python<super>16</super>; if none did, the component with "
    "the highest absolute score was used (two components were flagged in 49 of 13,624 recordings). Rather than each component's share "
    "of continuous-signal variance, its mixing-weighted contribution to each channel was evaluated on "
    "the R-peak-locked evoked average, thereby quantifying the fraction of heartbeat-evoked signal "
    "that the component explains. In addition, the effect of artifact removal was measured directly "
    "by comparing the HEP variance of each channel before and after excluding the ECG-related "
    "component through ICA back-projection, yielding the realised percentage reduction in variance, "
    "100 × (pre - post)/pre, rather than the nominal share of the component. For Figure 2b, the "
    "ratio of ECG-related component variance to residual variance (SNR) was also computed for each "
    "channel-recording.",
    styles["Body"]))
story.append(Paragraph("2.5 Channel canonicalisation and minimum coverage", styles["H2"]))
story.append(Paragraph(
    "Channel labels were mapped to their scalp-side 10-20 site, and labels corresponding only to a "
    "reference electrode were excluded. Figure 2c–d includes all canonical sites recorded in at least "
    f"{TOPO['min_patients']} patients. Analyses requiring comparison across conditions or estimators "
    "at a common set of electrodes (see Sections 3.4–3.6, S2–S4) were restricted to sites present in at least "
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
    "lags of &plusmn;100 ms in 5 ms steps, extending the zero-lag analysis of Section 2.3; the peak absolute "
    "correlation and its lag were reported after per-channel sign alignment, as reference polarity "
    "is arbitrary. Mutual information was estimated with the Kraskov k-nearest-neighbour "
    "estimator<super>19</super> to capture both linear and nonlinear EEG-ECG dependence. Both measures "
    "were computed for three conditions: pre-ICA, post-ICA, and a non-heartbeat-locked control in "
    "which the same window was re-epoched around an equal number of pseudo-events placed uniformly at "
    "random instead of R-peaks; in this subsample, ICA was refitted and R-peaks were re-detected on "
    "the band-pass-filtered ECG. For the "
    "power spectral density (PSD) comparison (Figure S4), the patient's ECG evoked average was added "
    "as a fourth condition, and Welch PSD (dB) was computed for each condition at the six "
    "well-covered electrodes (see Section 2.5). Two further random subsamples supported Figure S2: in one "
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
    "variable using Pearson correlation and by tertile using one-way ANOVA. Differences by sex and "
    "by presence of a linked diagnosis were "
    "assessed with the Mann-Whitney U test. Differences among diagnosis categories (see Section 3.6) were "
    "assessed with the Kruskal-Wallis test followed by pairwise Mann-Whitney tests, with "
    "Benjamini-Hochberg false discovery rate (FDR) correction<super>20</super> across all pairwise "
    "comparisons. The association with continuous BMI was assessed using Pearson correlation. To retain one "
    "observation per patient, the BMI analysis used the longest available CFA window for each "
    "patient (30 min for 98% of patients), so absolute R² values in the BMI and BMI-adjusted sex "
    "analyses are not directly comparable with the 10-min estimates. BMI correlations were also "
    "estimated separately by sex, and a BMI-by-sex interaction "
    "term in an ordinary least-squares (OLS) model was used to test whether the slopes differed. A "
    "supplementary OLS model compared the sex coefficient before and after adjustment for BMI in the "
    "same subset. The relationship between model-free R² and the log-transformed ICA SNR (see Section 2.4) "
    "was assessed with Pearson correlation across channel-recordings (Figure 2b). The effect of "
    "segment duration (see Section 3.5) was assessed in patients with usable data at all six "
    "durations (5–60 min), using all EEG channels rather than only the six well-covered sites "
    "(Figure 3d reports channel-level means). Patient-mean R² was compared across the six "
    "durations with the Friedman test, and consecutive durations (and 5 vs. 60 min and 10 vs. 60 "
    "min) were compared with paired Wilcoxon signed-rank tests, Holm-corrected; Kendall's W was "
    "reported as the effect size. Sexes and diagnosis groups were compared with Mann-Whitney tests "
    "at each duration. Windows of each length were drawn with the same seeded procedure, so that shorter "
    "windows were usually nested within longer ones; the 45- and 60-min analyses used one recording "
    "per patient. Sample sizes differ between analyses because each requires different data: the "
    f"model-free estimator included all patients with a valid window ({CFA['n_patients']:,}); the ICA "
    f"estimator, patients with a converged decomposition ({ICA['n_patients']:,}); age, sex, and "
    f"diagnosis analyses, EHR-linked patients with known age and at least one well-covered site "
    f"({STRAT['n_with_age']:,}); BMI analyses, patients with a plausible BMI "
    f"({STRAT['bmi']['n']:,}); and the duration analysis, patients with data at all six durations "
    f"({DOSE['n_common_patients']:,}). For illustration (Figure 1), two example patients were "
    f"selected among Human Sleep Project patients with measured BMI and usable data from the same "
    f"recording at all six durations: a man with a BMI in the lowest range and low CFA, and a man "
    f"with a BMI in the highest range and high CFA. These examples were chosen to show the range "
    f"of CFA and are not used for inference. To test the effect of sleep stage (see Section 3.1), "
    f"a random subsample of {SENS['n_patients']} patients was staged in 30-s epochs with "
    f"YASA<super>21</super>; for each of wake, light sleep (N1+N2), N3, and REM, up to "
    f"{SENS['draws_per_cell']} windows per duration "
    f"({', '.join(str(m) for m in SENS['window_lengths_min'])} min) were drawn from contiguous "
    f"stretches of that stage, and CFA R² was computed as in Section 2.3. Stage and duration were "
    f"compared with one-way ANOVA.",
    styles["Body"]))
story.append(Paragraph("2.8 Proof of concept: ECG-free CFA cleaning", styles["H2"]))
story.append(Paragraph(
    f"Many EEG datasets have no ECG channel. We therefore tested whether the cohort can supply "
    f"what such recordings lack. Single-beat CFA was too weak relative to ongoing EEG to detect "
    f"heartbeats reliably from the EEG itself, but CFA has a largely fixed scalp pattern, which "
    f"allows cleaning without locating individual beats. In Human Sleep Project recordings with "
    f"the standard six-channel montage (F3, F4, C3, C4, O1, O2, mastoid-referenced), the dominant "
    f"spatial pattern of each patient's R-peak-locked EEG average (first singular vector) was "
    f"computed in a training set of {CD['n_train']} patients, sign-aligned, and averaged into one "
    f"population CFA pattern. For {CD['n_test']} different held-out patients, this pattern was "
    f"projected out of every EEG sample (signal-space projection); the ECG was not used for "
    f"cleaning. The held-out ECG was then used only for scoring: CFA R² (outside QRS) before "
    f"and after cleaning, compared with an upper bound obtained by projecting out each patient's "
    f"own ECG-derived pattern and with the pseudo-event chance level (see Section 2.6). To measure "
    f"preservation of brain activity, a synthetic neural HEP (Gaussian, −{CD_INJ_UV:.0f} µV at "
    f"+300 ms, frontal &gt; central &gt; occipital) was added before cleaning and the retained "
    f"fraction was measured; the cost to ongoing EEG was measured as the change in Welch power "
    f"(1–40 Hz). Patterns specific to sex and BMI group (&lt;27, 27–35, ≥35 kg/m²) were "
    f"also learned to test whether conditioning on these variables improves cleaning.",
    styles["Body"]))
story.append(Paragraph("3. Results", styles["H1"]))
story.append(Paragraph("3.1 Cohort", styles["H2"]))
sex_str = ", ".join(f"{k} n={v:,}" for k, v in COHORT["sex_counts"].items())
top3_dx = list(COHORT["top_diagnoses"].items())[:3]
top3_str = "; ".join(f"{k} (n={v:,})" for k, v in top3_dx)
story.append(Paragraph(
    f"Demographic data were available for {COHORT['n_age']:,} "
    f"of the {CFA['n_patients']:,} analysed patients (median age "
    f"{COHORT['age_median']:.0f} years, range {COHORT['age_min']:.0f}–{COHORT['age_max']:.0f}; "
    f"{sex_str}). This EHR-linked cohort consisted of clinically referred polysomnography patients with a high "
    f"diagnostic burden rather than a healthy community sample; the most prevalent diagnosis "
    f"categories were {top3_str} (categories are not mutually exclusive). Cohort composition is "
    f"shown in Supplementary Figure S1.",
    styles["Body"]))
STG = SENS["by_stage_mean"]
story.append(Paragraph(
    f"Analysis windows were placed at random start times across the whole recording, without "
    f"regard to sleep stage (see Section 2.2). They therefore sampled wake and all sleep stages "
    f"approximately in proportion to their duration across the night, and the cohort-level "
    f"estimates describe CFA over the sleep cycle as a whole rather than in a single stage. To test "
    f"whether the conclusions hold across stages, a subsample of {SENS['n_patients']} patients was "
    f"staged with YASA<super>21</super> (see Section 2.7) and CFA was estimated separately in "
    f"wake, light sleep (N1+N2), N3, and REM. Mean CFA R² was similar in all stages (wake "
    f"{STG['W']:.2f}, light sleep {STG['light_sleep']:.2f}, N3 {STG['N3']:.2f}, REM "
    f"{STG['R']:.2f}; ANOVA p = {SENS['p_stage_anova']:.2f}), whereas segment duration had a "
    f"significant effect in the same data (p = {SENS['p_length_anova']:.1e}). Within this "
    f"subsample, sleep stage therefore did not measurably change CFA, which suggests that the "
    f"main conclusions are not specific to one stage of the sleep cycle.",
    styles["Body"]))

story.append(Paragraph("3.2 CFA in individual patients", styles["H2"]))
EXA, EXB = EX
story.append(KeepTogether([
    fig("fig1_patient_examples.png"),
    Paragraph(
    f"<b>Figure 1.</b> Cardiac field artifact (CFA) in two example patients. Each head map shows the "
    f"per-channel CFA R² (variance of the R-peak-locked EEG average explained by the ECG average, "
    f"outside the ±50 ms QRS window) at each recorded electrode, computed from segments of 5, 10, "
    f"20, 30, 45, and 60 min drawn from the same recording; the last column shows the change "
    f"from 5 to 60 min. Patient A ({EXA['sex'].lower()}, {EXA['age']:.0f} years, BMI "
    f"{EXA['bmi']:.1f} kg/m²) showed almost no CFA at any duration (mean R² "
    f"{EXA['mean_r2']['5']:.2f}–{max(EXA['mean_r2'].values()):.2f}). Patient B "
    f"({EXB['sex'].lower()}, {EXB['age']:.0f} years, BMI {EXB['bmi']:.1f} kg/m²) showed high CFA "
    f"at every electrode, which increased from {EXB['mean_r2']['5']:.2f} at 5 min to "
    f"{EXB['mean_r2']['60']:.2f} at 60 min. Colour maps are interpolated from "
    f"{EXA['n_sites']} electrodes; electrode names mark the recording sites.",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"Figure 1 illustrates how much CFA can differ between people. In a lean man (Patient A, BMI "
    f"{EXA['bmi']:.1f} kg/m²), the ECG explained only {EXA['mean_r2']['10']*100:.0f}% of the "
    f"heartbeat-locked EEG variance at 10 min, and this value did not change with segment "
    f"duration. In a man with severe obesity (Patient B, BMI {EXB['bmi']:.1f} kg/m²), the ECG "
    f"explained {EXB['mean_r2']['10']*100:.0f}% at 10 min, so the heartbeat-locked EEG was almost "
    f"a scaled copy of the ECG at every electrode. In Patient B, CFA also increased with segment "
    f"duration, from {EXB['mean_r2']['5']:.2f} at 5 min to {EXB['mean_r2']['60']:.2f} at 60 min. "
    f"The cohort-level analyses below test whether these patterns, higher CFA with higher BMI and "
    f"with longer segments, hold across the population.",
    styles["Body"]))

story.append(Paragraph("3.3 CFA across the cohort", styles["H2"]))
story.append(KeepTogether([
    fig("fig2_cfa_overview.png", crop=False),
    Paragraph(
    f"<b>Figure 2.</b> Cardiac field artifact (CFA) across {CFA['n_patients']:,} patients "
    f"({CFA['n_rows']:,} channel-recordings, 10-min segments). (a) Model-free estimator: "
    f"distribution of per-channel R² between the R-peak-locked EEG average and the ECG average, "
    f"over the full epoch (mean {CFA['r2_full_mean']:.2f}) and outside the ±50 ms QRS-exclusion "
    f"window (mean {CFA['r2_excl_qrs_mean']:.2f}). (b) Model-based estimator ({ICA['n_patients']:,} "
    f"patients): ICA SNR (ECG-related component variance relative to residual variance; see Section 2.4) "
    f"plotted against the model-free CFA R² of the same channel-recording (logarithmic SNR axis; "
    f"line, least-squares fit of log SNR on R²; r = {ICA['r2_vs_snr_r']:.2f}, p {R2_SNR_P_STR}, "
    f"n = {ICA['r2_vs_snr_n']:,}). (c) Scalp map of mean CFA R² (outside QRS) at the "
    f"{TOPO['n_sites']} canonical 10-20 sites recorded in at least {TOPO['min_patients']} patients "
    f"({TOPO['n_rows']:,} channel-recordings; bipolar/mastoid-referenced labels were "
    f"canonicalised to their scalp-side site; {', '.join(TOPO['dropped_sites'])} were dropped for "
    f"falling below the patient floor). (d) Distribution of per-channel R² per site, sorted by "
    f"median (orange line); box, interquartile range; n per site is indicated. Coverage is "
    f"uneven, from the six standard montage sites (F3/F4/C3/C4/O1/O2; n &gt; 12,000 each) to "
    f"sites near the {TOPO['min_patients']}-patient threshold, whose estimates are less precise. "
    f"(e) Patient-mean CFA R² per clinical diagnosis category (mean, 95% CI, Welch; number of "
    f"patients in brackets); dotted line, EHR-linked patients with no recorded diagnosis "
    f"(mean {DX['no_dx_mean']:.2f}, n = {DX['no_dx_n']:,}; confounded by recording site, see "
    f"Limitations). Patients may belong to more than one category; categories are drawn from "
    f"fifteen predefined groups, and those with fewer than 10 patients are omitted; Kruskal-Wallis "
    f"test across all groups, p = {DX['p_kruskal']:.2g}. (f) Pairwise comparisons among "
    f"categories: Mann-Whitney p-values, Benjamini-Hochberg corrected across all "
    f"{DX['pairwise']['n_pairs']} tests ({DX['pairwise']['n_significant_fdr']} pairs significant "
    f"at q &lt; 0.05; colour scale saturates at q = 0.5); categories ordered as in (e).",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"Across the cohort, the ECG explained a substantial proportion of heartbeat-locked EEG "
    f"variance outside the conventional QRS-exclusion window (mean R² = "
    f"{CFA['r2_excl_qrs_mean']:.2f}, median {CFA['r2_excl_qrs_median']:.2f}; Figure 2a). This "
    f"value was only modestly lower than the full-epoch estimate (mean R² = "
    f"{CFA['r2_full_mean']:.2f}), so QRS exclusion removed only approximately "
    f"{(1 - CFA['r2_excl_qrs_mean']/CFA['r2_full_mean'])*100:.0f}% of the ECG-explained variance. "
    f"The distribution was broad, from channels with no detectable CFA to channels in which the "
    f"EEG average was almost identical to the ECG average. In the "
    f"{ZL['non_locked']['n']}-patient subsample with a chance-level reference (see Section 2.6), full-epoch "
    f"R² at some electrodes (F3, F4, C3, C4) was {ZL['pre_ica']['mean']:.2f} for R-peak-locked averages but "
    f"{ZL['non_locked']['mean']:.2f} (median {ZL['non_locked']['median']:.2f}) for averages "
    f"around random pseudo-events, so the observed R² far exceeded chance similarity between "
    f"finite-length averages.",
    styles["Body"]))
story.append(Paragraph(
    f"The model-based estimator agreed. Across all channels, the ECG-related ICA component "
    f"accounted for a median of {ICA['component_variance_fraction_median_unfiltered']*100:.0f}% of "
    f"heartbeat-locked EEG variance, and excluding it reduced this variance by a median of "
    f"{ICA['hep_pct_drop_median']*100:.0f}%. Channels with higher model-free R² also had higher "
    f"ICA SNR (Figure 2b). The dispersion in Figure 2b is expected, because a single, globally "
    f"selected ICA source does not load equally on all electrodes. Regression on the concurrently "
    f"recorded ECG has previously been proposed for estimating and removing CFA from "
    f"heartbeat-locked EEG, in linear<super>22</super> and, more recently, nonlinear "
    f"neural-network<super>23</super> form. Following this approach, we used the per-channel ECG "
    f"regression as the primary estimate and the ICA-based removal as convergent evidence.",
    styles["Body"]))
story.append(Paragraph(
    f"CFA was not uniformly distributed across the scalp (Figure 2c–d); it was highest at "
    f"{TOPO['highest_site']} (mean R² = {TOPO['highest_mean']:.2f}) and lowest at "
    f"{TOPO['lowest_site']} (mean R² = {TOPO['lowest_mean']:.2f}), an approximately "
    f"{TOPO['highest_mean']/max(TOPO['lowest_mean'], 1e-6):.1f}-fold range. Because sites outside "
    f"the standard montage came from a minority of recordings with different montages and "
    f"references, part of this range reflects montage and referencing rather than electrode "
    f"position; within the six standard, mastoid-referenced sites, mean R² ranged from "
    f"{TOPO['site_means']['F3']:.2f} (F3) to {TOPO['site_means']['F4']:.2f} (F4). A "
    f"site-independent CFA correction would therefore perform unevenly across electrodes.",
    styles["Body"]))

story.append(Paragraph("3.4 CFA increases with BMI and is higher in men", styles["H2"]))
BMI_STRAT = STRAT["bmi"]
CONF = S["sex_bmi_confound"]
BMI_WINDOW_COUNTS = ", ".join(
    f"{minutes} min: n={count:,}" for minutes, count in sorted(
        ((int(k), v) for k, v in BMI_STRAT["window_counts"].items())
    )
)
dose_str = "; ".join(f"{int(r['window_minutes'])} min: {r['mean']:.2f}" for r in DOSE["lengths"])
dose_by_min = {int(r["window_minutes"]): r for r in DOSE["lengths"]}
p_sex_dose = max(r["p_sex"] for r in DOSE["stratified_by_length"])
p_dx_dose_min = min(r["p_dx"] for r in DOSE["stratified_by_length"])
p_dx_dose_max = max(r["p_dx"] for r in DOSE["stratified_by_length"])
max_adjacent_p = max(WT_PAIR[(a, b)]["p_holm"] for a, b in [(5, 10), (10, 20), (20, 30), (30, 45), (45, 60)])
story.append(KeepTogether([
    fig("fig3_bmi_sex_duration.png", crop=False),
    Paragraph(
    f"<b>Figure 3.</b> Dependence of CFA on patient characteristics (a–c) and segment duration (d–f). "
    f"Patient-mean CFA R² (outside QRS) vs. (a) age, continuous "
    f"(n = {STRAT['n_with_age']:,}; Pearson r = {STRAT['r_age_pearson']:.2f}, "
    f"p = {STRAT['p_age_pearson']:.2g}; line = least-squares fit), (b) sex "
    f"(Mann-Whitney p = {STRAT['p_sex_mannwhitney']:.1e}; box shows quartiles; outliers omitted), "
    f"and (c) BMI, continuous, with sex-specific fits (n = {BMI_STRAT['n']:,}; Pearson "
    f"r = {BMI_STRAT['r_pearson']:.2f}, p = {BMI_STRAT['p_pearson']:.2g}; one observation per "
    f"patient, longest available window; {BMI_WINDOW_COUNTS}). Female n = "
    f"{BMI_STRAT['by_sex']['Female']['n']:,}, r = {BMI_STRAT['by_sex']['Female']['r_pearson']:.2f}, "
    f"p = {BMI_STRAT['by_sex']['Female']['p_pearson']:.1e}; male n = "
    f"{BMI_STRAT['by_sex']['Male']['n']:,}, r = {BMI_STRAT['by_sex']['Male']['r_pearson']:.2f}, "
    f"p = {BMI_STRAT['by_sex']['Male']['p_pearson']:.1e}; BMI-by-sex interaction "
    f"p = {BMI_STRAT['sex_interaction_p']:.2g}. (d–f) Patients with usable data at every duration "
    f"(n = {DOSE['n_common_patients']:,}). (d) Mean channel-level CFA R² (all EEG channels; ± 95% "
    f"CI) at each duration ({dose_str}); Friedman test across durations on patient-mean R², "
    f"p {WT_P_STR}; every consecutive step significant (paired Wilcoxon, Holm-corrected, all "
    f"p &le; {max_adjacent_p:.1g}). (e) Sex-stratified estimates; men had higher CFA at every "
    f"duration (Mann-Whitney, all p &le; {p_sex_dose:.3g}). (f) Diagnosis-stratified estimates; "
    f"differences were not significant (p = {p_dx_dose_min:.2g}–{p_dx_dose_max:.2g}).",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"BMI was positively correlated with CFA R² (Figure 3c; n = {BMI_STRAT['n']:,}, Pearson "
    f"r = {BMI_STRAT['r_pearson']:.2f}, p = {BMI_STRAT['p_pearson']:.1e}; mean BMI "
    f"{BMI_STRAT['bmi_mean']:.1f} kg/m²). Each 10 kg/m² increase in BMI was associated with an "
    f"increase of {BMI_STRAT['slope']*10:.3f} in patient-mean R². The correlation was positive "
    f"and significant in both women (r = {BMI_STRAT['by_sex']['Female']['r_pearson']:.2f}, "
    f"p = {BMI_STRAT['by_sex']['Female']['p_pearson']:.1e}) and men "
    f"(r = {BMI_STRAT['by_sex']['Male']['r_pearson']:.2f}, "
    f"p = {BMI_STRAT['by_sex']['Male']['p_pearson']:.1e}), and the BMI-by-sex interaction was not "
    f"significant (p = {BMI_STRAT['sex_interaction_p']:.2g}). The association, although weak at "
    f"the level of the individual (r² ≈ {BMI_STRAT['r_pearson']**2:.2f}), was consistent with the "
    f"diagnosis analysis, in which clinically obese patients had the highest CFA of all diagnosis "
    f"categories (see Section 3.6).",
    styles["Body"]))
story.append(Paragraph(
    f"Sex also had a clear effect. Patient-mean CFA R² was higher in men than in women "
    f"({STRAT['sex_means']['Male']:.2f} vs. {STRAT['sex_means']['Female']:.2f}, "
    f"p = {STRAT['p_sex_mannwhitney']:.1e}; Figure 3b). This difference was not explained by BMI: "
    f"women in this cohort had a higher mean BMI than men "
    f"({BMI_STRAT['by_sex']['Female']['bmi_mean']:.1f} vs. "
    f"{BMI_STRAT['by_sex']['Male']['bmi_mean']:.1f} kg/m²), and the male-versus-female "
    f"coefficient was unchanged after adjustment for BMI ({CONF['sex_unadj_coef']:.3f} "
    f"unadjusted vs. {CONF['sex_adj_coef']:.3f} adjusted, p = {CONF['sex_adj_p']:.2g}; "
    f"Supplementary Section S5), while BMI remained an independent predictor (p = {CONF['bmi_p']:.2g}). "
    f"The sex difference was also present at every segment duration (see Section 3.5). In contrast, age "
    f"showed no linear relationship with CFA R² (Figure 3a; r = {STRAT['r_age_pearson']:.2f}, "
    f"p = {STRAT['p_age_pearson']:.2g}); only the oldest age tertile was slightly lower than the "
    f"younger two ({STRAT['age_tertile_means']['Older']:.2f} vs. "
    f"{STRAT['age_tertile_means']['Younger']:.2f}/{STRAT['age_tertile_means']['Middle']:.2f}, "
    f"p = {STRAT['p_age_anova']:.1e}).",
    styles["Body"]))

story.append(Paragraph("3.5 Segment duration changes the estimated CFA", styles["H2"]))
story.append(Paragraph(
    f"Segment duration had a significant and systematic effect on the estimated CFA (Friedman "
    f"χ² = {WT['friedman_chi2']:,.0f}, p {WT_P_STR}, Kendall's W = {WT['kendall_w']:.2f}; "
    f"n = {WT['n']:,}). Mean CFA R² increased from {dose_by_min[5]['mean']:.2f} at 5 min to "
    f"{dose_by_min[10]['mean']:.2f} at 10 min, {dose_by_min[20]['mean']:.2f} at 20 min, "
    f"{dose_by_min[30]['mean']:.2f} at 30 min, {dose_by_min[45]['mean']:.2f} at 45 min, and "
    f"{dose_by_min[60]['mean']:.2f} at 60 min (Figure 3d), and every consecutive step was "
    f"significant after Holm correction. Between 5 and 60 min, R² increased in "
    f"{WT_PAIR[(5, 60)]['pct_patients_increase']:.0f}% of patients (mean increase "
    f"{WT_PAIR[(5, 60)]['mean_diff']:.2f}); between the 10-min windows of the main analysis and 60 "
    f"min, it increased in {WT_PAIR[(10, 60)]['pct_patients_increase']:.0f}% of patients (mean "
    f"increase {WT_PAIR[(10, 60)]['mean_diff']:.2f}). The gain per step decreased with duration, "
    f"from {dose_by_min[20]['mean']-dose_by_min[10]['mean']:.2f} R² units between 10 and 20 min "
    f"to {dose_by_min[60]['mean']-dose_by_min[45]['mean']:.2f} between 45 and 60 min, but no "
    f"plateau was reached within 60 min. Fewer channel-recordings contributed at 45 and 60 min "
    f"({dose_by_min[45]['n_rows']:,} vs. {dose_by_min[30]['n_rows']:,} at 30 min), so the "
    f"longest-window means may partly reflect a different channel composition; between 5 and 30 "
    f"min, however, channel composition was nearly constant ({dose_by_min[5]['n_rows']:,} to "
    f"{dose_by_min[30]['n_rows']:,} channel-recordings), and R² still increased monotonically.",
    styles["Body"]))
story.append(Paragraph(
    f"The sex difference was stable across durations: men had higher CFA than women at every "
    f"duration (Figure 3e; e.g. {DOSE['stratified_by_length'][0]['male_mean']:.2f} vs. "
    f"{DOSE['stratified_by_length'][0]['female_mean']:.2f} at 5 min and "
    f"{DOSE['stratified_by_length'][-1]['male_mean']:.2f} vs. "
    f"{DOSE['stratified_by_length'][-1]['female_mean']:.2f} at 60 min). Diagnosis differences "
    f"were not significant in this matched subset (Figure 3f), and the no-diagnosis mean was "
    f"numerically higher than the any-diagnosis mean at every duration, consistent with the "
    f"within-site reversal described in the Limitations.",
    styles["Body"]))

story.append(Paragraph("3.6 CFA by diagnosis category", styles["H2"]))
story.append(Paragraph(
    f"Among diagnosis categories (Figure 2e–f), CFA was highest in patients with {DX['highest_category']} "
    f"(+{DX['highest_diff']:.2f} R² units relative to the no-diagnosis reference, "
    f"n = {DX['categories'][DX['highest_category']]['n']:,}) and lowest in "
    f"{DX['lowest_category']} ({DX['lowest_diff']:+.2f}, "
    f"n = {DX['categories'][DX['lowest_category']]['n']:,}). Of {DX['pairwise']['n_pairs']} "
    f"pairwise comparisons between categories, {DX['pairwise']['n_significant_fdr']} remained "
    f"significant after FDR correction, of which "
    f"{sum('Obesity' in (q['a'], q['b']) for q in DX['pairwise']['significant_pairs'])} involved "
    f"Obesity, which differed significantly from every other category; the remainder involved "
    f"Obstructive Sleep Apnea. This pattern agrees with the BMI association (see Section 3.4). All "
    f"{DX['n_categories_total']} displayed categories lay above the no-diagnosis reference, but "
    f"because the reference group is confounded by recording site (see Limitations) and the "
    f"comparisons were unadjusted for age, sex, and BMI, these differences should not be "
    f"interpreted as disease effects.",
    styles["Body"]))

story.append(Paragraph("3.7 ECG-free cleaning with a cohort-trained spatial filter", styles["H2"]))
story.append(KeepTogether([
    fig("fig5_cleaning_demo.png"),
    Paragraph(
    f"<b>Figure 4.</b> ECG-free cleaning of CFA with a spatial filter learned from the cohort "
    f"(see Section 2.8). (a) Six seconds of EEG from Patient B (Figure 1) before (orange) and after "
    f"(black) cleaning; the ECG (*) is shown for reference only and was not used; dotted lines "
    f"mark R-peaks. (b) Population CFA pattern learned from {CD['n_train']} training patients. "
    f"(c) Heartbeat-locked average of Patient B at the channel with the largest reduction, before "
    f"and after cleaning (grey band, ±50 ms QRS window). (d) Patient-mean CFA R² in "
    f"{CD['n_test']} held-out patients before and after ECG-free cleaning, compared with an upper "
    f"bound that uses each patient's own ECG and with the pseudo-event chance level; grey lines "
    f"connect the same patient. (e) Mean CFA R² (± SEM) before and after cleaning across BMI bins "
    f"(bins with at least 10 patients).",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"The CFA scalp pattern was highly consistent across patients (median absolute cosine "
    f"similarity between individual patterns and the population pattern, "
    f"{CD['pattern_consistency_median_cos']:.2f}; Figure 4b). Projecting this single learned "
    f"pattern out of held-out recordings, with no ECG, reduced mean CFA R² from "
    f"{CD['r2_pre_mean']:.2f} to {CD['r2_clean_mean']:.2f} (a {CD['pct_reduction_mean']:.0f}% "
    f"reduction; Wilcoxon p = {CD['p_pre_vs_clean']:.1e}), and R² decreased in "
    f"{CD['pct_patients_reduced']:.0f}% of patients (Figure 4d). The ECG-based upper bound reached "
    f"{CD['r2_own_mean']:.2f} and the chance level was {CD['r2_null_mean']:.2f}, so the ECG-free "
    f"filter closed {CD_GAP:.0f}% of the gap between uncleaned data and the upper bound. In "
    f"Patient B, mean R² fell from {CD['example_r2_pre']:.2f} to {CD['example_r2_clean']:.2f}, and "
    f"the heartbeat-locked deflections visible in the raw EEG were largely removed (Figure 4a, c). "
    f"Cleaning reduced CFA at every BMI level, although residual CFA remained higher at higher BMI "
    f"(Figure 4e). Patterns learned separately by sex and BMI group gave a small further gain "
    f"(mean R² {CD['r2_grp_mean']:.2f} vs. {CD['r2_clean_mean']:.2f}; p = {CD['p_grp_vs_pop']:.2g}), "
    f"consistent with BMI and sex mainly changing the amount of CFA rather than its scalp pattern. "
    f"The cleaning retained a median of {CD['hep_retained_median']*100:.0f}% of an injected "
    f"synthetic neural HEP and changed ongoing EEG power by a median of "
    f"{CD['psd_change_db_median']:.1f} dB (1–40 Hz), the cost of removing one of six spatial "
    f"dimensions.",
    styles["Body"]))

# ---------------------------------------------------------------------
# Discussion
# ---------------------------------------------------------------------
story.append(Paragraph("4. Discussion", styles["H1"]))
story.append(Paragraph(
    f"In {CFA['n_patients']:,} patients, by far the largest sample in which the cardiac field "
    f"artifact has been measured, two complementary estimators showed that a substantial "
    f"proportion of heartbeat-locked scalp EEG variance is shared with the ECG even after "
    f"standard QRS exclusion. The sample size allowed us to move beyond describing CFA as a "
    f"generic nuisance and to show that it is a variable property of each person and each "
    f"analysis: it ranged from negligible to near-complete between individual patients (Figure 1), "
    f"varied approximately {TOPO['highest_mean']/max(TOPO['lowest_mean'], 1e-6):.1f}-fold across "
    f"scalp sites (Figure 2c–d), increased with BMI, was higher in men than in women (Figure 3a–c), and "
    f"depended strongly on segment duration (Figure 3d–f). These factors are common grouping "
    f"variables or analysis choices in EEG research, so uncorrected CFA can create, mask, or "
    f"inflate group differences in heartbeat-locked EEG measures.",
    styles["Body"]))
story.append(Paragraph(
    "Although CFA is most prominent around the QRS complex, the conventional ±30–50 ms "
    "exclusion<super>6-8</super> did not remove it: excluding ±50 ms reduced mean ECG-shared "
    f"variance (R²) only from {CFA['r2_full_mean']:.2f} to {CFA['r2_excl_qrs_mean']:.2f}, so a "
    "substantial ECG-correlated component extends into the nominal analysis interval. The "
    "positive relationship between the model-free R² and the ICA-derived SNR indicates that both "
    "methods detect the same contamination. Neither estimator establishes that every residual "
    "heartbeat-locked deflection is artifactual, and genuine cortical heartbeat-evoked activity is "
    "well documented<super>5,6,10</super>. Nevertheless, uncorrected heartbeat-locked EEG variance "
    "cannot be assumed to be of cortical origin.",
    styles["Body"]))
story.append(Paragraph(
    f"BMI was positively correlated with CFA, in women and in men, and clinically obese patients "
    f"had the highest CFA of all diagnosis categories, differing significantly from every other "
    f"category. The two examples in Figure 1 illustrate the extremes: almost no CFA in a lean man "
    f"and near-complete CFA in a man with severe obesity. At first sight this is counterintuitive: "
    f"adipose tissue is a poor electrical conductor, with a conductivity roughly an order of "
    f"magnitude lower than that of muscle<super>24</super>, and subcutaneous fat attenuates the "
    f"surface ECG, so that obesity is associated with low QRS voltage<super>12,13</super>. If fat "
    f"acted only as an insulating layer, the cardiac field reaching the scalp should be smaller, "
    f"not larger. The resolution lies in what R² measures. R² is scale-invariant: it indexes how "
    f"closely the shape of the heartbeat-locked scalp average follows the shape of the ECG, not "
    f"the amplitude of either. A uniform attenuation of the cardiac field would scale the CFA "
    f"waveform (and the ECG lead itself) without changing their correlation. R² increases only "
    f"when the cardiac component grows relative to everything else in the heartbeat-locked "
    f"average, that is, relative to residual background EEG and genuine neural responses. Several "
    f"physical changes in obesity could increase this relative share. First, obesity is "
    f"associated with increased left-ventricular mass and a leftward, more horizontal electrical "
    f"axis of the heart<super>13</super>, which change the strength and orientation of the cardiac "
    f"dipole relative to the head; in the mastoid-referenced derivations used here, CFA reflects "
    f"the difference in cardiac potential between the scalp electrode and the mastoid, which "
    f"depends strongly on dipole orientation. Second, low-conductivity tissue does not only "
    f"attenuate volume currents but also redistributes them, so a thick insulating layer around "
    f"the thorax and neck may change which fraction of the cardiac field projects onto the head. "
    f"Third, tissue between the brain and the electrode attenuates cortical EEG at the same "
    f"electrode. A trivial explanation, more heartbeats per window, is unlikely: heart rate was "
    f"slightly lower at higher BMI (r = {BMIHR['r_bmi_bpm']:.2f}), and adjusting for heart rate "
    f"did not reduce the BMI coefficient ({BMIHR['bmi_coef_unadj']:.4f} unadjusted vs. "
    f"{BMIHR['bmi_coef_adj']:.4f} adjusted per kg/m², n = {BMIHR['n']:,}). Which of these "
    f"mechanisms dominates cannot be determined from the present data; an analysis of absolute "
    f"CFA amplitude, which would be expected to decrease with BMI if insulation dominated, and "
    f"direct measures of body composition and heart position would be needed. "
    f"At the level of the individual, the "
    f"correlation was weak (r = {BMI_STRAT['r_pearson']:.2f}), so BMI alone cannot predict how "
    f"much CFA a given recording contains; at the level of the group, however, a difference in "
    f"mean BMI between study groups, which is common in clinical comparisons, will translate into "
    f"a systematic difference in CFA.",
    styles["Body"]))
story.append(Paragraph(
    f"Sex had a robust effect: CFA was higher in men than in women, at every segment duration, "
    f"and the difference was unchanged after adjustment for BMI. Because women in this cohort had "
    f"a higher mean BMI, BMI would, if anything, have reduced the sex difference. The sex effect "
    f"may reflect differences in heart size and position, thoracic and head geometry, or body-fat "
    f"distribution, none of which were measured here; cardiac parameters such as stroke volume "
    f"have been proposed to influence heartbeat-locked responses<super>11</super>. Whatever its origin, sex differences are "
    f"among the most frequently reported group effects in EEG research, and studies that compare "
    f"men and women on heartbeat-locked measures should therefore quantify CFA per channel and "
    f"show that the difference survives its correction.",
    styles["Body"]))
story.append(Paragraph(
    f"Segment duration had the largest and most consistent effect. In the duration-matched "
    f"{WT['n']:,}-patient analysis, mean R² rose from {DOSE['lengths'][0]['mean']:.2f} at 5 min and "
    f"{DOSE['lengths'][1]['mean']:.2f} at 10 min to {DOSE['lengths'][-1]['mean']:.2f} at 60 min, "
    f"every step was significant, and R² increased between 5 and 60 min in "
    f"{WT_PAIR[(5, 60)]['pct_patients_increase']:.0f}% of patients. Longer segments contain more "
    f"heartbeats and yield a more stable heartbeat-locked average<super>14</super>, which allows shared EEG-ECG "
    f"structure to emerge from background EEG. The practical consequences are twofold. First, "
    f"short segments, including the 10-min segments of our main analysis, underestimate CFA, so a "
    f"low CFA estimate from a short recording does not show that the data are clean. Second, "
    f"studies or groups that differ in recording length, or in the number of usable heartbeats, "
    f"will differ in apparent CFA for purely methodological reasons. The segment length is "
    f"therefore not a neutral analysis parameter: it should be reported, justified, and, where "
    f"possible, matched between groups, and the stability of results should be checked across "
    f"durations. Because R² depends on the number of averaged heartbeats, between-group "
    f"differences in heart rate could also contribute to differences in R²; this was not examined "
    f"in the present analysis.",
    styles["Body"]))
story.append(Paragraph(
    "BMI is an anthropometric index rather than a direct measure of adiposity; no direct measures "
    "of fat mass, thoracic geometry, or electrode impedance were available, and diagnosis "
    "categories overlap. We therefore interpret these findings as evidence that BMI, clinical "
    "obesity, and sex are confounders of CFA, not as evidence of a causal effect of body fat. "
    "Likewise, the associations of age and diagnosis category with CFA may reflect anatomy, "
    "cardiac physiology, medication, comorbidity, or recording conditions.",
    styles["Body"]))
story.append(Paragraph(
    "In practice, EEG preprocessing should combine concurrent ECG recording, channel-level "
    "assessment of the R-peak-locked EEG-ECG relationship, and removal or regression of "
    "ECG-related components, followed by verification that heartbeat-locked variance in the "
    "cleaned signal is reduced relative to the uncleaned data while remaining above a "
    "non-heartbeat-locked noise floor. Reporting only the QRS mask or the selected ICA component is "
    "insufficient. We recommend that studies report the segment duration and number of "
    "heartbeats, match them between groups where possible, report channel-level CFA metrics before "
    "and after cleaning, and test whether results are robust to adjustment for BMI (or, "
    "preferably, direct body-composition measures) and sex. These recommendations extend recent "
    "methodological reviews of HEP analysis and reporting<super>8,9</super> to EEG analyses more "
    "generally.",
    styles["Body"]))
story.append(Paragraph(
    f"For recordings without ECG, the proof of concept in Section 3.7 suggests a practical route: "
    f"because the scalp pattern of CFA is consistent across people, a spatial filter learned from "
    f"a large cohort with ECG removed about {CD['pct_reduction_mean']:.0f}% of CFA in new patients "
    f"without any cardiac reference. The filter is specific to the montage and reference on which "
    f"it was trained, it removes one spatial dimension of the EEG, and it does not remove all CFA. "
    f"It should therefore be validated on other montages, and it is best regarded as a first "
    f"step towards a trained tool that combines learned cardiac patterns with information about "
    f"the participant, such as BMI and sex, and the recording length.",
    styles["Body"]))

story.append(Paragraph("5. Conclusions", styles["H1"]))
story.append(Paragraph(
    f"In {CFA['n_patients']:,} patients, cardiac field artifact was a substantial and systematic "
    f"component of scalp EEG<super>1,2</super>, consistent with its long-recognised role as an EEG "
    f"artifact<super>3,4</super>: the ECG explained a mean R² of {CFA['r2_excl_qrs_mean']:.2f} of "
    f"heartbeat-locked EEG variance outside the QRS-exclusion window, and ICA-based removal of the "
    f"ECG-related component reduced this variance by a median of "
    f"{ICA['hep_pct_drop_median']*100:.0f}%. CFA increased with BMI, was higher in men than in "
    f"women independently of BMI, varied across electrodes, and increased significantly with "
    f"segment duration up to the longest window examined (60 min). CFA should therefore be treated "
    f"as a participant-, channel-, and duration-dependent source of noise in EEG. Robust inference "
    f"requires ECG-informed channel-level cleaning with quantitative before-and-after "
    f"validation<super>8,9</super>, reported and justified segment lengths<super>14</super>, and adjustment for "
    f"BMI and sex. Without these safeguards, apparent group differences in EEG may partly reflect "
    f"the electrical field of the heart rather than brain activity<super>6,11</super>.",
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
    f"composition was nearly constant (see Section 3.5). Sixth, the chance-level "
    f"(pseudo-event) R² was estimated only in a subsample of {ZL['non_locked']['n']} patients and "
    f"only over the full epoch (see Section 3.3); a null for the outside-QRS estimate was not computed. "
    f"Seventh, windows were not selected by sleep stage, so vigilance state was not controlled and "
    f"may differ between groups and segment durations; CFA did not differ between stages in a "
    f"staged subsample, but that subsample was small (n = {SENS['n_patients']}; see Section 3.1). Eighth, the cohort was predominantly a "
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
    "data use agreement. No attempt was made to re-identify participants.",
    styles["Body"]))

story.append(Paragraph("Data availability statement", styles["H2"]))
story.append(Paragraph(
    "The Human Sleep Project dataset is publicly available, subject to a data use agreement, from "
    "the Brain Data Science Platform (<font face='Courier'>https://bdsp.io/content/hsp/</font>). Analysis code is available from "
    "the corresponding author upon reasonable request (nircafri@mail.tau.ac.il).",
    styles["Body"]))

story.append(Paragraph("Funding", styles["H2"]))
story.append(Paragraph(
    "",
    styles["Body"]))

story.append(Paragraph("Declaration of competing interests", styles["H2"]))
story.append(Paragraph(
    "The authors declare no competing interests.",
    styles["Body"]))


# ---------------------------------------------------------------------
# Supplementary analysis: window-length and sleep-stage sensitivity
# ---------------------------------------------------------------------
story.append(PageBreak())
story.append(Paragraph("Supplementary analysis", styles["H1"]))

story.append(Paragraph("S1. Cohort composition", styles["H2"]))
story.append(KeepTogether([
    fig("figS1_cohort.png", 6.6 * inch),
    Paragraph(
    f"<b>Figure S1.</b> Cohort composition of the {COHORT['n_demographics']:,} Human Sleep Project patients. "
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
        "<b>Table S1.</b> Diagnosis categories and the case-insensitive keywords matched against EHR "
        "diagnosis descriptions. Keyword matching is inclusive; for example, \"infarction\" also "
        "matches myocardial infarction.",
        styles["Caption"]),
    dx_table,
]))

story.append(Paragraph("S2. Variance and entropy vs. a non-heartbeat-locked noise floor", styles["H2"]))
story.append(KeepTogether([
    fig("figS3_post_ica_variance.png", 6.6 * inch),
    Paragraph(
    f"<b>Figure S2.</b> Absolute HEP variance before vs. after excluding the ECG-related "
    f"ICA component, alongside a non-heartbeat-locked control (n = {S['post_ica_variance']['n_patients']:,} "
    f"ICA patients with at least one well-covered site; "
    f"{S['post_ica_variance']['n_patients_control']:,}-patient control subsample, Section 2.6; "
    f"medians on a log axis are reported rather than means). "
    f"(a) Pooled over channels at the four core frontal-central electrodes "
    f"({'/'.join(S['post_ica_variance']['core_electrodes'])}): median "
    f"{S['post_ica_variance']['core_pre_median']:.2f} µV² pre-ICA to "
    f"{S['post_ica_variance']['core_post_median']:.2f} µV² post-ICA (a "
    f"{S['post_ica_variance']['core_pct_drop']:.0f}% drop) vs. "
    f"{S['post_ica_variance']['core_non_locked_median']:.2f} µV² for the non-locked control (the "
    f"same windows re-epoched around random pseudo-events instead of R-peaks). "
    f"(b) The same three-way comparison broken out per electrode, all six sites (see Section 2.5). "
    f"(c) Spectral entropy (normalised Shannon entropy of the evoked waveform's Welch power spectrum, "
    f"0–1) of the same three conditions, core electrode average (n = {S['post_ica_variance']['n_patients_entropy']:,} "
    f"patients, separate subsample). (d) Spectral entropy per electrode.",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"Removal of the ECG-related component reduced median core-electrode HEP variance by "
    f"{S['post_ica_variance']['core_pct_drop']:.0f}%; this ratio of group medians is larger than, and not "
    f"directly comparable to, the median per-channel reduction reported in Section 3.3 "
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
    fig("figS4_crosscorr_mi.png", 6.6 * inch),
    Paragraph(
    f"<b>Figure S3.</b> EEG-ECG cross-correlation and mutual information for pre-ICA, post-ICA, and "
    f"non-heartbeat-locked control conditions (n = {CC['n_patients']:,} patients; separate "
    f"subsample in which ICA was refitted to obtain paired evoked waveforms; see Section 2.6). In the control "
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
    f"regression estimator (see Section 2.3). Peak correlation decreased from pre-ICA "
    f"(mean |r| = {CC['core_peak_r_pre_mean']:.2f}) to post-ICA "
    f"(mean |r| = {CC['core_peak_r_post_mean']:.2f}) to the non-locked control "
    f"(mean |r| = {CC['core_peak_r_non_locked_mean']:.2f}), and mutual information decreased "
    f"correspondingly, from {CC['core_mi_pre_median']:.2f} to {CC['core_mi_post_median']:.2f} to "
    f"{CC['core_mi_non_locked_median']:.2f} nats (panel c). Post-ICA values remained above the "
    f"non-locked floor at every well-covered electrode (panels S3b, S3d), indicating that a "
    f"non-trivial EEG-ECG relationship persists after cleaning, in agreement with Section 3.3 and "
    f"Supplementary Figure S2. Mutual information, which is also sensitive to nonlinear dependence, "
    f"followed the same ordering as peak correlation, giving no indication of substantial EEG-ECG "
    f"dependence beyond that captured by the linear measures of Section 3.3.",
    styles["Body"]))



PSD = S["psd_comparison"]
story.append(Paragraph("S4. Power spectral density: pre-ICA, post-ICA, non-locked control, and ECG", styles["H2"]))
story.append(Paragraph(
    f"To complement the correlation- and variance-based summaries, we compared the spectral content "
    f"of each evoked waveform. Welch power spectral density was computed for the pre-ICA, post-ICA, "
    f"and non-locked evoked waveforms described in Section S3 and for the patient’s ECG evoked "
    f"average, in the same {PSD['n_patients']:,}-patient subsample.",
    styles["Body"]))
story.append(KeepTogether([
    fig("figS5_psd_comparison.png", 6.6 * inch, crop=False),
    Paragraph(
    f"<b>Figure S4.</b> Power spectral density (Welch, dB) of the pre-ICA, post-ICA, "
    f"non-heartbeat-locked control, and ECG evoked waveforms (n = {PSD['n_patients']:,} patients). "
    f"(a) Group average: core four-electrode ({'/'.join(PSD['core_electrodes'])}) average, all four "
    f"conditions overlaid (mean \u00b1 SEM). (b) Per-electrode: the same four-way overlay for each "
    f"of the six well-covered sites (see Section 2.5).",
        styles["Caption"]),
]))
story.append(Paragraph(
    "The ECG spectrum was broadband and structured, dominated by the sharp QRS transient, whereas "
    "the non-locked EEG control lacked this QRS-dominated structure. Pre-ICA and post-ICA EEG spectra lay between these two "
    "references and appeared visually closer to the ECG than the non-locked floor did, providing "
    "spectral-domain support for the conclusions of Section 3.3 and Section S3. Spectral shape was "
    "visually similar across the six electrodes (panel b), suggesting that CFA varies across the "
    "scalp mainly in magnitude (Figure 2c) rather than in spectral content. Spectra are shown up "
    "to 100 Hz, the upper edge of the analysis band-pass filter (see Section 2.2).",
    styles["Body"]))

CONF = S["sex_bmi_confound"]
story.append(Paragraph("S5. Adjustment of the sex effect for BMI", styles["H2"]))
story.append(Paragraph(
    f"Male patients had higher CFA R² than female patients (see Section 3.4), whereas mean BMI was higher in "
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
    fig("figS6_sex_bmi_confound.png", 4.5 * inch),
    Paragraph(
    f"<b>Figure S5.</b> Male-versus-female CFA R² (outside QRS) OLS coefficient, unadjusted vs. "
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
    '1. Dirlich G, Vogl L, Plaschke M, Strian F. Cardiac field effects on the EEG. '
    'Electroencephalogr Clin Neurophysiol. 1997;102(4):307-315.',
    '2. Dirlich G, Dietl T, Vogl L, Strian F. Topography and morphology of heart action-related '
    'EEG potentials. Electroencephalogr Clin Neurophysiol. 1998;108(3):299-305.',
    '3. Urigüen JA, Garcia-Zapirain B. EEG artifact removal—state-of-the-art and guidelines. J '
    'Neural Eng. 2015;12(3):031001. doi:10.1088/1741-2560/12/3/031001.',
    '4. Jiang X, Bian G-B, Tian Z. Removal of artifacts from EEG signals: a review. Sensors. '
    '2019;19(5):987. doi:10.3390/s19050987.',
    '5. Schandry R, Sparrer B, Weitkunat R. From the heart to the brain: a study of heartbeat '
    'contingent scalp potentials. Int J Neurosci. 1986;30:261-275.',
    '6. Park H-D, Blanke O. Heartbeat-evoked cortical responses: underlying mechanisms, '
    'functional roles, and methodological considerations. NeuroImage. 2019;197:502-511.',
    '7. Coll M-P, Hobson H, Bird G, Murphy J. Systematic review and meta-analysis of the '
    'relationship between the heartbeat-evoked potential and interoception. Neurosci Biobehav '
    'Rev. 2021;122:190-200.',
    '8. Steinfath TP, et al. Heartbeat-evoked responses in M/EEG: a systematic review of methods '
    'with suggestions for analysis and reporting. Psychophysiology. 2026;63(4):e70297. '
    'doi:10.1111/psyp.70297.',
    '9. Virjee R-I, Kandasamy R, Garfinkel SN, Carmichael DW, Yogarajah M. Review of methods to '
    'derive the heartbeat-evoked potential: past practices and future directions. Soc Cogn Affect'
    ' Neurosci. 2026:nsag057. doi:10.1093/scan/nsag057.',
    '10. Kern M, Aertsen A, Schulze-Bonhage A, Ball T. Heart cycle-related effects on event-'
    'related potentials, spectral power changes, and connectivity patterns in the human ECoG. '
    'NeuroImage. 2013;81:178-190.',
    '11. Buot A, Azzalini D, Chaumon M, Tallon-Baudry C. Does stroke volume influence heartbeat '
    'evoked responses? Biol Psychol. 2021;165:108165. doi:10.1016/j.biopsycho.2021.108165.',
    '12. Tochikubo O, Miyajima E, Shigemasa T, Ishii M. Relation between body fat-corrected ECG '
    'voltage and ambulatory blood pressure in patients with essential hypertension. Hypertension.'
    ' 1999;33(5):1159-1163. doi:10.1161/01.HYP.33.5.1159.',
    '13. Fraley MA, Birchem JA, Senkottaiyan N, Alpert MA. Obesity and the electrocardiogram. '
    'Obes Rev. 2005;6(4):275-281.',
    '14. Boudewyn MA, Luck SJ, Farrens JL, Kappenman ES. How many trials does it take to get a '
    'significant ERP effect? It depends. Psychophysiology. 2018;55(6):e13049.',
    '15. Li Q, Wen S, Sun H, Ganglberger W, Tripathi A, Turley N, et al.; Westover MB. The Human '
    'Sleep Project (HSP). Brain Data Science Platform. 2026. doi:10.60508/m3sw-rz13.',
    '16. Gramfort A, et al. MEG and EEG data analysis with MNE-Python. Front Neurosci. '
    '2013;7:267.',
    '17. Hyvarinen A, Oja E. Independent component analysis: algorithms and applications. Neural '
    'Netw. 2000;13(4-5):411-430.',
    '18. Ablin P, Cardoso J-F, Gramfort A. Faster independent component analysis by '
    'preconditioning with Hessian approximations. IEEE Trans Signal Process. '
    '2018;66(15):4040-4049.',
    '19. Kraskov A, Stögbauer H, Grassberger P. Estimating mutual information. Phys Rev E. '
    '2004;69(6):066138.',
    '20. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful '
    'approach to multiple testing. J R Stat Soc Series B. 1995;57(1):289-300.',
    '21. Vallat R, Walker MP. An open-source, high-performance tool for automated sleep staging. '
    'eLife. 2021;10:e70092. doi:10.7554/eLife.70092.',
    '22. Pérez JJ, Guijarro E, Barcia JA. Suppression of the cardiac electric field artifact from'
    ' the heart action evoked potential. Med Biol Eng Comput. 2005;43(5):572-581.',
    '23. Arnau S, Sharifian F, Wascher E, Larra MF. Removing the cardiac field artefact from the '
    'EEG using neural network regression. Psychophysiology. 2023:e14323. doi:10.1111/psyp.14323.',
    '24. Gabriel S, Lau RW, Gabriel C. The dielectric properties of biological tissues: II. '
    'Measurements in the frequency range 10 Hz to 20 GHz. Phys Med Biol. 1996;41(11):2251-2269.',
]
for r in refs:
    story.append(Paragraph(r, styles["Ref"]))

doc = SimpleDocTemplate(
    OUT_PDF, pagesize=LETTER,
    topMargin=0.8 * inch, bottomMargin=0.8 * inch,
    leftMargin=0.9 * inch, rightMargin=0.9 * inch,
    title="Cleaning the Heart's Noise from Brain Signals: A Large-Cohort Study of Cardiac Field Artifact in EEG",
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
