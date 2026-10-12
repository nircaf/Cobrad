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
    first = int(np.argmax(ink))  # first title row
    top, r = 0, first
    while crop and r < len(ink):
        gap_end = r
        while gap_end < len(ink) and not ink[gap_end]:
            gap_end += 1
        if gap_end - r >= min_gap and r > first:
            # only a short top block is a suptitle; a tall one is a panel row (e.g. Figure 4a-b)
            if r - first < 0.15 * len(ink):
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
CD_P_STR = "&lt; 1e-300" if CD["p_pre_vs_clean"] == 0 else f"= {CD['p_pre_vs_clean']:.1e}"
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
    "Cardiac Field Artifact Removal (CFAR): A Large-Cohort Study Cleaning the Heart's Noise from Brain Signals",
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
    f"The heart's electrical field reaches the scalp as a cardiac field artifact (CFA) in every EEG "
    f"recording, adding noise to any analysis and confounding heartbeat-evoked potentials (HEPs). "
    f"CFA has been studied only in small samples and is usually assumed constant across people and "
    f"removed by excluding the QRS complex. We quantified CFA in {CFA['n_patients']:,} "
    f"polysomnography patients ({CFA['n_rows']:,} channel-recordings) from the Human Sleep Project "
    f"by regressing each channel's R-peak-locked EEG average on the ECG average outside a ±50 ms QRS "
    f"interval. The ECG explained a mean R² of {CFA['r2_excl_qrs_mean']:.2f}; removing the "
    f"ECG-related independent component reduced heartbeat-locked variance by a median of "
    f"{ICA['hep_pct_drop_median']*100:.0f}%. CFA ranged from negligible to near-complete between "
    f"patients and varied across electrodes. It increased with BMI "
    f"(r = {STRAT['bmi']['r_pearson']:.2f}, p = {STRAT['bmi']['p_pearson']:.1e}), was higher in "
    f"men than women ({STRAT['sex_means']['Male']:.2f} vs. {STRAT['sex_means']['Female']:.2f}, "
    f"p = {STRAT['p_sex_mannwhitney']:.1e}) independently of BMI, and rose with segment duration "
    f"from {DOSE['lengths'][0]['mean']:.2f} at 5 min to {DOSE['lengths'][-1]['mean']:.2f} at "
    f"{DOSE['lengths'][-1]['window_minutes']:.0f} min (Friedman p {WT_P_STR}). Because its scalp "
    f"pattern was consistent across patients, we developed CFAR (Cardiac Field Artifact Removal), "
    f"an ECG-informed model that, once trained, removes CFA from EEG without ECG: in "
    f"{CD['n_test']:,} held-out patients, it achieved {CD_GAP:.0f}% of the cleaning possible with the "
    f"patient's own ECG (R² {CD['r2_pre_mean']:.2f} to {CD['r2_clean_mean']:.2f}, vs. "
    f"{CD['r2_own_mean']:.2f}) and cut the cardiac share of ongoing EEG "
    f"variance from {CD['frac_pre_mean']*100:.1f}% to {CD['frac_clean_mean']*100:.1f}%, retaining "
    f"{CD['hep_retained_median']*100:.0f}% of a simulated neural HEP. EEG studies "
    f"should estimate CFA per channel, report segment length, and account for BMI and sex; CFAR can "
    f"clean recordings without ECG.",
    styles["Body"]))
story.append(Paragraph("Highlights", styles["H2"]))
for h in [
    f"In {CFA['n_patients']:,} patients, the ECG explained {CFA['r2_excl_qrs_mean']*100:.0f}% of "
    f"heartbeat-locked EEG variance after QRS exclusion.",
    f"CFA rose with BMI (r = {STRAT['bmi']['r_pearson']:.2f}, p = {STRAT['bmi']['p_pearson']:.1e}) "
    f"and segment duration (R² {DOSE['lengths'][0]['mean']:.2f} at 5 min, "
    f"{DOSE['lengths'][-1]['mean']:.2f} at 60 min).",
    f"CFA was higher in men ({STRAT['sex_means']['Male']:.2f} vs. {STRAT['sex_means']['Female']:.2f}), "
    f"independent of BMI (p = {S['sex_bmi_confound']['sex_adj_p']:.1e}).",
    f"Without ECG, CFAR achieved {CD_GAP:.0f}% of ECG-based cleaning, and removed more CFA in obese "
    f"patients (p = {CD['p_removed_obese_vs_lean']:.1g}).",
]:
    story.append(Paragraph("• " + h, styles["Body"]))
story.append(Paragraph(
    "Keywords: cardiac field artifact; CFAR; EEG artifact; ECG; body mass index; sex differences; "
    "segment duration; heartbeat-evoked potential; independent component analysis; "
    "polysomnography", styles["Kw"]))
story.append(PageBreak())

# ---------------------------------------------------------------------
# Introduction
# ---------------------------------------------------------------------
story.append(Paragraph("1. Introduction", styles["H1"]))
story.append(Paragraph(
    "The heart is one of the body's strongest bioelectrical generators, and its field is "
    "volume-conducted to the scalp. Every EEG recording therefore contains a cardiac field artifact "
    "(CFA): a heartbeat-locked deflection of non-cortical origin<super>1-4</super>. CFA adds "
    "structured noise to any EEG measure<super>3,4</super> and is preserved, not averaged out, in "
    "R-peak-locked analyses. The heartbeat-evoked potential (HEP), used to study cardiac "
    "interoception<super>5-7</super>, is the clearest example: CFA and cortical responses to the "
    "heartbeat share the same time-locking event<super>1,6</super>. The standard mitigation is to "
    "exclude ±30–50 ms around the QRS complex and, in some studies, to remove an ECG-related "
    "independent component<super>6-8</super>.",
    styles["Body"]))
story.append(Paragraph(
    "CFA has been characterised only in single cohorts of a few tens of "
    "participants<super>1,2</super>, too small to detect modest differences between people. It is "
    "therefore usually treated as a constant property of the recording, removed by a fixed QRS "
    "exclusion. Recent reviews note that heterogeneous CFA handling limits reproducibility and "
    "clinical interpretation<super>7-9</super>. If CFA differs systematically between people or "
    "analysis choices, a group difference in heartbeat-locked EEG may reflect cardiac "
    "electrophysiology, tissue conductivity, or electrode geometry rather than brain "
    "activity<super>2,6,10,11</super>.",
    styles["Body"]))
story.append(Paragraph(
    "Three factors are of particular interest. First, body composition alters the conduction path "
    "from heart to scalp: subcutaneous fat attenuates surface-ECG voltage, correcting voltage for "
    "body fat changes its relation to cardiac structure<super>12</super>, and obesity is associated "
    "with low QRS voltage and a shifted cardiac axis<super>13</super>. BMI, though an indirect "
    "measure of adiposity, is available in most clinical datasets. Second, men and women differ in "
    "thoracic anatomy, heart size, and fat distribution, so CFA may differ by sex, a common grouping "
    "variable. Third, CFA is estimated from averages over heartbeats, and averages over fewer events "
    "are less reliable<super>14</super>, so apparent contamination may depend on segment duration. "
    "None of these factors has been examined at the population level.",
    styles["Body"]))
story.append(Paragraph(
    f"We quantified CFA in clinical polysomnography recordings from {CFA['n_patients']:,} patients "
    f"in the Human Sleep Project, a public, curated dataset<super>15</super>, the largest sample in "
    f"which CFA has been measured. We used two estimators: a model-free regression of each channel's "
    f"R-peak-locked EEG average on the patient's R-peak-locked ECG average, and ECG-informed ICA. We "
    f"show the range of CFA in individual patients, map its scalp distribution, and test its "
    f"dependence on BMI, sex, age, and segment duration (5–60 min). We hypothesised that "
    f"ECG-related variance would persist outside the QRS interval, increase with BMI, differ by sex, "
    f"and increase with segment duration. Finally, we developed CFAR (Cardiac "
    f"Field Artifact Removal), a model trained on the cohort's ECG-informed CFA patterns that "
    f"removes CFA from EEG recorded without ECG.",
    styles["Body"]))

# ---------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------
story.append(Paragraph("2. Methods", styles["H1"]))
story.append(Paragraph("2.1 Cohort and recordings", styles["H2"]))
story.append(Paragraph(
    f"This is a secondary analysis of existing polysomnography recordings from the Human Sleep "
    f"Project<super>15</super>, a public, curated dataset of clinical recordings with linked "
    f"electronic health records (EHR), distributed through the Brain Data Science Platform. BMI was "
    f"computed from each patient's median EHR height and weight; values outside 10–80 kg/m² were "
    f"excluded. EEG channels were identified by matching labels to 10-20/10-10 electrode names, "
    f"which excluded intracranial and auxiliary channels; at least two were required. The ECG "
    f"channel was the first label matching ECG/EKG. Cohort characteristics are in Section 3.1.",
    styles["Body"]))
story.append(Paragraph("2.2 Window selection, R-peak detection, and preprocessing", styles["H2"]))
story.append(Paragraph(
    "For the main analysis, one 10-min window per recording was selected by a seeded random search "
    "(uniform start times; up to 10 attempts). R-peaks were detected on the ECG before the analysis "
    "filter: the median-subtracted signal was band-pass filtered at 5–25 Hz (third-order zero-phase "
    "Butterworth) and rectified, making detection polarity-independent, and peaks were identified "
    "with a minimum spacing of 300 ms and a prominence ≥3 × the median absolute deviation. A window "
    "was accepted if (i) ≥70% of EEG channels passed quality criteria (≥99.9% finite samples; SD "
    "0.1–500 µV; peak-to-peak ≤5,000 µV; &lt;20% flat samples), (ii) mean heart rate was 35–180 "
    "beats/min, and (iii) ≥70% of R-R intervals were 0.33–2.0 s. EEG and ECG were then band-pass "
    "filtered from 1 Hz to min(100 Hz, Nyquist − 0.5 Hz) (zero-phase FIR, "
    "MNE-Python<super>16</super>; 86% of recordings were sampled at 200 Hz) and analysed in µV at "
    "native sampling rate, without re-referencing, interpolation, or amplitude-based rejection, so "
    "that CFA was estimated in minimally processed data. For the duration analysis, the procedure "
    "was repeated with 5-, 20-, 30-, 45-, and 60-min windows (Section 2.7). Parameters are "
    "summarised in Table 1.",
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
    "HEP epochs spanned −300 to 400 ms around each R-peak, without baseline correction; epochs "
    "beyond the window edges were discarded, and recordings with fewer than 20 epochs were excluded "
    "(median 940 epochs per 10-min window). For each patient and channel, the R-peak-locked EEG "
    "average was compared with the ECG average over the same epochs. The squared zero-lag Pearson "
    "correlation (R²) between the two waveforms, the fraction of HEP variance shared with the ECG, "
    "was the CFA index. R² was computed over the full epoch and outside a ±50 ms QRS-exclusion "
    "window<super>8</super>. This estimator does not depend on ICA isolating a cardiac source; it "
    "tests directly how far the averaged scalp waveform is a scaled, offset copy of the averaged ECG. "
    "For patient-level analyses (Section 2.7), R² was averaged across the patient's channels at the "
    "six well-covered sites (Section 2.5) and across recordings (patient-mean CFA R²).",
    styles["Body"]))
story.append(Paragraph("2.4 Model-based CFA estimator: ECG-informed ICA", styles["H2"]))
story.append(Paragraph(
    "In parallel, ICA<super>17</super> (extended Picard<super>18</super>; min(15, channels − 1) "
    "components; 500 iterations; fixed seed) was fitted to the filtered EEG of the same window. "
    "ECG-related components were identified by correlation with the ECG channel (MNE-Python "
    "default threshold<super>16</super>); if none exceeded it, the highest-scoring component was "
    "used (two components were flagged in 49 of 13,624 recordings). Each component's "
    "mixing-weighted contribution to each channel was evaluated on the R-peak-locked average, "
    "quantifying the fraction of heartbeat-evoked signal it explains. Removal was also measured "
    "directly, as the reduction in each channel's HEP variance after back-projection without the "
    "component, 100 × (pre − post)/pre. For Figure 2b, the ratio of component to residual variance "
    "(SNR) was computed per channel-recording.",
    styles["Body"]))
story.append(Paragraph("2.5 Channel canonicalisation and minimum coverage", styles["H2"]))
story.append(Paragraph(
    "Channel labels were mapped to their scalp-side 10-20 site; reference-only labels were excluded. "
    "Figure 2c–d includes sites recorded in at least "
    f"{TOPO['min_patients']} patients. Analyses comparing conditions or estimators at common "
    "electrodes (Sections 3.4–3.6, S2–S4) used sites present in ≥50% of patients, which retained "
    "six sites (F3, F4, C3, C4, O1, O2; each in 96–100% of patients), the dataset's predominant "
    "montage.",
    styles["Body"]))

story.append(Paragraph("2.6 EEG-ECG cross-correlation and mutual information", styles["H2"]))
story.append(Paragraph(
    "The EEG-ECG relationship was further characterised in a separate subsample of 150 patients "
    f"(Figures S3–S4; {S['crosscorr_mi']['n_patients']} with usable data). Lag-resolved "
    "cross-correlation was the Pearson correlation between each channel's evoked waveform and the "
    "ECG evoked average at lags of &plusmn;100 ms in 5-ms steps; the peak absolute correlation and "
    "its lag were reported after per-channel sign alignment, because reference polarity is "
    "arbitrary. Mutual information was estimated with the Kraskov k-nearest-neighbour "
    "estimator<super>19</super> to capture linear and nonlinear dependence. Both were computed "
    "pre-ICA, post-ICA, and for a non-heartbeat-locked control in which the same window was "
    "re-epoched around an equal number of random pseudo-events; in this subsample, ICA was refitted "
    "and R-peaks were re-detected on the filtered ECG. For the power spectral density (PSD) "
    "comparison (Figure S4), the ECG evoked average was added as a fourth condition, and Welch PSD "
    "(dB) was computed at the six well-covered electrodes. Two further random subsamples supported "
    f"Figure S2: in one ({S['post_ica_variance']['n_patients_control']:,} patients), ICA windows "
    "were re-epoched around pseudo-events, without refitting, to estimate a variance noise floor; "
    f"in the other ({S['post_ica_variance']['n_patients_entropy']:,} patients), ICA was refitted and "
    "the spectral entropy of each evoked waveform (normalised Shannon entropy of its Welch spectrum) "
    "was computed for the three conditions.",
    styles["Body"]))
story.append(Paragraph("2.7 Statistics", styles["H2"]))
story.append(Paragraph(
    "Diagnoses were assigned to 15 predefined categories by case-insensitive keyword matching on EHR "
    "diagnosis descriptions (Table S1); categories overlap. EHR-linked patients without a recorded "
    "diagnosis formed the no-diagnosis group, which included patients from a site without diagnosis "
    "tables (see Limitations); patients without an EHR link were excluded from diagnosis analyses. "
    "Age was tested as a continuous variable (Pearson correlation) and by tertile (one-way ANOVA). "
    "Sex and presence of a diagnosis were tested with the Mann-Whitney U test. Diagnosis categories "
    "(Section 3.6) were compared with the Kruskal-Wallis test and pairwise Mann-Whitney tests, "
    "corrected by the Benjamini-Hochberg false discovery rate (FDR)<super>20</super>. BMI was "
    "tested with Pearson correlation, using the longest available window per patient (30 min for "
    "98%) to keep one observation per patient; absolute R² in the BMI and BMI-adjusted sex analyses "
    "is therefore not comparable with the 10-min estimates. BMI correlations were also computed by "
    "sex, and a BMI-by-sex interaction term in an ordinary least-squares (OLS) model tested whether "
    "slopes differed. A further OLS model compared the sex coefficient with and without BMI "
    "adjustment. Model-free R² and log ICA SNR (Section 2.4) were correlated across "
    "channel-recordings (Pearson; Figure 2b). Segment duration (Section 3.5) was tested in patients "
    "with data at all six durations (5–60 min), using all EEG channels (Figure 3d shows "
    "channel-level means). Patient-mean R² was compared across durations with the Friedman test "
    "(effect size, Kendall's W), and consecutive durations, 5 vs. 60 min, and 10 vs. 60 min with "
    "Holm-corrected paired Wilcoxon signed-rank tests. Sexes and diagnosis groups were compared at "
    "each duration with Mann-Whitney tests. Windows of each length were drawn with the same seeded "
    "procedure, so shorter windows were usually nested within longer ones; the 45- and 60-min "
    "analyses used one recording per patient. Sample sizes differ because analyses need different "
    f"data: model-free estimator, all patients with a valid window ({CFA['n_patients']:,}); ICA, "
    f"patients with a converged decomposition ({ICA['n_patients']:,}); age, sex, and diagnosis, "
    f"EHR-linked patients with known age and at least one well-covered site "
    f"({STRAT['n_with_age']:,}); BMI, patients with a plausible BMI ({STRAT['bmi']['n']:,}); "
    f"duration, patients with data at all six durations ({DOSE['n_common_patients']:,}). For "
    f"illustration (Figure 1), two patients with measured BMI and data from the same recording at "
    f"all durations were chosen at the low and high ends of BMI and CFA; they were not used for "
    f"inference. To test sleep stage (Section 3.1), a random subsample of {SENS['n_patients']} "
    f"patients was staged in 30-s epochs with YASA<super>21</super>; for wake, light sleep "
    f"(N1+N2), N3, and REM, up to {SENS['draws_per_cell']} windows per duration "
    f"({', '.join(str(m) for m in SENS['window_lengths_min'])} min) were drawn from contiguous "
    f"stretches of each stage, and CFA R² was computed as in Section 2.3. Stage and duration were "
    f"compared with one-way ANOVA.",
    styles["Body"]))
story.append(Paragraph("2.8 CFAR: ECG-free cardiac field artifact removal", styles["H2"]))
story.append(Paragraph(
    f"Many EEG datasets lack an ECG. We therefore developed CFAR (Cardiac Field Artifact Removal), "
    f"an ECG-informed model: the ECG is needed to train it, not to apply it. Single-beat CFA was too weak to detect heartbeats from the EEG, "
    f"but CFA has a largely fixed scalp pattern, which allows cleaning without locating beats. In "
    f"recordings with the standard six-channel, mastoid-referenced montage (F3, F4, C3, C4, O1, "
    f"O2), each patient's CFA pattern (first singular vector of the R-peak-locked EEG average) was "
    f"computed, sign-aligned, and averaged, weighted by the patient's CFA R², into one population "
    f"pattern. All {CD['n_test']:,} eligible patients were used with {CD['n_folds']}-fold "
    f"cross-fitting: each patient was cleaned with a pattern learned from the other folds (about "
    f"{CD['n_train']:,} patients), projected out of every sample of the continuous EEG "
    f"(signal-space projection). The ECG was used only for scoring: CFA R² (outside QRS) before and "
    f"after cleaning, compared with a gold standard that projects out the patient's own ECG-derived "
    f"pattern. Cardiac noise in the ongoing EEG was quantified as the share of continuous EEG "
    f"variance carried by the ECG-locked template overlap-added at every beat. To measure "
    f"preservation of brain activity, a synthetic neural HEP (Gaussian, −{CD_INJ_UV:.0f} µV at "
    f"+300 ms, frontal &gt; central &gt; occipital) was added before cleaning and the retained "
    f"fraction was measured; the cost to ongoing EEG was the change in Welch power (1–40 Hz). To "
    f"test whether participant information helps, a ridge regression trained in the same folds "
    f"predicted each patient's pattern from BMI (median-imputed, with a missing indicator), age, "
    f"sex, number of diagnoses, and diagnosis categories.",
    styles["Body"]))
story.append(Paragraph("3. Results", styles["H1"]))
story.append(Paragraph("3.1 Cohort", styles["H2"]))
sex_str = ", ".join(f"{k} n={v:,}" for k, v in COHORT["sex_counts"].items())
top3_dx = list(COHORT["top_diagnoses"].items())[:3]
top3_str = "; ".join(f"{k} (n={v:,})" for k, v in top3_dx)
story.append(Paragraph(
    f"Demographics were available for {COHORT['n_age']:,} of the {CFA['n_patients']:,} patients "
    f"(median age {COHORT['age_median']:.0f} years, range "
    f"{COHORT['age_min']:.0f}–{COHORT['age_max']:.0f}; {sex_str}). The EHR-linked cohort was "
    f"clinically referred, with a high diagnostic burden; the most prevalent categories were "
    f"{top3_str} (Figure S1).",
    styles["Body"]))
STG = SENS["by_stage_mean"]
story.append(Paragraph(
    f"Windows were placed at random across the recording (Section 2.2), so they sampled wake and "
    f"sleep stages roughly in proportion to their duration, and cohort estimates describe CFA over "
    f"the whole night. In {SENS['n_patients']} patients staged with YASA<super>21</super> (Section "
    f"2.7), mean CFA R² was similar across stages (wake {STG['W']:.2f}, light sleep "
    f"{STG['light_sleep']:.2f}, N3 {STG['N3']:.2f}, REM {STG['R']:.2f}; ANOVA "
    f"p = {SENS['p_stage_anova']:.2f}), whereas segment duration had a significant effect in the "
    f"same data (p = {SENS['p_length_anova']:.1e}). The main conclusions are therefore not specific "
    f"to one sleep stage.",
    styles["Body"]))

story.append(Paragraph("3.2 CFA in individual patients", styles["H2"]))
EXA, EXB = EX
story.append(KeepTogether([
    fig("fig1_patient_examples.png", 4.8 * inch, crop=False),
    Paragraph(
    f"<b>Figure 1.</b> CFA in two example patients. Head maps show per-electrode CFA R² (variance "
    f"of the R-peak-locked EEG average explained by the ECG average, outside the ±50 ms QRS window) "
    f"at the {EXA['n_sites']} recorded electrodes, from a 5-min (left) and a 60-min (right) segment "
    f"of the same recording; colour is interpolated between electrodes. Patient A "
    f"({EXA['sex'].lower()}, {EXA['age']:.0f} years, BMI {EXA['bmi']:.1f} kg/m²): mean R² "
    f"{EXA['mean_r2']['5']:.2f} at 5 min, {EXA['mean_r2']['60']:.2f} at 60 min. Patient B "
    f"({EXB['sex'].lower()}, {EXB['age']:.0f} years, BMI {EXB['bmi']:.1f} kg/m²): "
    f"{EXB['mean_r2']['5']:.2f} and {EXB['mean_r2']['60']:.2f}. Patients were chosen from "
    f"recordings with at least 10 electrodes at the low and high ends of BMI; individually, BMI "
    f"explains little of the variation in CFA (Section 3.4).",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"Figure 1 shows how CFA differs between people and segment lengths. In Patient A (BMI "
    f"{EXA['bmi']:.1f} kg/m²), the ECG explained {EXA['mean_r2']['5']*100:.0f}% of heartbeat-locked "
    f"EEG variance at 5 min and {EXA['mean_r2']['60']*100:.0f}% at 60 min; in Patient B (BMI "
    f"{EXB['bmi']:.1f} kg/m²), {EXB['mean_r2']['5']*100:.0f}% and {EXB['mean_r2']['60']*100:.0f}%, "
    f"so the longer segment revealed twice as much CFA, spread over most of the scalp. The analyses "
    f"below test whether higher CFA with higher BMI and longer segments holds across the cohort.",
    styles["Body"]))

story.append(Paragraph("3.3 CFA across the cohort", styles["H2"]))
story.append(KeepTogether([
    fig("fig2_cfa_overview.png", crop=False),
    Paragraph(
    f"<b>Figure 2.</b> CFA across {CFA['n_patients']:,} patients ({CFA['n_rows']:,} "
    f"channel-recordings, 10-min segments). (a) Per-channel R² between R-peak-locked EEG and ECG "
    f"averages, over the full epoch (mean {CFA['r2_full_mean']:.2f}) and outside the ±50 ms QRS "
    f"window (mean {CFA['r2_excl_qrs_mean']:.2f}). (b) ICA SNR (ECG-component to residual "
    f"variance; Section 2.4; {ICA['n_patients']:,} patients) vs. model-free R² of the same "
    f"channel-recording (log axis; line, least-squares fit; r = {ICA['r2_vs_snr_r']:.2f}, "
    f"p {R2_SNR_P_STR}, n = {ICA['r2_vs_snr_n']:,}). (c) Mean CFA R² (outside QRS) at the "
    f"{TOPO['n_sites']} 10-20 sites recorded in at least {TOPO['min_patients']} patients "
    f"({TOPO['n_rows']:,} channel-recordings; referenced labels mapped to their scalp site; "
    f"{', '.join(TOPO['dropped_sites'])} fell below this floor). (d) Per-site R² distribution, "
    f"sorted by median (orange line; box, interquartile range; n per site shown); the six standard "
    f"sites (F3/F4/C3/C4/O1/O2) have n &gt; 12,000 each, and sites near the floor are less "
    f"precise. (e) Patient-mean CFA R² per diagnosis category (mean, 95% CI, Welch; n in brackets; "
    f"categories overlap; those with fewer than 10 patients omitted); dotted line, EHR-linked "
    f"patients with no recorded diagnosis (mean {DX['no_dx_mean']:.2f}, n = {DX['no_dx_n']:,}; "
    f"confounded by site, see Limitations); Kruskal-Wallis p = {DX['p_kruskal']:.2g}. (f) Pairwise "
    f"Mann-Whitney p-values, Benjamini-Hochberg corrected across all {DX['pairwise']['n_pairs']} "
    f"tests ({DX['pairwise']['n_significant_fdr']} significant at q &lt; 0.05; colour saturates at "
    f"q = 0.5); order as in (e).",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"Outside the QRS-exclusion window, the ECG explained a mean R² of "
    f"{CFA['r2_excl_qrs_mean']:.2f} (median {CFA['r2_excl_qrs_median']:.2f}; Figure 2a), only "
    f"modestly below the full-epoch mean ({CFA['r2_full_mean']:.2f}), so QRS exclusion removed only "
    f"about {(1 - CFA['r2_excl_qrs_mean']/CFA['r2_full_mean'])*100:.0f}% of the ECG-explained "
    f"variance. Values ranged from no detectable CFA to EEG averages almost identical to the ECG. In "
    f"the {ZL['non_locked']['n']}-patient subsample with a chance-level reference (Section 2.6), "
    f"full-epoch R² at F3, F4, C3, and C4 was {ZL['pre_ica']['mean']:.2f} for R-peak-locked "
    f"averages but {ZL['non_locked']['mean']:.2f} (median {ZL['non_locked']['median']:.2f}) around "
    f"random pseudo-events, so the observed R² far exceeded chance.",
    styles["Body"]))
story.append(Paragraph(
    f"The model-based estimator agreed: the ECG-related ICA component accounted for a median of "
    f"{ICA['component_variance_fraction_median_unfiltered']*100:.0f}% of heartbeat-locked EEG "
    f"variance, and excluding it reduced this variance by a median of "
    f"{ICA['hep_pct_drop_median']*100:.0f}%. Channels with higher model-free R² had higher ICA SNR "
    f"(Figure 2b); the dispersion is expected, because one globally selected source does not load "
    f"equally on all electrodes. Regression on the concurrent ECG has been proposed for estimating "
    f"and removing CFA, in linear<super>22</super> and nonlinear neural-network<super>23</super> "
    f"form; we therefore used per-channel ECG regression as the primary estimate and ICA as "
    f"convergent evidence.",
    styles["Body"]))
story.append(Paragraph(
    f"CFA varied across the scalp (Figure 2c–d), from {TOPO['highest_mean']:.2f} at "
    f"{TOPO['highest_site']} to {TOPO['lowest_mean']:.2f} at {TOPO['lowest_site']}, a "
    f"{TOPO['highest_mean']/max(TOPO['lowest_mean'], 1e-6):.1f}-fold range. Sites outside the "
    f"standard montage came from a minority of recordings with other montages and references, so "
    f"part of this range reflects referencing rather than position; within the six standard "
    f"mastoid-referenced sites, mean R² ranged from {TOPO['site_means']['F3']:.2f} (F3) to "
    f"{TOPO['site_means']['F4']:.2f} (F4). A site-independent correction would therefore perform "
    f"unevenly.",
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
    f"<b>Figure 3.</b> CFA vs. patient characteristics (a–c) and segment duration (d–f). "
    f"Patient-mean CFA R² (outside QRS) vs. (a) age (n = {STRAT['n_with_age']:,}; "
    f"r = {STRAT['r_age_pearson']:.2f}, p = {STRAT['p_age_pearson']:.2g}; line, least-squares fit), "
    f"(b) sex (Mann-Whitney p = {STRAT['p_sex_mannwhitney']:.1e}; box, quartiles; outliers "
    f"omitted), and (c) BMI with sex-specific fits (n = {BMI_STRAT['n']:,}; "
    f"r = {BMI_STRAT['r_pearson']:.2f}, p = {BMI_STRAT['p_pearson']:.2g}; one observation per "
    f"patient, longest window; {BMI_WINDOW_COUNTS}). Female: n = "
    f"{BMI_STRAT['by_sex']['Female']['n']:,}, r = {BMI_STRAT['by_sex']['Female']['r_pearson']:.2f}, "
    f"p = {BMI_STRAT['by_sex']['Female']['p_pearson']:.1e}; male: n = "
    f"{BMI_STRAT['by_sex']['Male']['n']:,}, r = {BMI_STRAT['by_sex']['Male']['r_pearson']:.2f}, "
    f"p = {BMI_STRAT['by_sex']['Male']['p_pearson']:.1e}; interaction "
    f"p = {BMI_STRAT['sex_interaction_p']:.2g}. (d–f) Patients with data at every duration "
    f"(n = {DOSE['n_common_patients']:,}). (d) Mean channel-level R² (all EEG channels; ± 95% CI) "
    f"per duration ({dose_str}); Friedman p {WT_P_STR}; every consecutive step significant "
    f"(Holm-corrected paired Wilcoxon, all p &le; {max_adjacent_p:.1g}). (e) By sex; men higher at "
    f"every duration (Mann-Whitney, all p &le; {p_sex_dose:.3g}). (f) By diagnosis; not "
    f"significant (p = {p_dx_dose_min:.2g}–{p_dx_dose_max:.2g}).",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"BMI correlated positively with CFA R² (Figure 3c; n = {BMI_STRAT['n']:,}, "
    f"r = {BMI_STRAT['r_pearson']:.2f}, p = {BMI_STRAT['p_pearson']:.1e}; mean BMI "
    f"{BMI_STRAT['bmi_mean']:.1f} kg/m²): each 10 kg/m² added {BMI_STRAT['slope']*10:.3f} to "
    f"patient-mean R². The correlation held in women (r = "
    f"{BMI_STRAT['by_sex']['Female']['r_pearson']:.2f}, "
    f"p = {BMI_STRAT['by_sex']['Female']['p_pearson']:.1e}) and men "
    f"(r = {BMI_STRAT['by_sex']['Male']['r_pearson']:.2f}, "
    f"p = {BMI_STRAT['by_sex']['Male']['p_pearson']:.1e}), with no BMI-by-sex interaction "
    f"(p = {BMI_STRAT['sex_interaction_p']:.2g}). Though weak individually "
    f"(r² ≈ {BMI_STRAT['r_pearson']**2:.2f}), it agrees with the diagnosis analysis, in which obese "
    f"patients had the highest CFA of all categories (Section 3.6).",
    styles["Body"]))
story.append(Paragraph(
    f"CFA R² was higher in men than in women ({STRAT['sex_means']['Male']:.2f} vs. "
    f"{STRAT['sex_means']['Female']:.2f}, p = {STRAT['p_sex_mannwhitney']:.1e}; Figure 3b). BMI did "
    f"not explain this: women had a higher mean BMI "
    f"({BMI_STRAT['by_sex']['Female']['bmi_mean']:.1f} vs. "
    f"{BMI_STRAT['by_sex']['Male']['bmi_mean']:.1f} kg/m²), and the sex coefficient was unchanged "
    f"after BMI adjustment ({CONF['sex_unadj_coef']:.3f} vs. {CONF['sex_adj_coef']:.3f}, "
    f"p = {CONF['sex_adj_p']:.2g}; Section S5), with BMI an independent predictor "
    f"(p = {CONF['bmi_p']:.2g}). The sex difference held at every duration (Section 3.5). Age showed "
    f"no linear relationship with CFA (Figure 3a; r = {STRAT['r_age_pearson']:.2f}, "
    f"p = {STRAT['p_age_pearson']:.2g}); only the oldest tertile was slightly lower "
    f"({STRAT['age_tertile_means']['Older']:.2f} vs. "
    f"{STRAT['age_tertile_means']['Younger']:.2f}/{STRAT['age_tertile_means']['Middle']:.2f}, "
    f"p = {STRAT['p_age_anova']:.1e}).",
    styles["Body"]))

story.append(Paragraph("3.5 Segment duration changes the estimated CFA", styles["H2"]))
story.append(Paragraph(
    f"Segment duration had a significant, systematic effect (Friedman "
    f"χ² = {WT['friedman_chi2']:,.0f}, p {WT_P_STR}, Kendall's W = {WT['kendall_w']:.2f}; "
    f"n = {WT['n']:,}). Mean CFA R² rose from {dose_by_min[5]['mean']:.2f} at 5 min to "
    f"{dose_by_min[10]['mean']:.2f} (10 min), {dose_by_min[20]['mean']:.2f} (20), "
    f"{dose_by_min[30]['mean']:.2f} (30), {dose_by_min[45]['mean']:.2f} (45), and "
    f"{dose_by_min[60]['mean']:.2f} at 60 min (Figure 3d), with every consecutive step significant "
    f"after Holm correction. From 5 to 60 min, R² increased in "
    f"{WT_PAIR[(5, 60)]['pct_patients_increase']:.0f}% of patients (mean "
    f"+{WT_PAIR[(5, 60)]['mean_diff']:.2f}); from the 10-min main-analysis windows to 60 min, in "
    f"{WT_PAIR[(10, 60)]['pct_patients_increase']:.0f}% (mean +{WT_PAIR[(10, 60)]['mean_diff']:.2f}). "
    f"Gains shrank with duration, from {dose_by_min[20]['mean']-dose_by_min[10]['mean']:.2f} between "
    f"10 and 20 min to {dose_by_min[60]['mean']-dose_by_min[45]['mean']:.2f} between 45 and 60 min, "
    f"without reaching a plateau. Fewer channel-recordings contributed at 45 and 60 min "
    f"({dose_by_min[45]['n_rows']:,} vs. {dose_by_min[30]['n_rows']:,} at 30 min), which may change "
    f"channel composition, but between 5 and 30 min composition was nearly constant "
    f"({dose_by_min[5]['n_rows']:,} to {dose_by_min[30]['n_rows']:,} channel-recordings) and R² "
    f"still rose monotonically.",
    styles["Body"]))
story.append(Paragraph(
    f"Men had higher CFA than women at every duration (Figure 3e; e.g. "
    f"{DOSE['stratified_by_length'][0]['male_mean']:.2f} vs. "
    f"{DOSE['stratified_by_length'][0]['female_mean']:.2f} at 5 min, "
    f"{DOSE['stratified_by_length'][-1]['male_mean']:.2f} vs. "
    f"{DOSE['stratified_by_length'][-1]['female_mean']:.2f} at 60 min). Diagnosis differences were "
    f"not significant (Figure 3f); the no-diagnosis mean was numerically higher at every duration, "
    f"consistent with the within-site reversal in the Limitations.",
    styles["Body"]))

story.append(Paragraph("3.6 CFA by diagnosis category", styles["H2"]))
story.append(Paragraph(
    f"Among diagnosis categories (Figure 2e–f), CFA was highest in {DX['highest_category']} "
    f"(+{DX['highest_diff']:.2f} R² vs. the no-diagnosis reference, "
    f"n = {DX['categories'][DX['highest_category']]['n']:,}) and lowest in "
    f"{DX['lowest_category']} ({DX['lowest_diff']:+.2f}, "
    f"n = {DX['categories'][DX['lowest_category']]['n']:,}). Of {DX['pairwise']['n_pairs']} "
    f"pairwise comparisons, {DX['pairwise']['n_significant_fdr']} survived FDR correction; "
    f"{sum('Obesity' in (q['a'], q['b']) for q in DX['pairwise']['significant_pairs'])} involved "
    f"Obesity, which differed from every other category, and the rest involved Obstructive Sleep "
    f"Apnea, in line with the BMI association (Section 3.4). All {DX['n_categories_total']} "
    f"categories lay above the no-diagnosis reference, but this reference is confounded by site "
    f"(see Limitations) and comparisons were not adjusted for age, sex, or BMI, so these are not "
    f"disease effects.",
    styles["Body"]))

story.append(Paragraph("3.7 CFAR removes CFA without ECG", styles["H2"]))
story.append(KeepTogether([
    fig("fig5_cleaning_demo.png", crop=False),
    Paragraph(
    f"<b>Figure 4.</b> CFAR: ECG-free cardiac field artifact removal (Section 2.8). "
    f"(a) Five seconds of continuous EEG at C4 from an example patient (male, 72 years, BMI "
    f"34.2 kg/m²) before (orange) and after (black) CFAR; dashed lines, R-peaks (from the ECG, "
    f"for display only); chosen as a clear high-CFA case. (b) The CFAR filter: the single CFA scalp "
    f"pattern learned from the cohort and subtracted from every EEG sample. (c) Patient-mean CFA R² "
    f"in {CD['n_test']:,} patients (cross-fitted, so each is held out from its own filter) before "
    f"and after cleaning, and with the gold standard. (d) Mean CFA R² (± SEM) across BMI bins in men "
    f"and women, before (solid) and after (dashed) cleaning (known BMI, "
    f"n = {CD['n_test_with_bmi']:,}; bins with at least 10 patients).",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"The CFA scalp pattern was highly consistent across patients (median absolute cosine "
    f"similarity to the population pattern, {CD['pattern_consistency_median_cos']:.2f}; Figure 4b). "
    f"CFAR, which projects this pattern out of held-out recordings without ECG, reduced mean CFA R² from "
    f"{CD['r2_pre_mean']:.2f} to {CD['r2_clean_mean']:.2f} ({CD['pct_reduction_mean']:.0f}% "
    f"reduction; Wilcoxon p {CD_P_STR}), and R² fell in {CD['pct_patients_reduced']:.0f}% of "
    f"patients (Figure 4c). The gold standard reached {CD['r2_own_mean']:.2f}, so CFAR "
    f"closed {CD_GAP:.0f}% of the gap. In the example patient, mean R² fell from "
    f"{CD['example_r2_pre']:.2f} to {CD['example_r2_clean']:.2f}, and the deflection after every "
    f"R-peak was removed (Figure 4a). Obese patients (BMI ≥ 30 kg/m², n = {CD['obese_n']:,}) had "
    f"more CFA removed than lean patients (BMI &lt; 25, n = {CD['lean_n']:,}; R² "
    f"−{CD['obese_removed']:.2f} vs. −{CD['lean_removed']:.2f}, Mann-Whitney "
    f"p = {CD['p_removed_obese_vs_lean']:.2g}), because they had more to remove: the share removed "
    f"was similar ({CD['obese_pct_reduction']:.0f}% vs. {CD['lean_pct_reduction']:.0f}%). Residual CFA "
    f"remained higher at high BMI and in men (Figure 4d). In the ongoing EEG, the cardiac share of "
    f"variance fell from a median of {CD['frac_pre_median']*100:.2f}% to "
    f"{CD['frac_clean_median']*100:.2f}% (mean {CD['frac_pre_mean']*100:.1f}% to "
    f"{CD['frac_clean_mean']*100:.1f}%; median reduction {CD['frac_drop_median_pct']:.0f}%) and "
    f"decreased in {CD['pct_patients_frac_reduced']:.0f}% of patients. Predicting each patient's "
    f"pattern from BMI, age, sex, and diagnoses gave no gain (median cosine similarity to the "
    f"patient's own pattern {CD['pattern_cos_cov_median']:.3f} vs. "
    f"{CD['pattern_cos_pop_median']:.3f}; mean R² {CD['r2_cov_mean']:.3f} vs. "
    f"{CD['r2_clean_mean']:.3f}; p = {CD['p_cov_vs_pop']:.2g}; better in only "
    f"{CD['pct_patients_cov_better']:.0f}% of patients): these factors change the amount of CFA, "
    f"not its scalp pattern, and projection removes the pattern regardless of amount. Cleaning "
    f"retained a median {CD['hep_retained_median']*100:.0f}% of a synthetic neural HEP and changed "
    f"ongoing EEG power by a median of {CD['psd_change_db_median']:.1f} dB (1–40 Hz), the cost of "
    f"removing one of six spatial dimensions.",
    styles["Body"]))

# ---------------------------------------------------------------------
# Discussion
# ---------------------------------------------------------------------
story.append(Paragraph("4. Discussion", styles["H1"]))
story.append(Paragraph(
    f"In {CFA['n_patients']:,} patients, by far the largest CFA sample, two estimators showed that "
    f"much of the heartbeat-locked scalp EEG variance is shared with the ECG even after QRS "
    f"exclusion. CFA was not a generic nuisance but a property of each person and analysis: it "
    f"ranged from negligible to near-complete between patients (Figure 1), varied "
    f"{TOPO['highest_mean']/max(TOPO['lowest_mean'], 1e-6):.1f}-fold across sites (Figure 2c–d), "
    f"increased with BMI, was higher in men (Figure 3a–c), and depended strongly on segment "
    f"duration (Figure 3d–f). Because these are common grouping variables or analysis choices, "
    f"uncorrected CFA can create, mask, or inflate group differences in heartbeat-locked EEG.",
    styles["Body"]))
story.append(Paragraph(
    "The conventional ±30–50 ms QRS exclusion<super>6-8</super> did not remove CFA: excluding "
    f"±50 ms reduced mean R² only from {CFA['r2_full_mean']:.2f} to {CFA['r2_excl_qrs_mean']:.2f}, "
    "so ECG-correlated signal extends into the analysis interval. The agreement between model-free "
    "R² and ICA SNR shows that both detect the same contamination. Neither proves that every "
    "residual heartbeat-locked deflection is artifactual, and cortical heartbeat-evoked activity is "
    "well documented<super>5,6,10</super>; still, uncorrected heartbeat-locked EEG variance cannot "
    "be assumed cortical.",
    styles["Body"]))
story.append(Paragraph(
    f"BMI correlated positively with CFA in women and men, and obese patients had the highest CFA "
    f"of all diagnosis categories; in Figure 1, CFA was low at a BMI of {EXA['bmi']:.0f} kg/m² and "
    f"several times higher at {EXB['bmi']:.0f} kg/m². This seems counterintuitive: adipose tissue "
    f"conducts roughly ten times less than muscle<super>24</super>, and subcutaneous fat attenuates "
    f"the surface ECG, so obesity is associated with low QRS voltage<super>12,13</super>. If fat only "
    f"insulated, less cardiac field should reach the scalp. The resolution lies in what R² "
    f"measures. R² is scale-invariant: it indexes how closely the shape of the heartbeat-locked "
    f"scalp average follows the ECG, not amplitude. Uniform attenuation scales both without "
    f"changing their correlation; R² rises only when the cardiac component grows relative to "
    f"everything else in the average, namely background EEG and neural responses. Several changes "
    f"in obesity could raise this share. First, obesity is associated with greater left-ventricular "
    f"mass and a leftward, more horizontal cardiac axis<super>13</super>, changing the strength and "
    f"orientation of the cardiac dipole; in mastoid-referenced derivations, CFA is the cardiac "
    f"potential difference between scalp and mastoid, which depends strongly on dipole orientation. "
    f"Second, low-conductivity tissue redistributes volume currents as well as attenuating them, "
    f"which may change how much of the field projects onto the head. Third, tissue between brain "
    f"and electrode attenuates cortical EEG at the same electrode. More heartbeats per window is "
    f"unlikely to explain the effect: heart rate was slightly lower at higher BMI "
    f"(r = {BMIHR['r_bmi_bpm']:.2f}), and adjusting for heart rate did not reduce the BMI "
    f"coefficient ({BMIHR['bmi_coef_unadj']:.4f} vs. {BMIHR['bmi_coef_adj']:.4f} per kg/m², "
    f"n = {BMIHR['n']:,}). Which mechanism dominates cannot be resolved here; this needs absolute "
    f"CFA amplitude, which should fall with BMI if insulation dominated, and direct measures of "
    f"body composition and heart position. Individually the correlation was weak "
    f"(r = {BMI_STRAT['r_pearson']:.2f}), so BMI cannot predict a recording's CFA, but a difference "
    f"in mean BMI between study groups, common in clinical comparisons, will produce a systematic "
    f"difference in CFA.",
    styles["Body"]))
story.append(Paragraph(
    f"CFA was higher in men at every segment duration, unchanged by BMI adjustment; because women "
    f"here had a higher mean BMI, BMI would if anything have reduced the difference. It may reflect "
    f"differences in heart size and position, thoracic and head geometry, or fat distribution, none "
    f"measured here; cardiac parameters such as stroke volume may also influence heartbeat-locked "
    f"responses<super>11</super>. Sex differences are among the most reported group effects in EEG, "
    f"so studies comparing men and women on heartbeat-locked measures should quantify CFA per "
    f"channel and show that the difference survives its correction.",
    styles["Body"]))
story.append(Paragraph(
    f"Segment duration had the largest, most consistent effect: in {WT['n']:,} duration-matched "
    f"patients, mean R² rose from {DOSE['lengths'][0]['mean']:.2f} at 5 min and "
    f"{DOSE['lengths'][1]['mean']:.2f} at 10 min to {DOSE['lengths'][-1]['mean']:.2f} at 60 min, "
    f"every step was significant, and R² increased from 5 to 60 min in "
    f"{WT_PAIR[(5, 60)]['pct_patients_increase']:.0f}% of patients. Longer segments contain more "
    f"heartbeats and give a more stable average<super>14</super>, letting shared EEG-ECG structure "
    f"emerge from background EEG. Two consequences follow. First, short segments, including our "
    f"10-min main analysis, underestimate CFA, so a low estimate from a short recording does not "
    f"show that the data are clean. Second, groups that differ in recording length or usable "
    f"heartbeats will differ in apparent CFA for methodological reasons alone. Segment length "
    f"should therefore be reported, justified, and matched between groups where possible, and "
    f"results checked across durations. Because R² depends on the number of averaged beats, group "
    f"differences in heart rate could also affect R²; this was not examined.",
    styles["Body"]))
story.append(Paragraph(
    "BMI is an anthropometric index, not a direct measure of adiposity; fat mass, thoracic "
    "geometry, and electrode impedance were unavailable, and diagnosis categories overlap. We "
    "therefore treat BMI, obesity, and sex as confounders of CFA, not as evidence of a causal effect "
    "of body fat. Associations of age and diagnosis with CFA may likewise reflect anatomy, cardiac "
    "physiology, medication, comorbidity, or recording conditions.",
    styles["Body"]))
story.append(Paragraph(
    "In practice, EEG preprocessing should combine concurrent ECG, channel-level assessment of the "
    "R-peak-locked EEG-ECG relationship, and removal or regression of ECG-related components, then "
    "verify that heartbeat-locked variance falls after cleaning while staying above a "
    "non-heartbeat-locked noise floor. Reporting only the QRS mask or the selected ICA component is "
    "insufficient. Studies should report segment duration and number of heartbeats, match them "
    "between groups where possible, report channel-level CFA before and after cleaning, and test "
    "robustness to adjustment for BMI (or, better, direct body-composition measures) and sex. These "
    "recommendations extend recent HEP reviews<super>8,9</super> to EEG analysis generally.",
    styles["Body"]))
story.append(Paragraph(
    f"For recordings without ECG, CFAR offers a practical route. Because the CFA scalp "
    f"pattern is consistent across people, CFAR, learned once from patients with "
    f"ECG, cleaned the heart signal from the continuous EEG of unseen patients without a cardiac "
    f"reference: it removed about {CD['pct_reduction_mean']:.0f}% of heartbeat-locked CFA, cut the "
    f"cardiac share of ongoing EEG variance by about "
    f"{(1 - CD['frac_clean_mean'] / CD['frac_pre_mean']) * 100:.0f}%, and closed {CD_GAP:.0f}% of "
    f"the gap to cleaning with the patient's own ECG. Because it acts on every sample, it cleans "
    f"ongoing EEG, not only heartbeat-locked averages, and can be applied retrospectively to "
    f"existing datasets. Conditioning on BMI, sex, age, and diagnoses gave no benefit, because these "
    f"change the amount of CFA, not its pattern; they remain useful for identifying who needs "
    f"cleaning most, since CFA before and after cleaning was highest in men and at high BMI "
    f"(Figure 4d). CFAR is specific to the montage and reference it was trained on, removes "
    f"one of six spatial dimensions, retained {CD['hep_retained_median']*100:.0f}% of a simulated "
    f"neural HEP, and leaves some CFA; it should be validated on other montages before routine use.",
    styles["Body"]))

story.append(Paragraph("5. Conclusions", styles["H1"]))
story.append(Paragraph(
    f"In {CFA['n_patients']:,} patients, CFA was a substantial, systematic component of scalp "
    f"EEG<super>1-4</super>: the ECG explained a mean R² of {CFA['r2_excl_qrs_mean']:.2f} of "
    f"heartbeat-locked EEG variance outside the QRS window, and ICA removal of the ECG-related "
    f"component reduced this variance by a median of {ICA['hep_pct_drop_median']*100:.0f}%. CFA "
    f"increased with BMI, was higher in men independently of BMI, varied across electrodes, and "
    f"increased with segment duration up to 60 min. It is therefore a participant-, channel-, and "
    f"duration-dependent noise source. Robust inference requires ECG-informed channel-level "
    f"cleaning with before-and-after validation<super>8,9</super>, reported and justified segment "
    f"lengths<super>14</super>, and adjustment for BMI and sex; without ECG, CFAR "
    f"removes about half of the CFA. Otherwise, group differences in EEG may partly reflect the heart "
    f"rather than the brain<super>6,11</super>.",
    styles["Body"]))

story.append(Paragraph("Limitations", styles["H2"]))
story.append(Paragraph(
    f"First, diagnosis records covered only part of the Human Sleep Project, so the no-diagnosis "
    f"group came largely from a site without diagnosis tables (mean R² "
    f"{DXS['I0003_no_dx']['mean']:.2f}, n = {DXS['I0003_no_dx']['n']}). Within the site with "
    f"records, patients without a diagnosis had higher R² than those with one "
    f"({DXS['I0002_no_dx']['mean']:.2f}, n = {DXS['I0002_no_dx']['n']}, vs. "
    f"{DXS['I0002_any_dx']['mean']:.2f}, n = {DXS['I0002_any_dx']['n']:,}). The "
    f"with/without-diagnosis comparison is therefore confounded by site and is not a disease effect; "
    f"comparisons among categories are unaffected. Second, detected heart rate was high for sleep "
    f"(median {CFA['bpm_quartiles']['0.5']:.0f} beats/min, interquartile range "
    f"{CFA['bpm_quartiles']['0.25']:.0f}–{CFA['bpm_quartiles']['0.75']:.0f}); the custom R-peak "
    f"detector was not validated against an established algorithm, and occasional T-wave detections "
    f"would mix T-wave-locked epochs into the averages. Third, R² cannot separate volume-conducted "
    f"CFA from neural activity with an ECG-like time course, and epochs ended 400 ms after the "
    f"R-peak, so later HEP intervals overlapping the T wave were not covered. Fourth, CFA was "
    f"estimated in native references, mostly contralateral mastoid (e.g., F3-M2, F4-M1); because "
    f"the cardiac field projects differently onto each reference, site values and hemispheric "
    f"asymmetries may not generalise to average-referenced data.",
    styles["Body"]))
story.append(Paragraph(
    f"Fifth, 10-min estimates are likely lower bounds, because CFA R² rose monotonically with "
    f"duration, including from 5 to 30 min, where channel composition was nearly constant (Section "
    f"3.5). Sixth, chance-level (pseudo-event) R² was estimated only in "
    f"{ZL['non_locked']['n']} patients and only over the full epoch (Section 3.3). Seventh, windows "
    f"were not selected by sleep stage, so vigilance was not controlled and may differ between "
    f"groups and durations; CFA did not differ by stage in a small staged subsample "
    f"(n = {SENS['n_patients']}; Section 3.1). Eighth, the cohort was clinically referred, with a "
    f"high diagnostic burden (Figure S1c), which limits generalisation of absolute R² to healthy "
    f"volunteers, and underweight patients were under-represented ({BMI_STRAT['underweight_n']:,} "
    f"of {BMI_STRAT['n']:,} with BMI &lt;18.5 kg/m²). Finally, as a secondary analysis, recording "
    f"parameters and clinical annotations could not be controlled.",
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
    f"<b>Figure S1.</b> Cohort composition ({COHORT['n_demographics']:,} Human Sleep Project "
    f"patients). (a) Age (median {COHORT['age_median']:.0f} years, range "
    f"{COHORT['age_min']:.0f}–{COHORT['age_max']:.0f}). (b) Sex ({sex_str}). (c) The ten most "
    f"prevalent diagnosis categories, led by {top3_str}; categories overlap. The cohort is "
    f"clinically referred, not a healthy community sample.",
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
    f"<b>Figure S2.</b> HEP variance before and after excluding the ECG-related ICA component, vs. "
    f"a non-heartbeat-locked control (n = {S['post_ica_variance']['n_patients']:,} ICA patients with "
    f"at least one well-covered site; {S['post_ica_variance']['n_patients_control']:,}-patient "
    f"control subsample, Section 2.6; medians, log axis). (a) Four core electrodes "
    f"({'/'.join(S['post_ica_variance']['core_electrodes'])}): median "
    f"{S['post_ica_variance']['core_pre_median']:.2f} µV² pre-ICA, "
    f"{S['post_ica_variance']['core_post_median']:.2f} µV² post-ICA "
    f"({S['post_ica_variance']['core_pct_drop']:.0f}% drop), and "
    f"{S['post_ica_variance']['core_non_locked_median']:.2f} µV² for the control (same windows "
    f"re-epoched around random pseudo-events). (b) Per electrode, all six sites. (c) Spectral "
    f"entropy (normalised Shannon entropy of the evoked Welch spectrum) of the same conditions, core "
    f"electrodes (n = {S['post_ica_variance']['n_patients_entropy']:,}, separate subsample). (d) "
    f"Spectral entropy per electrode.",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"Removing the ECG-related component reduced median core-electrode HEP variance by "
    f"{S['post_ica_variance']['core_pct_drop']:.0f}%; this ratio of group medians is larger than, "
    f"and not comparable to, the median per-channel reduction in Section 3.3 "
    f"({ICA['hep_pct_drop_median']*100:.0f}%). The control's noise floor "
    f"({S['post_ica_variance']['core_non_locked_median']:.2f} µV²) was well below both pre-ICA "
    f"({S['post_ica_variance']['core_pre_median']:.2f} µV²) and post-ICA "
    f"({S['post_ica_variance']['core_post_median']:.2f} µV²) values, and post-ICA variance stayed "
    f"above it at every electrode (panel b), so component removal did not reduce the "
    f"heartbeat-locked signal to noise. Spectral entropy, a variance-independent measure, fell from "
    f"pre-ICA ({S['post_ica_variance']['core_entropy_pre_median']:.2f}) to post-ICA "
    f"({S['post_ica_variance']['core_entropy_post_median']:.2f}) to the control "
    f"({S['post_ica_variance']['core_entropy_non_locked_median']:.2f}); the post-ICA decrease was "
    f"small, and post-ICA values exceeded the control at every electrode (panel d).",
    styles["Body"]))

story.append(Paragraph("S3. EEG-ECG cross-correlation and mutual information", styles["H2"]))
CC = S["crosscorr_mi"]
story.append(KeepTogether([
    fig("figS4_crosscorr_mi.png", 6.6 * inch),
    Paragraph(
    f"<b>Figure S3.</b> EEG-ECG cross-correlation and mutual information pre-ICA, post-ICA, and for "
    f"a non-heartbeat-locked control (n = {CC['n_patients']:,} patients; separate subsample with "
    f"ICA refitted; Section 2.6). The control re-epochs the pre-ICA EEG around random pseudo-events "
    f"and correlates it with the R-peak-locked ECG average, giving the chance-level relationship. "
    f"(a) Cross-correlation vs. lag between the core-electrode "
    f"({'/'.join(CC['core_electrodes'])}) HEP average and the ECG average, sign-aligned per channel; "
    f"mean ± SEM across patients. (b) Mean peak |cross-correlation| per electrode. (c) Mutual "
    f"information, core electrodes. (d) Mutual information per electrode.",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"Cross-correlation peaked near zero lag in both heartbeat-locked conditions (median "
    f"{CC['core_peak_lag_ms_pre_median']:.0f} ms pre-ICA, {CC['core_peak_lag_ms_post_median']:.0f} "
    f"ms post-ICA), consistent with the instantaneous volume conduction assumed by the zero-lag "
    f"estimator (Section 2.3). Peak |r| fell from {CC['core_peak_r_pre_mean']:.2f} pre-ICA to "
    f"{CC['core_peak_r_post_mean']:.2f} post-ICA and {CC['core_peak_r_non_locked_mean']:.2f} for "
    f"the control, and mutual information from {CC['core_mi_pre_median']:.2f} to "
    f"{CC['core_mi_post_median']:.2f} to {CC['core_mi_non_locked_median']:.2f} nats (panel c). "
    f"Post-ICA values stayed above the control at every well-covered electrode (panels b, d), so "
    f"an EEG-ECG relationship persists after cleaning, in agreement with Section 3.3 and Figure S2. "
    f"Mutual information, which also captures nonlinear dependence, followed the same order as "
    f"correlation, giving no sign of dependence beyond the linear measures.",
    styles["Body"]))



PSD = S["psd_comparison"]
story.append(Paragraph("S4. Power spectral density: pre-ICA, post-ICA, non-locked control, and ECG", styles["H2"]))
story.append(Paragraph(
    f"Welch power spectral density was computed for the pre-ICA, post-ICA, and non-locked evoked "
    f"waveforms of Section S3 and for the ECG evoked average, in the same "
    f"{PSD['n_patients']:,}-patient subsample.",
    styles["Body"]))
story.append(KeepTogether([
    fig("figS5_psd_comparison.png", 6.6 * inch, crop=False),
    Paragraph(
    f"<b>Figure S4.</b> Power spectral density (Welch, dB) of the pre-ICA, post-ICA, "
    f"non-heartbeat-locked control, and ECG evoked waveforms (n = {PSD['n_patients']:,}). (a) Core "
    f"electrodes ({'/'.join(PSD['core_electrodes'])}), mean \u00b1 SEM. (b) Each of the six "
    f"well-covered sites.",
        styles["Caption"]),
]))
story.append(Paragraph(
    "The ECG spectrum was broadband, dominated by the sharp QRS transient, which the non-locked "
    "control lacked. Pre-ICA and post-ICA EEG spectra lay between these references and closer to "
    "the ECG than the control, supporting Sections 3.3 and S3 in the spectral domain. Spectral shape "
    "was similar across electrodes (panel b), so CFA varies across the scalp mainly in magnitude "
    "(Figure 2c), not spectral content. Spectra end at 100 Hz, the upper edge of the analysis filter "
    "(Section 2.2).",
    styles["Body"]))

CONF = S["sex_bmi_confound"]
story.append(Paragraph("S5. Adjustment of the sex effect for BMI", styles["H2"]))
story.append(Paragraph(
    f"Men had higher CFA R² than women (Section 3.4), but women had a higher mean BMI "
    f"({BMI_STRAT['by_sex']['Female']['bmi_mean']:.1f} vs. "
    f"{BMI_STRAT['by_sex']['Male']['bmi_mean']:.1f} kg/m²), and BMI itself was associated with "
    f"CFA. In patients with BMI (n = {CONF['n']:,}), OLS regression of patient-mean CFA R² on sex "
    f"gave a male-versus-female coefficient of {CONF['sex_unadj_coef']:.3f} (95% CI "
    f"{CONF['sex_unadj_ci_lo']:.3f} to {CONF['sex_unadj_ci_hi']:.3f}, p = "
    f"{CONF['sex_unadj_p']:.2g}). Adding BMI did not attenuate it ({CONF['sex_adj_coef']:.3f}; 95% "
    f"CI {CONF['sex_adj_ci_lo']:.3f} to {CONF['sex_adj_ci_hi']:.3f}, p = {CONF['sex_adj_p']:.2g}), "
    f"and BMI was an independent predictor ({CONF['bmi_coef']:.4f} per kg/m², "
    f"p = {CONF['bmi_p']:.2g}).",
    styles["Body"]))
story.append(KeepTogether([
    fig("figS6_sex_bmi_confound.png", 4.5 * inch),
    Paragraph(
    f"<b>Figure S5.</b> Male-versus-female CFA R² (outside QRS) OLS coefficient, without and with "
    f"BMI adjustment (n = {CONF['n']:,}); points, coefficient; bars, 95% CI.",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"The sex coefficient rose slightly after BMI adjustment, as expected given women's higher "
    f"BMI, so BMI does not explain the sex effect; sex and BMI were independently associated with "
    f"CFA.",
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
    title="Cardiac Field Artifact Removal (CFAR): A Large-Cohort Study Cleaning the Heart's Noise from Brain Signals",
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
