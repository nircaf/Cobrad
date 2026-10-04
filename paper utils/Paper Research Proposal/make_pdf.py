#!/usr/bin/env python3
"""Assemble the PhD research-proposal PDF (TAU life-sciences DLP format) from
inline text + Paper1/figures/*.png, using reportlab Platypus.

Body: ~10 pages, 1.5 line spacing, 12 pt. Figures embedded at the end.
Sources: Cafri_HEP_conference_abstract.pdf (HEP sleep-stage & age gradient) and
16_diagnosis_sleep_stage_comparison_dashboard.py (diagnosis x sleep-stage HEP).

  source venv/bin/activate && python3 "paper utils/Paper Research Proposal/make_pdf.py"
"""
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

# Hebrew glyphs: DejaVuSans covers them; reportlab has no bidi, so pure-Hebrew
# lines are reversed here to render right-to-left visually.
pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))

def he(s):
    """Reverse a pure-Hebrew line for RTL visual order."""
    return s[::-1]

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIG_DIR = os.path.join(ROOT, "paper utils", "Paper1", "figures")
PAPERS_DIR = os.path.join(ROOT, "papers")
os.makedirs(PAPERS_DIR, exist_ok=True)
OUT_PDF = os.path.join(PAPERS_DIR, "Cafri_HEP_diagnosis_research_proposal.pdf")

# --- Preliminary-results numbers (from the conference abstract + Paper1 caches) ---
P = dict(
    corpus_n=90166, montage_n=2443,
    rem_ds_n=2136, rem_ds_p=0.0050, rem_ds_ch="15/19",
    rem_ls_n=2202, rem_ls_p=0.0149, rem_ls_ch="1/19",
    age_n_old=1144, age_n_young=1299, age_p=0.0050, age_ch="9/19",
    align_cohort_a=2634, align_pool=8240,
    align_diag=dict(af=1240, hf=1658, stroke=1701, dem=947),
    align_r_af=0.9995, align_cos_af=0.9977, align_pperm_af=0.94,
)

# ---------------------------------------------------------------------
# Styles: 12 pt, 1.5 spacing (leading = 18)
# ---------------------------------------------------------------------
base = getSampleStyleSheet()
BODY_LEAD = 18
styles = {
    "CoverTitleEn": ParagraphStyle("CoverTitleEn", parent=base["Title"], fontName="Helvetica-Bold",
                                   fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=10),
    "CoverTitleHe": ParagraphStyle("CoverTitleHe", parent=base["Title"], fontName="DejaVu-Bold",
                                   fontSize=13, leading=20, alignment=TA_CENTER, spaceAfter=10),
    "CoverMeta": ParagraphStyle("CoverMeta", parent=base["Normal"], fontName="Helvetica",
                                fontSize=12, leading=18, alignment=TA_CENTER, spaceAfter=6),
    "H1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=13.5,
                         spaceBefore=16, spaceAfter=8, textColor=colors.HexColor("#111111")),
    "H2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12,
                         spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#222222")),
    "Body": ParagraphStyle("Body", parent=base["Normal"], fontName="Times-Roman", fontSize=12,
                           leading=BODY_LEAD, alignment=TA_JUSTIFY, spaceAfter=8),
    "Bullet": ParagraphStyle("Bullet", parent=base["Normal"], fontName="Times-Roman", fontSize=12,
                             leading=BODY_LEAD, alignment=TA_JUSTIFY, spaceAfter=4,
                             leftIndent=18, bulletIndent=6),
    "Caption": ParagraphStyle("Caption", parent=base["Normal"], fontName="Helvetica", fontSize=9.5,
                              leading=12.5, textColor=colors.HexColor("#333333"), spaceAfter=12,
                              spaceBefore=3),
    "Ref": ParagraphStyle("Ref", parent=base["Normal"], fontName="Times-Roman", fontSize=10.5,
                          leading=14, spaceAfter=4, leftIndent=16, firstLineIndent=-16),
}


def P_(txt, s="Body"):
    story.append(Paragraph(txt, styles[s]))


def B_(txt):
    story.append(Paragraph(f"•&nbsp;{txt}", styles["Bullet"]))


story = []

# ===================================================================
# COVER PAGE
# ===================================================================
story.append(Spacer(1, 1.5 * cm))
P_("Tel Aviv University, George S. Wise Faculty of Life Sciences<br/>"
   "School of Neurobiology, Biochemistry and Biophysics", "CoverMeta")
story.append(Spacer(1, 1.2 * cm))
P_("Research Proposal for the Ph.D. Degree", "CoverMeta")
story.append(Spacer(1, 1.0 * cm))
P_("Heart-Brain Modulation Across Sleep Stages and Clinical Diagnoses in a Large Cohort",
   "CoverTitleEn")
story.append(Spacer(1, 0.4 * cm))
P_(he("מודולציה בין הלב למוח בשלבי שינה במגוון אבחנות קליניות במדגם רחב"), "CoverTitleHe")
story.append(Spacer(1, 1.6 * cm))
P_("Student: Nir Cafri", "CoverMeta")
P_("Main supervisor: Prof. Pablo Blinder, Department of Neurobiology, "
   "School of Neurobiology, Biochemistry and Biophysics, George S. Wise Faculty of Life Sciences, "
   "Tel Aviv University, Tel Aviv, Israel", "CoverMeta")
P_("Co-supervisor: Dr. Felix Benninger, Department of Neurology, Rabin Medical Center "
   "(Beilinson Hospital) and Tel Aviv University", "CoverMeta")
story.append(Spacer(1, 1.6 * cm))
P_("Submission date: ________________", "CoverMeta")
story.append(Spacer(1, 1.4 * cm))
P_("Supervisor signature: ______________________", "CoverMeta")
P_("Co-supervisor signature: ______________________", "CoverMeta")
story.append(PageBreak())

# ===================================================================
# ABSTRACT (~250 words)
# ===================================================================
P_("Summary", "H1")
P_(
    "The heartbeat-evoked potential (HEP) is an R-peak-locked EEG deflection that indexes cortical "
    "processing of cardiac afferent signals and is used as a neural marker of interoception. Prior "
    "work reports that HEP magnitude follows a vigilance-state gradient across sleep and rises with "
    "age, but these effects have been measured in small, single-site samples, have not been "
    "topographically mapped, and have never been compared against clinical diagnosis in the same "
    "framework. <b>Objectives.</b> This project asks (i) whether the sleep-stage and age HEP effects "
    "share a cortical generator and topography, and (ii) whether patients with cardiovascular, "
    "cerebrovascular or neurodegenerative diagnoses show an HEP sleep-stage profile that departs from "
    "the diagnosis-free reference, or merely a rescaled version of it. <b>Methodology.</b> Using the "
    "Human Sleep Project, a multi-centre polysomnography corpus of ~90,000 patients, HEP epochs are "
    "extracted per sleep stage from a standardised EEG/ECG cleaning pipeline. Sleep-stage and "
    "age contrasts are tested with nonparametric cluster-mass permutation tests, electrode by "
    "electrode. Diagnosis groups are then compared to a demographically matched diagnosis-free "
    "cohort using the same statistics plus a waveform-alignment permutation test and a "
    "cross-validated classifier. <b>Preliminary results.</b> In 2,136 patients the REM−deep-sleep "
    "HEP difference is robust (p = 0.005, 15/19 electrodes) and an older-vs-younger REM contrast "
    "matches it in magnitude (p = 0.005, 9/19 electrodes). In a first diagnosis analysis the "
    "sleep-stage HEP delta of diagnosed subgroups is nearly collinear with the diagnosis-free delta "
    "(Pearson r > 0.999), suggesting diagnosis rescales rather than reshapes the interoceptive "
    "gradient. The proposal develops this into a full topographic and clinical characterisation.",
    "Body")
story.append(PageBreak())

# ===================================================================
# 1. INTRODUCTION
# ===================================================================
P_("1. Introduction", "H1")

P_("1.1 The heartbeat-evoked potential and cortical interoception", "H2")
P_(
    "Every cardiac cycle generates afferent volleys, from arterial baroreceptors, "
    "mechanoreceptors and chemoreceptors, that reach the nucleus of the solitary tract and, via "
    "thalamic and parabrachial relays, the insular, somatosensory and cingulate cortices (Critchley "
    "and Garfinkel, 2017). Time-locking the EEG to the electrocardiographic R-peak and averaging "
    "isolates the heartbeat-evoked potential (HEP), a low-amplitude deflection typically largest "
    "200–400 ms after the R-peak over fronto-central and central electrodes (Park and Blanke, "
    "2019). HEP amplitude scales with attention directed to the heartbeat, with interoceptive "
    "accuracy, and with self-referential processing in the default-mode network (Babo-Rebelo et al., "
    "2016), and it is therefore widely used as a non-invasive index of how strongly the cortex "
    "represents the internal state of the body.",
    "Body")
P_(
    "The principal methodological threat to the HEP is the cardiac field artifact (CFA): the "
    "electrical field of the heart volume-conducts to the scalp with a fixed phase relationship to "
    "the same R-peak used for epoching, so it survives trial averaging exactly as a genuine cortical "
    "response does (Park and Blanke, 2019). Current methodological guidance is to exclude a short "
    "window around the QRS complex, commonly ±50 ms, from every statistical test, and "
    "to report the retained interval, the number of heartbeats, and channel-level pre/post-cleaning "
    "metrics (Steinfath et al., 2026). Any HEP study, and in particular any group comparison, has to "
    "demonstrate that a reported effect peaks outside this excluded window and is not tracking a "
    "between-group difference in cardiac electrophysiology or thoracic geometry.",
    "Body")

P_("1.2 The HEP changes with vigilance state and with age", "H2")
P_(
    "Central–autonomic coupling is not static across the night. Slow oscillations and sleep "
    "spindles gate autonomic outflow, heart-rate variability shifts toward vagal dominance in "
    "non-REM sleep, and phasic REM is accompanied by autonomic surges (de Zambotti et al., 2018). "
    "Against this background, Lechinger et al. (2015) reported that heartbeat-related EEG amplitude "
    "follows a vigilance gradient, with larger responses in wakefulness and REM than in deep sleep, "
    "and interactions with spindles and slow oscillations. Independently, ageing has been linked to "
    "a larger HEP: older adults show higher HEP amplitude than young adults during rest and during "
    "attention orienting (Kamp et al., 2021; Aprile et al., 2025), a change that has been "
    "interpreted as altered interoceptive gain or reduced afferent inhibition.",
    "Body")
P_(
    "Three gaps follow directly from this literature. First, the sleep-stage and age effects have "
    "been established in separate small cohorts (tens of participants), so it is unknown whether they "
    "reflect one modulatory axis, arousability, or two. Second, neither effect has been "
    "topographically mapped with adequate electrode coverage and multiple-comparison control, so it "
    "is unclear whether they share a cortical generator. Third, and most important for clinical "
    "translation, no study has asked whether these normative gradients are preserved, amplified or "
    "distorted in patients with disease. This matters because the diseases most likely to affect the "
    "HEP, atrial fibrillation, heart failure, stroke and neurodegeneration, are exactly "
    "the conditions that also alter sleep architecture and autonomic tone, so a naive case–"
    "control HEP comparison that is not matched for age and not stratified by sleep stage can easily "
    "mistake a physiological shift for a disease signature.",
    "Body")

P_("1.3 Why a large clinical polysomnography corpus", "H2")
P_(
    "Answering these questions requires a sample that is simultaneously large (for electrode-wise "
    "permutation statistics and demographic matching), diagnostically annotated, and recorded with "
    "synchronous EEG and ECG across sleep stages. Clinical polysomnography corpora meet all three "
    "requirements. The Human Sleep Project provides ~90,000 overnight recordings, of which a subset "
    f"(~{P['montage_n']:,}) carry the full 19-electrode 10–20 montage needed for topographic "
    "analysis, together with linked diagnosis codes. Multi-hospital clinical corpora have already "
    "been used successfully for population-scale neurophysiology in epilepsy imaging (Cafri et al., "
    "2025), and the same design is appropriate here: it trades the homogeneity of a laboratory "
    "sample for the statistical power and diagnostic range that the present questions need, provided "
    "that cohort heterogeneity is handled by explicit matching and by re-using one fixed, "
    "quality-controlled analysis pipeline for every contrast.",
    "Body")
P_(
    "<b>Problem statement.</b> The HEP is being adopted as an interoceptive biomarker, yet its two "
    "best-documented physiological modulators, sleep depth and age, are uncharacterised "
    "topographically and unquantified at scale, and their behaviour in disease is unknown. Without "
    "that baseline, HEP differences reported between patient and control groups cannot be attributed "
    "to altered cortical interoception rather than to unmatched age, unbalanced sleep-stage "
    "composition, or the cardiac field artifact. This proposal establishes the normative sleep-stage "
    "and age topography of the HEP in a large corpus and then tests, within the same framework, "
    "whether and how clinical diagnosis changes it.",
    "Body")
story.append(PageBreak())

# ===================================================================
# 2. RESEARCH OBJECTIVES
# ===================================================================
P_("2. Research objectives and hypotheses", "H1")
P_(
    "<b>Central question.</b> Do the sleep-stage and age gradients of the heartbeat-evoked potential "
    "reflect a single, topographically organised axis of cortical interoceptive gain, and is that "
    "axis preserved or altered in patients with cardiovascular, cerebrovascular and neurodegenerative "
    "disease?",
    "Body")
P_("From this, four sub-questions:", "Body")
B_("<b>Q1, Sleep-stage topography.</b> Across which electrodes, and in which post-R-peak time "
   "window, does the HEP differ between REM, light sleep and deep sleep? Hypothesis (H1): REM and "
   "light sleep differ minimally, while both diverge sharply from deep sleep over fronto-central "
   "sites, i.e. the HEP tracks arousability rather than a graded sleep-depth continuum.")
B_("<b>Q2, Age.</b> Within a single sleep stage (REM), does an older-vs-younger contrast "
   "produce a spatial pattern that overlaps the deep-sleep–REM pattern of Q1? Hypothesis (H2): "
   "the older-minus-younger REM difference is spatially correlated with, and comparable in "
   "magnitude to, the deep-sleep–REM difference, ageing shifts interoceptive gain along "
   "the same axis as sleep depth.")
B_("<b>Q3, Shared generator.</b> Do the Q1 and Q2 effects localise to a common cortical "
   "source? Hypothesis (H3): both project to insular / somatosensory-opercular and cingulate "
   "sources, consistent with a single central-interoceptive network being up- and down-regulated.")
B_("<b>Q4, Diagnosis.</b> For each diagnosis group (atrial fibrillation, heart failure, "
   "stroke, cognitive impairment / dementia), matched to a diagnosis-free cohort for age and sex, "
   "is the sleep-stage HEP profile (a) unchanged, (b) uniformly rescaled, or (c) reshaped (a "
   "different topography or time course)? Hypothesis (H4): most diagnoses rescale the normative "
   "gradient without reshaping it (the delta waveform stays collinear with the reference), whereas "
   "conditions with direct central-autonomic involvement, stroke and dementia, show a "
   "genuine change in topography.")
P_(
    "The objectives are ordered so that a negative result at one stage still leaves the later "
    "stages answerable: Q1–Q2 are descriptive and will yield an interpretable map regardless of "
    "outcome; Q3 depends on source-localisation feasibility (addressed in Risks); Q4 is the "
    "translational payoff and is designed to be robust to the exact form of the Q1 result.",
    "Body")
story.append(PageBreak())

# ===================================================================
# 3. APPROACHES / STAGES OF IMPLEMENTATION
# ===================================================================
P_("3. Approaches, stages of implementation", "H1")
P_(
    "The work is organised into five stages that map onto the four research questions plus a "
    "consolidation stage. Each stage below states its rationale, the key analytic choices, the "
    "expected duration, the foreseen difficulties, and the fallback if the primary plan fails. "
    "Durations assume a four-year Ph.D. and are summarised in the Gantt chart in Section 6.",
    "Body")

P_("Stage 0, Corpus assembly and pipeline freeze (months 1–6)", "H2")
P_(
    "Rationale: every downstream contrast must run on identically processed data, so the pipeline is "
    "fixed and version-locked before any hypothesis test. Tasks: ingest EDF recordings and "
    "hypnograms; identify EEG and ECG channels by a 10–20 / 10–10 name whitelist (excluding "
    "intracranial and auxiliary channels present in this heterogeneous corpus); band-pass filter "
    "(0.5–100 Hz) and notch at the detected line frequency with MNE-Python (Gramfort et al., "
    "2013); detect R-peaks; epoch −300 to +400 ms around each R-peak; reject bad epochs with "
    "AutoReject; standardise per channel. Link diagnosis codes and demographics from the hospital "
    "records. Deliverable: a frozen per-patient, per-sleep-stage HEP cache with a documented "
    "quality-control manifest. Difficulty: channel-naming and montage heterogeneity across sites; "
    "mitigated by the whitelist plus manual audit of a random 200-recording sample. Fallback: if "
    "full-montage yield is lower than expected, restrict topographic analyses to the six "
    "consistently present sites (F3/F4/C3/C4/O1/O2) and treat the full montage as a secondary "
    "analysis.",
    "Body")

P_("Stage 1, Sleep-stage HEP topography, Q1 (months 6–14)", "H2")
P_(
    "Within each patient, compute the per-stage HEP evoked average per electrode, then the paired "
    "REM−deep-sleep, REM−light-sleep and light-sleep−deep-sleep differences. Test each "
    "with a cluster-mass permutation test over electrodes × time (Maris and Oostenveld, 2007), "
    "200–000–500 permutations, α = 0.01, with the ±50 ms QRS window excluded from "
    "every test. Report per-electrode corrected p-value topomaps and the peak significant latency. "
    "Difficulty: unequal epoch counts per stage bias the evoked-average noise floor; mitigated by "
    "epoch-count matching within patient before averaging and by a non-heartbeat-locked "
    "surrogate-event control that re-epochs the same window around random pseudo-events. Fallback: "
    "if paired within-patient data are too sparse for one stage pair, fall back to an "
    "age-and-sex-matched between-patient contrast.",
    "Body")

P_("Stage 2, Age contrast and axis overlap, Q2 (months 12–20)", "H2")
P_(
    "Within REM only, split patients at the median age and run the same electrode-wise permutation "
    "contrast (older − younger). Quantify the spatial overlap between the Stage-2 map and the "
    "Stage-1 REM−deep-sleep map by the Pearson correlation of their per-electrode effect sizes, "
    "with a spin/permutation null. Repeat with age as a continuous predictor in a per-electrode "
    "linear mixed model (fixed effects: age, sex, stage; random intercept: patient; random slope: "
    "site) to check that the median split is not creating the effect. Difficulty: age confounds "
    "with diagnosis burden and medication; mitigated by repeating the contrast within the "
    "diagnosis-free subset. Fallback: if the continuous model does not converge, report the "
    "tertile ANOVA used in preliminary work.",
    "Body")

P_("Stage 3, Source localisation, Q3 (months 20–28)", "H2")
P_(
    "Project the significant sensor-level differences to cortex with a template-MRI boundary-element "
    "forward model and dSPM/eLORETA (Gramfort et al., 2013), and test whether the Stage-1 and "
    "Stage-2 source maps overlap in insular, opercular-somatosensory and cingulate regions of "
    "interest. Difficulty: this is a clinical corpus without individual MRIs and with a sparse "
    "montage, so source estimates are coarse; this stage is explicitly exploratory. Fallback: if "
    "source localisation is judged unreliable, substitute a sensor-space generator analysis, "
    "spatial-pattern similarity, topographic PCA, and comparison against published HEP source "
    "topographies, which is sufficient to address whether the two effects share a spatial "
    "signature.",
    "Body")

P_("Stage 4, Diagnosis comparison, Q4 (months 24–42)", "H2")
P_(
    "For each of the four index diagnoses, build a diagnosis-free comparison cohort matched on age "
    "(±3 years), sex and recording site by nearest-neighbour matching. Three complementary "
    "analyses per diagnosis: (i) the Stage-1 electrode-wise permutation contrast, run diagnosis vs "
    "matched reference within each sleep stage; (ii) a waveform-alignment permutation test that "
    "asks whether the diagnosis group's sleep-stage delta waveform is collinear with the reference "
    "delta (observed cosine / Pearson distance vs a label-shuffling null), a positive result "
    "means ‘rescaled, not reshaped’; (iii) a cross-validated classifier (logistic "
    "regression and gradient boosting) trained on HEP + heart-rate + T-wave features to test "
    "whether diagnosis is decodable from the interoceptive profile, with permutation-based chance "
    "estimation and feature-importance reporting. Difficulty: comorbidity overlap (many patients "
    "carry several index diagnoses); mitigated by a ‘single-diagnosis-only’ sensitivity "
    "analysis and by modelling comorbidity count as a covariate. Fallback: if matched-cohort sizes "
    "are too small for a given diagnosis, pool into broader categories (any-cardiac, any-"
    "cerebrovascular, any-neurodegenerative).",
    "Body")

P_("Stage 5, Consolidation, CFA control analyses, and writing (months 40–48)", "H2")
P_(
    "Re-run every headline contrast with (a) an ECG-informed ICA cleaning step and (b) a per-channel "
    "regression of the HEP evoked average on the patient's own ECG evoked average, confirming that "
    "each reported effect survives both and peaks outside the QRS window. Assemble three "
    "manuscripts (sleep-stage/age topography; source/generator; diagnosis comparison) and the "
    "dissertation.",
    "Body")
story.append(PageBreak())

# ===================================================================
# 4. MATERIALS AND METHODS (<= 1 page)
# ===================================================================
P_("4. Materials and methods", "H1")
P_(
    "<b>Data.</b> The Human Sleep Project multi-centre clinical polysomnography corpus "
    f"(~{P['corpus_n']:,} overnight recordings; ~{P['montage_n']:,} with the full 19-electrode "
    "10–20 montage), accessed under approval from the Brain Data Science Platform "
    "(bdsp.io/content/hsp). Each recording provides synchronous scalp EEG and ECG, a "
    "technician-scored hypnogram, and linked demographic and ICD diagnosis codes. No new data are "
    "collected.",
    "Body")
P_(
    "<b>Design.</b> Retrospective, observational, within- and between-subject contrasts on existing "
    "recordings; no intervention. Primary units of analysis are per-patient, per-sleep-stage, "
    "per-electrode HEP evoked averages.",
    "Body")
P_(
    "<b>Signal processing.</b> EEG/ECG read and filtered (0.5–100 Hz band-pass, harmonic line "
    "notch) with MNE-Python (Gramfort et al., 2013); resampling to 256 Hz; bad-channel repair via "
    "the PREP pipeline (Bigdely-Shamlo et al., 2015) then interpolation; residual-artifact removal "
    "with AutoReject (Jas et al., 2017); per-channel standardisation. HEP epochs −300 to +400 "
    "ms about each R-peak; ±50 ms QRS window excluded from all tests (Steinfath et al., 2026).",
    "Body")
P_(
    "<b>Statistics.</b> Nonparametric cluster-mass permutation tests over electrodes × time "
    "(Maris and Oostenveld, 2007), α = 0.01; per-electrode linear mixed models for continuous "
    "age; Mann–Whitney / Kruskal–Wallis for skewed scalar summaries with Benjamini–"
    "Hochberg FDR control; nearest-neighbour cohort matching for diagnosis contrasts; "
    "cross-validated logistic regression and gradient boosting (scikit-learn; Pedregosa et al., "
    "2011) with permutation chance estimation for the decoding analysis.",
    "Body")
P_(
    "<b>Computing.</b> Analyses run on the laboratory's Linux compute server; all pipeline code is "
    "version-controlled and re-used unchanged across contrasts.",
    "Body")
story.append(PageBreak())

# ===================================================================
# 5. PRELIMINARY RESULTS
# ===================================================================
P_("5. Preliminary results", "H1")
P_(
    "The following results are from the student's own unpublished analyses of the corpus and "
    "establish that the experimental system can answer the research questions. They are summarised "
    "in Figure 1 at the end of this document.",
    "Body")

P_("5.1 A robust, wide-field sleep-stage HEP difference", "H2")
P_(
    f"In N = {P['rem_ds_n']:,} patients with paired REM and deep-sleep epochs, the REM−deep-sleep "
    f"HEP difference was significant by cluster-mass permutation test (p = {P['rem_ds_p']:.3f}; "
    f"significant at {P['rem_ds_ch']} electrodes), with the effect peaking well outside the excluded "
    f"QRS window. The REM−light-sleep difference was markedly smaller and did not survive "
    f"threshold (N = {P['rem_ls_n']:,}; p = {P['rem_ls_p']:.3f}; {P['rem_ls_ch']} electrodes). This "
    "two-tier pattern, REM ≈ light sleep ≫ deep sleep, is the basis for "
    "hypothesis H1 and shows the corpus has the power and montage coverage for electrode-wise "
    "topographic inference.",
    "Body")

P_("5.2 An age effect of comparable magnitude, along the same axis", "H2")
P_(
    f"Within REM, an older-vs-younger median-split contrast (N = {P['age_n_old']:,} vs "
    f"{P['age_n_young']:,}) was significant (p = {P['age_p']:.3f}; {P['age_ch']} electrodes) and "
    "comparable in magnitude to the REM−deep-sleep contrast of 5.1. This is the preliminary "
    "support for H2: ageing appears to move the HEP along the same sleep-depth axis rather than "
    "producing an orthogonal pattern. Formal spatial-overlap testing (Stage 2) has not yet been "
    "run.",
    "Body")

P_("5.3 First diagnosis comparison: rescaling, not reshaping", "H2")
P_(
    f"An initial diagnosis analysis matched a diagnosis-free cohort (n = {P['align_cohort_a']:,}) to "
    f"four diagnosis subgroups, atrial fibrillation (n = {P['align_diag']['af']:,}), heart "
    f"failure (n = {P['align_diag']['hf']:,}), stroke / cerebrovascular (n = "
    f"{P['align_diag']['stroke']:,}) and cognitive impairment / dementia (n = "
    f"{P['align_diag']['dem']:,}), and compared the sleep-stage HEP delta vectors. For atrial "
    f"fibrillation the diagnosis delta was almost perfectly collinear with the diagnosis-free delta "
    f"(Pearson r = {P['align_r_af']:.4f}; cosine similarity {P['align_cos_af']:.4f}; alignment "
    f"permutation p ≈ {P['align_pperm_af']:.2f}, i.e. no detectable change in shape). This "
    "directly motivates Q4's three-way ‘unchanged / rescaled / reshaped’ framing and "
    "suggests that the interesting deviations, if any, will be concentrated in the "
    "central-autonomic diagnoses (stroke, dementia), which is where Stage 4 is powered to look.",
    "Body")

story.append(PageBreak())

# ===================================================================
# 6. TIMELINE AND TASKS
# ===================================================================
P_("6. Timeline and tasks", "H1")
P_(
    "Durations are in project months over a four-year Ph.D. Overlap between stages is intentional: "
    "analysis of one stage proceeds while the next stage's data are prepared.",
    "Body")

gantt = [
    ["Stage / task", "Objective", "Months", "Y1", "Y2", "Y3", "Y4"],
    ["0 Corpus assembly & pipeline freeze", "Infrastructure", "1–6", "■", "", "", ""],
    ["1 Sleep-stage HEP topography", "Q1 / H1", "6–14", "■", "■", "", ""],
    ["2 Age contrast & axis overlap", "Q2 / H2", "12–20", "", "■", "", ""],
    ["3 Source / generator analysis", "Q3 / H3", "20–28", "", "■", "■", ""],
    ["4 Diagnosis comparison", "Q4 / H4", "24–42", "", "", "■", "■"],
    ["5 CFA controls, consolidation, writing", "All", "40–48", "", "", "", "■"],
    ["Manuscript 1 (topography)", "", "16–22", "", "■", "", ""],
    ["Manuscript 2 (generator)", "", "28–34", "", "", "■", ""],
    ["Manuscript 3 (diagnosis) + thesis", "", "42–48", "", "", "", "■"],
]
t = Table(gantt, colWidths=[6.0 * cm, 3.0 * cm, 2.0 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm])
t.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9.5),
    ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dddddd")),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
    ("ALIGN", (2, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
]))
story.append(t)
story.append(Spacer(1, 10))

# ===================================================================
# 7. RISKS AND SOLUTIONS
# ===================================================================
P_("7. Risks and solutions", "H1")
B_("<b>Cardiac field artifact drives a group effect.</b> Every headline contrast is re-run with "
   "ECG-informed ICA and with per-channel HEP-on-ECG regression, and only effects that survive both "
   "and peak outside the ±50 ms QRS window are reported. Diagnosis contrasts are additionally "
   "age/sex/site matched, since CFA covaries with body habitus and sex.")
B_("<b>Cohort heterogeneity / montage sparsity.</b> If full-19-electrode yield is low, topographic "
   "claims are restricted to the six consistently covered sites and the full montage becomes a "
   "secondary analysis; between-site effects are modelled as a random factor.")
B_("<b>Diagnosis subgroups too small after matching.</b> Pool into broad categories (any-cardiac, "
   "any-cerebrovascular, any-neurodegenerative); report the matched sample size and minimum "
   "detectable effect for every contrast.")
B_("<b>Comorbidity confounding.</b> Sensitivity analysis on single-diagnosis-only patients plus "
   "comorbidity-count covariate; pre-register the primary contrast for each diagnosis.")
story.append(PageBreak())

# ===================================================================
# REFERENCES
# ===================================================================
P_("References", "H1")
refs = [
    "Aprile F, et al. (2025) The heartbeat-evoked potential in young and older adults during "
    "attention orienting. Psychophysiology, e70057.",
    "Babo-Rebelo M, Richter CG, Tallon-Baudry C (2016) Neural responses to heartbeats in the "
    "default network encode the self in spontaneous thoughts. Journal of Neuroscience, "
    "36(30):7829–7840.",
    "Bigdely-Shamlo N, Mullen T, Kothe C, Su K-M, Robbins KA (2015) The PREP pipeline: standardized "
    "preprocessing for large-scale EEG analysis. Frontiers in Neuroinformatics, 9:16.",
    "Cafri N, Mirloo S, Zarhin D, Kamintsky L, Serlin Y, Alhadeed L, et al.; Benninger F (2025) "
    "Imaging blood-brain barrier dysfunction in drug-resistant epilepsy: a multi-center feasibility "
    "study. Epilepsia, 66(1):195–206.",
    "Critchley HD, Garfinkel SN (2017) Interoception and emotion. Current Opinion in Psychology, "
    "17:7–14.",
    "de Zambotti M, Trinder J, Silvani A, Colrain IM, Baker FC (2018) Dynamic coupling between the "
    "central and autonomic nervous systems during sleep: a review. Neuroscience and Biobehavioral "
    "Reviews, 90:84–103.",
    "Gramfort A, Luessi M, Larson E, Engemann DA, Strohmeier D, Brodbeck C, et al. (2013) MEG and "
    "EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7:267.",
    "Jas M, Engemann DA, Bekhti Y, Raimondo F, Gramfort A (2017) Autoreject: automated artifact "
    "rejection for MEG and EEG data. NeuroImage, 159:417–429.",
    "Kamp S-M, et al. (2021) Older adults show a higher heartbeat-evoked potential than young adults "
    "and a negative association with everyday metacognition. Brain Research (PMID 33406407).",
    "Lechinger J, Heib DPJ, Gruber W, Schabus M, Klimesch W (2015) Heartbeat-related EEG amplitude "
    "and phase modulations from wakefulness to deep sleep: interactions with sleep spindles and slow "
    "oscillations. Psychophysiology, 52(11):1441–1450.",
    "Maris E, Oostenveld R (2007) Nonparametric statistical testing of EEG- and MEG-data. Journal of "
    "Neuroscience Methods, 164(1):177–190.",
    "Park H-D, Blanke O (2019) Heartbeat-evoked cortical responses: underlying mechanisms, "
    "functional roles, and methodological considerations. NeuroImage, 197:502–511.",
    "Pedregosa F, et al. (2011) Scikit-learn: machine learning in Python. Journal of Machine "
    "Learning Research, 12:2825–2830.",
    "Steinfath TP, et al. (2026) Heartbeat-evoked responses in M/EEG: a systematic review of methods "
    "with suggestions for analysis and reporting. Psychophysiology (PMID 41943417).",
]
for r in refs:
    story.append(Paragraph(r, styles["Ref"]))
story.append(PageBreak())

# ===================================================================
# FIGURES (at end, <= 5 pages)
# ===================================================================
P_("Supporting figures", "H1")

story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "abstract_3panel.png"),
          width=16.5 * cm, height=16.5 * cm * (2429 / 5539)),
    Paragraph(
        "Figure 1. Preliminary sleep-stage and age HEP contrasts (student's unpublished analysis). "
        "Each panel: mean Δ HEP waveform (left) and per-electrode corrected-significance topomap "
        "(right); red = cluster-significant (p &lt; 0.01), grey = QRS window excluded from testing. "
        f"A: REM − deep sleep (N = {P['rem_ds_n']:,}; cluster p = {P['rem_ds_p']:.3f}; "
        f"{P['rem_ds_ch']} electrodes). B: REM − light sleep (N = {P['rem_ls_n']:,}; "
        f"p = {P['rem_ls_p']:.3f}; {P['rem_ls_ch']}). C: older − younger within REM "
        f"(N = {P['age_n_old']:,} vs {P['age_n_young']:,}; p = {P['age_p']:.3f}; {P['age_ch']}). "
        "All effects peak outside the excluded QRS window.",
        styles["Caption"]),
]))

# ---------------------------------------------------------------------
doc = SimpleDocTemplate(
    OUT_PDF, pagesize=A4,
    leftMargin=2.5 * cm, rightMargin=2.5 * cm, topMargin=2.2 * cm, bottomMargin=2.2 * cm,
    title="HEP diagnosis research proposal", author="Nir Cafri",
)
doc.build(story)
print("wrote", OUT_PDF)
