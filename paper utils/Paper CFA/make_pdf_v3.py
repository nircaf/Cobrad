#!/usr/bin/env python3
"""v3 manuscript (NeuroImage format) -> papers/Cafri_CFA_EEG_paper_v3.pdf.

Narrative: how CFAR works (Fig 1) -> CFA in one patient and the cohort (Fig 2) -> how much CFAR cleans (Fig 3)
-> independent cohort, patient-specific CFAR vs. ICA (Fig 4) -> who carries the most CFA, before and after CFAR:
sex, BMI, age (Fig 5) and diagnoses (Fig 6) -> segment duration (Fig 7). Each population figure opens with
an individual example. NeuroImage (Elsevier) style: abstract <= 250 words, 3-5 highlights <= 85 characters,
numbered sections, Elsevier Harvard author-year references, Data and code availability and CRediT statements.

Inputs: paper_stats.json (incl. "v3" from make_v3_figures.py), window_stage_sensitivity_stats.json,
beilinson_summary.json, hep_cleaning_summary.json, stage_spectrum_summary.json, validate_ps_cfar_summary.json.
Run: venv/bin/python "paper utils/Paper CFA/make_pdf_v3.py"
"""
import json, math, os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
OUT_PDF = os.path.join(os.path.dirname(os.path.dirname(HERE)), "papers", "Cafri_CFA_EEG_paper_v3.pdf")
TITLE = "Cardiac field artifact in clinical EEG: who carries it and how to remove it without an ECG"


def load(name):
    return json.load(open(os.path.join(HERE, name)))


def fig(name, width=6.5 * inch):
    from PIL import Image as PILImage
    im = PILImage.open(os.path.join(FIG_DIR, name))
    return Image(os.path.join(FIG_DIR, name), width=width, height=width * im.height / im.width)


def fp(p, digits=2):
    """p-value as text; reportlab needs &lt; for '<'."""
    return "p &lt; 1e-300" if p == 0 else f"p = {p:.{digits}g}"


S = load("paper_stats.json")
SENS = load("window_stage_sensitivity_stats.json")
class Pending(dict):  # Beilinson run not finished: numbers print as "nan", Figure 3 shows a placeholder
    def __missing__(self, k):
        return float("nan")


B3 = load("beilinson_summary.json") if os.path.exists(os.path.join(HERE, "beilinson_summary.json")) else Pending()
HEPC = load("hep_cleaning_summary.json")
STS = load("stage_spectrum_summary.json")
VAL = load("validate_ps_cfar_summary.json")
V3 = S["v3"]
COHORT, CFA, ICA, STRAT, TOPO, DX = S["cohort"], S["cfa"], S["ica"], S["stratified"], S["topomap"], S["diagnosis"]
CD, BM, DOSE, DFIT = S["cleaning_demo"], S["cleaning_benchmark"], S["dose_response"], S["dose_fit"]
WT = S["v2"]["window_test"]
WT_PAIR = {(q["a"], q["b"]): q for q in WT["pairs"]}
BMIHR, CONF, DXS = S["v2"]["bmi_hr"], S["sex_bmi_confound"], S["stratified"]["dx_by_site"]
ZL = S["crosscorr_mi"]["zero_lag_r2_full_epoch"]
F1, F4, F5, F6 = V3["fig1_example"], V3["fig4"], V3["fig5"], V3["fig6_example"]
EXM, EXW = V3["fig4_examples"]
DMAX = int(DOSE["lengths"][-1]["window_minutes"])
dose_by_min = {int(r["window_minutes"]): r for r in DOSE["lengths"]}
CD_GAP = (CD["r2_pre_mean"] - CD["r2_clean_mean"]) / (CD["r2_pre_mean"] - CD["r2_own_mean"]) * 100
BMI_STRAT = STRAT["bmi"]
_f, _m = BMI_STRAT["by_sex"]["Female"], BMI_STRAT["by_sex"]["Male"]
_z = (math.atanh(_f["r_pearson"]) - math.atanh(_m["r_pearson"])) / math.sqrt(1 / (_f["n"] - 3) + 1 / (_m["n"] - 3))
P_R_SEX_DIFF = math.erfc(abs(_z) / math.sqrt(2))
STEPS = {s["minutes"]: s for s in F6["steps"]}

base = getSampleStyleSheet()
st = {
    "Title": ParagraphStyle("T", parent=base["Title"], fontName="Helvetica-Bold", fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=6),
    "Run": ParagraphStyle("R", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=9, alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceAfter=8),
    "Author": ParagraphStyle("A", parent=base["Normal"], fontName="Helvetica", fontSize=10.5, alignment=TA_CENTER, spaceAfter=4),
    "Affil": ParagraphStyle("Af", parent=base["Normal"], fontName="Helvetica", fontSize=7.8, leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceAfter=4),
    "H1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=12.5, spaceBefore=14, spaceAfter=6),
    "H2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10.3, spaceBefore=10, spaceAfter=4),
    "H3": ParagraphStyle("H3", parent=base["Normal"], fontName="Helvetica-BoldOblique", fontSize=9.6, spaceBefore=6, spaceAfter=2),
    "Body": ParagraphStyle("B", parent=base["Normal"], fontName="Times-Roman", fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6),
    "Cap": ParagraphStyle("C", parent=base["Normal"], fontName="Helvetica", fontSize=8.2, leading=10.8, textColor=colors.HexColor("#222222"), spaceBefore=4, spaceAfter=12),
    "Kw": ParagraphStyle("K", parent=base["Normal"], fontName="Helvetica", fontSize=8.8, leading=11.5, spaceAfter=4),
    "Ref": ParagraphStyle("Rf", parent=base["Normal"], fontName="Times-Roman", fontSize=8.8, leading=11.5, leftIndent=14, firstLineIndent=-14, spaceAfter=3),
}
story = []
P = lambda text, style="Body": story.append(Paragraph(text, st[style]))


def figure(name, caption, width=6.5 * inch):
    if not os.path.exists(os.path.join(FIG_DIR, name)):
        story.append(Paragraph(f"<b>[{name} pending]</b>", st["Body"]))
        story.append(Paragraph(caption, st["Cap"]))
        return
    story.append(KeepTogether([fig(name, width), Paragraph(caption, st["Cap"])]))


# =============================================================================
# Title page
# =============================================================================
P(TITLE, "Title")
P("Nir Cafri,<super>1,3</super> Felix Benninger<super>2,3</super> and Pablo Blinder<super>1,2</super>", "Author")
P("<super>1</super>Department of Neurobiology, School of Neurobiology, Biochemistry and Biophysics, George S. Wise "
  "Faculty of Life Sciences, Tel Aviv University, Tel Aviv, Israel<br/><super>2</super>Sagol School of Neuroscience, "
  "Tel Aviv University, Tel Aviv, Israel<br/><super>3</super>Department of Neurology, Rabin Medical Center, Beilinson "
  "Hospital and Tel Aviv University, Petah Tikva, Israel", "Affil")
P("Correspondence to: Pablo Blinder, Tel Aviv University, Tel Aviv 6997801, Israel. E-mail: pb@tauex.tau.ac.il", "Affil")
story.append(Spacer(1, 6))
story.append(HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#888888")))

P("Abstract", "H1")
ABSTRACT = (
    f"Every scalp EEG contains the heart's electrical field, the cardiac field artifact (CFA), usually treated as a "
    f"small, constant nuisance removed by excluding the QRS complex. We quantified CFA in {CFA['n_patients']:,} "
    f"patients from the Human Sleep Project as the share of each channel's heartbeat-evoked potential (HEP; the R-peak-locked EEG average) explained by "
    f"the ECG outside a ±50 ms QRS window (R²). Mean R² was {CFA['r2_excl_qrs_mean']:.2f}, ranging from none to "
    f"near-complete. Because the CFA scalp pattern was consistent across patients, a single learned projection, CFAR "
    f"(Cardiac Field Artifact Removal), removed CFA from held-out recordings without an ECG, leaving a median "
    f"{BM['cfar_pow_pct']:.0f}% of the ECG-explained HEP variance versus {BM['ica_pow_pct']:.0f}% after "
    f"ECG-informed independent component analysis (ICA), while preserving {BM['cfar_hep']*100:.0f}% of a simulated "
    f"neural HEP. Using the patient's own ECG, patient-specific CFAR left "
    f"{BM['own_pow_pct']:.0f}%, and in an independent clinical cohort recorded on other montages it left less "
    f"artifact than ICA ({B3['left_own_median']:.0f}% vs. {B3['left_ica_median']:.0f}%, n = {B3['n']}), at a "
    f"higher cost to neural signal. CFA was higher in men and "
    f"increased with body mass index. After CFAR, the sex difference shrank by "
    f"{(1 - F4['sex_median_diff_r2_clean'] / F4['sex_median_diff_r2_pre'])*100:.0f}%, the age association "
    f"disappeared, and significant differences between diagnosis groups fell from {F5['n_sig_pairs_r2_pre']} to "
    f"{F5['n_sig_pairs_r2_clean']} of {F5['n_pairs']} pairs. Estimated CFA rose with segment duration, from "
    f"{dose_by_min[5]['mean']:.2f} at 5 min to {dose_by_min[DMAX]['mean']:.2f} at {DMAX} min. CFA is thus a person- "
    f"and analysis-dependent confound that can create spurious group differences in EEG; it should be measured and "
    f"removed per channel, and CFAR does so without an ECG.")
assert len(ABSTRACT.split()) <= 250, len(ABSTRACT.split())  # NeuroImage limit
P(ABSTRACT)
P("Highlights", "H2")
HIGHLIGHTS = [
    f"CFA explained a mean {CFA['r2_excl_qrs_mean']*100:.0f}% of the HEP in {CFA['n_patients']:,} patients",
    f"CFAR removes cardiac artifact without ECG: {BM['cfar_pow_pct']:.0f}% left vs. {BM['ica_pow_pct']:.0f}% after ICA",
    "Patient-specific CFAR left less artifact than ICA in a second hospital",
    "Most diagnosis-group differences in EEG vanished after cardiac cleaning",
    "CFA estimates rise with segment duration; short recordings look cleaner",
]
for h in HIGHLIGHTS:
    assert len(h) <= 85, (len(h), h)  # NeuroImage: <= 85 characters incl. spaces
    P("• " + h)
P("<b>Keywords:</b> Cardiac field artifact; EEG; Heartbeat-evoked potential; Artifact removal; "
  "Independent component analysis; Body mass index; Polysomnography", "Kw")
story.append(PageBreak())

# =============================================================================
# Introduction
# =============================================================================
P("1. Introduction", "H1")
P("The heart is one of the body's strongest electrical generators, and its field is volume-conducted to the scalp. "
  "Every EEG recording therefore contains a cardiac field artifact (CFA): a heartbeat-locked deflection of "
  "non-cortical origin (Dirlich <i>et al.</i>, 1997, 1998). CFA adds structured noise to any EEG measure (Urigüen "
  "and Garcia-Zapirain, 2015; Jiang <i>et al.</i>, 2019) and survives, rather than averages out, in R-peak-locked "
  "analyses. The heartbeat-evoked potential (HEP), used to study cardiac interoception (Schandry <i>et al.</i>, "
  "1986; Coll <i>et al.</i>, 2021), is the clearest example: CFA and the cortical response to the heartbeat share "
  "the same time-locking event (Park and Blanke, 2019). The usual mitigation is to exclude ±30–50 ms around the QRS "
  "complex and sometimes to remove an ECG-related independent component (Park and Blanke, 2019; Steinfath "
  "<i>et al.</i>, 2026).")
P("CFA has been characterised only in single cohorts of a few tens of participants (Dirlich <i>et al.</i>, 1997, "
  "1998), too few to detect differences between people, so it is usually treated as a fixed property of the "
  "recording. Yet the ECG that would allow its removal is often left unused: of 132 heartbeat-evoked studies, only "
  "69 used it to remove cardiac components (Steinfath <i>et al.</i>, 2026), and about 38% did not report whether "
  "CFA was corrected at all (Virjee <i>et al.</i>, 2026). Many clinical and research EEG datasets have no ECG. If "
  "CFA differs systematically between people, a group difference in EEG may reflect cardiac electrophysiology, "
  "tissue conductivity or electrode geometry rather than the brain (Kern <i>et al.</i>, 2013; Park and Blanke, "
  "2019; Buot <i>et al.</i>, 2021). Body composition alters the conduction path from heart to scalp (Fraley "
  "<i>et al.</i>, 2005), men and women differ in thoracic anatomy, and averages over fewer events are less "
  "reliable (Boudewyn <i>et al.</i>, 2018), so CFA may depend on BMI, sex and the length of the analysed segment.")
P(f"Here we measured CFA in {CFA['n_patients']:,} patients from the Human Sleep Project (Li <i>et al.</i>, 2026), "
  f"by far the largest sample in which it has been studied, and built a method to remove it. The paper follows the "
  f"order in which a reader would use these results. We first show what CFA looks like in one typical patient and "
  f"across the cohort, and introduce CFAR (Cardiac Field Artifact Removal), which learns the heart's scalp pattern "
  f"from patients with an ECG and removes it from recordings without one. We then measure how much more CFAR "
  f"cleans than ICA, and test the comparison in an independent hospital cohort recorded on different montages. "
  f"Finally, we ask who carries the most CFA, by sex, BMI, age and diagnosis, before and after CFAR, and how the "
  f"estimate depends on segment duration.")

# =============================================================================
# Materials and methods
# =============================================================================
P("2. Materials and methods", "H1")
P("2.1. Cohorts and recordings", "H2")
P(f"<i>Human Sleep Project.</i> This is a secondary analysis of clinical polysomnography recordings from the Human "
  f"Sleep Project (HSP; Li <i>et al.</i>, 2026), a public, curated dataset with linked electronic health records "
  f"(EHR) distributed through the Brain Data Science Platform. BMI was computed from each patient's median EHR "
  f"height and weight (values outside 10–80 kg/m² excluded). EEG channels were identified by matching labels to "
  f"10-20/10-10 electrode names (at least two required); the ECG was the first channel labelled ECG/EKG. Most "
  f"recordings used a six-channel, contralateral-mastoid montage (F3, F4, C3, C4, O1, O2).")
P(f"<i>Rabin Medical Center (Beilinson).</i> To test CFAR on recordings from a different hospital, equipment and "
  f"montage, we analysed {B3['n_total']} clinical EEG recordings with a concurrent ECG lead from the Department of "
  f"Neurology, Rabin Medical Center, Beilinson Hospital: 68 in a longitudinal bipolar "
  f"(double-banana, 18 derivations) and 10 in a referential montage (19 10-20 electrodes plus Fpz and Oz), sampled "
  f"at 256 Hz. Of these, {B3['n']} ({B3['n_bipolar']} bipolar, {B3['n_referential']} referential) passed the ECG "
  f"quality check below. One referential recording with typical CFA served as the 19-electrode example "
  f"in Fig. 2.")
P("2.2. Preprocessing and CFA estimation", "H2")
P("Because CFA rose with segment length and approached a plateau by about 45 min (Fig. 7), the main analysis used "
  "one 45-min window per recording, chosen by a seeded random search (up to 10 attempts). R-peaks were detected on "
  "the median-subtracted ECG after 5–25 Hz band-pass filtering and rectification (minimum spacing 300 ms, prominence "
  "≥3 × median absolute deviation). A window was accepted if ≥70% of EEG channels passed quality criteria (Table 1), "
  "mean heart rate was 35–180 beats/min, and ≥70% of R-R intervals were 0.33–2.0 s. EEG and ECG were then filtered "
  "from 1 Hz to min(100 Hz, Nyquist − 0.5 Hz) (MNE-Python; Gramfort <i>et al.</i>, 2013) and analysed without "
  "re-referencing, interpolation or amplitude-based rejection. In the Beilinson leads, tall T waves were sometimes "
  "detected as beats; there, peaks below half the typical R height were discarded and recordings without a "
  "plausible rhythm (35–120 beats/min, R-R coefficient of variation &lt;0.25) were excluded "
  f"({B3['n_excluded']} of {B3['n_total']}).")
P("HEP epochs spanned −300 to 400 ms around each R-peak, without baseline correction (≥20 epochs; median 4,186 per "
  "45-min window). For each channel, the R-peak-locked EEG average (the heartbeat-evoked potential, HEP) was compared with the ECG average over "
  "the same beats. The CFA index was the squared zero-lag Pearson correlation (R²) between the two waveforms outside "
  "a ±50 ms QRS-exclusion window: the share of the HEP's shape explained by the ECG. Patient-level analyses averaged "
  "R² over the patient's channels (patient-mean CFA R²). Because R² is scale-free, cleaning was also scored by the "
  "ECG-explained HEP variance, R² × the variance of the HEP outside the QRS, which falls when the artifact shrinks.")
PREPROC = [
    ("Window selection", "Seeded random search, uniform start times, up to 10 attempts; 45 min (main), 5–90 min (duration analysis)"),
    ("R-peak detection", "Median subtraction; 5–25 Hz third-order zero-phase Butterworth; rectification; spacing ≥300 ms; prominence ≥3 × MAD; Beilinson: R-height filter"),
    ("EEG channel quality", "≥99.9% finite samples; SD 0.1–500 µV; peak-to-peak ≤5,000 µV; &lt;20% flat samples"),
    ("Window acceptance", "≥70% of channels pass; heart rate 35–180 beats/min; ≥70% of R-R intervals 0.33–2.0 s"),
    ("Analysis filter", "Zero-phase FIR, 1 Hz to min(100 Hz, Nyquist − 0.5 Hz), EEG and ECG"),
    ("HEP epochs", "−300 to 400 ms around each R-peak; no baseline correction; ≥20 epochs; ±50 ms QRS exclusion for R²"),
]
tbl = Table([[Paragraph("<b>Step</b>", st["Cap"]), Paragraph("<b>Parameters</b>", st["Cap"])]]
            + [[Paragraph(a, st["Cap"]), Paragraph(b, st["Cap"])] for a, b in PREPROC], colWidths=[1.5 * inch, 5.0 * inch])
tbl.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
story.append(KeepTogether([Paragraph("<b>Table 1.</b> Preprocessing and quality-control parameters.", st["Cap"]), tbl]))
P("2.3. ECG-informed independent component analysis", "H2")
P("ICA (Hyvärinen and Oja, 2000; extended Picard, Ablin <i>et al.</i>, 2018; min(15, channels − 1) components, "
  "fixed seed) was fitted to the filtered EEG of each window. Components correlated with the ECG channel (MNE-Python "
  "threshold) were removed; if none passed, the highest-scoring component was removed. This is the standard "
  "ECG-informed ICA. ECG-free ICA, with heart components chosen by ICLabel (Pion-Tonachini <i>et al.</i>, 2019), "
  f"found one in only {BM['ica_iclabel_found_pct']:.0f}% of patients and is reported in the text only.")
P("2.4. CFAR: cardiac field artifact removal", "H2")
P(f"CFAR is a single linear projection learned with an ECG and applied without one. For every training patient, the "
  f"first singular vector of the HEP gives that patient's CFA scalp pattern. Patterns were "
  f"sign-aligned and averaged, weighted by CFA R², into one population pattern <i>u</i>. CFAR replaces every sample "
  f"<i>x</i> of the continuous EEG by <i>x</i> − (<i>u</i>·<i>x</i>)<i>u</i>, a signal-space projection (Uusitalo "
  f"and Ilmoniemi, 1997), so it needs neither the ECG nor the timing of individual beats. In the six-channel HSP "
  f"montage, all {CD['n_test']:,} eligible patients were cleaned with {CD['n_folds']}-fold cross-fitting: each "
  f"patient's filter was learned from the other folds (about {CD['n_train']:,} patients). <i>Patient-specific "
  f"CFAR</i> (ps-CFAR) uses the same projection with the patient's own pattern, learned from that recording's ECG. "
  f"It needs the ECG, adapts to any montage, and was used for the Beilinson cohort, where no population filter "
  f"existed for its montages.")
P("2.5. Scoring the cleaning", "H2")
P("Each method was scored per patient on (i) the ECG-explained HEP variance left, as a percentage of raw, and (ii) "
  "neural signal preserved: a synthetic neural HEP (Gaussian, −2 µV at +300 ms, frontal &gt; central &gt; "
  "posterior) was added to the EEG before cleaning and the recovered fraction measured. Their ratio, neural kept ÷ "
  "cardiac left, is the net gain over raw EEG (1× = no change). In the Beilinson cohort each method was re-fitted on "
  "the injected data, so a method that learns from the HEP can absorb part of the neural response. Medians are "
  "reported with 95% bootstrap confidence intervals. Validation analyses that hold out data, compare power spectra "
  "by sleep stage and score the HEP itself are in the Supplementary material.")
P("2.6. Demographic, diagnostic and duration analyses", "H2")
P(f"Diagnoses were assigned to 15 predefined categories by keyword matching on EHR descriptions (Table S1). For the analyses before and after CFAR (Figs. 5 and 6), patient-mean CFA R² was computed on the same "
  f"{F4['n']:,} six-channel patients before and after cross-fitted CFAR. Age and BMI were tested with Pearson "
  f"correlation, sex with the Mann-Whitney U test, and diagnosis categories with pairwise Mann-Whitney tests "
  f"corrected by the Benjamini-Hochberg false discovery rate (Benjamini and Hochberg, 1995). Whole-cohort statistics "
  f"(all montages) and the BMI adjustment of the sex effect (ordinary least squares) are reported where noted. "
  f"Segment duration was tested in the {DOSE['n_common_patients']:,} patients with data at all eight durations "
  f"(5–90 min) by the Friedman test and Holm-corrected paired Wilcoxon tests. Example patients were chosen by "
  f"predefined rules (closest to a group mean, or of similar age with contrasting BMI and sex) and were not used for "
  f"inference.")
P("2.7. Statistical analysis", "H2")
P("All tests were two-sided unless stated. Effect sizes are reported with their tests; with thousands of patients, "
  "small effects reach significance, so we emphasise magnitudes. Analyses were run in Python with MNE-Python, SciPy "
  "and statsmodels.")

# =============================================================================
# Results
# =============================================================================
P("3. Results", "H1")
P("3.1. CFAR: from the HEP to a learned projection", "H2")
figure("fig1_v3_cfar.png",
       f"<b>Fig. 1. How CFAR works.</b> (<b>A</b>) Five seconds of F4 EEG and the simultaneous ECG from a high-CFA HSP "
       f"patient. Each ECG R-peak (dashed) defines one epoch, −300 to 400 ms (shaded). (<b>B</b>) Aligned to the R-peak, "
       f"single-beat epochs (grey; 40 shown) average into the "
       f"heartbeat-evoked potential (HEP, blue), which follows the ECG average (dotted) beyond the QRS (grey band). "
       f"(<b>C</b>) The CFAR algorithm. Training (once, with an ECG): for each patient, R-peaks from the ECG, epoching, "
       f"averaging to the multichannel HEP, and the first spatial singular vector uᵢ; the R²-weighted mean over patients "
       f"is the population filter u (inset: u on the scalp). Inference (no ECG): every sample x of a new recording is "
       f"replaced by (I − uuᵀ)x, a single linear projection layer. (<b>D</b>) Four seconds of F4 and O1: raw EEG (faint) "
       f"and after CFAR (green), with the ECG; shaded, QRS and T wave. (<b>E</b>) Each EEG sample as one dot in the "
       f"F4–O1 plane: samples during the QRS (red) lie along the CFA direction <i>u</i>; projecting <i>u</i> out leaves "
       f"the orthogonal component (green; two-channel illustration of the six-channel projection).")
P(f"Every heartbeat adds the same electrical deflection to the scalp EEG. Cutting the EEG into windows aligned to the "
  f"ECG R-peaks and averaging them cancels the background EEG and keeps what is time-locked to the heart, the HEP "
  f"(Fig. 1A and B). In a high-CFA patient the HEP is essentially a copy of the ECG, extending well beyond the QRS "
  f"interval that is usually excluded. Across patients, this deflection had a similar scalp pattern (median |cosine| "
  f"to the population pattern {CD['pattern_consistency_median_cos']:.2f}): during each heartbeat the multichannel EEG "
  f"moves along one fixed direction in electrode space, whereas between beats it does not (Fig. 1E). CFAR exploits "
  f"this. It learns the direction once, from patients with an ECG, by taking the dominant spatial component of each "
  f"patient's HEP and averaging these over patients, and then removes it from every sample of any new recording with a "
  f"single projection (Fig. 1C). Inference needs neither the ECG nor the timing of individual beats, and the "
  f"heartbeat-locked deflections disappear from the continuous EEG (Fig. 1D).")

P("3.2. The heart's field reaches the whole scalp in a typical patient and across the cohort", "H2")
figure("fig2_v3_cfa_cohort.png",
       f"<b>Fig. 2. Cardiac field artifact in one patient and across the cohort.</b> (<b>A</b>–<b>C</b>) A Rabin Medical "
       f"Center patient with typical CFA, 19-electrode referential montage. (<b>A</b>) CFA R² per electrode (outside "
       f"QRS); mean {F1['mean_r2']:.2f}, near the cohort median ({V3['cohort_median_r2']:.2f}). (<b>B</b>) Five seconds "
       f"of {F1['site']} EEG with the simultaneous ECG (scaled); grey bands, QRS. (<b>C</b>) {F1['site']} HEP "
       f"({F1['n_beats']:,} beats) with the ECG average (dotted); grey, QRS window excluded from R² (R² "
       f"{F1['site_r2']:.2f}). (<b>D</b>–<b>F</b>) All {CFA['n_patients']:,} patients. (<b>D</b>) Channel R² over the "
       f"full epoch and outside the QRS. (<b>E</b>) Mean R² per electrode (sites with ≥100 patients). (<b>F</b>) R² "
       f"distribution per electrode (box, interquartile range; line, median).")
P(f"In a typical patient, the heartbeat left a deflection on every electrode (Fig. 2A–C). Across 19 electrodes the "
  f"ECG explained a mean {F1['mean_r2']*100:.0f}% of the HEP shape outside the QRS, reaching "
  f"{F1['site_r2']:.2f} at {F1['site']}. This patient was typical: across {CFA['n_patients']:,} HSP patients, mean R² "
  f"outside the QRS was {CFA['r2_excl_qrs_mean']:.2f} (median {CFA['r2_excl_qrs_median']:.2f}), only modestly below "
  f"the full-epoch value ({CFA['r2_full_mean']:.2f}; Fig. 2D), so the conventional QRS exclusion removed only about "
  f"{(1 - CFA['r2_excl_qrs_mean']/CFA['r2_full_mean'])*100:.0f}% of the ECG-explained signal. Around random "
  f"pseudo-events R² was {ZL['non_locked']['mean']:.2f}, far below the {ZL['pre_ica']['mean']:.2f} found for "
  f"R-peak-locked averages, and ECG-informed ICA attributed a median "
  f"{ICA['component_variance_fraction_median_unfiltered']*100:.0f}% of HEP variance to the cardiac "
  f"component (Figs. S2 and S3). CFA varied across the scalp, from {TOPO['highest_mean']:.2f} at "
  f"{TOPO['highest_site']} to {TOPO['lowest_mean']:.2f} at {TOPO['lowest_site']} (Fig. 2E and F).")

P("3.3. CFAR removes more cardiac artifact than ICA", "H2")
figure("fig2_v3_cleaning.png",
       f"<b>Fig. 3. CFAR cleans more of the cardiac artifact than ICA.</b> (<b>A</b>) Five seconds of "
       f"{CD['example_site']} EEG from a high-CFA HSP patient, raw (orange) and after CFAR (black); dashed lines, "
       f"R-peaks. (<b>B</b>) HEP at {CD['hep_site']} in a second high-CFA patient before and after CFAR; grey, QRS "
       f"window. (<b>C</b>) The CFAR filter: the population CFA scalp pattern. (<b>D</b>) Cleaning pipelines in "
       f"{BM['n']:,} patients (CFAR cross-fitted). Left, ECG-explained HEP variance left (% of raw; lower is "
       f"better). Right, simulated neural HEP preserved (%). Medians with 95% bootstrap CIs; hatched, ECG used. "
       f"(<b>E</b>) Net gain per patient: neural signal preserved ÷ ECG-explained variance left, relative to raw EEG "
       f"(dashed, 1×).")
P(f"CFAR reduced mean CFA R² from {CD['r2_pre_mean']:.2f} to {CD['r2_clean_mean']:.2f} in held-out patients "
  f"(Wilcoxon {fp(CD['p_pre_vs_clean'])}), with R² falling in {CD['pct_patients_reduced']:.0f}% of patients and "
  f"the deflection after every R-peak removed in the examples (Fig. 3A and B). Measured as ECG-explained HEP "
  f"variance, CFAR left a median {BM['cfar_pow_pct']:.0f}% of the raw value, against {BM['ica_pow_pct']:.0f}% "
  f"after ECG-informed ICA, which needs an ECG, and {BM['ica_iclabel_pow_pct']:.0f}% after ECG-free ICA, which "
  f"found a heart component in only {BM['ica_iclabel_found_pct']:.0f}% of patients (Fig. 3D). Applying ICA after "
  f"CFAR left {BM['cfar_ica_pow_pct']:.0f}%, and patient-specific CFAR left {BM['own_pow_pct']:.0f}%. The cost was "
  f"modest: CFAR preserved {BM['cfar_hep']*100:.0f}% of a simulated neural HEP (ICA "
  f"{BM['ica_hep']*100:.0f}%). Combining both, the neural-to-cardiac ratio rose {BM['cfar_gain']:.1f}-fold with "
  f"CFAR and {BM['own_gain']:.1f}-fold with patient-specific CFAR, against {BM['ica_gain']:.1f}-fold with ICA "
  f"(Fig. 3E). In the ongoing EEG, the cardiac share of variance fell by a median "
  f"{CD['frac_drop_median_pct']:.0f}%. Predicting each patient's pattern from BMI, age, sex and diagnoses did not "
  f"improve on the single population pattern (mean R² {CD['r2_cov_mean']:.3f} vs. {CD['r2_clean_mean']:.3f}).")
P(f"Three checks guarded against an optimistic estimate (Figs. S10–S12). Learned on one half of a "
  f"recording and scored on the other, patient-specific CFAR still raised the neural-to-cardiac ratio "
  f"{VAL['gain_cross_median']:.1f}-fold (vs. {VAL['gain_in_median']:.1f}-fold in-sample and "
  f"{VAL['gain_pop_median']:.1f}-fold for CFAR), and a pattern learned from random event times gave "
  f"{VAL['gain_fake_median']:.2f}-fold, i.e. no cleaning; the ECG could no longer be reconstructed from the cleaned "
  f"EEG (R² {VAL['ecg_r2_raw_median']:.3f} to {VAL['ecg_r2_clean_median']:.4f}). Across {STS['n']} whole nights "
  f"scored by technicians, the EEG spectrum after patient-specific CFAR and after ICA matched the raw spectrum in "
  f"every sleep stage (shape r {STS['shape_r_own']:.3f} and {STS['shape_r_ica']:.3f}), whereas the HEP spectrum "
  f"did not (r {STS['shape_r_hep']:.2f}; peak frequency off by a median {STS['peak_err_hep']:.1f} Hz). Within the "
  f"200–600 ms window used in HEP studies, patient-specific CFAR left {HEPC['winpow_pct_own']:.0f}% of the "
  f"ECG-explained power, CFAR {HEPC['winpow_pct_cfar']:.0f}% and ICA {HEPC['winpow_pct_ica']:.0f}%.")

P("3.4. In an independent hospital cohort, patient-specific CFAR reveals heart-free EEG where ICA does not", "H2")
figure("fig3_v3_beilinson.png",
       f"<b>Fig. 4. What patient-specific CFAR reveals that ICA does not, in an independent clinical cohort (Rabin "
       f"Medical Center, n = {B3['n']}).</b> Each recording was cleaned on its own montage ({B3['n_bipolar']} bipolar, "
       f"{B3['n_referential']} referential). (<b>A</b>) HEP at {B3['example']['channel']} of one referential recording, the "
       f"channel where ICA left the most ECG shape: raw EEG (dashed), ICA and patient-specific CFAR (ps-CFAR), with the "
       f"ECG average (dotted); display low-pass 30 Hz. (<b>B</b>) ECG-explained HEP power over the heartbeat (channel "
       f"mean, median over patients; log scale). (<b>C</b>) Share of each patient's channels free of CFA (R² &lt; 0.1); "
       f"lines join one patient, bars are medians. (<b>D</b>) Mean CFA R² per electrode in the {B3['n_referential']} "
       f"19-electrode recordings.")
P(f"The HSP recordings shared one montage and one reference. To test the cleaning elsewhere, we analysed {B3['n']} "
  f"recordings from Rabin Medical Center on their own bipolar or referential montages ({B3['n_excluded']} excluded "
  f"for an implausible ECG rhythm). In many of them ICA did not change the HEP at all: at "
  f"{B3['example']['channel']} of one patient, the HEP after ICA was still a copy of the ECG (R² "
  f"{B3['example']['r2_ica']:.2f}, raw {B3['example']['r2_raw']:.2f}), whereas after patient-specific CFAR the ECG "
  f"shape was gone (R² {B3['example']['r2_own']:.2f}; Fig. 4A). Across patients, ICA barely lowered the ECG-explained "
  f"HEP power over the heartbeat, including the T-wave interval, whereas patient-specific CFAR lowered it about "
  f"{B3['twave_power']['ica'] / B3['twave_power']['own']:.0f}-fold relative to ICA between 150 and 400 ms (Fig. 4B). "
  f"The median share of channels free of CFA rose from {B3['cfa_free_pct_raw']:.0f}% (raw) and "
  f"{B3['cfa_free_pct_ica']:.0f}% (ICA) to {B3['cfa_free_pct_own']:.0f}% (Wilcoxon vs. ICA "
  f"{fp(B3['p_free_own_vs_ica'])}; Fig. 4C), and in the 19-electrode recordings the scalp map of CFA after ICA was "
  f"nearly identical to raw (mean R² {B3['ref_mean_r2_ica']:.2f} vs. {B3['ref_mean_r2_raw']:.2f}), whereas after "
  f"patient-specific CFAR it was largely cleared ({B3['ref_mean_r2_own']:.2f}; Fig. 4D). Overall, patient-specific "
  f"CFAR left a median {B3['left_own_median']:.0f}% of the ECG-explained HEP variance against {B3['left_ica_median']:.0f}% "
  f"after ICA ({fp(B3['p_left'])}). The cost was larger than in the HSP: re-fitted on data containing an injected "
  f"neural response, patient-specific CFAR kept a median {B3['hep_own_median']*100:.0f}% of it, against "
  f"{B3['hep_ica_median']*100:.0f}% for ICA (Fig. S13), because where CFA is weak its pattern also captures "
  f"heartbeat-locked brain activity. For heartbeat-evoked analyses on such montages, a population filter learned on "
  f"other patients of the same montage is preferable.")

P("3.5. Men and people with higher BMI carry more cardiac artifact; CFAR removes most of the sex difference", "H2")
figure("fig4_v3_demographics.png",
       f"<b>Fig. 5. CFA by sex, BMI and age, before and after CFAR.</b> (<b>A</b> and <b>B</b>) Two patients of "
       f"similar age (60-min segments): (<b>A</b>) a man with high BMI ({EXM['bmi']:.1f} kg/m²; mean R² "
       f"{EXM['mean_r2']:.2f}) and (<b>B</b>) a woman with low BMI ({EXW['bmi']:.1f} kg/m²; R² "
       f"{EXW['mean_r2']:.2f}). Left, R² per electrode; right, {EXM['site']} HEP raw (orange) and after "
       f"patient-specific CFAR (green), with the ECG average (dotted). (<b>C</b>–<b>E</b>) The {F4['n']:,} "
       f"six-channel HSP patients, before (raw) and after cross-fitted CFAR. (<b>C</b>) Age. (<b>D</b>) Sex; Δ, "
       f"male minus female median. (<b>E</b>) BMI by sex (n = {F4['bmi_n_Female'] + F4['bmi_n_Male']:,}); solid, "
       f"raw; dashed, after CFAR; least-squares lines.")
P(f"In a man with high BMI the ECG explained {EXM['mean_r2']*100:.0f}% of the HEP across the "
  f"scalp and the {EXM['site']} HEP was a copy of the ECG; patient-specific CFAR reduced R² at {EXM['site']} from "
  f"{EXM['site_r2']:.2f} to {EXM['site_r2_own']:.2f} (Fig. 5A). In a woman of the same age with low BMI, R² was "
  f"{EXW['mean_r2']:.2f} and the HEP showed no ECG shape, before or after cleaning (Fig. 5B).")
P(f"Across the cohort, men had higher CFA than women (median R² {F4['sex_median_r2_pre']['Male']:.2f} vs. "
  f"{F4['sex_median_r2_pre']['Female']:.2f}; whole cohort {fp(STRAT['p_sex_mannwhitney'])}), a difference not "
  f"explained by BMI (sex coefficient {CONF['sex_unadj_coef']:.3f} unadjusted, {CONF['sex_adj_coef']:.3f} after "
  f"BMI adjustment; Fig. S5). CFA increased with BMI (r = {F4['bmi_r_all_r2_pre']:.2f}; each 10 kg/m² "
  f"added {F4['bmi_slope10_Female_r2_pre']:.3f} R² in women and {F4['bmi_slope10_Male_r2_pre']:.3f} in men) and "
  f"decreased slightly with age (r = {F4['age_r_r2_pre']:.2f}). After CFAR, the sex difference shrank from "
  f"{F4['sex_median_diff_r2_pre']:.3f} to {F4['sex_median_diff_r2_clean']:.3f} R² "
  f"({(1 - F4['sex_median_diff_r2_clean'] / F4['sex_median_diff_r2_pre'])*100:.0f}% smaller; Fig. 5D) and the age "
  f"association disappeared (r = {F4['age_r_r2_clean']:.2f}, {fp(F4['age_p_r2_clean'])}; Fig. 5C). The BMI "
  f"association remained (r = {F4['bmi_r_all_r2_clean']:.2f}), with the slope roughly halved "
  f"({F4['bmi_slope10_Female_r2_clean']:.3f} and {F4['bmi_slope10_Male_r2_clean']:.3f} per 10 kg/m²; Fig. 5E): "
  f"the residual CFA after one projection is itself larger in people with higher BMI.")

P("3.6. Apparent differences between diagnoses are largely cardiac artifact", "H2")
EX5 = V3["fig5_examples"]
figure("fig5_v3_diagnosis.png",
       f"<b>Fig. 6. CFA by diagnosis, before and after CFAR.</b> (<b>A</b> and <b>B</b>) The patient closest to the "
       f"group mean in the highest-CFA ({EX5[0]['category']}) and lowest-CFA ({EX5[1]['category']}) category: HEP at "
       f"the highest-CFA channel, raw (orange) and after CFAR (black), with the ECG average (dotted). (<b>C</b>) "
       f"Patient-mean CFA R² per diagnosis category (mean ± 95% CI; n in brackets), raw and after CFAR; dotted lines, "
       f"patients without a recorded diagnosis. (<b>D</b>) Pairwise Mann-Whitney tests, FDR-corrected: upper triangle, raw EEG; lower triangle, CFAR. "
       f"Categories overlap.")
P(f"Diagnosis categories differed in CFA (Fig. 6C): it was highest in {EX5[0]['category']} (mean R² "
  f"{F5[EX5[0]['category'] + '|r2_pre']:.2f}) and lowest in {EX5[1]['category']} "
  f"({F5[EX5[1]['category'] + '|r2_pre']:.2f}), and {F5['n_sig_pairs_r2_pre']} of {F5['n_pairs']} pairwise "
  f"comparisons survived FDR correction, most involving Obesity or Obstructive Sleep Apnea, in line with the BMI "
  f"association. Examples from the two extremes both carried a clear ECG-shaped HEP that CFAR flattened (Fig. 6A and "
  f"B). After CFAR, categories converged (R² {F5[EX5[1]['category'] + '|r2_clean']:.2f} to "
  f"{F5[EX5[0]['category'] + '|r2_clean']:.2f}) and only {F5['n_sig_pairs_r2_clean']} pair remained significant "
  f"(Fig. 6D). Most apparent differences in the HEP between these clinical groups therefore reflected "
  f"the cardiac field rather than the brain. Comparisons were not adjusted for age, sex or BMI, and the undiagnosed "
  f"group is confounded by recording site (see Limitations).")

P("3.7. Estimated CFA grows with segment duration", "H2")
figure("fig6_v3_duration.png",
       f"<b>Fig. 7. CFA estimates depend on segment duration.</b> (<b>A</b>) One HSP patient whose estimate rose like "
       f"the cohort's, using the first 5, 10, 20 and 60 min of the same window. Top, CFA R² per electrode (scalp maps) "
       f"with the patient-mean R²; bottom, {F6['site']} HEP (blue) and ECG average (dotted), with {F6['site']} R² and the "
       f"number of beats averaged. (<b>B</b>–<b>D</b>) Patients "
       f"with data at all eight durations (n = {DOSE['n_common_patients']:,}). (<b>B</b>) Mean R² ± 95% CI with a "
       f"saturating fit (dashed). (<b>C</b>) By sex. (<b>D</b>) By number of diagnoses.")
P(f"As beats were added, background EEG averaged out and the ECG shape emerged from the HEP: in one patient, "
  f"{F6['site']} R² rose from {STEPS[5]['site_r2']:.2f} with {STEPS[5]['n_beats']:,} beats (5 min) to "
  f"{STEPS[60]['site_r2']:.2f} with {STEPS[60]['n_beats']:,} beats (60 min), and the patient-mean R² from "
  f"{STEPS[5]['mean_r2']:.2f} to {STEPS[60]['mean_r2']:.2f}, as CFA appeared at more electrodes (Fig. 7A). Across "
  f"{WT['n']:,} duration-matched patients, mean R² rose from {dose_by_min[5]['mean']:.2f} at 5 min and "
  f"{dose_by_min[10]['mean']:.2f} at 10 min to {dose_by_min[DMAX]['mean']:.2f} at {DMAX} min (Friedman "
  f"χ² = {WT['friedman_chi2']:,.0f}, {fp(WT['friedman_p'])}, Kendall's W = {WT['kendall_w']:.2f}; Fig. 7B), "
  f"every consecutive step was significant, and R² increased from 5 to {DMAX} min in "
  f"{WT_PAIR[(5, DMAX)]['pct_patients_increase']:.0f}% of patients. Gains shrank with duration, converging to "
  f"R²∞ = {DFIT['asymptote']:.3f}. The sex difference held at every duration (Fig. 7C), and diagnosis groups "
  f"did not differ (Fig. 7D). In repeat windows from the same night, reliability (ICC) also rose with duration "
  f"(Figs. S8 and S9), and CFA did not differ by sleep stage in a staged subsample "
  f"(n = {SENS['n_patients']}, {fp(SENS['p_stage_anova'])}).")

# =============================================================================
# Discussion
# =============================================================================
P("4. Discussion", "H1")
P(f"In {CFA['n_patients']:,} patients, the heart's field was a large, systematic part of the scalp HEP. "
  f"It was present across the scalp in a typical patient, survived the standard QRS exclusion, and varied from "
  f"negligible to near-complete between people. Because its scalp pattern was shared, a single projection learned "
  f"once from patients with an ECG, CFAR, removed it from recordings without one and left less artifact than ICA, "
  f"the current standard; with the patient's own ECG, patient-specific CFAR did better still, including in an "
  f"independent hospital cohort recorded on different montages. Cleaning changed the clinical picture: much of the "
  f"sex difference, the age association and almost all differences between diagnosis groups in the HEP "
  f"were carried by the cardiac field.")
P("Why does CFAR outperform ICA? ICA must find a component whose time course resembles the ECG; when the cardiac "
  "field is weak or mixed with brain activity, part of it stays in the brain components, and ECG-free ICA rarely "
  "finds it at all. CFAR does not search: it removes a known spatial direction. The price is that brain activity "
  "sharing this direction is also removed. CFAR preserved most of a simulated neural HEP, but continuous-EEG power "
  "fell by 1–2 dB, and patient-specific CFAR, which learns from the patient's own heartbeat-locked average, absorbed "
  "part of a real heartbeat-locked brain response (Figs. S10–S12). For HEP studies, where the "
  "heartbeat-locked response is the outcome, the population filter learned on other patients is therefore the safer "
  "choice; patient-specific CFAR is best when the aim is the cleanest ongoing EEG.")
P(f"BMI was positively associated with CFA. This seems counterintuitive, because adipose tissue conducts poorly "
  f"(Gabriel <i>et al.</i>, 1996) and obesity lowers surface-ECG voltage (Fraley <i>et al.</i>, 2005; Tochikubo "
  f"<i>et al.</i>, 1999). R² is scale-free, however: it rises when the cardiac component grows relative to "
  f"everything else in the average. Obesity shifts the cardiac axis and increases ventricular mass, changing the "
  f"orientation of the cardiac dipole relative to the mastoid reference, and tissue between brain and electrode "
  f"attenuates cortical EEG at the same time. Heart rate did not explain the effect (BMI coefficient "
  f"{BMIHR['bmi_coef_unadj']:.4f} vs. {BMIHR['bmi_coef_adj']:.4f} per kg/m² after adjustment). The BMI association "
  f"persisted after CFAR, so residual CFA remains largest in the patients who had the most, and studies comparing "
  f"groups that differ in BMI should verify cleaning per channel. The sex difference, in contrast, was mostly "
  f"removed, consistent with a difference in the amount, not the pattern, of the cardiac field.")
P(f"Segment duration had the largest, most consistent effect on the estimate. Longer segments average more beats "
  f"and reduce background EEG in the average (Boudewyn <i>et al.</i>, 2018), letting the ECG shape emerge. Short "
  f"segments, including the 10-min segments common in HEP studies, therefore underestimate CFA, so a low estimate "
  f"does not show that the data are clean, and groups that differ in usable recording time will differ in apparent "
  f"CFA for methodological reasons alone.")
P("These results lead to concrete recommendations. Record the ECG and use it: measure CFA per channel, remove it, and "
  "show that HEP variance falls after cleaning while staying above a non-heartbeat-locked noise floor "
  "(Steinfath <i>et al.</i>, 2026; Virjee <i>et al.</i>, 2026). Report segment duration and the number of beats, and "
  "match them between groups. Test whether group differences in EEG survive cardiac cleaning, especially between "
  "sexes, BMI levels and clinical diagnoses. Where no ECG exists, CFAR provides a validated alternative.")
P("4.1. Limitations", "H2")
P(f"First, diagnosis records covered only part of the HSP, so the undiagnosed group came largely from a site without "
  f"diagnosis tables (mean R² {DXS['I0003_no_dx']['mean']:.2f}); comparisons among categories are unaffected, but "
  f"the with/without-diagnosis contrast is confounded by site. Second, R-peak detection was not validated against an "
  f"established algorithm; in the Beilinson leads it counted some T waves until an R-height filter was added, and "
  f"occasional T-wave detections in the HSP would mix T-wave-locked epochs into averages. Third, R² cannot separate "
  f"volume-conducted CFA from neural activity with an ECG-like time course, and epochs ended 400 ms after the "
  f"R-peak. Fourth, the population CFAR filter is specific to the montage and reference it was learned on; other "
  f"montages need their own filter or patient-specific CFAR. Fifth, CFAR removes one spatial dimension and with it "
  f"some brain activity (Figs. S10–S12). Sixth, BMI is an indirect measure of adiposity, the cohort was "
  f"clinically referred, and windows were not selected by sleep stage. Finally, as a secondary analysis, recording "
  f"parameters could not be controlled.")
P("5. Conclusions", "H1")
P(f"The cardiac field artifact is a person-, channel- and duration-dependent confound in clinical EEG that the QRS "
  f"exclusion does not remove. It is larger in men and with higher BMI, and it accounts for most apparent differences "
  f"in the HEP between diagnosis groups. CFAR removes it without an ECG, more completely than ICA, and "
  f"patient-specific CFAR extends this to any montage when the ECG is available. Without such cleaning, group "
  f"differences in EEG may partly reflect the heart rather than the brain.")

# =============================================================================
# Back matter (NeuroImage)
# =============================================================================
P("Data and code availability", "H1")
P("The Human Sleep Project dataset is publicly available under a data use agreement from the Brain Data Science "
  "Platform (<font face='Courier'>https://bdsp.io/content/hsp/</font>). The Rabin Medical Center recordings cannot be "
  "shared publicly because of patient privacy; de-identified derived measures are available from the corresponding "
  "author on reasonable request. Analysis code, including CFAR, is available from the authors on request.")
P("Ethics statement", "H1")
P("Human Sleep Project data were de-identified under the HIPAA Safe Harbor standard and shared under an approved "
  "institutional review board protocol (#2022P000417) with a waiver of informed consent, and were accessed under "
  "the Brain Data Science Platform data use agreement. Use of the Rabin Medical Center recordings was approved by the "
  "institutional Helsinki committee [approval number to be added by the authors].")
P("CRediT authorship contribution statement", "H1")
P("[To be completed by the authors.]")
P("Declaration of competing interest", "H1")
P("The authors declare no competing interests.")
P("Acknowledgements", "H1")
P("The Human Sleep Project has received support from the Glenn Foundation and the American Federation of Aging "
  "Research (AFAR), the American Academy of Sleep Medicine, the National Institutes of Health (R01NS102190, "
  "R01NS102574, R01NS107291, RF1AG064312, RF1NS120947, R01AG073410, R01HL161253, R01NS126282, R01AG073598), the "
  "National Science Foundation (NSF 2014431) and the Henry and Allison McCance Center for Brain Health.")
P("Funding", "H1")
P("[To be completed by the authors.]")
P("Supplementary materials", "H1")
P("Supplementary material associated with this article (Figs. S1–S13, Table S1) follows the references.")

# =============================================================================
# References (Elsevier Harvard: author-year, alphabetical)
# =============================================================================
P("References", "H1")
REFS = [
    "Ablin, P., Cardoso, J.-F., Gramfort, A., 2018. Faster independent component analysis by preconditioning with Hessian approximations. IEEE Trans. Signal Process. 66, 4040–4049.",
    "Arnau, S., Sharifian, F., Wascher, E., Larra, M.F., 2023. Removing the cardiac field artefact from the EEG using neural network regression. Psychophysiology 60, e14323.",
    "Benjamini, Y., Hochberg, Y., 1995. Controlling the false discovery rate: a practical and powerful approach to multiple testing. J. R. Stat. Soc. B 57, 289–300.",
    "Boudewyn, M.A., Luck, S.J., Farrens, J.L., Kappenman, E.S., 2018. How many trials does it take to get a significant ERP effect? It depends. Psychophysiology 55, e13049.",
    "Buot, A., Azzalini, D., Chaumon, M., Tallon-Baudry, C., 2021. Does stroke volume influence heartbeat evoked responses? Biol. Psychol. 165, 108165.",
    "Coll, M.-P., Hobson, H., Bird, G., Murphy, J., 2021. Systematic review and meta-analysis of the relationship between the heartbeat-evoked potential and interoception. Neurosci. Biobehav. Rev. 122, 190–200.",
    "Dirlich, G., Vogl, L., Plaschke, M., Strian, F., 1997. Cardiac field effects on the EEG. Electroencephalogr. Clin. Neurophysiol. 102, 307–315.",
    "Dirlich, G., Dietl, T., Vogl, L., Strian, F., 1998. Topography and morphology of heart action-related EEG potentials. Electroencephalogr. Clin. Neurophysiol. 108, 299–305.",
    "Fraley, M.A., Birchem, J.A., Senkottaiyan, N., Alpert, M.A., 2005. Obesity and the electrocardiogram. Obes. Rev. 6, 275–281.",
    "Gabriel, S., Lau, R.W., Gabriel, C., 1996. The dielectric properties of biological tissues: II. Measurements in the frequency range 10 Hz to 20 GHz. Phys. Med. Biol. 41, 2251–2269.",
    "Gramfort, A., Luessi, M., Larson, E., Engemann, D.A., Strohmeier, D., Brodbeck, C., Goj, R., Jas, M., Brooks, T., Parkkonen, L., Hämäläinen, M., 2013. MEG and EEG data analysis with MNE-Python. Front. Neurosci. 7, 267.",
    "Hyvärinen, A., Oja, E., 2000. Independent component analysis: algorithms and applications. Neural Netw. 13, 411–430.",
    "Jiang, X., Bian, G.-B., Tian, Z., 2019. Removal of artifacts from EEG signals: a review. Sensors 19, 987.",
    "Kern, M., Aertsen, A., Schulze-Bonhage, A., Ball, T., 2013. Heart cycle-related effects on event-related potentials, spectral power changes, and connectivity patterns in the human ECoG. NeuroImage 81, 178–190.",
    "Li, Q., Wen, S., Sun, H., Ganglberger, W., Tripathi, A., Turley, N., et al., Westover, M.B., 2026. The Human Sleep Project (HSP). Brain Data Science Platform. https://doi.org/10.60508/m3sw-rz13.",
    "Park, H.-D., Blanke, O., 2019. Heartbeat-evoked cortical responses: underlying mechanisms, functional roles, and methodological considerations. NeuroImage 197, 502–511.",
    "Pion-Tonachini, L., Kreutz-Delgado, K., Makeig, S., 2019. ICLabel: an automated electroencephalographic independent component classifier, dataset, and website. NeuroImage 198, 181–197.",
    "Schandry, R., Sparrer, B., Weitkunat, R., 1986. From the heart to the brain: a study of heartbeat contingent scalp potentials. Int. J. Neurosci. 30, 261–275.",
    "Steinfath, T.P., et al., 2026. Heartbeat-evoked responses in M/EEG: a systematic review of methods with suggestions for analysis and reporting. Psychophysiology 63, e70297.",
    "Tochikubo, O., Miyajima, E., Shigemasa, T., Ishii, M., 1999. Relation between body fat-corrected ECG voltage and ambulatory blood pressure in patients with essential hypertension. Hypertension 33, 1159–1163.",
    "Urigüen, J.A., Garcia-Zapirain, B., 2015. EEG artifact removal—state-of-the-art and guidelines. J. Neural Eng. 12, 031001.",
    "Uusitalo, M.A., Ilmoniemi, R.J., 1997. Signal-space projection method for separating MEG or EEG into components. Med. Biol. Eng. Comput. 35, 135–140.",
    "Virjee, R.-I., Kandasamy, R., Garfinkel, S.N., Carmichael, D.W., Yogarajah, M., 2026. Review of methods to derive the heartbeat-evoked potential: past practices and future directions. Soc. Cogn. Affect. Neurosci. nsag057.",
]
for r in REFS:
    P(r, "Ref")

# =============================================================================
# Supplementary material
# =============================================================================
story.append(PageBreak())
P("Supplementary material", "H1")
SUPP = [
    ("figS1_cohort.png", f"<b>Fig. S1. Cohort composition</b> ({COHORT['n_demographics']:,} HSP patients). (<b>A</b>) Age. (<b>B</b>) Sex. (<b>C</b>) The ten most prevalent diagnosis categories (overlapping)."),
    ("figS3_post_ica_variance.png", "<b>Fig. S2. HEP variance and spectral entropy before and after removing the ECG-related ICA component, against a non-heartbeat-locked noise floor.</b> (<b>A</b>) Variance, core electrodes. (<b>B</b>) Variance per electrode. (<b>C</b>) Spectral entropy, core electrodes. (<b>D</b>) Spectral entropy per electrode."),
    ("figS4_crosscorr_mi.png", f"<b>Fig. S3. EEG–ECG cross-correlation and mutual information</b> before and after ICA and for a non-heartbeat-locked control (n = {S['crosscorr_mi']['n_patients']})."),
    ("figS5_psd_comparison.png", f"<b>Fig. S4. Power spectral density of raw EEG, HEP, patient-specific CFAR and ICA (dashed)</b> (n = {S['psd_comparison']['n_patients']}). (<b>A</b>) Example patient. (<b>B</b>) Core electrodes. (<b>C</b>) Each electrode."),
    ("figS6_sex_bmi_confound.png", "<b>Fig. S5. The sex effect on CFA is unchanged by BMI adjustment</b> (male-minus-female coefficient ± 95% CI)."),
    ("figS10_cfar_pipeline.png", "<b>Fig. S6. The CFAR pipeline.</b> (<b>A</b>) Training (ECG used) and inference (no ECG). (<b>B</b>) Learning curve. (<b>C</b>) Projection weights. (<b>D</b>) Patients with detectable CFA before and after CFAR."),
    ("figS11_cfar_own_pipeline.png", "<b>Fig. S7. The patient-specific CFAR pipeline</b>, in the layout of Fig. S6."),
    ("figS8_window_optimization.png", "<b>Fig. S8. CFA R² across segment durations</b> in patients with every duration."),
    ("figS9_window_reliability.png", "<b>Fig. S9. Test–retest reliability of CFA R² versus window length</b>, repeat windows from the same night."),
    ("figS_stage_spectrum.png", f"<b>Fig. S10. The cleaned EEG keeps the raw spectrum in every sleep stage; the HEP does not</b> (n = {STS['n']} technician-scored nights). (<b>A</b>) Relative spectra, all stages. (<b>B</b>) Per stage. (<b>C</b>) Peak frequency by stage. (<b>D</b>) Agreement with raw EEG per patient and stage."),
    ("figS_hep_cleaning.png", f"<b>Fig. S11. HEP after ICA, CFAR and patient-specific CFAR</b> (n = {HEPC['n']}). (<b>A</b>) Grand-average frontal HEP. (<b>B</b>) ECG-explained HEP power over time. (<b>C</b>) Power left in the 200–600 ms window. (<b>D</b>) HEP–ECG similarity. (<b>E</b>) Recovery of a known injected brain response when each method is re-fitted."),
    ("validation/1_cross_half.png", f"<b>Fig. S12. Held-out validation of patient-specific CFAR</b> (n = {VAL['n']}). Pattern learned on one half of the recording and scored on the other half; a pattern learned from random event times as control."),
    ("figS_beilinson_cost.png", "<b>Fig. S13. Cost of patient-specific CFAR in the Rabin Medical Center cohort.</b> Per patient, ICA versus ps-CFAR: ECG-explained HEP variance left, synthetic neural HEP kept after re-fitting on the injected data, and net gain (neural kept ÷ cardiac left)."),
]
for name, cap in SUPP:
    if os.path.exists(os.path.join(FIG_DIR, name)):
        figure(name, cap)

DX_RULES = [("Obstructive Sleep Apnea", "sleep apnea"), ("Hypertension", "hypertension"), ("Diabetes", "diabetes"),
            ("Heart Failure", "heart failure"), ("Coronary Artery Disease", "coronary; atherosclerotic heart"),
            ("Heart Transplant", "heart transplant; transplant status"), ("Atrial Fibrillation", "atrial fibrillation; atrial flutter"),
            ("COPD", "chronic obstructive"), ("Obesity", "obesity; obese"), ("Depression", "depressive; depression"),
            ("Anxiety", "anxiety"), ("Cognitive Impairment / Dementia", "cognitive; alzheimer; dementia"),
            ("Stroke / Cerebrovascular", "infarction; cerebrovascular; stroke; hemorrhage"), ("Kidney Disease", "kidney; renal"), ("Anemia", "anemia")]
t = Table([[Paragraph("<b>Category</b>", st["Cap"]), Paragraph("<b>Keywords (any match)</b>", st["Cap"])]]
          + [[Paragraph(a, st["Cap"]), Paragraph(b, st["Cap"])] for a, b in DX_RULES], colWidths=[2.4 * inch, 4.1 * inch])
t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
story.append(KeepTogether([Paragraph("<b>Table S1.</b> <b>Diagnosis categories</b> and the case-insensitive "
                                     "keywords matched against EHR diagnosis descriptions.", st["Cap"]), t]))


def page_number(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(A4[0] / 2, 0.5 * inch, str(doc_.page))
    canvas.restoreState()


doc = SimpleDocTemplate(OUT_PDF, pagesize=A4, topMargin=0.8 * inch, bottomMargin=0.8 * inch, leftMargin=0.85 * inch,
                        rightMargin=0.85 * inch, title=TITLE, author="Nir Cafri")
doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
print("Wrote", OUT_PDF, os.path.getsize(OUT_PDF), "bytes")
