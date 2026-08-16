"""
Manuscript: the clinical role of the heartbeat-evoked potential (HEP) across
diagnostic categories -- assembled into a Nature-style research article using
reportlab Platypus, following the exact style/structure of
Paper1/make_combined_nature_paper.py.

Pulls all numbers from already-computed, on-disk results (no fabricated
numbers):
  - Paper1/hep_diagnosis_long_df.pkl              (cohort N's, per-category N)
  - Paper1/fig2_distribution_results.json         (Fig 2 Kruskal-Wallis)
  - Paper1/fig3_mixedmodel_results.json           (Fig 3 mixed-effects model)
  - Paper1/fig4_diagnosis_waveforms_results.json  (Fig 4 cluster permutation)
Figures reused from Paper1/figures/ (already rendered by the upstream
build_fig{1,2,3,4}_*.py scripts; none regenerated here).

Run: source venv/bin/activate && python3 Paper1/make_hep_diagnosis_paper.py
"""
import json
import os
import pickle

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
OUT_DIR = os.path.join(REPO, "Paper1")
FIG_DIR = os.path.join(OUT_DIR, "figures")
OUT_PDF = os.path.join(OUT_DIR, "Cafri_HEP_diagnosis_clinical_role_paper.pdf")

with open(os.path.join(OUT_DIR, "hep_diagnosis_long_df.pkl"), "rb") as f:
    DATA = pickle.load(f)
with open(os.path.join(OUT_DIR, "fig2_distribution_results.json")) as f:
    F2 = json.load(f)
with open(os.path.join(OUT_DIR, "fig3_mixedmodel_results.json")) as f:
    F3 = json.load(f)
with open(os.path.join(OUT_DIR, "fig4_diagnosis_waveforms_results.json")) as f:
    F4 = json.load(f)

meta_df = DATA["meta_df"]
long_df = DATA["long_df"]
avg_df = DATA["avg_df"]
AMP_WINDOW = DATA["amp_window"]
STAGES = DATA["stages"]

N_PATIENTS_TOTAL = int(meta_df["patient_id"].nunique())
N_ROWS_LONG = int(len(long_df))
N_ELECTRODES = int(long_df["electrode"].nunique())
BROAD_COUNTS = meta_df["broad_group"].value_counts().to_dict()

from collections import Counter
CAT_COUNTS = Counter()
for cats in meta_df["categories"]:
    CAT_COUNTS.update(cats)
CAT_TABLE = sorted(CAT_COUNTS.items(), key=lambda kv: -kv[1])
N_UNKNOWN = int((meta_df["n_categories"] == 0).sum())

F3_COEF = {c["term"]: c for c in F3["coefficients"]}
TOP_DX = F4["top_diagnoses"]  # list of [name, n]
ALPHA = 0.05


def fmt_p(p):
    if p is None or (isinstance(p, float) and p != p):
        return "n/a"
    return f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}"


def fmt_num(x, nd=3):
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return f"{x:.{nd}f}"


styles = getSampleStyleSheet()
styles.add(ParagraphStyle("PaperTitle", parent=styles["Title"], fontSize=16, leading=20,
                           fontName="Helvetica-Bold", spaceAfter=6))
styles.add(ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=11, leading=14,
                           fontName="Helvetica-Oblique", textColor=colors.HexColor("#333333"),
                           spaceAfter=10))
styles.add(ParagraphStyle("Author", parent=styles["Normal"], fontSize=11, leading=14, spaceAfter=2))
styles.add(ParagraphStyle("Affil", parent=styles["Normal"], fontSize=9.5, leading=12,
                           textColor=colors.HexColor("#444444")))
styles.add(ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13, leading=16,
                           fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6))
styles.add(ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, leading=14,
                           fontName="Helvetica-BoldOblique", spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.1, leading=14,
                           alignment=TA_JUSTIFY, spaceAfter=7))
styles.add(ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8.6, leading=11.3,
                           alignment=TA_JUSTIFY, spaceAfter=12, textColor=colors.HexColor("#222222")))
styles.add(ParagraphStyle("Ref", parent=styles["Normal"], fontSize=9, leading=12.5, spaceAfter=4,
                           leftIndent=14, firstLineIndent=-14))
styles.add(ParagraphStyle("Kw", parent=styles["Normal"], fontSize=9.5, leading=13, spaceBefore=6))

story = []

# ===========================================================================
# Title page
# ===========================================================================
story.append(Paragraph(
    "The clinical role of the heartbeat-evoked potential across neurological "
    "and non-neurological diagnoses: a sleep-stage-resolved analysis in "
    f"{N_PATIENTS_TOTAL:,} patients",
    styles["PaperTitle"]))
story.append(Paragraph(
    "Diagnosis-associated modulation of cortical heartbeat-evoked potentials (HEP), assessed "
    "channel-averaged and per-electrode, across sleep stages, with mixed-effects modelling "
    "controlling for age, sex, heart rate, and cardiac field artifact contamination",
    styles["Subtitle"]))
story.append(Spacer(1, 6))
story.append(Paragraph("Nir Cafri<sup>1,3</sup>, Felix Benninger<sup>2,3</sup>, Pablo Blinder<sup>1,2</sup>", styles["Author"]))
story.append(Paragraph(
    "<sup>1</sup>Department of Neurobiology, School of Neurobiology, Biochemistry and Biophysics, "
    "George S. Wise Faculty of Life Sciences, Tel Aviv University, Tel Aviv, Israel<br/>"
    "<sup>2</sup>Sagol School of Neuroscience, Tel Aviv University, Tel Aviv, Israel<br/>"
    "<sup>3</sup>Department of Neurology, Rabin Medical Center, Beilinson Hospital and Tel-Aviv "
    "University, Petah Tikva, Israel<br/>"
    "Correspondence: nircafri@mail.tau.ac.il", styles["Affil"]))
story.append(Spacer(1, 10))
story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#888888")))
story.append(Spacer(1, 10))

# ---------------------------------------------------------------------
# Structured abstract
# ---------------------------------------------------------------------
kw_light = next(r for r in F2["kw_table"] if r["stage"] == "light_sleep")
kw_n3 = next(r for r in F2["kw_table"] if r["stage"] == "N3")
kw_rem = next(r for r in F2["kw_table"] if r["stage"] == "R")
dx_neuro = F3_COEF["C(neuro_group, Treatment('Unknown'))[T.Neurological]"]
dx_nonneuro = F3_COEF["C(neuro_group, Treatment('Unknown'))[T.Non-neurological]"]
cfa_term = F3_COEF["cfa_z"]

story.append(Paragraph("Abstract", styles["H1"]))
story.append(Paragraph(
    f"The heartbeat-evoked potential (HEP) -- a scalp-EEG deflection time-locked to the cardiac "
    f"R-peak that indexes cortical processing of cardiac afferent signals -- has been proposed as a "
    f"marker of interoceptive and autonomic-cortical coupling, but its behaviour across clinical "
    f"diagnostic categories, and whether it differs between neurological and non-neurological "
    f"disease, remains poorly characterised at scale. We analysed channel-averaged and per-electrode "
    f"HEP amplitude (0.15-0.5&nbsp;s post-R-peak) from {N_PATIENTS_TOTAL:,} patients undergoing "
    f"clinical polysomnography (Harvard Electroencephalography cohort), across three sleep stages "
    f"(light sleep, N3, REM) and {N_ELECTRODES} scalp electrodes, categorised into 15 non-exclusive "
    f"EHR-derived diagnosis groups and split into neurological (n={BROAD_COUNTS.get('Neurological', 0) + BROAD_COUNTS.get('Both', 0):,} "
    f"with &ge;1 neurological diagnosis) vs. non-neurological (n={BROAD_COUNTS.get('Non-neurological', 0):,}) "
    f"vs. an undiagnosed reference cohort (n={N_UNKNOWN:,}). HEP amplitude differed significantly "
    f"across these three groups at every sleep stage (Kruskal-Wallis, light sleep "
    f"H={kw_light['H_stat']:.1f}, p={fmt_p(kw_light['p_value'])}; N3 H={kw_n3['H_stat']:.1f}, "
    f"p={fmt_p(kw_n3['p_value'])}; REM H={kw_rem['H_stat']:.1f}, p={fmt_p(kw_rem['p_value'])}). "
    f"A linear mixed-effects model of amplitude (fixed effects: diagnosis category, sleep stage, "
    f"scalp region, their interaction, age, sex, heart rate, and cardiac-field-artifact contamination; "
    f"random intercept per patient; N={F3['n_obs']:,} observations, {F3['n_patients']:,} patients) "
    f"confirmed an independent effect of neurological diagnosis "
    f"(&beta;={fmt_num(dx_neuro['estimate'])}&nbsp;&mu;V, 95% CI [{fmt_num(dx_neuro['ci_lo'])}, "
    f"{fmt_num(dx_neuro['ci_hi'])}], p={fmt_p(dx_neuro['p_value'])}) after adjustment for cardiac "
    f"field artifact (&beta;={fmt_num(cfa_term['estimate'])}&nbsp;&mu;V per SD, p={fmt_p(cfa_term['p_value'])}), "
    f"heart rate, age, and sex. Grand-average waveform analysis of the "
    f"{len(TOP_DX)} largest diagnostic categories against the reference cohort, using cluster-based "
    f"permutation testing with jittered null distributions, localised diagnosis-associated deviations "
    f"largely within the classic 150-500&nbsp;ms HEP window. These results indicate that the HEP "
    f"carries diagnosis- and disease-category-relevant information beyond what is explained by heart "
    f"rate and cardiac-field-artifact contamination alone, motivating its further evaluation as a "
    f"non-invasive marker of altered brain-heart coupling in clinical populations.",
    styles["Body"]))

story.append(Paragraph("Keywords: heartbeat-evoked potential, interoception, sleep stage, EEG, "
                        "clinical diagnosis, mixed-effects model, cardiac field artifact", styles["Kw"]))
story.append(PageBreak())

# ===========================================================================
# Introduction
# ===========================================================================
story.append(Paragraph("Introduction", styles["H1"]))
story.append(Paragraph(
    "The brain continuously processes signals arising from the heartbeat. This cardiac afferent "
    "processing is measurable at the scalp as the heartbeat-evoked potential (HEP), a slow cortical "
    "deflection time-locked to the electrocardiographic R-peak that emerges after the cardiac field "
    "artifact has subsided, typically studied in a window of roughly 150-500&nbsp;ms post-R-peak "
    "(Schandry, 1981; Montoya et al., 1993). HEP amplitude has been linked to interoceptive accuracy, "
    "bodily self-consciousness, and attentional and affective state (Park &amp; Blanke, 2019; "
    "Babo-Rebelo et al., 2016; Critchley &amp; Garfinkel, 2017), and modulates with sleep stage and "
    "arousal, consistent with a role for the HEP as a window onto ongoing brain-heart coupling rather "
    "than a fixed anatomical response.", styles["Body"]))
story.append(Paragraph(
    "Because brain-heart coupling is plausibly altered by disease that affects either the central "
    "nervous system (e.g. stroke, neurodegeneration) or peripheral/autonomic physiology (e.g. "
    "cardiovascular and metabolic disease, sleep apnea), the HEP has been proposed as a candidate "
    "biomarker of altered interoceptive or autonomic-cortical function across a range of clinical "
    "conditions. However, most prior HEP work has been conducted in small, single-diagnosis cohorts, "
    "leaving open whether HEP differences are specific to individual diseases, shared broadly across "
    "neurological conditions, or better explained by nuisance covariates such as heart rate or "
    "residual cardiac-field-artifact (CFA) contamination of the EEG signal -- a well-known confound "
    "given the volume conduction of the heart's electrical field into scalp electrodes near the "
    "R-peak.", styles["Body"]))
story.append(Paragraph(
    "Here we leverage a large, clinically heterogeneous polysomnography cohort with EHR-derived "
    "diagnosis annotations to ask, at scale: (1) does HEP amplitude differ between patients with a "
    "neurological diagnosis, patients with only non-neurological diagnoses, and an undiagnosed "
    "reference cohort; (2) does this difference survive adjustment for sleep stage, scalp region, age, "
    "sex, heart rate, and CFA contamination in a single mixed-effects model; and (3) which of the "
    "largest individual diagnostic categories show grand-average waveform deviations from the "
    "reference cohort, and in which time windows. We address all three questions on the same "
    "channel-averaged and per-electrode HEP amplitude metric, computed identically across sleep "
    "stages, diagnosis categories, and analyses.", styles["Body"]))

# ===========================================================================
# Results
# ===========================================================================
story.append(Paragraph("Results", styles["H1"]))

story.append(Paragraph("Cohort and diagnosis categories", styles["H2"]))
cat_rows_txt = "; ".join(f"{name} (n={n:,})" for name, n in CAT_TABLE)
story.append(Paragraph(
    f"From the Harvard Electroencephalography sleep cohort we extracted single-patient, "
    f"channel-averaged and per-electrode HEP traces for {N_PATIENTS_TOTAL:,} unique patients across "
    f"up to {len(STAGES)} sleep stages (light sleep, N3, REM), yielding {N_ROWS_LONG:,} "
    f"(patient &times; stage &times; electrode) observations across {N_ELECTRODES} scalp electrodes "
    f"(Fig.&nbsp;1). Patients were assigned to 15 non-exclusive EHR-derived diagnosis categories by "
    f"keyword-matched ICD-10/diagnosis-name search (Methods): {cat_rows_txt}. Because categories are "
    f"non-exclusive, {int(meta_df.loc[meta_df['n_categories'] > 1, 'patient_id'].nunique()):,} "
    f"patients carried more than one diagnosis category; {N_UNKNOWN:,} patients had no matched "
    f"diagnosis category and formed the undiagnosed reference ('Unknown') cohort. For the primary "
    f"neurological-vs-non-neurological analyses, patients with &ge;1 neurological category "
    f"(Cognitive Impairment/Dementia or Stroke/Cerebrovascular) were classed Neurological regardless "
    f"of comorbid non-neurological categories; patients with &ge;1 non-neurological category and zero "
    f"neurological categories were classed Non-neurological.", styles["Body"]))

# Fig 1
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "fig1_overview.png"), width=6.4 * inch, height=4.8 * inch),
    Paragraph(
        "<b>Figure 1. Single-patient HEP traces and grand average, by electrode and sleep stage.</b> "
        "Rows: three representative scalp electrodes (F3 frontal, C3 central, O1 occipital), each "
        "present in essentially the entire sparse 6-electrode montage that dominates this cohort "
        "(&ge;97% of recordings). Columns: sleep stage. Thin, low-alpha lines: a random subsample of "
        "60 patients' single-channel HEP traces per panel (seed 42); bold line: grand average across "
        "all available patients for that electrode/stage (n given per panel, deduplicated to one "
        "recording per patient); shaded band: &plusmn;1 SEM. Grey vertical band: the "
        "&plusmn;50&nbsp;ms QRS-complex window excluded from all cluster-permutation statistics "
        "(cardiac field artifact). A time-locked deflection is visible in the grand average of every "
        "panel, consistent with a genuine population-level HEP.",
        styles["Caption"]),
]))

story.append(Paragraph("HEP amplitude differs by diagnosis category and sleep stage", styles["H2"]))
n_by_grp = F2["n_by_group_stage"]
story.append(Paragraph(
    f"Using the channel-averaged HEP amplitude in the standard 150-500&nbsp;ms window "
    f"(Methods), we compared Neurological (light sleep n={n_by_grp['Neurological|light_sleep']:,}, "
    f"N3 n={n_by_grp['Neurological|N3']:,}, REM n={n_by_grp['Neurological|R']:,}), Non-neurological "
    f"(light sleep n={n_by_grp['Non-neurological|light_sleep']:,}, N3 n={n_by_grp['Non-neurological|N3']:,}, "
    f"REM n={n_by_grp['Non-neurological|R']:,}), and Unknown-reference (light sleep "
    f"n={n_by_grp['Unknown|light_sleep']:,}, N3 n={n_by_grp['Unknown|N3']:,}, REM n={n_by_grp['Unknown|R']:,}) "
    f"patients at each sleep stage (Fig.&nbsp;2). Amplitude distributions differed significantly "
    f"across the three groups at every stage (Kruskal-Wallis; light sleep H={kw_light['H_stat']:.1f}, "
    f"p={fmt_p(kw_light['p_value'])}, q={fmt_p(kw_light['q_value'])}; N3 H={kw_n3['H_stat']:.1f}, "
    f"p={fmt_p(kw_n3['p_value'])}, q={fmt_p(kw_n3['q_value'])}; REM H={kw_rem['H_stat']:.1f}, "
    f"p={fmt_p(kw_rem['p_value'])}, q={fmt_p(kw_rem['q_value'])}; Benjamini-Hochberg across stages). "
    f"A sensitivity analysis restricted to patients with exactly one diagnosis category (removing "
    f"comorbidity mixing) reproduced this pattern at every stage "
    f"(all q&lt;{max(r['q_value'] for r in F2['kw_table_single_category']):.1e}), indicating the "
    f"group difference is not solely an artifact of multi-morbid patients being pooled into both "
    f"groups.", styles["Body"]))

story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "fig2_distribution.png"), width=6.4 * inch, height=2.6 * inch),
    Paragraph(
        "<b>Figure 2. Population distribution of HEP amplitude by diagnosis category and sleep "
        "stage.</b> Violin + box plots of channel-averaged HEP amplitude (0.15-0.5&nbsp;s window) "
        "for the Unknown reference cohort (grey), Non-neurological (blue), and Neurological (orange) "
        "groups, one panel per sleep stage (<b>A</b> Light Sleep, <b>B</b> N3, <b>C</b> REM). Box: "
        "median and IQR; whiskers omitted for legibility (outliers not shown, full range in violin). "
        "P-values: Kruskal-Wallis test across the three groups, this panel's stage only.",
        styles["Caption"]),
]))

story.append(Paragraph("A mixed-effects model isolates an independent diagnosis effect", styles["H2"]))
region_terms = {k: v for k, v in F3_COEF.items() if k.startswith("C(region")}
stage_terms = {k: v for k, v in F3_COEF.items() if k.startswith("C(stage")}
hr_term = F3_COEF["hr_z"]
sex_term = F3_COEF["sex_bin"]
age_term = F3_COEF["age_z"]
story.append(Paragraph(
    f"To test whether the neuro/non-neuro amplitude difference survives adjustment for sleep stage, "
    f"scalp topography, and physiological/artifact covariates, we fit a linear mixed-effects model "
    f"(restricted maximum likelihood; statsmodels MixedLM) of HEP amplitude with fixed effects for "
    f"diagnosis group (Neurological / Non-neurological / Unknown reference), sleep stage, scalp "
    f"region (5 levels: frontal, central, parietal, temporal, occipital -- electrodes binned into "
    f"regions to keep the fixed-effect count legible, Methods), the diagnosis&times;stage interaction, "
    f"age, sex, heart rate, and CFA contamination, with a random intercept per patient "
    f"(N={F3['n_obs']:,} observations, {F3['n_patients']:,} patients; log-likelihood="
    f"{F3['log_likelihood']:.0f}, AIC={F3['aic']:.0f}; approximate marginal "
    f"R&sup2;={F3['marginal_r2_approx']:.3f}, conditional R&sup2;={F3['conditional_r2_approx']:.3f}; "
    f"Fig.&nbsp;3). The neurological diagnosis effect remained significant after this adjustment "
    f"(&beta;={fmt_num(dx_neuro['estimate'])}&nbsp;&mu;V relative to the reference cohort, 95% CI "
    f"[{fmt_num(dx_neuro['ci_lo'])}, {fmt_num(dx_neuro['ci_hi'])}], p={fmt_p(dx_neuro['p_value'])}), "
    f"whereas the non-neurological effect was smaller and not significant at &alpha;=0.05 "
    f"(&beta;={fmt_num(dx_nonneuro['estimate'])}&nbsp;&mu;V, 95% CI [{fmt_num(dx_nonneuro['ci_lo'])}, "
    f"{fmt_num(dx_nonneuro['ci_hi'])}], p={fmt_p(dx_nonneuro['p_value'])}). CFA contamination was the "
    f"single strongest predictor in the model (&beta;={fmt_num(cfa_term['estimate'])}&nbsp;&mu;V per "
    f"SD, p={fmt_p(cfa_term['p_value'])}), confirming that residual cardiac-field-artifact leakage "
    f"is an important nuisance source in HEP amplitude and underscoring the need to control for it "
    f"explicitly rather than assume the post-QRS window is artifact-free. Heart rate "
    f"(&beta;={fmt_num(hr_term['estimate'])}&nbsp;&mu;V per SD, p={fmt_p(hr_term['p_value'])}) and "
    f"sex (&beta;={fmt_num(sex_term['estimate'])}&nbsp;&mu;V for male vs. female, "
    f"p={fmt_p(sex_term['p_value'])}) were also independently associated with amplitude, while age "
    f"was not (&beta;={fmt_num(age_term['estimate'])}&nbsp;&mu;V per SD, p={fmt_p(age_term['p_value'])}). "
    f"Sleep stage effects replicated the pattern visible in Fig.&nbsp;1 (N3 lower than light sleep, "
    f"REM higher than light sleep), and scalp region effects showed occipital and parietal electrodes "
    f"with the lowest amplitude relative to the central reference level. None of the "
    f"diagnosis&times;stage interaction terms reached significance, indicating the diagnosis effect "
    f"on amplitude did not differ detectably across sleep stages in this model. Full per-electrode "
    f"(rather than per-region) descriptive statistics are provided in Supplementary Table S1 "
    f"(Paper1/fig3_electrode_supplementary_table.csv).", styles["Body"]))

story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "fig3_mixedmodel.png"), width=6.4 * inch, height=6.0 * inch),
    Paragraph(
        "<b>Figure 3. Mixed-effects model of HEP amplitude.</b> Forest plot of fixed-effect "
        "estimates &plusmn;95% CI from the linear mixed-effects model described in the text "
        f"(N={F3['n_obs']:,} observations, {F3['n_patients']:,} patients; random intercept per "
        "patient), ranked by absolute effect magnitude. Colour indicates term family: Diagnosis "
        "(orange), Diagnosis&times;Stage interaction (pink), Stage (blue), Region (green), other "
        "covariates -- age, sex, heart rate, CFA contamination (grey). Reference levels: diagnosis "
        "= Unknown, stage = Light Sleep, region = central. Estimates are in &mu;V (heart rate, age, "
        "CFA contamination z-scored; sex coded Male=1).",
        styles["Caption"]),
]))

story.append(Paragraph("Diagnosis-specific grand-average waveforms", styles["H2"]))
top_dx_txt = "; ".join(f"{name} (n={n:,})" for name, n in TOP_DX)
n_diag_windows = sum(
    1 for g in F4["waveform_results"] if g != "Unknown"
    for s in F4["waveform_results"][g] if F4["waveform_results"][g][s].get("cluster_windows")
)
story.append(Paragraph(
    f"We recomputed observed N per diagnosis category (rather than assuming any fixed shortlist) and "
    f"selected the {len(TOP_DX)} largest by patient count for waveform analysis: {top_dx_txt}. For "
    f"each category and sleep stage we computed the grand-average, channel-averaged HEP waveform "
    f"and compared it, using the cluster-based permutation test with pynapple-jittered null "
    f"distributions and Fisher-combined per-patient significance (6_hep_group_comparison."
    f"permutation_cluster_jitter_test; 100 permutations; &plusmn;50&nbsp;ms QRS window excluded), "
    f"against the same test applied independently to the Unknown reference cohort at that stage "
    f"(Fig.&nbsp;4; Methods). Windows significant in a diagnostic group but not in the reference "
    f"cohort are marked as candidate diagnosis-associated deviations. This procedure identified "
    f"{n_diag_windows} (diagnosis, stage) combinations with at least one such candidate window, "
    f"concentrated within or adjacent to the canonical 150-500&nbsp;ms HEP window, consistent with "
    f"disease-associated modulation of the same cortical response rather than a broadly different "
    f"waveform shape.", styles["Body"]))

story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "fig4_diagnosis_waveforms.png"), width=6.0 * inch, height=6.6 * inch),
    Paragraph(
        "<b>Figure 4. Grand-average HEP waveforms by diagnostic category, vs. undiagnosed reference "
        "cohort.</b> One panel per sleep stage (<b>A</b> Light Sleep, <b>B</b> N3, <b>C</b> REM). "
        "Black trace: Unknown reference cohort grand average &plusmn;SEM. Coloured traces: grand "
        f"average &plusmn;SEM for each of the {len(TOP_DX)} largest diagnostic categories (legend, "
        "panel A). Grey vertical band: &plusmn;50&nbsp;ms QRS window excluded from cluster "
        "statistics. Coloured bars beneath each trace: time windows where "
        "permutation_cluster_jitter_test found a significant cluster (p&lt;0.05) for that diagnosis "
        "group but not for the reference cohort at the same stage (Methods).",
        styles["Caption"]),
]))

# ===========================================================================
# Discussion
# ===========================================================================
story.append(Paragraph("Discussion", styles["H1"]))
story.append(Paragraph(
    "In a clinically heterogeneous polysomnography cohort spanning thousands of patients, we found "
    "that HEP amplitude differs systematically between patients with a neurological diagnosis, "
    "patients with only non-neurological diagnoses, and an undiagnosed reference cohort, and that "
    "this difference for the neurological group persists after adjusting for sleep stage, scalp "
    "region, age, sex, heart rate, and -- critically -- residual cardiac field artifact contamination "
    "in a single mixed-effects model. This pattern is broadly consistent with the view that altered "
    "brain-heart coupling accompanies neurological disease, plausibly reflecting disrupted central "
    "autonomic or interoceptive processing pathways (Park &amp; Blanke, 2019; Critchley &amp; "
    "Garfinkel, 2017), though the present cross-sectional, EHR-diagnosis-derived design cannot "
    "establish causal direction or rule out that the neurological category is enriched for other, "
    "unmeasured confounds.", styles["Body"]))
story.append(Paragraph(
    "A central methodological finding is the size of the CFA-contamination effect: it was the single "
    "largest-magnitude fixed effect in the mixed model, larger than sleep stage, scalp region, or "
    "diagnosis. This reinforces that any clinical HEP analysis -- including this one -- must "
    "explicitly quantify and adjust for residual cardiac-field leakage into the post-QRS EEG window, "
    "rather than relying on the QRS-exclusion window alone to guarantee a 'clean' interoceptive "
    "signal. Our CFA covariate (cfa_r2_excl_qrs, the fraction of post-QRS-window EEG variance "
    "explained by the concurrent ECG trace) is one defensible operationalisation of this contamination "
    "and is described fully in Methods; readers using a different CFA metric may find different "
    "residual associations.", styles["Body"]))
story.append(Paragraph(
    "Several simplifications were necessary to make this analysis tractable at this scale, and are "
    "worth stating plainly rather than glossing over. First, the mixed-effects model represents "
    "diagnosis as the same 3-level Neurological/Non-neurological/Unknown grouping used for the "
    "population-distribution analysis (Fig.&nbsp;2), rather than as 15 non-exclusive binary diagnosis "
    "indicators; the latter would introduce substantial collinearity (most patients carry several "
    "comorbid categories) and would not fit legibly into a single coefficient plot. The grand-average "
    "waveform analysis (Fig.&nbsp;4) instead examines individual diagnostic categories directly, "
    "providing a complementary, disease-specific view. Second, the 24-electrode montage was collapsed "
    "into 5 scalp regions for the mixed model to keep the fixed-effect count interpretable; full "
    "per-electrode descriptive statistics are retained in a supplementary table. Third, the "
    "diagnosis-vs-reference significance marks in Fig.&nbsp;4 use the only comparison the existing "
    "cluster-permutation implementation supports -- an independent one-sample test (vs. zero) within "
    "each group -- rather than a genuine two-sample cluster statistic; a window flagged here means "
    "the diagnostic group shows a significant deflection from baseline that the reference cohort does "
    "not (or vice versa) at that stage, which is suggestive of a group difference but not a formal "
    "test of it. Future work extending the cluster-permutation framework to a true two-sample cluster "
    "test would sharpen this comparison.", styles["Body"]))
story.append(Paragraph(
    "Taken together, these results support the HEP as a physiologically meaningful, diagnosis-"
    "sensitive signal in a large clinical cohort, while highlighting cardiac field artifact as a "
    "confound that must be modelled explicitly, and motivate future work with two-sample cluster "
    "statistics, dense-electrode montages, and longitudinal or interventional designs to test whether "
    "HEP changes track disease progression or treatment response.", styles["Body"]))

# ===========================================================================
# Methods
# ===========================================================================
story.append(Paragraph("Methods", styles["H1"]))

story.append(Paragraph("Cohort, EHR diagnosis categorisation, and cohort split", styles["H2"]))
story.append(Paragraph(
    "All data are from the Harvard Electroencephalography clinical polysomnography cohort processed "
    "by this repository's existing HEP pipeline (6_hep_group_comparison.py; 16_diagnosis_sleep_stage_"
    "comparison_dashboard.py), reused here without modification via mod16.load_patient_data "
    "(min_eeg_channels=None, i.e. the full sparse+dense electrode pool) and the on-disk "
    "individuals_cache*.pkl produced by get_group_individuals. Diagnosis categories were derived "
    "from EHR diagnosis-name text (bdsp_i0006_diagnosis / icd10_codes parquet tables) via 15 "
    "non-exclusive, keyword-matched category rules (_DIAG_CATEGORIES in dashboard 16; identical "
    "rules were previously applied to build Paper CFA/demographics_combined.parquet, which we use "
    "directly as the primary diagnosis-category and age/sex source, with mod16.load_ehr_data / "
    "mod16._patient_demographics as a fallback for the minority of patients not present in that "
    "cache). A patient was classed Neurological if they carried &ge;1 of Cognitive "
    "Impairment/Dementia or Stroke/Cerebrovascular (regardless of comorbid non-neurological "
    "categories), Non-neurological if they carried &ge;1 other category and zero neurological "
    "categories, and Unknown (reference) if they carried zero categories at all "
    "(mod16.select_non_diagnosis_cohort logic, applied here directly on the category list).",
    styles["Body"]))

story.append(Paragraph("HEP epoching, preprocessing, and amplitude metric", styles["H2"]))
story.append(Paragraph(
    "HEP epochs were extracted by the repository's existing pipeline (6_hep_group_comparison."
    "process_file_data / get_group_individuals), not reimplemented here: ECG is cleaned and R-peaks "
    "detected with a robust peak detector; EEG is epoched relative to each detected R-peak over a "
    "window of -0.3 to +0.4&nbsp;s; epochs failing amplitude/flatline/roughness/spectral-power-ratio "
    "quality checks are excluded before averaging (HEP_WINDOW_* thresholds in 6_hep_group_comparison.py); "
    "per-patient, per-electrode average HEP traces are the (patient_id, hep_data, times, ch_names, "
    "rpeaks, ...) tuples returned by get_group_individuals and cached to individuals_cache*.pkl. From "
    "each such trace we additionally required it pass mod16._is_valid_hep_trace (reject flatline, "
    "extreme-amplitude, or high-roughness traces) before inclusion in any analysis here. The scalar "
    "HEP amplitude metric used throughout (Figs.&nbsp;2-3, and as the mixed-model outcome) is the "
    f"mean of the (baseline-uncorrected) trace over {AMP_WINDOW[0]}-{AMP_WINDOW[1]}&nbsp;s post-R-peak "
    "(AMP_WINDOW, identical to Paper1/build_diagnosis_alignment_analysis.py), computed per electrode "
    "for the per-electrode long-format table and, separately, on the channel-averaged trace "
    "(mod16._patient_hep_trace over the montage-channel union) for the channel-averaged metric used "
    "in Figs.&nbsp;2-3. Multiple recordings for the same canonical patient ID "
    "(mod16._canonical_patient_id, which collapses stage/session suffixes) within the same sleep "
    "stage were deduplicated to one recording per patient per stage.", styles["Body"]))

story.append(Paragraph("Covariates: age, sex, heart rate, cardiac field artifact", styles["H2"]))
story.append(Paragraph(
    "Age and sex were taken from Paper CFA/demographics_combined.parquet where available, falling "
    "back to mod16._patient_demographics (COBRAD clinical sheet + EHR demographics cache) otherwise. "
    "Heart rate (mean ECG beats-per-minute, qc_ecg_bpm) and cardiac field artifact magnitude "
    "(cfa_r2_excl_qrs: the fraction of post-QRS-window EEG variance explained by the concurrent ECG "
    "trace, i.e. residual cardiac-field leakage into the analysis window after excluding the QRS "
    "complex itself) were taken from Paper CFA/cfa_combined.parquet, averaged per patient across all "
    "available recordings/channels/windows in that table (this table does not cover every HEP patient "
    "in the sparse-montage pool; the mixed-effects model is fit on the subset with complete "
    "age/sex/HR/CFA/region/stage/diagnosis data, N given in Results).", styles["Body"]))

story.append(Paragraph("Cluster-based permutation testing", styles["H2"]))
story.append(Paragraph(
    "Group-level HEP significance was assessed with the repository's existing "
    "permutation_cluster_jitter_test (6_hep_group_comparison.py), used without modification: for a "
    "matrix of one channel-averaged trace per patient, the function computes a per-patient cluster "
    "permutation p-value (against pynapple-jittered circular-shift null traces), Fisher-combines "
    "these into a group-level p-value, and separately performs a group-level cluster-mass permutation "
    "test on the one-sample t-statistic (vs. zero) with the &plusmn;50&nbsp;ms window around the "
    "R-peak excluded from cluster detection (cardiac field artifact dominates this window). We ran "
    "this test independently, per sleep stage, on each diagnostic group's patient-average-HEP matrix "
    "and on the Unknown reference cohort's matrix (100 permutations, 0.1&nbsp;s jitter). Because this "
    "function performs a one-sample (vs. zero) test rather than a two-sample group-difference test, "
    "we do not have a native two-sample cluster statistic available in this codebase; the "
    "diagnosis-associated windows shown in Fig.&nbsp;4 are windows significant in the diagnostic "
    "group's one-sample test but not in the reference cohort's one-sample test at the same stage, "
    "which we report as a candidate/suggestive marker rather than a formal group-difference test "
    "(see Discussion for this limitation).", styles["Body"]))

story.append(Paragraph("Mixed-effects model", styles["H2"]))
story.append(Paragraph(
    "The mixed-effects model (Fig.&nbsp;3) was fit with statsmodels.formula.api.mixedlm, REML, on "
    "the per-(patient, stage, electrode) long-format table restricted to rows with non-missing age, "
    "sex, heart rate, CFA, region, stage, and diagnosis group: hep_amplitude_uv ~ "
    "C(neuro_group, Treatment('Unknown')) * C(stage, Treatment('light_sleep')) + "
    "C(region, Treatment('central')) + age_z + sex_bin + hr_z + cfa_z, with a random intercept per "
    "patient (groups=patient_id). Age, heart rate, and CFA were z-scored; sex was coded Male=1, "
    "Female=0. Electrodes were mapped to 5 canonical scalp regions (frontal: Fp1/Fp2/F7/F3/Fz/F4/F8; "
    "central: C3/Cz/C4; parietal: P3/Pz/P4; temporal: T3/T7/T4/T8/T5/P7/T6/P8; occipital: O1/Oz/O2) to "
    "keep the electrode fixed effect legible in one coefficient plot, per the analysis plan; full "
    "per-electrode descriptive statistics are exported separately (Paper1/"
    "fig3_electrode_supplementary_table.csv). Diagnosis was represented as the same 3-level "
    "Neurological/Non-neurological/Unknown grouping used in Fig.&nbsp;2, rather than 15 non-exclusive "
    "binary indicators, for the collinearity/interpretability reasons discussed above. We found that "
    "statsmodels' default lbfgs optimizer converged MixedLM to a degenerate boundary solution on this "
    "dataset (random-intercept variance forced to exactly zero, infinite log-likelihood, and grossly "
    "inflated covariate standard errors) -- confirmed on a 1,500-patient subsample where lbfgs alone "
    "diverged from bfgs/cg/powell, which agreed to 5 decimal places; we therefore fit the full model "
    "with the bfgs optimizer (cg as an automatic fallback if bfgs failed to converge). Approximate "
    "marginal/conditional R&sup2; were computed as variance-of-fitted-fixed-effects / total variance, "
    "and (variance-of-fitted-fixed-effects + random-intercept variance) / total variance respectively, "
    "with total variance = fixed-effect variance + random-intercept variance + residual variance -- an "
    "approximation of the Nakagawa &amp; Schielzeth pseudo-R&sup2; rather than an exact computation.",
    styles["Body"]))

story.append(Paragraph("Statistics summary", styles["H2"]))
story.append(Paragraph(
    "Group comparisons of the population HEP-amplitude distribution (Fig.&nbsp;2) used the "
    "Kruskal-Wallis H test across the three diagnosis groups, per sleep stage, with Benjamini-Hochberg "
    "FDR correction across stages (mod16.benjamini_hochberg). All p-values reported are two-sided. "
    "Analyses were run with the project's existing venv (Python 3.11; statsmodels, scipy, "
    "matplotlib, reportlab); no raw EDF/EEG data were reprocessed for this manuscript beyond what the "
    "existing pipeline scripts already cache on disk.", styles["Body"]))

# ===========================================================================
# References
# ===========================================================================
story.append(Paragraph("References", styles["H1"]))
refs = [
    "1. Schandry R. Heart beat perception and emotional experience. Psychophysiology. 1981;18(4):483-488.",
    "2. Montoya P, Schandry R, Müller A. Heartbeat evoked potentials (HEP): topography and "
    "influence of cardiac awareness and focus of attention. Electroencephalography and Clinical "
    "Neurophysiology. 1993;88(3):163-172.",
    "3. Park H-D, Blanke O. Heartbeat-evoked cortical responses: state of the art and future "
    "directions. NeuroImage. 2019;197:502-511.",
    "4. Babo-Rebelo M, Richter CG, Tallon-Baudry C. Neural responses to heartbeats in the default "
    "network encode the self in spontaneous thoughts. Journal of Neuroscience. 2016;36(30):7829-7840.",
    "5. Critchley HD, Garfinkel SN. Interoception and emotion. Current Opinion in Psychology. "
    "2017;17:7-14.",
]
for r in refs:
    story.append(Paragraph(r, styles["Ref"]))

doc = SimpleDocTemplate(
    OUT_PDF, pagesize=LETTER,
    leftMargin=0.85 * inch, rightMargin=0.85 * inch,
    topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    title="The clinical role of the heartbeat-evoked potential across diagnoses",
    author="Nir Cafri",
)
doc.build(story)
print(f"Saved {OUT_PDF}")
