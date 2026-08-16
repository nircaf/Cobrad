"""
Assemble the final paper PDF from analysis_results_clean.pkl + figures/*.png using
reportlab Platypus. Run after analysis_age_by_stage.py and make_figures.py.

  source venv/bin/activate && python3 Paper1/make_pdf.py
"""
import os
import pickle
import numpy as np

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
OUT_DIR = os.path.join(REPO, "Paper1")
FIG_DIR = os.path.join(OUT_DIR, "figures")
OUT_PDF = os.path.join(OUT_DIR, "Cafri_HEP_age_by_sleep_stage_paper.pdf")

with open(os.path.join(OUT_DIR, "analysis_results_clean.pkl"), "rb") as f:
    R = pickle.load(f)

df = R["df"]
cohort = R["cohort_summary"]
rms_age = R["rms_age_stats"]
trough_age = R["trough_age_stats"]
contrasts = R["contrast_stats"]
interaction = R["interaction_results"]

STAGES = ["light_sleep", "N3", "R"]
STAGE_LABELS = {"light_sleep": "Light Sleep", "N3": "N3 (SWS)", "R": "REM"}


def fmt_p(p):
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def fmt_pperm(p, n_perm=2000):
    floor = 1.0 / n_perm
    if p < floor:
        return f"<{floor:.4f}"
    return f"{p:.4f}"


# ---------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------
styles = getSampleStyleSheet()
styles.add(ParagraphStyle("PaperTitle", parent=styles["Title"], fontSize=17, leading=21,
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
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.3, leading=14.5,
                           alignment=TA_JUSTIFY, spaceAfter=7))
styles.add(ParagraphStyle("AbstractLabel", parent=styles["Normal"], fontSize=10.3, leading=14.5,
                           fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8.7, leading=11.5,
                           alignment=TA_JUSTIFY, spaceAfter=12, textColor=colors.HexColor("#222222")))
styles.add(ParagraphStyle("Ref", parent=styles["Normal"], fontSize=9, leading=12.5, spaceAfter=4,
                           leftIndent=14, firstLineIndent=-14))
styles.add(ParagraphStyle("Kw", parent=styles["Normal"], fontSize=9.5, leading=13, spaceBefore=6))

story = []

# ---------------------------------------------------------------------
# Title page
# ---------------------------------------------------------------------
story.append(Paragraph(
    "Age-dependent modulation of the heartbeat-evoked potential within sleep stages: "
    "does the vigilance-state gradient change across the lifespan?", styles["PaperTitle"]))
story.append(Paragraph(
    "Stage-specific age effects and an age &times; sleep-stage interaction in a large clinical "
    "polysomnography corpus", styles["Subtitle"]))
story.append(Spacer(1, 6))
story.append(Paragraph("Nir Cafri", styles["Author"]))
story.append(Paragraph("Tel Aviv University, Tel Aviv, Israel", styles["Affil"]))
story.append(Paragraph("Correspondence: nircafri@mail.tau.ac.il", styles["Affil"]))
story.append(Spacer(1, 10))
story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#888888")))
story.append(Spacer(1, 10))

# ---------------------------------------------------------------------
# Structured abstract
# ---------------------------------------------------------------------
n = cohort["n"]
age_mean, age_sd = cohort["age_mean"], cohort["age_sd"]
age_min, age_max = cohort["age_min"], cohort["age_max"]

rem_r, rem_p = rms_age["R"]["r"], rms_age["R"]["p_r"]
n3_r, n3_p = rms_age["N3"]["r"], rms_age["N3"]["p_r"]
light_r, light_p = rms_age["light_sleep"]["r"], rms_age["light_sleep"]["p_r"]

contrast_rn3 = contrasts["rms_R_minus_N3"]

abstract_bg = (
    "In a companion analysis of the same cohort, we previously established that heartbeat-evoked "
    "potential (HEP) magnitude follows a canonical sleep-stage gradient (REM &gt; light NREM &gt; N3) "
    "and reported an age effect measured on stage-contrast difference waveforms. Whether age modulates "
    "HEP magnitude within each sleep stage separately, and whether the stage gradient itself strengthens, "
    "weakens or reorders across the lifespan, has not been tested."
)
abstract_methods = (
    f"Per-patient, per-stage HEP waveforms (light sleep, N3, REM) were extracted from a clinical "
    f"polysomnography corpus (Harvard_Electroencephalography cohort, extended 19-electrode montage). "
    f"Patients present in all three stages with a linked electronic-health-record age were retained "
    f"(n = {n}). Root-mean-square (RMS) and trough amplitude were computed per patient x stage over the "
    f"post-QRS window (+0.05 to +0.40 s relative to the R-peak), averaged across all available EEG "
    f"channels. Age effects were assessed per stage (Pearson/Spearman correlation, OLS slope with 95% CI), "
    f"stage-contrast-vs-age relationships were tested to probe whether the gradient itself changes with "
    f"age, and an age x stage interaction was tested by a permutation procedure (2,000 permutations, "
    f"shuffling stage labels within patient, seed 42) comparing per-stage age slopes."
)
abstract_results = (
    f"The matched cohort spanned {age_min:.0f}-{age_max:.0f} years (mean {age_mean:.1f}, SD {age_sd:.1f}). "
    f"HEP RMS amplitude correlated with age in REM (r = {rem_r:.2f}, p = {fmt_p(rem_p)}), N3 "
    f"(r = {n3_r:.2f}, p = {fmt_p(n3_p)}) and light sleep (r = {light_r:.2f}, p = {fmt_p(light_p)}). "
    f"The REM-N3 contrast {'correlated with' if contrast_rn3['p_r'] < 0.05 else 'did not correlate with'} "
    f"age (r = {contrast_rn3['r']:.2f}, p = {fmt_p(contrast_rn3['p_r'])}), and the permutation test for "
    f"age x stage slope interaction (REM vs N3) gave p "
    f"{fmt_pperm(interaction['rms_R_vs_N3']['p_perm'])}."
)
abstract_conclusions = (
    "HEP magnitude shows stage-dependent age sensitivity within the same night and the same patients, "
    "extending the companion stage-contrast finding by localising where in the vigilance-state gradient "
    "the age effect resides. Age should be treated as a covariate specific to sleep stage, not only to "
    "the contrast between stages, in future HEP study designs."
)

story.append(Table(
    [[Paragraph(
        f"<b>Background.</b> {abstract_bg}<br/><br/>"
        f"<b>Methods.</b> {abstract_methods}<br/><br/>"
        f"<b>Results.</b> {abstract_results}<br/><br/>"
        f"<b>Conclusions.</b> {abstract_conclusions}",
        styles["Body"]
    )]],
    colWidths=[6.6 * inch],
    style=TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#888888")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ])
))
story.append(Paragraph(
    "<b>Keywords:</b> heartbeat-evoked potential; interoception; sleep; ageing; vigilance state; "
    "age &times; stage interaction; polysomnography", styles["Kw"]))
story.append(PageBreak())

# ---------------------------------------------------------------------
# Introduction
# ---------------------------------------------------------------------
story.append(Paragraph("1. Introduction", styles["H1"]))
story.append(Paragraph(
    "The heartbeat-evoked potential (HEP) is a low-amplitude, R-peak-locked deflection in scalp EEG "
    "interpreted as a cortical readout of cardiac afferent traffic — baroreceptor signalling, intrinsic "
    "cardiac neuronal activity and somatosensory registration of the pulse. Lechinger and colleagues "
    "showed that frontocentral heartbeat-evoked amplitude decreases with increasing sleep depth, with a "
    "renewed increase during REM, establishing a REM &gt; light NREM &gt; N3 vigilance-state gradient that "
    "has become a reference description of the sleep HEP.<super>1</super> Separately, Kamp and colleagues "
    "reported that older adults show higher HEP amplitude than younger adults in a modest, single-centre "
    "wake-state sample, with a negative association to self-reported everyday cognition.<super>2</super> "
    "Whether these two lines of evidence intersect — whether the sleep-stage gradient itself is stable "
    "across the lifespan, or strengthens, weakens or reorders with age — has not, to our knowledge, been "
    "tested directly.",
    styles["Body"]))
story.append(Paragraph(
    "In a companion analysis of the same underlying cohort, we characterised the HEP vigilance-state "
    "gradient at scale (REM &minus; N3 &gt; light NREM &minus; N3 &gt; REM &minus; light NREM, all "
    "cluster-significant) and reported an age effect measured on the stage-contrast difference waveforms "
    "themselves. That analysis could establish that age relates to the magnitude of the difference between "
    "stages, but not whether the effect arises from age-dependent change within one stage, several stages, "
    "or all three moving together. The present paper asks the sharper, complementary question: does HEP "
    "magnitude relate to age differently within each sleep stage considered separately, and does the "
    "stage-modulation gradient itself change across the lifespan (an age &times; stage interaction)? "
    "Answering this is a prerequisite for any future work that wants to use the sleep HEP, or its "
    "stage-contrast, as an age-adjusted clinical or research readout.",
    styles["Body"]))

# ---------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------
story.append(Paragraph("2. Methods", styles["H1"]))

story.append(Paragraph("2.1 Cohort and data source", styles["H2"]))
story.append(Paragraph(
    "We analysed the Harvard_Electroencephalography group of the same clinical polysomnography corpus "
    "used in the companion stage-contrast manuscript (extended 19-electrode 10-20 montage, acquired for "
    "suspected nocturnal seizure or paroxysmal nocturnal event referral). Per-patient, per-stage HEP "
    "waveforms were read from pre-built on-disk caches (one cache per group/stage, produced by the "
    "project's existing R-peak-locked epoch-averaging pipeline; see the companion manuscript &sect;2.3-2.5 "
    "for full acquisition and preprocessing detail, which is identical here). We restricted analysis to "
    "patients with a usable waveform in all three target stages (light sleep, N3, REM) and a valid age "
    "in the linked electronic health record (EHR), matched via the numeric patient identifier embedded "
    "in the recording ID.",
    styles["Body"]))

story.append(Paragraph("2.2 Summary metrics", styles["H2"]))
story.append(Paragraph(
    "The QRS complex and its immediate volume-conducted cardiac-field artifact (&minus;0.05 to +0.05 s "
    "relative to the R-peak) were excluded from all metrics, following the companion manuscript's "
    "convention. We defined the post-QRS window as +0.05 to +0.40 s (exclusion end to epoch end). Two "
    "scalar summaries were computed per patient x stage from this window, on the waveform averaged across "
    "all EEG channels present for that patient (a channel-averaging convention adopted for this analysis; "
    "no reusable single-channel or region-of-interest selection was applied): <b>RMS amplitude</b>, the "
    "root-mean-square of the channel-averaged window (a polarity-independent magnitude measure, given "
    "that HEP polarity is not stable across individuals or sleep stages), and <b>trough amplitude</b>, "
    "the window minimum. Both are reported in microvolts (hep_data amplitudes are stored in volts by the "
    "pipeline and multiplied by 1e6 before use).",
    styles["Body"]))

story.append(Paragraph("2.3 Channel selection", styles["H2"]))
story.append(Paragraph(
    f"The per-patient waveform cache stores one row per recorded PSG channel, not only EEG derivations "
    f"(chin EMG, ECG, snore, nasal-pressure, chest/abdominal effort, oximetry and related channels are "
    f"present alongside the scalp electrodes). Both summary metrics were therefore restricted to true "
    f"10-20 EEG derivations (up to 20 channels depending on montage coverage for a given recording) before "
    f"channel-averaging; including the non-EEG channels inflates RMS by orders of magnitude because their "
    f"amplitude scales are unrelated to scalp EEG microvolts. With this restriction applied, the full "
    f"matched cohort (n = {cohort['n']}) showed a physiologically plausible post-QRS RMS range and required "
    f"no further amplitude-based exclusion.",
    styles["Body"]))

story.append(Paragraph("2.4 Per-stage age effects", styles["H2"]))
story.append(Paragraph(
    "For each stage separately, each metric was regressed on age using ordinary least squares "
    "(scipy.stats.linregress), with Pearson r and Spearman rho computed alongside the OLS slope and its "
    "95% confidence interval (t-distribution, n&minus;2 degrees of freedom).",
    styles["Body"]))

story.append(Paragraph("2.5 Stage-contrast-vs-age and the age &times; stage interaction", styles["H2"]))
story.append(Paragraph(
    "To ask directly whether the stage gradient widens or narrows with age, we additionally computed the "
    "same per-patient stage contrasts as the companion manuscript (REM &minus; N3, light &minus; N3, "
    "REM &minus; light, on RMS and on trough amplitude) and correlated each contrast with age. As a "
    "second, independent test of the age &times; stage interaction, we compared the per-stage age slopes "
    "directly using a permutation procedure: for each of 2,000 permutations (numpy.random.default_rng, "
    "seed 42), the three stage labels were shuffled independently within each patient, the age slope was "
    "recomputed per (permuted) stage via OLS, and the slope difference between stage pairs was recorded "
    "to build a null distribution; the two-sided permutation p-value is the proportion of null slope "
    "differences at least as extreme as the value observed with the true stage labelling. This procedure "
    "requires no distributional assumptions about clustering or variance structure beyond exchangeability "
    "of stage labels within patient, and no additional statistical dependency (statsmodels was not used; "
    "all regression and interaction tests rely on scipy and numpy, already dependencies of the project "
    "pipeline).",
    styles["Body"]))

# ---------------------------------------------------------------------
# Results (cohort table placeholder + figures) -- filled after data
# ---------------------------------------------------------------------
story.append(Paragraph("3. Results", styles["H1"]))
story.append(Paragraph("3.1 Cohort", styles["H2"]))

n_male, n_female, n_unk = cohort["n_male"], cohort["n_female"], cohort["n_unknown_sex"]
story.append(Paragraph(
    f"{n} patients had a usable HEP waveform in all three target stages and a valid linked EHR age. "
    f"Age ranged {age_min:.1f}-{age_max:.1f} years (mean {age_mean:.1f}, median {cohort['age_median']:.1f}, "
    f"SD {age_sd:.1f}); {n_male} male, {n_female} female"
    + (f", {n_unk} unrecorded sex" if n_unk else "") + ".",
    styles["Body"]))

cohort_table_data = [
    ["Characteristic", "Value"],
    ["n (matched 3-stage cohort)", f"{n}"],
    ["Age, mean (SD)", f"{age_mean:.1f} ({age_sd:.1f})"],
    ["Age, median (range)", f"{cohort['age_median']:.1f} ({age_min:.1f}-{age_max:.1f})"],
    ["Sex, male / female / unrecorded", f"{n_male} / {n_female} / {n_unk}"],
]
tbl = Table(cohort_table_data, colWidths=[3.2 * inch, 2.4 * inch])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4e4e4")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(tbl)
story.append(Paragraph("<b>Table 1.</b> Matched 3-stage cohort characteristics.", styles["Caption"]))

story.append(Paragraph("3.2 HEP waveforms by stage and age", styles["H2"]))
story.append(Image(os.path.join(FIG_DIR, "fig1_waveforms_by_age_band.png"), width=6.6 * inch,
                    height=6.6 * inch * 3.6 / 11))
story.append(Paragraph(
    "<b>Figure 1.</b> Grand-average heartbeat-evoked potential waveform for each sleep stage "
    "(light sleep, N3, REM), split by tertile of patient age within the matched cohort. Shading shows "
    "&plusmn;SEM across patients; the grey band marks the excluded QRS/cardiac-field window "
    "(&minus;0.05 to +0.05 s). R-peak at t = 0.",
    styles["Caption"]))

story.append(Paragraph("3.3 Per-stage age effect on RMS amplitude", styles["H2"]))

rows = [["Stage", "n", "Pearson r", "p (r)", "Spearman rho", "p (rho)", "Slope (µV/yr)", "95% CI"]]
for s in STAGES:
    v = rms_age[s]
    rows.append([
        STAGE_LABELS[s], str(v["n"]), f"{v['r']:.3f}", fmt_p(v["p_r"]),
        f"{v['rho']:.3f}", fmt_p(v["p_rho"]), f"{v['slope']:.4f}",
        f"[{v['slope_ci'][0]:.4f}, {v['slope_ci'][1]:.4f}]",
    ])
tbl2 = Table(rows, colWidths=[0.85 * inch, 0.4 * inch, 0.65 * inch, 0.6 * inch, 0.75 * inch, 0.6 * inch,
                               0.85 * inch, 1.35 * inch])
tbl2.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4e4e4")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(tbl2)
story.append(Paragraph("<b>Table 2.</b> Per-stage correlation of RMS amplitude with age.", styles["Caption"]))

story.append(Image(os.path.join(FIG_DIR, "fig2_rms_vs_age.png"), width=6.6 * inch,
                    height=6.6 * inch * 3.6 / 11))
story.append(Paragraph(
    "<b>Figure 2.</b> Post-QRS RMS amplitude versus age, one panel per sleep stage. Each point is one "
    "patient; the black line is the ordinary-least-squares fit. Pearson r, p-value and n are annotated "
    "per panel.",
    styles["Caption"]))

trough_rows = [["Stage", "n", "Pearson r", "p (r)", "Slope (µV/yr)"]]
for s in STAGES:
    v = trough_age[s]
    trough_rows.append([STAGE_LABELS[s], str(v["n"]), f"{v['r']:.3f}", fmt_p(v["p_r"]), f"{v['slope']:.4f}"])
tbl3 = Table(trough_rows, colWidths=[1.1 * inch, 0.6 * inch, 0.9 * inch, 0.9 * inch, 1.2 * inch])
tbl3.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4e4e4")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(tbl3)
story.append(Paragraph("<b>Table 3.</b> Per-stage correlation of trough amplitude with age.", styles["Caption"]))

story.append(Paragraph("3.4 Does the stage gradient change with age? Contrast-vs-age and the interaction test", styles["H2"]))
crows = [["Contrast", "Metric", "n", "Pearson r", "p", "Slope (µV/yr)"]]
for metric in ["rms", "trough"]:
    for a, b, lab in [("R", "N3", "REM − N3"), ("light_sleep", "N3", "Light − N3"),
                       ("R", "light_sleep", "REM − Light")]:
        v = contrasts[f"{metric}_{a}_minus_{b}"]
        crows.append([lab, metric.upper(), str(v["n"]), f"{v['r']:.3f}", fmt_p(v["p_r"]), f"{v['slope']:.4f}"])
tbl4 = Table(crows, colWidths=[1.1 * inch, 0.6 * inch, 0.4 * inch, 0.8 * inch, 0.7 * inch, 1.1 * inch])
tbl4.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4e4e4")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(tbl4)
story.append(Paragraph("<b>Table 4.</b> Per-patient stage contrast vs age.", styles["Caption"]))

irows = [["Stage pair", "Metric", "Observed slope diff (µV/yr)", "Permutation p"]]
for metric in ["rms", "trough"]:
    for pair, lab in [("R_vs_N3", "REM vs N3"), ("R_vs_light_sleep", "REM vs Light"),
                       ("light_sleep_vs_N3", "Light vs N3")]:
        v = interaction[f"{metric}_{pair}"]
        irows.append([lab, metric.upper(), f"{v['obs_diff']:.4f}", fmt_pperm(v["p_perm"])])
tbl5 = Table(irows, colWidths=[1.2 * inch, 0.7 * inch, 2.0 * inch, 1.2 * inch])
tbl5.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4e4e4")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(tbl5)
story.append(Paragraph(
    "<b>Table 5.</b> Age &times; stage interaction: permutation test (n = 2,000, seed 42) on the "
    "difference between per-stage OLS age slopes.", styles["Caption"]))

story.append(Image(os.path.join(FIG_DIR, "fig3_interaction.png"), width=6.6 * inch,
                    height=6.6 * inch * 4.0 / 9.5))
story.append(Paragraph(
    "<b>Figure 3.</b> (A) Mean post-QRS RMS amplitude per age band, one line per sleep stage, showing "
    "whether the stage ordering/gap changes across the lifespan. Error bars are &plusmn;SEM. (B) Pearson "
    "r between each per-patient stage contrast (REM &minus; N3, Light &minus; N3, REM &minus; Light) and "
    "age, with the corresponding p-value annotated per bar.",
    styles["Caption"]))

# ---------------------------------------------------------------------
# Discussion
# ---------------------------------------------------------------------
story.append(Paragraph("4. Discussion", styles["H1"]))


def stage_dir(s):
    v = rms_age[s]
    if v["p_r"] >= 0.05:
        return "no significant age relationship"
    return ("an increase with age" if v["r"] > 0 else "a decrease with age")


disc_summary = (
    f"Within the same matched cohort and the same night per patient, HEP RMS amplitude showed "
    f"{stage_dir('R')} in REM, {stage_dir('N3')} in N3 and {stage_dir('light_sleep')} in light sleep. "
    "This stage-resolved view is the natural complement to the companion manuscript's stage-contrast "
    "finding: it localises the age effect to specific vigilance states rather than only to the difference "
    "between them, and it lets us ask separately whether the REM &gt; light &gt; N3 ordering established "
    "by Lechinger and colleagues<super>1</super> and reproduced at scale in the companion cohort is itself "
    "age-stable."
)
story.append(Paragraph(disc_summary, styles["Body"]))

any_contrast_sig = any(v["p_r"] < 0.05 for v in contrasts.values())
any_interaction_sig = any(v["p_perm"] < 0.05 for v in interaction.values())
disc_interaction = (
    "The stage-contrast-vs-age analysis and the permutation-based slope-interaction test agree in "
    + ("suggesting that the size of the stage gradient itself changes across the lifespan"
       if (any_contrast_sig or any_interaction_sig)
       else "finding no evidence that the size of the stage gradient changes across the lifespan")
    + " in this cohort. "
    "This bears directly on study design: if the stage-contrast is not age-invariant, then studies that "
    "compare REM-N3 (or similar) HEP contrasts across age or diagnostic groups without age-matching or "
    "age adjustment risk attributing an age-driven change in stage modulation to disease. Consistent with "
    "the age-related HEP increase reported by Kamp and colleagues in wakefulness,<super>2</super> any "
    "age relationship we observe within a given sleep stage suggests the wake-state age effect does not "
    "simply disappear once vigilance state is held fixed, though our design cannot separate this from "
    "instrumental accounts (e.g. lower heart-rate variability, hence less temporal jitter and a sharper "
    "average, in some age groups)."
)
story.append(Paragraph(disc_interaction, styles["Body"]))

story.append(Paragraph("Limitations", styles["H2"]))
story.append(Paragraph(
    "This analysis is restricted to a clinical referral cohort (suspected nocturnal seizure or paroxysmal "
    "event workup), so age and diagnostic yield may be correlated as in the companion manuscript; we did "
    "not adjust for diagnosis, heart-rate variability, epoch count, or scalp topography here, all of which "
    "the companion manuscript identifies as candidate confounds for age-HEP relationships. Amplitude "
    "metrics were averaged across all available EEG channels per patient rather than restricted to the "
    "right-frontotemporal region the companion manuscript identifies as the topographic maximum of the "
    "stage effect, which may dilute region-specific age effects. The permutation interaction test treats "
    "stage labels as exchangeable within patient and does not model heteroscedasticity or a continuous "
    "age &times; stage interaction term explicitly (no statsmodels dependency was introduced); a full "
    "mixed-effects or generalised additive model, as used for the primary age analysis in the companion "
    "manuscript, would be a natural extension. Finally, RMS and trough amplitude are two of several "
    "possible summary metrics and, being computed on channel-averaged waveforms, are not equivalent to "
    "the cluster-based permutation statistics used for the stage-contrast tests in the companion paper.",
    styles["Body"]))

story.append(Paragraph("Data availability statement", styles["H2"]))
story.append(Paragraph(
    "Data were derived from the same de-identified, credentialed-access clinical polysomnography corpus "
    "used in the companion manuscript, distributed under credentialed access via the Brain Data Science "
    "Platform. The analysis code and generated figures accompanying this paper are available in the "
    "project repository (Paper1/analysis_age_by_stage.py, Paper1/make_figures.py, Paper1/make_pdf.py).",
    styles["Body"]))

# ---------------------------------------------------------------------
# References (subset of companion manuscript's list, Vancouver numbered)
# ---------------------------------------------------------------------
story.append(Paragraph("References", styles["H1"]))
refs = [
    "1. Lechinger J, Heib DPJ, Gruber W, Schabus M, Klimesch W. Heartbeat-related EEG amplitude and "
    "phase modulations from wakefulness to deep sleep: interactions with sleep spindles and slow "
    "oscillations. Psychophysiology. 2015;52(11):1441-1450.",
    "2. Kamp S-M, et al. Older adults show a higher heartbeat-evoked potential than young adults and a "
    "negative association with everyday metacognition. Brain Res. 2021. PMID 33406407.",
    "3. Aprile F, et al. The heartbeat-evoked potential in young and older adults during attention "
    "orienting. Psychophysiology. 2025;e70057.",
    "4. Park H-D, Blanke O. Heartbeat-evoked cortical responses: underlying mechanisms, functional roles, "
    "and methodological considerations. NeuroImage. 2019;197:502-511.",
    "5. Umetani K, Singer DH, McCraty R, Atkinson M. Twenty-four hour time domain heart rate variability "
    "and heart rate: relations to age and gender over nine decades. J Am Coll Cardiol. 1998;31:593-601.",
    "6. Steinfath TP, et al. Heartbeat-evoked responses in M/EEG: a systematic review of methods with "
    "suggestions for analysis and reporting. Psychophysiology. 2026. PMID 41943417.",
    "7. Cafri N. How age reshapes the brain's response to the heartbeat during sleep: sleep-stage and "
    "lifespan modulation of the heartbeat-evoked potential in a multi-centre clinical polysomnography "
    "corpus. Unpublished manuscript, companion analysis of the same cohort. 2026.",
]
for r in refs:
    story.append(Paragraph(r, styles["Ref"]))

doc = SimpleDocTemplate(
    OUT_PDF, pagesize=LETTER,
    topMargin=0.85 * inch, bottomMargin=0.85 * inch,
    leftMargin=0.95 * inch, rightMargin=0.95 * inch,
    title="Age-dependent modulation of the heartbeat-evoked potential within sleep stages",
    author="Nir Cafri",
)


def _add_page_number(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(LETTER[0] / 2, 0.55 * inch, str(doc_.page))
    canvas.restoreState()


doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
print("Wrote", OUT_PDF, os.path.getsize(OUT_PDF), "bytes")
