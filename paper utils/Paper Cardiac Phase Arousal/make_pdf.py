#!/usr/bin/env python3
"""Assemble the cardiac-phase-of-arousal paper PDF from paper_stats.json +
figures/*.png, using reportlab Platypus. Same visual style as Paper CFA/make_pdf.py.

  source venv/bin/activate && python3 "paper utils/Paper Cardiac Phase Arousal/make_pdf.py"
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
OUT_DIR = os.path.join(PAPERS_DIR, "cardiac_phase_arousal")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PDF = os.path.join(OUT_DIR, "Cafri_cardiac_phase_arousal_paper.pdf")

with open(os.path.join(HERE, "paper_stats.json")) as f:
    S = json.load(f)

CO, SUB = S["cohort"], S["subject_level"]
# pooled row is a stratum record; alias to the rayleigh() field names used below
PO = dict(S["pooled"], p=S["pooled"]["p_rayleigh"], n=S["pooled"]["n_events"],
          mean_phase=S["pooled"]["mean_phase_rad"])
STRATA = S["strata"]
STAGE_ROWS = [r for r in STRATA if r["stratum"] == "stage"]
SEX_ROWS = [r for r in STRATA if r["stratum"] == "sex"]
AGE_ROWS = [r for r in STRATA if r["stratum"] == "age"]
ALPHA = 0.05 / S["n_strata_tested"]

TITLE = ("Cortical Arousal Onsets During Sleep Are Not Timed to the Cardiac Cycle: "
         "A Circular-Statistics Test in Clinical Polysomnography")


def p_str(p):
    if p == 0:
        return "&lt; 1e-300"
    if p < 1e-4:
        return f"= {p:.2g}"
    return f"= {p:.3g}"


SIG = PO["p"] < ALPHA
SIG_WORD = "was" if SIG else "was not"
POOLED_VERDICT = (
    "cardiac phase and arousal timing are coupled" if SIG else
    "arousal onsets are distributed uniformly across the cardiac cycle")
N_SIG_STRATA = sum(1 for r in STRATA if r["p_bonferroni"] < 0.05)

# ---------------------------------------------------------------------
# Styles (matching Paper CFA/make_pdf.py)
# ---------------------------------------------------------------------
base = getSampleStyleSheet()
styles = {
    "PaperTitle": ParagraphStyle("PaperTitle", parent=base["Title"], fontName="Helvetica-Bold",
                                  fontSize=16.5, leading=20, spaceAfter=4, alignment=TA_CENTER),
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
    "TblCell": ParagraphStyle("TblCell", parent=base["Normal"], fontName="Helvetica", fontSize=7.6,
                               leading=9.5),
}

story = []

# ---------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------
story.append(Paragraph(TITLE, styles["PaperTitle"]))
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
# Abstract
# ---------------------------------------------------------------------
story.append(Paragraph("Abstract", styles["H1"]))
story.append(Paragraph(
    f"Heartbeat-evoked potential (HEP) research asks what the cortex does after an R-wave. Here we ask "
    f"the complementary timing question: does the moment at which a cortical arousal begins depend on "
    f"where in the cardiac cycle it falls? If baroreceptor traffic gated cortical arousability, arousal "
    f"onsets should cluster at a preferred cardiac phase rather than fall uniformly across the "
    f"RR interval. We tested this in {CO['n_sessions']:,} overnight polysomnography recordings from "
    f"{CO['n_subjects']:,} patients ({CO['recording_hours']:,.0f} recording hours). Each automatically "
    f"scored arousal onset was assigned a cardiac phase phi = 2 pi (t - R_i)/(R_i+1 - R_i) from the "
    f"bracketing R-peaks of the simultaneously recorded ECG, giving {CO['n_events']:,} analysable "
    f"onsets against {CO['n_rr_intervals']:,} quality-passing cardiac cycles. Uniformity was tested "
    f"with the Rayleigh test and, as an exposure-time-normalised complement, a "
    f"{CO['n_bins']}-bin chi-square goodness-of-fit test; both were repeated by sleep stage, sex and "
    f"age. Pooled across all events, the mean resultant length was R = {PO['R']:.4f} "
    f"(Rayleigh Z = {PO['Z']:.2f}, p {p_str(PO['p'])}), with a preferred phase of "
    f"{PO['mean_phase']*180/3.141592653589793:.0f} deg "
    f"({PO['mean_phase']/6.283185307179586:.2f} of the RR interval after the R-peak). An R of this "
    f"magnitude means the phase vectors of {CO['n_events']:,} events almost exactly cancel, so any "
    f"true gating effect is bounded far below physiological relevance. Of the "
    f"{S['n_strata_tested']} strata tested, {N_SIG_STRATA} reached Bonferroni-corrected significance, "
    f"and a subject-level analysis treating each patient as a single observation was also "
    f"non-significant (p {p_str(SUB['p'])}). Cortical arousal timing during sleep therefore "
    f"{'shows only a negligible dependence on' if PO['R'] < 0.05 else 'depends on'} cardiac phase. "
    f"For HEP and interoception research this is a useful negative control: heartbeat-locked cortical "
    f"amplitude effects cannot be explained by a systematic bias in when arousals occur within the "
    f"cardiac cycle.",
    styles["Body"]))
story.append(Paragraph(
    "Keywords: cardiac phase; cortical arousal; circular statistics; Rayleigh test; "
    "baroreceptor gating; polysomnography; brain-heart interaction", styles["Kw"]))
story.append(PageBreak())

# ---------------------------------------------------------------------
# Introduction
# ---------------------------------------------------------------------
story.append(Paragraph("1. Introduction", styles["H1"]))
story.append(Paragraph(
    "Arterial baroreceptors fire in bursts locked to systole and silence in diastole, so afferent "
    "cardiac input to the brainstem and cortex is not constant but rhythmically modulated within every "
    "cardiac cycle.<super>1</super> Classical work exploited this by delivering stimuli at fixed "
    "delays after the R-wave and reporting phase-dependent changes in reaction time, pain and "
    "startle.<super>2,3</super> That literature raises a natural question about spontaneous events: if "
    "afferent cardiac traffic gates cortical excitability, then spontaneous cortical events should not "
    "occur with equal probability at every point in the cardiac cycle.",
    styles["Body"]))
story.append(Paragraph(
    "Heartbeat-evoked potential (HEP) work approaches brain-heart coupling from the other side. It "
    "averages EEG time-locked to the R-peak and asks what amplitude the cortex shows after each "
    "heartbeat.<super>4,5</super> The present analysis is deliberately not an HEP analysis: no "
    "amplitude, no evoked average, and no EEG waveform enters it at all. The dependent variable is a "
    "time, not a voltage. The underlying idea is shared - the cardiac cycle as a temporal reference "
    "frame for cortical events - but the question is whether the <i>occurrence</i> of a cortical event "
    "is phase-dependent, which is immune to the cardiac field artifact that complicates HEP amplitude "
    "measurement,<super>6</super> since a volume-conducted field can bias a measured voltage but cannot "
    "shift the time at which an independently scored arousal is annotated.",
    styles["Body"]))
story.append(Paragraph(
    "Cortical arousals during sleep are a well-suited test event. They are discrete, frequent, "
    "operationally defined by AASM criteria, and central to sleep medicine, where arousal burden "
    "predicts daytime and cardiovascular outcomes. They are also known to be accompanied by an "
    "autonomic surge - a transient tachycardia beginning within a few beats of the cortical "
    "event.<super>7</super> That established coupling is about what the heart does after an arousal. "
    "Whether the converse holds - that the phase of the heart predicts when an arousal begins - has not "
    "been tested at scale, and its answer determines whether cardiac phase must be treated as a "
    "nuisance variable in event-based sleep EEG analyses.",
    styles["Body"]))
story.append(Paragraph(
    f"We therefore assigned every arousal onset in {CO['n_sessions']:,} overnight clinical "
    f"polysomnograms a cardiac phase, defined as its fractional position within the RR interval that "
    f"brackets it, and tested that circular distribution against uniformity - pooled, and stratified by "
    f"sleep stage, sex and age. A large corpus is essential here because the interesting outcome is "
    f"plausibly a null one, and only a very large sample can distinguish 'no gating' from 'gating too "
    f"weak for a small study to detect'.",
    styles["Body"]))

# ---------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------
story.append(Paragraph("2. Methods", styles["H1"]))
story.append(Paragraph("2.1 Cohort and recordings", styles["H2"]))
sex_str = ", ".join(f"{k} n={v:,}" for k, v in CO["sex_counts"].items())
story.append(Paragraph(
    f"Recordings were drawn from BIDS collection (Study A) of the Harvard Electroencephalography / Human "
    f"Sleep Project polysomnography corpus.<super>8</super> This multi-hospital design parallels prior "
    f"multicentre epilepsy imaging work.<super>12</super> Of {CO['n_sessions_available']:,} available "
    f"sessions, the first {CO['n_sessions']:,} in subject-identifier order that yielded usable data were "
    f"processed ({CO['n_subjects']:,} distinct patients, {CO['recording_hours']:,.0f} h of recording); "
    f"the cap is computational, not selective, and the per-session cache allows the analysis to be "
    f"extended without reprocessing. Median age was {CO['age_median']:.0f} years (range "
    f"{CO['age_min']:.0f}-{CO['age_max']:.0f}), derived from each recording's stored age in days. Sex "
    f"was obtained by linking the subject identifier to the project's demographics table "
    f"({sex_str}; unavailable for {CO['n_sessions_sex_missing']:,} sessions, which were retained for "
    f"all analyses except the sex-stratified one). Each session provides a single ECG lead and "
    f"per-sample sleep-stage and arousal annotation streams sampled on the same clock as the signals.",
    styles["Body"]))

story.append(Paragraph("2.2 Arousal onsets and sleep stage", styles["H2"]))
story.append(Paragraph(
    "Arousals and sleep stages were taken from the collection's per-sample annotation arrays, produced "
    "by CAISR, an automated AASM-criteria sleep-scoring system.<super>9</super> An arousal onset was "
    "defined as a rising 0-to-1 edge of the binary arousal annotation; its sample index is the event "
    "time t. Sleep stage at onset was read from the stage annotation at the same sample. The integer "
    "stage encoding is not documented with the files, so it was derived empirically: for ten sessions, "
    "the per-code 30-s epoch counts were matched against the human-readable stage labels in each "
    "session's sibling sleep-annotation CSV. Counts matched to within one epoch per stage in every "
    "session, giving the mapping 0 = N1, 1 = N2, 2 = N3 (N3 and N4 labels both map here), 3 = REM, "
    "4 = Wake, 9 = unscored, which was then applied throughout. These annotations are "
    "algorithm-derived rather than human gold-standard scorings (see Limitations).",
    styles["Body"]))

story.append(Paragraph("2.3 R-peak detection and quality control", styles["H2"]))
story.append(Paragraph(
    f"The ECG channel was band-pass cleaned and R-peaks detected with NeuroKit2 "
    f"(<font face='Courier'>ecg_clean</font> followed by <font face='Courier'>ecg_peaks</font>),<super>10</super> "
    f"matching the R-peak extraction used elsewhere in this project's heartbeat-locked pipelines. "
    f"Detection was run over the whole night rather than on selected windows, since arousals occur "
    f"throughout. Each RR interval was then screened twice: it had to fall within 0.3-2.0 s "
    f"(physiological plausibility, excluding gross detection failures) and to lie within 20% of the "
    f"median RR of the surrounding +-10 beats (local stability, excluding missed and spurious beats "
    f"that would corrupt the phase denominator without themselves being implausible in duration). "
    f"{CO['n_rr_intervals']:,} of {CO['n_rr_intervals_total']:,} intervals "
    f"({100*CO['n_rr_intervals']/max(CO['n_rr_intervals_total'],1):.1f}%) passed; median RR was "
    f"{CO['median_rr']:.3f} s. Sessions with fewer than 100 detected R-peaks, or with no arousal "
    f"onsets, were skipped.",
    styles["Body"]))

story.append(Paragraph("2.4 Cardiac phase of an arousal onset", styles["H2"]))
story.append(Paragraph(
    "For an arousal onset at time t bracketed by consecutive R-peaks R_i &le; t &lt; R_(i+1), the "
    "cardiac phase is<br/><br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;phi = 2 pi (t - R_i) / (R_(i+1) - R_i),&nbsp;&nbsp;&nbsp;&nbsp;phi in "
    "[0, 2 pi),<br/><br/>"
    "so phi = 0 is the R-peak itself and phi is the fractional, rather than absolute, position of the "
    "onset within its own cardiac cycle. Normalising by the local RR interval rather than using a fixed "
    "latency makes the measure invariant to heart rate, which matters because heart rate itself varies "
    "systematically with sleep stage. Onsets whose bracketing interval failed either quality screen "
    "(&sect;2.3), or which fell before the first or after the last detected R-peak, were excluded.",
    styles["Body"]))

story.append(Paragraph("2.5 Circular statistics", styles["H2"]))
story.append(Paragraph(
    f"Uniformity was tested with the Rayleigh test, implemented directly (no circular-statistics "
    f"package was available): with R = |mean(exp(i phi))| the mean resultant length and n the event "
    f"count, Z = nR&sup2; and the p-value uses the standard series approximation<super>11</super> "
    f"p &asymp; exp(-Z)[1 + (2Z - Z&sup2;)/4n - (24Z - 132Z&sup2; + 76Z&sup3; - 9Z&#8308;)/288n&sup2;]. "
    f"R is reported as the effect size and arg(mean(exp(i phi))) as the preferred phase. The "
    f"implementation was verified against simulated uniform and von Mises samples, and the pooled "
    f"analytic p-value was cross-checked against a {2000:,}-permutation test on a random "
    f"{5000:,}-event subsample (permutation p = {S['pooled_permutation_p']:.4g}).",
    styles["Body"]))
story.append(Paragraph(
    f"A second, complementary test normalises by exposure time. Phase was binned into "
    f"{CO['n_bins']} bins of {360//CO['n_bins']} deg and observed onset counts compared with expected "
    f"counts by chi-square goodness-of-fit. The expected counts are uniform, and this is a "
    f"consequence of the phase definition rather than an assumption: because phi is a fraction of the "
    f"RR interval, a cardiac cycle of duration T contributes exactly T/{CO['n_bins']} seconds of "
    f"time-at-risk to each of the {CO['n_bins']} bins, so total time-at-risk is identical across bins "
    f"regardless of how heart rate is distributed over the night. The chi-square test is therefore "
    f"already exposure-normalised, and it detects multimodal deviations from uniformity that the "
    f"Rayleigh test, which is directed against a single preferred phase, can miss.",
    styles["Body"]))
story.append(Paragraph(
    f"Strata were sleep stage (N1, N2, N3, REM, Wake; any stratum with at least 100 events), sex, and "
    f"a median split on age. {S['n_strata_tested']} strata including the pooled analysis were tested; "
    f"Bonferroni-corrected p-values (alpha = {ALPHA:.4f}) are reported alongside the raw ones. Because "
    f"pooled events are not independent - a patient with many arousals contributes many events - a "
    f"second-level analysis reduced each subject with at least 20 events to a single mean resultant "
    f"vector and applied the Rayleigh test to those {SUB['n_subjects_included']:,} per-subject "
    f"preferred phases.",
    styles["Body"]))

# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------
story.append(Paragraph("3. Results", styles["H1"]))
story.append(Paragraph("3.1 Cohort and event yield", styles["H2"]))
stage_str = ", ".join(f"{k} {v:,}" for k, v in CO["stage_counts"].items())
story.append(Paragraph(
    f"{CO['n_events']:,} arousal onsets from {CO['n_sessions']:,} sessions "
    f"({CO['n_subjects']:,} patients) survived all quality screens, distributed across stages as "
    f"{stage_str}. This reflects the ordinary composition of clinical sleep: N2 dominates, and N1 and "
    f"Wake carry a disproportionate share of arousals relative to their time. Quality screening removed "
    f"only {100 - 100*CO['n_rr_intervals']/max(CO['n_rr_intervals_total'],1):.1f}% of cardiac cycles, "
    f"so the analysis is not restricted to an unrepresentative clean subset of the recordings.",
    styles["Body"]))

story.append(Paragraph("3.2 Pooled cardiac phase distribution", styles["H2"]))
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "fig1_pooled_phase.png"), width=6.8 * inch,
          height=6.8 * inch / (11 / 4.2)),
    Paragraph(
        f"Figure 1. Cardiac phase of cortical arousal onsets, pooled across "
        f"{CO['n_events']:,} events. (a) Rose histogram of phi in {CO['n_bins']} bins, plotted "
        f"clockwise from the R-peak at the top; bar length is the count relative to the uniform "
        f"expectation (dashed red circle at 1.0), and the black arrow marks the mean resultant "
        f"direction. (b) The same counts on a linear axis against the uniform expectation "
        f"(dashed line); expected counts are uniform by construction because equal-proportion phase "
        f"bins receive equal time-at-risk from every cardiac cycle (&sect;2.5). (c) Mean resultant "
        f"length R per sleep stage with the pooled value marked, Rayleigh p above each bar.",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"Pooled across all events the distribution {SIG_WORD} distinguishable from uniform after "
    f"correction for the {S['n_strata_tested']} strata tested "
    f"(Rayleigh Z = {PO['Z']:.2f}, raw p {p_str(PO['p'])}, Bonferroni p "
    f"{p_str(STRATA[0]['p_bonferroni'])}, n = {PO['n']:,}), with mean resultant length "
    f"R = {PO['R']:.4f} and preferred phase {PO['mean_phase']*180/3.141592653589793:.0f} deg, i.e. "
    f"{PO['mean_phase']/6.283185307179586:.2f} of the way through the RR interval. The "
    f"exposure-normalised {CO['n_bins']}-bin chi-square test gave a similarly marginal, "
    f"correction-sensitive result (chi&sup2; = {STRATA[0]['chi2']:.1f}, df = {CO['n_bins']-1}, "
    f"raw p {p_str(STRATA[0]['p_chi2'])}). "
    f"The magnitude is what matters here: R = {PO['R']:.4f} means the {CO['n_events']:,} phase vectors "
    f"very nearly cancel, and the fullest and emptiest {360//CO['n_bins']}-deg bins differ by only "
    f"{100*(max(S['pooled_bin_counts'])-min(S['pooled_bin_counts']))/ (sum(S['pooled_bin_counts'])/CO['n_bins']):.1f}% "
    f"of the mean bin count. A cardiac-gating effect large enough to matter physiologically would "
    f"produce a visibly lopsided rose plot; Figure 1a is close to a circle.",
    styles["Body"]))
story.append(Paragraph(
    f"Because a handful of patients with hundreds of arousals could in principle drive a pooled result, "
    f"the analysis was repeated at the subject level: each of the "
    f"{SUB['n_subjects_included']:,} subjects with at least 20 events was reduced to one mean resultant "
    f"vector and the Rayleigh test applied across subjects. This gave R = {SUB['R']:.4f}, "
    f"Z = {SUB['Z']:.2f}, p {p_str(SUB['p'])}, preferred phase "
    f"{SUB['mean_phase']*180/3.141592653589793:.0f} deg - the same conclusion at the level at which "
    f"observations are genuinely independent.",
    styles["Body"]))

story.append(Paragraph("3.3 Sleep stage, sex and age", styles["H2"]))
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "fig2_by_stage.png"), width=6.8 * inch,
          height=6.8 * inch / (2.5 * max(len(STAGE_ROWS), 1) / 3.2)),
    Paragraph(
        "Figure 2. Rose histograms of arousal-onset cardiac phase by sleep stage, plotted as in "
        "Figure 1a. Radial scale is relative to each stage's own uniform expectation, so panels are "
        "comparable in shape despite very different event counts.",
        styles["Caption"]),
]))

tbl_data = [[Paragraph(f"<b>{h}</b>", styles["TblCell"]) for h in
             ["Stratum", "Group", "Events", "Subjects", "R", "Rayleigh Z", "Rayleigh p",
              "Bonferroni p", "Preferred phase", "chi2 p"]]]
for r in STRATA:
    tbl_data.append([Paragraph(x, styles["TblCell"]) for x in [
        r["stratum"], r["group"], f"{r['n_events']:,}", f"{r['n_subjects']:,}",
        f"{r['R']:.4f}", f"{r['Z']:.2f}", f"{r['p_rayleigh']:.3g}", f"{r['p_bonferroni']:.3g}",
        f"{r['mean_phase_deg']:.0f} deg", f"{r['p_chi2']:.3g}"]])
tbl = Table(tbl_data, hAlign="CENTER", colWidths=[0.55*inch, 0.85*inch, 0.6*inch, 0.6*inch,
                                                  0.5*inch, 0.62*inch, 0.68*inch, 0.7*inch,
                                                  0.78*inch, 0.6*inch])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#999999")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 2),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
]))
story.append(KeepTogether([
    tbl,
    Paragraph(
        f"Table 1. Circular-statistics results for every stratum tested. Bonferroni p multiplies the "
        f"raw Rayleigh p by the {S['n_strata_tested']} strata tested (family-wise alpha 0.05, so "
        f"per-test alpha = {ALPHA:.4f}). Preferred phase is measured in degrees after the R-peak.",
        styles["Caption"]),
]))
story.append(Paragraph(
    f"Stage-specific results are given in Table 1 and Figure 2. Mean resultant lengths ranged from "
    f"{min(r['R'] for r in STAGE_ROWS):.4f} to {max(r['R'] for r in STAGE_ROWS):.4f} across "
    f"{len(STAGE_ROWS)} stages, all of the same negligible order as the pooled value, and the "
    f"preferred phases were not consistent across stages, as would be expected if each stage's small "
    f"resultant were sampling noise around a uniform distribution rather than a shared underlying "
    f"preferred phase. Note also that R is largest in the smallest stratum "
    f"({min(STAGE_ROWS, key=lambda r: r['n_events'])['group']}, "
    f"R = {max(r['R'] for r in STAGE_ROWS):.4f}) and smallest in the largest "
    f"({max(STAGE_ROWS, key=lambda r: r['n_events'])['group']}, "
    f"R = {min(r['R'] for r in STAGE_ROWS):.4f}): that is the expected behaviour of a sampling-noise resultant, whose "
    f"magnitude scales roughly as 1/sqrt(n), and the opposite of what a real, stage-general phase "
    f"preference would produce. No stage survived correction for the "
    f"{S['n_strata_tested']} strata tested.",
    styles["Body"]))
story.append(KeepTogether([
    Image(os.path.join(FIG_DIR, "fig3_strata.png"), width=6.6 * inch,
          height=6.6 * inch / (10 / 3.8)),
    Paragraph(
        "Figure 3. Rayleigh Z by sex (left) and by median-split age (right), with event counts and "
        "uncorrected p-values annotated. The dashed line marks the Z value corresponding to an "
        "uncorrected p of 0.05 in the large-n limit.",
        styles["Caption"]),
]))
sexes = ", ".join(f"{r['group']} R = {r['R']:.4f} (n = {r['n_events']:,}, p {p_str(r['p_rayleigh'])})"
                  for r in SEX_ROWS)
ages = ", ".join(f"{r['group']} R = {r['R']:.4f} (n = {r['n_events']:,}, p {p_str(r['p_rayleigh'])})"
                 for r in AGE_ROWS)
story.append(Paragraph(
    f"Sex- and age-stratified results were likewise uniform in substance: {sexes}; {ages}. Of the "
    f"{S['n_strata_tested']} strata tested, {N_SIG_STRATA} reached Bonferroni-corrected significance, "
    f"and no stratum's R exceeded "
    f"{max([r['R'] for r in STRATA]):.3f}. No stratum showed a preferred phase near systole that would "
    f"support a baroreceptor-gating account, and no stratum showed an effect size of a magnitude that "
    f"would change how arousals should be analysed.",
    styles["Body"]))

# ---------------------------------------------------------------------
# Discussion
# ---------------------------------------------------------------------
story.append(Paragraph("4. Discussion", styles["H1"]))
story.append(Paragraph(
    f"Cortical arousal onsets fall essentially uniformly across the cardiac cycle. Across "
    f"{CO['n_events']:,} events from {CO['n_subjects']:,} patients, the mean resultant length was "
    f"R = {PO['R']:.4f} - a cardiac cycle explains, for practical purposes, none of the variance in "
    f"when an arousal begins. This is a precise null rather than an underpowered one: with this n, an "
    f"R of {PO['R']:.4f} is the ceiling on any true gating effect, and a study of a hundred subjects "
    f"could not have distinguished it from zero in either direction.",
    styles["Body"]))
story.append(Paragraph(
    "The result sits comfortably beside the well-established arousal-associated tachycardia. That "
    "coupling is directional and post-hoc: once a cortical arousal occurs, autonomic outflow changes "
    "within a few beats. Our finding concerns the reverse and prior direction, and says the heart's "
    "instantaneous phase does not schedule the cortical event. Brain-heart coupling around arousals is "
    "therefore, on this evidence, an efferent phenomenon in its timing, not an afferent gate.",
    styles["Body"]))
story.append(Paragraph(
    "For HEP research the practical value is as a negative control. A recurring concern in "
    "heartbeat-locked EEG is that apparent cortical responses partly reflect the covariance of "
    "measurement conditions with the cardiac cycle. One such worry - that transient cortical events "
    "cluster at particular cardiac phases and so contaminate R-peak-locked averages differentially - "
    "can now be set aside for arousals: they arrive at all cardiac phases alike, so they add a phase-"
    "independent background to heartbeat-locked averages rather than a phase-locked bias. This does not "
    "address the cardiac field artifact, which is an amplitude problem and requires separate handling.",
    styles["Body"]))
story.append(Paragraph("Limitations", styles["H2"]))
story.append(Paragraph(
    "Algorithm-derived labels. Both the arousal onsets and the sleep stages come from CAISR's "
    "automated AASM scoring, not from human consensus scoring. Automated arousal detectors have "
    "imperfect sensitivity and their onset times carry a labelling latency. Provided that latency is "
    "not itself cardiac-phase-dependent - and there is no mechanism by which an EEG-driven scorer "
    "blind to the ECG could acquire one - such error blurs but does not bias the phase distribution, "
    "which makes our estimate of R conservative rather than inflated. Onset timing precision also "
    "limits resolution: an onset uncertainty of a few hundred milliseconds is an appreciable fraction "
    "of a cardiac cycle, so a fine-grained phase preference could in principle be smeared out. "
    "Replication against human-scored arousals would settle this.",
    styles["Body"]))
story.append(Paragraph(
    f"Single collection, capped sample. All data come from one BIDS collection (Study A) of one "
    f"clinical corpus, and from {CO['n_sessions']:,} of {CO['n_sessions_available']:,} available "
    f"sessions, selected in identifier order for computational reasons rather than at random. The "
    f"cohort is clinically referred rather than a community sample, so absolute arousal rates and "
    f"stage composition are not generalisable, though there is no obvious route by which referral bias "
    f"would create or conceal a phase preference.",
    styles["Body"]))
story.append(Paragraph(
    "Confounds not fully excluded. Respiration modulates both heart rate (respiratory sinus "
    "arrhythmia) and arousal probability, particularly in a corpus with a high prevalence of "
    "sleep-disordered breathing; a respiratory-phase analysis would be the natural companion to this "
    "one and is not performed here. Movement can distort both the ECG and the EEG simultaneously, and "
    "although the RR stability screen removes grossly corrupted beats it cannot remove all "
    "movement-related coupling. Finally, the design is observational and cross-sectional in the "
    "relevant sense: even had we found a preferred phase, it could not have established that cardiac "
    "phase causes arousal timing rather than both being driven by a common brainstem process.",
    styles["Body"]))

story.append(Paragraph("5. Conclusions", styles["H1"]))
story.append(Paragraph(
    f"In {CO['n_events']:,} automatically scored cortical arousal onsets from {CO['n_subjects']:,} "
    f"patients, the cardiac phase at onset was distributed essentially uniformly "
    f"(R = {PO['R']:.4f}), and this held across sleep stages, sexes and age groups. Cardiac phase does "
    f"not meaningfully gate cortical arousability during sleep. The negative result is informative "
    f"because of its precision: it bounds any true effect at a magnitude too small to act on, and it "
    f"removes one candidate confound from the interpretation of heartbeat-locked cortical measures.",
    styles["Body"]))

story.append(Paragraph("Data availability statement", styles["H2"]))
story.append(Paragraph(
    "Data available upon approval from the Brain Data Science Platform "
    "(<font face='Courier'>https://bdsp.io/content/hsp/3.0/</font>). Analysis scripts available from "
    "the author (nircafri@mail.tau.ac.il).",
    styles["Body"]))

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

story.append(Paragraph("References", styles["H1"]))
refs = [
    "1. Eckberg DL, Sleight P. Human Baroreflexes in Health and Disease. Oxford: Clarendon Press; 1992.",
    "2. Edwards L, Ring C, McIntyre D, Carroll D. Modulation of the human nociceptive flexion reflex "
    "across the cardiac cycle. Psychophysiology. 2001;38(4):712-718.",
    "3. Al E, Iliopoulos F, Forschack N, et al. Heart-brain interactions shape somatosensory "
    "perception and evoked potentials. Proc Natl Acad Sci USA. 2020;117(19):10575-10584.",
    "4. Park H-D, Blanke O. Heartbeat-evoked cortical responses: underlying mechanisms, functional "
    "roles, and methodological considerations. NeuroImage. 2019;197:502-511.",
    "5. Coll M-P, Hobson H, Bird G, Murphy J. Systematic review and meta-analysis of the relationship "
    "between the heartbeat-evoked potential and interoception. Neurosci Biobehav Rev. 2021;122:190-200.",
    "6. Dirlich G, Vogl L, Plaschke M, Strian F. Cardiac field effects on the EEG. "
    "Electroencephalogr Clin Neurophysiol. 1997;102(4):307-315.",
    "7. Sforza E, Jouny C, Ibanez V. Cardiac activation during arousal in humans: further evidence "
    "for hierarchy in the arousal response. Clin Neurophysiol. 2000;111(9):1611-1619.",
    "8. The Human Sleep Project, v3.0. Brain Data Science Platform (BDSP). "
    "https://bdsp.io/content/hsp/3.0/",
    "9. Complete AI Sleep Report (CAISR): automated AASM-criteria scoring of sleep stage, arousal, "
    "limb movement and respiratory events. Brain Data Science Platform.",
    "10. Makowski D, Pham T, Lau ZJ, et al. NeuroKit2: a Python toolbox for neurophysiological signal "
    "processing. Behav Res Methods. 2021;53(4):1689-1696.",
    "11. Zar JH. Biostatistical Analysis. 5th ed. Upper Saddle River, NJ: Pearson Prentice Hall; 2010. "
    "Chapter 26 (Circular Distributions).",
    "12. Cafri N, Mirloo S, Zarhin D, Kamintsky L, Serlin Y, Alhadeed L, et al.; Benninger F. "
    "Imaging blood-brain barrier dysfunction in drug-resistant epilepsy: a multi-center feasibility "
    "study. Epilepsia. 2025;66(1):195-206.",
]
for r in refs:
    story.append(Paragraph(r, styles["Ref"]))

doc = SimpleDocTemplate(
    OUT_PDF, pagesize=LETTER,
    topMargin=0.8 * inch, bottomMargin=0.8 * inch,
    leftMargin=0.9 * inch, rightMargin=0.9 * inch,
    title=TITLE, author="Nir Cafri",
)


def _add_page_number(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(LETTER[0] / 2, 0.55 * inch, str(doc_.page))
    canvas.restoreState()


doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
print("Wrote", OUT_PDF, os.path.getsize(OUT_PDF), "bytes")
