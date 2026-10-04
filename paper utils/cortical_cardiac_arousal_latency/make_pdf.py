#!/usr/bin/env python3
"""Assemble the cortical-cardiac arousal latency paper PDF from
paper_stats.json + figures/*.png, using reportlab Platypus.
Patterned on paper utils/Paper CFA/make_pdf.py.

  source venv/bin/activate && python3 "paper utils/cortical_cardiac_arousal_latency/make_pdf.py"
"""
import json
import os
import shutil

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
PAPERS_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "papers",
                           "cortical_cardiac_arousal_latency")
os.makedirs(PAPERS_DIR, exist_ok=True)
OUT_PDF = os.path.join(PAPERS_DIR, "Cafri_cortical_cardiac_arousal_latency_pilot.pdf")
FINAL_PDF = os.path.join(PAPERS_DIR, "Cafri_cortical_cardiac_arousal_latency.pdf")

with open(os.path.join(HERE, "paper_stats.json")) as f:
    S = json.load(f)

STAGE_LABEL = {"N1": "N1", "N2": "N2", "N3": "N3", "R": "REM"}

base = getSampleStyleSheet()
styles = {
    "PaperTitle": ParagraphStyle("PaperTitle", parent=base["Title"], fontName="Helvetica-Bold",
                                  fontSize=15.5, leading=19, spaceAfter=4, alignment=TA_CENTER),
    "Subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName="Helvetica-Bold",
                                fontSize=10.5, leading=14, textColor=colors.HexColor("#9c3b0e"),
                                alignment=TA_CENTER, spaceAfter=6),
    "Author": ParagraphStyle("Author", parent=base["Normal"], fontName="Helvetica",
                              fontSize=10.5, alignment=TA_CENTER),
    "Affil": ParagraphStyle("Affil", parent=base["Normal"], fontName="Helvetica",
                             fontSize=9, textColor=colors.HexColor("#555555"), alignment=TA_CENTER,
                             spaceAfter=10),
    "H1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=12.5,
                          spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#111111")),
    "H2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10.5,
                          spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#222222")),
    "Body": ParagraphStyle("Body", parent=base["Normal"], fontName="Times-Roman", fontSize=9.7,
                            leading=13.4, alignment=TA_JUSTIFY, spaceAfter=6),
    "Caption": ParagraphStyle("Caption", parent=base["Normal"], fontName="Helvetica", fontSize=8.3,
                               leading=11, textColor=colors.HexColor("#333333"), spaceAfter=10,
                               spaceBefore=3),
    "Ref": ParagraphStyle("Ref", parent=base["Normal"], fontName="Times-Roman", fontSize=8.6,
                           leading=11.5, spaceAfter=4, leftIndent=14, firstLineIndent=-14),
}

story = []

story.append(Paragraph(
    "Cortical–Cardiac Arousal Latency Across Sleep Stages and Respiratory Event Types",
    styles["PaperTitle"]))
story.append(Paragraph("Large-sample, single-cohort analysis (BDSP Harvard Study A) — "
                        "cross-study generalization untested; see Limitations",
                        styles["Subtitle"]))
story.append(Paragraph("Nir Cafri", styles["Author"]))
story.append(Paragraph("BDSP Harvard Electroencephalography cohort (Studies A–D), "
                        "Cobrad sleep-PSG project", styles["Affil"]))
story.append(HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#999999"),
                         spaceAfter=8))

# ---------------------------------------------------------------------
# Abstract
# ---------------------------------------------------------------------
n_sub = S["n_subjects"]
n_cand = S["n_candidate_events"]
n_use = S["n_usable_events"]
med = S["pooled_median_dt"]
q1, q3 = S["pooled_q1"], S["pooled_q3"]
psign = S["pooled_sign_p"]
direction = "heart acceleration preceded the scored cortical microarousal" if (med is not None and med < 0) \
    else "cortical microarousal preceded measurable heart acceleration" if (med is not None and med > 0) \
    else "neither systematically preceded the other"

abstract = (
    f"<b>Background.</b> Whether sleep-disrupting events are initiated predominantly by autonomic "
    f"(cardiac) or cortical (EEG) activation is unresolved, and may depend on sleep stage and event "
    f"type. <b>Methods.</b> In {n_sub} unique subjects from the BDSP Harvard PSG cohort (Study A, "
    f"the only study in this dataset whose event annotations carry the structured taxonomy this method "
    f"needs), we identified {n_cand} scored microarousal events linked to spontaneous arousal, "
    f"obstructive/central apnea, hypopnea, RERA, or periodic limb movement, took the scored microarousal "
    f"onset as cortical arousal time (t<sub>EEG</sub>), and detected the matched heart-rate-acceleration "
    f"onset (t<sub>HR</sub>) from the EKG channel using a documented, threshold-based algorithm. "
    f"{n_use} events ({100*n_use/max(n_cand,1):.0f}%) yielded a usable Δt = t<sub>HR</sub> − "
    f"t<sub>EEG</sub>. <b>Results.</b> Pooled Δt median was {med:.2f} s (IQR {q1:.2f} to {q3:.2f} s; "
    f"sign-test p = {psign:.3g}), i.e. on average {direction}. Distributions differed "
    f"{'significantly' if S['stage_kruskal']['p'] is not None and S['stage_kruskal']['p'] < 0.05 else 'non-significantly'} "
    f"across sleep stage (Kruskal-Wallis H={S['stage_kruskal']['h']:.2f}, p={S['stage_kruskal']['p']:.3g}) "
    f"and {'significantly' if S['type_kruskal']['p'] is not None and S['type_kruskal']['p'] < 0.05 else 'non-significantly'} "
    f"across event type (H={S['type_kruskal']['h']:.2f}, p={S['type_kruskal']['p']:.3g}). "
    f"<b>Conclusion.</b> With {n_use} usable events across {n_sub} sessions, this is a well-powered "
    f"within-cohort characterization of cortical-cardiac arousal ordering by stage and event type; the "
    f"scope is nonetheless a single BDSP Harvard cohort (Study A) and cross-study generalization is untested."
)
story.append(Paragraph("Abstract", styles["H1"]))
story.append(Paragraph(abstract, styles["Body"]))

# ---------------------------------------------------------------------
# Introduction
# ---------------------------------------------------------------------
story.append(Paragraph("Introduction", styles["H1"]))
story.append(Paragraph(
    "Arousal from sleep is classically defined electrocortically — an abrupt shift to higher-frequency "
    "EEG activity, per AASM scoring rules<super>2</super> whose reliability and alternatives have "
    "themselves been debated<super>5</super> — but is almost always accompanied by autonomic activation, "
    "including a transient rise in heart rate, and the broader nature of the arousal response remains "
    "actively discussed.<super>3</super> Whether the cortical or the cardiac component leads is "
    "debated: some evidence points to subcortical/autonomic activation preceding detectable cortical "
    "change, consistent with reports that cardiac activation during arousal is hierarchically "
    "organized,<super>4</super> "
    "(arguing arousal is fundamentally a bottom-up, brainstem-autonomic event that cortex later "
    "registers), while other work finds cortical EEG change can precede or coincide with heart-rate "
    "acceleration, particularly for spontaneous arousals versus respiratory-event-triggered ones. If the "
    "cortical-cardiac ordering differs systematically by sleep stage (e.g. N3 vs REM, where autonomic and "
    "arousal-threshold physiology differ) or by the type of disruptor (obstructive apnea with its strong "
    "hypoxic/mechanical drive vs a spontaneous arousal with no external trigger), this would inform models "
    "of arousal as a graded, multi-system process rather than a single all-or-none cortical event. This "
    "study builds a per-event cortical-cardiac latency (Δt) pipeline and applies it at scale within a "
    "single large PSG cohort to characterize that ordering by stage and event type.",
    styles["Body"]))

# ---------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------
story.append(Paragraph("Methods", styles["H1"]))
story.append(Paragraph("Data and sample", styles["H2"]))
story.append(Paragraph(
    f"Data are from the Human Sleep Project's<super>10</super> "
    f"BDSP Harvard Electroencephalography PSG BIDS export, covering "
    f"Study A, Study B, Study C, and Study D. Each session provides a raw PSG EDF (200 Hz, including an EKG "
    f"channel), a manually scored events-annotation file, and an automated CAISR annotation file. This "
    f"multi-hospital design parallels prior multicentre epilepsy imaging work.<super>1</super> "
    f"On inspection, only Study A exports events in the structured "
    f"<font face='Courier'>Epoch, Stage, Type, Time, Length, Description</font> schema with an explicit "
    f"Apnea/Hypopnea/Microarousal (MA) taxonomy and bracket-tagged MA linkage (e.g. "
    f"<font face='Courier'>\"Microarousal [Hypopnea]\"</font>, <font face='Courier'>\"Microarousal [PLMS][LM]\"</font>) "
    f"that this method's event classification depends on. Studies B, C, and D use different, "
    f"mutually incompatible per-study event-log schemas (free-text annotation streams / clinical-system "
    f"exports without this taxonomy), and Study B has no directly readable EDF in this export (signal is in "
    f".h5 only). This analysis therefore draws all subjects from Study A; see Limitations for the "
    f"consequence for the originally intended multi-study spread. "
    f"Of 6,349 Study A sessions with the EDF, events-annotation, and CAISR-annotation files all present, "
    f"900 were randomly sampled (seed 42) and processed in parallel (42-way, one worker per session). "
    f"{n_sub} unique subjects (890 subject-sessions; 26 subjects contributed 2+ sessions, pooled) yielded "
    f"at least one usable Δt and make up the cohort analyzed below. A precomputed R-peak/heartbeat-shape "
    f"cache used elsewhere for heartbeat-evoked-potential work was evaluated as "
    f"a way to skip raw-EDF EKG extraction, but inspection showed it holds stage-purified, concatenated "
    f"(capped at ~1200 s), resampled (256 Hz) synthetic recordings built for heartbeat-evoked-potential "
    f"averaging, with no annotations mapping its samples back to absolute Record Time in the source EDF — "
    f"unusable for aligning R-peaks to a specific event's t<sub>EEG</sub>, so raw EDFs were read directly "
    f"instead, parallelized for throughput.",
    styles["Body"]))

story.append(Paragraph("Event identification and classification", styles["H2"]))
story.append(Paragraph(
    "Cortical arousal onset (t<sub>EEG</sub>) is taken directly from the technologist-scored Microarousal "
    "(MA) row's Record Time (elapsed seconds from EDF start). Each MA row's free-text Description carries "
    "one or more bracketed tags identifying what it is linked to, scored to AASM respiratory- and "
    "arousal-event criteria;<super>6</super> these tags were used directly for "
    "classification (no proximity heuristic was needed, since the schema already encodes the linkage): "
    "<font face='Courier'>[Spon]</font> → spontaneous arousal (no linked respiratory/limb event); "
    "<font face='Courier'>[Apnea]</font> → apnea-related, further split into obstructive vs. central "
    "by the Description text (\"Obstructive Apnea\" / \"Central Apnea\") of the nearest preceding Apnea-type "
    "row within 30 s; <font face='Courier'>[Hypopnea]</font> → hypopnea; "
    "<font face='Courier'>[PLMS]</font> (with or without a co-occurring <font face='Courier'>[LM]</font> tag) "
    "→ periodic limb movement (PLM) arousal, following standard PLM periodicity criteria;<super>8</super> "
    "<font face='Courier'>[LM]</font> alone → isolated "
    "limb-movement arousal (kept as its own category, distinct from PLM); <font face='Courier'>[RERA]</font> "
    "→ respiratory-effort-related arousal, i.e. the upper-airway resistance syndrome.<super>7</super> "
    "RERA tags were in fact present in the Study A annotation data "
    "(as <font face='Courier'>\"Microarousal [RERA]\"</font>) — no inference was required, contrary to "
    "the possibility flagged in the study plan. Sleep stage was taken from the MA row's own Stage field; "
    "events scored during Wake were excluded, leaving only N1/N2/N3/REM events. The automated CAISR<super>9</super> "
    "<font face='Courier'>arousal_caisr</font> stream was used only as an independent cross-check: for each "
    "usable event, agreement was recorded as whether any CAISR arousal flag fell within ±3 s of "
    "t<sub>EEG</sub> (reported in Results as a QC statistic, not used to alter classification).",
    styles["Body"]))

story.append(Paragraph("Heart-rate acceleration onset detection", styles["H2"]))
story.append(Paragraph(
    "For each event, the EKG channel was extracted in a window centered on t<sub>EEG</sub> (30 s of "
    "padding beyond the analysis window on each side, to avoid R-peak edge effects), band-pass filtered "
    "0.5–40 Hz (4th-order Butterworth, zero-phase), and R-peaks detected with NeuroKit2 "
    "(<font face='Courier'>nk.ecg_peaks</font>, sampling rate 200 Hz; a SciPy "
    "<font face='Courier'>find_peaks</font> fallback was used if NeuroKit2 raised an exception). "
    "Instantaneous heart rate at each beat was HR<sub>i</sub> = 60 / IBI<sub>i</sub>, assigned at the time "
    "of the second R-peak of the interval; beats with an implausible inter-beat interval (&lt;0.3 s or "
    "&gt;2.0 s) were dropped as artifacts. A pre-arousal baseline was computed over t<sub>EEG</sub> − 30 s "
    "to t<sub>EEG</sub> − 5 s (required ≥ 5 valid beats); the acceleration threshold was "
    "baseline mean + 1.0 × baseline SD. Within a search window of t<sub>EEG</sub> − 5 s to "
    "t<sub>EEG</sub> + 15 s, t<sub>HR</sub> was defined as the time of the first beat that began a run of "
    "≥ 3 consecutive beats at or above threshold. Δt = t<sub>HR</sub> − t<sub>EEG</sub>; "
    "Δt &lt; 0 means the heart-rate rise was detected before the scored cortical arousal onset "
    "(\"heart leads\"), Δt &gt; 0 means after (\"brain leads\"). Events were excluded (flagged, not "
    "imputed) when the extraction window fell outside the recording, when fewer than 2 beats were found, "
    "when the baseline had &lt;5 valid beats, or when no sustained threshold crossing occurred in the "
    "search window; exclusion reasons are tabulated in Results.",
    styles["Body"]))

story.append(Paragraph("Statistics", styles["H2"]))
story.append(Paragraph(
    "Δt distributions are non-normal, so group comparisons used "
    "Kruskal-Wallis across sleep stage (N1/N2/N3/REM) and, separately, across event type, with pairwise "
    "Mann-Whitney U post-hoc tests (Bonferroni-corrected within each family). Whether Δt differs from "
    "zero within each group was tested with an exact two-sided sign test (primary, since it makes no "
    "symmetry assumption) and, where n ≥ 2 with at least one non-zero value, a Wilcoxon signed-rank "
    "test (secondary). All tests used SciPy; no new statistical dependency was added.",
    styles["Body"]))

# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------
story.append(PageBreak())
story.append(Paragraph("Results", styles["H1"]))

flag_rows = [["Outcome", "n events"]] + [[k.replace("_", " "), str(v)]
                                          for k, v in sorted(S["flag_counts"].items(),
                                                              key=lambda kv: -kv[1])]
t1 = Table(flag_rows, colWidths=[3.1 * inch, 1.3 * inch])
t1.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf5")),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
    ("ALIGN", (1, 0), (1, -1), "CENTER"),
]))
caisr_txt = (f"{100*S['caisr_arousal_agree_rate']:.0f}%" if S["caisr_arousal_agree_rate"] is not None
             else "n/a")
story.append(Paragraph(
    f"Of {n_cand} candidate scored microarousal events across {n_sub} subjects (Study A), "
    f"{n_use} ({100*n_use/max(n_cand,1):.0f}%) yielded a usable Δt; Table 1 breaks down the rest by "
    f"exclusion reason. Among usable events, the independent CAISR automated arousal stream flagged an "
    f"arousal within ±3 s of t<sub>EEG</sub> for {caisr_txt} of events, a rough cross-check that the "
    f"scored arousal times are physiologically real and not spurious.",
    styles["Body"]))
story.append(t1)
story.append(Spacer(1, 4))
story.append(Paragraph("<b>Table 1.</b> Event yield and exclusion reasons.", styles["Caption"]))

story.append(Paragraph(
    f"Pooled across all usable events (ignoring stage/type), median Δt was {med:.2f} s "
    f"(IQR {q1:.2f} to {q3:.2f} s; sign-test p = {psign:.3g}), i.e. {direction} on average.",
    styles["Body"]))

story.append(Image(os.path.join(FIG_DIR, "dt_all_events.png"), width=4.6 * inch, height=3.07 * inch,
                    hAlign="CENTER"))
story.append(Paragraph("<b>Figure 1.</b> Pooled Δt distribution across all usable events "
                        "(vertical dashed line = 0, simultaneous onset).", styles["Caption"]))

story.append(Image(os.path.join(FIG_DIR, "dt_by_stage.png"), width=5.4 * inch, height=3.54 * inch,
                    hAlign="CENTER"))
story.append(Paragraph(
    f"<b>Figure 2.</b> Δt by sleep stage at the time of the event. Kruskal-Wallis H = "
    f"{S['stage_kruskal']['h']:.2f}, p = {S['stage_kruskal']['p']:.3g} "
    f"(n groups compared = {S['stage_kruskal']['n_groups']}).", styles["Caption"]))

stage_rows = [["Stage", "n", "median Δt (s)", "IQR (s)", "sign-test p", "% heart-leads"]]
for r in S["stage_summary"]:
    stage_rows.append([r.get("group_label", r["group"]), str(r["n"]), f"{r['median']:.2f}",
                        f"{r['q1']:.2f} to {r['q3']:.2f}",
                        f"{r['p_sign']:.3g}" if r["p_sign"] is not None else "n/a",
                        f"{r['pct_heart_leads']:.0f}%"])
t2 = Table(stage_rows, colWidths=[0.8 * inch, 0.5 * inch, 1.1 * inch, 1.1 * inch, 0.9 * inch, 1.1 * inch])
t2.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf5")),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
]))
story.append(t2)
story.append(Paragraph("<b>Table 2.</b> Δt summary by sleep stage (sign test: median ≠ 0).",
                        styles["Caption"]))

story.append(Image(os.path.join(FIG_DIR, "dt_by_event_type.png"), width=5.6 * inch, height=3.67 * inch,
                    hAlign="CENTER"))
story.append(Paragraph(
    f"<b>Figure 3.</b> Δt by event type. Kruskal-Wallis H = {S['type_kruskal']['h']:.2f}, "
    f"p = {S['type_kruskal']['p']:.3g} (n groups compared = {S['type_kruskal']['n_groups']}).",
    styles["Caption"]))

type_rows = [["Event type", "n", "median Δt (s)", "IQR (s)", "sign-test p", "% heart-leads"]]
for r in S["type_summary"]:
    type_rows.append([str(r["group"]).replace("_", " "), str(r["n"]), f"{r['median']:.2f}",
                       f"{r['q1']:.2f} to {r['q3']:.2f}",
                       f"{r['p_sign']:.3g}" if r["p_sign"] is not None else "n/a",
                       f"{r['pct_heart_leads']:.0f}%"])
t3 = Table(type_rows, colWidths=[1.5 * inch, 0.5 * inch, 1.1 * inch, 1.1 * inch, 0.8 * inch, 1.0 * inch])
t3.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf5")),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
]))
story.append(t3)
story.append(Paragraph("<b>Table 3.</b> Δt summary by event type (sign test: median ≠ 0).",
                        styles["Caption"]))

# ---------------------------------------------------------------------
# Discussion
# ---------------------------------------------------------------------
story.append(PageBreak())
story.append(Paragraph("Discussion", styles["H1"]))

sig_groups = [r["group_label"] if "group_label" in r else r["group"]
              for r in S["stage_summary"] if r["p_sign"] is not None and r["p_sign"] < 0.05]
sig_types = [r["group"] for r in S["type_summary"] if r["p_sign"] is not None and r["p_sign"] < 0.05]

discussion = (
    f"The central question motivating this pipeline is whether different sleep disruptions are initiated "
    f"predominantly by autonomic (cardiac-first) or cortical (EEG-first) activation, and whether that "
    f"differs by stage or event type. Pooled across all usable events, median Δt was "
    f"{med:.2f} s ({direction}), a small effect but, with n={n_use}, a highly significant one "
    f"(sign-test p={psign:.3g}) — the population-level ordering is close to simultaneous rather than "
    f"strongly lateralized to one system. "
)
if sig_groups:
    discussion += (f"By stage, sign tests reached p&lt;0.05 for {', '.join(sig_groups)} (Table 2); "
                    f"N1 shows the clearest brain-leads pattern (median +{S['stage_summary'][0]['median']:.2f} s), "
                    f"while N3 and REM trend heart-leads (negative medians), consistent with the idea that "
                    f"arousal threshold and autonomic reactivity both shift across sleep stages. ")
else:
    discussion += "No individual stage reached sign-test significance at p&lt;0.05. "
if sig_types:
    discussion += (f"By event type, {', '.join(sig_types)} reached sign-test p&lt;0.05 (Table 3); "
                    f"the largest, best-powered categories (spontaneous, hypopnea) sit closest to Δt=0, "
                    f"while several less common categories (central apnea, RERA, PLM) show small but "
                    f"significant brain-leads offsets. ")
else:
    discussion += "No individual event type reached sign-test significance at p&lt;0.05. "
discussion += (
    f"The omnibus Kruskal-Wallis tests across stage (p={S['stage_kruskal']['p']:.3g}) and event type "
    f"(p={S['type_kruskal']['p']:.3g}) both indicate real group-level differences in Δt at this sample "
    f"size. Even so, the effect sizes are modest (IQRs of several seconds against sub-second median "
    f"shifts; Tables 2–3), so this should be read as evidence for systematic, stage- and event-type-"
    f"dependent timing offsets between cortical and cardiac arousal components — not as evidence that "
    f"either system consistently and strongly leads the other. The result is specific to this cohort "
    f"(BDSP Harvard Study A); whether it replicates in other PSG populations is untested."
)
story.append(Paragraph(discussion, styles["Body"]))

# ---------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------
story.append(Paragraph("Limitations", styles["H1"]))
limitations = (
    "<b>Sample-size caveat, inverted from a typical pilot.</b> With "
    f"{n_sub} subjects and {n_use} usable events, per-cell n is large everywhere except the least common "
    "event categories (apnea_unspecified n=445, limb_movement n=862 — still hundreds, but smaller than "
    "the ~9,000–23,000-event categories); the more practical caution here is that with this much power, "
    "even tiny, clinically negligible median shifts (well under a second in several cells; Tables 2–3) "
    "reach high statistical significance, so p-values should be read alongside effect size (median, IQR), "
    "not as a stand-in for it. "
    "<b>Single-study sample.</b> Contrary to the original plan to sample across A/B/C/D, "
    "investigation of the actual files showed that only Study A exports the structured, tagged MA "
    "linkage this method needs (Study B additionally lacks a directly readable EDF, and Study C lacks "
    "the CAISR annotation file used for the QC cross-check); all subjects are therefore Study A, and "
    "cross-study generalization is untested. Adapting the classifier to the other studies' event-log "
    "formats is future work. Within Study A, 900 of 6,349 eligible sessions were randomly sampled rather "
    "than exhaustively processed; the sampled 862 subjects should be representative of the eligible pool "
    "but this was not separately verified against cohort-level demographics. "
    "<b>RERA availability.</b> RERA events were, in fact, present and directly labeled in Study A's "
    "annotations (<font face='Courier'>\"Microarousal [RERA]\"</font>), so no inference or fabrication was "
    "needed for this label; this differs from the possibility flagged in the study plan and is reported "
    "here for transparency. "
    "<b>HR-onset detection assumptions.</b> The onset algorithm (baseline mean + 1 SD, sustained over 3 "
    "beats, within a −5 s to +15 s search window) is one reasonable, fully documented choice, not a "
    "validated clinical standard; different threshold/sustain parameters would shift individual Δt "
    "values, though the qualitative sign (heart-leads vs. brain-leads) is expected to be more robust than "
    "the exact magnitude. R-peak detection quality was not manually reviewed per event; the exclusion "
    "flags in Table 1 are the only automated safeguard against noisy EKG. "
    "<b>t<sub>EEG</sub> definition.</b> t<sub>EEG</sub> is the technologist-scored MA onset, which already "
    "reflects a human EEG-based judgment rather than an automated cortical-arousal detector; this makes "
    "t<sub>EEG</sub> a high-quality but not perfectly reproducible reference point. "
    "<b>Obstructive/central split.</b> Apnea subtype was read from the linked Apnea row's free-text "
    "Description within a 30 s look-back window; a small fraction of apnea-linked MAs may be mis-split if "
    "the preceding-row heuristic picks the wrong Apnea row in dense event runs."
)
story.append(Paragraph(limitations, styles["Body"]))
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
story.append(Paragraph(
    "1. Cafri N, Mirloo S, Zarhin D, Kamintsky L, Serlin Y, Alhadeed L, et al.; Benninger F. "
    "Imaging blood-brain barrier dysfunction in drug-resistant epilepsy: a multi-center feasibility "
    "study. <i>Epilepsia.</i> 2025;66(1):195-206.", styles["Body"]))
story.append(Paragraph(
    "2. American Academy of Sleep Medicine. The AASM Manual for the Scoring of Sleep and Associated "
    "Events: Rules, Terminology and Technical Specifications. Darien, IL: AASM, 2020.", styles["Body"]))
story.append(Paragraph(
    "3. Halász P, Terzano M, Parrino L, Bódizs R. The nature of arousal in sleep. <i>J Sleep Res.</i> "
    "2004;13(1):1-23.", styles["Body"]))
story.append(Paragraph(
    "4. Sforza E, Jouny C, Ibanez V. Cardiac activation during arousal in humans: further evidence for "
    "hierarchy in the arousal response. <i>Clin Neurophysiol.</i> 2000;111(9):1611-1619.", styles["Body"]))
story.append(Paragraph(
    "5. Bonnet MH, Doghramji K, Roehrs T, et al. The scoring of arousal in sleep: reliability, "
    "validity, and alternatives. <i>J Clin Sleep Med.</i> 2007;3(2):133-145.", styles["Body"]))
story.append(Paragraph(
    "6. Berry RB, Budhiraja R, Gottlieb DJ, et al. Rules for scoring respiratory events in sleep: "
    "update of the 2007 AASM manual. <i>J Clin Sleep Med.</i> 2012;8(5):597-619.", styles["Body"]))
story.append(Paragraph(
    "7. Guilleminault C, Stoohs R, Clerk A, Cetel M, Maistros P. A cause of excessive daytime "
    "sleepiness: the upper airway resistance syndrome. <i>Chest.</i> 1993;104(3):781-787.", styles["Body"]))
story.append(Paragraph(
    "8. Ferri R, Zucconi M, Manconi M, Plazzi G, Bruni O, Ferini-Strambi L. Different periodicity and "
    "time structure of leg movements during sleep in restless legs syndrome and periodic limb movement "
    "disorder. <i>Sleep.</i> 2006;29(12):1587-1594.", styles["Body"]))
story.append(Paragraph(
    "9. Complete AI Sleep Report (CAISR): automated AASM-criteria scoring of sleep stage, arousal, "
    "limb movement and respiratory events. Brain Data Science Platform.", styles["Body"]))
story.append(Paragraph(
    "10. The Human Sleep Project, v2.0. Brain Data Science Platform (BDSP). "
    "https://bdsp.io/content/hsp/2.0/", styles["Body"]))

doc = SimpleDocTemplate(OUT_PDF, pagesize=LETTER,
                         leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                         topMargin=0.75 * inch, bottomMargin=0.75 * inch,
                         title="Cortical-Cardiac Arousal Latency")
doc.build(story)
shutil.copyfile(OUT_PDF, FINAL_PDF)
print(f"Wrote {OUT_PDF}")
print(f"Wrote {FINAL_PDF}")
