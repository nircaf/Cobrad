"""
Assemble the diagnosis-cohort HEP stage-delta alignment paper PDF from
Paper1/diagnosis_alignment_results.json (produced by
Paper1/build_diagnosis_alignment_analysis.py), using reportlab Platypus.
Patterned on Paper1/make_pdf_v2.py's structure and visual style.

  source venv/bin/activate && python3 Paper1/make_diagnosis_alignment_pdf.py
"""
import os
import json

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable,
)

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
OUT_DIR = os.path.join(REPO, "Paper1")
FIG_DIR = os.path.join(OUT_DIR, "figures")
OUT_PDF = os.path.join(OUT_DIR, "Cafri_diagnosis_cohort_alignment_paper.pdf")

with open(os.path.join(OUT_DIR, "diagnosis_alignment_results.json")) as f:
    R = json.load(f)

N_A = R["n_cohort_a"]
N_B_POOL = R["n_cohort_b_pool_6ch"]
DIAG_N = R["diag_n"]
ALIGN = R["alignment_results"]
AGE_OVERALL = R["age_overall"]
AGE = R["age_results"]
PSD_KW = R["psd_kw_table"]
PSD_N = R["psd_n_by_group_stage"]
PSD_FIG = R.get("psd_fig_path")
N_PERM = R["n_perm"]
SEL_CH = R["selected_channels"]

DIAG_GROUPS = [
    "Atrial Fibrillation",
    "Heart Failure",
    "Stroke / Cerebrovascular",
    "Cognitive Impairment / Dementia",
]
DELTA_PAIR_LABELS = ["Light→N3", "N3→REM", "Light→REM"]
ALPHA = 0.05


def fmt_p(p):
    if p is None or (isinstance(p, float) and (p != p)):
        return "n/a"
    return f"{p:.4f}" if p >= 0.0001 else "<0.0001"


def fmt_num(x, nd=2):
    if x is None or (isinstance(x, float) and (x != x)):
        return "n/a"
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------------
# Styles (copied from make_pdf_v2.py)
# ---------------------------------------------------------------------
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
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.3, leading=14.5,
                           alignment=TA_JUSTIFY, spaceAfter=7))
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
    "Does any diagnosis hide an undiagnosed heartbeat-evoked-potential signature? "
    "A stage-delta alignment test across four diagnosis cohorts",
    styles["PaperTitle"]))
story.append(Paragraph(
    "Within-subject sleep-stage HEP amplitude deltas, permutation-tested for alignment "
    "with an undiagnosed referral cohort, plus age and EEG power-spectral comparisons, "
    "in a large clinical polysomnography corpus",
    styles["Subtitle"]))
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
aligned_groups = [g for g in DIAG_GROUPS if ALIGN.get(g, {}).get("q_perm") is not None
                   and ALIGN[g]["q_perm"] == ALIGN[g]["q_perm"] and ALIGN[g]["q_perm"] >= ALPHA]
diverged_groups = [g for g in DIAG_GROUPS if ALIGN.get(g, {}).get("q_perm") is not None
                    and ALIGN[g]["q_perm"] == ALIGN[g]["q_perm"] and ALIGN[g]["q_perm"] < ALPHA]
untestable_groups = [g for g in DIAG_GROUPS if g not in aligned_groups and g not in diverged_groups]

abstract_bg = (
    "Heartbeat-evoked potential (HEP) amplitude follows a canonical within-subject sleep-stage "
    "gradient (light sleep, N3, REM). We ask whether this stage-delta “fingerprint”, measured "
    "in an undiagnosed general-referral cohort, is also present — and therefore hidden — in "
    "patients carrying one of four cardio-/cerebro-/cognitive diagnoses whose recordings use a much "
    "sparser electrode montage."
)
abstract_methods = (
    f"Cohort A ({N_A} patients, no linked ICD-10 diagnosis, \"Susp. Epilepsy\" reference cohort, "
    f"&gt;10 standard 10-20 EEG channels) and Cohort B ({N_B_POOL} diagnosed patients with a sparse "
    f"6-electrode montage, split non-exclusively into Atrial Fibrillation, Heart Failure, "
    f"Stroke/Cerebrovascular, and Cognitive Impairment/Dementia sub-cohorts) were drawn from the same "
    f"Harvard_Electroencephalography clinical polysomnography corpus via the project's existing "
    f"get_group_individuals/select_non_diagnosis_cohort machinery. Per-patient HEP amplitude "
    f"(mean of the channel-averaged trace in the 0.15-0.5 s post-R-peak window) was computed per sleep "
    f"stage, and within-subject stage-to-stage deltas (Light&rarr;N3, N3&rarr;REM, Light&rarr;REM) were "
    f"reduced to a per-cohort mean delta vector. Each Cohort-B sub-cohort's vector was compared to "
    f"Cohort A's via Euclidean distance, Pearson correlation, cosine similarity, and a "
    f"{N_PERM}-permutation label-shuffle test (BH-FDR corrected across the 4 sub-cohorts). Age "
    f"(Mann-Whitney U, BH-FDR corrected) and EEG band-power PSD (Welch, Kruskal-Wallis per band"
    f"&times;stage, BH-FDR corrected) were compared the same way."
)
if aligned_groups:
    align_sentence = (
        f"{', '.join(aligned_groups)} showed a stage-delta vector statistically indistinguishable "
        f"from Cohort A's (BH-q &ge; {ALPHA})"
        + (f", while {', '.join(diverged_groups)} diverged significantly (BH-q &lt; {ALPHA})"
           if diverged_groups else "") + "."
    )
elif diverged_groups:
    align_sentence = f"All testable sub-cohorts ({', '.join(diverged_groups)}) diverged significantly from Cohort A's stage-delta pattern (BH-q &lt; {ALPHA})."
else:
    align_sentence = "No sub-cohort had sufficient matched-stage N for a reliable alignment test."
abstract_results = (
    f"{align_sentence} Age differed between Cohort A and Cohort B overall "
    f"(median {fmt_num(AGE_OVERALL['median1'],1)} vs {fmt_num(AGE_OVERALL['median2'],1)} years, "
    f"p={fmt_p(AGE_OVERALL['p'])})."
)
abstract_conclusions = (
    "A stage-delta vector that is statistically indistinguishable from the undiagnosed reference "
    "cohort's is consistent with (but does not prove) an undiagnosed/subclinical process sharing the "
    "reference cohort's autonomic-cortical coupling profile; a vector that diverges significantly is "
    "evidence the diagnosis measurably alters this coupling. Given the small sparse-montage sub-cohort "
    "sizes, these findings should be treated as hypothesis-generating."
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
    "<b>Keywords:</b> heartbeat-evoked potential; interoception; sleep stage; diagnosis cohort; "
    "permutation test; BH-FDR; electroencephalography power spectral density; polysomnography",
    styles["Kw"]))
story.append(PageBreak())

# ---------------------------------------------------------------------
# Introduction
# ---------------------------------------------------------------------
story.append(Paragraph("1. Introduction", styles["H1"]))
story.append(Paragraph(
    "The heartbeat-evoked potential (HEP) is an R-peak-locked scalp EEG deflection reflecting cortical "
    "processing of cardiac afferent signals, and its amplitude follows a reproducible within-subject "
    "sleep-stage gradient. A companion analysis of this same corpus established that gradient at scale "
    "in an undiagnosed general-referral (\"Susp. Epilepsy\") cohort. Separately, a large fraction of this "
    "corpus's diagnosed patients &mdash; carrying cardiovascular, cerebrovascular, or cognitive "
    "diagnoses &mdash; were recorded with a much sparser (~6-electrode) montage rather than the full "
    "10-20 clinical PSG cap, precluding the topographically-resolved cluster-permutation approach used "
    "for the extended-montage cohort.",
    styles["Body"]))
story.append(Paragraph(
    "Here we ask a narrower, more directly interpretable question with that sparse-montage data: does "
    "any of four diagnosis groups &mdash; Atrial Fibrillation, Heart Failure, Stroke/Cerebrovascular, and "
    "Cognitive Impairment/Dementia &mdash; show a within-subject HEP stage-delta \"fingerprint\" (the "
    "vector of amplitude changes between light sleep, N3 and REM) that is statistically indistinguishable "
    "from the undiagnosed reference cohort's, versus one that diverges significantly? A sub-cohort whose "
    "pattern is aligned with the reference cohort's is one candidate way an underlying "
    "autonomic-cortical coupling profile could look similar to an undiagnosed patient's despite a "
    "different clinical diagnosis; a diverging sub-cohort indicates the diagnosis measurably perturbs "
    "this coupling. We pair the stage-delta alignment test with two supporting comparisons on the same "
    "cohorts: age distribution, and resting EEG band-power (PSD) by sleep stage.",
    styles["Body"]))

# ---------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------
story.append(Paragraph("2. Methods", styles["H1"]))

story.append(Paragraph("2.1 Cohorts and data source", styles["H2"]))
story.append(Paragraph(
    f"Data were drawn from the Harvard_Electroencephalography group of the project's existing clinical "
    f"polysomnography corpus, loaded via the dashboard's unmodified "
    f"<font face='Courier'>get_group_individuals</font>/<font face='Courier'>load_patient_data</font> "
    f"caches (no cache regeneration). <b>Cohort A</b> (undiagnosed reference): patients with no linked "
    f"ICD-10 diagnosis in the EHR (<font face='Courier'>select_non_diagnosis_cohort</font>, the "
    f"\"Susp. Epilepsy\" cohort used throughout this project) and &gt;10 standard 10-20 EEG channels "
    f"per the existing <font face='Courier'>individuals_cache_min11eeg_*</font> cache "
    f"(&ge;11 via <font face='Courier'>_standard_1020_channel_count</font>, i.e. strictly &gt;10): "
    f"n = {N_A} unique patients. <b>Cohort B</b> (diagnosed, sparse montage): the full unfiltered patient "
    f"pool was loaded and each recording's exact standard-10-20 channel count was computed with the same "
    f"function; patients with exactly 6 such channels (the observed sparse-montage cluster, see &sect;3.1) "
    f"were retained, n = {N_B_POOL} unique 6-electrode patients pooled across the 4 diagnosis groups. "
    f"Membership in each of the 4 diagnosis groups (Atrial Fibrillation, Heart Failure, "
    f"Stroke/Cerebrovascular, Cognitive Impairment/Dementia) is non-exclusive, following "
    f"23_brain_body_synchrony_dysrhythmia_dashboard.py's <font face='Courier'>build_grouped_df</font> "
    f"convention: a patient with two of the four diagnoses appears in both sub-cohorts.",
    styles["Body"]))

t_cohort_rows = [["Cohort", "n (unique patients)"]]
t_cohort_rows.append(["Cohort A (Susp. Epilepsy, undiagnosed, >10 EEG ch)", str(N_A)])
t_cohort_rows.append(["Cohort B pool (diagnosed, 6-electrode montage)", str(N_B_POOL)])
for g in DIAG_GROUPS:
    t_cohort_rows.append([f"  • {g}", str(DIAG_N.get(g, "n/a"))])
tblc = Table(t_cohort_rows, colWidths=[4.6 * inch, 1.6 * inch])
tblc.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4e4e4")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.6),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(tblc)
story.append(Paragraph(
    "<b>Table 1.</b> Cohort sizes after all filters (electrode-count and diagnosis-category membership). "
    "Diagnosis sub-cohort membership is non-exclusive; sub-cohort n's need not sum to the Cohort B pool.",
    styles["Caption"]))

story.append(Paragraph("2.2 HEP amplitude and stage-delta vectors", styles["H2"]))
story.append(Paragraph(
    f"Per-patient, per-stage HEP amplitude was the mean of the channel-averaged HEP trace "
    f"(<font face='Courier'>mod16._patient_hep_trace</font>, over the union of "
    f"{len(SEL_CH)} available 10-20 channels across both cohorts) inside the canonical post-R-peak "
    f"analysis window (0.15-0.5 s), exactly as computed by "
    f"23_brain_body_synchrony_dysrhythmia_dashboard.py's <font face='Courier'>compute_amplitudes</font>. "
    f"For each cohort/sub-cohort, patients with a usable amplitude in all three stages contributed a "
    f"3-element within-subject delta vector (Light&rarr;N3, N3&rarr;REM, Light&rarr;REM), and the "
    f"cohort/sub-cohort's mean delta vector is the elementwise mean across its patients.",
    styles["Body"]))

story.append(Paragraph("2.3 Alignment test", styles["H2"]))
story.append(Paragraph(
    f"For each Cohort-B diagnosis sub-cohort we computed: (i) Pearson correlation and cosine similarity "
    f"between its mean delta vector and Cohort A's (3 stage-pair elements); (ii) Euclidean distance "
    f"between the two mean vectors; and (iii) a {N_PERM}-permutation label-shuffle test &mdash; the "
    f"sub-cohort's and Cohort A's matched-3-stage patients were pooled, group labels were shuffled "
    f"{N_PERM} times preserving the original group sizes, and the Euclidean distance between the two "
    f"resulting group means was recomputed each time to build a null distribution. The permutation "
    f"p-value is the fraction of null distances at least as large as the observed distance: a sub-cohort "
    f"whose observed distance to Cohort A is no larger than expected under this null is \"aligned\" with "
    f"Cohort A's stage-delta pattern; one whose distance is significantly larger \"diverges\". Raw "
    f"p-values were BH-FDR corrected (<font face='Courier'>mod16.benjamini_hochberg</font>) across the "
    f"family of 4 sub-cohort tests; significance is judged on q &lt; {ALPHA}, not raw p.",
    styles["Body"]))

story.append(Paragraph("2.4 Age and PSD comparisons", styles["H2"]))
story.append(Paragraph(
    "Age was extracted per patient via the dashboard's existing clinical/EHR demographic lookup "
    "(<font face='Courier'>mod16._patient_demographics</font>) and compared with a two-sided "
    "Mann-Whitney U test: Cohort A vs Cohort B overall (reported as a single descriptive comparison, not "
    "part of the corrected family), and Cohort A vs each of the 4 diagnosis sub-cohorts (BH-FDR corrected "
    "across those 4 tests). EEG power-spectral density was computed per sampled recording by loading the "
    "patient's raw per-stage pickle, applying the project's existing "
    "<font face='Courier'>drop_non_eeg_channels</font>/<font face='Courier'>process_file_data</font> "
    "cleaning pipeline, and running Welch's method "
    "(<font face='Courier'>scipy.signal.welch</font>, matching 6_hep_group_comparison.py's own PSD usage) "
    "on the continuous cleaned EEG, channel-averaged; band power (delta 0.5-4, theta 4-8, alpha 8-12, "
    f"beta 12-30 Hz) was the trapezoidal-rule integral of the mean PSD over each band, matching "
    f"16_diagnosis_sleep_stage_comparison_dashboard.py's own band-power convention. To bound compute, up "
    f"to {R['psd_max_per_group_stage']} recordings per group×stage cell were sampled (highest "
    f"R-peak count first, following the dashboard's own "
    f"<font face='Courier'>_hr_psd_candidate_rows</font>/sampling convention). Band power was compared "
    f"across the 5 groups (Cohort A + 4 diagnoses) per stage with Kruskal-Wallis, BH-FDR corrected across "
    f"the full band&times;stage family (4 bands &times; 3 stages = 12 tests).",
    styles["Body"]))

# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------
story.append(Paragraph("3. Results", styles["H1"]))

story.append(Paragraph("3.1 Headline finding: stage-delta alignment", styles["H2"]))
t_align_rows = [["Sub-cohort", "N (matched 3-stage)", "Distance to A", "Pearson r", "Cosine sim", "q_perm", "Verdict (q<0.05)"]]
for g in DIAG_GROUPS:
    a = ALIGN.get(g, {})
    q = a.get("q_perm")
    if q is None or q != q:
        verdict = "not testable"
    elif q < ALPHA:
        verdict = "DIVERGES"
    else:
        verdict = "ALIGNED"
    t_align_rows.append([
        g, str(a.get("n_sub", "n/a")),
        fmt_num(a.get("obs_distance")), fmt_num(a.get("pearson_r")),
        fmt_num(a.get("cosine_sim")), fmt_p(q), verdict,
    ])
tbla = Table(t_align_rows, colWidths=[1.75 * inch, 0.85 * inch, 0.85 * inch, 0.7 * inch, 0.75 * inch, 0.65 * inch, 0.85 * inch])
tbla.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4e4e4")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 7.6),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(tbla)
story.append(Paragraph(
    f"<b>Table 2.</b> Stage-delta alignment of each Cohort-B diagnosis sub-cohort against Cohort A "
    f"(Cohort A matched-3-stage N reported in text). Distance = Euclidean distance between mean 3-element "
    f"stage-delta vectors (&mu;V). q_perm = BH-FDR corrected {N_PERM}-permutation label-shuffle p-value. "
    f"\"ALIGNED\" = not significantly different from Cohort A's pattern (q &ge; 0.05); \"DIVERGES\" = "
    f"significantly different (q &lt; 0.05).",
    styles["Caption"]))

n_a_matched = ALIGN.get(DIAG_GROUPS[0], {}).get("n_a", "n/a")
story.append(Paragraph(
    f"Cohort A contributed {n_a_matched} patients with a usable HEP amplitude in all three sleep stages "
    f"for the alignment test. {align_sentence} "
    + (f"Of the tested sub-cohorts, the closest match to Cohort A's pattern was "
       f"{min((g for g in DIAG_GROUPS if ALIGN.get(g,{}).get('obs_distance') is not None and ALIGN[g]['obs_distance']==ALIGN[g]['obs_distance']), key=lambda g: ALIGN[g]['obs_distance'], default='n/a')}"
       f", and the most divergent was "
       f"{max((g for g in DIAG_GROUPS if ALIGN.get(g,{}).get('obs_distance') is not None and ALIGN[g]['obs_distance']==ALIGN[g]['obs_distance']), key=lambda g: ALIGN[g]['obs_distance'], default='n/a')}."
       if any(ALIGN.get(g, {}).get("obs_distance") == ALIGN.get(g, {}).get("obs_distance") for g in DIAG_GROUPS) else ""),
    styles["Body"]))

story.append(Paragraph("3.2 Age comparison", styles["H2"]))
t_age_rows = [["Comparison", "N (A)", "N (other)", "Median age A", "Median age other", "p", "q"]]
t_age_rows.append([
    "Cohort A vs Cohort B (overall)", str(AGE_OVERALL["n1"]), str(AGE_OVERALL["n2"]),
    fmt_num(AGE_OVERALL["median1"], 1), fmt_num(AGE_OVERALL["median2"], 1),
    fmt_p(AGE_OVERALL["p"]), "— (single test)",
])
for g in DIAG_GROUPS:
    a = AGE.get(g, {})
    t_age_rows.append([
        f"Cohort A vs {g}", str(a.get("n1", "n/a")), str(a.get("n2", "n/a")),
        fmt_num(a.get("median1"), 1), fmt_num(a.get("median2"), 1),
        fmt_p(a.get("p")), fmt_p(a.get("q")),
    ])
tblage = Table(t_age_rows, colWidths=[2.0 * inch, 0.6 * inch, 0.6 * inch, 0.95 * inch, 1.0 * inch, 0.6 * inch, 0.85 * inch])
tblage.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4e4e4")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 7.6),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(tblage)
story.append(Paragraph(
    "<b>Table 3.</b> Age (Mann-Whitney U, two-sided). q-values BH-FDR corrected across the 4 "
    "Cohort-A-vs-diagnosis-sub-cohort tests only; the overall Cohort A vs Cohort B comparison is reported "
    "as a single uncorrected descriptive test.", styles["Caption"]))

story.append(Paragraph("3.3 PSD / EEG band power", styles["H2"]))
if PSD_KW:
    sig_rows = [r for r in PSD_KW if r.get("q_value") is not None and r["q_value"] == r["q_value"] and r["q_value"] < ALPHA]
    story.append(Paragraph(
        f"Across the 4 bands &times; 3 stages tested (12 Kruskal-Wallis tests, BH-FDR corrected), "
        f"{len(sig_rows)} of {len(PSD_KW)} reached q &lt; {ALPHA}"
        + (": " + "; ".join(f"{r['band']}/{r['stage']} (H={fmt_num(r['H_stat'])}, q={fmt_p(r['q_value'])})" for r in sig_rows) + "."
           if sig_rows else ", i.e. no band×stage cell showed a significant 5-group difference after correction."),
        styles["Body"]))
    t_psd_rows = [["Band", "Stage", "H", "p", "q"]]
    for r in PSD_KW:
        t_psd_rows.append([r["band"], r["stage"], fmt_num(r["H_stat"]), fmt_p(r["p_value"]), fmt_p(r["q_value"])])
    tblpsd = Table(t_psd_rows, colWidths=[1.0 * inch, 1.2 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch])
    tblpsd.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4e4e4")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(tblpsd)
    story.append(Paragraph(
        "<b>Table 4.</b> Kruskal-Wallis test of EEG band power across the 5 groups (Cohort A + 4 "
        "diagnosis sub-cohorts), per band and sleep stage. q BH-FDR corrected across all 12 tests.",
        styles["Caption"]))
    if PSD_FIG and os.path.exists(PSD_FIG):
        story.append(Image(PSD_FIG, width=6.6 * inch, height=6.6 * inch * (4.2 / 15)))
        story.append(Paragraph(
            "<b>Figure 1.</b> Mean channel-averaged EEG power spectral density (Welch, log10 power) by "
            "group and sleep stage, 0-45 Hz.", styles["Caption"]))
else:
    story.append(Paragraph(
        "The PSD comparison was infeasible with the recordings actually available for the sampled "
        "sub-cohorts (no usable Welch PSD could be computed for any sampled recording) and is reported "
        "here as a limitation rather than a fabricated result (see &sect;5).",
        styles["Body"]))

# ---------------------------------------------------------------------
# Discussion
# ---------------------------------------------------------------------
story.append(Paragraph("4. Discussion", styles["H1"]))
story.append(Paragraph(
    align_sentence + " Interpreted alongside the age and PSD comparisons in the same cohorts, this "
    "pattern should be read as a description of which diagnosis groups' autonomic-cortical coupling "
    "profile, measured through this within-subject stage-delta lens, is or is not distinguishable from "
    "an undiagnosed referral population's &mdash; not as a claim about diagnostic sensitivity or "
    "specificity, given the modest sparse-montage sub-cohort sizes.",
    styles["Body"]))

story.append(Paragraph("Limitations", styles["H2"]))
story.append(Paragraph(
    f"<b>Small sparse-montage sub-cohorts.</b> The 6-electrode diagnosed sub-cohorts are markedly "
    f"smaller than Cohort A (see Table 1), which widens the alignment permutation test's null "
    f"distribution and reduces power to detect a true divergence; an \"ALIGNED\" verdict should be read "
    f"as \"not distinguishable given the available N,\" not as proof of equivalence. "
    f"<b>Non-exclusive diagnosis groups.</b> Patients with multiple qualifying diagnoses contribute to "
    f"more than one sub-cohort, so the 4 sub-cohort tests are not statistically independent; this is the "
    f"same convention used by 23_brain_body_synchrony_dysrhythmia_dashboard.py and is retained for "
    f"consistency, but should be kept in mind when comparing sub-cohort verdicts to each other. "
    f"<b>Only 3 stage-delta elements.</b> The alignment vectors have only 3 dimensions (light→N3, "
    f"N3→REM, light→REM), so the Pearson correlation reported alongside the permutation test is "
    f"statistically under-powered (n=3) and included only as a descriptive complement to the "
    f"permutation-based significance call, which is the analysis's primary evidence. "
    f"<b>Electrode-count filter is exact, not a montage identity check.</b> The 6-electrode Cohort-B "
    f"filter selects recordings with exactly 6 standard 10-20 channel names present; it does not verify "
    f"that all such patients used literally the same physical montage. <b>PSD sampling.</b> To bound "
    f"compute, PSD band power was computed on a capped, R-peak-count-ranked sample per group×stage "
    f"cell rather than every patient (see &sect;2.4); this could bias band-power estimates toward "
    f"recordings with cleaner ECG (and possibly cleaner EEG).",
    styles["Body"]))

story.append(Paragraph("Data availability statement", styles["H2"]))
story.append(Paragraph(
    "Data were derived from the same de-identified, credentialed-access clinical polysomnography corpus "
    "used throughout this project, distributed under credentialed access via the Brain Data Science "
    "Platform. All statistics and figures were produced by calling the project's existing dashboard "
    "modules (6_hep_group_comparison.py, 16_diagnosis_sleep_stage_comparison_dashboard.py) directly, "
    "following 23_brain_body_synchrony_dysrhythmia_dashboard.py's cohort-construction pattern; the "
    "computation script (Paper1/build_diagnosis_alignment_analysis.py) and PDF assembly script "
    "(Paper1/make_diagnosis_alignment_pdf.py) accompany this paper in the project repository.",
    styles["Body"]))

doc = SimpleDocTemplate(
    OUT_PDF, pagesize=LETTER,
    topMargin=0.8 * inch, bottomMargin=0.8 * inch,
    leftMargin=0.9 * inch, rightMargin=0.9 * inch,
    title="Does any diagnosis hide an undiagnosed heartbeat-evoked-potential signature?",
    author="Nir Cafri",
)


def _add_page_number(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(LETTER[0] / 2, 0.55 * inch, str(doc_.page))
    canvas.restoreState()


doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
print("Wrote", OUT_PDF, os.path.getsize(OUT_PDF), "bytes")
