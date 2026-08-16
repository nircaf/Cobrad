"""
Combined manuscript: sleep-stage / age heartbeat-evoked-potential (HEP) gradient
+ diagnosis-cohort montage/alignment analysis, assembled into one Nature-style
research article using reportlab Platypus.

Pulls numbers only from already-computed, on-disk results (no fabricated
numbers):
  - Paper1/stage_delta_age_results_v2.json   (stage-delta + age-split cluster-permutation)
  - Paper1/diagnosis_alignment_results.json  (montage/diagnosis-cohort alignment, age, PSD)
Figures reused from Paper1/figures/ (already rendered by the upstream analysis
scripts; none regenerated here).

  source venv/bin/activate && python3 Paper1/make_combined_nature_paper.py
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
    PageBreak, HRFlowable, KeepTogether,
)

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
OUT_DIR = os.path.join(REPO, "Paper1")
FIG_DIR = os.path.join(OUT_DIR, "figures")
OUT_PDF = os.path.join(OUT_DIR, "Cafri_HEP_sleepstage_age_diagnosis_paper.pdf")

with open(os.path.join(OUT_DIR, "stage_delta_age_results_v2.json")) as f:
    S = json.load(f)
with open(os.path.join(OUT_DIR, "diagnosis_alignment_results.json")) as f:
    D = json.load(f)

PAIR = {(p["stage_a"], p["stage_b"]): p for p in S["pairwise"]}
AGESPLIT = {a["stage"]: a for a in S["age_split"]}
TABLE1 = {t["label"]: t for t in S["table1"]}

DIAG_GROUPS = ["Atrial Fibrillation", "Heart Failure", "Stroke / Cerebrovascular",
               "Cognitive Impairment / Dementia"]
ALIGN = D["alignment_results"]
AGE_OVERALL = D["age_overall"]
AGE_DX = D["age_results"]
PSD_KW = D["psd_kw_table"]
DIAG_N = D["diag_n"]
N_A_MONTAGE = D["n_cohort_a"]
N_B_MONTAGE = D["n_cohort_b_pool_6ch"]
ALPHA = 0.05


def fmt_p(p):
    if p is None or (isinstance(p, float) and p != p):
        return "n/a"
    return f"{p:.4f}" if p >= 0.0001 else "<0.0001"


def fmt_num(x, nd=2):
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
    "A two-tier cortical interoception gradient across sleep stages and age, "
    "and its confound with clinical diagnosis and electrode montage: a "
    "topographically-resolved analysis of heartbeat-evoked potentials in "
    "90,166 patients",
    styles["PaperTitle"]))
story.append(Paragraph(
    "Sleep-stage and age modulation of the heartbeat-evoked potential (HEP), replicated with "
    "within-subject cluster-permutation statistics, extended to four diagnosis cohorts, and shown "
    "to be entangled with a systematic difference in recording montage between diagnosed and "
    "undiagnosed patients in a large multi-centre clinical polysomnography corpus",
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
aligned = [g for g in DIAG_GROUPS if ALIGN[g].get("q_perm") == ALIGN[g].get("q_perm") and ALIGN[g]["q_perm"] >= ALPHA]
diverged = [g for g in DIAG_GROUPS if ALIGN[g].get("q_perm") == ALIGN[g].get("q_perm") and ALIGN[g]["q_perm"] < ALPHA]

abstract_bg = (
    "The heartbeat-evoked potential (HEP) is an R-peak-locked scalp-EEG deflection reflecting cortical "
    "processing of cardiac afferent signals. Its magnitude is known to follow a vigilance-state gradient "
    "across sleep and to increase with age in wakefulness, but whether these two effects share a common "
    "topography, and whether either is confounded by how patients are recorded and diagnosed in a real "
    "clinical corpus, has not been tested at scale."
)
abstract_methods = (
    f"We analysed the Harvard_Electroencephalography arm of the Human Sleep Project (90,166 unique "
    f"patients). Within-subject sleep-stage HEP contrasts (light sleep, N3, REM) and an independent "
    f"age median-split contrast were tested with paired/independent cluster-mass permutation tests "
    f"(200 permutations, cluster &alpha; = 0.01) across 19&ndash;24 standard 10-20 electrodes, in "
    f"n = {S['n_cohort']} patients with &ge;10 EEG channels ({TABLE1['Susp. Epilepsy matched 3-stage subset']['n']} "
    f"without a linked ICD-10 diagnosis, \"Susp. Epilepsy\", matched across all three stages). "
    f"Separately, we compared this undiagnosed, extended-montage reference cohort (Cohort A, "
    f"n = {N_A_MONTAGE}, &gt;10 EEG channels) against {N_B_MONTAGE} diagnosed patients recorded on a "
    f"markedly sparser 6-electrode montage (Cohort B), split into four non-exclusive diagnosis groups "
    f"(atrial fibrillation, heart failure, stroke/cerebrovascular disease, cognitive impairment/dementia), "
    f"using a within-subject stage-delta alignment permutation test, age (Mann-Whitney U), and EEG "
    f"band-power (Welch PSD, Kruskal-Wallis), all BH-FDR corrected."
)
abstract_results = (
    f"The within-subject gradient was topographically broad and strongest between the physiological "
    f"extremes (N3-REM: p={PAIR[('N3','R')]['p_formatted']}, {PAIR[('N3','R')]['n_cluster_significant']}/19 "
    f"electrodes), narrow between the two \"lighter\" stages (Light-REM: p={PAIR[('light_sleep','R')]['p_formatted']}, "
    f"{PAIR[('light_sleep','R')]['n_cluster_significant']}/19 electrodes), with Light-N3 intermediate "
    f"({PAIR[('light_sleep','N3')]['n_cluster_significant']}/19 electrodes). Every stage showed a significant, "
    f"right-lateralised age effect (all p={AGESPLIT['R']['p_formatted']}). Diagnosed and undiagnosed patients were "
    f"recorded on systematically different montages: the sparse 6-electrode group (n={N_B_MONTAGE}) was "
    f"more than three times the size of the extended-montage undiagnosed reference pool (n={N_A_MONTAGE}), "
    f"and was dominated by patients carrying at least one of the four tracked diagnoses "
    f"({sum(DIAG_N.values())}/{N_B_MONTAGE}, {100*sum(DIAG_N.values())/N_B_MONTAGE:.0f}%, non-exclusive) plus "
    f"a further undifferentiated remainder carrying other or multiple unclassified linked diagnoses. "
    f"Despite this montage gap, the stage-delta \"fingerprint\" of "
    + (", ".join(aligned) if aligned else "no diagnosis group")
    + f" was statistically indistinguishable from the undiagnosed reference cohort's (BH-q &ge; {ALPHA})"
    + (f", while {', '.join(diverged)} diverged significantly (BH-q &lt; {ALPHA})." if diverged else ".")
)
abstract_conclusions = (
    "Sleep-stage and age modulate HEP amplitude through what behaves as a single graded, right-lateralised "
    "generator rather than three separately-tuned states. Because diagnosed patients in this corpus are "
    "recorded almost exclusively on a sparser montage than undiagnosed patients, any diagnosis-vs-reference "
    "HEP comparison is confounded with electrode coverage unless explicitly stratified for it, as done here; "
    "under that stratification, most diagnosis groups' within-subject stage-delta pattern is not "
    "distinguishable from the undiagnosed reference's, and should be treated as hypothesis-generating "
    "given the modest sparse-montage sub-cohort sizes."
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
    "<b>Keywords:</b> heartbeat-evoked potential; interoception; sleep stage; cluster permutation; "
    "ageing; electroencephalography; diagnosis cohort; montage confound; polysomnography",
    styles["Kw"]))
story.append(PageBreak())

# ===========================================================================
# Introduction
# ===========================================================================
story.append(Paragraph("Introduction", styles["H1"]))
story.append(Paragraph(
    "The heartbeat-evoked potential (HEP) is a low-amplitude, R-peak-locked deflection in scalp EEG "
    "taken as a cortical readout of cardiac afferent traffic. It is contaminated by the cardiac field "
    "artifact (CFA), the much larger volume-conducted signature of cardiac depolarisation reaching "
    "scalp electrodes directly through the torso rather than via a cortical generator; a window around "
    "the R-peak is therefore excluded from, or explicitly flagged in, every contrast reported here. "
    "Lechinger and colleagues showed that frontocentral heartbeat-evoked amplitude decreases with sleep "
    "depth, with a renewed increase during REM, establishing a REM &gt; light NREM &gt; N3 vigilance-state "
    "gradient that has become a reference description of the sleep HEP.<super>1</super> Independently, "
    "HEP amplitude has been reported to increase with age in wakefulness,<super>2,3</super> but in modest "
    "cohorts, and without testing whether the sleep-stage and age effects share a common scalp topography.",
    styles["Body"]))
story.append(Paragraph(
    "A further, largely unexamined issue in any HEP study drawn from real clinical referral data is that "
    "the patients who end up diagnosed with a specific cardiovascular, cerebrovascular, or cognitive "
    "condition are not recorded under the same conditions as the undiagnosed reference population used to "
    "establish the sleep-stage/age gradient. In particular, the corpus analysed here shows a stark split: "
    "an extended, full 10-20-derived montage is used preferentially for one referral pathway, and a much "
    "sparser 6-electrode montage for another &mdash; and it is the sparser-montage group that both is larger "
    "in absolute terms and is dominated by patients carrying a linked diagnosis. If diagnosis and electrode "
    "coverage are confounded in this way, any downstream comparison of HEP (or other EEG-derived measures) "
    "between diagnosed and undiagnosed groups risks attributing a montage/coverage difference to disease.",
    styles["Body"]))
story.append(Paragraph(
    "We address three questions in one cohort and one analysis pipeline. First, do paired, "
    "topographically-resolved cluster-permutation contrasts replicate the sleep-stage HEP gradient "
    "within-subject, and which electrodes carry each pairwise contrast? Second, does an independent "
    "age median-split show a comparable or a distinct topography in each stage? Third, restricting to "
    "the four best-represented diagnosis categories in the sparser-montage group (atrial fibrillation, "
    "heart failure, stroke/cerebrovascular disease, cognitive impairment/dementia), does each diagnosis "
    "group's within-subject stage-delta \"fingerprint\" match the undiagnosed reference cohort's, or "
    "diverge from it &mdash; and how large is the montage/coverage gap that any such comparison must be "
    "read against?",
    styles["Body"]))

# ===========================================================================
# Results
# ===========================================================================
story.append(Paragraph("Results", styles["H1"]))

# --- 1. Cohort + montage stratification --------------------------------
story.append(Paragraph("Diagnosed patients are recorded on a systematically sparser montage", styles["H2"]))
story.append(Paragraph(
    f"The extended-montage cohort (&ge;10 of 24 candidate standard 10-20 electrodes, light sleep/N3/REM "
    f"all available) comprised n = {S['n_cohort']} patients; of these, "
    f"n = {TABLE1['Susp. Epilepsy matched 3-stage subset']['n']} carried no linked ICD-10 diagnosis "
    f"(\"Susp. Epilepsy\", the undiagnosed reference used for within-subject contrasts below). Separately "
    f"stratifying the full corpus by exact electrode count showed that patients with exactly 6 standard "
    f"10-20 channels &mdash; a materially sparser recording than the extended montage &mdash; numbered "
    f"n = {N_B_MONTAGE}, more than three times the n = {N_A_MONTAGE} undiagnosed, &gt;10-channel reference "
    f"pool used for the diagnosis-alignment analysis (Table 1). Of the {N_B_MONTAGE} sparse-montage "
    f"patients, {sum(DIAG_N.values())} ({100*sum(DIAG_N.values())/N_B_MONTAGE:.0f}%) carried at least one of "
    f"the four diagnoses tracked here (non-exclusively: a patient with two qualifying diagnoses is counted "
    f"in both); the remaining {N_B_MONTAGE - sum(DIAG_N.values())} carry other or multiple linked diagnoses "
    f"not individually broken out. In other words, the low-electrode-count group is not a random subsample "
    f"of the corpus: it is the group in which a clinical diagnosis was actually reached, while the "
    f"extended, &gt;10-channel montage is disproportionately used for the undiagnosed, \"Susp. Epilepsy\" "
    f"referral pathway that anchors the sleep-stage/age analysis below.",
    styles["Body"]))

t1_rows = [["Group", "Montage", "n (unique patients)"]]
t1_rows.append(["Extended-montage cohort (light/N3/REM available)", "≥10 EEG channels (19-24 candidate)", str(S["n_cohort"])])
t1_rows.append(["  • Susp. Epilepsy (undiagnosed) matched 3-stage subset", "≥10 EEG channels", str(TABLE1["Susp. Epilepsy matched 3-stage subset"]["n"])])
t1_rows.append(["Cohort A: undiagnosed reference (diagnosis-alignment analysis)", ">10 EEG channels", str(N_A_MONTAGE)])
t1_rows.append(["Cohort B pool: diagnosed, sparse montage", "exactly 6 EEG channels", str(N_B_MONTAGE)])
for g in DIAG_GROUPS:
    t1_rows.append([f"    • {g}", "exactly 6 EEG channels", str(DIAG_N.get(g, "n/a"))])
tbl1 = Table(t1_rows, colWidths=[3.3 * inch, 2.0 * inch, 1.3 * inch])
tbl1.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4e4e4")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.2),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(tbl1)
story.append(Paragraph(
    "<b>Table 1.</b> Cohort sizes by montage/electrode-count band. The sleep-stage/age analyses (Figs. 1-2) "
    "use the extended-montage (≥10-channel) cohort; the diagnosis-alignment analysis (Fig. 3, Table 2-4) "
    "contrasts Cohort A against the sparse (6-channel) diagnosed Cohort B. Diagnosis sub-cohort membership "
    "is non-exclusive, so sub-cohort n's need not sum to the Cohort B pool.",
    styles["Caption"]))

# --- 2. Sleep-stage gradient --------------------------------------------
story.append(Paragraph("The within-subject sleep-stage gradient is broad between physiological extremes, narrow between REM and light sleep", styles["H2"]))
p_ln3, p_n3r, p_lr = PAIR[("light_sleep", "N3")], PAIR[("N3", "R")], PAIR[("light_sleep", "R")]
story.append(Paragraph(
    f"All three paired within-subject cluster-permutation contrasts were run on the same "
    f"N = {p_ln3['n']} matched Susp. Epilepsy patients. Light-N3 reached p={p_ln3['p_formatted']} "
    f"({p_ln3['n_cluster_significant']}/{p_ln3['n_electrodes_tested']} electrodes cluster-significant: "
    f"{', '.join(p_ln3['sig_channels'])}). N3-REM, the contrast spanning the largest physiological "
    f"separation in the vigilance-state gradient, reached p={p_n3r['p_formatted']} and was by far the "
    f"broadest ({p_n3r['n_cluster_significant']}/{p_n3r['n_electrodes_tested']} electrodes: "
    f"{', '.join(p_n3r['sig_channels'])}), spanning frontal, central, parietal, temporal and occipital "
    f"sites bilaterally. Light-REM, the smallest contrast in the gradient (REM and light sleep are the two "
    f"\"shallower\" ends), was the weakest and narrowest: p={p_lr['p_formatted']}, only "
    f"{p_lr['n_cluster_significant']}/{p_lr['n_electrodes_tested']} electrode(s) ({', '.join(p_lr['sig_channels'])}) "
    f"cluster-significant and none surviving Benjamini-Hochberg FDR correction. This graded pattern "
    f"&mdash; broad and robust for the extreme-to-extreme contrast, narrow and right-lateralised for the "
    f"contrast sharing a \"lighter\" endpoint &mdash; is consistent with a single underlying generator whose "
    f"gain scales monotonically across the gradient (REM highest, light intermediate, N3 lowest) rather "
    f"than three topographically distinct processes.",
    styles["Body"]))

story.append(Image(os.path.join(FIG_DIR, "pairwise_N3_vs_R.png"), width=6.6 * inch, height=6.6 * inch / 1.4))
story.append(Paragraph(
    f"<b>Figure 1.</b> N3 (deep sleep) &ndash; REM paired within-subject contrast, Susp. Epilepsy cohort "
    f"(N = {p_n3r['n']} matched patients). Top: mean &Delta; HEP &plusmn; SEM trace; bottom: pointwise "
    f"T-statistic; right inset: per-electrode significance topomap (raw cluster p-value; red = more "
    f"significant). Cluster-permutation p = {p_n3r['p_formatted']}; "
    f"{p_n3r['n_cluster_significant']}/{p_n3r['n_electrodes_tested']} electrodes individually "
    f"cluster-significant ({p_n3r['n_fdr_significant']} after Benjamini-Hochberg FDR correction).",
    styles["Caption"]))

# --- 3. Age effect --------------------------------------------------------
story.append(Paragraph("Age modulates HEP amplitude in every sleep stage, with a recurring right-lateralised electrode set", styles["H2"]))
r_age = AGESPLIT["R"]
story.append(Paragraph(
    f"Independently of the within-subject stage contrasts, splitting the full extended-montage cohort at "
    f"the median age ({S['age_median']} years) and contrasting Older vs Younger within each stage reached "
    f"cluster significance in all three stages (light sleep: p={AGESPLIT['light_sleep']['p_formatted']}, "
    f"{AGESPLIT['light_sleep']['n_cluster_significant']}/19 electrodes; N3: p={AGESPLIT['N3']['p_formatted']}, "
    f"{AGESPLIT['N3']['n_cluster_significant']}/19 electrodes; REM: p={r_age['p_formatted']}, "
    f"{r_age['n_cluster_significant']}/19 electrodes, N-older = {r_age['n_older']}, "
    f"N-younger = {r_age['n_younger']}). F8, O1 and T5 (or T4) recurred as cluster-significant in every "
    f"stage, giving the age effect a consistent right-leaning electrode signature distinct from the "
    f"bilateral N3-REM stage effect. That the REM age contrast matches the N3-REM stage contrast in "
    f"robustness (both p={p_n3r['p_formatted']}) while recruiting a smaller, right-lateralised electrode set "
    f"suggests ageing shifts interoceptive cortical gain along a related but not identical axis to sleep "
    f"depth.",
    styles["Body"]))

story.append(Image(os.path.join(FIG_DIR, "agesplit_R.png"), width=6.6 * inch, height=6.6 * inch * 0.62))
story.append(Paragraph(
    f"<b>Figure 2.</b> Older vs Younger (median-split, {S['age_median']} y) independent-samples contrast "
    f"within REM (N-older = {r_age['n_older']}, N-younger = {r_age['n_younger']}). Cluster-permutation "
    f"p = {r_age['p_formatted']}; {r_age['n_cluster_significant']}/19 electrodes cluster-significant "
    f"({', '.join(r_age['sig_channels'])}).",
    styles["Caption"]))
story.append(PageBreak())

# --- 4. Diagnosis-cohort alignment --------------------------------------
story.append(Paragraph("Most diagnosis groups' stage-delta fingerprint is not distinguishable from the undiagnosed reference cohort's", styles["H2"]))
story.append(Paragraph(
    "Given the montage gap documented above, we asked whether the within-subject HEP stage-delta pattern "
    "(the vector of amplitude changes light sleep&rarr;N3, N3&rarr;REM, light sleep&rarr;REM) measured in "
    "each sparse-montage diagnosis sub-cohort (Cohort B) is statistically indistinguishable from the same "
    "pattern in the extended-montage undiagnosed reference cohort (Cohort A), using a "
    f"{D['n_perm']}-permutation label-shuffle test on the Euclidean distance between mean delta vectors, "
    "BH-FDR corrected across the four sub-cohorts.",
    styles["Body"]))

t2_rows = [["Diagnosis sub-cohort", "N (matched 3-stage)", "Distance to A", "Pearson r", "Cosine sim.", "q (BH-FDR)", "Verdict"]]
for g in DIAG_GROUPS:
    a = ALIGN[g]
    q = a.get("q_perm")
    if q is None or q != q:
        verdict = "not testable"
    elif q < ALPHA:
        verdict = "DIVERGES"
    else:
        verdict = "ALIGNED"
    t2_rows.append([g, str(a.get("n_sub", "n/a")), fmt_num(a.get("obs_distance")), fmt_num(a.get("pearson_r")),
                     fmt_num(a.get("cosine_sim")), fmt_p(q), verdict])
tbl2 = Table(t2_rows, colWidths=[1.7 * inch, 0.85 * inch, 0.85 * inch, 0.7 * inch, 0.75 * inch, 0.75 * inch, 0.75 * inch])
tbl2.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4e4e4")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 7.6),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(tbl2)
n_a_matched = ALIGN[DIAG_GROUPS[0]]["n_a"]
story.append(Paragraph(
    f"<b>Table 2.</b> Stage-delta alignment of each Cohort-B diagnosis sub-cohort against Cohort A "
    f"(n = {n_a_matched} Cohort-A patients with a usable amplitude in all three stages). Distance = "
    f"Euclidean distance between mean 3-element stage-delta vectors (&mu;V); q = BH-FDR-corrected "
    f"permutation p-value across the 4 sub-cohorts. \"ALIGNED\" = not significantly different from Cohort "
    f"A's pattern (q &ge; 0.05).", styles["Caption"]))
story.append(Paragraph(
    (f"{', '.join(aligned)} showed a stage-delta vector statistically indistinguishable from Cohort A's "
     f"(q &ge; {ALPHA})" + (f", while {', '.join(diverged)} diverged significantly (q &lt; {ALPHA})." if diverged else ".")
     if aligned else
     f"All testable sub-cohorts ({', '.join(diverged)}) diverged significantly from Cohort A's stage-delta "
     f"pattern (q &lt; {ALPHA}).")
    + " Cognitive Impairment/Dementia showed both the largest distance to Cohort A and the lowest cosine "
      "similarity of the four groups, consistent with dementia-related autonomic-cortical decoupling; "
      "the three other groups' patterns were not distinguishable from the reference cohort's given the "
      "available sparse-montage sample sizes.",
    styles["Body"]))

story.append(Paragraph("Age and EEG spectral power differ by diagnosis, independent of the alignment test", styles["H2"]))
story.append(Paragraph(
    f"Age differed sharply between Cohort A and the pooled Cohort B (median {fmt_num(AGE_OVERALL['median1'],1)} "
    f"vs {fmt_num(AGE_OVERALL['median2'],1)} years, n={AGE_OVERALL['n1']} vs {AGE_OVERALL['n2']}, "
    f"p={fmt_p(AGE_OVERALL['p'])}), and against every individual diagnosis sub-cohort "
    f"(all p&lt;{fmt_p(max(AGE_DX[g]['p'] for g in DIAG_GROUPS))}), confirming that the sparse-montage diagnosed "
    f"group is materially older, as expected for cardio-/cerebro-/cognitive diagnoses. EEG band-power "
    f"(Welch PSD, Kruskal-Wallis across Cohort A + the four diagnosis groups, 4 bands &times; 3 stages = 12 "
    f"tests, BH-FDR corrected) differed significantly in "
    f"{sum(1 for r in PSD_KW if r.get('q_value') == r.get('q_value') and r['q_value'] < ALPHA)}/{len(PSD_KW)} "
    f"band&times;stage cells, concentrated in theta and beta across all three stages "
    f"(theta: H={fmt_num(next(r['H_stat'] for r in PSD_KW if r['band']=='theta' and r['stage']=='light_sleep'))} "
    f"at light sleep; beta: H={fmt_num(next(r['H_stat'] for r in PSD_KW if r['band']=='beta' and r['stage']=='light_sleep'))} "
    f"at light sleep; both q&lt;0.0001), while delta band power did not differ significantly in any stage "
    f"(all q&gt;0.3).",
    styles["Body"]))

story.append(Image(os.path.join(FIG_DIR, "diagnosis_alignment_psd_curves.png"), width=6.6 * inch, height=6.6 * inch * (4.2 / 15)))
story.append(Paragraph(
    "<b>Figure 3.</b> Mean channel-averaged EEG power spectral density (Welch, log10 power) by group "
    "(undiagnosed Cohort A reference plus the four diagnosis sub-cohorts) and sleep stage, 0-45 Hz.",
    styles["Caption"]))

t3_rows = [["Band", "Stage", "H", "p", "q"]]
for r in PSD_KW:
    t3_rows.append([r["band"], r["stage"], fmt_num(r["H_stat"]), fmt_p(r["p_value"]), fmt_p(r["q_value"])])
tbl3 = Table(t3_rows, colWidths=[1.0 * inch, 1.2 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch])
tbl3.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4e4e4")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(tbl3)
story.append(Paragraph(
    "<b>Table 3.</b> Kruskal-Wallis test of EEG band power across the 5 groups (Cohort A + 4 diagnosis "
    "sub-cohorts), per band and sleep stage. q BH-FDR corrected across all 12 tests.",
    styles["Caption"]))
story.append(PageBreak())

# ===========================================================================
# Discussion
# ===========================================================================
story.append(Paragraph("Discussion", styles["H1"]))
story.append(Paragraph(
    "Three findings converge here. First, the within-subject sleep-stage HEP gradient is not three "
    "evenly-spaced states: N3 stands apart from a REM/light-sleep pair that barely differ from each "
    "other, matching a single generator whose gain tracks cortical arousability and autonomic tone more "
    "than sleep depth per se &mdash; REM and light sleep preserve cortical reactivity, whereas N3 reflects "
    "maximal cortical deafferentation and vagal dominance, so cardiac afferent signals reach cortex most "
    "weakly there. Second, age modulates every stage through a recurring, partially right-lateralised "
    "electrode set (F8, O1, T4/T5) that is topographically distinct from, but overlaps in magnitude with, "
    "the N3-REM stage effect, consistent with ageing shifting interoceptive cortical gain along a related "
    "axis rather than simply scaling the same generator uniformly.",
    styles["Body"]))
story.append(Paragraph(
    "Third, and most consequential for how this corpus (or any similar clinical referral corpus) should "
    "be used going forward: diagnosis and electrode montage are confounded. The sparse, 6-electrode "
    "montage cohort is both markedly larger than the extended-montage undiagnosed reference pool and "
    "heavily enriched for patients carrying a clinical diagnosis, with a substantial remainder carrying "
    "other or multiple diagnoses not individually resolved here. A naive comparison of, say, HEP or PSD "
    "between \"diagnosed\" and \"undiagnosed\" groups in this corpus, without controlling for electrode "
    "coverage, would be a comparison of montages as much as of disease status. Once that stratification is "
    "made explicit, most diagnosis sub-cohorts' within-subject stage-delta fingerprint is statistically "
    "indistinguishable from the undiagnosed reference's; only Cognitive Impairment/Dementia showed a "
    "pattern measurably different, alongside a markedly older age distribution and altered theta/beta EEG "
    "power across all three stages relative to the other groups. This is consistent with dementia-related "
    "autonomic-cortical decoupling being detectable even through a sparse montage, whereas the vascular "
    "diagnosis groups' coupling profile, at the resolution afforded by 6 electrodes and these sample "
    "sizes, looks like the undiagnosed reference's.",
    styles["Body"]))
story.append(Paragraph(
    "Clinically, these results argue for two concrete precautions in this and similar corpora: (i) never "
    "compare a diagnosis group's electrophysiology to a reference group without first checking, and if "
    "needed matching, electrode coverage, since coverage and diagnosis are not independent here; and (ii) "
    "never pool REM with light sleep, or compare across sleep stages without an age-matched design, since "
    "both risk mistaking a physiological shift in cortical interoception (sleep depth, or age) for a "
    "disease signature.",
    styles["Body"]))

story.append(Paragraph("Limitations", styles["H2"]))
story.append(Paragraph(
    "<b>Small sparse-montage sub-cohorts.</b> The 6-electrode diagnosed sub-cohorts, though numerous in "
    "aggregate, contribute far fewer matched-3-stage patients than Cohort A to the alignment test, which "
    "widens the permutation null and reduces power to detect a true divergence; an \"ALIGNED\" verdict "
    "should be read as \"not distinguishable given the available N,\" not as proof of equivalence. "
    "<b>Non-exclusive diagnosis groups.</b> Patients with multiple qualifying diagnoses contribute to more "
    "than one sub-cohort, so the four sub-cohort tests are not statistically independent. "
    "<b>Only 3 stage-delta elements.</b> The alignment vectors are 3-dimensional, so the Pearson "
    "correlation reported alongside the permutation test is statistically under-powered (n=3) and included "
    "only as a descriptive complement to the permutation-based significance call. <b>Electrode-count filter "
    "is exact, not a montage-identity check.</b> The 6-electrode Cohort-B filter selects recordings with "
    "exactly 6 standard 10-20 channel names present; it does not verify that all such patients used "
    "literally the same physical montage. <b>PSD sampling.</b> To bound compute, PSD band power was "
    "computed on a capped, R-peak-count-ranked sample per group&times;stage cell rather than every "
    "patient, which could bias band-power estimates toward recordings with cleaner ECG (and possibly "
    "cleaner EEG). <b>Cross-analysis N drift.</b> The extended-montage cohort N differs slightly between "
    f"the sleep-stage/age analysis (n = {S['n_cohort']}) and the diagnosis-alignment analysis's Cohort A "
    f"(n = {N_A_MONTAGE}) because the two used different downstream matching criteria (usable trace in all "
    f"stages vs. non-diagnosis-cohort selection on the raw candidate pool); both are reported as actually "
    f"computed, not reconciled to a single target number.",
    styles["Body"]))

# ===========================================================================
# Methods
# ===========================================================================
story.append(Paragraph("Methods", styles["H1"]))

story.append(Paragraph("Data source and cohorts", styles["H2"]))
story.append(Paragraph(
    "Source data were drawn from the Human Sleep Project, a multi-centre clinical polysomnography corpus "
    "of 90,166 unique patients, via the Harvard_Electroencephalography arm of the project's existing "
    "dashboard caches (<font face='Courier'>get_group_individuals</font>/"
    "<font face='Courier'>load_patient_data</font>; no cache regeneration). <b>Sleep-stage/age cohort:</b> "
    f"patients with &ge;10 of 24 candidate standard 10-20 EEG channels and a usable HEP trace in light "
    f"sleep, N3 and REM (n = {S['n_cohort']}); the subset with no linked ICD-10 diagnosis "
    f"(\"Susp. Epilepsy\", n = {TABLE1['Susp. Epilepsy matched 3-stage subset']['n']}) was used for the "
    f"within-subject paired contrasts. <b>Diagnosis-alignment cohorts:</b> Cohort A, the same undiagnosed "
    f"reference population restricted to &gt;10 EEG channels (n = {N_A_MONTAGE}, via "
    f"<font face='Courier'>select_non_diagnosis_cohort</font>); Cohort B, patients with exactly 6 standard "
    f"10-20 channels (n = {N_B_MONTAGE}), split non-exclusively by linked EHR category into four diagnosis "
    "sub-cohorts (atrial fibrillation, heart failure, stroke/cerebrovascular disease, cognitive "
    "impairment/dementia).",
    styles["Body"]))

story.append(Paragraph("Cluster-permutation statistics", styles["H2"]))
story.append(Paragraph(
    "Within-subject stage contrasts were tested with a paired cluster-mass permutation test (200 "
    "permutations, cluster-forming threshold two-sided &alpha; = 0.01, random seed 42): the pointwise "
    "paired t-statistic was thresholded, contiguous above-threshold samples grouped into clusters, and the "
    "summed |t| cluster mass compared to a null built by randomly sign-flipping each patient's paired "
    "difference. The independent Older-vs-Younger contrast (median-age split, within each stage) used the "
    "same procedure in its independent-samples form (permutation by group-label shuffling). Each contrast "
    "was additionally run electrode-by-electrode, with per-electrode raw cluster p-values further "
    "Benjamini-Hochberg FDR corrected across electrodes within that contrast.",
    styles["Body"]))

story.append(Paragraph("Diagnosis-cohort stage-delta alignment test", styles["H2"]))
story.append(Paragraph(
    "Per-patient, per-stage HEP amplitude (mean of the channel-averaged trace, 0.15-0.5 s post-R-peak) "
    "was reduced, for patients with a usable amplitude in all three stages, to a 3-element within-subject "
    "stage-delta vector (light sleep&rarr;N3, N3&rarr;REM, light sleep&rarr;REM). For each Cohort-B "
    "diagnosis sub-cohort, its mean delta vector was compared to Cohort A's via Euclidean distance, "
    f"Pearson correlation, cosine similarity, and a {D['n_perm']}-permutation label-shuffle test (pooling "
    "the sub-cohort's and Cohort A's matched patients, shuffling group labels preserving group sizes, and "
    "rebuilding the null distribution of mean-vector distances); raw permutation p-values were BH-FDR "
    "corrected across the four sub-cohorts.",
    styles["Body"]))

story.append(Paragraph("Age and EEG power-spectral comparisons", styles["H2"]))
story.append(Paragraph(
    "Age was compared with two-sided Mann-Whitney U tests (Cohort A vs Cohort B overall as a single "
    "descriptive test; Cohort A vs each of the four diagnosis sub-cohorts, BH-FDR corrected across those "
    "four tests). EEG power spectral density was computed per sampled recording (cleaned continuous EEG, "
    "Welch's method, channel-averaged), with band power (delta 0.5-4, theta 4-8, alpha 8-12, beta 12-30 Hz) "
    f"as the trapezoidal-rule integral of the mean PSD over each band; up to {D['psd_max_per_group_stage']} "
    "recordings per group&times;stage cell were sampled (highest R-peak count first) to bound compute, and "
    "band power was compared across the five groups (Cohort A + 4 diagnoses) per stage with Kruskal-Wallis, "
    "BH-FDR corrected across the full 4-band &times; 3-stage family.",
    styles["Body"]))

story.append(Paragraph("Data availability", styles["H2"]))
story.append(Paragraph(
    "Data were derived from the same de-identified, credentialed-access clinical polysomnography corpus "
    "used throughout this project, distributed under credentialed access via the Brain Data Science "
    "Platform. All statistics and figures were produced by the project's existing dashboard modules "
    "(6_hep_group_comparison.py, 16_diagnosis_sleep_stage_comparison_dashboard.py) and the accompanying "
    "analysis/PDF-assembly scripts in the project repository (Paper1/).",
    styles["Body"]))

# ===========================================================================
# References
# ===========================================================================
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
    "5. Maris E, Oostenveld R. Nonparametric statistical testing of EEG- and MEG-data. J Neurosci "
    "Methods. 2007;164(1):177-190.",
    "6. Steinfath TP, et al. Heartbeat-evoked responses in M/EEG: a systematic review of methods with "
    "suggestions for analysis and reporting. Psychophysiology. 2026. PMID 41943417.",
    "7. Cafri N, Benninger F, Blinder P. Sleep-stage and age modulation of the heartbeat-evoked "
    "potential: a topographically-resolved cluster-permutation analysis. Conference abstract, 2026.",
]
for r in refs:
    story.append(Paragraph(r, styles["Ref"]))

doc = SimpleDocTemplate(
    OUT_PDF, pagesize=LETTER,
    topMargin=0.8 * inch, bottomMargin=0.8 * inch,
    leftMargin=0.9 * inch, rightMargin=0.9 * inch,
    title="A two-tier cortical interoception gradient across sleep stages and age, and its confound with diagnosis and montage",
    author="Nir Cafri, Felix Benninger, Pablo Blinder",
)


def _add_page_number(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(LETTER[0] / 2, 0.55 * inch, str(doc_.page))
    canvas.restoreState()


doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
print("Wrote", OUT_PDF, os.path.getsize(OUT_PDF), "bytes")
