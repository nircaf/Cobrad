"""
Assemble the 1-page conference abstract: title/authors, a short structured
abstract, and the 3-panel delta-HEP figure (REM-N3, REM-Light, Older-Younger
REM) from make_abstract_figures.py. Single page, generous font sizes, no
overlapping elements.

Run: source venv/bin/activate && python3 Paper1/make_abstract_pdf.py
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
OUT_DIR = os.path.join(REPO, "Paper1")
FIG_PATH = os.path.join(OUT_DIR, "figures", "abstract_3panel.png")
OUT_PDF = os.path.join(OUT_DIR, "Cafri_HEP_conference_abstract.pdf")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("AbsTitle", parent=styles["Title"], fontSize=13.5, leading=16,
                           spaceAfter=3, alignment=TA_CENTER))
styles.add(ParagraphStyle("AbsAuthor", parent=styles["Normal"], fontSize=10, leading=12,
                           alignment=TA_CENTER, spaceAfter=1))
styles.add(ParagraphStyle("AbsAffil", parent=styles["Normal"], fontSize=6.6, leading=7.9,
                           alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=4))
styles.add(ParagraphStyle("AbsH2", parent=styles["Heading2"], fontSize=10.3, leading=12.5,
                           spaceBefore=3, spaceAfter=1.5, textColor=colors.HexColor("#1a1a1a")))
styles.add(ParagraphStyle("AbsBody", parent=styles["Normal"], fontSize=9.6, leading=12.6,
                           spaceAfter=3, alignment=TA_JUSTIFY))
styles.add(ParagraphStyle("AbsCaption", parent=styles["Normal"], fontSize=8.4, leading=10.6,
                           textColor=colors.HexColor("#333333"), spaceBefore=4))

doc = SimpleDocTemplate(
    OUT_PDF, pagesize=letter,
    topMargin=0.3 * inch, bottomMargin=0.22 * inch,
    leftMargin=0.6 * inch, rightMargin=0.6 * inch,
)

story = []
story.append(Paragraph(
    "Heartbeat-Evoked Potentials Reveal a Sleep-Stage and Age Gradient "
    "in Cortical Interoception",
    styles["AbsTitle"]))
story.append(Paragraph(
    "Nir Cafri<super>1,3</super>, Felix Benninger<super>2,3</super>, Pablo Blinder<super>1,2</super> "
    "&nbsp;&middot;&nbsp; Correspondence: nircafri@mail.tau.ac.il",
    styles["AbsAuthor"]))
story.append(Paragraph(
    "<super>1</super>Department of Neurobiology, School of Neurobiology, Biochemistry and Biophysics, "
    "George S. Wise Faculty of Life Sciences, Tel Aviv University, Tel Aviv, Israel<br/>"
    "<super>2</super>Sagol School of Neuroscience, Tel Aviv University, Tel Aviv, Israel<br/>"
    "<super>3</super>Department of Neurology, Rabin Medical Center, Beilinson Hospital and "
    "Tel-Aviv University, Petah Tikva, Israel",
    styles["AbsAffil"]))

story.append(Paragraph("Introduction", styles["AbsH2"]))
story.append(Paragraph(
    "The heartbeat-evoked potential (HEP) is an R-peak-locked EEG deflection reflecting cortical processing of "
    "cardiac afferent signals, contaminated by the cardiac field artifact; the &minus;50 to +50 ms window around "
    "the R-peak is excluded from every test. Prior work found HEP magnitude follows a vigilance-state gradient "
    "across sleep (REM &gt; light sleep [LS] &gt; deep sleep [DS]) and increases with age. Whether these effects "
    "share a generator, and their topography, has not been tested.",
    styles["AbsBody"]))

story.append(Paragraph("Methods", styles["AbsH2"]))
story.append(Paragraph(
    "Data were drawn from the Human Sleep Project, a multi-centre polysomnography corpus of 90,166 patients; "
    "2,443 had the full 19-electrode montage, a heterogeneous referral cohort. Sleep-stage contrasts were "
    "tested with paired cluster-mass permutation tests (200 permutations, &alpha; = 0.01); an Older-vs-Younger "
    "contrast (median split) was run within REM, each tested electrode-by-electrode.",
                            styles["AbsBody"]))

story.append(Paragraph("Results", styles["AbsH2"]))
story.append(Paragraph(
    "REM &minus; DS showed a robust post-QRS difference (N = 2136, p = 0.0050; 15/19 electrodes), as did "
    "Older &minus; Younger REM (N = 1144/1299, p = 0.0050; 9/19 electrodes). REM &minus; LS was smaller, "
    "missing threshold (N = 2202, p = 0.0149; 1/19 electrodes). All three peaked outside the excluded CFA "
    "window, indicating genuine response.",
    styles["AbsBody"]))

story.append(Paragraph("Conclusion", styles["AbsH2"]))
story.append(Paragraph(
    "The topography reveals a two-tier gradient: REM and LS barely differ, while both diverge sharply from DS, "
    "tracking arousability more than sleep depth &mdash; REM and LS preserve cortical reactivity, DS reflects "
    "maximal deafferentation and vagal dominance. The REM age effect matches the DS contrast in magnitude, "
    "suggesting ageing shifts interoceptive gain along the sleep-depth axis. Comparisons not age-matched, or "
    "pooling REM with LS with DS, risk mistaking a physiological shift for disease.",
    styles["AbsBody"]))

from PIL import Image as PILImage
_fig_w, _fig_h = PILImage.open(FIG_PATH).size
_img_width = 7.2 * inch  # full text width (8.5in page - 0.6in margins each side), was 5.8in
img = Image(FIG_PATH, width=_img_width, height=_img_width * (_fig_h / _fig_w))
story.append(img)
story.append(Paragraph(
    "<b>Figure.</b> Each panel: mean &Delta; HEP waveform (left) and per-electrode significance topomap "
    "(right). Red = cluster-significant (p &lt; 0.01); grey = CFA-excluded window. <b>A</b>: REM &minus; DS. "
    "<b>B</b>: REM &minus; LS (within-subject, paired, Susp. Epilepsy cohort). <b>C</b>: Older &minus; "
    "Younger, REM (age median-split).",
    styles["AbsCaption"]))

doc.build(story)
print(f"wrote {OUT_PDF} ({os.path.getsize(OUT_PDF)} bytes)")
