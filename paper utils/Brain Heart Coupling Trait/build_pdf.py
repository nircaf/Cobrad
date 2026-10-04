#!/usr/bin/env python3
"""Build the brain-heart coupling trait protocol PDF from manuscript.md."""

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer
import re


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "manuscript.md"
OUTPUT = HERE.parents[1] / "papers" / "Cafri_brain_heart_coupling_trait_paper.pdf"


def inline_markup(text: str) -> str:
    text = escape(text)
    while "**" in text:
        text = text.replace("**", "<b>", 1)
        if "**" not in text:
            break
        text = text.replace("**", "</b>", 1)
    return text


def footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(0.72 * inch, 0.48 * inch, "Within-night and repeat-PSG brain-heart coupling")
    canvas.drawRightString(7.78 * inch, 0.48 * inch, str(document.page))
    canvas.restoreState()


base = getSampleStyleSheet()
styles = {
    "title": ParagraphStyle(
        "PaperTitle", parent=base["Title"], fontName="Helvetica-Bold",
        fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=12,
        textColor=colors.HexColor("#17233c"),
    ),
    "subtitle": ParagraphStyle(
        "Subtitle", parent=base["Normal"], fontName="Helvetica",
        fontSize=11.5, leading=15, alignment=TA_CENTER, spaceAfter=14,
        textColor=colors.HexColor("#40516f"),
    ),
    "h1": ParagraphStyle(
        "H1", parent=base["Heading1"], fontName="Helvetica-Bold",
        fontSize=13, leading=16, spaceBefore=14, spaceAfter=6,
        textColor=colors.HexColor("#17233c"),
    ),
    "h2": ParagraphStyle(
        "H2", parent=base["Heading2"], fontName="Helvetica-Bold",
        fontSize=10.5, leading=13, spaceBefore=9, spaceAfter=4,
        textColor=colors.HexColor("#30486d"),
    ),
    "body": ParagraphStyle(
        "Body", parent=base["BodyText"], fontName="Times-Roman",
        fontSize=9.6, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=6,
    ),
    "meta": ParagraphStyle(
        "Meta", parent=base["Normal"], fontName="Helvetica-Oblique",
        fontSize=9.5, leading=13, alignment=TA_CENTER, spaceAfter=10,
        textColor=colors.HexColor("#555555"),
    ),
    "list": ParagraphStyle(
        "List", parent=base["BodyText"], fontName="Times-Roman",
        fontSize=9.6, leading=13.2, leftIndent=18, firstLineIndent=-12, spaceAfter=4,
    ),
    "equation": ParagraphStyle(
        "Equation", parent=base["BodyText"], fontName="Times-Italic",
        fontSize=10, leading=14, alignment=TA_CENTER, spaceBefore=5, spaceAfter=8,
    ),
    "caption": ParagraphStyle(
        "Caption", parent=base["Normal"], fontName="Helvetica",
        fontSize=8.2, leading=10.8, textColor=colors.HexColor("#444444"),
        spaceBefore=3, spaceAfter=10,
    ),
}


lines = SOURCE.read_text(encoding="utf-8").splitlines()
story = []
paragraph = []
title_seen = False


def flush_paragraph():
    if not paragraph:
        return
    value = " ".join(part.strip() for part in paragraph)
    paragraph.clear()
    equation_starts = ("y_ij =", "Var(y) =", "ICC =")
    style = styles["equation"] if value.startswith(equation_starts) else styles["body"]
    story.append(Paragraph(inline_markup(value), style))
    if value.startswith("We screened the local Harvard polysomnography archive"):
        story.append(Paragraph(
            "This multi-hospital design parallels prior multicentre epilepsy imaging work [16].",
            styles["body"]))


for line in lines:
    stripped = line.strip()
    if not stripped:
        flush_paragraph()
        continue
    if stripped.startswith("# "):
        flush_paragraph()
        story.append(Spacer(1, 0.65 * inch))
        story.append(Paragraph(inline_markup(stripped[2:]), styles["title"]))
        title_seen = True
    elif stripped.startswith("## "):
        flush_paragraph()
        value = stripped[3:]
        if title_seen and value.startswith("Five-Minute"):
            story.append(Paragraph(inline_markup(value), styles["subtitle"]))
        else:
            story.append(Paragraph(inline_markup(value), styles["h1"]))
    elif stripped.startswith("### "):
        flush_paragraph()
        story.append(Paragraph(inline_markup(stripped[4:]), styles["h2"]))
    elif stripped.startswith("!["):
        flush_paragraph()
        match = re.match(r"!\[(.+)\]\((.+)\)", stripped)
        if match:
            caption, relpath = match.groups()
            img = Image(str(HERE / relpath), width=6.85 * inch, height=2.42 * inch)
            img._restrictSize(6.85 * inch, 3.35 * inch)
            story.append(KeepTogether([img, Paragraph(inline_markup(caption), styles["caption"])]))
    elif stripped.startswith("**Study protocol") or stripped.startswith("**Original research") or stripped.startswith("**Keywords:"):
        flush_paragraph()
        story.append(Paragraph(inline_markup(stripped), styles["meta"]))
        if stripped.startswith("**Study protocol"):
            story.append(Spacer(1, 0.2 * inch))
    elif re.match(r"^\d+\.\s", stripped):
        flush_paragraph()
        story.append(Paragraph(inline_markup(stripped), styles["list"]))
    else:
        paragraph.append(stripped)

flush_paragraph()
story.append(Paragraph(
    "16. Cafri N, Mirloo S, Zarhin D, Kamintsky L, Serlin Y, Alhadeed L, et al.; Benninger F. "
    "Imaging blood-brain barrier dysfunction in drug-resistant epilepsy: a multi-center feasibility "
    "study. <i>Epilepsia.</i> 2025;66(1):195-206.", styles["list"]))

doc = SimpleDocTemplate(
    str(OUTPUT), pagesize=LETTER, rightMargin=0.72 * inch, leftMargin=0.72 * inch,
    topMargin=0.68 * inch, bottomMargin=0.68 * inch,
    title="Brain-Heart Coupling Is State-Labile Within Nights but Its Nightly Mean Recurs Across Polysomnograms",
    author="Nir Cafri",
    subject="Real-data five-minute variance decomposition and repeat-PSG reliability",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUTPUT)
