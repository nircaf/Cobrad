"""Build the CFAR-cleaned HEP conference abstract."""
from __future__ import annotations

import os
import pickle

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
HERE = os.path.join(REPO, "paper utils", "Paper1")
DATA = os.path.join(HERE, "abstract_data_cfar.pkl")
FIGURE = os.path.join(HERE, "figures", "abstract_3panel_cfar.png")
OUTPUT = os.path.join(REPO, "papers", "Cafri_HEP_CFAR_conference_abstract.pdf")

with open(DATA, "rb") as f:
    data = pickle.load(f)
p = data["panels"]
sig = {k: sum(r.get("fdr_significant", False) for r in v["channel_results"].values())
       for k, v in p.items()}

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("T", parent=styles["Title"], fontSize=14, leading=16,
                          alignment=TA_CENTER, spaceAfter=4))
styles.add(ParagraphStyle("A", parent=styles["Normal"], fontSize=9.5, leading=11.5,
                          alignment=TA_CENTER, spaceAfter=2))
styles.add(ParagraphStyle("Aff", parent=styles["Normal"], fontSize=7, leading=8.2,
                          alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=5))
styles.add(ParagraphStyle("H", parent=styles["Heading2"], fontSize=10.5, leading=12,
                          spaceBefore=4, spaceAfter=1))
styles.add(ParagraphStyle("B", parent=styles["Normal"], fontSize=9.2, leading=11.5,
                          alignment=TA_JUSTIFY, spaceAfter=3))
styles.add(ParagraphStyle("C", parent=styles["Normal"], fontSize=8, leading=9.7,
                          textColor=colors.HexColor("#333333"), spaceBefore=3))

doc = SimpleDocTemplate(OUTPUT, pagesize=letter, topMargin=.30*inch, bottomMargin=.25*inch,
                        leftMargin=.60*inch, rightMargin=.60*inch,
                        title="CFAR-cleaned heartbeat-evoked potentials")
story = [
    Paragraph("CFAR-Cleaned Heartbeat-Evoked Potentials Reveal a Sleep-Stage and Age Gradient in Cortical Interoception", styles["T"]),
    Paragraph("Nir Cafri<super>1,3</super>, Felix Benninger<super>2,3</super>, Pablo Blinder<super>1,2</super> &nbsp;&middot;&nbsp; Correspondence: nircafri@mail.tau.ac.il", styles["A"]),
    Paragraph("<super>1</super>Department of Neurobiology, George S. Wise Faculty of Life Sciences, Tel Aviv University, Israel<br/>"
              "<super>2</super>Sagol School of Neuroscience, Tel Aviv University, Israel<br/>"
              "<super>3</super>Department of Neurology, Rabin Medical Center, Beilinson Hospital, Petah Tikva, Israel", styles["Aff"]),
    Paragraph("Introduction", styles["H"]),
    Paragraph("The heartbeat-evoked potential (HEP) is an R-peak-locked EEG response associated with cortical processing of cardiac afferent signals and interoception.<super>1,2</super> Its interpretation is limited by the cardiac field artifact (CFA), which can reproduce ECG morphology on the scalp. We tested whether previously observed sleep-stage and age gradients persist after Cardiac Field Artifact Removal (CFAR), a cohort-trained signal-space projection that removes the reproducible cardiac scalp direction without requiring ECG during application.", styles["B"]),
    Paragraph("Methods", styles["H"]),
    Paragraph("Data were drawn from the Human Sleep Project multi-centre polysomnography corpus. The analysis used the CFAR-compatible six-site montage (F3/F4/C3/C4/O1/O2). For each patient and stage, the validated unit-length population CFA direction was projected out of the six-channel HEP matrix: X<sub>clean</sub> = X &minus; uu<super>T</super>X. REM&minus;deep sleep (DS) and REM&minus;light sleep (LS) were tested within patients; Older&minus;Younger REM used a median age split. Cluster-mass permutation tests used 200 permutations and &alpha; = 0.01, excluded &minus;50 to +50 ms around the R-peak, and electrode results were Benjamini&ndash;Hochberg corrected.", styles["B"]),
    Paragraph("Results", styles["H"]),
    Paragraph(
        f"After CFAR, REM &minus; DS included {p['A']['result']['matrix_a'].shape[0]} matched patients "
        f"(cluster p = {p['A']['result']['contrast_p']:.4f}; {sig['A']}/6 electrodes FDR-significant). "
        f"REM &minus; LS included {p['B']['result']['matrix_a'].shape[0]} matched patients "
        f"(p = {p['B']['result']['contrast_p']:.4f}; {sig['B']}/6 electrodes). "
        f"Older &minus; Younger REM included {p['C']['result']['matrix_a'].shape[0]}/"
        f"{p['C']['result']['matrix_b'].shape[0]} patients (p = {p['C']['result']['contrast_p']:.4f}; "
        f"{sig['C']}/6 electrodes).", styles["B"]),
    Paragraph("Conclusion", styles["H"]),
    Paragraph("Explicit removal of the population cardiac direction provides a stricter test of HEP physiology. Effects that remain after CFAR are less readily explained by volume-conducted cardiac activity, while attenuation relative to the uncleaned analysis quantifies its contribution. The six-channel validation montage limits spatial inference; denser-montage CFAR replication is required.", styles["B"]),
]

w, h = PILImage.open(FIGURE).size
story.append(Image(FIGURE, width=7.2*inch, height=7.2*inch*h/w))
story.append(Paragraph("<b>Figure.</b> CFAR-cleaned contrasts. Top: mean &Delta; HEP &plusmn; SEM; red indicates cluster-significant intervals and grey indicates the excluded QRS window. Bottom: six-electrode FDR-corrected significance topography. <b>A</b>, REM &minus; DS; <b>B</b>, REM &minus; LS; <b>C</b>, Older &minus; Younger in REM.", styles["C"]))
story.append(Paragraph("<b>Acknowledgements.</b> The Human Sleep Project has received support from the Glenn Foundation and AFAR, the American Academy of Sleep Medicine, NIH, NSF, and the Henry and Allison McCance Center for Brain Health.", styles["C"]))
story.append(Paragraph("<b>References.</b> 1. Park HD, Blanke O. <i>NeuroImage.</i> 2019;197:502&ndash;511. 2. Critchley HD, Garfinkel SN. <i>Curr Opin Psychol.</i> 2017;17:7&ndash;14. 3. Maris E, Oostenveld R. <i>J Neurosci Methods.</i> 2007;164:177&ndash;190. 4. Steinfath TP, et al. <i>Psychophysiology.</i> 2026. PMID 41943417.", styles["C"]))

doc.build(story)
print(f"wrote {OUTPUT} ({os.path.getsize(OUTPUT)} bytes)")
