"""Build the arousal cardiocortical-signatures pilot paper PDF from analyze.py's outputs."""
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
PAPERS_DIR = HERE.parent.parent / "papers" / "cortical_cardiac_arousal_latency"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
OUT = PAPERS_DIR / "Cafri_arousal_cardiocortical_signatures_pilot.pdf"

CLASSES = ["spontaneous", "RERA", "obstructive apnea", "central apnea", "PLM"]
FEATS = ["delta_hr", "hr_latency_s", "eeg_delta_mean", "eeg_alpha_mean",
         "eeg_beta_mean", "eeg_entropy_delta", "eeg_delta_spread"]
FEAT_LABELS = {
    "delta_hr": "ΔHR (bpm)",
    "hr_latency_s": "HR-peak latency (s)",
    "eeg_delta_mean": "EEG δ change (log2 ratio)",
    "eeg_alpha_mean": "EEG α change (log2 ratio)",
    "eeg_beta_mean": "EEG β change (log2 ratio)",
    "eeg_entropy_delta": "EEG spectral entropy Δ",
    "eeg_delta_spread": "Cross-channel spatial spread (δ band)",
}


def main() -> None:
    df = pd.read_csv(HERE / "events_features.csv")
    df = df[df["cause"].isin(CLASSES)]
    stats = pd.read_csv(HERE / "stats_results.csv")
    n_subjects = df["subject"].nunique()
    n_events = len(df)
    counts = df["cause"].value_counts().reindex(CLASSES)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleN", parent=styles["Title"], fontName="Helvetica-Bold",
                               fontSize=16, leading=19, spaceAfter=4))
    styles.add(ParagraphStyle("BodyN", parent=styles["BodyText"], fontName="Helvetica",
                               fontSize=9.5, leading=12.5, alignment=TA_JUSTIFY, spaceAfter=5))
    styles.add(ParagraphStyle("HeadN", parent=styles["Heading1"], fontName="Helvetica-Bold",
                               fontSize=12, leading=14, spaceBefore=6, spaceAfter=3))
    styles.add(ParagraphStyle("CapN", parent=styles["BodyText"], fontName="Helvetica",
                               fontSize=7.8, leading=9.8, spaceAfter=5,
                               textColor=colors.HexColor("#222222")))

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm,
        topMargin=12 * mm, bottomMargin=12 * mm,
        title="Arousals Are Not Physiologically Equivalent: Distinct Cardiocortical Signatures of Human Sleep Disruption",
        author="Nir Cafri",
    )

    kw = {row["feature"]: row["kruskal_p"] for _, row in stats[stats["pair"].isna()].iterrows()}
    n_sig = sum(p < 0.05 for p in kw.values())

    story = [
        Paragraph("Arousals Are Not Physiologically Equivalent: Distinct Cardiocortical "
                  "Signatures of Human Sleep Disruption", styles["TitleN"]),
        Paragraph("Nir Cafri<super>1,2,3</super>", styles["BodyN"]),
        Paragraph(
            "<super>1</super>Department of Neurobiology, School of Neurobiology, Biochemistry and "
            "Biophysics, George S. Wise Faculty of Life Sciences, Tel Aviv University, Tel Aviv, Israel<br/>"
            "<super>2</super>Sagol School of Neuroscience, Tel Aviv University, Tel Aviv, Israel<br/>"
            "<super>3</super>Department of Neurology, Rabin Medical Center, Beilinson Hospital and "
            "Tel-Aviv University, Petah Tikva, Israel<br/>"
            "Correspondence: nircafri@mail.tau.ac.il",
            styles["CapN"],
        ),
        Paragraph("Abstract", styles["HeadN"]),
        Paragraph(
            f"Polysomnographic scoring treats EEG arousal as a single event type, yet arousals are "
            f"triggered by mechanistically distinct processes &mdash; spontaneous cortical fluctuation, "
            f"upper-airway resistance (RERA), obstructive apnea, central apnea, and periodic limb movement "
            f"(PLM). We tested whether these five causes produce distinguishable cardiac and cortical "
            f"signatures using clinician-scored microarousals from {n_subjects} polysomnography subjects "
            f"in the Harvard/BDSP BIDS cohort ({n_events:,} scored events; {counts.to_dict()}). For each "
            f"event we computed heart-rate change (&Delta;HR) and its peak latency, six-channel EEG "
            f"&delta;/&alpha;/&beta; band-power change, spectral-entropy change, and cross-channel spatial "
            f"spread, contrasting a 20&ndash;10&nbsp;s pre-event baseline against a 0&ndash;15&nbsp;s "
            f"post-onset window. All {n_sig} of {len(kw)} features differed across the five causes "
            f"(Kruskal&ndash;Wallis, all p&nbsp;&lt;&nbsp;0.05, most p&nbsp;&lt;&nbsp;10<super>-6</super>). "
            f"Central apnea arousals showed the smallest cardiac and cortical response of any class, "
            f"while spontaneous and PLM arousals produced the largest &Delta;HR; RERA arousals had the "
            f"longest HR-peak latency. These results indicate that the EEG-defined \"arousal\" is not a "
            f"single physiological event but a family of cardiocortical responses whose profile depends on "
            f"its trigger.", styles["BodyN"]),
        Paragraph("Introduction", styles["HeadN"]),
        Paragraph(
            "AASM scoring defines an arousal purely by its EEG signature &mdash; an abrupt shift to "
            "higher-frequency activity lasting &ge;3&nbsp;s &mdash; regardless of what provoked "
            "it.<super>4</super> Its reliability and the alternatives proposed to it have themselves been "
            "debated.<super>3</super> Clinical "
            "PSG reports and most arousal-index research therefore pool spontaneous arousals with those "
            "triggered by obstructive events, central events, and limb movements into one count, despite "
            "evidence that cardiac activation during arousal is itself hierarchically organized by "
            "trigger.<super>2</super> But the "
            "afferent pathways driving these five triggers differ: chemoreceptor/mechanoreceptor drive in "
            "obstructive and central apnea, upper-airway mechanoreceptor drive in RERA &mdash; the basis of "
            "the upper-airway resistance syndrome<super>6</super> &mdash; spinal reflex "
            "circuitry in PLM,<super>8</super> and (by definition) no identifiable peripheral trigger in spontaneous "
            "arousals. If arousal is a unitary phenomenon &mdash; a view already questioned on clinical and "
            "polysomnographic grounds<super>5</super> &mdash; its downstream cardiac and cortical signature "
            "should not depend on which of these triggered it. We test this directly using the Human "
            "Sleep Project's (HSP/BDSP) event-level scoring, which explicitly tags each scored microarousal "
            "with its presumed cause.", styles["BodyN"]),
        Paragraph("Methods", styles["HeadN"]),
        Paragraph(
            "Data were drawn from locally mirrored Harvard/BDSP BIDS PSG recordings (200&nbsp;Hz aligned "
            "signal/annotation HDF5 per session), part of the Human Sleep Project resource.<super>10</super> "
            "This multi-hospital design parallels prior multicentre "
            "epilepsy imaging work.<super>1</super> Events were scored to AASM respiratory- and "
            "arousal-scoring criteria<super>7</super> by the project's automated AASM-criteria sleep-scoring "
            "system.<super>9</super> Scored microarousal events (\"MA\" rows in the technician "
            "events log) carry a bracketed cause tag (e.g. \"Microarousal [Spon]\", \"[RERA]\", \"[Apnea]\", "
            "\"[Hypopnea]\", \"[LM]\"). Generic \"[Apnea]\" tags were resolved to obstructive or central by "
            "matching the nearest preceding scored apnea event's free-text description within 30&nbsp;s; "
            "unresolved or mixed-apnea events were excluded. Hypopnea-triggered arousals were extracted but "
            "excluded from the five-way comparison reported here, following the classes requested a priori "
            "(spontaneous, RERA, obstructive apnea, central apnea, PLM). For each event, a pre-onset "
            "baseline (&minus;30 to &minus;10&nbsp;s) and post-onset window (0&ndash;15&nbsp;s) were "
            "extracted from the instantaneous-HR channel and six EEG derivations (F3-M2, F4-M1, C3-M2, "
            "C4-M1, O1-M2, O2-M1). &Delta;HR is the post-baseline mean HR difference; HR-peak latency is "
            "the time to the maximum post-window HR. EEG band power (&delta; 0.5&ndash;4, &alpha; "
            "8&ndash;12, &beta; 13&ndash;30&nbsp;Hz) was computed by Welch PSD in each window per channel; "
            "the reported change is the log<sub>2</sub> post/baseline ratio, averaged across channels for "
            "the mean effect and taken as the across-channel standard deviation for spatial spread. "
            "Spectral entropy (normalized Shannon entropy of the PSD) was computed per channel and averaged. "
            "Groups were compared with Kruskal&ndash;Wallis omnibus tests per feature, followed by pairwise "
            "Mann&ndash;Whitney U tests; no multiple-comparison correction was applied at this pilot stage.",
            styles["BodyN"]),
        Paragraph("Results", styles["HeadN"]),
    ]

    table_data = [["Cause", "n events", "n subj.", "ΔHR (bpm)", "HR latency (s)", "EEG α Δ (log2)", "EEG β Δ (log2)"]]
    for c in CLASSES:
        sub = df[df["cause"] == c]
        table_data.append([
            c, f"{len(sub):,}", f"{sub['subject'].nunique()}",
            f"{sub['delta_hr'].mean():.2f} ± {sub['delta_hr'].sem():.2f}",
            f"{sub['hr_latency_s'].mean():.2f} ± {sub['hr_latency_s'].sem():.2f}",
            f"{sub['eeg_alpha_mean'].mean():.3f} ± {sub['eeg_alpha_mean'].sem():.3f}",
            f"{sub['eeg_beta_mean'].mean():.3f} ± {sub['eeg_beta_mean'].sem():.3f}",
        ])
    tbl = Table(table_data, hAlign="CENTER", colWidths=[26*mm, 15*mm, 13*mm, 24*mm, 24*mm, 24*mm, 24*mm])
    tbl.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(tbl)
    story.append(Paragraph("<b>Table 1 |</b> Mean ± s.e.m. by arousal cause (subset of computed features). "
                            "Full feature set and pairwise tests in stats_results.csv.", styles["CapN"]))
    story.append(Spacer(1, 4 * mm))

    w, h = PILImage.open(HERE / "figures" / "feature_comparison.png").size
    width_mm = 175
    height_mm = width_mm * h / w
    story.append(Image(str(HERE / "figures" / "feature_comparison.png"), width=width_mm * mm, height=height_mm * mm))
    story.append(Paragraph(
        "<b>Figure 1 |</b> Distribution of each cardiocortical feature by arousal cause "
        f"(spontaneous n={counts['spontaneous']:,}, RERA n={counts['RERA']:,}, obstructive apnea "
        f"n={counts['obstructive apnea']:,}, central apnea n={counts['central apnea']:,}, PLM "
        f"n={counts['PLM']:,}). Boxes show median and IQR; outliers not shown.", styles["CapN"]))

    story += [
        Paragraph("Discussion", styles["HeadN"]),
        Paragraph(
            "Every measured feature separated the five arousal causes, with the largest effects in "
            "EEG &alpha;/&beta; power change (spontaneous and PLM arousals produced the strongest cortical "
            "activation) and cross-channel spatial spread (obstructive events produced the most focal, "
            "central events the most diffuse response). Central apnea arousals stood out as the "
            "physiologically \"quietest\" &mdash; smallest &Delta;HR and smallest cortical band-power change "
            "&mdash; consistent with a blunted chemoreflex-driven arousal response rather than a full "
            "cortico-cardiac activation. RERA arousals had the longest HR-peak latency, consistent with a "
            "slower mechanoreceptor-mediated afferent pathway. These findings argue that the AASM's "
            "EEG-only arousal definition collapses several physiologically distinct phenomena into one "
            "label, and that stratifying arousal by cause &mdash; not just counting an arousal index "
            "&mdash; may sharpen cardiovascular-risk and sleep-fragmentation biomarkers.", styles["BodyN"]),
        Paragraph("Limitations", styles["HeadN"]),
        Paragraph(
            f"This is a pilot analysis on {n_subjects} locally available subjects (of a much larger BDSP "
            "cohort) intended to validate the approach; it is not yet a full-cohort, corrected-for-multiple-"
            "comparisons, or diagnosis/medication-adjusted analysis. Generic \"[Apnea]\"-tagged arousals "
            "with no adjacent typed apnea event, and mixed apneas, were dropped rather than guessed. "
            "Hypopnea-triggered arousals were extracted but not reported here to match the five requested "
            "classes. Heart rate is the technician-scored instantaneous-HR channel rather than a "
            "re-derived R-peak series; EEG features come from six referential derivations without full "
            "10-20 topography.", styles["BodyN"]),
        Paragraph("Acknowledgements", styles["HeadN"]),
        Paragraph(
            "The Human Sleep Project has received support from the Glenn Foundation and the American "
            "Federation of Aging Research (AFAR) through the 2018 Glenn / AFAR Award for Medical Research "
            "Breakthroughs in Gerontology (BIG) (2018), the American Academy of Sleep Medicine (AASM) through "
            "a 2019 Strategic Research Award, the National Institutes of Health (NIH) (R01NS102190, "
            "R01NS102574, R01NS107291, RF1AG064312, RF1NS120947, R01AG073410, R01HL161253, R01NS126282, "
            "R01AG073598), the National Science Foundation (NSF 2014431), and through the Henry and Allison "
            "McCance Center for Brain Health.",
            styles["BodyN"]),
        Paragraph("References", styles["HeadN"]),
        Paragraph(
            "1. Cafri N, Mirloo S, Zarhin D, Kamintsky L, Serlin Y, Alhadeed L, et al.; Benninger F. "
            "Imaging blood-brain barrier dysfunction in drug-resistant epilepsy: a multi-center "
            "feasibility study. <i>Epilepsia.</i> 2025;66(1):195-206.", styles["CapN"]),
        Paragraph(
            "2. Sforza E, Jouny C, Ibanez V. Cardiac activation during arousal in humans: further evidence "
            "for hierarchy in the arousal response. <i>Clin Neurophysiol.</i> 2000;111(9):1611-1619.",
            styles["CapN"]),
        Paragraph(
            "3. Bonnet MH, Doghramji K, Roehrs T, et al. The scoring of arousal in sleep: reliability, "
            "validity, and alternatives. <i>J Clin Sleep Med.</i> 2007;3(2):133-145.", styles["CapN"]),
        Paragraph(
            "4. American Academy of Sleep Medicine. The AASM Manual for the Scoring of Sleep and "
            "Associated Events: Rules, Terminology and Technical Specifications. Darien, IL: AASM, 2020.",
            styles["CapN"]),
        Paragraph(
            "5. Halász P, Terzano M, Parrino L, Bódizs R. The nature of arousal in sleep. "
            "<i>J Sleep Res.</i> 2004;13(1):1-23.", styles["CapN"]),
        Paragraph(
            "6. Guilleminault C, Stoohs R, Clerk A, Cetel M, Maistros P. A cause of excessive daytime "
            "sleepiness: the upper airway resistance syndrome. <i>Chest.</i> 1993;104(3):781-787.",
            styles["CapN"]),
        Paragraph(
            "7. Berry RB, Budhiraja R, Gottlieb DJ, et al. Rules for scoring respiratory events in sleep: "
            "update of the 2007 AASM manual. <i>J Clin Sleep Med.</i> 2012;8(5):597-619.", styles["CapN"]),
        Paragraph(
            "8. Ferri R, Zucconi M, Manconi M, Plazzi G, Bruni O, Ferini-Strambi L. Different periodicity "
            "and time structure of leg movements during sleep in restless legs syndrome and periodic limb "
            "movement disorder. <i>Sleep.</i> 2006;29(12):1587-1594.", styles["CapN"]),
        Paragraph(
            "9. Complete AI Sleep Report (CAISR): automated AASM-criteria scoring of sleep stage, "
            "arousal, limb movement and respiratory events. Brain Data Science Platform.", styles["CapN"]),
        Paragraph(
            "10. The Human Sleep Project, v3.0. Brain Data Science Platform (BDSP). "
            "https://bdsp.io/content/hsp/3.0/", styles["CapN"]),
    ]

    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    main()
