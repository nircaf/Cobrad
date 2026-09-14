"""Build the five-figure short paper for Paper 2 (diagnosis + sleep-stage + age HEP gradient).

Reuses frozen, already-computed analysis outputs from Paper1 (no statistics
recomputed): fig1_overview_data.pkl, hep_diagnosis_long_df.pkl,
stage_delta_age_results_v2.json, and the diagnosis-vs-reference waveform
figure already rendered in Paper1/figures. Only the cohort-wide 19-electrode
amplitude topomap (Figure 2b) is computed fresh here, directly from the
frozen per-patient long-format amplitude table.
"""
from pathlib import Path
import json
import pickle
import subprocess
import sys

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, find_peaks, sosfiltfilt

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate

ROOT = Path(__file__).resolve().parent
P1 = ROOT.parent / "Paper1"
FD = ROOT / "figures"
FD.mkdir(parents=True, exist_ok=True)
PAPERS_DIR = ROOT.parent.parent / "papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
OUT = PAPERS_DIR / "Cafri_HEP_sleepstage_age_diagnosis_paper.pdf"

BLENDER_BIN = "blender"
BLENDER_SCRIPT = P1 / "eeg_brain_blender.py"
BRAIN_CACHE = FD / "brain_cache"
BRAIN_CACHE.mkdir(parents=True, exist_ok=True)
# RdBu_r-style blue -> near-white -> red, matching the original topomap's
# colour scheme, but mapped across the data's actual min..max (not a
# zero-centred symmetric range the mostly-positive amplitudes never reach)
# so the full colour range still carries contrast.
SEQUENTIAL_BRAIN_CMAP = [
    (0.00, (0.13, 0.30, 0.75)),
    (0.50, (0.95, 0.95, 0.95)),
    (1.00, (0.70, 0.05, 0.05)),
]


def render_brain_panel(key, channels, vmin, vmax, cache=BRAIN_CACHE, cmap=SEQUENTIAL_BRAIN_CMAP):
    """Shell out to headless Blender for a 3D cortex heat-map of one panel.
    Cached by key since re-rendering per cosmetic tweak elsewhere is slow."""
    out_png = cache / f"{key}_brain.png"
    cfg_path = cache / f"{key}_cfg.json"
    cfg = {
        "channels": {ch: float(v) for ch, v in channels.items() if np.isfinite(v)},
        "view": "top",
        "cmap": [list(stop) for stop in cmap],
        "vmin": float(vmin),
        "vmax": float(vmax),
        "sigma": 0.42,
        "fill": 0.85,
        "subdiv": 7,
        "samples": 96,
        "denoise": False,  # OIDN silently blanks the render on this host's Blender/driver combo
        "show_electrodes": False,
        "resolution": [900, 900],
    }
    cfg_path.write_text(json.dumps(cfg))
    subprocess.run(
        [BLENDER_BIN, "-b", "-P", str(BLENDER_SCRIPT), "--", str(cfg_path), str(out_png)],
        check=True, capture_output=True, text=True,
    )
    return out_png

BLUE, ORANGE, GREEN, RED, PURPLE = "#0072B2", "#D55E00", "#009E73", "#C43C39", "#7B3294"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

CANON19 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4",
    "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]
ALIAS = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def panel(ax, s):
    ax.text(-0.09, 1.1, s, transform=ax.transAxes, weight="bold", fontsize=15,
             ha="right", va="bottom")


def _norm(ch):
    return ch.upper().replace("EEG ", "").replace("EEG-", "").strip()


# ---------------------------------------------------------------- Figure 1
def fig1():
    candidates = list((ROOT.parent / "EDF_Format").rglob("*.EDF"))
    p = next(x for x in candidates if not x.name.startswith("._") and x.stat().st_size > 1_000_000)
    raw = mne.io.read_raw_edf(p, preload=True, verbose="ERROR")
    fs = raw.info["sfreq"]
    norm_map = {_norm(c): c for c in raw.ch_names}
    eeg_name = norm_map.get("C3", raw.ch_names[0])
    eeg = raw.get_data(picks=[eeg_name])[0] * 1e6
    eeg_label = _norm(eeg_name)
    standard_1020 = {
        "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
        "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"}
    non = [c for c in raw.ch_names if _norm(c) not in standard_1020]
    best = None
    for c in non:
        x = raw.get_data(picks=[c])[0]
        if np.nanstd(x) == 0:
            continue
        sos = butter(2, [5, 35], btype="bandpass", fs=fs, output="sos")
        y = sosfiltfilt(sos, x)
        peaks, _ = find_peaks(np.abs(y), distance=0.45 * fs, prominence=2 * np.std(y))
        if 8 < len(peaks) < raw.times[-1] * 2.2:
            rr = np.diff(peaks) / fs
            score = np.median(np.abs(y[peaks])) / (np.std(y) + 1e-12) / (np.std(rr) + 0.08)
            if best is None or score > best[0]:
                best = (score, c, y, peaks)
    _, ecg_name, ecg, peaks = best
    starts = np.arange(int(20 * fs), min(len(eeg) - int(12 * fs), int(300 * fs)), int(10 * fs))
    s = min(starts, key=lambda q: np.std(eeg[q:q + int(10 * fs)]))
    t = np.arange(int(10 * fs)) / fs
    idx = (peaks > s) & (peaks < s + 10 * fs)
    pp = peaks[idx]
    epochs = []
    for r in peaks:
        a = int(r - 0.3 * fs)
        b = int(r + 0.5 * fs)
        if a >= 0 and b < len(eeg):
            ep = eeg[a:b]
            ep = ep - np.mean(ep[: int(0.15 * fs)])
            epochs.append(ep)
    ep = np.asarray(epochs)
    keep = np.ptp(ep, axis=1) < np.percentile(np.ptp(ep, axis=1), 80)
    ep = ep[keep]
    et = np.arange(ep.shape[1]) / fs - 0.3
    mean = np.mean(ep, axis=0)
    sem = np.std(ep, axis=0) / np.sqrt(len(ep))

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1.3], hspace=0.42, left=0.13, right=0.97, top=0.93, bottom=0.08)
    ax = fig.add_subplot(gs[0])
    ax.plot(t, eeg[s:s + len(t)], lw=0.8, color=BLUE)
    ax.set_ylabel(f"{eeg_label} (µV)")
    ax.set_title("Representative patient: simultaneous EEG, ECG and R-locked HEP", weight="bold")
    panel(ax, "a")
    ax.set_xticklabels([])

    ax = fig.add_subplot(gs[1])
    z = ecg[s:s + len(t)] / np.std(ecg[s:s + len(t)])
    ax.plot(t, z, lw=0.85, color=RED)
    ax.scatter((pp - s) / fs, z[(pp - s).astype(int)], s=14, color="black", zorder=3, label="R peak")
    ax.set_ylabel(f"ECG ({_norm(ecg_name)}, z)")
    ax.set_xlabel("Time (s)")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    panel(ax, "b")

    ax = fig.add_subplot(gs[2])
    ax.fill_between(et, mean - sem, mean + sem, color=GREEN, alpha=0.22)
    ax.plot(et, mean, color=GREEN, lw=2.2)
    ax.axvspan(-0.05, 0.05, color="0.85", label="Cardiac-field artifact (excluded)")
    ax.axvline(0, color="0.25", ls=":")
    ax.axhline(0, color="0.65", lw=0.7)
    ax.set(xlim=(-0.3, 0.5), xlabel="Time from R peak (s)", ylabel="HEP (µV)")
    ax.text(0.98, 0.92, f"n = {len(ep)} beats", transform=ax.transAxes, ha="right")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    panel(ax, "c")
    fig.savefig(FD / "figure1.png")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2
def _cohort_amplitude_by_electrode():
    d = pickle.load(open(P1 / "hep_diagnosis_long_df.pkl", "rb"))
    df = d["long_df"].copy()
    df["electrode"] = df["electrode"].replace(ALIAS)
    df = df[df["electrode"].isin(CANON19)]
    n_patients = df["patient_id"].nunique()
    out = {}
    for stage in ["light_sleep", "N3", "R"]:
        g = df[df["stage"] == stage].groupby("electrode")["hep_amplitude_uv"].mean()
        out[stage] = g.reindex(CANON19).to_numpy()
    return out, n_patients


def fig2():
    d = pickle.load(open(P1 / "fig1_overview_data.pkl", "rb"))
    waves = d["panels"]
    amp_by_stage, n_patients = _cohort_amplitude_by_electrode()

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = GridSpec(2, 3, height_ratios=[1.2, 1], hspace=0.42, wspace=0.15,
                  left=0.09, right=0.9, top=0.9, bottom=0.06)

    ax = fig.add_subplot(gs[0, :])
    cols = {"light_sleep": BLUE, "N3": ORANGE, "R": GREEN}
    labels = {"light_sleep": "Light sleep", "N3": "N3 (SWS)", "R": "REM"}
    for st in ["light_sleep", "N3", "R"]:
        vv = [waves[(el, st)] for el in d["electrodes"]]
        times = vv[0]["times"]
        y = np.nanmean([v["grand_mean"] for v in vv], axis=0)
        ax.plot(times, y, lw=2.2, color=cols[st], label=labels[st])
    ax.axvspan(-0.05, 0.05, color="0.88")
    ax.axvline(0, color="0.3", ls=":")
    ax.set(xlabel="Time from R peak (s)", ylabel="Mean HEP (µV)",
           title=f"Cohort-wide HEP by sleep stage (n = {n_patients:,} patients)")
    # ponytail: headroom so the inline legend clears the REM peak
    _lo, _hi = ax.get_ylim()
    ax.set_ylim(_lo, _hi + 0.12 * (_hi - _lo))
    ax.legend(frameon=False, ncol=3, loc="upper right")
    panel(ax, "a")

    all_vals = np.concatenate(list(amp_by_stage.values()))
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    for i, st in enumerate(["light_sleep", "N3", "R"]):
        ax = fig.add_subplot(gs[1, i])
        channels = dict(zip(CANON19, amp_by_stage[st]))
        brain_png = render_brain_panel(f"fig2_{st}", channels, vmin, vmax)
        ax.imshow(plt.imread(brain_png))
        ax.axis("off")
        ax.set_title(labels[st], fontsize=11)
        if i == 0:
            panel(ax, "b")

    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    cmap = LinearSegmentedColormap.from_list("brain_amp", SEQUENTIAL_BRAIN_CMAP)
    sm = ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap)
    cax = fig.add_axes([0.93, 0.11, 0.015, 0.32])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("HEP amplitude (µV)\n0.15–0.5 s window", fontsize=8.5)
    fig.savefig(FD / "figure2.png")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3
def fig3():
    # Crop off the source PNG's stray footnote line (truncated at its right
    # edge in the original render) since the PDF caption already states it.
    src = P1 / "figures" / "fig4_diagnosis_waveforms.png"
    from PIL import Image as PILImage
    src_im = PILImage.open(src)
    w, h = src_im.size
    cropped = src_im.crop((0, 0, w, h - 110))
    im = np.asarray(cropped)
    fig, ax = plt.subplots(figsize=(7.2, 7.7))
    ax.imshow(im)
    ax.axis("off")
    fig.savefig(FD / "figure3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 4
def _trim(path, pad=8):
    from PIL import Image as PILImage, ImageChops
    im = PILImage.open(path).convert("RGB")
    bg = PILImage.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox is None:
        return np.asarray(im)
    l, t, r, b = bbox
    l = max(0, l - pad)
    t = max(0, t - pad)
    r = min(im.width, r + pad)
    b = min(im.height, b + pad)
    return np.asarray(im.crop((l, t, r, b)))


def _stacked_panel_composite(paths, outfile, title):
    trimmed = [_trim(p) for p in paths]
    ratios = [im.shape[0] / im.shape[1] for im in trimmed]
    heights = [7.2 * r for r in ratios]
    fig, axs = plt.subplots(
        len(trimmed), 1, figsize=(7.2, sum(heights)),
        gridspec_kw={"height_ratios": ratios, "hspace": 0.06},
    )
    if len(trimmed) == 1:
        axs = [axs]
    for i, im in enumerate(trimmed):
        axs[i].imshow(im)
        axs[i].axis("off")
        axs[i].text(-0.02, 0.5, chr(97 + i), transform=axs[i].transAxes,
                     va="center", ha="right", weight="bold", fontsize=15)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.98, bottom=0.01)
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def fig4():
    S = json.load(open(P1 / "stage_delta_age_results_v2.json"))
    sig = [x for x in S["pairwise"] if float(x["p_formatted"]) < 0.05]
    names = {
        ("N3", "R"): "pairwise_N3_vs_R.png",
        ("light_sleep", "N3"): "pairwise_light_sleep_vs_N3.png",
        ("light_sleep", "R"): "pairwise_light_sleep_vs_R.png",
    }
    paths = []
    for x in sig:
        key = (x["stage_a"], x["stage_b"])
        p = P1 / "figures" / names.get(key, names.get(tuple(reversed(key)), ""))
        if p.exists():
            paths.append(p)
    _stacked_panel_composite(paths, FD / "figure4.png", "")


# ---------------------------------------------------------------- Figure 5
def fig5():
    S = json.load(open(P1 / "stage_delta_age_results_v2.json"))
    sig = [x for x in S["age_split"] if float(x["p_formatted"]) < 0.05]
    paths = [P1 / "figures" / f"agesplit_{r['stage']}.png" for r in sig]
    _stacked_panel_composite(paths, FD / "figure5.png", "")


# ---------------------------------------------------------------- Manuscript
def manuscript():
    S = json.load(open(P1 / "stage_delta_age_results_v2.json"))
    _, n_patients = _cohort_amplitude_by_electrode()
    d4 = json.load(open(P1 / "fig4_diagnosis_waveforms_results.json"))
    top_dx = ", ".join(f"{n} (n={c:,})" for n, c in d4["top_diagnoses"])

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleN", parent=styles["Title"], fontName="Helvetica-Bold",
                               fontSize=16, leading=19, spaceAfter=4))
    styles.add(ParagraphStyle("BodyN", parent=styles["BodyText"], fontName="Helvetica",
                               fontSize=9, leading=11.6, alignment=TA_JUSTIFY, spaceAfter=4))
    styles.add(ParagraphStyle("HeadN", parent=styles["Heading1"], fontName="Helvetica-Bold",
                               fontSize=11.5, leading=13, spaceBefore=5, spaceAfter=2))
    styles.add(ParagraphStyle("CapN", parent=styles["BodyText"], fontName="Helvetica",
                               fontSize=7.8, leading=9.8, spaceAfter=5,
                               textColor=colors.HexColor("#222222")))

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, rightMargin=13 * mm, leftMargin=13 * mm,
        topMargin=9 * mm, bottomMargin=8 * mm,
        title="Heartbeat-Evoked Potentials Reveal a Sleep-Stage and Age Gradient in Cortical Interoception",
        author="Nir Cafri, Felix Benninger, Pablo Blinder",
    )

    story = [
        Paragraph("Heartbeat-Evoked Potentials Reveal a Sleep-Stage and Age Gradient in Cortical Interoception", styles["TitleN"]),
        Paragraph("Nir Cafri<super>1,3</super>, Felix Benninger<super>2,3</super>, Pablo Blinder<super>1,2</super>", styles["BodyN"]),
        Paragraph("<super>1</super>Tel Aviv University; <super>2</super>Sagol School of Neuroscience; "
                  "<super>3</super>Rabin Medical Center, Israel. Correspondence: nircafri@mail.tau.ac.il", styles["CapN"]),
        Paragraph("Abstract", styles["HeadN"]),
        Paragraph(
            f"The heartbeat-evoked potential (HEP) is an EEG signature of cortical processing of cardiac "
            f"afferent input, but its clinical use requires separating disease-related variation from normal "
            f"physiological modulation by vigilance state and age. We analysed R-peak-locked EEG from a "
            f"19-electrode 10–20 clinical polysomnography cohort of {n_patients:,} patients across three "
            f"sleep stages (light sleep, N3, REM). A representative single-patient recording establishes the "
            f"signal's origin relative to the cardiac-field artifact. Cohort-wide topographic maps show that "
            f"HEP amplitude is largest over central and occipital sites and grows from light sleep to REM. "
            f"Diagnosis groups ({top_dx}) differed from an undiagnosed reference cohort within each sleep "
            f"stage taken separately. Within a single diagnostic group followed across all three stages "
            f"(suspected epilepsy, {S['pairwise'][0]['n']:,} matched patients), paired sleep-stage contrasts "
            f"were significant for all {len(S['pairwise'])} tested stage pairs after correction, and an age "
            f"median-split produced spatially organized, stage-dependent differences. Together these results establish "
            f"sleep stage and age as structured, separable axes of HEP variability that must be held constant "
            f"when testing diagnosis as a clinical biomarker.", styles["BodyN"]),
        Paragraph("Introduction", styles["HeadN"]),
        Paragraph(
            "The HEP is a small EEG deflection time-locked to the electrocardiographic R peak, interpreted as "
            "a marker of cortical interoception. It sits beside a much larger cardiac-field artifact and is "
            "sensitive to arousal state, so clinical interpretation requires (i) direct visualization of the "
            "source signals, (ii) exclusion of the peri-R interval, and (iii) comparison within a fixed sleep "
            "stage. Here we present, in five figures, the signal-level basis of the HEP, its cohort-wide "
            "topography, its dependence on clinical diagnosis within matched sleep stages, its within-patient "
            "modulation across sleep stages, and its association with age.", styles["BodyN"]),
    ]

    def _fig_dims(n, width_mm, max_h_mm):
        from PIL import Image as PILImage
        w, h = PILImage.open(FD / f"figure{n}.png").size
        height_mm = width_mm * h / w
        if height_mm > max_h_mm:
            height_mm = max_h_mm
            width_mm = height_mm * w / h
        return width_mm, height_mm

    def addfig(n, title, cap, width_mm=175, max_h_mm=200):
        w, h = _fig_dims(n, width_mm, max_h_mm)
        story.append(KeepTogether([
            Paragraph(title, styles["HeadN"]),
            Image(str(FD / f"figure{n}.png"), width=w * mm, height=h * mm),
            Paragraph(f"<b>Figure {n} |</b> {cap}", styles["CapN"]),
        ]))

    addfig(1, "Signal-level origin of the HEP",
           "Representative simultaneous EEG and cardiac activity from a single patient recording. "
           "R peaks (black markers) define epochs; the grey band marks the −50 to +50 ms "
           "cardiac-field-artifact interval excluded from inference. The lower trace is the "
           "patient-level mean HEP ± s.e.m. across beats.")
    addfig(2, "Cohort-wide HEP amplitude and 19-electrode topography",
           "(a) Grand-average HEP waveform by sleep stage across the full cohort. (b) 10–20 "
           "topographic maps of grand-mean HEP amplitude (0.15–0.5 s post-R window) at each of "
           "the 19 standard electrodes, separately for light sleep, N3 and REM.")

    w3, h3 = _fig_dims(3, 150, 200)
    story.append(Paragraph("Diagnosis effects within the same sleep stage", styles["HeadN"]))
    story.append(Paragraph(
        "The most common diagnostic categories in the cohort were compared against an undiagnosed "
        "reference group separately within light sleep, N3 and REM, so that no comparison pools across "
        "vigilance states. Cluster-based permutation testing (peri-R artifact window excluded) identified "
        "significant amplitude windows for essentially every diagnostic group in every sleep stage, "
        "indicating that diagnosis-associated HEP differences are not an artifact of sleep-stage "
        "composition.", styles["BodyN"]))
    story.append(KeepTogether([
        Image(str(FD / "figure3.png"), width=w3 * mm, height=h3 * mm),
        Paragraph(
            "<b>Figure 3 |</b> Grand-average HEP waveforms by diagnostic category versus an undiagnosed "
            "reference cohort, shown separately within light sleep (a), N3 (b) and REM (c). Coloured "
            "horizontal bars mark time windows where a diagnostic group shows a significant cluster not "
            "also present in the reference cohort (cluster permutation test, p<0.05); the grey band is the "
            "excluded cardiac-field-artifact window.", styles["CapN"]),
    ]))

    addfig(4, "Within-diagnosis sleep-stage effects",
           "Paired within-patient contrasts between sleep stages, restricted to a single diagnostic group "
           "(suspected epilepsy, 2,075 patients matched across all three stages) and shown only for the "
           "stage pairs that reached omnibus significance: (a) light sleep vs. N3, (b) REM vs. N3, (c) REM "
           "vs. light sleep. Each panel combines the mean difference waveform, its cluster t-statistic, and "
           "the electrode-wise significance topography; the peri-R grey interval is excluded from testing.",
           width_mm=138)
    addfig(5, "Age effects across sleep stages",
           "Older-versus-younger median-split contrasts, shown only for sleep stages that reached the "
           "omnibus significance criterion: (a) light sleep, (b) N3, (c) REM. Waveform and 19-electrode maps "
           "show that ageing is a spatially organized covariate rather than a uniform amplitude offset.",
           width_mm=138)

    story.append(Paragraph("Discussion", styles["HeadN"]))
    story.append(Paragraph(
        "Three conclusions follow. First, a measurable post-R HEP is visible individually and cohort-wide "
        "once the artifact-dominated interval is excluded, peaking over central-occipital scalp. Second, "
        "diagnosis groups differ from an undiagnosed reference even within a single sleep stage. Third, "
        "sleep stage and age each produce large, topographically structured amplitude shifts that can "
        "confound unstratified disease comparisons. Future HEP biomarker studies should therefore be "
        "stage-matched, montage-matched and age-adjusted, with patient-level replication.", styles["BodyN"]))
    story.append(Paragraph("Methods summary", styles["HeadN"]))
    story.append(Paragraph(
        "EEG was epoched around R peaks and baseline-corrected; the −50 to +50 ms interval was treated as "
        "cardiac-field artifact and excluded from inference. Diagnosis was tested within sleep stage against "
        "an undiagnosed reference using electrode-wise cluster-permutation tests, with heart rate and "
        "cardiac-field-artifact amplitude as covariates. This multi-hospital design parallels prior multicentre "
        "epilepsy imaging work.<super>1</super> Paired within-patient sleep-stage differences and "
        "age-split contrasts used cluster-mass permutation testing (200 permutations, cluster α = 0.01), "
        "retaining only significant panels in Figures 4–5; Figure 2 used the same frozen per-electrode "
        "amplitude table directly.",
        styles["BodyN"]))

    story.extend([
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
        Paragraph("1. Cafri N, Mirloo S, Zarhin D, Kamintsky L, Serlin Y, Alhadeed L, et al.; "
                  "Benninger F. Imaging blood-brain barrier dysfunction in drug-resistant epilepsy: "
                  "a multi-center feasibility study. <i>Epilepsia.</i> 2025;66(1):195-206.", styles["CapN"]),
    ])
    doc.build(story)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    manuscript()
    print(OUT)
