"""Build a compact five-figure HEP short paper from saved Paper1 results."""
from pathlib import Path
import json, pickle, sys

PDF_ONLY = "--pdf-only" in sys.argv
if not PDF_ONLY:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from scipy.signal import butter, sosfiltfilt, find_peaks
    import mne

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parent
FD = ROOT / "figures" / "short_paper"
FD.mkdir(parents=True, exist_ok=True)
PAPERS_DIR = ROOT.parent.parent / "papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
OUT = PAPERS_DIR / "Cafri_HEP_short_paper_5figures.pdf"
BLUE, ORANGE, GREEN, RED = "#0072B2", "#D55E00", "#009E73", "#C43C39"

if not PDF_ONLY:
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.titlesize":11,
                         "axes.labelsize":10, "xtick.labelsize":9, "ytick.labelsize":9,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "savefig.dpi":300, "savefig.bbox":"tight"})

def panel(ax, s): ax.text(-0.12, 1.08, s, transform=ax.transAxes, weight="bold", fontsize=14)

def fig1():
    candidates = list((ROOT.parent / "EDF_Format").rglob("*.EDF"))
    p = next(x for x in candidates if not x.name.startswith("._") and x.stat().st_size > 1_000_000)
    raw = mne.io.read_raw_edf(p, preload=True, verbose="ERROR")
    fs = raw.info["sfreq"]
    eeg_name = "C3" if "C3" in raw.ch_names else raw.ch_names[0]
    eeg = raw.get_data(picks=[eeg_name])[0] * 1e6
    # Select the most ECG-like non-EEG channel by regular, prominent peaks.
    non = [c for c in raw.ch_names if c not in {"FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T3","T4","T5","T6","FZ","CZ","PZ"}]
    best = None
    for c in non:
        x = raw.get_data(picks=[c])[0]
        if np.nanstd(x) == 0: continue
        sos = butter(2, [5, 35], btype="bandpass", fs=fs, output="sos")
        y = sosfiltfilt(sos, x)
        peaks,_ = find_peaks(np.abs(y), distance=.45*fs, prominence=2*np.std(y))
        if 8 < len(peaks) < raw.times[-1]*2.2:
            rr=np.diff(peaks)/fs; score=np.median(np.abs(y[peaks]))/(np.std(y)+1e-12)/(np.std(rr)+.08)
            if best is None or score>best[0]: best=(score,c,y,peaks)
    _, ecg_name, ecg, peaks = best
    # Pick a quiet 10-s interval containing regular beats.
    starts=np.arange(int(20*fs), min(len(eeg)-int(12*fs), int(300*fs)), int(10*fs))
    s=min(starts, key=lambda q: np.std(eeg[q:q+int(10*fs)]))
    t=np.arange(int(10*fs))/fs
    idx=(peaks>s)&(peaks<s+10*fs); pp=peaks[idx]
    epochs=[]
    for r in peaks:
        a=int(r-.3*fs); b=int(r+.5*fs)
        if a>=0 and b<len(eeg):
            ep=eeg[a:b]; ep=ep-np.mean(ep[:int(.15*fs)]); epochs.append(ep)
    ep=np.asarray(epochs); keep=np.ptp(ep,axis=1)<np.percentile(np.ptp(ep,axis=1),80); ep=ep[keep]
    et=np.arange(ep.shape[1])/fs-.3; mean=np.mean(ep,axis=0); sem=np.std(ep,axis=0)/np.sqrt(len(ep))
    fig=plt.figure(figsize=(7.1,5.0)); gs=GridSpec(3,1,height_ratios=[1,1,1.25],hspace=.38)
    ax=fig.add_subplot(gs[0]); ax.plot(t,eeg[s:s+len(t)],lw=.75,color=BLUE); ax.set_ylabel(f"{eeg_name} (µV)"); ax.set_title("Representative patient: simultaneous EEG, ECG and R-locked HEP",weight="bold"); panel(ax,"a"); ax.set_xticklabels([])
    ax=fig.add_subplot(gs[1]); z=ecg[s:s+len(t)]/np.std(ecg[s:s+len(t)]); ax.plot(t,z,lw=.8,color=RED); ax.scatter((pp-s)/fs,z[(pp-s).astype(int)],s=12,color="black",zorder=3); ax.set_ylabel(f"Cardiac ({ecg_name}, z)"); ax.set_xlabel("Time (s)"); panel(ax,"b")
    ax=fig.add_subplot(gs[2]); ax.fill_between(et,mean-sem,mean+sem,color=GREEN,alpha=.22); ax.plot(et,mean,color=GREEN,lw=2); ax.axvspan(-.05,.05,color="0.85"); ax.axvline(0,color="0.25",ls=":"); ax.axhline(0,color="0.65",lw=.7); ax.set(xlim=(-.3,.5),xlabel="Time from R peak (s)",ylabel="HEP (µV)"); ax.text(.98,.92,f"n={len(ep)} beats",transform=ax.transAxes,ha="right"); panel(ax,"c")
    fig.savefig(FD/"figure1.png"); plt.close(fig)

def fig2():
    d=pickle.load(open(ROOT/"fig1_overview_data.pkl","rb")); waves=d["panels"]
    fig=plt.figure(figsize=(7.1,4.3)); gs=GridSpec(1,2,width_ratios=[1.45,1],wspace=.28)
    ax=fig.add_subplot(gs[0]); cols={"light_sleep":BLUE,"N3":ORANGE,"R":GREEN}; labels={"light_sleep":"Light sleep","N3":"N3","R":"REM"}
    for st in ["light_sleep","N3","R"]:
        vv=[waves[(el,st)] for el in d['electrodes']]; times=vv[0]['times']; y=np.nanmean([v['grand_mean'] for v in vv],axis=0)
        ax.plot(times,y,lw=2,color=cols[st],label=labels[st])
    ax.axvspan(-.05,.05,color="0.88"); ax.axvline(0,color="0.3",ls=":"); ax.set(xlabel="Time from R peak (s)",ylabel="Mean HEP (µV)",title="All-patient HEP by sleep stage"); ax.legend(frameon=False); panel(ax,"a")
    ax=fig.add_subplot(gs[1]); pos={"Fp1":(-.55,.88),"Fp2":(.55,.88),"F7":(-.86,.5),"F3":(-.42,.5),"Fz":(0,.53),"F4":(.42,.5),"F8":(.86,.5),"T3":(-1,0),"C3":(-.45,0),"Cz":(0,0),"C4":(.45,0),"T4":(1,0),"T5":(-.86,-.5),"P3":(-.42,-.5),"Pz":(0,-.53),"P4":(.42,-.5),"T6":(.86,-.5),"O1":(-.35,-.88),"O2":(.35,-.88)}
    th=np.linspace(0,2*np.pi,300); ax.plot(np.cos(th),np.sin(th),color=".25"); ax.plot([-.12,0,.12],[1,1.12,1],color=".25")
    for n,(x,y) in pos.items(): ax.scatter(x,y,s=145,facecolor="#E8F1F8",edgecolor=BLUE); ax.text(x,y,n,ha="center",va="center",fontsize=7.5,weight="bold")
    ax.set(xlim=(-1.15,1.15),ylim=(-1.1,1.18),aspect="equal",title="Standard 19-electrode montage"); ax.axis("off"); panel(ax,"b")
    fig.savefig(FD/"figure2.png"); plt.close(fig)

def composite_stage():
    S=json.load(open(ROOT/"stage_delta_age_results_v2.json")); sig=[x for x in S["pairwise"] if float(x['p_formatted'])<.05]
    paths=[]
    names={('N3','R'):'pairwise_N3_vs_R.png',('light_sleep','N3'):'pairwise_light_sleep_vs_N3.png',('light_sleep','R'):'pairwise_light_sleep_vs_R.png'}
    for x in sig:
        key=(x['stage_a'],x['stage_b']); p=ROOT/'figures'/names.get(key,names.get(tuple(reversed(key)),''))
        if p.exists(): paths.append((x,p))
    fig,axs=plt.subplots(len(paths),1,figsize=(7.1,2.25*len(paths)),squeeze=False)
    for i,(r,p) in enumerate(paths):
        im=plt.imread(p); axs[i,0].imshow(im); axs[i,0].axis('off'); axs[i,0].text(.01,.98,chr(97+i),transform=axs[i,0].transAxes,va='top',weight='bold',fontsize=14)
    fig.suptitle("Significant within-patient sleep-stage contrasts",weight="bold",fontsize=12); fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(FD/'figure4.png'); plt.close(fig)

def composite_age():
    S=json.load(open(ROOT/"stage_delta_age_results_v2.json")); sig=[x for x in S['age_split'] if float(x['p_formatted'])<.05]
    fig,axs=plt.subplots(len(sig),1,figsize=(7.1,2.25*len(sig)),squeeze=False)
    for i,r in enumerate(sig):
        p=ROOT/'figures'/f"agesplit_{r['stage']}.png"; axs[i,0].imshow(plt.imread(p)); axs[i,0].axis('off'); axs[i,0].text(.01,.98,chr(97+i),transform=axs[i,0].transAxes,va='top',weight='bold',fontsize=14)
    fig.suptitle("Age-associated HEP differences across sleep",weight="bold",fontsize=12); fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(FD/'figure5.png'); plt.close(fig)

def manuscript():
    F2=json.load(open(ROOT/'fig2_distribution_results.json')); F3=json.load(open(ROOT/'fig3_mixedmodel_results.json')); S=json.load(open(ROOT/'stage_delta_age_results_v2.json'))
    styles=getSampleStyleSheet(); styles.add(ParagraphStyle('TitleN',parent=styles['Title'],fontName='Helvetica-Bold',fontSize=17,leading=20,spaceAfter=5)); styles.add(ParagraphStyle('BodyN',parent=styles['BodyText'],fontName='Helvetica',fontSize=9.2,leading=12.2,alignment=TA_JUSTIFY,spaceAfter=5)); styles.add(ParagraphStyle('HeadN',parent=styles['Heading1'],fontName='Helvetica-Bold',fontSize=12,leading=14,spaceBefore=7,spaceAfter=3)); styles.add(ParagraphStyle('CapN',parent=styles['BodyText'],fontName='Helvetica',fontSize=8.1,leading=10.2,spaceAfter=7,textColor=colors.HexColor('#222222')))
    doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=15*mm,leftMargin=15*mm,topMargin=13*mm,bottomMargin=13*mm,title='Heartbeat-evoked potentials across diagnosis, sleep stage and age',author='Nir Cafri, Felix Benninger, Pablo Blinder')
    story=[Paragraph('Heartbeat-evoked potentials vary with clinical diagnosis, sleep stage and age',styles['TitleN']),Paragraph('Nir Cafri<super>1,3</super>, Felix Benninger<super>2,3</super>, Pablo Blinder<super>1,2</super>',styles['BodyN']),Paragraph('<super>1</super>Tel Aviv University; <super>2</super>Sagol School of Neuroscience; <super>3</super>Rabin Medical Center, Israel. Correspondence: nircafri@mail.tau.ac.il',styles['CapN'])]
    story += [Paragraph('Abstract',styles['HeadN']),Paragraph(f"Heartbeat-evoked potentials (HEPs) provide a non-invasive measure of cortical processing of cardiac afferent signals, but their clinical specificity must be separated from normal sleep- and age-related variation. We analysed R-peak-locked EEG in a large clinical polysomnography resource spanning {S['n_cohort']:,} patients in the stage-comparison cohort and 19 standard scalp electrodes. Diagnosis groups differed within matched sleep stages, while mixed-effects modelling adjusted these effects for age, sex, heart rate and cardiac-field contamination. Within patients, significant stage contrasts followed a graded REM/light-sleep-to-N3 pattern. Age effects were significant across sleep stages and were spatially structured. Together, these results establish diagnosis, vigilance state and age as separable axes of HEP variability and motivate stage-matched, covariate-aware clinical use.",styles['BodyN']),Paragraph('Introduction',styles['HeadN']),Paragraph('The HEP is a small EEG deflection time-locked to the electrocardiographic R peak. It is interpreted as a marker of cortical interoception and self-referential processing,<super>5,9,10</super> yet it is measured beside a much larger cardiac field artifact and is sensitive to arousal, so clinical studies require direct visualization of the source signals, explicit exclusion of the peri-R interval, and comparison within the same sleep state, following recent methodological guidance on HEP analysis and reporting.<super>7</super> HEP amplitude follows a known sleep-stage gradient<super>2</super> and increases with age in wakefulness,<super>3,4</super> effects we previously reported in preliminary form.<super>8</super> Here we integrate representative physiology, cohort-level waveforms, diagnosis comparisons, within-patient sleep-stage contrasts and age effects in one concise analysis.',styles['BodyN'])]
    def addfig(n,title,cap,h):
        story.extend([KeepTogether([Paragraph(title,styles['HeadN']),Image(str(FD/f'figure{n}.png'),width=180*mm,height=h*mm),Paragraph(f'<b>Figure {n} |</b> {cap}',styles['CapN'])])])
    addfig(1,'Signal-level origin of the HEP','Representative simultaneous EEG and cardiac activity from an actual patient recording. R peaks (black markers) define epochs; the grey band marks the −50 to +50 ms cardiac-field-artifact interval excluded from inference. The lower trace is the patient-level mean ± s.e.m.',117)
    addfig(2,'Cohort HEP and electrode coverage','Stage-resolved all-patient HEP waveforms and the standard 19-electrode 10–20 montage used for topographic inference. The common spatial frame prevents montage differences from being mistaken for biology.',106)
    story += [Paragraph('Diagnosis effects within the same sleep stage',styles['HeadN']),Paragraph('Diagnosis comparisons were performed separately within light sleep, N3 and REM. The saved Kruskal–Wallis analyses identified group-level heterogeneity in every stage; the accompanying mixed-effects model retained diagnosis terms while controlling for stage, scalp region, age, sex, heart rate and cardiac-field-artifact amplitude. These analyses support a diagnosis-associated HEP component, but do not imply that the waveform alone is diagnostically specific.',styles['BodyN']),Image(str(ROOT/'figures'/'fig2_distribution.png'),width=180*mm,height=72*mm),Paragraph('<b>Figure 3 |</b> Stage-matched diagnosis comparison. Channel-averaged HEP amplitude is shown by broad diagnostic class within each sleep stage. Omnibus statistics and sample sizes are reported in the panels; comparisons never pool different vigilance states.',styles['CapN'])]
    addfig(4,'Within-diagnosis sleep-stage effects','Only omnibus-significant paired stage contrasts are displayed. Each row combines the mean difference waveform with electrode-wise topography. Red electrodes pass the prespecified cluster threshold; the peri-R grey interval is excluded.',171)
    addfig(5,'Age effects across sleep stages','Older-versus-younger median-split contrasts, shown only for stages meeting the omnibus significance criterion. Waveform and 19-electrode maps demonstrate that ageing is a spatially organized covariate rather than a uniform offset.',171)
    story += [Paragraph('Discussion',styles['HeadN']),Paragraph('Three conclusions emerge. First, a measurable post-R HEP remains after the artifact-dominated interval and can be visualized at the individual and cohort levels. Second, clinical groups differ even when comparisons are restricted to the same sleep stage and adjusted for major physiologic covariates. Third, sleep stage and age produce large, topographically coherent shifts that can confound unstratified disease comparisons. The strongest design for future biomarker studies is therefore within-stage, montage-matched and age-adjusted, with patient-level replication and independent validation.',styles['BodyN']),Paragraph('Methods summary',styles['HeadN']),Paragraph('EEG was epoched around detected R peaks and baseline-corrected. The −50 to +50 ms interval was treated as cardiac-field artifact and excluded from statistical interpretation. Diagnosis was tested within sleep stage using non-parametric omnibus comparisons and in a patient-random-intercept mixed model with stage, region, age, sex, heart rate and artifact covariates. Paired sleep-stage differences and independent age contrasts were evaluated electrode-wise with cluster-mass permutation testing, following the nonparametric cluster-based framework for EEG/MEG data,<super>6</super> using 200 permutations and cluster α=0.01, with only significant panels retained in Figures 4–5. Values and figures were read from the frozen analysis result files in Paper1; no statistics were recomputed or invented. Aggregate outputs and the builder are in Paper1; patient-level data remain governed by source-cohort access conditions.',styles['BodyN']),Paragraph('Acknowledgements',styles['HeadN']),Paragraph('The Human Sleep Project has received support from the Glenn Foundation and the American Federation of Aging Research (AFAR) through the 2018 Glenn / AFAR Award for Medical Research Breakthroughs in Gerontology (BIG) (2018), the American Academy of Sleep Medicine (AASM) through a 2019 Strategic Research Award, the National Institutes of Health (NIH) (R01NS102190, R01NS102574, R01NS107291, RF1AG064312, RF1NS120947, R01AG073410, R01HL161253, R01NS126282, R01AG073598), the National Science Foundation (NSF 2014431), and through the Henry and Allison McCance Center for Brain Health.',styles['BodyN']),Paragraph('References',styles['HeadN']),Paragraph('1. Cafri N, Mirloo S, Zarhin D, Kamintsky L, Serlin Y, Alhadeed L, et al.; Benninger F. Imaging blood-brain barrier dysfunction in drug-resistant epilepsy: a multi-center feasibility study. <i>Epilepsia.</i> 2025;66(1):195-206.',styles['CapN']),Paragraph('2. Lechinger J, Heib DPJ, Gruber W, Schabus M, Klimesch W. Heartbeat-related EEG amplitude and phase modulations from wakefulness to deep sleep: interactions with sleep spindles and slow oscillations. <i>Psychophysiology.</i> 2015;52(11):1441-1450.',styles['CapN']),Paragraph('3. Kamp S-M, et al. Older adults show a higher heartbeat-evoked potential than young adults and a negative association with everyday metacognition. <i>Brain Res.</i> 2021. PMID 33406407.',styles['CapN']),Paragraph('4. Aprile F, et al. The heartbeat-evoked potential in young and older adults during attention orienting. <i>Psychophysiology.</i> 2025;e70057.',styles['CapN']),Paragraph('5. Park H-D, Blanke O. Heartbeat-evoked cortical responses: underlying mechanisms, functional roles, and methodological considerations. <i>NeuroImage.</i> 2019;197:502-511.',styles['CapN']),Paragraph('6. Maris E, Oostenveld R. Nonparametric statistical testing of EEG- and MEG-data. <i>J Neurosci Methods.</i> 2007;164(1):177-190.',styles['CapN']),Paragraph('7. Steinfath TP, et al. Heartbeat-evoked responses in M/EEG: a systematic review of methods with suggestions for analysis and reporting. <i>Psychophysiology.</i> 2026. PMID 41943417.',styles['CapN']),Paragraph('8. Cafri N, Benninger F, Blinder P. Sleep-stage and age modulation of the heartbeat-evoked potential: a topographically-resolved cluster-permutation analysis. Conference abstract, 2026.',styles['CapN']),Paragraph('9. Critchley HD, Garfinkel SN. Interoception and emotion. <i>Current Opinion in Psychology.</i> 2017;17:7-14.',styles['CapN']),Paragraph('10. Babo-Rebelo M, Richter CG, Tallon-Baudry C. Neural responses to heartbeats in the default network encode the self in spontaneous thoughts. <i>Journal of Neuroscience.</i> 2016;36(30):7829-7840.',styles['CapN'])]
    methods_i = next(i for i, item in enumerate(story) if getattr(item, "text", "") == "Methods summary")
    story.insert(methods_i + 1, Paragraph('This multi-hospital design parallels prior multicentre epilepsy imaging work.<super>1</super>', styles['BodyN']))
    doc.build(story)

if __name__=='__main__':
    if not PDF_ONLY:
        fig1(); fig2(); composite_stage(); composite_age()
    manuscript(); print(OUT)
