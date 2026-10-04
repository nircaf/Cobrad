# Brain–Heart Coupling Is State-Labile Within Nights but Its Nightly Mean Recurs Across Polysomnograms

## Five-Minute Variance Decomposition and Long-Term Repeat-PSG Reliability

**Original research article**

## Abstract

Brain–heart coupling is often compared between groups or sleep stages, but it is unknown whether it varies more across one person’s night or between people. We analyzed 8,408 quality-controlled, non-overlapping 5-minute windows from 159 real polysomnograms in 80 individuals with repeated studies. Coupling was the mean magnitude-squared coherence from 0.04–0.40 Hz between interpolated heart rate and delta-band (0.5–4 Hz) EEG amplitude envelopes across six standard channels. Unequal-cluster variance decomposition attributed 17.5% of total window variance to subjects (ICC 0.175, 95% bootstrap CI 0.100–0.253) and 82.5% to within-subject variation. Adjustment for sleep stage, time of night, stage purity, and beat count changed the ICC little (0.166, 0.087–0.224). Stage-specific ICCs ranged from 0.068 in REM to 0.494 in N3. Despite low five-minute reliability, the aggregated nightly mean recurred across 79 repeat pairs with good absolute agreement (ICC[A,1] 0.714; Pearson r=0.766) after a median 1.22 years and up to 6.90 years. A channel-by-stage fingerprint was only weakly identifying (4/75, 5.3%; chance 1.3%). Thus, momentary coupling is predominantly state-labile, while aggregation reveals a reproducible person-level coupling level. The data support a nightly scalar trait but not a strong multivariate biometric fingerprint.

**Keywords:** brain–heart coupling; polysomnography; heartbeat-evoked potential; intraclass correlation; variance components; test–retest reliability; physiological fingerprint; sleep

## 1. Introduction

The brain and heart interact continuously through autonomic, baroreceptive, respiratory, and central neural pathways.[1,2] During sleep, this interaction can be estimated under relatively standardized behavioral conditions and across several recurring physiological states.[3] Yet a basic measurement question remains unresolved. A coupling value observed in one segment may reflect a stable characteristic of the individual, transient variation over the night, sleep stage, recording noise, or cardiac field contamination of the electroencephalogram (EEG).[4,5]

Most studies summarize a recording at the participant level or test average differences between sleep stages and clinical groups. Averaging improves precision but hides the variation needed to determine whether coupling behaves like a trait. Sleep EEG spectra themselves show strong individual specificity across nights,[6–8] but it is unknown whether brain–heart coupling has comparable individuality. Repeated measurement within a night permits a direct decomposition: how much variability lies between people, and how much lies among windows from the same person? If between-person variance dominates after accounting for sleep state and technical covariates, coupling may be a physiological fingerprint. If within-person variance dominates, a single nightly summary is unlikely to characterize an individual reliably.

We applied a two-timescale analysis. First, coupling was calculated every 5 minutes throughout each PSG, and variance was partitioned into between-subject and within-subject components. Second, repeat PSGs tested whether the person-specific nightly level and multichannel pattern returned after months or years. This second test is essential because same-night stability can arise from montage, electrode impedance, posture, or persistent artifact. Long-term agreement across independently acquired nights provides a stronger trait criterion.

![Figure 1. Analysis overview. Every PSG is reduced to quality-controlled, non-overlapping 5-minute coupling estimates. A hierarchical model separates person-level from window-level variance; independent repeat PSGs then test whether the scalar level and multichannel fingerprint return.](figures/figure1_pipeline.png)

## 2. Research questions and hypotheses

### 2.1 Primary objective

We quantified the proportion of total 5-minute brain–heart coupling variance attributable to differences between subjects.

### 2.2 Secondary objectives

1. Determine how much between-subject variance remains after adjustment for sleep stage, time of night, stage purity, and accepted beat count.
2. Estimate ICC separately in wake, N1, N2, N3, and REM sleep.
3. Test whether person-level coupling estimates and multichannel coupling fingerprints reproduce in repeat PSGs separated by months or years.

### 2.3 Prespecified hypotheses

**H1:** Between-subject variance is greater than zero and the adjusted ICC is meaningfully above zero.

**H2:** The adjusted ICC is higher within a fixed sleep stage than in the unadjusted analysis because stage transitions contribute to within-person variability.

**H3:** Subject-level coupling values and spatial fingerprints show positive test–retest reliability across repeat PSGs.

**Trait interpretation rule:** Coupling supports an individual physiological trait only if person-level variation recurs in an independently acquired PSG. A high same-night ICC alone is insufficient, and a reproducible nightly scalar does not by itself establish a high-dimensional fingerprint.

![Figure 2. Real five-minute coupling trajectories and variance decomposition. Eight first-session PSGs spanning the distribution of nightly mean coupling are vertically offset for visualization; points are colored by sleep stage. Across all 8,408 retained windows, within-subject variance was substantially larger than between-subject variance.](figures/figure2_icc_estimand.png)

## 3. Methods

### 3.1 Study population

We screened the local Harvard polysomnography archive for subjects with at least two sessions containing synchronized EEG and ECG, sleep-stage annotations, at least four hours of recording, sampling rate of at least 100 Hz, and the common six-channel montage F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, and O2-M1. From eligible repeated subjects, 80 were selected using a fixed random seed (20260820), and the first two chronologically available sessions were analyzed. One session produced no windows meeting final stage and signal criteria, leaving 159 PSGs. Large-scale harmonized sleep resources demonstrate the feasibility and value of secondary PSG analyses.[9]

### 3.2 Five-minute windows

Each recording was divided from its start into consecutive, non-overlapping 300-second windows. Windows were not shifted to stage boundaries. The primary stage label was the modal label among ten 30-second epochs. Analyses required a recognized stage and at least 80% stage purity.

For every window we retained time from recording start, modal stage, stage purity, accepted heartbeat count, the six channel-level coupling values, and their robust median.

### 3.3 Signal preprocessing

ECG was zero-phase band-pass filtered at 5–25 Hz. R peaks were detected from the polarity with the stronger robust peak distribution using a 300-ms refractory period and prominence threshold of 2.5 median-absolute-deviation units. RR intervals outside 0.33–1.72 s were rejected. Windows required 150–900 detected peaks. EEG channels were rejected when flat, non-finite, or containing excursions greater than 30 robust standard deviations; at least four of six channels had to remain.

The metric was intentionally based on slow covariation between heart-rate dynamics and EEG delta amplitude rather than R-peak-locked EEG voltage, reducing direct sensitivity to the cardiac field. Nevertheless, electrical and motion contamination cannot be completely excluded from scalp PSG,[4,5] so “coupling” here denotes a reproducible EEG–ECG signal phenotype rather than proven directed neural communication.

### 3.4 Primary coupling metric

The primary metric was calculated independently in every 5-minute window. Instantaneous heart rate was obtained from accepted RR intervals and interpolated at 4 Hz. Each EEG channel was zero-phase filtered at 0.5–4 Hz; its analytic amplitude envelope was computed with the Hilbert transform, log transformed, and interpolated at 4 Hz. Magnitude-squared coherence between detrended heart rate and the delta envelope was estimated with 64-second segments and 50% overlap. Coupling was the mean coherence from 0.04–0.40 Hz, then the median across valid EEG channels. This band captures slow autonomic–cortical cofluctuation while excluding the heartbeat carrier frequency. Heart-rate processing followed established measurement principles.[10]

All filters, thresholds, channel names, window duration, seed, and frequency ranges are encoded in the accompanying analysis script and were applied identically across subjects and sessions.


### 3.5 Variance decomposition

Let y_ij denote coupling for subject i in 5-minute window j. The variance-components model is

y_ij = beta_0 + u_i + epsilon_ij,

where u_i ~ N(0, sigma_subject^2) is the subject-specific deviation and epsilon_ij ~ N(0, sigma_within^2) is window-level variation. Thus,

Var(y) = sigma_subject^2 + sigma_within^2,

and the intraclass correlation coefficient is

ICC = sigma_subject^2 / (sigma_subject^2 + sigma_within^2).

ICC is the expected correlation between two exchangeable 5-minute windows from the same subject. ICC form, model, unit, and agreement definition must be explicit because different formulations answer different questions.[11–13] This subject/within-subject decomposition parallels the intraclass effect decomposition (ICED) framework used to partition reliability of repeated neuroimaging measures.[15] Variance components were estimated by unequal-cluster one-way ANOVA method of moments. Negative between-subject estimates at the parameter boundary were truncated to zero. We report both components, ICC, and percentile 95% confidence intervals from 500 subject-level bootstrap resamples.

For the adjusted decomposition, coupling was first residualized using ordinary least squares with categorical sleep stage, linear and quadratic recording hour, stage purity, and accepted beat count. The same method-of-moments decomposition was then applied to residualized values. Subject-level resampling retained all windows from each selected subject and therefore preserved the observed serial dependence within recordings.

The unadjusted ICC answers how reproducible two randomly selected windows from the same person are in practice. The adjusted ICC asks whether individuals differ after measured state and technical factors are held constant.

![Figure 3. Overall and sleep-stage-specific ICCs from real PSG windows. Points show ICCs and bars show subject-bootstrap 95% confidence intervals. Adjustment for stage, time, purity, and beat count changed the overall estimate little; N3 had the largest point estimate and REM the smallest.](figures/figure3_state_adjustment.png)

### 3.6 Sleep-stage and temporal analyses

Separate variance components were estimated for wake, N1, N2, N3, and REM when at least 20 subjects contributed qualifying windows. Coupling trajectories were displayed against recording hour and sleep stage.

### 3.7 From a scalar trait to a physiological fingerprint

A single average may be stable even when spatial organization is not. We constructed a fingerprint vector for each subject and night containing mean coupling for each of six EEG channels within each available sleep stage. Pearson correlation measured fingerprint similarity. A fingerprint required at least 12 channel-by-stage elements present in both visits.

Fingerprint specificity was evaluated by blinded identification: for every repeat PSG, we selected the most correlated first-visit fingerprint among all eligible subjects. We report top-1 accuracy and median same-person and different-person correlations.

### 3.8 Repeat-PSG analysis

For subjects with two technically adequate PSGs, the visit-level scalar estimate was the mean of qualifying window coupling values. Inter-visit time was calculated from EDF recording start dates.

Long-term reliability was assessed with the two-way mixed-effects, single-measure absolute-agreement ICC(A,1).[11–13] We additionally calculated Pearson and Spearman correlations and Bland–Altman agreement.[14] Treatment and medication changes were unavailable and therefore could not be modeled.

Evidence that the fingerprint “comes back” was evaluated separately for the scalar nightly level and the multichannel-by-stage pattern. This distinction prevents good agreement of a nightly mean from being overstated as biometric identification.

![Figure 4. Repeat-PSG reliability of the real nightly mean. The left panel compares first and repeat PSG coupling with identity and fitted lines. The right panel shows Bland–Altman differences and 95% limits of agreement.](figures/figure4_reliability_design.png)

![Figure 5. Repeat-PSG multichannel-by-stage fingerprint performance. Same-person correlations were modestly higher than different-person correlations, but blinded top-1 identification remained low. The matrix shows the 30 repeat observations with the largest maximum similarity for visualization.](figures/figure5_repeat_fingerprint.png)

### 3.9 Missingness and weighting

No coupling outcomes were imputed. Unequal numbers of windows were retained because the variance estimator explicitly accounts for cluster size. The subject-level bootstrap resampled people, not individual windows, preventing long recordings from creating artificial independent replication.

## 4. Results

### 4.1 Cohort and usable windows

The analysis retained 8,408 five-minute windows from 159 PSG sessions in 80 subjects. Sessions contributed a median of 53 qualifying windows (265 minutes). All 80 subjects contributed to the within-person decomposition; 79 contributed two usable nightly means to the repeat-PSG analysis.

### 4.2 Within-night versus between-person variance

Between-subject variance was 0.000185 and within-subject variance was 0.000870 coherence-squared units. Therefore, 17.5% of total variance was attributable to subjects (ICC 0.175, 95% bootstrap CI 0.100–0.253), while 82.5% occurred among windows within subjects. After adjustment, between-subject variance was 0.000168 and within-subject variance was 0.000845, yielding ICC 0.166 (0.087–0.224). Adjustment therefore did not explain the predominance of within-person variability.

### 4.3 Sleep-stage-specific reliability

Within-night reliability varied by sleep state. N3 had the largest point estimate (ICC 0.494, 95% CI 0.068–0.591; 720 windows from 61 subjects), followed by N2 (0.312, 0.190–0.414; 4,331 windows from 80 subjects) and N1 (0.292, 0.107–0.424; 178 windows from 53 subjects). Reliability was low in wake (0.090, 0.059–0.122; 2,021 windows from 77 subjects) and REM (0.068, 0.026–0.104; 1,158 windows from 75 subjects). The wide N3 interval reflects fewer contributors and heterogeneous window counts.

### 4.4 Repeat-PSG recovery of the nightly scalar

Across 79 repeat pairs, the nightly mean showed good absolute agreement: ICC(A,1)=0.714. Rank and linear consistency were also positive (Pearson r=0.766; Spearman rho=0.632). The median interval was 1.22 years and the maximum was 6.90 years. Thus, a measure dominated by within-person variability at five minutes became reproducible when aggregated across the night.

### 4.5 Multichannel-by-stage fingerprint

Seventy-five subjects had at least 12 common channel-by-stage elements across both PSGs. The median correlation was 0.211 for the same person and 0.084 between different people. Nevertheless, only 4 of 75 repeat fingerprints selected the correct first PSG as their most similar candidate (5.3%, compared with 1.3% nominal chance). The scalar nightly level therefore carried substantially stronger repeat information than the detailed channel-by-stage pattern.

## 5. Interpretation

Two randomly selected five-minute windows from the same person were only modestly more similar than windows from different people. The metric is consequently state-labile at short timescales. In contrast, averaging many windows suppressed transient variation and exposed a person-level nightly mean that recurred in later PSGs.

The observed pattern corresponds to low within-night window reliability but high between-night reliability of aggregated means. It supports a weak momentary signal that stabilizes through aggregation, not a fixed coupling value throughout the night. The low identification accuracy further argues against describing the present multichannel representation as a biometric fingerprint.

ICC is population- and context-dependent: it can increase because subjects are more heterogeneous or decrease because measurement noise is greater. We therefore present raw variance components, cohort composition, window reliability, and long-term agreement alongside ICC. “Trait” denotes reproducible person-level variation under the studied recording conditions, not immutability or causal biological identity.

## 6. Discussion

This study asked a simple but consequential question: is brain–heart coupling more variable across one person’s night or across people? The answer depended on timescale. At five minutes, within-person variability was approximately five times the between-person component, and measured sleep state and time did not materially change that conclusion. At the nightly level, however, repeat PSG agreement was good even across intervals extending to 6.9 years.

The stage analysis shows that reliability is itself state-dependent. N3 approached moderate window-level ICC, whereas REM and wake were low. This difference may reflect more stationary cortical and autonomic physiology in consolidated N3, but the present observational analysis cannot distinguish physiology from stage-dependent signal quality. The stability of the nightly mean is consistent with prior demonstrations that sleep EEG contains individual-specific information,[6–8] while extending that idea to coordinated cardiac–cortical dynamics.

The scalar and multivariate results should not be conflated. Good ICC(A,1) of the nightly average indicates reproducible rank and absolute level after aggregation. Yet channel-by-stage correlations overlapped extensively between same-person and different-person pairs, and top-1 identification was low. The current representation is therefore useful as a repeatable scalar phenotype but inadequate as a biometric signature.

### 6.1 Limitations

This real-data study is a single-archive analysis of 80 reproducibly sampled subjects with a harmonized montage, and montage eligibility may limit generalizability. Repeat PSGs were clinically acquired rather than scheduled research retests; disease course, treatment, medication, and recording indication were unavailable. Windows were adjacent and temporally correlated, although subject-level bootstrap resampling preserved that dependence for confidence intervals. Coherence is nondirectional and may include common respiratory, arousal, movement, or residual cardiac-field influences. The analysis did not establish that the coupling is neural, nor did it compare alternate coupling definitions or window lengths. Finally, the random sample is adequate for the primary decomposition but gives imprecise N1 and N3 estimates.

### 6.2 Conclusion

Five-minute brain–heart coupling varied predominantly within individuals, with only 16.6% of adjusted variance attributable to subjects. Nevertheless, averaging across the night produced a scalar coupling level that returned with good absolute agreement months to years later. Brain–heart coupling is therefore not a fixed moment-to-moment fingerprint; it is a state-labile signal with a reproducible aggregated person-level component. Future work should test whether this scalar stability survives explicit cardiac-field correction, respiratory adjustment, treatment changes, and independent cohorts.

## Acknowledgements

The Human Sleep Project has received support from the Glenn Foundation and the American Federation of Aging Research (AFAR) through the 2018 Glenn / AFAR Award for Medical Research Breakthroughs in Gerontology (BIG) (2018), the American Academy of Sleep Medicine (AASM) through a 2019 Strategic Research Award, the National Institutes of Health (NIH) (R01NS102190, R01NS102574, R01NS107291, RF1AG064312, RF1NS120947, R01AG073410, R01HL161253, R01NS126282, R01AG073598), the National Science Foundation (NSF 2014431), and through the Henry and Allison McCance Center for Brain Health.

## References

1. Scheitz JF, Villringer A, Mikail N, Gebhard C, Endres M. Bidirectional brain–heart interactions in health and disease. Nat Rev Neurol. 2026;22:209–225. doi:10.1038/s41582-025-01180-w.
2. Critchley HD, Harrison NA. Visceral influences on brain and behavior. Neuron. 2013;77:624–638. doi:10.1016/j.neuron.2013.02.008.
3. de Zambotti M, Trinder J, Silvani A, Colrain IM, Baker FC. Dynamic coupling between the central and autonomic nervous systems during sleep: a review. Neurosci Biobehav Rev. 2018;90:84–103. doi:10.1016/j.neubiorev.2018.03.027.
4. Park HD, Blanke O. Heartbeat-evoked cortical responses: underlying mechanisms, functional roles, and methodological considerations. NeuroImage. 2019;197:502–511. doi:10.1016/j.neuroimage.2019.04.081.
5. Virjee RI, Kandasamy R, Garfinkel SN, Carmichael DW, Yogarajah M. Review of methods to derive the heartbeat-evoked potential: past practices and future directions. Soc Cogn Affect Neurosci. 2026:nsag057. doi:10.1093/scan/nsag057.
6. Finelli LA, Achermann P, Borbély AA. Individual “fingerprints” in human sleep EEG topography. Neuropsychopharmacology. 2001;25:S57–S62. doi:10.1016/S0893-133X(01)00320-7.
7. Lewandowski A, Rosipal R, Dorffner G. On the individuality of sleep EEG spectra. J Psychophysiol. 2013;27:105–112. doi:10.1027/0269-8803/a000092.
8. Eggert T, Dorn H, Danker-Hopfe H. The fingerprint-like pattern of nocturnal brain activity demonstrated in young individuals is also present in senior adulthood. Nat Sci Sleep. 2022;14:109–120. doi:10.2147/NSS.S336379.
9. Dean DA II, Goldberger AL, Mueller R, et al. Scaling up scientific discovery in sleep medicine: the National Sleep Research Resource. Sleep. 2016;39:1151–1164. doi:10.5665/sleep.5774.
10. Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. Heart rate variability: standards of measurement, physiological interpretation and clinical use. Circulation. 1996;93:1043–1065. doi:10.1161/01.CIR.93.5.1043.
11. Shrout PE, Fleiss JL. Intraclass correlations: uses in assessing rater reliability. Psychol Bull. 1979;86:420–428. doi:10.1037/0033-2909.86.2.420.
12. McGraw KO, Wong SP. Forming inferences about some intraclass correlation coefficients. Psychol Methods. 1996;1:30–46. doi:10.1037/1082-989X.1.1.30.
13. Koo TK, Li MY. A guideline of selecting and reporting intraclass correlation coefficients for reliability research. J Chiropr Med. 2016;15:155–163. doi:10.1016/j.jcm.2016.02.012.
14. Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. Lancet. 1986;1:307–310. doi:10.1016/S0140-6736(86)90837-8.
15. Brandmaier AM, Wenger E, Bodammer NC, Kühn S, Raz N, Lindenberger U. Assessing reliability in neuroimaging research through intra-class effect decomposition (ICED). eLife. 2018;7:e35718. doi:10.7554/eLife.35718.
