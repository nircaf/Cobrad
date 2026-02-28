# Methodological Overview of \`6_hep_group_comparison.py\`

This document provides a high-level summary of the data processing pipeline, cleaning algorithms, and analytical functions implemented in `6_hep_group_comparison.py`. It is structured to facilitate adaptation for the methodology section of a scientific paper describing the extraction and analysis of Heartbeat Evoked Potentials (HEP).

---

## 1. High-Level Pipeline Overview

The script is a comprehensive analysis suite designed to process electrophysiological recordings (Electrocardiography [ECG] and Electroencephalography [EEG]), specifically to extract, clean, and analyze the brain's responses time-locked to the heartbeat (HEP).

The core pipeline executed by the script is as follows:
1. **Data Ingestion:** Loads continuous EDF recordings and extracts sampling frequencies.
2. **ECG Signal Polarity Correction:** Automatically detects and corrects inverted ECG leads.
3. **High-Fidelity ECG Cleaning:** Removes baseline wander, high-frequency artifacts, and extreme outliers while preserving the physiological morphology of the R-peaks.
4. **Robust R-peak Detection:** Identifies QRS complexes, refines peak localization, and enforces physiological minimum-distance constraints.
5. **Epoching and Averaging (HEP Computation):** Extracts peri-event segments of EEG data time-locked to the identified R-peaks and computes the average evoked potential.
6. **Statistical Evaluation:** Employs non-parametric, cluster-based permutation testing with subject-level jitter to calculate the statistical significance of HEP waveforms between groups or conditions.

---

## 2. Signal Processing and Cleaning Methodology

### 2.1 Upstream Pre-Cleaning (Pickled Data Framework)
Prior to the analyses run in `6_hep_group_comparison.py`, the core data ingestion pipeline often relies on pre-cleaned patient data aggregated by `HEP_parquet_generation.py` (via the `edf_cleaning.py` module). This upstream process efficiently packages raw continuous data into Pickle (`.pkl`) files after executing a robust pre-processing pipeline:
- **Resampling & Basic Filtering:** Signals are resampled to 256 Hz to standardize temporal resolution across subjects. Broad environmental noise is addressed via a 0.5 Hz high-pass filter, a low-pass filter (Nyquist - 0.1 Hz), and 50 Hz (with harmonics) notch filtering to eliminate electrical line noise.
- **Robust Channel Interpolation:** It uses the `PrepPipeline` algorithm (with robust manual variance and correlation thresholds as a fallback) to identify anomalous EEG channels, interpolating them over valid neighbors or dropping them if head-shape digitization is unavailable.
- **Artifact Rejection:** Extreme motion or environmental artifact epochs are filtered using the `AutoReject` algorithm before serialization.

By relying on these standardized, sanitized `.pkl` representations, the downstream HEP script ensures a stable source of neural data requiring minimal secondary EEG-level cleaning, isolating the remaining cleaning procedures exclusively to the ECG tracks.

### 2.2 ECG Polarity Correction
To ensure consistency in R-peak detection across different patients or recording setups, the pipeline applies an automated polarity check (`fix_inverted_ecg` and `decide_inversion_from_template`). 
- **Template Generation:** An initial pass detects peaks based on absolute signal amplitude. A median beat template is constructed by averaging localized epochs around these provisional peaks.
- **Inversion Criteria:** The algorithm compares the amplitude of the maximal positive deflection to the minimal negative deflection within the template. If the negative deflection is greater than 90% of the absolute positive peak, the ECG signal is considered inverted and subsequently flipped.

### 2.3 High-Fidelity ECG Cleaning
The `clean_ecg_high_fidelity` function implements a multi-stage filtering approach specifically tuned to preserve R-peak morphology and amplitude:
1. **Gentle Median Filtering:** A narrow 20 ms median filter is applied. This short window effectively dampens sudden spikes or "pops" in the signal without degrading the narrow structure of the R-wave.
2. **Bandpass Filtering:** The signal passes through a 2nd-order Butterworth bandpass filter (3.0 Hz – 40.0 Hz). This range effectively eliminates low-frequency baseline drift (e.g., respiration artifacts) and high-frequency muscle noise, isolating the primary energies of the cardiac cycle.
3. **Outlier Clipping (Z-Score Thresholding):** To handle extreme transient artifacts, samples with absolute Z-scores greater than a strict threshold (default 5.0) are clipped. Clipping, rather than median replacement, prevents "flat-top" artifacts from disrupting continuous signal topology.

### 2.4 Robust R-Peak Detection
The `detect_rpeaks_robust` function localizes the exact timing of ventricular depolarizations:
- **Primary Detection:** The algorithm initially leverages the well-validated `wfdb` XQRS detector to identify candidate QRS complexes.
- **Refinement:** The algorithm searches a 100 ms window (± 50 ms) around each candidate to find the exact local continuous maximum, ensuring millisecond-level precision critical for phase-locked HEP alignment.
- **Physiological Filtering:** A stringent minimum distance filter is applied. Peaks occurring within 501 ms of the preceding peak are rejected, successfully mitigating false-positive double-detections (e.g., mistaking T-waves for R-peaks).

---

## 3. Analytical Functions and Statistical Testing

### 3.1 Heartbeat Evoked Potential (HEP) Extraction
The \`compute_hep_avg\` and \`compute_ecg_hep_avg\` functions utilize the `pynapple` framework to generate continuous peri-event time series. By default, segments ranging from -0.3 seconds to +0.4 seconds relative to the R-peak are extracted. Segments are averaged within-subject to yield the individual participant's HEP waveform across all active EEG sensors.

### 3.2 Non-Parametric Permutation Testing
The routine uses \`permutation_cluster_jitter_test\` to contrast HEP waveforms (e.g., across patient cohorts or sleep stages). 
- **Method:** It applies a cluster-based permutation test with controlled temporal jitter. 
- **Jitter Mechanism:** Under the null hypothesis, each subject's data is randomly shifted (jittered) in time by up to a specified duration. Permutations are compared against the true clusters to evaluate significance while respecting the highly autocorrelated structure of electrophysiological data.

### 3.3 Repetitive Pattern / Noise Detection
A unique analytical module (\`analyze_ecg_repetitive_pattern\`) inspects the ECG for structural motifs. It extracts segments, calculates cross-correlations, and identifies highly recurrent, stereotyped signal segments against a randomized surrogate threshold, quantifying the repetition rate of these continuous templates across the recording.

---

## 4. Glossary of Core Functions

- **\`process_file_data(raw, patient_id)\`**: The master orchestration function that wraps loading, polarity fixation, high-fidelity cleaning, and robust peak detection for an individual EDF file.
- **\`fix_inverted_ecg(...)\` \/ \`decide_inversion_from_template(...)\`**: Constructs a median beat and forces correct geometric polarity for inverted leads.
- **\`clean_ecg_high_fidelity(...)\`**: The three-step sequential cleaner (median → bandpass → clipping) dedicated to R-wave preservation.
- **\`detect_rpeaks_robust(...)\`**: Multi-stage peak locator establishing absolute synchronization points for the epoching process.
- **\`compute_hep_avg(...)\`**: Computes the mean time-locked EEG response against the defined R-peak timestamps.
- **\`permutation_cluster_jitter_test(...)\`**: Evaluates statistical differences between HEP time-windows accounting for multiple comparisons.
- **\`handle_single_patient_view(...)\` & \`run_compare_groups_analysis(...)\`**: Graphical frontend handlers executing either single-subject comprehensive review or aggregate group-level plotting and testing.
