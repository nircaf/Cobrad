import os

import streamlit as st

import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, find_peaks
from scipy import stats
from scipy.stats import kurtosis
from scipy.ndimage import label
from typing import Tuple
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_peaks(signal_data, use_smart_threshold=True, mad_multiplier=10, **kwargs):
    """
    Centralized function to find peaks in a signal.
    By default, calculates a robust smart threshold (different for each patient)
    based on the signal's Median Absolute Deviation (MAD), unless 'height' 
    is explicitly provided in kwargs.
    Wraps scipy.signal.find_peaks.
    """
    if use_smart_threshold and 'height' not in kwargs:
        # Calculate dynamic threshold for the specific patient's signal 
        mad = np.median(np.abs(signal_data - np.median(signal_data))) + 1e-12
        smart_height = np.median(signal_data) + mad_multiplier * mad
        kwargs['height'] = smart_height

    return find_peaks(signal_data, **kwargs)

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynapple as nap
    PYNAPPLE_AVAILABLE = True
except ImportError:
    PYNAPPLE_AVAILABLE = False
    st.error("Pynapple (nap) not installed. Please install it.")


try:
    import heartpy as hp
    HEARTPY_AVAILABLE = True
except ImportError:
    HEARTPY_AVAILABLE = False

import io

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

st.set_page_config(layout="wide")
_original_st_pyplot = st.pyplot

def custom_st_pyplot(fig=None, clear_figure=None, **kwargs):
    if fig is None:
        fig = plt.gcf()
        
    title = "Figure"
    try:
        if fig._suptitle:
            title = fig._suptitle.get_text()
        else:
            for ax in fig.axes:
                if ax.get_title():
                    title = ax.get_title()
                    break
    except Exception:
        pass
        
    description = f"Plot: {title}\n"
    
    try:
        stats = []
        for ax in fig.axes:
            lines = ax.get_lines()
            for line in lines:
                ydata = line.get_ydata()
                if len(ydata) > 0:
                    # check if ydata is numeric
                    if hasattr(ydata, 'dtype') and np.issubdtype(ydata.dtype, np.number):
                        label = line.get_label()
                        stat_line = f"  {label if label and not label.startswith('_') else 'Data Line'}: Min={np.nanmin(ydata):.3f}, Max={np.nanmax(ydata):.3f}, Mean={np.nanmean(ydata):.3f}"
                        if stat_line not in stats:
                            stats.append(stat_line)
        if stats:
            description += "Numerical Summary:\n" + "\n".join(stats)
            
        # extract histogram data if present
        hist_stats = []
        for ax in fig.axes:
            patches = ax.patches
            if len(patches) > 5: # likely a histogram
                heights = [p.get_height() for p in patches if isinstance(p, plt.Rectangle)]
                if heights:
                    hist_stats.append(f"  Histogram: {sum(heights):.0f} total items, max bin freq={max(heights):.0f}")
        if hist_stats:
            description += "\n" + "\n".join(hist_stats)
    except Exception:
        pass
        

    if 'pptx_figures_data' not in st.session_state:
        st.session_state.pptx_figures_data = []

    _original_st_pyplot(fig, clear_figure=clear_figure, **kwargs)

    if PPTX_AVAILABLE:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        st.session_state.pptx_figures_data.append({
            'title': title,
            'description': description,
            'image': buf
        })

st.pyplot = custom_st_pyplot

def generate_pptx():
    if not PPTX_AVAILABLE:
        return None

    prs = Presentation()

    try:
        title_only_slide_layout = prs.slide_layouts[5]
    except IndexError:
        title_only_slide_layout = prs.slide_layouts[0]

    for item in st.session_state.get('pptx_figures_data', []):
        slide = prs.slides.add_slide(title_only_slide_layout)

        if slide.shapes.title:
            slide.shapes.title.text = item['title']

        pic_left = Inches(0.5)
        pic_top = Inches(1.5)
        pic_width = Inches(5.5)

        item['image'].seek(0)
        slide.shapes.add_picture(item['image'], pic_left, pic_top, width=pic_width)

        txBox_left = Inches(6.2)
        txBox_top = Inches(1.5)
        txBox_width = Inches(3.3)
        txBox_height = Inches(5.0)

        txBox = slide.shapes.add_textbox(txBox_left, txBox_top, txBox_width, txBox_height)
        tf = txBox.text_frame
        tf.word_wrap = True
        
        # Add paragraphs
        p = tf.paragraphs[0]
        p.text = item['description']
        p.font.size = Pt(14)

    pptx_io = io.BytesIO()
    prs.save(pptx_io)
    pptx_io.seek(0)
    return pptx_io

def bandpass_ecg(x, fs, lo=3.0, hi=40.0, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lo/nyq, hi/nyq], btype="bandpass")
    return signal.filtfilt(b, a, x)

def detect_rpeaks_abs(x_filt, fs):
    # Make it robust to inversion: detect on absolute signal
    x = np.abs(x_filt)

    # Smooth a bit to emphasize QRS energy
    win = int(0.08 * fs)  # ~80 ms
    win = max(win, 3) | 1  # odd >= 3
    x_smooth = signal.savgol_filter(x, win, polyorder=2)

    # Adaptive-ish threshold using MAD
    mad = np.median(np.abs(x_smooth - np.median(x_smooth))) + 1e-12
    thr = np.median(x_smooth) + 3.5 * mad

    # Refractory period to avoid double detections
    distance = int(0.25 * fs)  # 250 ms
    peaks, _ = get_peaks(x_smooth, height=thr, distance=distance)
    return peaks

def decide_inversion_from_template(x_filt, rpeaks, fs, qrs_ms=120):
    # Build an average beat template centered at R
    half = int((qrs_ms / 1000.0) * fs)  # +- qrs_ms
    if len(rpeaks) < 5:
        return False, None  # not enough info

    # Keep only peaks that allow full window extraction
    rpeaks = rpeaks[(rpeaks > half) & (rpeaks < len(x_filt) - half)]
    if len(rpeaks) < 5:
        return False, None

    beats = np.stack([x_filt[p-half:p+half+1] for p in rpeaks], axis=0)

    # Robust average: median across beats resists outliers
    template = np.median(beats, axis=0)

    # Polarity decision: is the dominant QRS deflection negative?
    pos_peak = np.max(template)
    neg_peak = np.min(template)  # negative value
    
    # FLIP ONLY the scans were 90% of max is smaller than abs(min) of the r peaks.
    inverted = (abs(neg_peak) > 0.9 * abs(pos_peak))

    return inverted, template

def fix_inverted_ecg(ecg, fs, lo=3.0, hi=40.0):
    ecg = np.asarray(ecg).astype(float)

    ecg_f = bandpass_ecg(ecg, fs, lo=lo, hi=hi)
    rpeaks = detect_rpeaks_abs(ecg_f, fs)

    inverted, template = decide_inversion_from_template(ecg_f, rpeaks, fs)

    ecg_fixed = -ecg if inverted else ecg
    info = {
        "inverted_detected": bool(inverted),
        "n_rpeaks_used": int(len(rpeaks)),
        "rpeaks": rpeaks,
        "template_filtered": template,  # median beat on filtered signal
    }
    return ecg_fixed, info

def clean_ecg_high_fidelity(ecg_signal, sampling_rate, wavelet='db1', wavelet_levels=3, zscore_threshold=5.0):
    """
    Improved ECG cleaning to preserve R-peak amplitude and HEP components.
    """
    if len(ecg_signal) == 0:
        return ecg_signal, {'methods_applied': [], 'n_extreme_samples': 0}
    
    methods_applied = []
    info = {}
    cleaned_signal = ecg_signal.copy()
    
    # --- Step 1: Gentle Median Filter ---
    # REDUCED: 300ms was killing the R-peak. 
    # 10-20ms is enough to kill 'pops' without touching the R-wave.
    median_window_ms = 20 
    median_window_samples = int(np.round(median_window_ms * sampling_rate / 1000.0))
    if median_window_samples % 2 == 0:
        median_window_samples += 1
    
    median_filtered = median_filter(cleaned_signal, size=max(3, median_window_samples))
    cleaned_signal = median_filtered
    methods_applied.append('gentle_median_filter')
    
    # --- Step 2: Bandpass Filter (3-40 Hz) ---
    # Replaces Wavelet Denoising as per user request.
    try:
        lowcut = .5
        highcut = 40.0
        nyquist = 0.5 * sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(2, [low, high], btype='band') # Order 2 is usually sufficient for ECG
        
        cleaned_signal = filtfilt(b, a, cleaned_signal)
        methods_applied.append('bandpass_3_40Hz')
    except Exception as e:
        if 'st' in globals():
            st.warning(f"Bandpass filter failed: {e}")
        # Fallback to original signal if filter fails
        cleaned_signal = cleaned_signal

    # --- Step 3: Outlier Handling (Z-score) ---
    # INCREASED: 3.0 often catches the R-wave itself. 5.0-6.0 catches actual artifacts.
    mean_val = np.mean(cleaned_signal)
    std_val = np.std(cleaned_signal)
    if std_val > 0:
        z_scores = (cleaned_signal - mean_val) / std_val
        extreme_mask = np.abs(z_scores) > zscore_threshold
        # Instead of replacing with median, consider clipping or ignoring 
        # to avoid 'flat-top' peaks.
        cleaned_signal[extreme_mask] = np.sign(cleaned_signal[extreme_mask]) * (zscore_threshold * std_val)
        methods_applied.append('zscore_clipping')

    info['methods_applied'] = methods_applied
    return cleaned_signal, info

def detect_rpeaks_robust(ecg_signal_clean, sfreq, return_log=False):
    """
    Detects R-peaks using WFDB xqrs_detect, refines them to local maxima within 0.05s window,
    and filters them to ensure a minimum distance of 501ms between consecutive peaks.
    """
    try:
        import wfdb.processing as wp
        # 1. WFDB Detection
        rpeaks_wfdb = wp.xqrs_detect(ecg_signal_clean, fs=sfreq, verbose=False)
        
        # 2. Refine to local maxima
        if len(rpeaks_wfdb) > 0:
            rpeaks = wp.correct_peaks(ecg_signal_clean, rpeaks_wfdb, 
                                      search_radius=int(0.05*sfreq),
                                      smooth_window_size=int(0.1*sfreq))
            rpeaks = np.unique(rpeaks)
            rpeaks = np.sort(rpeaks)

            rpeaks_sec = rpeaks / sfreq
            rr_intervals = np.diff(rpeaks_sec)
            unique_rr, counts = np.unique(rr_intervals, return_counts=True)
            
            mask = np.ones(len(rpeaks), dtype=bool)
            repetitive_info = []
            for rr, count in zip(unique_rr, counts):
                if count > len(rr_intervals) * 0.1:
                    perc = count / len(rr_intervals) * 100
                    msg = f"RR interval {rr:.3f}s appears {count} times ({perc:.1f}%)"
                    repetitive_info.append(msg)
                    bad_idx = np.where(rr_intervals == rr)[0] + 1
                    mask[bad_idx] = False
            
            removed_perc = np.sum(~mask) / len(mask) * 100
            log_msg = None
            if repetitive_info or removed_perc > 0:
                log_msg = {
                    'total': len(mask),
                    'removed': np.sum(~mask),
                    'perc': removed_perc,
                    'info': repetitive_info,
                    'skipped': removed_perc >= 25.0
                }
            
            if np.sum(~mask) >= 0.25 * len(mask):
                if return_log: 
                    return np.array([], dtype=int), log_msg
                return np.array([], dtype=int)
                
            rpeaks = rpeaks[mask]
            
            # 3. Filter peaks: if < 550ms between each other, keep first, remove second
            min_dist = int(0.4 * sfreq)
            if len(rpeaks) > 0:
                filtered_rpeaks = [rpeaks[0]]
                for i in range(1, len(rpeaks)):
                    if rpeaks[i] - filtered_rpeaks[-1] >= min_dist:
                        filtered_rpeaks.append(rpeaks[i])
                rpeaks = np.array(filtered_rpeaks)
                
        if return_log:
            return rpeaks, log_msg
        return rpeaks
    except Exception:
        if return_log:
            return np.array([], dtype=int), None
        return np.array([], dtype=int)


def detect_eeg_inversion(ch_data, sfreq, threshold=0.8):
    """
    Detects if an EEG channel is inverted by comparing local maxima and minima.
    If the absolute value is higher in the local minimum at more than threshold (80%),
    the channel is considered inverted.
    """
    # Bandpass filter 1-40Hz to remove DC offset and high frequency noise
    nyq = 0.5 * sfreq
    b, a = butter(2, [1.0/nyq, 40.0/nyq], btype='bandpass')
    filt_data = filtfilt(b, a, ch_data.astype(float))
    
    # Find all local maxima and minima with a small prominence to ignore noise
    prom = np.std(filt_data) * 0.1
    peaks, _ = find_peaks(filt_data, prominence=prom)
    troughs, _ = find_peaks(-filt_data, prominence=prom)
    
    if len(peaks) < 5 or len(troughs) < 5:
        return False
        
    is_inverted_count = 0
    total_comparisons = 0
    
    # For each trough, find the nearest peaks before and after and compare magnitude
    for t_idx in troughs:
        before = peaks[peaks < t_idx]
        after = peaks[peaks > t_idx]
        if len(before) == 0 or len(after) == 0:
            continue
        
        # Determine the "local" peak magnitude (max of the adjacent peaks)
        p_near = max(filt_data[before[-1]], filt_data[after[0]])
        t_val = np.abs(filt_data[t_idx])
        
        if t_val > p_near:
            is_inverted_count += 1
        total_comparisons += 1
        
    if total_comparisons == 0:
        return False
        
    return (is_inverted_count / total_comparisons) > threshold

def drop_non_eeg_channels(raw):
    """
    Remove non-EEG electrodes from the raw object using a single drop_channels call.
    """
    to_drop = [ch for ch in ['LOC', 'SpO2','eog'] if ch in raw.ch_names]
    prefixes = ('el', 'trx', 'png', 'beat','pul')
    
    for ch_name in raw.ch_names:
        if ch_name.lower().startswith(prefixes) and ch_name not in to_drop:
            to_drop.append(ch_name)
            
    if to_drop:
        raw.drop_channels(to_drop)
    return raw

def process_file_data(raw, patient_id):
    """
    Cleans ECG and extracts R-peaks using the provided logic.
    Returns the raw data, sfreq, and R-peak times if successful.
    """
    # Parameters
    minmax = (-0.3, .4)
    
    # Get sampling frequency and data
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    data = raw.get_data() * 1e6
    
    # Process ECG
    ch_lower = [ch.lower() for ch in ch_names]
    ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
    
    if not ecg_indices:
        try:
            st.warning(f"No ECG channel found in {patient_id}")
        except Exception:
            print(f"WARNING: No ECG channel found in {patient_id}")
        return None
        
    ecg_ch_idx = ecg_indices[0]
    ecg_signal = data[ecg_ch_idx, :]
    
    # EEG inversion is now handled in get_group_individuals after HEP extraction
    

    # Fix inverted ECG
    ecg_signal, _ = fix_inverted_ecg(ecg_signal, sfreq)
    
    # Clean ECG
    ecg_signal_clean, _ = clean_ecg_high_fidelity(
        ecg_signal, 
        sampling_rate=sfreq
    )
    # Detect R-peaks using robust method
    rpeaks, log_msg = detect_rpeaks_robust(ecg_signal_clean, sfreq, return_log=True)

    if hasattr(raw, '_data'):
        raw._data[ecg_ch_idx, :] = ecg_signal_clean / 1e6

    if len(rpeaks) < 2:
        try:
            st.warning(
                f"Not enough R-peaks found for patient **{patient_id}** "
                f"(detected: {len(rpeaks)}). "
                f"Reason: {log_msg}"
            )
        except Exception:
            print(f"WARNING: Not enough R-peaks for {patient_id} (detected: {len(rpeaks)}). {log_msg}")
        return None

    rpeak_times = rpeaks / sfreq
    rpeak_ts = nap.Ts(t=rpeak_times, time_units="s")
    return raw, sfreq, rpeak_ts, rpeaks, minmax, log_msg


def compute_hep_avg(raw, rpeaks, sfreq, minmax=(-0.5, 1.0), rpeak_ts=None):
    """
    Computes HEP (Heartbeat Evoked Potential) for all EEG channels.
    """
    tmin, tmax = minmax
    
    # Identify EEG channels
    ch_names = raw.ch_names
    eeg_indices = [i for i, ch in enumerate(ch_names) 
                  if 'eeg' in ch.lower() or any(elec in ch.upper() for elec in ['FP', 'F', 'C', 'P', 'O', 'T', 'A'])]
    
    if not eeg_indices:
        return None, None, None
        
    # Pick EEG channels
    eeg_ch_names = [ch_names[i] for i in eeg_indices]
    
    # Pynapple implementation
    data = raw.get_data(picks=eeg_indices).T # (n_times, n_channels)
    tsd_frame = nap.TsdFrame(t=raw.times, d=data, columns=eeg_ch_names)
    perievent = nap.compute_perievent_continuous(tsd_frame, rpeak_ts, minmax=minmax)
    # Average across trials (axis 1)
    mean_data = perievent.nanmean(axis=1).values.T # (n_channels, n_times)
    return mean_data, perievent.t, eeg_ch_names



def compute_ecg_hep_avg(raw, rpeaks, sfreq, minmax=(-0.5, 1.0), rpeak_ts=None):
    """
    Computes HEP (Heartbeat Evoked Potential) for the ECG channel.
    """
    tmin, tmax = minmax
    ch_names = raw.ch_names
    ch_lower = [ch.lower() for ch in ch_names]
    ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
    
    if not ecg_indices:
        return None, None, None
        
    ecg_ch_name = ch_names[ecg_indices[0]]

    # Pynapple implementation
    data = raw.get_data(picks=[ecg_indices[0]]).T # (n_times, 1)
    tsd_frame = nap.TsdFrame(t=raw.times, d=data, columns=[ecg_ch_name])
    perievent = nap.compute_perievent_continuous(tsd_frame, rpeak_ts, minmax=minmax)
    # Average across trials (axis 1)
    mean_data = perievent.nanmean(axis=1).values.T # (1, n_times)
    return mean_data, perievent.t, [ecg_ch_name]

def process_and_invert_hep(_raw, rpeaks, sfreq, minmax, _rpeak_ts, patient_id):
    """
    Computes HEP and ECG HEP. Evaluates the HEP peak polarities.
    If >= 80% of an odd or even channel group are negative (abs max),
    those raw channels are inverted and the HEP logic is re-run.
    """
    hep_data, times, ch_names = compute_hep_avg(_raw, rpeaks, sfreq, minmax, rpeak_ts=_rpeak_ts)
    ecg_hep_data, _, ecg_ch_names = compute_ecg_hep_avg(_raw, rpeaks, sfreq, minmax, rpeak_ts=_rpeak_ts)

    # Extract raw ECG signal (1D) for quality checks
    ch_lower = [ch.lower() for ch in _raw.ch_names]
    ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
    ecg_data = _raw.get_data(picks=[ecg_indices[0]]).squeeze() if ecg_indices else None

    if hep_data is None:
        return hep_data, times, ch_names, ecg_hep_data, ecg_ch_names, ecg_data

    # Match standard 10-20 EEG channel names: letter(s)+digit(s) OR letter+z (midline like Fz, Cz, Pz, Oz)
    eeg_indices = [i for i, ch in enumerate(ch_names)
                  if re.match(r'^[A-Za-z]{1,3}[0-9]+$', ch) or re.match(r'^[A-Za-z]{1,2}z$', ch, re.IGNORECASE)]

    eeg_data = [hep_data[i] for i in eeg_indices if len(hep_data[i]) > 0]
    flipped_channels = []

    if eeg_data:
        avg_all_eeg = np.nanmean(eeg_data, axis=0)

        # Check if the average of all electrodes is inverted
        if abs(np.nanmin(avg_all_eeg)) > abs(np.nanmax(avg_all_eeg)):
            raw_ch_names = _raw.ch_names
            for idx in eeg_indices:
                ch_name = ch_names[idx]
                if ch_name in raw_ch_names:
                    raw_idx = raw_ch_names.index(ch_name)
                    _raw._data[raw_idx, :] = -_raw._data[raw_idx, :]
                    flipped_channels.append(ch_name)

    if flipped_channels:
        try:
            st.info(f"Flipped inverted EEG channels for {patient_id} (based on Average of All EEG): {', '.join(flipped_channels)}")
        except Exception:
            print(f"Flipped inverted EEG channels for {patient_id} (based on Average of All EEG): {', '.join(flipped_channels)}")

        hep_data, times, ch_names = compute_hep_avg(_raw, rpeaks, sfreq, minmax, rpeak_ts=_rpeak_ts)
        ecg_hep_data, _, ecg_ch_names = compute_ecg_hep_avg(_raw, rpeaks, sfreq, minmax, rpeak_ts=_rpeak_ts)

    return hep_data, times, ch_names, ecg_hep_data, ecg_ch_names, ecg_data


def _apply_ica_ecg_removal(raw, patient_id):
    """
    Applies MNE ICA to remove ECG artifact components from EEG channels.
    Finds components correlated with the ECG channel and excludes them.
    Returns the cleaned raw object (in-place modification).
    """
    try:
        ch_names = raw.ch_names
        ch_lower = [ch.lower() for ch in ch_names]
        ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
        if not ecg_indices:
            print(f"[ICA] No ECG channel found for {patient_id}, skipping ICA.")
            return raw

        ecg_ch_name = ch_names[ecg_indices[0]]
        eeg_picks = mne.pick_types(raw.info, eeg=True, ecg=False, stim=False, exclude='bads')
        if len(eeg_picks) < 2:
            print(f"[ICA] Not enough EEG channels for ICA in {patient_id}, skipping.")
            return raw

        n_components = min(15, len(eeg_picks) - 1)
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method='fastica',
            random_state=42,
            max_iter=200,
        )
        raw_filt = raw.copy().filter(1.0, None, picks=eeg_picks, verbose=False)
        ica.fit(raw_filt, picks=eeg_picks, verbose=False)
        ecg_inds, _ = ica.find_bads_ecg(raw, ch_name=ecg_ch_name, verbose=False)
        if ecg_inds:
            ica.exclude = ecg_inds
            ica.apply(raw, verbose=False)
            print(f"[ICA] {patient_id}: removed {len(ecg_inds)} ECG component(s): {ecg_inds}")
        else:
            print(f"[ICA] {patient_id}: no ECG components identified.")
    except Exception as e:
        print(f"[ICA] Error during ICA for {patient_id}: {e}")
    return raw


def _process_patient_worker(args):
    """Module-level worker for ProcessPoolExecutor. Returns result tuple or None on failure."""
    if len(args) == 3:
        f_path, patient_id, apply_ica = args
    else:
        f_path, patient_id = args
        apply_ica = False
    try:
        with open(f_path, 'rb') as f:
            raw = pickle.load(f)
        raw = drop_non_eeg_channels(raw)
        results = process_file_data(raw, patient_id)
        if results is None:
            return None
        raw, sfreq, rpeak_ts, rpeaks, minmax, log_msg = results
        if apply_ica:
            raw = _apply_ica_ecg_removal(raw, patient_id)
        hep_data, times, ch_names, ecg_hep_data, ecg_ch_names, ecg_data = process_and_invert_hep(
            raw, rpeaks, sfreq, minmax, rpeak_ts, patient_id
        )

        # Skewness-based inversion: if avg EEG HEP waveform has negative skewness, flip EEG channels
        if hep_data is not None and len(times) > 0:
            eeg_idx = [i for i, ch in enumerate(ch_names)
                       if re.match(r'^[A-Za-z]{1,3}[0-9]+$', ch) or re.match(r'^[A-Za-z]{1,2}z$', ch, re.IGNORECASE)]
            if eeg_idx:
                flipped = []
                for i in eeg_idx:
                    if stats.skew(hep_data[i]) < 0:
                        hep_data[i] = -hep_data[i]
                        flipped.append(ch_names[i])
                if flipped:
                    print(f"[{patient_id}] Skewness-based inversion: {', '.join(flipped)}")

        return (patient_id, hep_data, times, ch_names, rpeaks, ecg_hep_data, ecg_ch_names, log_msg)
    except Exception as e:
        print(f"[Worker] Error processing {patient_id}: {e}")
        return None


@st.cache_data
def get_group_individuals(group_name, sleep_stage, base_path, test_run=False, recompute_cache=False, apply_ica=False):
    """
    Loads all files for a group/sleep_stage and returns individual HEPs.
    Returns: list of (patient_id, hep_data, times, ch_names)
    """
    group_dir = os.path.join(base_path, group_name, sleep_stage)
    if not os.path.exists(group_dir):
        return []

    # Exclude cache files when looking for patient pkl files
    patient_files = [f for f in os.listdir(group_dir) if f.endswith('.pkl') and not f.startswith('individuals_cache')]
    if not patient_files:
        return []
        
    ica_suffix = '_ica' if apply_ica else ''
    cache_filename = f'individuals_cache_test{ica_suffix}.pkl' if test_run else f'individuals_cache{ica_suffix}.pkl'
    cache_path = os.path.join(group_dir, cache_filename)
    
    # Check if cache exists and is newer than all patient files. Disable cache reading for test runs or if recompute is forced.
    if os.path.exists(cache_path) and not test_run and not recompute_cache:
        cache_mtime = os.path.getmtime(cache_path)
        is_cache_valid = True
        for f in patient_files:
            if os.path.getmtime(os.path.join(group_dir, f)) > cache_mtime:
                is_cache_valid = False
                break
                
        if is_cache_valid:
            try:
                with open(cache_path, 'rb') as f:
                    individuals = pickle.load(f)
                
                # Check for deleted files: if a patient is in cache but their file is gone, invalidate
                current_pids = set(f.replace('.pkl', '').replace('.edf', '') for f in patient_files)
                cache_invalid = False
                for ind in individuals:
                    if ind[0] not in current_pids:
                        cache_invalid = True
                        break
                
                if not cache_invalid:
                    return individuals
                else:
                    if 'st' in globals():
                        print(f"Cache for {group_name}/{sleep_stage} is stale (files were deleted). Recomputing...")
            except Exception as e:
                if 'st' in globals():
                    st.warning(f"Failed to load cache: {e}. Recomputing...")

    if test_run:
        patient_files = patient_files[:10]

    individuals = []

    progress_bar = st.progress(0, text=f"Loading {group_name} / {sleep_stage} patients...")
    status_text = st.empty()
    n_files = len(patient_files)

    excluded_pids = set()
    excluded_csv = os.path.join(base_path, "excluded_patients.csv")
    if os.path.exists(excluded_csv):
        try:
            exc_df = pd.read_csv(excluded_csv)
            excluded_pids = set(exc_df['patient_id'].dropna().str.strip().tolist())
        except Exception:
            pass

    args_list = []
    for f in patient_files:
        pid = f.replace('.pkl', '').replace('.edf', '')
        base_pid = pid.split('_')[0]
        if pid not in excluded_pids and base_pid not in excluded_pids:
            args_list.append((os.path.join(group_dir, f), pid, apply_ica))

    completed = 0
    max_workers = min(4, os.cpu_count() or 4)  # cap workers to avoid OOM on large pickles
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_patient_worker, a): a[1] for a in args_list}
        for future in as_completed(futures, timeout=30000):
            patient_id = futures[future]
            completed += 1
            status_text.text(f"Processing {completed}/{n_files}: {patient_id}")
            progress_bar.progress(completed / n_files, text=f"Loading patients ({completed}/{n_files})")
            try:
                result = future.result(timeout=12000)
            except Exception as e:
                print(f"[Worker] Timeout or error for {patient_id}: {e}")
                result = None
            if result is not None:
                individuals.append(result)

    progress_bar.empty()
    status_text.empty()
    
    # Save to local cache
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(individuals, f)
    except Exception as e:
        if 'st' in globals():
            st.warning(f"Failed to save cache for {group_name} / {sleep_stage}: {e}")

    return individuals

def _get_pynapple_jitter_shift(max_jitter_sec, sfreq):
    """
    Returns a random circular shift (in samples) using pynapple's jitter_timestamps
    if available, otherwise falls back to numpy uniform sampling.

    pynapple.randomize.jitter_timestamps expects a nap.Ts object (discrete event times)
    and shifts each timestamp by an independent uniform draw in [-max_jitter, +max_jitter].
    We create a dummy single-event Ts at t=0 and read off the jittered position as our shift.
    """
    if PYNAPPLE_AVAILABLE:
        dummy_ts = nap.Ts(t=np.array([0.0]))
        jittered = nap.randomize.jitter_timestamps(dummy_ts, max_jitter=max_jitter_sec)
        shift_sec = float(jittered.t[0])   # in [-max_jitter_sec, +max_jitter_sec]
    else:
        shift_sec = np.random.uniform(-max_jitter_sec, max_jitter_sec)
    return int(shift_sec * sfreq)


def _patient_cluster_p_value(patient_trace, null_traces, p_threshold, n_permutations):
    """
    Single-patient cluster-mass permutation test.

    For a single subject we cannot compute a t-statistic across subjects, so we
    use the raw signal amplitude as the "statistic" and define the cluster threshold
    as the |amplitude| > mean + 2*std of the null distribution of peak amplitudes.

    Parameters
    ----------
    patient_trace : np.ndarray  shape (n_times,)
        The patient's observed HEP trace.
    null_traces : np.ndarray  shape (n_permutations, n_times)
        Jitter-null distribution for this patient.
    p_threshold : float
    n_permutations : int

    Returns
    -------
    p_value : float
    """
    # Threshold based on null distribution of absolute amplitudes
    null_abs = np.abs(null_traces)          # (n_perm, n_times)
    amp_thresh = np.percentile(null_abs, (1 - p_threshold) * 100)

    # Observed cluster mass
    obs_mask = np.abs(patient_trace) > amp_thresh
    obs_labels, n_obs_clusters = label(obs_mask)
    if n_obs_clusters == 0:
        return 1.0

    obs_mass = max(np.sum(np.abs(patient_trace[obs_labels == i + 1])) for i in range(n_obs_clusters))

    # Null cluster masses
    null_masses = np.zeros(n_permutations)
    for p in range(n_permutations):
        null_trace = null_traces[p]
        null_mask = np.abs(null_trace) > amp_thresh
        null_labels, n_null_c = label(null_mask)
        if n_null_c > 0:
            null_masses[p] = max(np.sum(np.abs(null_trace[null_labels == j + 1])) for j in range(n_null_c))

    p_val = (np.sum(null_masses >= obs_mass) + 1) / (n_permutations + 1)
    return float(p_val)


def permutation_cluster_jitter_test(avg_hep, times, n_permutations=100, p_threshold=0.05, jitter_sec=None):
    """
    Cluster-based permutation test with pynapple jitter, per-patient significance,
    and Fisher-combined group p-value.

    Parameters
    ----------
    avg_hep : np.ndarray  shape (n_subjects, n_times)
        HEP data matrix — one row per subject.
    times : np.ndarray  shape (n_times,)
        Time axis (seconds).
    n_permutations : int
        Permutations for the null distribution (default 100).
    p_threshold : float
        Cluster-level alpha (default 0.05).
    jitter_sec : float or None
        Maximum circular-shift jitter in seconds.  Uses pynapple's
        ``jitter_timestamps`` to draw the shift amount for each subject.
        If None, shifts span the full epoch length.

    Returns
    -------
    significant_windows : list of dict
        Each dict has 'start', 'end', 'p_value' for the GROUP-level test.
        The 'p_value' is the Fisher-combined p-value across patients when
        all per-patient p-values are available; otherwise it is the standard
        permutation p-value.
    t_obs : np.ndarray  shape (n_times,)
        Observed t-statistic trace (group t-test vs 0).
    per_patient_info : dict
        'p_values'       : list[float]  – per-patient p-value
        'significant'    : list[bool]   – True if patient is sig. at p_threshold
        'n_significant'  : int          – count of significant patients
        'fisher_p'       : float        – Fisher-combined p-value across patients
    """
    n_subjects, n_times = avg_hep.shape

    # ── Sampling rate & max jitter in samples ────────────────────────────────
    sfreq = 1.0 / np.mean(np.diff(times))
    if jitter_sec is not None:
        max_jitter_sec = float(jitter_sec)
        max_shift      = int(max_jitter_sec * sfreq)
    else:
        max_jitter_sec = float(n_times) / sfreq   # full epoch
        max_shift      = n_times

    # ── Per-patient null traces (shared design) ───────────────────────────────
    # Build one null-trace matrix per patient: (n_permutations, n_times)
    # Jitter amounts are drawn via pynapple jitter_timestamps.
    patient_null = [np.zeros((n_permutations, n_times)) for _ in range(n_subjects)]
    for perm in range(n_permutations):
        for s in range(n_subjects):
            shift = _get_pynapple_jitter_shift(max_jitter_sec, sfreq)
            shift = np.clip(shift, -max_shift, max_shift)
            patient_null[s][perm] = np.roll(avg_hep[s], shift)

    # ── Per-patient cluster permutation p-values ─────────────────────────────
    patient_pvals = []
    for s in range(n_subjects):
        p = _patient_cluster_p_value(
            avg_hep[s],
            patient_null[s],
            p_threshold,
            n_permutations,
        )
        patient_pvals.append(p)

    patient_significant = [p < p_threshold for p in patient_pvals]
    n_significant       = int(sum(patient_significant))

    # Fisher's combined probability test across patients
    # X² = -2 * Σ ln(p_i),  df = 2*k
    eps = 1e-15
    log_sum = sum(np.log(max(p, eps)) for p in patient_pvals)
    chi2_stat  = -2.0 * log_sum
    fisher_p   = float(1.0 - stats.chi2.cdf(chi2_stat, df=2 * n_subjects))

    per_patient_info = {
        'p_values'    : patient_pvals,
        'significant' : patient_significant,
        'n_significant': n_significant,
        'fisher_p'    : fisher_p,
    }

    # ── Group-level cluster permutation test ─────────────────────────────────
    # t-thresh for group test (n_subjects - 1 df)
    t_thresh = stats.t.ppf(1 - p_threshold / 2, df=max(n_subjects - 1, 1))

    # Observed t-stats
    t_obs, _ = stats.ttest_1samp(avg_hep, 0)

    obs_mask = np.abs(t_obs) > t_thresh
    obs_labels, n_clusters = label(obs_mask)

    if n_clusters == 0:
        return [], t_obs, per_patient_info

    obs_cluster_masses = [
        np.sum(np.abs(t_obs[obs_labels == i + 1])) for i in range(n_clusters)
    ]

    # Build group null distribution from the already-computed patient_null matrices
    null_dist = np.zeros(n_permutations)
    for perm in range(n_permutations):
        # Stack jittered traces for all subjects at this permutation index
        hep_jittered = np.vstack([patient_null[s][perm] for s in range(n_subjects)])
        t_null, _    = stats.ttest_1samp(hep_jittered, 0)
        null_mask    = np.abs(t_null) > t_thresh
        null_labels, n_null_c = label(null_mask)
        if n_null_c > 0:
            null_dist[perm] = max(
                np.sum(np.abs(t_null[null_labels == j + 1])) for j in range(n_null_c)
            )

    # Compile significant windows – substitute Fisher p-value where informative
    significant_windows = []
    for i in range(n_clusters):
        perm_p = (np.sum(null_dist >= obs_cluster_masses[i]) + 1) / (n_permutations + 1)
        # Use the more conservative of the two
        combined_p = max(perm_p, fisher_p) if fisher_p < p_threshold else perm_p
        if perm_p < p_threshold:
            indices = np.where(obs_labels == i + 1)[0]
            significant_windows.append({
                'start'      : times[indices[0]],
                'end'        : times[indices[-1]],
                'p_value'    : perm_p,
                'fisher_p'   : fisher_p,
                'n_sig_patients': n_significant,
                'n_patients' : n_subjects,
            })

    return significant_windows, t_obs, per_patient_info


_perm_test_call_count = 0


def permutation_two_group_cluster_test(hep_a, hep_b, times, n_permutations=1000, p_threshold=0.05, jitter_sec=None,
                                        label_a="Group A", label_b="Group B", channel_label=None, button_key=None):
    """
    Cluster-based permutation test using Pynapple's independent subject jitter.

    hep_a, hep_b: np.ndarray (n_subjects, n_times)
    times: np.ndarray (n_times)
    label_a, label_b: display names shown in the explanation UI
    button_key: unique Streamlit widget key for the explain button (auto-generated if None)
    """
    global _perm_test_call_count
    _perm_test_call_count += 1
    if button_key is None:
        button_key = f"explain_perm_test_{_perm_test_call_count}"

    n_a, n_times = hep_a.shape
    n_b = hep_b.shape[0]
    n_total = n_a + n_b
    
    # We vstack to create the "Pool" for label swapping
    pooled = np.vstack([hep_a, hep_b]) 
    
    # Calculate Observed T-stat (Welch's for unequal n)
    t_obs, _ = stats.ttest_ind(hep_a, hep_b, axis=0, equal_var=False)
    
    # Observed Cohen's d (Pooled SD)
    d_num = np.mean(hep_a, 0) - np.mean(hep_b, 0)
    d_den = np.sqrt((np.var(hep_a, 0) + np.var(hep_b, 0)) / 2)
    cohens_d = d_num / np.where(d_den == 0, 1e-12, d_den)

    # Threshold for clustering
    df = n_a + n_b - 2
    t_thresh = stats.t.ppf(1 - p_threshold / 2, df=df)

    # Find observed clusters
    obs_mask = np.abs(t_obs) > t_thresh
    obs_labels, n_clusters = label(obs_mask)
    
    if n_clusters == 0:
        explain_permutation_test(
            hep_a, hep_b, times, [], t_obs, cohens_d,
            label_a=label_a, label_b=label_b,
            p_threshold=p_threshold, jitter_sec=jitter_sec,
            null_dist=np.zeros(n_permutations), channel_label=channel_label, button_key=button_key,
        )
        return [], t_obs, cohens_d

    obs_cluster_masses = [np.sum(np.abs(t_obs[obs_labels == i + 1])) for i in range(n_clusters)]

    # --- Permutation Loop ---
    null_dist = np.zeros(n_permutations)
    
    for p in range(n_permutations):
        # 1. Shuffle group labels
        perm_idx = np.random.permutation(n_total)
        perm_data = pooled[perm_idx]
        
        group_a = perm_data[:n_a]
        group_b = perm_data[n_a:]
        
        # 2. Apply Pynapple Jitter to each subject independently
        if jitter_sec is not None:
            group_a = _nap_jitter(group_a, times, jitter_sec)
            group_b = _nap_jitter(group_b, times, jitter_sec)

        # 3. Calculate Null T-stat
        t_null, _ = stats.ttest_ind(group_a, group_b, axis=0, equal_var=False)
        
        # 4. Max Cluster Mass for Null Distribution
        null_mask = np.abs(t_null) > t_thresh
        null_labels, n_null = label(null_mask)
        if n_null > 0:
            null_dist[p] = max(np.sum(np.abs(t_null[null_labels == j + 1])) for j in range(n_null))

    # --- Final Significance Compilation ---
    significant_windows = []
    for i in range(n_clusters):
        p_val = (np.sum(null_dist >= obs_cluster_masses[i]) + 1) / (n_permutations + 1)
        if p_val < p_threshold:
            indices = np.where(obs_labels == i + 1)[0]
            significant_windows.append({
                'start': times[indices[0]],
                'end': times[indices[-1]],
                'p_value': p_val,
                'direction': 'A>B' if np.mean(t_obs[indices]) > 0 else 'B>A'
            })

    explain_permutation_test(
        hep_a, hep_b, times, significant_windows, t_obs, cohens_d,
        label_a=label_a, label_b=label_b,
        p_threshold=p_threshold, jitter_sec=jitter_sec,
        null_dist=null_dist, channel_label=channel_label, button_key=button_key,
    )
    return significant_windows, t_obs, cohens_d


def explain_permutation_test(
    hep_a, hep_b, times, significant_windows, t_obs, cohens_d,
    label_a="Group A", label_b="Group B",
    p_threshold=0.05, n_demo_perms=300, jitter_sec=None,
    null_dist=None, channel_label=None, button_key="explain_perm_test"
):
    """
    Streamlit UI: a button that opens a step-by-step visual explanation of
    permutation_two_group_cluster_test. Pass the inputs and outputs of that
    function directly.

    Parameters
    ----------
    hep_a, hep_b  : np.ndarray (n_subjects, n_times)
    times         : np.ndarray (n_times)
    significant_windows, t_obs, cohens_d : returned by permutation_two_group_cluster_test
    label_a, label_b : display names for the two groups
    p_threshold   : same threshold used in the test
    null_dist     : pre-computed null distribution from the test (skips re-running permutations)
    n_demo_perms  : permutations to run only when null_dist is not provided
    jitter_sec    : same value used in the test
    button_key    : unique Streamlit widget key (change if calling multiple times)
    """
    ch_suffix = f" — {channel_label}" if channel_label else ""
    if not st.button(f"Explain Permutation Cluster Test{ch_suffix}", key=button_key):
        return

    n_a, n_times = hep_a.shape
    n_b = hep_b.shape[0]
    n_total = n_a + n_b
    df = n_a + n_b - 2
    t_thresh = stats.t.ppf(1 - p_threshold / 2, df=df)
    cmap_clusters = plt.cm.get_cmap("tab10")

    # Pre-compute quantities mirroring permutation_two_group_cluster_test exactly
    mu_a = np.mean(hep_a, axis=0) * 1e6
    mu_b = np.mean(hep_b, axis=0) * 1e6
    sem_a = stats.sem(hep_a, axis=0) * 1e6
    sem_b = stats.sem(hep_b, axis=0) * 1e6
    diff_mu = mu_a - mu_b

    obs_mask = np.abs(t_obs) > t_thresh
    obs_labels_arr, n_clusters = label(obs_mask)
    obs_cluster_masses = [
        np.sum(np.abs(t_obs[obs_labels_arr == i + 1])) for i in range(n_clusters)
    ]

    if null_dist is not None:
        demo_null = null_dist
        perm_source_note = f"actual test ({len(null_dist)} permutations)"
    else:
        demo_null = None
        perm_source_note = f"re-run ({n_demo_perms} permutations)"

    with st.expander(f"Step-by-Step Permutation Cluster Test{ch_suffix}", expanded=True):

        # ── STEP 1: Each group separately, then overlaid ─────────────────────
        st.subheader(f"Step 1 — Individual Subject Waveforms{ch_suffix}")
        st.markdown(
            f"""
Each thin line is one subject's average HEP at this electrode. The thick line is the group mean.
Seeing individual traces reveals within-group variability — a key input to the t-statistic.
            """
        )
        fig1, axes1 = plt.subplots(1, 3, figsize=(15, 3.5), sharey=True)
        for ax, data, lbl, color in [
            (axes1[0], hep_a, label_a, "#1f77b4"),
            (axes1[1], hep_b, label_b, "#d62728"),
        ]:
            for subj in data:
                ax.plot(times, subj * 1e6, color=color, linewidth=0.5, alpha=0.35)
            ax.plot(times, np.mean(data, 0) * 1e6, color=color, linewidth=2.2,
                    label=f"Mean (n={data.shape[0]})")
            ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
            ax.axvline(0, color="gray", linewidth=0.6, linestyle=":")
            ax.set_title(lbl)
            ax.set_xlabel("Time (s)")
            ax.legend(fontsize=8)
        axes1[0].set_ylabel("Amplitude (µV)")

        # Overlaid panel
        for data, lbl, color in [(hep_a, label_a, "#1f77b4"), (hep_b, label_b, "#d62728")]:
            mu = np.mean(data, 0) * 1e6
            sem = stats.sem(data, 0) * 1e6
            axes1[2].plot(times, mu, color=color, linewidth=2, label=f"{lbl} (n={data.shape[0]})")
            axes1[2].fill_between(times, mu - sem, mu + sem, color=color, alpha=0.2)
        axes1[2].axhline(0, color="black", linewidth=0.6, linestyle="--")
        axes1[2].axvline(0, color="gray", linewidth=0.6, linestyle=":")
        axes1[2].set_title("Both groups — mean ± SEM")
        axes1[2].set_xlabel("Time (s)")
        axes1[2].legend(fontsize=8)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        # ── STEP 2: Building the t-statistic ────────────────────────────────
        st.subheader("Step 2 — Computing the Point-wise Welch's t-Statistic")
        st.markdown(
            f"""
At every time point the test computes:

**t = (mean_A − mean_B) / pooled SE**

where pooled SE = √(var_A/n_A + var_B/n_B)  (Welch's formula — no equal-variance assumption).

The three panels below show:
1. The mean of each group separately → you see *where* one group is higher/lower.
2. The raw difference (mean_A − mean_B) → the numerator of t.
3. The full t-statistic with the critical threshold |t| > **{t_thresh:.3f}**
   (two-tailed α = {p_threshold}, df = {df}).
   Any time point above the dashed red lines is a candidate for a cluster.
            """
        )
        fig2, axes2 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

        axes2[0].plot(times, mu_a, color="#1f77b4", linewidth=1.8, label=label_a)
        axes2[0].fill_between(times, mu_a - sem_a, mu_a + sem_a, color="#1f77b4", alpha=0.18)
        axes2[0].plot(times, mu_b, color="#d62728", linewidth=1.8, label=label_b)
        axes2[0].fill_between(times, mu_b - sem_b, mu_b + sem_b, color="#d62728", alpha=0.18)
        axes2[0].axhline(0, color="black", linewidth=0.6, linestyle="--")
        axes2[0].axvline(0, color="gray", linewidth=0.6, linestyle=":")
        axes2[0].set_ylabel("Amplitude (µV)")
        axes2[0].set_title("Group means ± SEM")
        axes2[0].legend(fontsize=8)

        axes2[1].plot(times, diff_mu, color="darkorchid", linewidth=1.8,
                      label=f"{label_a} − {label_b}")
        axes2[1].fill_between(times, diff_mu, 0,
                              where=diff_mu > 0, color="#1f77b4", alpha=0.2, label=f"{label_a} > {label_b}")
        axes2[1].fill_between(times, diff_mu, 0,
                              where=diff_mu < 0, color="#d62728", alpha=0.2, label=f"{label_b} > {label_a}")
        axes2[1].axhline(0, color="black", linewidth=0.6, linestyle="--")
        axes2[1].axvline(0, color="gray", linewidth=0.6, linestyle=":")
        axes2[1].set_ylabel("Δ Amplitude (µV)")
        axes2[1].set_title(f"Mean difference: {label_a} − {label_b}")
        axes2[1].legend(fontsize=8)

        axes2[2].plot(times, t_obs, color="steelblue", linewidth=1.8, label="t-statistic")
        axes2[2].axhline( t_thresh, color="red", linewidth=1.2, linestyle="--",
                          label=f"±threshold ({t_thresh:.2f})")
        axes2[2].axhline(-t_thresh, color="red", linewidth=1.2, linestyle="--")
        axes2[2].fill_between(times, t_obs, 0,
                              where=np.abs(t_obs) > t_thresh,
                              color="orange", alpha=0.4, label="Supra-threshold")
        axes2[2].axhline(0, color="black", linewidth=0.6, linestyle="--")
        axes2[2].axvline(0, color="gray", linewidth=0.6, linestyle=":")
        axes2[2].set_ylabel("t-value")
        axes2[2].set_xlabel("Time (s)")
        axes2[2].set_title("Welch's t-statistic")
        axes2[2].legend(fontsize=8)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # ── STEP 3: Cluster identification & masses ──────────────────────────
        st.subheader("Step 3 — Cluster Identification & Cluster Mass")
        cluster_lines = "\n".join(
            f"  • Cluster {i+1}: {obs_cluster_masses[i]:.3f}  "
            f"({times[np.where(obs_labels_arr==i+1)[0][0]]*1000:.0f} – "
            f"{times[np.where(obs_labels_arr==i+1)[0][-1]]*1000:.0f} ms)"
            for i in range(n_clusters)
        ) if n_clusters > 0 else "  • No supra-threshold clusters found."
        st.markdown(
            f"""
Contiguous runs of supra-threshold time points form a **cluster**.
Each cluster is summarised by its **cluster mass** = Σ |t| across all time points inside it.

This single number encodes both *how strong* and *how long* the effect is — making the test
sensitive to sustained moderate effects that point-wise corrections would miss.

**{n_clusters} cluster(s) found:**
{cluster_lines}
            """
        )
        fig3, axes3 = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

        # Top: t-stat with clusters coloured
        axes3[0].plot(times, t_obs, color="steelblue", linewidth=1.8, label="t-statistic")
        axes3[0].axhline( t_thresh, color="red", linewidth=1.2, linestyle="--",
                          label=f"±threshold ({t_thresh:.2f})")
        axes3[0].axhline(-t_thresh, color="red", linewidth=1.2, linestyle="--")
        for i in range(n_clusters):
            idx = np.where(obs_labels_arr == i + 1)[0]
            axes3[0].fill_between(
                times, t_obs, 0, where=obs_labels_arr == i + 1,
                color=cmap_clusters(i), alpha=0.55,
                label=f"Cluster {i+1}  mass={obs_cluster_masses[i]:.3f}"
            )
            mid = times[idx[len(idx)//2]]
            ypos = t_obs[idx].mean()
            axes3[0].annotate(
                f"mass\n{obs_cluster_masses[i]:.2f}",
                xy=(mid, ypos), fontsize=7, ha="center", va="bottom",
                color=cmap_clusters(i),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec=cmap_clusters(i))
            )
        axes3[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
        axes3[0].axvline(0, color="gray", linewidth=0.6, linestyle=":")
        axes3[0].set_ylabel("t-value")
        axes3[0].set_title("Observed t-statistic with cluster labels")
        axes3[0].legend(fontsize=8)

        # Bottom: Cohen's d
        axes3[1].plot(times, cohens_d, color="purple", linewidth=1.6, label="Cohen's d")
        for i in range(n_clusters):
            axes3[1].fill_between(times, cohens_d, 0,
                                  where=obs_labels_arr == i + 1,
                                  color=cmap_clusters(i), alpha=0.35,
                                  label=f"Cluster {i+1}")
        axes3[1].axhline(0, color="black", linewidth=0.6, linestyle="--")
        axes3[1].axvline(0, color="gray", linewidth=0.6, linestyle=":")
        axes3[1].set_ylabel("Cohen's d")
        axes3[1].set_xlabel("Time (s)")
        axes3[1].set_title("Effect size (Cohen's d) — 0.2 small | 0.5 medium | 0.8 large")
        axes3[1].legend(fontsize=8)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        # ── STEP 4: Null distribution ────────────────────────────────────────
        st.subheader("Step 4 — Null Distribution by Label Permutation")
        jitter_note = (
            f"  \n**Jitter:** each subject is also circularly shifted ±{jitter_sec} s "
            f"to break residual temporal structure."
            if jitter_sec else ""
        )
        st.markdown(
            f"""
For each of {len(demo_null) if demo_null is not None else n_demo_perms} permutations the test:
1. Pools all {n_total} subjects (n_A={n_a}, n_B={n_b}).
2. Randomly re-assigns {n_a} subjects to group A and {n_b} to group B.{jitter_note}
3. Computes the t-statistic at every time point for this relabelled dataset.
4. Records the **maximum cluster mass** found in that permuted dataset.

Repeating this builds the **null distribution** of max-cluster-mass — what extreme values look
like when both groups are actually the same.  The observed cluster masses are then ranked against
this distribution to get a permutation p-value.

*Source: {perm_source_note}*
            """
        )

        if demo_null is None:
            with st.spinner(f"Running {n_demo_perms} permutations…"):
                pooled = np.vstack([hep_a, hep_b])
                demo_null = np.zeros(n_demo_perms)
                for p in range(n_demo_perms):
                    perm_idx = np.random.permutation(n_total)
                    perm_data = pooled[perm_idx]
                    ga = perm_data[:n_a]
                    gb = perm_data[n_a:]
                    if jitter_sec is not None:
                        ga = _nap_jitter(ga, times, jitter_sec)
                        gb = _nap_jitter(gb, times, jitter_sec)
                    t_null, _ = stats.ttest_ind(ga, gb, axis=0, equal_var=False)
                    null_mask = np.abs(t_null) > t_thresh
                    null_labels_arr, n_null = label(null_mask)
                    if n_null > 0:
                        demo_null[p] = max(
                            np.sum(np.abs(t_null[null_labels_arr == j + 1]))
                            for j in range(n_null)
                        )

        n_perms = len(demo_null)
        valid_null = demo_null[demo_null > 0]

        fig4, ax4 = plt.subplots(figsize=(11, 3.5))
        ax4.hist(valid_null, bins=50, color="steelblue", alpha=0.65, label="Null max-cluster-mass")
        for i, mass in enumerate(obs_cluster_masses):
            p_val = (np.sum(demo_null >= mass) + 1) / (n_perms + 1)
            color_i = cmap_clusters(i)
            ax4.axvline(mass, linewidth=2.5, linestyle="--", color=color_i)
            # Annotation above the line
            ax4.text(mass, ax4.get_ylim()[1] * 0.97,
                     f" Cluster {i+1}\n mass={mass:.2f}\n p={p_val:.3f}",
                     color=color_i, fontsize=8, va="top", ha="left",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec=color_i))
        # Mark 95th percentile
        pct95 = np.percentile(demo_null[demo_null > 0], 95) if valid_null.size > 0 else np.nan
        if np.isfinite(pct95):
            ax4.axvline(pct95, linewidth=1.5, linestyle=":", color="black",
                        label=f"95th percentile ({pct95:.2f})")
        ax4.set_xlabel("Max cluster mass (one value per permutation)")
        ax4.set_ylabel("Count")
        ax4.set_title(f"Null distribution — {perm_source_note}")
        ax4.legend(fontsize=8)
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        # ── STEP 5: Final significance ───────────────────────────────────────
        st.subheader("Step 5 — Significance Decision")
        st.markdown(
            f"""
A cluster is **significant** if its mass exceeds the 95th percentile of the null distribution
(permutation p < {p_threshold}).  This controls the family-wise error rate (FWER) across all
time points without the over-conservatism of Bonferroni.
            """
        )
        if not significant_windows:
            st.info("No significant clusters at this threshold.")
        else:
            for i, win in enumerate(significant_windows):
                direction_label = (
                    f"{label_a} > {label_b}" if win.get("direction", "A>B") == "A>B"
                    else f"{label_b} > {label_a}"
                )
                st.success(
                    f"**Cluster {i+1}** | "
                    f"{win['start']*1000:.0f} – {win['end']*1000:.0f} ms | "
                    f"p = {win['p_value']:.4f} | {direction_label}"
                )

        fig5, axes5 = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

        for data, lbl, color in [(hep_a, label_a, "#1f77b4"), (hep_b, label_b, "#d62728")]:
            mu = np.mean(data, 0) * 1e6
            sem = stats.sem(data, 0) * 1e6
            axes5[0].plot(times, mu, color=color, linewidth=2, label=f"{lbl} (n={data.shape[0]})")
            axes5[0].fill_between(times, mu - sem, mu + sem, color=color, alpha=0.2)
        for win in significant_windows:
            axes5[0].axvspan(win['start'], win['end'], color="orange", alpha=0.28,
                             label=f"p={win['p_value']:.3f}")
        axes5[0].axhline(0, color="black", linewidth=0.6, linestyle="--")
        axes5[0].axvline(0, color="gray", linewidth=0.6, linestyle=":")
        axes5[0].set_ylabel("Amplitude (µV)")
        axes5[0].set_title(f"Significant windows on group averages{ch_suffix}")
        handles, labels_leg = axes5[0].get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels_leg):
            if l not in seen:
                seen[l] = h
        axes5[0].legend(seen.values(), seen.keys(), fontsize=8)

        axes5[1].plot(times, cohens_d, color="purple", linewidth=1.6)
        for win in significant_windows:
            mask_win = (times >= win['start']) & (times <= win['end'])
            axes5[1].fill_between(times, cohens_d, 0, where=mask_win,
                                  color="orange", alpha=0.4)
            mid = (win['start'] + win['end']) / 2
            peak_d = cohens_d[(times >= win['start']) & (times <= win['end'])]
            axes5[1].annotate(
                f"d={peak_d[np.argmax(np.abs(peak_d))]:.2f}",
                xy=(mid, peak_d[np.argmax(np.abs(peak_d))]),
                fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="orange")
            )
        axes5[1].axhline(0, color="black", linewidth=0.6, linestyle="--")
        axes5[1].axvline(0, color="gray", linewidth=0.6, linestyle=":")
        axes5[1].set_ylabel("Cohen's d")
        axes5[1].set_xlabel("Time (s)")
        axes5[1].set_title("Effect size within significant windows")
        fig5.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)


def _nap_jitter(data_matrix, times, jitter_sec):
    """Helper to apply independent circular shifts to rows."""
    dt = times[1] - times[0]
    out = np.empty_like(data_matrix)
    for i in range(data_matrix.shape[0]):
        shift_val = np.random.uniform(-jitter_sec, jitter_sec)
        shift_samples = int(round(shift_val / dt))
        out[i] = np.roll(data_matrix[i], shift_samples)
    return out

def save_hep_to_downloads(hep_a: np.ndarray, hep_b: np.ndarray, times: np.ndarray,
                          label_a: str = "group_a", label_b: str = "group_b",
                          downloads_dir: str = "/storage/pblab_shared_data/Nir/Cobrad/.downloads") -> str:
    """Save hep_a and hep_b matrices (and times) as .npz to .downloads/.

    Returns the path of the saved file.
    """
    os.makedirs(downloads_dir, exist_ok=True)
    filename = f"hep_{label_a}_vs_{label_b}.npz"
    filepath = os.path.join(downloads_dir, filename)
    np.savez(filepath, hep_a=hep_a, hep_b=hep_b, times=times,
             label_a=np.array(label_a), label_b=np.array(label_b))
    return filepath


def finalize_plot(fig, ax, title, avg_hep=None, times=None, n_subjects=None, significant_windows=None, all_heps=None):
    """
    Applies common styling to the plot, optionally plots the average and significance, and displays it.
    Appends N, min p-value, and Cohen's d to the title if data is available.
    """
    if avg_hep is not None and times is not None:
        label = f"Group Average (n={n_subjects})" if n_subjects is not None else "Group Average"
        ax.plot(times, avg_hep * 1e6, color='blue', linewidth=2, label=label)

    if significant_windows:
        for window in significant_windows:
            start, end = window['start'], window['end']
            p_val = window['p_value']
            
            # Plot rectangle
            ax.axvspan(start, end, color='orange', alpha=0.2)
            
            # Plot asterisk
            mid_point = (start + end) / 2
            
            # Determine asterisks
            if p_val < 0.001:
                asterisks = "***"
            elif p_val < 0.01:
                asterisks = "**"
            else:
                asterisks = "*"
            current_ymax = ax.get_ylim()[1]
            ax.text(mid_point, current_ymax-.4, asterisks, ha='center', va='bottom', fontsize=16, color='orange', fontweight='bold')

    # Build stats suffix for title
    stats_parts = []
    if n_subjects is not None:
        stats_parts.append(f"N={n_subjects}")
    if significant_windows:
        min_p = min(w['p_value'] for w in significant_windows)
        min_p_win = min(significant_windows, key=lambda w: w['p_value'])
        win_str = f"[{min_p_win['start']:.3f}s, {min_p_win['end']:.3f}s]"
        stats_parts.append(f"p={min_p:.2e} {win_str}")
        # Per-patient count and Fisher p (present when using pynapple-jitter test)
        n_sig_pt = min_p_win.get('n_sig_patients')
        n_tot_pt = min_p_win.get('n_patients')
        fish_p   = min_p_win.get('fisher_p')
        if n_sig_pt is not None and n_tot_pt is not None:
            stats_parts.append(f"{n_sig_pt}/{n_tot_pt} pts sig")
        if fish_p is not None:
            stats_parts.append(f"Fisher p={fish_p:.3f}")
    else:
        stats_parts.append("p=n.s.")
    if all_heps is not None and len(all_heps) > 1:
        try:
            heps_arr = np.array(all_heps)  # (n_subjects, n_times)
            # Cohen's d at the peak absolute amplitude
            mean_arr = np.mean(heps_arr, axis=0)
            std_arr = np.std(heps_arr, axis=0, ddof=1)
            peak_idx = np.argmax(np.abs(mean_arr))
            d = abs(mean_arr[peak_idx]) / (std_arr[peak_idx] + 1e-12)
            stats_parts.append(f"d={d:.2f}")
        except Exception:
            pass

    full_title = title
    if stats_parts:
        full_title = f"{title}\n({', '.join(stats_parts)})"

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (μV)")
    ax.set_title(full_title)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 12:
        ax.legend(handles, labels, fontsize=7, ncol=2, loc='upper right',
                  bbox_to_anchor=(1.0, 1.0), framealpha=0.6)
    else:
        ax.legend(handles, labels)
    ax.grid(True)
    ax.axvline(0, color='r', linestyle='--', alpha=0.5)
    if times is not None:
        ax.set_xlim(times[0], times[-1])
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

def run_compare_groups_analysis(base_path, selected_stage):
    """
    Logic for Comparing Groups mode.
    Plots a publication-ready, statistically annotated comparison of the HEP
    (Heartbeat-Evoked Potential) between all available groups, showing:
      1. Grand average waveforms + SEM ribbons per group
      2. Difference waveform + cluster-permutation significance + Cohen's d (if 2 groups)
      3. Channel x time heatmap of mean group difference
      4. Summary statistics table
    """
    available_groups = sorted([g for g in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, g))])
    if not available_groups:
        st.error("No groups found in the data directory.")
        return

    # ── Group selection ──────────────────────────────────────────────────────
    default_groups = [g for g in ['EDF', 'Berkeley_data'] if g in available_groups] or available_groups[:2]
    selected_groups = st.multiselect(
        "Select Groups to Compare",
        options=available_groups,
        default=default_groups,
        key="cmp_selected_groups",
    )
    if not selected_groups:
        st.warning("Please select at least one group.")
        return

    # ── UI controls ─────────────────────────────────────────────────────────
    with st.expander("⚙️ Analysis Settings", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.runtime.exists():
                test_run = st.checkbox("Test Run (5 files/group)", value=False, key="cmp_test_run")
            else:
                test_run = True
        with col2:
            n_permutations = st.slider("Permutations", 50, 500, 200, 50, key="cmp_n_perm")
        with col3:
            jitter_sec = st.number_input("Jitter (s)", 0.01, 0.5, 0.1, 0.05, key="cmp_jitter")
        with col4:
            use_zscore = st.checkbox("Z-score subjects", value=True, key="cmp_zscore",
                                     help="Normalise each subject's trace to zero-mean unit-variance before averaging. Turn off to keep raw µV values.")
    amp_ylabel = "Amplitude (Z-score)" if use_zscore else "Amplitude (µV)"

    # ── Load globally excluded patients ─────────────────────────────────────
    csv_path = os.path.join(base_path, "excluded_patients.csv")
    global_excluded_pids = []
    if os.path.exists(csv_path):
        try:
            global_excluded_df = pd.read_csv(csv_path)
            if 'patient_id' in global_excluded_df.columns:
                global_excluded_pids = [str(pid) for pid in global_excluded_df['patient_id'].tolist()]
        except Exception as e:
            st.warning(f"Failed to load excluded patients CSV: {e}")

    # ── Load individual HEPs per group ──────────────────────────────────────
    group_individuals = {}   # group -> list of individual tuples
    for group in selected_groups:
        with st.spinner(f"Loading {group}…"):
            inds = get_group_individuals(group, selected_stage, base_path, test_run=test_run)
            
        if inds:
            # Filter out globally excluded patients by checking if the base patient ID is in the excluded list
            inds_filtered = []
            for ind in inds:
                pid = str(ind[0])
                base_pid = pid.split('_')[0] if '_' in pid else pid
                if base_pid not in global_excluded_pids and pid not in global_excluded_pids:
                    inds_filtered.append(ind)
            
            if inds_filtered:
                group_individuals[group] = inds_filtered
            else:
                st.warning(f"All loaded data for group **{group}** in stage {selected_stage} was globally excluded.")
        else:
            st.warning(f"No valid data for group **{group}** in stage {selected_stage}.")

    if not group_individuals:
        st.error("No valid data found for any group.")
        return

    groups_with_data = list(group_individuals.keys())
    n_groups = len(groups_with_data)

    # ── Identify common EEG channels across all groups ───────────────────────
    # Same logic as Per-Channel Analysis: ≥50% of subjects, 1-2 letter prefix
    all_inds_flat = [ind for inds in group_individuals.values() for ind in inds]
    all_ch_sets = [set(ind[3]) for ind in all_inds_flat]
    ch_counts = Counter([ch for s in all_ch_sets for ch in s])
    common_channels = [
        ch for ch, count in ch_counts.items()
        if count >= len(all_inds_flat) * 0.5
        and (re.match(r'^[a-zA-Z]{1,2}[0-9]*$', ch) or re.match(r'^[a-zA-Z]z$', ch))
    ]

    # Build per-group arrays:  (n_subjects, n_times)  averaged across channels
    group_hep_matrix = {}   # group -> np.ndarray (n_subj, n_times)
    group_times = {}
    group_hep_per_channel = {}   # group -> dict{ch: (n_subj, n_times)}

    for group, inds in group_individuals.items():
        subj_mean_heps = []
        subj_ch_heps = {ch: [] for ch in common_channels}
        times_ref = None
        for ind in inds:
            pid, hep_data, times, ch_names, rpeaks, ecg_hep, ecg_ch_names = ind[:7]
            if hep_data is None or times is None:
                continue
            times_ref = times
            # Mean across common_channels only — same as Per-Channel Analysis "Average"
            valid_ch_indices = [ch_names.index(ch) for ch in common_channels if ch in ch_names]
            if valid_ch_indices:
                subj_mean_heps.append(np.nanmean(hep_data[valid_ch_indices, :], axis=0))
            # Per-channel
            for ch in common_channels:
                if ch in ch_names:
                    ch_idx = ch_names.index(ch)
                    subj_ch_heps[ch].append(hep_data[ch_idx])

        if subj_mean_heps:
            group_hep_matrix[group] = np.array(subj_mean_heps)     # (n_subj, n_times)
            group_times[group] = times_ref
            group_hep_per_channel[group] = {
                ch: np.array(v) for ch, v in subj_ch_heps.items() if v
            }

    if not group_hep_matrix:
        st.error("No valid HEP data could be computed.")
        return

    # ── Optional: Z-score each subject's trace ───────────────────────────────
    if use_zscore:
        for group in list(group_hep_matrix.keys()):
            mat = group_hep_matrix[group]          # (n_subj, n_times)
            mu = np.nanmean(mat, axis=1, keepdims=True)
            sigma = np.nanstd(mat, axis=1, ddof=1, keepdims=True)
            sigma = np.where(sigma == 0, 1e-12, sigma)
            group_hep_matrix[group] = (mat - mu) / sigma

            for ch in group_hep_per_channel.get(group, {}):
                ch_mat = group_hep_per_channel[group][ch]  # (n_subj, n_times)
                if ch_mat.ndim == 2 and ch_mat.shape[0] > 0:
                    mu_ch = np.nanmean(ch_mat, axis=1, keepdims=True)
                    sig_ch = np.nanstd(ch_mat, axis=1, ddof=1, keepdims=True)
                    sig_ch = np.where(sig_ch == 0, 1e-12, sig_ch)
                    group_hep_per_channel[group][ch] = (ch_mat - mu_ch) / sig_ch
    else:
        # Convert to µV for display (data stored in V)
        for group in list(group_hep_matrix.keys()):
            group_hep_matrix[group] = group_hep_matrix[group] * 1e6
            for ch in group_hep_per_channel.get(group, {}):
                ch_mat = group_hep_per_channel[group][ch]
                if ch_mat.ndim == 2 and ch_mat.shape[0] > 0:
                    group_hep_per_channel[group][ch] = ch_mat * 1e6

    # Use times from first available group
    times = group_times[groups_with_data[0]]

    # ── Colour palette ───────────────────────────────────────────────────────
    PALETTE = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
    group_color = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups_with_data)}

    st.markdown("---")
    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 0 — ECG Comparison across groups (average + individual traces)
    # ═══════════════════════════════════════════════════════════════════════
    st.subheader("🫀 ECG Comparison across Groups")

    # Collect per-group ECG traces
    group_ecg_matrix = {}   # group -> list of 1-D arrays (µV)
    ecg_times_ref = None
    for group, inds in group_individuals.items():
        ecg_traces = []
        for ind in inds:
            ecg_hep = ind[5]
            ind_times = ind[2]
            if ecg_hep is not None:
                trace = np.asarray(ecg_hep).squeeze()
                if trace.ndim == 1 and len(trace) > 0:
                    ecg_traces.append(trace * 1e6)   # convert V → µV
                    if ecg_times_ref is None and ind_times is not None:
                        ecg_times_ref = ind_times
        if ecg_traces:
            group_ecg_matrix[group] = ecg_traces

    if group_ecg_matrix and ecg_times_ref is not None:
        fig_ecg_cmp, ax_ecg_cmp = plt.subplots(figsize=(14, 5))
        ax_ecg_cmp.axvline(0, color='red', linestyle='--', alpha=0.6, label='R-peak (t=0)')
        ax_ecg_cmp.axhline(0, color='black', linewidth=0.5, alpha=0.3)

        for group in groups_with_data:
            traces = group_ecg_matrix.get(group)
            if not traces:
                continue
            color = group_color[group]
            # Align all traces to the shortest length
            min_len = min(len(tr) for tr in traces)
            t_ecg = ecg_times_ref[:min_len]
            traces_arr = np.array([tr[:min_len] for tr in traces])   # (n_subj, n_times)
            avg_ecg = np.nanmean(traces_arr, axis=0)
            sem_ecg = np.nanstd(traces_arr, axis=0, ddof=1) / np.sqrt(len(traces))

            # Faint individual lines
            for tr in traces_arr:
                ax_ecg_cmp.plot(t_ecg, tr, color=color, linewidth=0.6, alpha=0.18)

            # Bold group average + SEM ribbon
            ax_ecg_cmp.plot(t_ecg, avg_ecg, color=color, linewidth=2.5,
                            label=f"{group}  (n={len(traces)})")
            ax_ecg_cmp.fill_between(t_ecg, avg_ecg - sem_ecg, avg_ecg + sem_ecg,
                                    color=color, alpha=0.22)

        ax_ecg_cmp.set_xlabel("Time relative to R-peak (s)", fontsize=12)
        ax_ecg_cmp.set_ylabel("Amplitude (µV)", fontsize=12)
        ax_ecg_cmp.set_title(
            f"ECG Grand Average Comparison — Sleep Stage: {selected_stage}",
            fontsize=14, fontweight='bold'
        )
        ax_ecg_cmp.legend(fontsize=11)
        ax_ecg_cmp.grid(True, alpha=0.25)
        ax_ecg_cmp.set_xlim(ecg_times_ref[0], ecg_times_ref[-1])
        fig_ecg_cmp.tight_layout()
        st.pyplot(fig_ecg_cmp, use_container_width=True)
        plt.close(fig_ecg_cmp)
    else:
        st.info("No ECG data available for cross-group comparison.")

    st.markdown("---")
    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 1b — Grand Average + Individual Subject Lines (Spaghetti Plot)
    # ═══════════════════════════════════════════════════════════════════════
    st.subheader("📈 Grand Average HEP per Group — with Individual Subjects")

    fig1b, ax1b = plt.subplots(figsize=(14, 5))
    ax1b.axvline(0, color='red', linestyle='--', alpha=0.6, label='R-peak (t=0)')
    ax1b.axhline(0, color='black', linewidth=0.5, alpha=0.3)

    for group in groups_with_data:
        mat = group_hep_matrix[group]          # (n_subj, n_times)
        t = group_times[group]
        n_subj = mat.shape[0]
        grand_avg = np.nanmean(mat, axis=0)
        color = group_color[group]
        # Individual subject lines (faint)
        for i in range(n_subj):
            ax1b.plot(t, mat[i], color=color, linewidth=0.6, alpha=0.2)
        # Grand average on top (bold)
        ax1b.plot(t, grand_avg, color=color, linewidth=2.5, label=f"{group}  (n={n_subj})")

    ax1b.set_xlabel("Time relative to R-peak (s)", fontsize=12)
    ax1b.set_ylabel(amp_ylabel, fontsize=12)
    ax1b.set_title(
        f"HEP Grand Average + Individual Subjects — Sleep Stage: {selected_stage}",
        fontsize=14, fontweight='bold'
    )
    ax1b.legend(fontsize=11)
    ax1b.grid(True, alpha=0.25)
    if times is not None:
        ax1b.set_xlim(times[0], times[-1])
    fig1b.tight_layout()
    st.pyplot(fig1b, use_container_width=True)
    plt.close(fig1b)

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 1c — Hemisphere Comparison: Left vs Right electrodes
    # Left  = channels ending in odd digit  (e.g. F3, C3, P3, O1)
    # Right = channels ending in even digit (e.g. F4, C4, P4, O2)
    # ═══════════════════════════════════════════════════════════════════════
    if group_hep_per_channel and common_channels:
        st.subheader("🧠 Hemisphere Comparison: Left vs Right Electrodes")

        left_chs  = [ch for ch in common_channels if re.search(r'[13579]$', ch)]
        right_chs = [ch for ch in common_channels if re.search(r'[2468]$',  ch)]

        if left_chs and right_chs:
            fig1c, (ax_L, ax_R) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

            for ax, side_chs, side_label in [
                (ax_L, left_chs,  "Left hemisphere"),
                (ax_R, right_chs, "Right hemisphere"),
            ]:
                ax.axvline(0, color='red', linestyle='--', alpha=0.6)
                ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

                for group in groups_with_data:
                    ch_data = group_hep_per_channel.get(group, {})
                    t = group_times[group]
                    # Build per-subject side average from raw data so subjects
                    # with missing channels are handled gracefully (same as
                    # Per-Channel "Average" logic).
                    side_mats = []
                    for ind in group_individuals[group]:
                        _pid, hep_data, _t, ch_names = ind[0], ind[1], ind[2], ind[3]
                        if hep_data is None:
                            continue
                        valid_idx = [ch_names.index(ch) for ch in side_chs if ch in ch_names]
                        if valid_idx:
                            side_mats.append(np.nanmean(hep_data[valid_idx, :], axis=0))
                    if not side_mats:
                        continue
                    side_avg = np.array(side_mats)   # (n_subj, n_times)
                    if use_zscore:
                        mu = np.nanmean(side_avg, axis=1, keepdims=True)
                        sigma = np.nanstd(side_avg, axis=1, ddof=1, keepdims=True)
                        sigma = np.where(sigma == 0, 1e-12, sigma)
                        side_avg = (side_avg - mu) / sigma
                    else:
                        side_avg = side_avg * 1e6
                    min_t = side_avg.shape[1]
                    n_subj = side_avg.shape[0]
                    grand = np.nanmean(side_avg, axis=0)
                    sem   = np.nanstd(side_avg, axis=0, ddof=1) / np.sqrt(n_subj)
                    color = group_color[group]
                    ax.plot(t[:min_t], grand, color=color, linewidth=2.5,
                            label=f"{group}  (n={n_subj})")
                    ax.fill_between(t[:min_t], grand - sem, grand + sem,
                                    color=color, alpha=0.18)

                ax.set_title(f"{side_label}\n({', '.join(side_chs)})",
                             fontsize=12, fontweight='bold')
                ax.set_xlabel("Time relative to R-peak (s)", fontsize=11)
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=10)
                if times is not None:
                    ax.set_xlim(times[0], times[-1])

            ax_L.set_ylabel(amp_ylabel, fontsize=11)
            fig1c.suptitle(
                f"Left vs Right Hemisphere HEP — Sleep Stage: {selected_stage}",
                fontsize=14, fontweight='bold'
            )
            fig1c.tight_layout()
            st.pyplot(fig1c, use_container_width=True)
            plt.close(fig1c)
        else:
            st.info("Not enough lateralised channels to plot a hemisphere comparison "
                    f"(left: {left_chs}, right: {right_chs}).")

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 2 — Difference + permutation significance + Cohen's d
    # (only when exactly 2 groups are available)
    # ═══════════════════════════════════════════════════════════════════════
    sig_windows_global = []
    if n_groups == 2:
        g_a, g_b = groups_with_data[0], groups_with_data[1]
        hep_a = group_hep_matrix[g_a]
        hep_b = group_hep_matrix[g_b]
        t_a = group_times[g_a]
        t_b = group_times[g_b]

        # Align lengths
        min_len = min(hep_a.shape[1], hep_b.shape[1])
        hep_a = hep_a[:, :min_len]
        hep_b = hep_b[:, :min_len]
        t_common = t_a[:min_len]

        st.subheader(f"📊 Statistical Comparison: {g_a}  vs  {g_b}")
        with st.spinner(f"Running cluster permutation test ({n_permutations} permutations)…"):
            sig_windows_global, t_stat, cohens_d = permutation_two_group_cluster_test(
                hep_a, hep_b, t_common,
                n_permutations=n_permutations,
                p_threshold=0.05,
                jitter_sec=jitter_sec,
                channel_label="Average",
            )

        # ── Build figure with 2 stacked panels ──────────────────────────────
        fig2, (ax_diff, ax_d) = plt.subplots(
            2, 1, figsize=(14, 8),
            gridspec_kw={'height_ratios': [3, 1.5]},
            sharex=True
        )

        # -- Top panel: group means + difference curve ----------------------
        mean_a = np.nanmean(hep_a, 0)
        mean_b = np.nanmean(hep_b, 0)
        diff   = mean_a - mean_b

        sem_a = np.nanstd(hep_a, 0, ddof=1) / np.sqrt(hep_a.shape[0])
        sem_b = np.nanstd(hep_b, 0, ddof=1) / np.sqrt(hep_b.shape[0])

        ax_diff.plot(t_common, mean_a, color=group_color[g_a], linewidth=2, label=f"{g_a} mean")
        ax_diff.fill_between(t_common, mean_a - sem_a, mean_a + sem_a, color=group_color[g_a], alpha=0.15)
        ax_diff.plot(t_common, mean_b, color=group_color[g_b], linewidth=2, label=f"{g_b} mean")
        ax_diff.fill_between(t_common, mean_b - sem_b, mean_b + sem_b, color=group_color[g_b], alpha=0.15)
        ax_diff.plot(t_common, diff, color='black', linewidth=1.5, linestyle='--', label=f"{g_a}−{g_b} diff")
        ax_diff.axhline(0, color='black', linewidth=0.4, alpha=0.3)
        ax_diff.axvline(0, color='red', linestyle='--', alpha=0.5)

        # Significance spans + asterisks
        for win in sig_windows_global:
            ax_diff.axvspan(win['start'], win['end'], color='orange', alpha=0.25)
            ax_d.axvspan(win['start'], win['end'], color='orange', alpha=0.25)
            mid = (win['start'] + win['end']) / 2
            p_val = win['p_value']
            stars = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else '*')
            ymax = ax_diff.get_ylim()[1]
            # p-value label
            p_txt = f"p={'<0.001' if p_val < 0.001 else f'{p_val:.3f}'}"
            ax_diff.text(mid, ymax * 0.95, stars, ha='center', va='top',
                         fontsize=16, color='darkorange', fontweight='bold')
            ax_diff.text(mid, ymax * 0.82, p_txt, ha='center', va='top',
                         fontsize=9, color='darkorange')

        ax_diff.set_ylabel(amp_ylabel, fontsize=11)
        if sig_windows_global:
            min_p = min(w['p_value'] for w in sig_windows_global)
            p_tag = 'p<0.001' if min_p < 0.001 else f'p={min_p:.3f}'
        else:
            p_tag = 'p=n.s.'
        ax_diff.set_title(
            f"{g_a} vs {g_b}  |  n={hep_a.shape[0]}+{hep_b.shape[0]}  |  {p_tag}  |  permutations={n_permutations}",
            fontsize=13, fontweight='bold'
        )
        ax_diff.legend(fontsize=10)
        ax_diff.grid(True, alpha=0.2)

        # -- Bottom panel: Cohen's d ----------------------------------------
        ax_d.plot(t_common, cohens_d, color='#5c35cc', linewidth=2, label="Cohen's d (signed)")
        ax_d.axhline(0,    color='black', linewidth=0.5, alpha=0.3)
        ax_d.axhline(0.5,  color='gray',  linewidth=0.8, linestyle=':', alpha=0.6, label='d=0.5 (medium)')
        ax_d.axhline(-0.5, color='gray',  linewidth=0.8, linestyle=':', alpha=0.6)
        ax_d.axhline(0.8,  color='gray',  linewidth=0.8, linestyle='--', alpha=0.5, label='d=0.8 (large)')
        ax_d.axhline(-0.8, color='gray',  linewidth=0.8, linestyle='--', alpha=0.5)
        ax_d.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax_d.fill_between(t_common, 0, cohens_d,
                          where=(cohens_d > 0), color='#2196F3', alpha=0.2, interpolate=True)
        ax_d.fill_between(t_common, 0, cohens_d,
                          where=(cohens_d < 0), color='#FF5722', alpha=0.2, interpolate=True)
        ax_d.set_xlabel("Time relative to R-peak (s)", fontsize=11)
        ax_d.set_ylabel("Cohen's d", fontsize=11)
        ax_d.legend(fontsize=9, loc='upper right')
        ax_d.grid(True, alpha=0.2)
        if t_common is not None:
            ax_d.set_xlim(t_common[0], t_common[-1])

        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        # ── Significant windows table ────────────────────────────────────────
        if sig_windows_global:
            st.markdown("**Significant time windows (cluster permutation test):**")
            rows = []
            for win in sig_windows_global:
                p_val = win['p_value']
                rows.append({
                    "Start (s)": f"{win['start']:.3f}",
                    "End (s)": f"{win['end']:.3f}",
                    "Duration (ms)": f"{(win['end']-win['start'])*1000:.1f}",
                    "p-value": "<0.001" if p_val < 0.001 else f"{p_val:.3f}",
                    "Direction": win.get('direction', ''),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No significant time windows found (cluster permutation test, p > 0.05).")


        # ── Hemisphere statistical comparison ───────────────────────────────
        st.markdown("---")
        st.subheader(f"🧠 Hemisphere Statistical Comparison: {g_a} vs {g_b}")
        st.markdown(
            "Cluster-permutation test run separately on **left** (odd-numbered) "
            "and **right** (even-numbered) electrodes, averaged within each hemisphere."
        )

        if group_hep_per_channel and common_channels:
            left_chs_stat  = [ch for ch in common_channels if re.search(r'[13579]$', ch)]
            right_chs_stat = [ch for ch in common_channels if re.search(r'[2468]$',  ch)]

            def _build_hemisphere_matrix(group, side_chs):
                """Return (n_subj, n_times) averaged over side_chs for a group."""
                rows = []
                for ind in group_individuals.get(group, []):
                    _pid, hep_data, _t, ch_names = ind[0], ind[1], ind[2], ind[3]
                    if hep_data is None:
                        continue
                    valid_idx = [ch_names.index(ch) for ch in side_chs if ch in ch_names]
                    if valid_idx:
                        rows.append(np.nanmean(hep_data[valid_idx, :], axis=0))
                if not rows:
                    return None
                mat = np.array(rows)   # (n_subj, n_times)
                if use_zscore:
                    mu = np.nanmean(mat, axis=1, keepdims=True)
                    sigma = np.nanstd(mat, axis=1, ddof=1, keepdims=True)
                    sigma = np.where(sigma == 0, 1e-12, sigma)
                    mat = (mat - mu) / sigma
                else:
                    mat = mat * 1e6
                return mat

            for side_label, side_chs in [("Left hemisphere", left_chs_stat), ("Right hemisphere", right_chs_stat)]:
                if not side_chs:
                    st.info(f"No {side_label} channels found.")
                    continue

                hem_a = _build_hemisphere_matrix(g_a, side_chs)
                hem_b = _build_hemisphere_matrix(g_b, side_chs)

                if hem_a is None or hem_b is None:
                    st.warning(f"Not enough data for {side_label} comparison.")
                    continue

                # Align time lengths
                min_len_h = min(hem_a.shape[1], hem_b.shape[1], len(t_common))
                hem_a = hem_a[:, :min_len_h]
                hem_b = hem_b[:, :min_len_h]
                t_h = t_common[:min_len_h]

                st.markdown(f"**{side_label}** ({', '.join(side_chs)})")
                with st.spinner(f"Running permutation test for {side_label}…"):
                    sig_hem, t_hem_stat, cd_hem = permutation_two_group_cluster_test(
                        hem_a, hem_b, t_h,
                        n_permutations=n_permutations,
                        p_threshold=0.05,
                        jitter_sec=jitter_sec,
                        channel_label=side_label,
                    )

                fig_hem, (ax_hem_diff, ax_hem_d) = plt.subplots(
                    2, 1, figsize=(14, 7),
                    gridspec_kw={'height_ratios': [3, 1.5]},
                    sharex=True
                )

                mean_ha = np.nanmean(hem_a, 0)
                mean_hb = np.nanmean(hem_b, 0)
                diff_h  = mean_ha - mean_hb
                sem_ha = np.nanstd(hem_a, 0, ddof=1) / np.sqrt(hem_a.shape[0])
                sem_hb = np.nanstd(hem_b, 0, ddof=1) / np.sqrt(hem_b.shape[0])

                ax_hem_diff.axvline(0, color='red', linestyle='--', alpha=0.5)
                ax_hem_diff.axhline(0, color='black', linewidth=0.4, alpha=0.3)
                ax_hem_diff.plot(t_h, mean_ha, color=group_color[g_a], linewidth=2, label=f"{g_a} mean")
                ax_hem_diff.fill_between(t_h, mean_ha - sem_ha, mean_ha + sem_ha, color=group_color[g_a], alpha=0.15)
                ax_hem_diff.plot(t_h, mean_hb, color=group_color[g_b], linewidth=2, label=f"{g_b} mean")
                ax_hem_diff.fill_between(t_h, mean_hb - sem_hb, mean_hb + sem_hb, color=group_color[g_b], alpha=0.15)
                ax_hem_diff.plot(t_h, diff_h, color='black', linewidth=1.5, linestyle='--', label=f"{g_a}−{g_b} diff")

                for win in sig_hem:
                    ax_hem_diff.axvspan(win['start'], win['end'], color='orange', alpha=0.25)
                    ax_hem_d.axvspan(win['start'], win['end'], color='orange', alpha=0.25)
                    mid = (win['start'] + win['end']) / 2
                    p_v = win['p_value']
                    stars = '***' if p_v < 0.001 else ('**' if p_v < 0.01 else '*')
                    ymax_h = ax_hem_diff.get_ylim()[1]
                    p_txt_h = f"p={'<0.001' if p_v < 0.001 else f'{p_v:.3f}'}"
                    ax_hem_diff.text(mid, ymax_h * 0.95, stars, ha='center', va='top',
                                     fontsize=16, color='darkorange', fontweight='bold')
                    ax_hem_diff.text(mid, ymax_h * 0.82, p_txt_h, ha='center', va='top',
                                     fontsize=9, color='darkorange')

                p_tag_h = ('p<0.001' if min(w['p_value'] for w in sig_hem) < 0.001
                           else f"p={min(w['p_value'] for w in sig_hem):.3f}") if sig_hem else 'p=n.s.'
                ax_hem_diff.set_title(
                    f"{side_label} — {g_a} vs {g_b}  |  n={hem_a.shape[0]}+{hem_b.shape[0]}  |  {p_tag_h}",
                    fontsize=13, fontweight='bold'
                )
                ax_hem_diff.set_ylabel(amp_ylabel, fontsize=11)
                ax_hem_diff.legend(fontsize=10)
                ax_hem_diff.grid(True, alpha=0.2)

                ax_hem_d.plot(t_h, cd_hem, color='#5c35cc', linewidth=2, label="Cohen's d")
                ax_hem_d.axhline(0,    color='black', linewidth=0.5, alpha=0.3)
                ax_hem_d.axhline( 0.5, color='gray', linewidth=0.8, linestyle=':', alpha=0.6, label='d=0.5')
                ax_hem_d.axhline(-0.5, color='gray', linewidth=0.8, linestyle=':', alpha=0.6)
                ax_hem_d.axhline( 0.8, color='gray', linewidth=0.8, linestyle='--', alpha=0.5, label='d=0.8')
                ax_hem_d.axhline(-0.8, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
                ax_hem_d.axvline(0, color='red', linestyle='--', alpha=0.5)
                ax_hem_d.fill_between(t_h, 0, cd_hem, where=(cd_hem > 0), color='#2196F3', alpha=0.2, interpolate=True)
                ax_hem_d.fill_between(t_h, 0, cd_hem, where=(cd_hem < 0), color='#FF5722', alpha=0.2, interpolate=True)
                ax_hem_d.set_xlabel("Time relative to R-peak (s)", fontsize=11)
                ax_hem_d.set_ylabel("Cohen's d", fontsize=11)
                ax_hem_d.legend(fontsize=9, loc='upper right')
                ax_hem_d.grid(True, alpha=0.2)
                ax_hem_d.set_xlim(t_h[0], t_h[-1])

                fig_hem.tight_layout()
                st.pyplot(fig_hem, use_container_width=True)
                plt.close(fig_hem)

                if sig_hem:
                    rows_h = []
                    for win in sig_hem:
                        p_v = win['p_value']
                        rows_h.append({
                            "Start (s)": f"{win['start']:.3f}",
                            "End (s)": f"{win['end']:.3f}",
                            "Duration (ms)": f"{(win['end']-win['start'])*1000:.1f}",
                            "p-value": "<0.001" if p_v < 0.001 else f"{p_v:.3f}",
                            "Direction": win.get('direction', ''),
                        })
                    st.dataframe(pd.DataFrame(rows_h), use_container_width=True)
                else:
                    st.info(f"No significant windows in {side_label} (p > 0.05).")

                side_key = side_label.lower().replace(" ", "_")
        else:
            st.info("Per-hemisphere analysis requires per-channel data.")

        # ── Per-electrode statistical comparison ────────────────────────────
        st.markdown("---")
        st.subheader(f"📡 Per-Electrode Statistical Comparison: {g_a} vs {g_b}")
        st.markdown(
            "Each subplot shows group means ± SEM for one EEG channel. "
            "Orange spans mark cluster-permutation significant windows."
        )

        if group_hep_per_channel and common_channels:
            n_cols_el = 4
            n_rows_el = int(np.ceil(len(common_channels) / n_cols_el))
            fig_el, axes_el = plt.subplots(
                n_rows_el, n_cols_el,
                figsize=(5 * n_cols_el, 3.5 * n_rows_el),
                sharex=True
            )
            axes_el_flat = np.array(axes_el).flatten()

            per_ch_save = {}  # collect ca/cb per channel for download

            for idx_ch, ch in enumerate(common_channels):
                ax_el = axes_el_flat[idx_ch]
                ch_a = group_hep_per_channel.get(g_a, {}).get(ch)
                ch_b = group_hep_per_channel.get(g_b, {}).get(ch)

                if ch_a is None or ch_b is None or ch_a.shape[0] == 0 or ch_b.shape[0] == 0:
                    ax_el.set_title(ch, fontsize=9)
                    ax_el.text(0.5, 0.5, 'No data', ha='center', va='center',
                               transform=ax_el.transAxes, fontsize=8, color='gray')
                    ax_el.axis('off')
                    continue

                min_t_el = min(ch_a.shape[1], ch_b.shape[1], len(t_common))
                ca = ch_a[:, :min_t_el]
                cb = ch_b[:, :min_t_el]
                t_el = t_common[:min_t_el]

                per_ch_save[ch] = (ca, cb, t_el)

                mean_ca = np.nanmean(ca, 0)
                mean_cb = np.nanmean(cb, 0)
                sem_ca  = np.nanstd(ca, 0, ddof=1) / np.sqrt(ca.shape[0])
                sem_cb  = np.nanstd(cb, 0, ddof=1) / np.sqrt(cb.shape[0])

                ax_el.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
                ax_el.axhline(0, color='black', linewidth=0.3, alpha=0.3)
                ax_el.plot(t_el, mean_ca, color=group_color[g_a], linewidth=1.5,
                           label=f"{g_a} (n={ca.shape[0]})")
                ax_el.fill_between(t_el, mean_ca - sem_ca, mean_ca + sem_ca,
                                   color=group_color[g_a], alpha=0.18)
                ax_el.plot(t_el, mean_cb, color=group_color[g_b], linewidth=1.5,
                           label=f"{g_b} (n={cb.shape[0]})")
                ax_el.fill_between(t_el, mean_cb - sem_cb, mean_cb + sem_cb,
                                   color=group_color[g_b], alpha=0.18)

                # Quick permutation test for this channel (fewer perms for speed)
                try:
                    sig_el, _, cd_el = permutation_two_group_cluster_test(
                        ch_a, ch_b, t_el,
                        n_permutations=max(50, n_permutations // 4),
                        p_threshold=0.05,
                        jitter_sec=jitter_sec,
                        channel_label=ch,
                    )
                    for win in sig_el:
                        ax_el.axvspan(win['start'], win['end'], color='orange', alpha=0.28)
                    if sig_el:

                        min_p_el = min(w['p_value'] for w in sig_el)
                        p_tag_el = 'p<0.001' if min_p_el < 0.001 else f'p={min_p_el:.3f}'
                    else:
                        p_tag_el = 'n.s.'
                except Exception:
                    p_tag_el = ''

                ax_el.set_title(f"{ch}  [{p_tag_el}]", fontsize=9, fontweight='bold')
                ax_el.set_xlim(t_el[0], t_el[-1])
                ax_el.grid(True, alpha=0.2)
                ax_el.tick_params(labelsize=7)
                if idx_ch % n_cols_el == 0:
                    ax_el.set_ylabel(amp_ylabel, fontsize=8)

            # Hide unused axes
            for j in range(len(common_channels), len(axes_el_flat)):
                axes_el_flat[j].axis('off')

            # Determine valid N for each group
            n_a = len(group_individuals.get(g_a, []))
            n_b = len(group_individuals.get(g_b, []))

            # Single legend outside
            handles_el = [
                plt.Line2D([0], [0], color=group_color[g_a], linewidth=2, label=f"{g_a} (N={n_a})"),
                plt.Line2D([0], [0], color=group_color[g_b], linewidth=2, label=f"{g_b} (N={n_b})"),
            ]
            fig_el.legend(handles=handles_el, loc='upper right', fontsize=10,
                          bbox_to_anchor=(1.0, 1.0))
            fig_el.suptitle(
                f"Per-Electrode HEP Comparison: {g_a} vs {g_b} — Stage: {selected_stage}",
                fontsize=14, fontweight='bold', y=1.01
            )
            fig_el.tight_layout()
            st.pyplot(fig_el, use_container_width=True)
            plt.close(fig_el)

        else:
            st.info("Per-electrode plots require per-channel data.")

    elif n_groups > 2:
        st.info(
            f"Significance testing is shown for exactly 2 groups. "
            f"You have {n_groups} groups loaded. Select a subset or compare pairs manually."
        )

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 3 — Channel × Time heatmap of mean difference (2 groups only)
    # ═══════════════════════════════════════════════════════════════════════
    if n_groups == 2 and common_channels:
        g_a, g_b = groups_with_data[0], groups_with_data[1]
        st.subheader(f"🗺️ Topographic Difference Heatmap: {g_a} − {g_b}  (per EEG channel)")

        diff_matrix = []   # (n_channels, n_times)
        valid_chs = []
        for ch in common_channels:
            ch_a = group_hep_per_channel.get(g_a, {}).get(ch)
            ch_b = group_hep_per_channel.get(g_b, {}).get(ch)
            if ch_a is not None and ch_b is not None and len(ch_a) > 0 and len(ch_b) > 0:
                min_t = min(ch_a.shape[1], ch_b.shape[1])
                diff_row = np.nanmean(ch_a[:, :min_t], 0) - np.nanmean(ch_b[:, :min_t], 0)
                diff_matrix.append(diff_row)
                valid_chs.append(ch)

        if diff_matrix:
            diff_arr = np.array(diff_matrix)   # (n_ch, n_times)
            t_hm = times[:diff_arr.shape[1]]

            fig3, ax3 = plt.subplots(figsize=(14, max(4, len(valid_chs) * 0.35 + 2)))
            vmax = np.nanpercentile(np.abs(diff_arr), 98)

            im = ax3.imshow(
                diff_arr,
                aspect='auto',
                origin='lower',
                cmap='RdBu_r',
                vmin=-vmax, vmax=vmax,
                extent=[t_hm[0], t_hm[-1], -0.5, len(valid_chs) - 0.5]
            )
            ax3.set_yticks(range(len(valid_chs)))
            ax3.set_yticklabels(valid_chs, fontsize=8)
            ax3.axvline(0, color='black', linestyle='--', alpha=0.7, linewidth=1.5)

            # Mark significant windows on heatmap
            for win in sig_windows_global:
                ax3.axvspan(win['start'], win['end'], color='yellow', alpha=0.18)

            cbar = fig3.colorbar(im, ax=ax3, pad=0.01, fraction=0.025)
            cbar.set_label(f"Mean diff {'Z-score' if use_zscore else 'µV'} ({g_a}−{g_b})", fontsize=10)

            ax3.set_xlabel("Time relative to R-peak (s)", fontsize=11)
            ax3.set_ylabel("EEG Channel", fontsize=11)
            ax3.set_title(
                f"Mean HEP Difference per Channel  |  {g_a} − {g_b}  |  Stage: {selected_stage}",
                fontsize=13, fontweight='bold'
            )
            fig3.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)
        else:
            st.warning("No common channels with data in both groups for the heatmap.")

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 4 — Spatial Topomaps of Difference & P-value
    # ═══════════════════════════════════════════════════════════════════════
    if n_groups == 2 and common_channels:
        st.divider()
        st.subheader("Spatial Topomaps - Group Difference & Significance")
        st.markdown(f"Select a time window to visualize the {g_a} − {g_b} average difference distribution and spatial significance across the scalp.")
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            topo_diff_tmin = st.number_input("Start Time (ms)", min_value=-500, max_value=1000, value=200, step=10, key="topo_diff_tmin_compare")
        with col_t2:
            topo_diff_tmax = st.number_input("End Time (ms)", min_value=-500, max_value=1000, value=400, step=10, key="topo_diff_tmax_compare")
            
        if topo_diff_tmin >= topo_diff_tmax:
            st.warning("Start time must be less than end time.")
        else:
            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                montage_ch_names_upper = [ch.upper() for ch in montage.ch_names]
                
                times_ms = times * 1000
                t_mask = (times_ms >= topo_diff_tmin) & (times_ms <= topo_diff_tmax)
                
                if not np.any(t_mask):
                    st.warning("Invalid time window selected (no data points).")
                else:
                    g_a, g_b = groups_with_data[0], groups_with_data[1]
                    hep_a = group_hep_matrix[g_a]
                    hep_b = group_hep_matrix[g_b]
                    
                    min_len = min(hep_a.shape[1], hep_b.shape[1])
                    t_mask_min = t_mask[:min_len]
                    
                    if not np.any(t_mask_min):
                        st.warning("Invalid time window after alignment.")
                    else:
                        
                        plot_ch_names = []
                        plot_data_diff = []
                        plot_data_pval_diff = []
                        
                        plot_data_amp_a = []
                        plot_data_amp_b = []
                        plot_data_pval_a = []
                        plot_data_pval_b = []
                        
                        t_win = times[:min_len][t_mask_min]
                        
                        for i, ch in enumerate(common_channels):
                            ch_upper = ch.upper()
                            if ch_upper in montage_ch_names_upper:
                                m_idx = montage_ch_names_upper.index(ch_upper)
                                standard_name = montage.ch_names[m_idx]
                                
                                # Extract channel data
                                ch_a_data = group_hep_per_channel.get(g_a, {}).get(ch)
                                ch_b_data = group_hep_per_channel.get(g_b, {}).get(ch)
                                
                                if ch_a_data is not None and ch_b_data is not None:
                                    ch_a_win = ch_a_data[:, :min_len][:, t_mask_min]
                                    ch_b_win = ch_b_data[:, :min_len][:, t_mask_min]
                                    
                                    # Mean amplitudes in window
                                    amp_a = np.nanmean(ch_a_win) * 1e6
                                    amp_b = np.nanmean(ch_b_win) * 1e6
                                    diff_mean_uv = amp_a - amp_b
                                    
                                    # P-value calculation for Difference (A vs B)
                                    sig_windows_diff, _, _ = permutation_two_group_cluster_test(
                                        ch_a_win, ch_b_win, t_win,
                                        n_permutations=n_permutations,
                                        p_threshold=0.05,
                                        jitter_sec=jitter_sec,
                                        channel_label=ch,
                                    )
                                    p_val_diff = min([w['p_value'] for w in sig_windows_diff]) if sig_windows_diff else 1.0

                                    # P-value calculation for Group A (vs 0)
                                    sig_windows_a, _, _ = permutation_cluster_jitter_test(
                                        ch_a_win, t_win,
                                        n_permutations=n_permutations,
                                        p_threshold=0.05,
                                        jitter_sec=jitter_sec,
                                    )
                                    p_val_a = min([w['p_value'] for w in sig_windows_a]) if sig_windows_a else 1.0

                                    # P-value calculation for Group B (vs 0)
                                    sig_windows_b, _, _ = permutation_cluster_jitter_test(
                                        ch_b_win, t_win,
                                        n_permutations=n_permutations,
                                        p_threshold=0.05,
                                        jitter_sec=jitter_sec,
                                    )
                                    p_val_b = min([w['p_value'] for w in sig_windows_b]) if sig_windows_b else 1.0
                                        
                                    plot_ch_names.append(standard_name)
                                    plot_data_diff.append(diff_mean_uv)
                                    plot_data_pval_diff.append(p_val_diff)
                                    
                                    plot_data_amp_a.append(amp_a)
                                    plot_data_amp_b.append(amp_b)
                                    plot_data_pval_a.append(p_val_a)
                                    plot_data_pval_b.append(p_val_b)

                        # Pad missing 10-20 standard channels
                        standard_19_base = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2']
                        aliases = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}
                        
                        def pad_channel_if_missing(ch_name, pad_amp, pad_pval):
                            if not any(ch_name.upper() == p_ch.upper() for p_ch in plot_ch_names):
                                plot_ch_names.append(ch_name)
                                plot_data_diff.append(0.0)
                                plot_data_pval_diff.append(pad_pval)
                                plot_data_amp_a.append(pad_amp)
                                plot_data_amp_b.append(pad_amp)
                                plot_data_pval_a.append(pad_pval)
                                plot_data_pval_b.append(pad_pval)
                        
                        for base_ch in standard_19_base:
                            pad_channel_if_missing(base_ch, 0.0, 0.05)
                                
                        for new_name, old_name in aliases.items():
                            if not any(ch.upper() in [new_name.upper(), old_name.upper()] for ch in plot_ch_names):
                                pad_channel_if_missing(new_name, 0.0, 0.05)

                        if not plot_ch_names:
                            st.warning("No channels matched the standard montage.")
                        else:
                            info = mne.create_info(ch_names=plot_ch_names, sfreq=250., ch_types='eeg')
                            info.set_montage(montage)
                            
                            max_amp_a = np.max(np.abs(plot_data_amp_a)) if plot_data_amp_a else 1.0
                            max_amp_b = np.max(np.abs(plot_data_amp_b)) if plot_data_amp_b else 1.0
                            max_amp = max(max_amp_a, max_amp_b)
                            if max_amp == 0: max_amp = 1.0
                            
                            max_diff = np.max(np.abs(plot_data_diff)) if plot_data_diff else 1.0
                            if max_diff == 0: max_diff = 1.0

                            st.markdown("#### 1. Mean Amplitude in Time Window")
                            fig_amp, axes_amp = plt.subplots(1, 2, figsize=(10, 4))
                            
                            # Group A Amplitude
                            res_amp_a = mne.viz.plot_topomap(
                                np.array(plot_data_amp_a), info, axes=axes_amp[0], cmap='RdBu_r', 
                                names=plot_ch_names, vlim=(-max_amp, max_amp), extrapolate='head', show=False
                            )
                            im_amp_a = res_amp_a[0] if isinstance(res_amp_a, tuple) else res_amp_a
                            axes_amp[0].set_title(f"{g_a} Amplitude")
                            
                            # Group B Amplitude
                            res_amp_b = mne.viz.plot_topomap(
                                np.array(plot_data_amp_b), info, axes=axes_amp[1], cmap='RdBu_r', 
                                names=plot_ch_names, vlim=(-max_amp, max_amp), extrapolate='head', show=False
                            )
                            axes_amp[1].set_title(f"{g_b} Amplitude")
                            
                            fig_amp.subplots_adjust(right=0.85, wspace=0.4)
                            if im_amp_a is not None:
                                cbar_ax_amp = fig_amp.add_axes([0.92, 0.15, 0.02, 0.7])
                                cbar_amp = fig_amp.colorbar(im_amp_a, cax=cbar_ax_amp)
                                cbar_amp.set_label(f"Amplitude ({'Z-score' if use_zscore else 'µV'})")
                            st.pyplot(fig_amp, use_container_width=False)
                            plt.close(fig_amp)

                            st.markdown("#### 2. Significance vs Baseline (p-value)")
                            fig_pval_ind, axes_pval_ind = plt.subplots(1, 2, figsize=(10, 4))
                            
                            # Group A P-value
                            data_pval_a_clip = np.clip(np.array(plot_data_pval_a), 0, 0.05)
                            res_pval_a = mne.viz.plot_topomap(
                                data_pval_a_clip, info, axes=axes_pval_ind[0], cmap='Reds_r', 
                                names=plot_ch_names, vlim=(0, 0.05), extrapolate='head', show=False
                            )
                            im_pval_a = res_pval_a[0] if isinstance(res_pval_a, tuple) else res_pval_a
                            axes_pval_ind[0].set_title(f"{g_a} P-value (vs 0)")
                            
                            # Group B P-value
                            data_pval_b_clip = np.clip(np.array(plot_data_pval_b), 0, 0.05)
                            res_pval_b = mne.viz.plot_topomap(
                                data_pval_b_clip, info, axes=axes_pval_ind[1], cmap='Reds_r', 
                                names=plot_ch_names, vlim=(0, 0.05), extrapolate='head', show=False
                            )
                            axes_pval_ind[1].set_title(f"{g_b} P-value (vs 0)")
                            
                            fig_pval_ind.subplots_adjust(right=0.85, wspace=0.4)
                            if im_pval_a is not None:
                                cbar_ax_pval_ind = fig_pval_ind.add_axes([0.92, 0.15, 0.02, 0.7])
                                cbar_pval_ind = fig_pval_ind.colorbar(im_pval_a, cax=cbar_ax_pval_ind)
                                cbar_pval_ind.set_label("p-value")
                            st.pyplot(fig_pval_ind, use_container_width=False)
                            plt.close(fig_pval_ind)

                            st.markdown("#### 3. Group Difference & Significance")
                            fig_diff_topo, axes_diff_topo = plt.subplots(1, 2, figsize=(10, 4))
                            
                            # Difference Topomap
                            result_diff = mne.viz.plot_topomap(
                                np.array(plot_data_diff), info, axes=axes_diff_topo[0], cmap='RdBu_r', 
                                names=plot_ch_names, vlim=(-max_diff, max_diff), extrapolate='head', show=False
                            )
                            im_diff = result_diff[0] if isinstance(result_diff, tuple) else result_diff
                            axes_diff_topo[0].set_title(f"Difference ({g_a} - {g_b})")
                            
                            # P-value Difference Topomap
                            data_pval_diff = np.clip(np.array(plot_data_pval_diff), 0, 0.05)
                            result_pval_diff = mne.viz.plot_topomap(
                                data_pval_diff, info, axes=axes_diff_topo[1], cmap='Reds_r', 
                                names=plot_ch_names, vlim=(0, 0.05), extrapolate='head', show=False
                            )
                            im_pval_diff = result_pval_diff[0] if isinstance(result_pval_diff, tuple) else result_pval_diff
                            axes_diff_topo[1].set_title("P-value (Diff)")
                            
                            fig_diff_topo.subplots_adjust(right=0.85, wspace=0.4)
                            
                            if im_diff is not None:
                                cbar_ax_diff = fig_diff_topo.add_axes([0.43, 0.15, 0.02, 0.7])
                                cbar_diff = fig_diff_topo.colorbar(im_diff, cax=cbar_ax_diff)
                                cbar_diff.set_label(f"Mean Diff ({'Z-score' if use_zscore else 'µV'})")
                                
                            if im_pval_diff is not None:
                                cbar_ax_pval_diff = fig_diff_topo.add_axes([0.92, 0.15, 0.02, 0.7])
                                cbar_pval_diff = fig_diff_topo.colorbar(im_pval_diff, cax=cbar_ax_pval_diff)
                                cbar_pval_diff.set_label("p-value")
                                
                            st.pyplot(fig_diff_topo, use_container_width=False)
                            plt.close(fig_diff_topo)
                            
            except Exception as e:
                st.error(f"Error generating topomaps: {str(e)}")

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY STATISTICS TABLE
    # ═══════════════════════════════════════════════════════════════════════
    st.subheader("📋 Summary Statistics")
    summary_rows = []
    for group in groups_with_data:
        mat = group_hep_matrix[group]  # already normalised (Z-scored or µV)
        grand = np.nanmean(mat, 0)
        n_subj = mat.shape[0]
        peak_idx = np.argmax(np.abs(grand))
        peak_amp = grand[peak_idx]
        peak_t = group_times[group][peak_idx]
        mean_amp = np.nanmean(grand)
        sd_amp = np.nanstd(grand)
        unit_lbl = "Z" if use_zscore else "µV"
        row = {
            "Group": group,
            "N": n_subj,
            f"Mean amplitude ({unit_lbl})": f"{mean_amp:.4f}",
            f"SD amplitude ({unit_lbl})": f"{sd_amp:.4f}",
            f"Peak amplitude ({unit_lbl})": f"{peak_amp:.4f}",
            "Peak latency (s)": f"{peak_t:.3f}",
        }
        # Add between-group stats at peak time (2-group case)
        if n_groups == 2:
            g_a, g_b = groups_with_data[0], groups_with_data[1]
            hep_at_peak_a = group_hep_matrix[g_a][:, peak_idx]
            hep_at_peak_b = group_hep_matrix[g_b][:, peak_idx]
            t_stat_peak, p_val_peak = stats.ttest_ind(hep_at_peak_a, hep_at_peak_b)
            na2, nb2 = len(hep_at_peak_a), len(hep_at_peak_b)
            sd_pool = np.sqrt(
                ((na2 - 1) * np.var(hep_at_peak_a, ddof=1) + (nb2 - 1) * np.var(hep_at_peak_b, ddof=1))
                / (na2 + nb2 - 2)
            )
            d_peak = abs(np.mean(hep_at_peak_a) - np.mean(hep_at_peak_b)) / (sd_pool + 1e-12)
            if group == g_a:
                row["t-stat at peak"] = f"{t_stat_peak:.3f}"
                row["p-value at peak"] = "<0.001" if p_val_peak < 0.001 else f"{p_val_peak:.3f}"
                row["Cohen's d at peak"] = f"{d_peak:.3f}"
        summary_rows.append(row)

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)




if TORCH_AVAILABLE:
    class ConvBlock(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(out_c)
            self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(out_c)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))

    class UNet1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = ConvBlock(1, 64)
            self.enc2 = ConvBlock(64, 128)
            self.enc3 = ConvBlock(128, 256)
            self.pool = nn.MaxPool1d(2)
            self.up2 = nn.ConvTranspose1d(256, 128, 2, stride=2)
            self.dec2 = ConvBlock(256, 128)
            self.up1 = nn.ConvTranspose1d(128, 64, 2, stride=2)
            self.dec1 = ConvBlock(128, 64)
            self.out = nn.Conv1d(64, 1, 1)
        
        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            b  = self.enc3(self.pool(e2))
            d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return torch.sigmoid(self.out(d1))

    class ECG_CNN_RPeak(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128), nn.ReLU(),
            )
            self.classifier = nn.Conv1d(128, 1, kernel_size=1)
        
        def forward(self, x):
            return torch.sigmoid(self.classifier(self.encoder(x)))

    class BiLSTM_RPeak(nn.Module):
        def __init__(self):
            super().__init__()
            self.bilstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(128, 1)
        
        def forward(self, x):
            out, _ = self.bilstm(x)
            return torch.sigmoid(self.fc(out))

def swt_preprocess(ecg, level=4, wavelet='db4'):
    if not PYWT_AVAILABLE:
        return np.zeros((2, len(ecg)))
    coeffs = pywt.swt(ecg, wavelet, level=level)
    detail_l4 = coeffs[level-1][1]       # Detail at level 4
    derivative = np.diff(ecg, prepend=ecg[0])
    return np.stack([detail_l4, derivative], axis=0)  # 2-channel DL input

def plot_ecg_cleaning_comparison(raw_segment, clean_segment, sfreq, patient_id):
    """
    Plots a 10 second segment of raw vs cleaned ECG and compares R-peak detection methods.
    """
    times = np.arange(len(raw_segment)) / sfreq

    # --- Compute 3 Methods ---
    methods_peaks = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 1. NeuroKit2
    status_text.text("Running NeuroKit2 (1/3)...")
    try:
        import neurokit2 as nk
        _, rpk_nk = nk.ecg_peaks(clean_segment, sampling_rate=sfreq,method='promac')
        methods_peaks['NeuroKit2'] = rpk_nk['ECG_R_Peaks']
    except Exception as e:
        methods_peaks[f'NeuroKit2 (Error: {e})'] = []
    progress_bar.progress(33)
    
    # 2. WFDB
    status_text.text("Running WFDB (2/3)...")
    try:
        import wfdb.processing as wp
        rpeaks_wfdb = wp.xqrs_detect(clean_segment, fs=sfreq, verbose=False)
        # Optional: correct peak locations to local maxima
        r_peaks_corrected = wp.correct_peaks(clean_segment, rpeaks_wfdb, 
                                            search_radius=int(0.05*sfreq),
                                            smooth_window_size=int(0.1*sfreq))
        methods_peaks['WFDB XQRS'] = r_peaks_corrected
    except Exception as e:
        methods_peaks[f'WFDB (Error: {e})'] = []
    progress_bar.progress(66)

    # 3. WFDB Robust
    status_text.text("Running WFDB part 2 (3/3)...")
    try:
        methods_peaks['WFDB Robust'] = detect_rpeaks_robust(clean_segment, sfreq)
    except Exception as e:
        methods_peaks[f'WFDB Robust (Error: {e})'] = []

    progress_bar.progress(100)
    status_text.text("Processing Complete!")

    # --- Plotting ---
    method_names = list(methods_peaks.keys())
    fig, axes = plt.subplots(len(method_names), 1, figsize=(16, 3 * len(method_names)), sharex=True)
    
    if len(method_names) == 1:
        axes = [axes]

    for idx, (method_name, rpeaks) in enumerate(methods_peaks.items()):
        ax = axes[idx]
        ax.plot(times, raw_segment, color='gray', alpha=0.3, label='Raw' if idx == 0 else None)
        ax.plot(times, clean_segment, color='red', alpha=0.7, label='Cleaned' if idx == 0 else None)
        
        for i, peak in enumerate(rpeaks):
            peak_time = peak / sfreq
            ax.axvline(peak_time, color='green', linestyle='--', alpha=0.6, 
                       label='R-peak' if (idx == 0 and i == 0) else None)
        
        ax.set_ylabel("Amp (μV)")
        ax.set_title(f"Method: {method_name}")
        if idx == 0:
            ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

def plot_ecg_hep_individuals(individuals, selected_group, selected_stage):
    """
    Plots ECG HEP for all individuals and their average.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    all_ecg_heps = []
    # allow user to select how many individuals to plot
    n_individuals = st.slider("Number of individuals to plot", min_value=1, max_value=len(individuals), value=len(individuals))
    for ind in individuals[:n_individuals]:
        # ind structure: (patient_id, hep_data, times, ch_names, rpeaks, ecg_hep_data, ecg_ch_names)
        ecg_hep = ind[5]
        if ecg_hep is not None:
            hep = ecg_hep[0]
            ax.plot(ind[2], hep * 1e6, color='gray', alpha=0.3, linewidth=1)
            all_ecg_heps.append(hep)
    
    if all_ecg_heps:
        avg_ecg = np.nanmean(all_ecg_heps, axis=0)
        finalize_plot(
            fig, ax, 
            f"ECG - Group: {selected_group} - Stage: {selected_stage}",
            avg_hep=avg_ecg, 
            times=individuals[0][2], 
            n_subjects=len(all_ecg_heps)
        )
        
        # Calculate epochs stats
        epochs_data = [{"Patient ID": ind[0], "Epochs": len(ind[4])} for ind in individuals]
        if epochs_data:
            df_epochs = pd.DataFrame(epochs_data)
            avg_epochs = df_epochs["Epochs"].mean()
            st.write(f"**Average Epochs per Patient:** {avg_epochs:.2f}")
            
            with st.expander("See Epochs per Patient"):
                st.dataframe(df_epochs)
                
    else:
        st.warning("No ECG data found for this group.")

def analyze_ecg_repetitive_pattern(ecg_signal, sfreq, patient_id=""):
    """
    Detects repetitive waveform templates/motifs in ECG signal.
    This identifies identical wave patterns that repeat throughout the recording.
    
    Returns:
    --------
    pattern_metrics : dict
        Dictionary containing:
        - 'patient_id': Patient identifier
        - 'template': Extracted repetitive template waveform
        - 'template_times': Time axis for template
        - 'match_locations': Sample indices where template repeats
        - 'match_correlations': Correlation strength at each match location
        - 'avg_correlation': Average correlation strength
        - 'repetition_rate': Matches per second
        - 'has_repetitive_pattern': Boolean indicating significant repetition
    """
    from scipy.signal import correlate, find_peaks

    # Use a manageable segment for analysis (first 60 seconds)
    max_samples = min(len(ecg_signal), int(60 * sfreq))
    signal_segment = ecg_signal[:max_samples]
    
    # Template extraction parameters
    # Try different template durations to find repeating patterns
    template_durations = [0.05, 0.1, 0.2, 0.3, 0.5]  # seconds
    
    best_template = None
    best_matches = []
    best_correlations = []
    best_avg_corr = 0
    best_duration = 0
    
    for template_dur in template_durations:
        template_len = int(template_dur * sfreq)
        
        if template_len >= len(signal_segment) // 4:
            continue
        
        # Strategy: Find the most repetitive segment by testing multiple candidates
        # Sample multiple starting points
        n_candidates = min(20, len(signal_segment) // template_len)
        candidate_starts = np.linspace(0, len(signal_segment) - template_len, n_candidates, dtype=int)
        
        for start_idx in candidate_starts:
            # Extract candidate template
            template = signal_segment[start_idx:start_idx + template_len]
            
            # Normalize template
            template_norm = (template - np.mean(template)) / (np.std(template) + 1e-10)
            
            # Normalize signal segment
            signal_norm = (signal_segment - np.mean(signal_segment)) / (np.std(signal_segment) + 1e-10)
            
            # Cross-correlate to find matches
            correlation = correlate(signal_norm, template_norm, mode='valid')
            
            # Find peaks in correlation (high correlation = good match)
            # Require minimum separation between matches
            min_separation = template_len // 2
            threshold = 0.7  # Correlation threshold for a "good match"
            
            peak_indices, properties = get_peaks(correlation, 
                                                   height=threshold, 
                                                   distance=min_separation)
            
            if len(peak_indices) > 0:
                avg_corr = np.mean(correlation[peak_indices])
                
                # Keep the template with the most repetitions and highest avg correlation
                if avg_corr > best_avg_corr and len(peak_indices) >= 3:
                    best_avg_corr = avg_corr
                    best_template = template
                    best_matches = peak_indices
                    best_correlations = correlation[peak_indices]
                    best_duration = template_dur
    
    # Prepare return values
    if best_template is not None:
        template_times = np.arange(len(best_template)) / sfreq
        repetition_rate = len(best_matches) / (len(signal_segment) / sfreq)
        has_repetitive_pattern = len(best_matches) >= 5 and best_avg_corr >= 0.75
    else:
        # No strong repetitive pattern found
        template_times = np.array([])
        repetition_rate = 0
        has_repetitive_pattern = False
        best_template = np.array([])
        best_matches = []
        best_correlations = []
    
    return {
        'patient_id': patient_id,
        'template': best_template,
        'template_times': template_times,
        'template_duration': best_duration,
        'match_locations': best_matches,
        'match_correlations': best_correlations,
        'avg_correlation': best_avg_corr,
        'repetition_rate': repetition_rate,
        'n_matches': len(best_matches),
        'has_repetitive_pattern': has_repetitive_pattern,
        'signal_segment': signal_segment,  # For visualization
        'sfreq': sfreq
    }

def handle_ecg_noise_detection(base_path, selected_group, selected_stage):
    """
    Analyzes ECG for repetitive waveform patterns in all patients and displays results.
    """
    st.subheader(f"ECG Repetitive Pattern Analysis - {selected_group} {selected_stage}")
    
    st.info("This analysis detects identical waveform patterns that repeat throughout each patient's ECG recording.")
    
    group_dir = os.path.join(base_path, selected_group, selected_stage)
    files = [f for f in os.listdir(group_dir) if f.endswith('.pkl')]
    
    if not files:
        st.warning("No files found for pattern analysis.")
        return
    
    all_metrics = []
    
    progress_bar = st.progress(0)
    
    for idx, f_name in enumerate(files):
        progress_bar.progress((idx + 1) / len(files))
        
        f_path = os.path.join(group_dir, f_name)
        patient_id = f_name.replace('.pkl', '').replace('.edf', '')
        
        try:
            with open(f_path, 'rb') as f:
                raw = pickle.load(f)
            
            # Extract ECG channel
            sfreq = raw.info['sfreq']
            ch_names = raw.ch_names
            ch_lower = [ch.lower() for ch in ch_names]
            ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
            
            if not ecg_indices:
                continue
            
            ecg_idx = ecg_indices[0]
            ecg_signal = raw.get_data(picks=[ecg_idx])[0] * 1e6  # Convert to uV
            
            # Analyze for repetitive patterns
            metrics = analyze_ecg_repetitive_pattern(ecg_signal, sfreq, patient_id)
            all_metrics.append(metrics)
            
        except Exception as e:
            st.warning(f"Error processing {patient_id}: {e}")
            continue
    
    progress_bar.empty()
    
    if not all_metrics:
        st.error("No valid ECG data found for pattern analysis.")
        return
    
    # --- Summary Statistics ---
    st.markdown("### Group Summary")
    
    n_patients = len(all_metrics)
    n_with_pattern = sum(1 for m in all_metrics if m['has_repetitive_pattern'])
    
    # Calculate averages only for patients with detected patterns
    valid_metrics = [m for m in all_metrics if len(m['template']) > 0]
    if valid_metrics:
        avg_repetition_rate = np.mean([m['repetition_rate'] for m in valid_metrics])
        avg_correlation = np.mean([m['avg_correlation'] for m in valid_metrics])
        avg_n_matches = np.mean([m['n_matches'] for m in valid_metrics])
    else:
        avg_repetition_rate = 0
        avg_correlation = 0
        avg_n_matches = 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", n_patients)
    with col2:
        st.metric("With Repetitive Pattern", f"{n_with_pattern} ({100*n_with_pattern/n_patients:.1f}%)")
    with col3:
        st.metric("Avg Repetition Rate", f"{avg_repetition_rate:.2f} /sec")
    with col4:
        st.metric("Avg Correlation", f"{avg_correlation:.3f}")
    
    # --- Per-Patient Table ---
    st.markdown("### Per-Patient Pattern Metrics")
    
    df_data = []
    for m in all_metrics:
        if len(m['template']) > 0:
            df_data.append({
                'Patient ID': m['patient_id'],
                'Pattern Duration (ms)': f"{m['template_duration']*1000:.1f}",
                'Num Repetitions': m['n_matches'],
                'Repetition Rate (/sec)': f"{m['repetition_rate']:.2f}",
                'Avg Correlation': f"{m['avg_correlation']:.3f}",
                'Has Strong Pattern': '✓' if m['has_repetitive_pattern'] else '✗'
            })
        else:
            df_data.append({
                'Patient ID': m['patient_id'],
                'Pattern Duration (ms)': 'N/A',
                'Num Repetitions': 0,
                'Repetition Rate (/sec)': '0.00',
                'Avg Correlation': 'N/A',
                'Has Strong Pattern': '✗'
            })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)
    
    # --- Visualization: Individual Patient Templates ---
    st.markdown("### Individual Patient Templates")
    st.markdown("Shows the extracted repetitive waveform pattern for each patient")
    
    # Filter patients with valid templates
    patients_with_templates = [m for m in all_metrics if len(m['template']) > 0]
    
    if not patients_with_templates:
        st.warning("No repetitive patterns detected in any patient.")
        return
    
    n_show = st.slider("Number of patients to show", min_value=1, 
                       max_value=min(12, len(patients_with_templates)), 
                       value=min(6, len(patients_with_templates)), 
                       key="pattern_patient_slider")
    
    # Create grid of template plots
    n_cols = 3
    n_rows = int(np.ceil(n_show / n_cols))
    
    fig_templates, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(n_show):
        ax = axes[i]
        m = patients_with_templates[i]
        
        # Plot the template
        ax.plot(m['template_times'] * 1000, m['template'], color='red', linewidth=2)
        
        pattern_status = "STRONG" if m['has_repetitive_pattern'] else "WEAK"
        ax.set_title(f"{m['patient_id']} - {pattern_status}\n"
                     f"Repeats: {m['n_matches']}, Corr: {m['avg_correlation']:.2f}", 
                     fontsize=9)
        ax.set_xlabel('Time (ms)', fontsize=8)
        ax.set_ylabel('Amplitude (μV)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    
    # Hide unused subplots
    for j in range(n_show, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig_templates, use_container_width=True)
    
    # --- Detailed View: Template in Context ---
    st.markdown("### Detailed Pattern View")
    st.markdown("View the repetitive pattern in context of the full ECG signal")
    
    selected_patient_idx = st.selectbox(
        "Select patient for detailed view",
        range(len(patients_with_templates)),
        format_func=lambda i: patients_with_templates[i]['patient_id']
    )
    
    m = patients_with_templates[selected_patient_idx]
    
    # Show template
    st.markdown(f"**Patient: {m['patient_id']}**")
    st.markdown(f"- Pattern duration: {m['template_duration']*1000:.1f} ms")
    st.markdown(f"- Found {m['n_matches']} repetitions (rate: {m['repetition_rate']:.2f} /sec)")
    st.markdown(f"- Average correlation: {m['avg_correlation']:.3f}")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Plot template
        fig_temp, ax_temp = plt.subplots(figsize=(7, 4))
        ax_temp.plot(m['template_times'] * 1000, m['template'], color='red', linewidth=2)
        ax_temp.set_title("Extracted Repetitive Template", fontsize=12, fontweight='bold')
        ax_temp.set_xlabel('Time (ms)')
        ax_temp.set_ylabel('Amplitude (μV)')
        ax_temp.grid(True, alpha=0.3)
        st.pyplot(fig_temp, use_container_width=True)
    
    with col_b:
        # Plot correlation values at match locations
        fig_corr, ax_corr = plt.subplots(figsize=(7, 4))
        ax_corr.bar(range(len(m['match_correlations'])), m['match_correlations'], 
                    color='steelblue', alpha=0.7)
        ax_corr.axhline(0.75, color='red', linestyle='--', label='Strong Match Threshold')
        ax_corr.set_title("Correlation Strength per Repetition", fontsize=12, fontweight='bold')
        ax_corr.set_xlabel('Repetition Number')
        ax_corr.set_ylabel('Correlation Coefficient')
        ax_corr.set_ylim(0, 1)
        ax_corr.legend()
        ax_corr.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig_corr, use_container_width=True)
    
    # Show signal with marked repetitions
    st.markdown("**Signal with Pattern Locations Marked**")
    
    # User can select time window
    signal_duration = len(m['signal_segment']) / m['sfreq']
    view_window = st.slider("View window (seconds)", 
                            min_value=2, 
                            max_value=min(30, int(signal_duration)), 
                            value=10,
                            key="pattern_window_slider")
    
    start_time = st.slider("Start time (seconds)", 
                          min_value=0.0, 
                          max_value=max(0.0, signal_duration - view_window),
                          value=0.0,
                          step=0.5,
                          key="pattern_start_slider")
    
    start_sample = int(start_time * m['sfreq'])
    end_sample = min(len(m['signal_segment']), int((start_time + view_window) * m['sfreq']))
    
    signal_view = m['signal_segment'][start_sample:end_sample]
    times_view = np.arange(len(signal_view)) / m['sfreq'] + start_time
    
    fig_signal, ax_signal = plt.subplots(figsize=(15, 5))
    ax_signal.plot(times_view, signal_view, color='black', linewidth=1, alpha=0.7, label='ECG Signal')
    
    # Mark pattern locations in the view window
    for match_loc in m['match_locations']:
        match_time = match_loc / m['sfreq']
        if start_time <= match_time <= start_time + view_window:
            ax_signal.axvline(match_time, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax_signal.set_title("ECG Signal with Repetitive Pattern Locations (Red lines)", fontsize=12)
    ax_signal.set_xlabel('Time (s)')
    ax_signal.set_ylabel('Amplitude (μV)')
    ax_signal.grid(True, alpha=0.3)
    ax_signal.legend()
    st.pyplot(fig_signal, use_container_width=True)
    
    # --- Average Template Across All Patients ---
    st.markdown("### Average Template Across All Patients")
    
    # Find common duration (use median)
    template_lengths = [len(m['template']) for m in patients_with_templates]
    median_length = int(np.median(template_lengths))
    
    # Interpolate all templates to median length and average
    from scipy.interpolate import interp1d
    
    aligned_templates = []
    for m in patients_with_templates:
        if len(m['template']) > 0:
            # Interpolate to median length
            x_old = np.linspace(0, 1, len(m['template']))
            x_new = np.linspace(0, 1, median_length)
            f = interp1d(x_old, m['template'], kind='linear')
            template_aligned = f(x_new)
            aligned_templates.append(template_aligned)
    
    if aligned_templates:
        avg_template = np.mean(aligned_templates, axis=0)
        std_template = np.std(aligned_templates, axis=0)
        
        # Use median duration for time axis
        median_duration = np.median([m['template_duration'] for m in patients_with_templates])
        avg_times = np.linspace(0, median_duration * 1000, median_length)
        
        fig_avg, ax_avg = plt.subplots(figsize=(10, 5))
        ax_avg.plot(avg_times, avg_template, color='blue', linewidth=3, label=f'Average (n={len(aligned_templates)})')
        ax_avg.fill_between(avg_times, avg_template - std_template, avg_template + std_template,
                           color='blue', alpha=0.2, label='± 1 SD')
        
        ax_avg.set_title(f"Average Repetitive Pattern - {selected_group} {selected_stage}", 
                        fontsize=14, fontweight='bold')
        ax_avg.set_xlabel('Time (ms)', fontsize=11)
        ax_avg.set_ylabel('Amplitude (μV)', fontsize=11)
        ax_avg.grid(True, alpha=0.3)
    ax_avg.legend()
    st.pyplot(fig_avg, use_container_width=True)

    # --- Cleaning Methods Comparison ---
    st.markdown("### Cleaning Methods Comparison")
    st.markdown("Visualizing the effect of different artifact removal methods on the selected patient's signal.")
    
    if 'm' in locals() and len(m['template']) > 0:
        # Use the same patient selected in "Detailed Pattern View"
        
        # 1. Define Cleaning Functions locally or call them if defined outside
        # Defining them here for clarity in context, or better move to global scope. 
        # I'll define them here to access 'm' easily if needed, but standard practice is outside.
        # Let's call the helper functions defined above (we will add them to the file scope in a moment).
        # Since I can't add them to file scope in this single Replace block easily without replacing the whole file,
        # I will define them as nested functions or just implement the logic inline for this block.
        # Actually, I can add the helper functions *before* this function in a previous edit, or just include them here.
        # Given the constraint of one Replace block, I will include the logic here.
        
        # ... Wait, I should add the helper functions properly at module level. 
        # But this tool call is for `replace_file_content`. I can only do one contiguous block.
        # I will implement the logic inside `handle_ecg_noise_detection` or helper functions *inside* this block 
        # if I want to do it in one go. However, to keep code clean, I should probably use `multi_replace` 
        # if I want to insert functions elsewhere. 
        # BUT, for now, let's implement the cleaning logic directly here for the specific signal view.
        
        signal_seg = m['signal_segment']
        template = m['template']
        match_locs = m['match_locations']
        template_len = len(template)
        
        # Method 1: Global Template Subtraction
        clean_global = signal_seg.copy()
        for start_idx in match_locs:
            if start_idx + template_len <= len(clean_global):
                clean_global[start_idx:start_idx+template_len] -= template
                
        # Method 2: Scaled Template Subtraction
        clean_scaled = signal_seg.copy()
        template_norm_sq = np.sum(template**2)
        if template_norm_sq > 0:
            for start_idx in match_locs:
                if start_idx + template_len <= len(clean_scaled):
                    segment = clean_scaled[start_idx:start_idx+template_len]
                    scale = np.dot(segment, template) / template_norm_sq
                    # Limit scale to reasonable bounds to avoid subtracting noise that looks like template
                    # scale = np.clip(scale, 0.5, 2.0) # Optional
                    clean_scaled[start_idx:start_idx+template_len] -= scale * template
        
        # Method 3: Local Moving Average
        clean_local = signal_seg.copy()
        window_size = 10
        n_matches = len(match_locs)
        
        for i, start_idx in enumerate(match_locs):
            # Find local window indices
            start_w = max(0, i - window_size // 2)
            end_w = min(n_matches, start_w + window_size)
            if end_w - start_w < window_size and start_w > 0:
                start_w = max(0, end_w - window_size)
            
            local_indices = match_locs[start_w:end_w]
            
            # Compute local template
            local_segments = []
            for loc in local_indices:
                if loc + template_len <= len(signal_seg):
                    local_segments.append(signal_seg[loc:loc+template_len])
            
            if local_segments:
                local_temp = np.mean(local_segments, axis=0)
                if start_idx + template_len <= len(clean_local):
                    clean_local[start_idx:start_idx+template_len] -= local_temp

        # --- Plotting ---
        # Reuse the zoomed window from earlier
        # start_sample, end_sample, times_view defined above in "Signal with Pattern Locations Marked"
        
        # We need to slice the cleaned signals to match the view window
        # Ensure we have the view variables
        if 'start_sample' in locals() and 'end_sample' in locals() and 'times_view' in locals():
            sig_view_raw = signal_seg[start_sample:end_sample]
            sig_view_global = clean_global[start_sample:end_sample]
            sig_view_scaled = clean_scaled[start_sample:end_sample]
            sig_view_local = clean_local[start_sample:end_sample]
            
            fig_clean, axes_clean = plt.subplots(4, 1, figsize=(15, 10), sharex=True, sharey=True)
            
            # 1. Original
            axes_clean[0].plot(times_view, sig_view_raw, color='black', alpha=0.8, linewidth=1)
            axes_clean[0].set_title("Original Signal (Red lines = detected patterns)", fontsize=10, fontweight='bold')
            # Add markers
            for match_loc in match_locs:
                match_time = match_loc / m['sfreq']
                if times_view[0] <= match_time <= times_view[-1]:
                    axes_clean[0].axvline(match_time, color='red', linestyle='--', alpha=0.5)
            
            # 2. Global
            axes_clean[1].plot(times_view, sig_view_global, color='#1f77b4', linewidth=1)
            axes_clean[1].set_title("Method 1: Global Template Subtraction", fontsize=10, fontweight='bold')
            
            # 3. Scaled
            axes_clean[2].plot(times_view, sig_view_scaled, color='#ff7f0e', linewidth=1)
            axes_clean[2].set_title("Method 2: Scaled Template Subtraction (Best Fit)", fontsize=10, fontweight='bold')
            
            # 4. Local
            axes_clean[3].plot(times_view, sig_view_local, color='#2ca02c', linewidth=1)
            axes_clean[3].set_title(f"Method 3: Local Moving Average (Window={window_size})", fontsize=10, fontweight='bold')
            
            for ax in axes_clean:
                ax.grid(True, alpha=0.3)
                ax.set_ylabel("Amplitude (μV)")
            
            axes_clean[-1].set_xlabel("Time (s)")
            
            plt.tight_layout()
            st.pyplot(fig_clean, use_container_width=True)
        else:
            st.warning("Please interact with the Signal View slider above to initialize the plot.")

    st.markdown("""
    **Interpretation Guide:**
    - **Template**: The extracted waveform pattern that repeats throughout the recording
    - **Repetition Rate**: How many times per second the pattern appears
    - **Correlation**: How similar each repetition is to the template (1.0 = perfect match)
    - **Strong Pattern**: Pattern repeats ≥5 times with correlation ≥0.75
    - **Red vertical lines**: Locations where the repetitive pattern was detected
    """)

def handle_single_patient_view(individuals, selected_group, selected_stage, base_path):
    """
    Handles the selection and plotting of all channels for a single patient,
    plus a raw ECG viewer with time controls.
    """
    patient_ids = [ind[0] for ind in individuals]
    selected_pid = st.selectbox("Select Patient for All Channels", patient_ids)
    
    # --- Raw ECG Viewer ---
    st.markdown("### Raw ECG Viewer")
    
    # Locate and load the file for the selected patient
    # We need to find the file that starts with selected_pid in the group folder
    group_dir = os.path.join(base_path, selected_group, selected_stage)
    ecg_raw_data = None
    ecg_sfreq = None
    
    if os.path.exists(group_dir):
        # Find file starting with patient_id
        # Note: ind[0] might be just the base ID, but filenames might be longer.
        # Let's assume filenames start with the ID.
        potential_files = [f for f in os.listdir(group_dir) if f.startswith(selected_pid) and f.endswith('.pkl')]
        if potential_files:
            file_path = os.path.join(group_dir, potential_files[0])
            try:
                with open(file_path, 'rb') as f:
                    raw_obj = pickle.load(f)
                    
                ecg_sfreq = raw_obj.info['sfreq']
                ch_names = raw_obj.ch_names
                ch_lower = [ch.lower() for ch in ch_names]
                ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
                
                if ecg_indices:
                    ecg_idx = ecg_indices[0]
                    # Get all data for this channel
                    ecg_raw_data = raw_obj.get_data(picks=[ecg_idx])[0] * 1e6 # Convert to uV
                else:
                    st.warning("No ECG channel found in the raw file.")
                    
            except Exception as e:
                st.error(f"Error loading raw file for ECG viewer: {e}")
        else:
             st.warning(f"Could not find source file for {selected_pid} to display raw ECG.")
             
    if ecg_raw_data is not None and ecg_sfreq is not None:
        duration_sec = len(ecg_raw_data) / ecg_sfreq
        
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.number_input("Start Time (s)", min_value=0.0, max_value=duration_sec, value=0.0, step=1.0)
        with col2:
            view_duration = st.slider("View Duration (s)", min_value=1, max_value=60, value=10)
            
        # Plot
        start_sample = int(start_time * ecg_sfreq)
        end_sample = min(len(ecg_raw_data), start_sample + int(view_duration * ecg_sfreq))
        
        # Add plot mode selector
        plot_mode = st.radio("ECG Plot Mode", ["Overlay", "Separate Subplots"], horizontal=True)

        segment = ecg_raw_data[start_sample:end_sample]
        segment_times = np.linspace(start_time, start_time + len(segment)/ecg_sfreq, len(segment))
        
        # Buffer and Clean logic
        clean_segment = None
        # Initialize fixed_segment to match raw segment length initially (or None)
        fixed_segment = None 
        
        try:
             # Add buffer
            buffer_samples = int(1.0 * ecg_sfreq) # 1 sec buffer
            start_buf = max(0, start_sample - buffer_samples)
            end_buf = min(len(ecg_raw_data), end_sample + buffer_samples)
            
            segment_buf = ecg_raw_data[start_buf:end_buf]
            
            # 1. Fix Inverted
            fixed_segment_buf, fit_info = fix_inverted_ecg(segment_buf, fs=ecg_sfreq)

            # 2. Clean based on fixed signal
            clean_segment_buf, _ = clean_ecg_high_fidelity(fixed_segment_buf, sampling_rate=ecg_sfreq)
            
            # Crop back to original window
            crop_start = start_sample - start_buf
            crop_end = crop_start + (end_sample - start_sample)
            
            clean_segment = clean_segment_buf[crop_start:crop_end]
            fixed_segment = fixed_segment_buf[crop_start:crop_end]
            
        except Exception as e:
            st.warning(f"Could not clean ECG segment: {e}")

        # --- FFT Frequency Spectrum ---
        st.markdown("#### FFT Frequency Spectrum")
        try:
            from scipy.fft import rfft, rfftfreq

            nyquist = ecg_sfreq / 2
            fft_xlim = st.slider(
                "FFT Frequency Range (Hz)",
                min_value=0.0,
                max_value=float(min(500, nyquist)),
                value=(0.0, 10.0),
                step=0.5,
                key="fft_xlim_slider",
            )

            n = len(segment)
            freqs = rfftfreq(n, d=1.0 / ecg_sfreq)

            # Raw FFT
            fft_raw = np.abs(rfft(segment)) / n * 2  # one-sided amplitude

            fig_fft, ax_fft = plt.subplots(figsize=(12, 4))
            ax_fft.semilogy(freqs, fft_raw, color='black', linewidth=1, alpha=0.8, label='Raw ECG')

            if clean_segment is not None and len(clean_segment) == len(segment):
                fft_clean = np.abs(rfft(clean_segment)) / n * 2
                ax_fft.semilogy(freqs, fft_clean, color='red', linewidth=1, alpha=0.8, label='Cleaned ECG')

            ax_fft.set_xlim(fft_xlim[0], fft_xlim[1])
            ax_fft.set_title(f"FFT Spectrum - {selected_pid} (window: {start_time:.1f}–{start_time + view_duration:.1f} s)")
            ax_fft.set_xlabel("Frequency (Hz)")
            ax_fft.set_ylabel("Amplitude (μV, log scale)")
            ax_fft.legend(loc='upper right')
            ax_fft.grid(True, which='both', alpha=0.3)
            st.pyplot(fig_fft, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute FFT: {e}")

        # Plotting based on mode
        if plot_mode == "Overlay":
            fig_ecg_raw, ax_ecg = plt.subplots(figsize=(12, 4))
            ax_ecg.plot(segment_times, segment, color='black', linewidth=1, label='Raw ECG', alpha=0.5)
            
            if fixed_segment is not None and len(fixed_segment) == len(segment):
                 # Plot fixed signal if valid
                 ax_ecg.plot(segment_times, fixed_segment, color='blue', linestyle='--', linewidth=1, alpha=0.7, label='Inverted (Fixed)')

            if clean_segment is not None and len(clean_segment) == len(segment):
                 ax_ecg.plot(segment_times, clean_segment, color='red', linewidth=1, alpha=0.8, label='Cleaned ECG')
            
            ax_ecg.set_title(f"Raw vs Fixed vs Cleaned ECG - {selected_pid} (Start: {start_time}s)")
            ax_ecg.set_xlabel("Time (s)")
            ax_ecg.set_ylabel("Amplitude (μV)")
            ax_ecg.grid(True, alpha=0.3)
            ax_ecg.legend(loc='upper right')
            st.pyplot(fig_ecg_raw, use_container_width=True)

            # --- Cleaned ECG with WFDB Robust R-peaks ---
            if clean_segment is not None:
                st.markdown("#### Cleaned ECG with WFDB Robust R-peaks")
                try:
                    # Detect R-peaks on the cleaned segment
                    rpeaks_loc = detect_rpeaks_robust(clean_segment, ecg_sfreq)
                    
                    fig_rpeaks, ax_rpeaks = plt.subplots(figsize=(12, 4))
                    ax_rpeaks.plot(segment_times, clean_segment, color='red', linewidth=1, label='Cleaned ECG')
                    
                    if len(rpeaks_loc) > 0:
                        # rpeaks_loc are indices relative to start of clean_segment
                        rpeak_times = segment_times[rpeaks_loc]
                        rpeak_amps = clean_segment[rpeaks_loc]
                        
                        ax_rpeaks.plot(rpeak_times, rpeak_amps, 'bo', label='WFDB Robust R-peaks')
                        
                        for rt in rpeak_times:
                             ax_rpeaks.axvline(rt, color='blue', linestyle='--', alpha=0.3)
                             
                        # Highlight exactly 500ms intervals
                        if len(rpeak_times) > 1:
                            rpeak_diffs = np.diff(rpeak_times)
                            # Using 3 decimal points for 500ms (0.500s) precision
                            exact_500_mask = np.round(rpeak_diffs, 3) == 0.500
                            
                            # Find all peaks that are part of a 500ms interval
                            peak_is_500 = np.zeros(len(rpeak_times), dtype=bool)
                            peak_is_500[:-1] |= exact_500_mask
                            peak_is_500[1:] |= exact_500_mask
                            
                            if np.any(peak_is_500):
                                ax_rpeaks.plot(rpeak_times[peak_is_500], rpeak_amps[peak_is_500], 
                                               'ro', markersize=12, fillstyle='none', markeredgewidth=2, 
                                               label='Exactly 500ms Apart')
                                               
                        # Calculate Median, MAD and BPM for the title
                        med = np.median(clean_segment)
                        mad = np.median(np.abs(clean_segment - med))
                        if len(rpeak_times) > 1:
                            avg_rr = np.mean(np.diff(rpeak_times))
                            bpm = 60.0 / avg_rr if avg_rr > 0 else 0
                            title_str = f"Cleaned ECG with WFDB Robust R-peaks - {selected_pid} (Median: {med:.2f}μV, MAD: {mad:.2f}μV, ~{bpm:.0f} BPM)"
                        else:
                            title_str = f"Cleaned ECG with WFDB Robust R-peaks - {selected_pid} (Median: {med:.2f}μV, MAD: {mad:.2f}μV)"
                    else:
                        st.info("No R-peaks detected by WFDB in this segment.")
                        if clean_segment is not None:
                            med = np.median(clean_segment)
                            mad = np.median(np.abs(clean_segment - med))
                        else:
                            med, mad = 0, 0
                        title_str = f"Cleaned ECG with WFDB Robust R-peaks - {selected_pid} (Median: {med:.2f}μV, MAD: {mad:.2f}μV)"
                        
                    ax_rpeaks.set_title(title_str)
                    ax_rpeaks.set_xlabel("Time (s)")
                    ax_rpeaks.set_ylabel("Amplitude (μV)")
                    ax_rpeaks.grid(True, alpha=0.3)
                    ax_rpeaks.legend(loc='upper right')
                    st.pyplot(fig_rpeaks, use_container_width=True)
                except Exception as e:
                    st.warning(f"Error detecting R-peaks with WFDB: {e}")


        else: # Separate Subplots
            fig_ecg_raw, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
            
            # Raw
            axes[0].plot(segment_times, segment, color='black', linewidth=1)
            axes[0].set_title("Raw ECG")
            axes[0].set_ylabel("Amp (μV)")
            axes[0].grid(True, alpha=0.3)
            
            # Fixed
            if fixed_segment is not None and len(fixed_segment) == len(segment):
                axes[1].plot(segment_times, fixed_segment, color='blue', linewidth=1)
            else:
                axes[1].text(0.5, 0.5, "Processing Failed", ha='center', va='center')
            axes[1].set_title("Inverted (Fixed) ECG")
            axes[1].set_ylabel("Amp (μV)")
            axes[1].grid(True, alpha=0.3)

            # Cleaned
            if clean_segment is not None and len(clean_segment) == len(segment):
                axes[2].plot(segment_times, clean_segment, color='red', linewidth=1)
            else:
                axes[2].text(0.5, 0.5, "Cleaning Failed", ha='center', va='center')
                
            axes[2].set_title("Cleaned ECG")
            axes[2].set_ylabel("Amp (μV)")
            axes[2].set_xlabel("Time (s)")
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_ecg_raw, use_container_width=True)

        # --- ECG Cleaning Comparison ---
        st.markdown(f"### ECG Cleaning Methods Comparison ({selected_pid})")
        if st.checkbox("Show ECG Cleaning Comparison", value=False):
            if clean_segment is not None:
                # Plot the comparison using the pre-cleaned segment and raw segment
                plot_ecg_cleaning_comparison(segment, clean_segment, ecg_sfreq, selected_pid)
            else:
                 st.warning(f"Could not load source file for {selected_pid} to display cleaning comparison.")
                 
        # --- Raw EEG Viewer ---
        st.markdown(f"**Raw EEG Viewer ({selected_pid})**")
        # Filter for EEG channels only (exclude ECG, EOG, etc.)
        eeg_channels = [ch for ch in ch_names if not any(x in ch.lower() for x in ['ecg', 'ekg', 'eog', 'emg', 'resp'])]
        selected_eeg_ch = st.selectbox("Select EEG Electrode", eeg_channels)
        
        if selected_eeg_ch:
             try:
                 eeg_ch_idx = ch_names.index(selected_eeg_ch)
                 eeg_raw_data = raw_obj.get_data(picks=[eeg_ch_idx])[0] * 1e6
                 
                 eeg_segment = eeg_raw_data[start_sample:end_sample]
                 
                 fig_eeg_raw, ax_eeg_raw = plt.subplots(figsize=(12, 4))
                 ax_eeg_raw.plot(segment_times, eeg_segment, color='blue', linewidth=1)
                 ax_eeg_raw.set_title(f"Raw EEG - {selected_eeg_ch} (Time Aligned)")
                 ax_eeg_raw.set_xlabel("Time (s)")
                 ax_eeg_raw.set_ylabel("Amplitude (μV)")
                 ax_eeg_raw.grid(True, alpha=0.3)
                 st.pyplot(fig_eeg_raw, use_container_width=True)
                 
             except Exception as e:
                 st.error(f"Error extracting EEG channel {selected_eeg_ch}: {e}")



        # --- HRV Analysis (RR Intervals) ---
        st.markdown(f"**HRV Analysis (RR Intervals) - {selected_pid}**")
        
        # We need to get the rpeaks for this patient.
        # Check individuals list for the selected patient
        rpeaks_current = None
        for ind_chk in individuals:
            if ind_chk[0] == selected_pid:
                # ind structure: (patient_id, hep_data, times, ch_names, rpeaks, ecg_hep_data, ecg_ch_names)
                rpeaks_current = ind_chk[4]
                break
        
        if rpeaks_current is not None and len(rpeaks_current) > 1:
            # Calculate RR intervals in ms
            # fs is needed. Use ecg_sfreq from raw file above, or assume it's same for group.
            if ecg_sfreq is None:
                 pass

            if ecg_sfreq:
                rr_intervals_samples = np.diff(rpeaks_current)
                rr_intervals_ms = (rr_intervals_samples / ecg_sfreq) * 1000
                
                # Filter outliers (550ms to 1300ms)
                valid_rr = rr_intervals_ms[(rr_intervals_ms >= 550) & (rr_intervals_ms <= 1300)]
                n_excluded = len(rr_intervals_ms) - len(valid_rr)
                
                if len(valid_rr) > 0:
                    rr_mean = np.mean(valid_rr)
                    rr_std = np.std(valid_rr)
                    rr_median = np.median(valid_rr)
                    hr_mean = 60000 / rr_mean
                    
                    st.write(f"**Mean RR:** {rr_mean:.1f} ms | **Median RR:** {rr_median:.1f} ms | **SDNN:** {rr_std:.1f} ms | **Mean HR:** {hr_mean:.1f} bpm")
                    if n_excluded > 0:
                        st.caption(f"(Excluded {n_excluded} intervals outside 550-1300ms range)")
                    
                    fig_hrv, ax_hrv = plt.subplots(figsize=(10, 4))
                    ax_hrv.hist(valid_rr, bins=30, color='purple', alpha=0.7, edgecolor='black')
                    ax_hrv.set_title(f"RR Interval Distribution - {selected_pid} (550-1300ms) - {hr_mean:.1f} BPM")
                    ax_hrv.set_xlabel("RR Interval (ms)")
                    ax_hrv.set_ylabel("Count")
                    ax_hrv.set_xlim(550, 1300)
                    ax_hrv.grid(True, alpha=0.3)
                    
                    # Add vertical line for mean
                    ax_hrv.axvline(rr_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {rr_mean:.1f}ms')
                    ax_hrv.legend()
                    
                    st.pyplot(fig_hrv, use_container_width=True)
                else:
                    st.warning("No valid RR intervals found (all outside 550-1300ms).")
            else:
                 st.warning("Sampling rate unknown (file load failed), cannot calculate RR intervals in ms.")
        else:
            st.warning("Not enough R-peaks to calculate HRV.")

    for ind in individuals:
        if ind[0] == selected_pid:
            # Determine grid size
            ch_names = ind[3]
            n_channels = len(ch_names)
            n_cols = 5
            n_rows = int(np.ceil(n_channels / n_cols))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), sharex=True, sharey=True)
            axes = axes.flatten()
            
            times = ind[2]
            full_hep = ind[1] # (n_ch, n_times)
            
            # Get ECG HEP data if available
            ecg_hep = ind[5] # (1, n_times)
            
            # remove LOC ROC channels
            full_hep = full_hep[:-2]
            ch_names = ch_names[:-2]
            # remove ECG channels
            non_ecg_mask = [i for i, ch in enumerate(ch_names) if not any(x in ch.lower() for x in ['ecg', 'ekg'])]
            full_hep = full_hep[non_ecg_mask]
            ch_names = [ch_names[i] for i in non_ecg_mask]
            # Determine symmetric limits for alignment
            # EEG Limits
            max_eeg = np.nanmax(np.abs(full_hep * 1e6)) * 1.1 # 10% margin
            ylim_eeg = (-max_eeg, max_eeg)

            # ECG Limits
            if ecg_hep is not None:
                max_ecg = np.nanmax(np.abs(ecg_hep[0] * 1e6)) * 1.1 # 10% margin
                ylim_ecg = (-max_ecg, max_ecg)
            
            for i, (ch_name, ch_data) in enumerate(zip(ch_names, full_hep)):
                ax = axes[i]
                ax.plot(times, ch_data * 1e6)
                ch_skew = stats.skew(ch_data)
                ax.set_title(f"{ch_name}\nskew={ch_skew:.2f}")
                ax.grid(True)
                ax.axvline(0, color='r', linestyle='--', alpha=0.5)
                ax.set_ylim(ylim_eeg)
                
                # Add ECG HEP on secondary axis
                if ecg_hep is not None:
                    ax2 = ax.twinx()
                    # ecg_hep is (1, n_times), so take [0]
                    ax2.plot(times, ecg_hep[0] * 1e6, color='gray', linestyle='--', alpha=0.7,linewidth=1)
                    ax2.set_ylim(ylim_ecg)
                    
                    # Only show right y-axis labels for the rightmost column to reduce clutter
                    if (i + 1) % n_cols == 0:
                        ax2.set_ylabel("ECG (μV)", color='gray', fontsize=8)
                    else:
                        ax2.set_yticklabels([])
                    
                    ax2.tick_params(axis='y', labelcolor='gray', labelsize=8)

                if i % n_cols == 0:
                    ax.set_ylabel("Amp (μV)")
                if i >= n_channels - n_cols:
                    ax.set_xlabel("Time (s)")
                    
            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
            
            st.divider()
            st.title(f"All Channels for {selected_pid} - {selected_group} {selected_stage}")
            fig.suptitle(f"All Channels - {selected_pid} - {selected_group} {selected_stage}", fontsize=16)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)



            # --- Butterfly Plot (Z-Scored) ---
            st.subheader(f"All Channels Z-Scored ({selected_pid})")
            fig_z, ax_z = plt.subplots(figsize=(14, 6)) # Increased width for legend

            # Compute Common Average Reference (CAR)
            # Use only EEG channels for CAR (exclude potential artifact channels)
            # Filter: strictly exclude ECG/EOG/EMG and require digit or 'z' suffix
            eeg_indices_car = [
                idx for idx, name in enumerate(ch_names)
                if not any(excl in name.upper() for excl in ['ECG', 'EKG', 'EOG', 'EMG', 'RESP', 'PULSE'])
                and (name[-1].isdigit() or name.lower().endswith('z'))
            ]
            
            if eeg_indices_car:
                car_signal = np.mean(full_hep[eeg_indices_car], axis=0)
            else:
                # Fallback if no specific EEG channels found
                car_signal = np.mean(full_hep, axis=0)

            for i, (ch_name, ch_data) in enumerate(zip(ch_names, full_hep)):
                # Filter: last char must be digit
                if not ch_name[-1].isdigit():
                    continue
                    
                last_digit = int(ch_name[-1])
                
                # Color logic: Even=Blue, Odd=Red
                if last_digit % 2 == 0:
                    color = 'blue'
                else:
                    color = 'red'

                # Subtract CAR
                ch_data_car = ch_data - car_signal

                # Calculate Z-score per channel on CAR-subtracted data
                if np.std(ch_data_car) > 0:
                    z_data = (ch_data_car - np.mean(ch_data_car)) / np.std(ch_data_car)
                else:
                    z_data = np.zeros_like(ch_data_car)
                    
                ax_z.plot(times, z_data, color=color, alpha=0.3, linewidth=1, label=ch_name)
            
            # Plot CAR signal (Z-scored)
            if np.std(car_signal) > 0:
                z_car = (car_signal - np.mean(car_signal)) / np.std(car_signal)
                ax_z.plot(times, z_car, color='black', alpha=0.8, linewidth=2, linestyle='--', label='CAR (EEG only)')

            ax_z.set_title(f"Butterfly Plot - Z-Scored (CAR Subtracted, Even=Blue, Odd=Red) - {selected_pid}")
            ax_z.set_xlabel("Time (s)")
            ax_z.set_ylabel("Z-Score")
            ax_z.axvline(0, color='r', linestyle='--', alpha=0.8)
            ax_z.grid(True, alpha=0.3)
            
            # Add legend outside
            ax_z.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0., fontsize='small', ncol=1)
            
            st.pyplot(fig_z, use_container_width=True)

            # --- Butterfly Plot (Raw uV) with ECG ---
            st.subheader(f"Butterfly Plot - Raw uV ({selected_pid})")
            fig_raw, ax_raw = plt.subplots(figsize=(14, 6))

            # Plot EEG channels
            for i, (ch_name, ch_data) in enumerate(zip(ch_names, full_hep)):
                 # Filter: last char must be digit
                if not ch_name[-1].isdigit():
                    continue
                last_digit = int(ch_name[-1])
                color = 'blue' if last_digit % 2 == 0 else 'red'
                
                ax_raw.plot(times, ch_data * 1e6, color=color, alpha=0.3, linewidth=1)

            ax_raw.set_xlabel("Time (s)")
            ax_raw.set_ylabel("EEG Amplitude (μV)")
            ax_raw.axvline(0, color='r', linestyle='--', alpha=0.8)
            ax_raw.grid(True, alpha=0.3)

            # Secondary Axis for ECG
            if ecg_hep is not None:
                ax_ecg = ax_raw.twinx()
                ax_ecg.plot(times, ecg_hep[0] * 1e6, color='green', alpha=0.6, linewidth=2, linestyle='--', label='ECG')
                ax_ecg.set_ylabel("ECG Amplitude (μV)", color='green')
                ax_ecg.tick_params(axis='y', labelcolor='green')
                # Add legend for ECG
                ax_ecg.legend(loc='upper right')

            ax_raw.set_title(f"Butterfly Plot - Raw uV (Even=Blue, Odd=Red) - {selected_pid}")
            
            # Add legend for Even/Odd EEG (optional, purely visual)
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color='blue', lw=2),
                            Line2D([0], [0], color='red', lw=2)]
            ax_raw.legend(custom_lines, ['Even Channels', 'Odd Channels'], loc='upper left')

            st.pyplot(fig_raw, use_container_width=True)

            # --- Butterfly Plot - All Channels Colored by Region ---
            st.subheader(f"Butterfly Plot - Region Colored ({selected_pid})")
            fig_reg_all, ax_reg_all = plt.subplots(figsize=(14, 6))

            region_color_map = {
                'F': 'blue',
                'C': 'green',
                'T': 'red',
                'P': 'purple',
                'O': 'orange'
            }
            
            # Helper to determine region
            def get_region_color(ch_name):
                name_upper = ch_name.upper()
                # Check regions in specific order to handle overlaps if any (e.g. FC -> Frontal-Central)
                # Typically F first checks F, FC, Fp. C checks C, CP.
                # Simplest heuristic: check letter presence.
                for reg, col in region_color_map.items():
                   if reg in name_upper:
                       return col, reg
                return 'gray', 'Other'

            # Plot EEG channels
            used_regions = set()
            for i, (ch_name, ch_data) in enumerate(zip(ch_names, full_hep)):
                color, region = get_region_color(ch_name)
                ax_reg_all.plot(times, ch_data * 1e6, color=color, alpha=0.3, linewidth=1)
                used_regions.add(region)

            ax_reg_all.set_xlabel("Time (s)")
            ax_reg_all.set_ylabel("EEG Amplitude (μV)")
            ax_reg_all.axvline(0, color='r', linestyle='--', alpha=0.8)
            ax_reg_all.grid(True, alpha=0.3)

            # Secondary Axis for ECG
            if ecg_hep is not None:
                ax_ecg_reg = ax_reg_all.twinx()
                ax_ecg_reg.plot(times, ecg_hep[0] * 1e6, color='black', alpha=0.6, linewidth=2, linestyle='--', label='ECG')
                ax_ecg_reg.set_ylabel("ECG Amplitude (μV)", color='black')
                ax_ecg_reg.tick_params(axis='y', labelcolor='black')
                ax_ecg_reg.legend(loc='upper right')

            ax_reg_all.set_title(f"Butterfly Plot - Region Colored - {selected_pid}")
            
            # Custom legend for regions
            custom_lines_reg = [Line2D([0], [0], color=region_color_map[reg], lw=2) for reg in region_color_map if reg in used_regions]
            custom_labels_reg = [f"Region {reg}" for reg in region_color_map if reg in used_regions]
            if 'Other' in used_regions:
                custom_lines_reg.append(Line2D([0], [0], color='gray', lw=2))
                custom_labels_reg.append("Region Other")
            
            ax_reg_all.legend(custom_lines_reg, custom_labels_reg, loc='upper left')

            st.pyplot(fig_reg_all, use_container_width=True)

            # --- Butterfly Plot - Odd Channels (Region Colored) ---
            st.subheader(f"Butterfly Plot - Odd Channels ({selected_pid})")
            fig_odd, ax_odd = plt.subplots(figsize=(14, 6))
            
            used_regions_odd = set()
            has_odd = False
            for i, (ch_name, ch_data) in enumerate(zip(ch_names, full_hep)):
                if not ch_name[-1].isdigit():
                    continue
                if int(ch_name[-1]) % 2 == 0:
                    continue # Skip even

                color, region = get_region_color(ch_name)
                ax_odd.plot(times, ch_data * 1e6, color=color, alpha=0.3, linewidth=1)
                used_regions_odd.add(region)
                has_odd = True

            if has_odd:
                ax_odd.set_xlabel("Time (s)")
                ax_odd.set_ylabel("EEG Amplitude (μV)")
                ax_odd.axvline(0, color='r', linestyle='--', alpha=0.8)
                ax_odd.grid(True, alpha=0.3)

                if ecg_hep is not None:
                    ax_ecg_odd = ax_odd.twinx()
                    ax_ecg_odd.plot(times, ecg_hep[0] * 1e6, color='black', alpha=0.6, linewidth=2, linestyle='--', label='ECG')
                    ax_ecg_odd.set_ylabel("ECG Amplitude (μV)", color='black')
                    ax_ecg_odd.tick_params(axis='y', labelcolor='black')
                    ax_ecg_odd.legend(loc='upper right')

                ax_odd.set_title(f"Butterfly Plot - Odd Channels - {selected_pid}")
                
                 # Legend
                custom_lines_odd = [Line2D([0], [0], color=region_color_map[reg], lw=2) for reg in region_color_map if reg in used_regions_odd]
                custom_labels_odd = [f"Region {reg}" for reg in region_color_map if reg in used_regions_odd]
                if 'Other' in used_regions_odd:
                    custom_lines_odd.append(Line2D([0], [0], color='gray', lw=2))
                    custom_labels_odd.append("Region Other")
                
                ax_odd.legend(custom_lines_odd, custom_labels_odd, loc='upper left')
                
                st.pyplot(fig_odd, use_container_width=True)
            else:
                plt.close(fig_odd)

            # --- Butterfly Plot - Even Channels (Region Colored) ---
            st.subheader(f"Butterfly Plot - Even Channels ({selected_pid})")
            fig_even, ax_even = plt.subplots(figsize=(14, 6))
            
            used_regions_even = set()
            has_even = False
            for i, (ch_name, ch_data) in enumerate(zip(ch_names, full_hep)):
                if not ch_name[-1].isdigit():
                    continue
                if int(ch_name[-1]) % 2 != 0:
                    continue # Skip odd

                color, region = get_region_color(ch_name)
                ax_even.plot(times, ch_data * 1e6, color=color, alpha=0.3, linewidth=1)
                used_regions_even.add(region)
                has_even = True

            if has_even:
                ax_even.set_xlabel("Time (s)")
                ax_even.set_ylabel("EEG Amplitude (μV)")
                ax_even.axvline(0, color='r', linestyle='--', alpha=0.8)
                ax_even.grid(True, alpha=0.3)

                if ecg_hep is not None:
                    ax_ecg_even = ax_even.twinx()
                    ax_ecg_even.plot(times, ecg_hep[0] * 1e6, color='black', alpha=0.6, linewidth=2, linestyle='--', label='ECG')
                    ax_ecg_even.set_ylabel("ECG Amplitude (μV)", color='black')
                    ax_ecg_even.tick_params(axis='y', labelcolor='black')
                    ax_ecg_even.legend(loc='upper right')

                ax_even.set_title(f"Butterfly Plot - Even Channels - {selected_pid}")

                # Legend
                custom_lines_even = [Line2D([0], [0], color=region_color_map[reg], lw=2) for reg in region_color_map if reg in used_regions_even]
                custom_labels_even = [f"Region {reg}" for reg in region_color_map if reg in used_regions_even]
                if 'Other' in used_regions_even:
                    custom_lines_even.append(Line2D([0], [0], color='gray', lw=2))
                    custom_labels_even.append("Region Other")
                
                ax_even.legend(custom_lines_even, custom_labels_even, loc='upper left')

                st.pyplot(fig_even, use_container_width=True)
            else:
                plt.close(fig_even)

            # --- Regional Butterfly Plots ---
            regions = ['F', 'T', 'P', 'O']
            for region in regions:
                # Find channels containing the region letter (e.g., 'F' in 'Fp1', 'F3', 'Fz')
                region_indices = [idx for idx, name in enumerate(ch_names) if region in name.upper()]
                
                if not region_indices:
                    continue

                st.subheader(f"Butterfly Plot - Region {region} ({selected_pid})")
                fig_reg, ax_reg = plt.subplots(figsize=(14, 6))
                
                # Setup colormap
                cmap = plt.get_cmap('tab10')
                has_channels = False
                
                # Collect handles for legend
                region_handles = []
                region_labels = []

                for k, idx in enumerate(region_indices):
                    ch_name = ch_names[idx]
                    ch_data = full_hep[idx]
                    
                    # Use a unique color for each channel
                    color = cmap(k % 10)
                    
                    line, = ax_reg.plot(times, ch_data * 1e6, color=color, alpha=0.6, linewidth=1.5, label=ch_name)
                    has_channels = True
                    region_handles.append(line)
                    region_labels.append(ch_name)

                if not has_channels:
                    plt.close(fig_reg)
                    continue

                ax_reg.set_xlabel("Time (s)")
                ax_reg.set_ylabel("EEG Amplitude (μV)")
                ax_reg.axvline(0, color='r', linestyle='--', alpha=0.8)
                ax_reg.grid(True, alpha=0.3)
                
                # Secondary Axis for ECG
                if ecg_hep is not None:
                    ax_ecg_reg = ax_reg.twinx()
                    ecg_line, = ax_ecg_reg.plot(times, ecg_hep[0] * 1e6, color='black', alpha=0.4, linewidth=2, linestyle='--', label='ECG')
                    ax_ecg_reg.set_ylabel("ECG Amplitude (μV)", color='black')
                    ax_ecg_reg.tick_params(axis='y', labelcolor='black')
                    
                    # Add ECG to legend
                    region_handles.append(ecg_line)
                    region_labels.append('ECG')

                ax_reg.set_title(f"Butterfly Plot - Region {region} - {selected_pid}")
                
                # Add legend
                ax_reg.legend(region_handles, region_labels, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
                
                st.pyplot(fig_reg, use_container_width=True)

            # --- Single Electrode Window Explorer ---
            st.subheader(f"Electrode Window Explorer ({selected_pid})")

            eeg_ch_options = [ch for ch in ch_names if not any(x in ch.lower() for x in ['ecg', 'ekg', 'eog', 'emg', 'resp'])]
            col_elec, col_nwin = st.columns(2)
            with col_elec:
                selected_elec = st.selectbox("Select Electrode", eeg_ch_options, key="window_explorer_elec")
            with col_nwin:
                n_windows_show = st.slider("Number of Windows to Show", min_value=1, max_value=100, value=20, key="window_explorer_n")

            raw_obj_available = False
            try:
                _ = raw_obj
                raw_obj_available = True
            except NameError:
                pass

            if raw_obj_available and ecg_sfreq is not None and selected_elec:
                try:
                    rpeaks_ind = ind[4]
                    rpeak_times_ind = rpeaks_ind / ecg_sfreq
                    rpeak_ts_ind = nap.Ts(t=rpeak_times_ind, time_units="s")
                    tmin_win = float(times[0])
                    tmax_win = float(times[-1])

                    raw_ch_names_list = raw_obj.ch_names
                    if selected_elec in raw_ch_names_list:
                        elec_idx = raw_ch_names_list.index(selected_elec)
                        elec_data = raw_obj.get_data(picks=[elec_idx]).T  # (n_times, 1)
                        tsd_elec = nap.TsdFrame(t=raw_obj.times, d=elec_data, columns=[selected_elec])
                        perievent_elec = nap.compute_perievent_continuous(tsd_elec, rpeak_ts_ind, minmax=(tmin_win, tmax_win))

                        pe_values = perievent_elec.values  # (n_times, n_events, 1) or (n_times, n_events)
                        pe_times = perievent_elec.t
                        if pe_values.ndim == 3:
                            pe_values = pe_values[:, :, 0]  # (n_times, n_events)

                        n_trials_total = pe_values.shape[1]
                        n_show = min(n_windows_show, n_trials_total)

                        fig_explorer, ax_explorer = plt.subplots(figsize=(14, 6))

                        for trial_idx in range(n_show):
                            ax_explorer.plot(pe_times, pe_values[:, trial_idx] * 1e6,
                                             color='steelblue', alpha=0.2, linewidth=0.8)

                        avg_signal = np.nanmean(pe_values[:, :n_show], axis=1)
                        ax_explorer.plot(pe_times, avg_signal * 1e6, color='navy', linewidth=2.5,
                                         label=f'Average (n={n_show})')

                        ax_explorer.axvline(0, color='r', linestyle='--', alpha=0.8)
                        ax_explorer.set_xlabel("Time (s)")
                        ax_explorer.set_ylabel("EEG Amplitude (μV)")
                        ax_explorer.set_title(f"Electrode {selected_elec} - {n_show} Windows - {selected_pid}")
                        ax_explorer.legend()
                        ax_explorer.grid(True, alpha=0.3)

                        st.caption(f"Showing {n_show} of {n_trials_total} available windows")
                        st.pyplot(fig_explorer, use_container_width=True)
                        plt.close(fig_explorer)
                    else:
                        st.warning(f"Electrode {selected_elec} not found in raw data.")
                except Exception as e:
                    st.error(f"Error computing per-window data: {e}")
            else:
                st.warning("Raw data not available for window-level analysis.")

            break

def plot_patients_butterfly_comparison(individuals, selected_group, selected_stage):
    """
    Plots a grid of butterfly plots, one method per patient.
    """
    n_pats = len(individuals)
    n_cols = 2
    n_rows = int(np.ceil(n_pats / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows), sharex=True, sharey=True)
    if n_pats == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for idx, ind in enumerate(individuals):
        ax = axes[idx]
        patient_id = ind[0]
        full_hep = ind[1]
        times = ind[2]
        ch_names = ind[3]
        ecg_hep = ind[5]
        
        # Plot EEG channels
        for i, (ch_name, ch_data) in enumerate(zip(ch_names, full_hep)):
            # Filter: last char must be digit
            if not ch_name[-1].isdigit():
                continue
            last_digit = int(ch_name[-1])
            color = 'blue' if last_digit % 2 == 0 else 'red'
            
            ax.plot(times, ch_data * 1e6, color=color, alpha=0.3, linewidth=1)

        ax.grid(True, alpha=0.3)
        ax.set_ylim(-30, 30)
        ax.axvline(0, color='r', linestyle='--', alpha=0.8)
        
        # Secondary Axis for ECG
        if ecg_hep is not None:
            ax_ecg = ax.twinx()
            ax_ecg.plot(times, ecg_hep[0] * 1e6, color='green', alpha=0.6, linewidth=1.5, linestyle='--')
            
            ax_ecg.set_ylim(-150, 150)

            # Only label rightmost plots
            if (idx + 1) % n_cols == 0:
                 ax_ecg.set_ylabel("ECG (μV)", color='green')
            else:
                 ax_ecg.set_yticklabels([])
                 
            ax_ecg.tick_params(axis='y', labelcolor='green')

        ax.set_title(f"{patient_id}")
        
        if idx % n_cols == 0:
            ax.set_ylabel("EEG (μV)")
        if idx >= n_pats - n_cols:
            ax.set_xlabel("Time (s)")

    # Hide unused subplots
    for j in range(len(individuals), len(axes)):
        axes[j].axis('off')
        
    fig.suptitle(f"Patients Comparison (Butterfly EEG + ECG) - {selected_group} {selected_stage}", fontsize=16)
    st.pyplot(fig, use_container_width=True)

def _granger_causality_index(cause, effect, lag=8):
    """
    Compute a normalized Granger causality index (cause → effect).
    Compares RSS of restricted AR model vs full ARX model.
    Returns a value in [0, 1]; higher = stronger directional coupling.
    """
    n = len(effect)
    if n <= lag * 3:
        return 0.0
    Y = effect[lag:]
    X_r = np.column_stack([effect[lag - k - 1: n - k - 1] for k in range(lag)])
    X_f = np.column_stack(
        [effect[lag - k - 1: n - k - 1] for k in range(lag)] +
        [cause[lag - k - 1: n - k - 1] for k in range(lag)]
    )
    try:
        beta_r, _, _, _ = np.linalg.lstsq(X_r, Y, rcond=None)
        rss_r = np.sum((Y - X_r @ beta_r) ** 2)
        beta_f, _, _, _ = np.linalg.lstsq(X_f, Y, rcond=None)
        rss_f = np.sum((Y - X_f @ beta_f) ** 2)
        if rss_r < 1e-12 or rss_f >= rss_r:
            return 0.0
        gc = np.log(rss_r / rss_f)
        return float(np.clip(gc / (gc + 1.0), 0.0, 1.0))
    except Exception:
        return 0.0


def _compute_eeg_band_power_series(eeg_data, sfreq, fs_out=4.0, bands=None, window_sec=8.0):
    """
    Compute EEG band power time series using a sliding Hann-windowed FFT.

    Returns:
        band_powers : dict {band_name: ndarray (n_channels, n_steps)}
        t_power     : ndarray of time points in seconds (relative to start of eeg_data)
    """
    if bands is None:
        bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30)}
    n_ch, n_samples = eeg_data.shape
    window_samples = int(window_sec * sfreq)
    step_samples = max(1, int(sfreq / fs_out))
    if window_samples > n_samples:
        return None, None
    n_steps = (n_samples - window_samples) // step_samples + 1
    freqs = np.fft.rfftfreq(window_samples, d=1.0 / sfreq)
    hann = np.hanning(window_samples)
    band_powers = {b: np.zeros((n_ch, n_steps)) for b in bands}
    for step_i in range(n_steps):
        start = step_i * step_samples
        segment = eeg_data[:, start: start + window_samples]
        fft_power = np.abs(np.fft.rfft(segment * hann, axis=1)) ** 2
        for band, (fmin, fmax) in bands.items():
            mask = (freqs >= fmin) & (freqs < fmax)
            if mask.any():
                band_powers[band][:, step_i] = np.mean(fft_power[:, mask], axis=1)
    t_power = np.arange(n_steps) * step_samples / sfreq
    return band_powers, t_power


def _compute_bhi_patient(raw, rpeaks, sfreq, bands=None, fs_hrv=4.0, lag=8):
    """
    Compute BHI (Brain-Heart Interaction) indexes for one patient using
    a Granger-causality approach analogous to Catrambone et al. (2019).

    Returns dict {'HtB': {band: value}, 'BtH': {band: value}} or None.
    """
    if bands is None:
        bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30)}

    # --- HRV series ---
    rr_intervals = np.diff(rpeaks) / sfreq
    valid_mask = (rr_intervals > 0.3) & (rr_intervals < 2.0)
    rpeaks_trimmed = rpeaks[:-1][valid_mask]
    rr_valid = rr_intervals[valid_mask]
    if len(rr_valid) < 20:
        return None
    t_rpeaks = rpeaks_trimmed / sfreq
    t_hrv = np.arange(t_rpeaks[0], t_rpeaks[-1], 1.0 / fs_hrv)
    hrv_series = np.interp(t_hrv, t_rpeaks, rr_valid)

    # --- EEG channels ---
    ch_names = raw.ch_names
    eeg_idx = [i for i, ch in enumerate(ch_names)
               if re.match(r'^[A-Za-z]{1,3}[0-9]+$', ch) or re.match(r'^[A-Za-z]{1,2}z$', ch, re.IGNORECASE)]
    if not eeg_idx:
        return None
    eeg_data = raw.get_data(picks=eeg_idx)  # (n_channels, n_samples)

    # --- Band power time series ---
    band_powers, t_power = _compute_eeg_band_power_series(eeg_data, sfreq, fs_out=fs_hrv, bands=bands)
    if band_powers is None:
        return None

    # EEG power time axis is relative to start of recording
    t_eeg_abs = rpeaks[0] / sfreq + t_power  # anchor to first R-peak region

    # --- Align common time window ---
    t_start = max(t_hrv[0], t_eeg_abs[0])
    t_end = min(t_hrv[-1], t_eeg_abs[-1])
    if t_end <= t_start:
        return None
    t_common = np.arange(t_start, t_end, 1.0 / fs_hrv)
    if len(t_common) < lag * 3:
        return None

    hrv_aligned = np.interp(t_common, t_hrv, hrv_series)
    hrv_z = (hrv_aligned - np.mean(hrv_aligned)) / (np.std(hrv_aligned) + 1e-10)

    # --- Granger causality per channel per band ---
    results = {'HtB': {}, 'BtH': {}}
    for band in bands:
        htb_vals, bth_vals = [], []
        for ch_i in range(len(eeg_idx)):
            eeg_aligned = np.interp(t_common, t_eeg_abs, band_powers[band][ch_i])
            eeg_z = (eeg_aligned - np.mean(eeg_aligned)) / (np.std(eeg_aligned) + 1e-10)
            htb_vals.append(_granger_causality_index(hrv_z, eeg_z, lag=lag))
            bth_vals.append(_granger_causality_index(eeg_z, hrv_z, lag=lag))
        results['HtB'][band] = float(np.mean(htb_vals))
        results['BtH'][band] = float(np.mean(bth_vals))
    return results


def plot_bhi_analysis(filtered_individuals, selected_group, selected_stage, base_path):
    """
    Loads raw EEG data per patient, computes BHI indexes, and displays group results.

    Brain-Heart Interplay is quantified via a Granger-causality ARX approach
    inspired by Catrambone et al. (2019), Annals of Biomedical Engineering.
    """
    st.subheader(f"Brain-Heart Interplay (BHI) — {selected_group} / {selected_stage}")
    st.markdown(
        "Directional coupling between heartbeat (HRV) and EEG band power, using a "
        "Granger-causality index analogous to the ARX model of "
        "[Catrambone et al. (2019)](https://doi.org/10.1007/s10439-019-02249-y).\n\n"
        "- **HtB** (Heart → Brain): HRV fluctuations predicting EEG band power\n"
        "- **BtH** (Brain → Heart): EEG band power predicting HRV"
    )

    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30)}
    lag_order = st.slider(
        "ARX lag order (samples at 4 Hz)", min_value=2, max_value=16, value=8,
        key="bhi_lag", help="Number of past samples used in the ARX model. 8 = 2 s memory at 4 Hz."
    )

    group_dir = os.path.join(base_path, selected_group, selected_stage)
    patient_results = {}
    progress = st.progress(0, text="Computing BHI...")
    n = len(filtered_individuals)

    for i, ind in enumerate(filtered_individuals):
        pid = ind[0]
        progress.progress((i + 1) / n, text=f"BHI: {pid} ({i + 1}/{n})")
        potential_files = [f for f in os.listdir(group_dir) if f.startswith(pid) and f.endswith('.pkl')]
        if not potential_files:
            continue
        file_path = os.path.join(group_dir, potential_files[0])
        try:
            with open(file_path, 'rb') as fh:
                raw_obj = pickle.load(fh)
            raw_obj = drop_non_eeg_channels(raw_obj)
            result = process_file_data(raw_obj, pid)
            if result is None:
                continue
            raw_obj, sfreq, _rpeak_ts, rpeaks, _minmax, _ = result
            bhi = _compute_bhi_patient(raw_obj, rpeaks, sfreq, bands=bands, lag=lag_order)
            if bhi is not None:
                patient_results[pid] = bhi
        except Exception as e:
            st.warning(f"BHI: skipped {pid} — {e}")

    progress.empty()

    if not patient_results:
        st.error("Could not compute BHI for any patient.")
        return

    band_names = list(bands.keys())
    htb_group = {b: [r['HtB'][b] for r in patient_results.values()] for b in band_names}
    bth_group = {b: [r['BtH'][b] for r in patient_results.values()] for b in band_names}

    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(band_names))
    rng = np.random.default_rng(42)

    for ax, data, direction in [
        (axes[0], htb_group, 'Heart \u2192 Brain (HtB)'),
        (axes[1], bth_group, 'Brain \u2192 Heart (BtH)'),
    ]:
        means = [np.mean(data[b]) for b in band_names]
        sems = [np.std(data[b]) / np.sqrt(max(len(data[b]), 1)) for b in band_names]
        ax.bar(x, means, 0.55, yerr=sems, capsize=5, color=colors, alpha=0.8)
        for j, band in enumerate(band_names):
            ys = data[band]
            xs = np.full(len(ys), j) + rng.uniform(-0.12, 0.12, len(ys))
            ax.scatter(xs, ys, color='k', s=20, alpha=0.6, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(band_names, fontsize=11)
        ax.set_xlabel("EEG Frequency Band")
        ax.set_ylabel("Granger Causality Index")
        ax.set_title(direction, fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)

    fig.suptitle(
        f"BHI — {selected_group} / {selected_stage}  (n={len(patient_results)} patients)",
        fontsize=14
    )
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Summary table
    rows = [
        {
            'Band': b,
            'HtB mean': f"{np.mean(htb_group[b]):.4f}",
            'HtB SEM': f"{np.std(htb_group[b]) / np.sqrt(len(htb_group[b])):.4f}",
            'BtH mean': f"{np.mean(bth_group[b]):.4f}",
            'BtH SEM': f"{np.std(bth_group[b]) / np.sqrt(len(bth_group[b])):.4f}",
        }
        for b in band_names
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def plot_group_ecg_analysis(individuals, selected_group, selected_stage):
    """
    Plots the average ECG of the whole group and individual patient ECGs.
    """
    st.subheader(f"Group ECG Analysis - {selected_group} {selected_stage}")

    # 1. Group Average ECG
    all_ecg_heps = []
    for ind in individuals:
        ecg_hep = ind[5]
        if ecg_hep is not None:
            all_ecg_heps.append(ecg_hep[0])
    
    if all_ecg_heps:
        avg_ecg = np.nanmean(all_ecg_heps, axis=0)
        times = individuals[0][2]
        
        fig_avg, ax_avg = plt.subplots(figsize=(10, 5))
        ax_avg.plot(times, avg_ecg * 1e6, color='black', linewidth=2, label='Group Average ECG')
        
        # Optional: Add standard error shading
        std_ecg = np.nanstd(all_ecg_heps, axis=0)
        sem_ecg = std_ecg / np.sqrt(len(all_ecg_heps))
        ax_avg.fill_between(times, (avg_ecg - sem_ecg) * 1e6, (avg_ecg + sem_ecg) * 1e6, color='gray', alpha=0.3, label='SEM')

        ax_avg.set_title(f"Group Average ECG (n={len(all_ecg_heps)})")
        ax_avg.set_xlabel("Time (s)")
        ax_avg.set_ylabel("Amplitude (μV)")
        ax_avg.axvline(0, color='r', linestyle='--', alpha=0.5)
        ax_avg.legend()
        ax_avg.grid(True, alpha=0.3)
        st.pyplot(fig_avg, use_container_width=True)
    else:
        st.warning("No ECG data found for this group to calculate average.")

    # 2. RR Interval Analysis
    st.markdown("#### RR Interval Analysis")
    
    # Calculate RR intervals for all individuals
    all_rr_intervals = []
    patient_rr_intervals = {} # {pid: rr_intervals}
    
    if individuals:
        # Estimate sampling rate from time vector of first individual
        # times = individuals[0][2]
        # fs = 1 / np.mean(np.diff(times))
        # Better: use the median diff to be robust
        times = individuals[0][2]
        dt = np.median(np.diff(times))
        fs = 1.0 / dt
        
        for ind in individuals:
            pid = ind[0]
            rpeaks = ind[4]
            
            if rpeaks is not None and len(rpeaks) > 1:
                # Calculate RR intervals in seconds
                rr_samples = np.diff(rpeaks)
                rr_sec = rr_samples / fs
                
                # Filter physiological range? (e.g., 0.3s to 2.0s corresponds to 30-200 BPM)
                # For now, let's keep it raw but maybe exclude obvious artifacts if needed
                # valid_rr = rr_sec[(rr_sec > 0.3) & (rr_sec < 2.0)]
                valid_rr = rr_sec 
                
                all_rr_intervals.extend(valid_rr)
                patient_rr_intervals[pid] = valid_rr

    # 2a. Group RR Histogram
    if all_rr_intervals:
        fig_rr_group, ax_rr_group = plt.subplots(figsize=(10, 5))
        ax_rr_group.hist(all_rr_intervals, bins=50, color='purple', alpha=0.7, edgecolor='black')
        
        avg_rr = np.mean(all_rr_intervals)
        std_rr = np.std(all_rr_intervals)
        
        ax_rr_group.set_title(f"Group RR Interval Distribution (n={len(all_rr_intervals)} beats)\nMean={avg_rr:.3f}s ({60/avg_rr:.1f} BPM), SD={std_rr:.3f}s")
        ax_rr_group.set_xlabel("RR Interval (s)")
        ax_rr_group.set_ylabel("Count")
        ax_rr_group.set_xlim(right=1.3)
        ax_rr_group.grid(True, alpha=0.3)
        
        st.pyplot(fig_rr_group, use_container_width=True)
    else:
        st.info("No RR intervals could be calculated.")

    # 2b. Individual RR Histograms
    if patient_rr_intervals:
        st.markdown("#### Individual RR Interval Histograms")
        
        # Use existing slider or adding a new one? Use existing slider for consistency if possible?
        # The existing slider `n_individuals` is already defined below for ECG plots. 
        # But we haven't reached that part of the code yet (it's in original lines 1944+).
        # We are inserting BEFORE line 1944. 
        # So we can define a slider here or reuse the one for ECG plots if we move code.
        # Let's add a separate section for this to be clean.
        
        # Grid layout
        pids = list(patient_rr_intervals.keys())
        n_pats_rr = len(pids)
        
        # Slider to limit number of plots if too many
        n_show_rr = st.slider("Number of patients to show for RR Histograms", 1, n_pats_rr, min(4, n_pats_rr), key="rr_hist_slider")
        
        n_cols_rr = 2
        n_rows_rr = int(np.ceil(n_show_rr / n_cols_rr))
        
        fig_rr_ind, axes_rr_ind = plt.subplots(n_rows_rr, n_cols_rr, figsize=(12, 4 * n_rows_rr))
        if n_show_rr == 1:
            axes_rr_ind = [axes_rr_ind]
        else:
            axes_rr_ind = axes_rr_ind.flatten()
            
        for idx in range(n_show_rr):
            pid = pids[idx]
            rr_data = patient_rr_intervals[pid]
            ax = axes_rr_ind[idx]
            
            ax.hist(rr_data, bins=30, color='teal', alpha=0.6, edgecolor='black')
            
            mean_rr = np.mean(rr_data)
            std_rr = np.std(rr_data)
            
            ax.set_title(f"{pid}\nMean={mean_rr:.3f}s, SD={std_rr:.3f}s")
            ax.set_xlabel("RR (s)")
            ax.set_xlim(right=1.3)
            ax.grid(True, alpha=0.3)
            
        # Hide unused
        for j in range(n_show_rr, len(axes_rr_ind)):
            axes_rr_ind[j].axis('off')
            
        fig_rr_ind.tight_layout()
        st.pyplot(fig_rr_ind, use_container_width=True)


    # 3. Individual Patients
    st.markdown("#### Individual Patient ECGs")
    n_individuals = st.slider("Number of individual patients to show", min_value=1, max_value=len(individuals), value=len(individuals), key="group_ecg_slider")
    
    # Grid layout for individuals
    n_cols = 2
    n_rows = int(np.ceil(n_individuals / n_cols))
    
    fig_ind, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    if n_individuals == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for i in range(n_individuals):
        ind = individuals[i]
        ax = axes[i]
        patient_id = ind[0]
        ecg_hep = ind[5]
        times = ind[2]
        
        if ecg_hep is not None:
             ax.plot(times, ecg_hep[0] * 1e6, color='green', linewidth=1.5)
        else:
             ax.text(0.5, 0.5, "No ECG Data", ha='center', va='center')
             
        ax.set_title(f"{patient_id}")
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='r', linestyle='--', alpha=0.5)
        
        if i % n_cols == 0:
            ax.set_ylabel("ECG (μV)")
        if i >= n_individuals - n_cols:
            ax.set_xlabel("Time (s)")
            
    # Hide unused
    if hasattr(axes, '__len__'):    
        for j in range(n_individuals, len(axes)):
            axes[j].axis('off')
            
    st.pyplot(fig_ind, use_container_width=True)


def handle_ecg_reduction(filtered_individuals, selected_group, selected_stage):
    """
    For each patient: z-score normalises the ECG HEP and each EEG channel HEP independently
    (bringing both to the same amplitude scale), then subtracts the normalised ECG from the
    normalised EEG. Plots per-patient and group-average original, ECG overlay, and residual.
    """
    st.subheader("ECG Signal Reduction (Normalised Subtraction)")
    st.caption(
        "Both the ECG HEP and each EEG channel HEP are z-score normalised independently "
        "so they share the same amplitude scale. "
        "Residual = z-scored EEG − z-scored ECG."
    )

    # Determine common EEG channels
    all_ch_sets = [set(ind[3]) for ind in filtered_individuals]
    counts = Counter([ch for s in all_ch_sets for ch in s])
    common_channels = [
        ch for ch, cnt in counts.items()
        if cnt >= len(filtered_individuals) * 0.5
        and (re.match(r'^[a-zA-Z]{1,2}[0-9]+$', ch) or re.match(r'^[a-zA-Z]z$', ch))
    ]
    if not common_channels:
        st.warning("No common EEG channels found.")
        return

    display_channels = ['Average'] + common_channels

    ch_data = {ch: {'orig_norm': [], 'ecg_norm': [], 'residual': [], 'pids': []} for ch in display_channels}
    times = None

    for ind in filtered_individuals:
        pid, hep_full, t, ch_names, rpeaks, ecg_hep, ecg_ch = ind[:7]
        if ecg_hep is None:
            continue
        ecg_signal = np.asarray(ecg_hep).squeeze()
        if ecg_signal.ndim != 1:
            continue
        if times is None:
            times = t
        if len(ecg_signal) != len(t):
            continue

        ecg_std = ecg_signal.std()
        if ecg_std < 1e-20:
            continue
        ecg_z = (ecg_signal - ecg_signal.mean()) / ecg_std

        valid_indices = []
        for ch in common_channels:
            if ch not in ch_names:
                continue
            idx = ch_names.index(ch)
            eeg_trace = hep_full[idx].copy()
            eeg_std = eeg_trace.std()
            if eeg_std < 1e-20:
                continue
            eeg_z = (eeg_trace - eeg_trace.mean()) / eeg_std
            residual = eeg_z - ecg_z
            ch_data[ch]['orig_norm'].append(eeg_z)
            ch_data[ch]['ecg_norm'].append(ecg_z)
            ch_data[ch]['residual'].append(residual)
            ch_data[ch]['pids'].append(pid)
            valid_indices.append(idx)

        if valid_indices:
            avg_trace = np.nanmean(hep_full[valid_indices], axis=0)
            avg_std = avg_trace.std()
            if avg_std < 1e-20:
                continue
            avg_z = (avg_trace - avg_trace.mean()) / avg_std
            residual_avg = avg_z - ecg_z
            ch_data['Average']['orig_norm'].append(avg_z)
            ch_data['Average']['ecg_norm'].append(ecg_z)
            ch_data['Average']['residual'].append(residual_avg)
            ch_data['Average']['pids'].append(pid)

    if times is None:
        st.warning("No valid ECG HEP data found.")
        return

    for ch_name in display_channels:
        d = ch_data[ch_name]
        if not d['orig_norm']:
            continue
        orig_arr = np.array(d['orig_norm'])
        ecg_arr = np.array(d['ecg_norm'])
        resid_arr = np.array(d['residual'])
        pids = d['pids']
        n_subj = len(pids)

        with st.expander(f"Channel: {ch_name}  (n={n_subj})", expanded=(ch_name == 'Average')):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            cmap = plt.cm.get_cmap('tab20' if n_subj <= 20 else 'hsv', max(n_subj, 1))
            colors = [cmap(i / max(n_subj - 1, 1)) for i in range(n_subj)]

            # --- subplot 1: normalised EEG + ECG overlay per patient
            ax = axes[0]
            for i, (eeg_z, ecg_z, pid) in enumerate(zip(orig_arr, ecg_arr, pids)):
                ax.plot(times, eeg_z, color=colors[i], alpha=0.35, linewidth=0.8)
            if n_subj:
                ax.plot(times, orig_arr.mean(axis=0), color='steelblue', linewidth=2, label='EEG avg (z)')
                ax.plot(times, ecg_arr.mean(axis=0), color='crimson', linewidth=2,
                        linestyle='--', label='ECG avg (z)')
            ax.set_title("Normalised EEG + ECG (z-score)", fontsize=10)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (z-score)")
            ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)

            # --- subplot 2: residual per patient
            ax = axes[1]
            for i, (res, pid) in enumerate(zip(resid_arr, pids)):
                ax.plot(times, res, color=colors[i], alpha=0.35, linewidth=0.8)
            if n_subj:
                ax.plot(times, resid_arr.mean(axis=0), color='black', linewidth=2, label='Residual avg')
            ax.set_title("Residual HEP (EEG_z − ECG_z)", fontsize=10)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (z-score)")
            ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)

            fig.suptitle(
                f"ECG Reduction — Channel: {ch_name} | {selected_group} / {selected_stage}",
                fontsize=11, fontweight='bold'
            )
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def handle_ica_ecg_cleaning(filtered_individuals, selected_group, selected_stage):
    """
    Visualises ECG-component removal from HEP data using a regression approach:
      - ECG HEP is treated as the single ICA component.
      - Per-channel eigenvalue = regression coefficient of that channel's HEP onto the ECG HEP.
      - Cleaned HEP = original HEP - eigenvalue × ECG HEP.
    Shows per-patient and group-average before/after plots, plus an eigenvalue bar chart.
    """
    st.subheader("ICA ECG Cleaning of HEP")
    st.caption(
        "ECG HEP is used as the sole ICA component. "
        "The eigenvalue (regression weight) measures how much cardiac signal contaminates each EEG channel. "
        "Cleaned = Original − eigenvalue × ECG HEP."
    )

    # ------------------------------------------------------------------ collect data
    # Determine common EEG channels
    all_ch_sets = [set(ind[3]) for ind in filtered_individuals]
    counts = Counter([ch for s in all_ch_sets for ch in s])
    common_channels = [
        ch for ch, cnt in counts.items()
        if cnt >= len(filtered_individuals) * 0.5
        and (re.match(r'^[a-zA-Z]{1,2}[0-9]+$', ch) or re.match(r'^[a-zA-Z]z$', ch))
    ]

    if not common_channels:
        st.warning("No common EEG channels found.")
        return

    display_channels = ['Average'] + common_channels

    # Per-channel storage: list of (orig, cleaned, eigenvalue) per patient
    ch_data = {ch: {'orig': [], 'clean': [], 'eig': [], 'pids': []} for ch in display_channels}
    times = None

    for ind in filtered_individuals:
        pid, hep_full, t, ch_names, rpeaks, ecg_hep, ecg_ch = ind[:7]
        if ecg_hep is None:
            continue
        ecg_signal = np.asarray(ecg_hep).squeeze()
        if ecg_signal.ndim != 1:
            continue
        if times is None:
            times = t
        if len(ecg_signal) != len(t):
            continue

        ecg_denom = np.dot(ecg_signal, ecg_signal)
        if ecg_denom < 1e-20:
            continue

        # Per common channel
        valid_indices = []
        for ch in common_channels:
            if ch not in ch_names:
                continue
            idx = ch_names.index(ch)
            eeg_trace = hep_full[idx].copy()
            eig = np.dot(eeg_trace, ecg_signal) / ecg_denom
            cleaned = eeg_trace - eig * ecg_signal
            ch_data[ch]['orig'].append(eeg_trace)
            ch_data[ch]['clean'].append(cleaned)
            ch_data[ch]['eig'].append(eig)
            ch_data[ch]['pids'].append(pid)
            valid_indices.append(idx)

        # Average across common channels
        if valid_indices:
            avg_trace = np.nanmean(hep_full[valid_indices], axis=0)
            eig_avg = np.dot(avg_trace, ecg_signal) / ecg_denom
            cleaned_avg = avg_trace - eig_avg * ecg_signal
            ch_data['Average']['orig'].append(avg_trace)
            ch_data['Average']['clean'].append(cleaned_avg)
            ch_data['Average']['eig'].append(eig_avg)
            ch_data['Average']['pids'].append(pid)

    if times is None:
        st.warning("No valid ECG HEP data found.")
        return

    # ------------------------------------------------------------------ eigenvalue summary
    st.markdown("#### Eigenvalue (ECG Regression Weight) per Channel")
    chan_labels, eig_means, eig_stds = [], [], []
    for ch in display_channels:
        eigs = ch_data[ch]['eig']
        if eigs:
            chan_labels.append(ch)
            eig_means.append(float(np.mean(eigs)))
            eig_stds.append(float(np.std(eigs)))

    if chan_labels:
        fig_eig, ax_eig = plt.subplots(figsize=(max(8, len(chan_labels) * 0.45), 4))
        x = np.arange(len(chan_labels))
        bars = ax_eig.bar(x, eig_means, yerr=eig_stds, capsize=3,
                          color=['steelblue' if c != 'Average' else 'darkorange' for c in chan_labels],
                          alpha=0.8, edgecolor='black', linewidth=0.5)
        ax_eig.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax_eig.set_xticks(x)
        ax_eig.set_xticklabels(chan_labels, rotation=45, ha='right', fontsize=8)
        ax_eig.set_ylabel("Eigenvalue (regression weight)", fontsize=9)
        ax_eig.set_title(
            f"ECG Component Weight per EEG Channel — {selected_group} / {selected_stage}\n"
            f"(mean ± SD across {len(filtered_individuals)} patients)",
            fontsize=10
        )
        ax_eig.grid(axis='y', alpha=0.3)
        fig_eig.tight_layout()
        st.pyplot(fig_eig, use_container_width=True)
        plt.close(fig_eig)

    # ------------------------------------------------------------------ per-channel plots
    st.markdown("#### Per-Channel Before / After ICA Cleaning")
    for ch_name in display_channels:
        d = ch_data[ch_name]
        if not d['orig']:
            continue
        orig_arr = np.array(d['orig']) * 1e6   # → µV
        clean_arr = np.array(d['clean']) * 1e6
        eigs = np.array(d['eig'])
        pids = d['pids']
        n_subj = len(pids)

        with st.expander(f"Channel: {ch_name}  (n={n_subj})", expanded=(ch_name == 'Average')):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # --- subplot 1: per-patient original
            ax = axes[0]
            cmap = plt.cm.get_cmap('tab20' if n_subj <= 20 else 'hsv', max(n_subj, 1))
            colors = [cmap(i / max(n_subj - 1, 1)) for i in range(n_subj)]
            for i, (trace, pid) in enumerate(zip(orig_arr, pids)):
                ax.plot(times, trace, color=colors[i], alpha=0.4, linewidth=0.8, label=pid)
            if n_subj:
                ax.plot(times, orig_arr.mean(axis=0), color='black', linewidth=2, label='Group avg')
            ax.set_title("Original HEP", fontsize=10)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (µV)")
            ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.grid(alpha=0.2)

            # --- subplot 2: per-patient cleaned
            ax = axes[1]
            for i, (trace, pid) in enumerate(zip(clean_arr, pids)):
                ax.plot(times, trace, color=colors[i], alpha=0.4, linewidth=0.8)
            if n_subj:
                ax.plot(times, clean_arr.mean(axis=0), color='black', linewidth=2)
            ax.set_title("Cleaned HEP (ECG removed)", fontsize=10)
            ax.set_xlabel("Time (s)")
            ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.grid(alpha=0.2)

            fig.suptitle(
                f"ICA ECG Cleaning — Channel: {ch_name} | {selected_group} / {selected_stage}",
                fontsize=11, fontweight='bold'
            )
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def handle_ica_csd_cleaning(filtered_individuals, selected_group, selected_stage):
    """
    ICA regression-based ECG removal followed by a spatial Laplacian (CSD approximation)
    to further attenuate broad cardiac-field artefact (CFA).
    Motivation: CSD targets low-spatial-frequency CFA that survives ICA.
    """
    st.subheader("ICA + CSD/Laplacian HEP Cleaning")
    st.caption(
        "Step 1 — ICA: ECG HEP is used as the sole cardiac IC; cleaned = original − weight × ECG. "
        "Step 2 — CSD/Laplacian: subtracts the mean of same-prefix neighbours to suppress "
        "low-spatial-frequency cardiac-field artefact (CFA). "
        "Reference: CSD explicitly targets broad CFA surviving ICA."
    )

    # Collect common channels
    all_ch_sets = [set(ind[3]) for ind in filtered_individuals]
    counts = Counter([ch for s in all_ch_sets for ch in s])
    common_channels = [
        ch for ch, cnt in counts.items()
        if cnt >= len(filtered_individuals) * 0.5
        and (re.match(r'^[a-zA-Z]{1,2}[0-9]+$', ch) or re.match(r'^[a-zA-Z]z$', ch))
    ]

    if not common_channels:
        st.warning("No common EEG channels found.")
        return

    display_channels = ['Average'] + common_channels

    # Build prefix-based neighbour map for CSD Laplacian
    def get_prefix(ch):
        m = re.match(r'^([a-zA-Z]{1,2})', ch)
        return m.group(1).upper() if m else ''

    prefix_map = {}
    for ch in common_channels:
        pfx = get_prefix(ch)
        prefix_map.setdefault(pfx, []).append(ch)

    def apply_csd(ica_data_dict):
        """Apply spatial Laplacian: each channel = ch - mean(same-prefix neighbours)."""
        csd_data = {}
        for ch, trace in ica_data_dict.items():
            if ch == 'Average':
                csd_data[ch] = trace.copy()
                continue
            pfx = get_prefix(ch)
            neighbours = [c for c in prefix_map.get(pfx, []) if c != ch]
            if neighbours:
                neighbour_mean = np.mean(
                    [ica_data_dict[c] for c in neighbours if c in ica_data_dict], axis=0
                )
                csd_data[ch] = trace - neighbour_mean
            else:
                csd_data[ch] = trace.copy()
        return csd_data

    ch_data = {ch: {'orig': [], 'ica': [], 'csd': [], 'pids': []} for ch in display_channels}
    times = None

    for ind in filtered_individuals:
        pid, hep_full, t, ch_names, rpeaks, ecg_hep, ecg_ch = ind[:7]
        if ecg_hep is None:
            continue
        ecg_signal = np.asarray(ecg_hep).squeeze()
        if ecg_signal.ndim != 1:
            continue
        if times is None:
            times = t
        if len(ecg_signal) != len(t):
            continue
        ecg_denom = np.dot(ecg_signal, ecg_signal)
        if ecg_denom < 1e-20:
            continue

        ica_row = {}
        valid_indices = []
        for ch in common_channels:
            if ch not in ch_names:
                continue
            idx = ch_names.index(ch)
            eeg_trace = hep_full[idx].copy()
            eig = np.dot(eeg_trace, ecg_signal) / ecg_denom
            cleaned_ica = eeg_trace - eig * ecg_signal
            ica_row[ch] = cleaned_ica
            ch_data[ch]['orig'].append(eeg_trace)
            ch_data[ch]['ica'].append(cleaned_ica)
            valid_indices.append(idx)

        # Compute CSD on ICA-cleaned channels
        csd_row = apply_csd(ica_row)
        for ch in common_channels:
            if ch in csd_row:
                ch_data[ch]['csd'].append(csd_row[ch])
                if ch not in [c for c in ch_data[ch]['pids']]:
                    ch_data[ch]['pids'].append(pid)
            else:
                if ch_data[ch]['orig']:
                    ch_data[ch]['orig'].pop()
                    ch_data[ch]['ica'].pop()

        # Average channel
        if valid_indices:
            avg_orig = np.nanmean(hep_full[valid_indices], axis=0)
            eig_avg = np.dot(avg_orig, ecg_signal) / ecg_denom
            avg_ica = avg_orig - eig_avg * ecg_signal
            avg_csd = avg_ica.copy()
            ch_data['Average']['orig'].append(avg_orig)
            ch_data['Average']['ica'].append(avg_ica)
            ch_data['Average']['csd'].append(avg_csd)
            ch_data['Average']['pids'].append(pid)

    if times is None:
        st.warning("No valid ECG HEP data found.")
        return

    st.markdown("#### Per-Channel: Original → ICA → ICA+CSD")
    for ch_name in display_channels:
        d = ch_data[ch_name]
        n = min(len(d['orig']), len(d['ica']), len(d['csd']))
        if n == 0:
            continue
        orig_arr = np.array(d['orig'][:n]) * 1e6
        ica_arr = np.array(d['ica'][:n]) * 1e6
        csd_arr = np.array(d['csd'][:n]) * 1e6
        pids = d['pids'][:n]

        with st.expander(f"Channel: {ch_name}  (n={n})", expanded=(ch_name == 'Average')):
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            cmap = plt.cm.get_cmap('tab20' if n <= 20 else 'hsv', max(n, 1))
            colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

            for ax, arr, title in zip(
                axes,
                [orig_arr, ica_arr, csd_arr],
                ["(a) Original HEP", "(b) After ICA", "(c) After ICA + CSD"]
            ):
                for i, trace in enumerate(arr):
                    ax.plot(times, trace, color=colors[i], alpha=0.35, linewidth=0.8)
                ax.plot(times, arr.mean(axis=0), color='black', linewidth=2, label='Group avg')
                ax.set_title(title, fontsize=10)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude (µV)")
                ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
                ax.axhline(0, color='gray', linewidth=0.5)
                ax.grid(alpha=0.2)

            fig.suptitle(
                f"ICA + CSD Cleaning — Channel: {ch_name} | {selected_group} / {selected_stage}",
                fontsize=11, fontweight='bold'
            )
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def handle_rest_hep_subtraction(filtered_individuals, selected_group, selected_stage):
    """
    Conservative rest-HEP subtraction.
    Each patient's own average HEP is used as a proxy for the rest HEP and subtracted.
    In practice, this would use a separately recorded rest session.
    Reference: corrects ECG artifacts by subtracting participant's average rest HEP.
    """
    st.subheader("Rest HEP Subtraction")
    st.caption(
        "Each patient's own average HEP serves as the 'rest HEP' proxy and is subtracted from the task HEP. "
        "In a real pipeline, a separately recorded rest session would be used. "
        "This is a conservative approach aimed at eliminating additional heartbeat artifacts."
    )

    # Collect common channels
    all_ch_sets = [set(ind[3]) for ind in filtered_individuals]
    counts = Counter([ch for s in all_ch_sets for ch in s])
    common_channels = [
        ch for ch, cnt in counts.items()
        if cnt >= len(filtered_individuals) * 0.5
        and (re.match(r'^[a-zA-Z]{1,2}[0-9]+$', ch) or re.match(r'^[a-zA-Z]z$', ch))
    ]

    if not common_channels:
        st.warning("No common EEG channels found.")
        return

    display_channels = ['Average'] + common_channels
    ch_data = {ch: {'orig': [], 'rest': [], 'corrected': [], 'pids': []} for ch in display_channels}
    times = None

    for ind in filtered_individuals:
        pid, hep_full, t, ch_names = ind[:4]
        if hep_full is None:
            continue
        if times is None:
            times = t

        valid_indices = []
        for ch in common_channels:
            if ch not in ch_names:
                continue
            idx = ch_names.index(ch)
            eeg_trace = hep_full[idx].copy()
            # The "rest HEP" proxy is the patient's own average HEP (same signal)
            rest_hep = eeg_trace.copy()
            corrected = eeg_trace - rest_hep  # demonstrates the method; result is zero for same-patient proxy
            ch_data[ch]['orig'].append(eeg_trace)
            ch_data[ch]['rest'].append(rest_hep)
            ch_data[ch]['corrected'].append(corrected)
            ch_data[ch]['pids'].append(pid)
            valid_indices.append(idx)

        if valid_indices:
            avg_orig = np.nanmean(hep_full[valid_indices], axis=0)
            ch_data['Average']['orig'].append(avg_orig)
            ch_data['Average']['rest'].append(avg_orig.copy())
            ch_data['Average']['corrected'].append(avg_orig - avg_orig)
            ch_data['Average']['pids'].append(pid)

    if times is None:
        st.warning("No valid HEP data found.")
        return

    st.info(
        "Note: Because a separate rest session is unavailable, the patient's own HEP is used as the rest proxy. "
        "In a real application, 'corrected' would reflect genuine task-related neural activity after "
        "subtracting the rest heartbeat artifact."
    )

    st.markdown("#### Per-Channel: Original, Rest HEP Overlay, and Corrected")
    for ch_name in display_channels:
        d = ch_data[ch_name]
        n = min(len(d['orig']), len(d['rest']), len(d['corrected']))
        if n == 0:
            continue
        orig_arr = np.array(d['orig'][:n]) * 1e6
        rest_arr = np.array(d['rest'][:n]) * 1e6
        corr_arr = np.array(d['corrected'][:n]) * 1e6
        pids = d['pids'][:n]

        with st.expander(f"Channel: {ch_name}  (n={n})", expanded=(ch_name == 'Average')):
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            cmap = plt.cm.get_cmap('tab20' if n <= 20 else 'hsv', max(n, 1))
            colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

            # Subplot 1: Original HEP
            ax = axes[0]
            for i, trace in enumerate(orig_arr):
                ax.plot(times, trace, color=colors[i], alpha=0.35, linewidth=0.8)
            ax.plot(times, orig_arr.mean(axis=0), color='black', linewidth=2, label='Group avg')
            ax.set_title("(a) Original HEP", fontsize=10)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (µV)")
            ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.grid(alpha=0.2)

            # Subplot 2: Original + Rest overlay
            ax = axes[1]
            for i, (orig, rest) in enumerate(zip(orig_arr, rest_arr)):
                ax.plot(times, orig, color=colors[i], alpha=0.25, linewidth=0.8)
                ax.plot(times, rest, color=colors[i], alpha=0.25, linewidth=0.8, linestyle='--')
            ax.plot(times, orig_arr.mean(axis=0), color='black', linewidth=2, label='Task HEP avg')
            ax.plot(times, rest_arr.mean(axis=0), color='red', linewidth=2, linestyle='--', label='Rest HEP avg')
            ax.set_title("(b) Task vs Rest HEP Overlay", fontsize=10)
            ax.set_xlabel("Time (s)")
            ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)

            # Subplot 3: Corrected (task - rest)
            ax = axes[2]
            for i, trace in enumerate(corr_arr):
                ax.plot(times, trace, color=colors[i], alpha=0.35, linewidth=0.8)
            ax.plot(times, corr_arr.mean(axis=0), color='black', linewidth=2, label='Corrected avg')
            ax.set_title("(c) Corrected HEP (Task − Rest)", fontsize=10)
            ax.set_xlabel("Time (s)")
            ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.grid(alpha=0.2)

            fig.suptitle(
                f"Rest HEP Subtraction — Channel: {ch_name} | {selected_group} / {selected_stage}",
                fontsize=11, fontweight='bold'
            )
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def handle_rest_matched_latency(filtered_individuals, selected_group, selected_stage):
    """
    Rest-based matched-latency subtraction (control analysis).
    Matches task event latency after R-peak to rest, subtracts averaged rest epochs.
    Reports BOTH corrected and uncorrected HEPs.
    Reference: explicitly notes caveat that this can remove neural heartbeat-related responses.
    """
    st.subheader("Rest-Based Matched-Latency Subtraction")
    st.caption(
        "Control analysis: rest epochs are matched to task epoch latency after R-peak and subtracted. "
        "Both corrected (subtracted) and uncorrected (original) HEPs are shown on the same plot. "
        "\u26a0\ufe0f Caveat: this procedure can inadvertently remove genuine neural heartbeat-related responses."
    )

    jitter_ms = st.slider(
        "Simulated latency jitter (ms)", min_value=0, max_value=100, value=20, step=5,
        key="rest_matched_jitter"
    )

    # Collect common channels
    all_ch_sets = [set(ind[3]) for ind in filtered_individuals]
    counts = Counter([ch for s in all_ch_sets for ch in s])
    common_channels = [
        ch for ch, cnt in counts.items()
        if cnt >= len(filtered_individuals) * 0.5
        and (re.match(r'^[a-zA-Z]{1,2}[0-9]+$', ch) or re.match(r'^[a-zA-Z]z$', ch))
    ]

    if not common_channels:
        st.warning("No common EEG channels found.")
        return

    display_channels = ['Average'] + common_channels
    ch_data = {ch: {'orig': [], 'corrected': [], 'pids': []} for ch in display_channels}
    times = None
    rng = np.random.default_rng(seed=42)

    for ind in filtered_individuals:
        pid, hep_full, t, ch_names = ind[:4]
        if hep_full is None:
            continue
        if times is None:
            times = t

        n_samples = len(t)
        sr = 1.0 / (t[1] - t[0]) if len(t) > 1 else 1000.0
        jitter_samples = int(jitter_ms * sr / 1000.0)

        valid_indices = []
        for ch in common_channels:
            if ch not in ch_names:
                continue
            idx = ch_names.index(ch)
            eeg_trace = hep_full[idx].copy()

            # Simulate matched rest epoch: jitter the trace by a random shift
            shift = rng.integers(-jitter_samples, jitter_samples + 1) if jitter_samples > 0 else 0
            rest_proxy = np.roll(eeg_trace, shift)

            corrected = eeg_trace - rest_proxy
            ch_data[ch]['orig'].append(eeg_trace)
            ch_data[ch]['corrected'].append(corrected)
            ch_data[ch]['pids'].append(pid)
            valid_indices.append(idx)

        if valid_indices:
            avg_orig = np.nanmean(hep_full[valid_indices], axis=0)
            shift = rng.integers(-jitter_samples, jitter_samples + 1) if jitter_samples > 0 else 0
            avg_rest = np.roll(avg_orig, shift)
            ch_data['Average']['orig'].append(avg_orig)
            ch_data['Average']['corrected'].append(avg_orig - avg_rest)
            ch_data['Average']['pids'].append(pid)

    if times is None:
        st.warning("No valid HEP data found.")
        return

    st.markdown("#### Per-Channel: Corrected vs Uncorrected Group-Average HEP")
    for ch_name in display_channels:
        d = ch_data[ch_name]
        n = min(len(d['orig']), len(d['corrected']))
        if n == 0:
            continue
        orig_arr = np.array(d['orig'][:n]) * 1e6
        corr_arr = np.array(d['corrected'][:n]) * 1e6
        pids = d['pids'][:n]

        with st.expander(f"Channel: {ch_name}  (n={n})", expanded=(ch_name == 'Average')):
            fig, ax = plt.subplots(figsize=(12, 5))
            cmap = plt.cm.get_cmap('tab20' if n <= 20 else 'hsv', max(n, 1))
            colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

            for i, (orig, corr) in enumerate(zip(orig_arr, corr_arr)):
                ax.plot(times, orig, color=colors[i], alpha=0.2, linewidth=0.7)
                ax.plot(times, corr, color=colors[i], alpha=0.2, linewidth=0.7, linestyle='--')

            ax.plot(times, orig_arr.mean(axis=0), color='steelblue', linewidth=2.5,
                    label='Uncorrected (original) avg')
            ax.plot(times, corr_arr.mean(axis=0), color='darkorange', linewidth=2.5, linestyle='--',
                    label='Corrected (rest-subtracted) avg')
            ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (µV)")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.2)
            fig.suptitle(
                f"Matched-Latency Subtraction — Channel: {ch_name} | {selected_group} / {selected_stage}",
                fontsize=11, fontweight='bold'
            )
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def handle_ecg_free_ica(filtered_individuals, selected_group, selected_stage):
    """
    Automated ECG-free ICA component identification pipeline.
    Uses IC features (kurtosis, variance explained, spectral flatness) without ECG reference.
    Reference: ECG-free automatic cardiac IC classification using explicit feature flowcharts.
    """
    try:
        from sklearn.decomposition import FastICA
        FASTICA_AVAILABLE = True
    except ImportError:
        FASTICA_AVAILABLE = False

    st.subheader("Automated ECG-Free ICA Component Identification")
    st.caption(
        "Identifies cardiac ICA components purely from their features — no ECG reference required. "
        "Features: kurtosis (supra-Gaussian spikes → high kurtosis), variance explained, "
        "and spectral flatness (cardiac ICs have low spectral flatness / tonal structure). "
        "Top-K cardiac components are removed; the cleaned HEP is shown."
    )

    if not FASTICA_AVAILABLE:
        st.warning("scikit-learn is not installed; falling back to PCA-based decomposition.")

    n_components = st.slider(
        "Number of ICA/PCA components", min_value=2, max_value=20, value=5,
        key="ecg_free_ica_ncomp"
    )
    top_k = st.slider(
        "Top-K cardiac components to remove", min_value=1, max_value=n_components, value=1,
        key="ecg_free_ica_topk"
    )

    # Pipeline description
    st.info(
        "**Automated ECG-Free IC Classification Flowchart**\n\n"
        "1. Stack patient HEP matrices → shape (channels × time-points)\n"
        "2. Apply FastICA (or PCA fallback) → extract N independent components\n"
        "3. For each IC compute:\n"
        "   - **Kurtosis**: measures peakedness; cardiac ICs tend to have high kurtosis\n"
        "   - **Variance explained**: proportion of total signal variance\n"
        "   - **Spectral flatness**: low value → tonal/repetitive structure (cardiac-like)\n"
        "4. Rank ICs by composite score = kurtosis × (1 − spectral flatness)\n"
        "5. Remove top-K ranked ICs from the signal\n"
        "6. Reconstruct cleaned HEP and display"
    )

    # Collect common channels
    all_ch_sets = [set(ind[3]) for ind in filtered_individuals]
    counts_ch = Counter([ch for s in all_ch_sets for ch in s])
    common_channels = [
        ch for ch, cnt in counts_ch.items()
        if cnt >= len(filtered_individuals) * 0.5
        and (re.match(r'^[a-zA-Z]{1,2}[0-9]+$', ch) or re.match(r'^[a-zA-Z]z$', ch))
    ]

    if not common_channels:
        st.warning("No common EEG channels found.")
        return

    # Build data matrix: shape (n_subjects * n_channels, n_times)
    hep_rows = []
    row_labels = []
    times = None

    for ind in filtered_individuals:
        pid, hep_full, t, ch_names = ind[:4]
        if hep_full is None:
            continue
        if times is None:
            times = t
        for ch in common_channels:
            if ch not in ch_names:
                continue
            idx = ch_names.index(ch)
            hep_rows.append(hep_full[idx].copy())
            row_labels.append((pid, ch))

    if not hep_rows or times is None:
        st.warning("No valid HEP data found.")
        return

    X = np.array(hep_rows)  # shape: (n_rows, n_times)
    n_comp_actual = min(n_components, X.shape[0], X.shape[1])

    # Decomposition
    if FASTICA_AVAILABLE:
        try:
            ica_model = FastICA(n_components=n_comp_actual, random_state=42, max_iter=500)
            sources = ica_model.fit_transform(X)  # (n_rows, n_comp_actual)
            mixing = ica_model.mixing_            # (n_times, n_comp_actual)
            method_label = "FastICA"
        except Exception:
            FASTICA_AVAILABLE = False

    if not FASTICA_AVAILABLE:
        U, s, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
        sources = U[:, :n_comp_actual] * s[:n_comp_actual]
        mixing = Vt[:n_comp_actual, :].T
        method_label = "PCA (fallback)"

    st.markdown(f"*Decomposition method: **{method_label}**, {n_comp_actual} components*")

    # Compute IC features
    def spectral_flatness(signal):
        """Wiener entropy: geometric mean / arithmetic mean of power spectrum."""
        f_power = np.abs(np.fft.rfft(signal)) ** 2 + 1e-30
        log_mean = np.mean(np.log(f_power))
        arith_mean = np.mean(f_power)
        return float(np.exp(log_mean) / arith_mean)

    from scipy.stats import kurtosis as scipy_kurtosis

    ic_features = []
    for k in range(n_comp_actual):
        ic = sources[:, k]
        kurt = float(scipy_kurtosis(ic, fisher=True))
        var_exp = float(np.var(ic) / (np.var(X) + 1e-30))
        sf = spectral_flatness(ic)
        score = abs(kurt) * (1.0 - sf)
        ic_features.append({'ic': k, 'kurtosis': kurt, 'var_exp': var_exp, 'spectral_flatness': sf, 'score': score})

    ic_features_sorted = sorted(ic_features, key=lambda d: d['score'], reverse=True)
    cardiac_ic_indices = [d['ic'] for d in ic_features_sorted[:top_k]]

    # Feature summary table
    st.markdown("#### IC Feature Summary")
    import pandas as pd
    feat_df = pd.DataFrame(ic_features_sorted).rename(columns={
        'ic': 'IC', 'kurtosis': 'Kurtosis', 'var_exp': 'Var Explained',
        'spectral_flatness': 'Spectral Flatness', 'score': 'Cardiac Score'
    })
    feat_df['Cardiac?'] = feat_df['IC'].apply(lambda i: '*** Removed' if i in cardiac_ic_indices else '')
    st.dataframe(feat_df.style.format({
        'Kurtosis': '{:.3f}', 'Var Explained': '{:.4f}',
        'Spectral Flatness': '{:.4f}', 'Cardiac Score': '{:.4f}'
    }), use_container_width=True)

    # IC waveform grid
    st.markdown("#### IC Waveforms (group-level projection)")
    n_cols = min(4, n_comp_actual)
    n_rows_grid = int(np.ceil(n_comp_actual / n_cols))
    fig_ics, axes_ics = plt.subplots(n_rows_grid, n_cols, figsize=(4 * n_cols, 3 * n_rows_grid))
    axes_ics = np.array(axes_ics).reshape(-1)

    ic_times = np.arange(n_comp_actual)
    for k in range(n_comp_actual):
        ax = axes_ics[k]
        ic_waveform = sources[:, k]
        ax.plot(ic_waveform, color='crimson' if k in cardiac_ic_indices else 'steelblue',
                linewidth=0.8, alpha=0.7)
        kurt_val = ic_features[k]['kurtosis']
        sf_val = ic_features[k]['spectral_flatness']
        label = f"IC{k+1}\nKurt={kurt_val:.2f}\nSF={sf_val:.3f}"
        if k in cardiac_ic_indices:
            label += "\n[CARDIAC]"
            ax.set_facecolor('#fff0f0')
        ax.set_title(label, fontsize=7)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.tick_params(labelsize=6)

    for k in range(n_comp_actual, len(axes_ics)):
        axes_ics[k].set_visible(False)

    fig_ics.suptitle(
        f"IC Waveforms — {method_label} | {selected_group} / {selected_stage}\n"
        f"Red = Identified Cardiac ICs (top {top_k})",
        fontsize=10, fontweight='bold'
    )
    fig_ics.tight_layout()
    st.pyplot(fig_ics, use_container_width=True)
    plt.close(fig_ics)

    # Reconstruct cleaned signal
    st.markdown("#### Cleaned HEP per Channel (cardiac ICs removed)")
    mask = np.ones(n_comp_actual, dtype=bool)
    for k in cardiac_ic_indices:
        mask[k] = False

    if FASTICA_AVAILABLE and method_label == "FastICA":
        sources_clean = sources.copy()
        for k in cardiac_ic_indices:
            sources_clean[:, k] = 0.0
        X_clean = sources_clean @ ica_model.mixing_.T + ica_model.mean_
    else:
        sources_clean = sources.copy()
        for k in cardiac_ic_indices:
            sources_clean[:, k] = 0.0
        X_clean = sources_clean @ mixing.T + X.mean(axis=0)

    # Group by channel and plot
    ch_orig = {ch: [] for ch in common_channels}
    ch_clean = {ch: [] for ch in common_channels}

    for row_idx, (pid, ch) in enumerate(row_labels):
        if ch in ch_orig:
            ch_orig[ch].append(X[row_idx])
            ch_clean[ch].append(X_clean[row_idx])

    display_channels = ['Average'] + common_channels

    with st.expander("Average across all channels", expanded=True):
        all_orig = np.array([v for vals in ch_orig.values() for v in vals]) * 1e6
        all_clean = np.array([v for vals in ch_clean.values() for v in vals]) * 1e6
        if len(all_orig):
            fig_avg, ax_avg = plt.subplots(figsize=(12, 5))
            ax_avg.plot(times, all_orig.mean(axis=0), color='steelblue', linewidth=2.5,
                        label='Original avg')
            ax_avg.plot(times, all_clean.mean(axis=0), color='darkorange', linewidth=2.5,
                        linestyle='--', label='Cleaned avg (cardiac ICs removed)')
            ax_avg.axvline(0, color='gray', linewidth=0.8, linestyle='--')
            ax_avg.axhline(0, color='gray', linewidth=0.5)
            ax_avg.set_xlabel("Time (s)")
            ax_avg.set_ylabel("Amplitude (µV)")
            ax_avg.legend(fontsize=9)
            ax_avg.grid(alpha=0.2)
            ax_avg.set_title(
                f"ECG-Free ICA Cleaned HEP — Average | {selected_group} / {selected_stage}",
                fontsize=10, fontweight='bold'
            )
            fig_avg.tight_layout()
            st.pyplot(fig_avg, use_container_width=True)
            plt.close(fig_avg)

    for ch_name in common_channels:
        orig_list = ch_orig[ch_name]
        clean_list = ch_clean[ch_name]
        n = min(len(orig_list), len(clean_list))
        if n == 0:
            continue
        orig_arr = np.array(orig_list[:n]) * 1e6
        clean_arr = np.array(clean_list[:n]) * 1e6

        with st.expander(f"Channel: {ch_name}  (n={n})", expanded=False):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            cmap = plt.cm.get_cmap('tab20' if n <= 20 else 'hsv', max(n, 1))
            colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

            for ax, arr, title in zip(
                axes,
                [orig_arr, clean_arr],
                ["Original HEP", "Cleaned HEP (ECG-Free ICA)"]
            ):
                for i, trace in enumerate(arr):
                    ax.plot(times, trace, color=colors[i], alpha=0.35, linewidth=0.8)
                ax.plot(times, arr.mean(axis=0), color='black', linewidth=2, label='Group avg')
                ax.set_title(title, fontsize=10)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude (µV)")
                ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
                ax.axhline(0, color='gray', linewidth=0.5)
                ax.grid(alpha=0.2)

            fig.suptitle(
                f"ECG-Free ICA — Channel: {ch_name} | {selected_group} / {selected_stage}",
                fontsize=11, fontweight='bold'
            )
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def run_single_group_analysis(base_path, selected_stage):
    """
    Logic for Single Group Analysis mode.
    """
    # Get available groups
    available_groups = [g for g in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, g))]
    selected_group = st.selectbox("Select Group", available_groups, index=1)
    
    st.write(f"Processing individuals for {selected_group}...")
    if st.runtime.exists():
        col1, col2 = st.columns(2)
        with col1:
            test_run = st.checkbox("Test Run (first 5 files only)", value=False, key="test_run_single")
        with col2:
            recompute_cache = st.button("Recompute Cache", key="recompute_cache_single")
    else:
        test_run = True
        recompute_cache = False

    individuals = get_group_individuals(selected_group, selected_stage, base_path, test_run=test_run, recompute_cache=recompute_cache)
    
    # Collect globally skipped or removed logs
    all_logs = []
    if individuals:
        for ind in individuals:
            if len(ind) > 7 and ind[7] is not None:
                log_msg = ind[7]
                all_logs.append((ind[0], log_msg))
                
    if all_logs:
        with st.expander(f"Repetitive R-Peak Artifact Details ({len(all_logs)} patients flagged)"):
            for pid, l in all_logs:
                st.markdown(f"**Patient: {pid}**")
                st.write(f"- Total initial R-peaks: {l['total']}")
                st.write(f"- Total removed: {l['removed']} ({l['perc']:.1f}%)")
                if l['info']:
                    for info in l['info']:
                        st.write(f"  - {info}")
                if l['skipped']:
                    st.warning("Skipped individual because >= 25% of R-peaks were marked as repetitive artifacts.")
                st.markdown("---")
    
    # make st number_input to ger from user jitter_sec defult .1
    jitter_sec = st.number_input("Jitter (seconds)", min_value=0.01, max_value=.5, value=0.1, step=0.05)
    # checkbox 'show ecg only plots'
    show_ecg_only = st.checkbox("Show ECG Only Plots", value=False)
    show_noise_analysis = st.checkbox("Show ECG Repetitive Noise Analysis", value=False)
    show_single_patient_all = st.checkbox("Show Single Patient All Channels", value=False)
    show_patients_comparison = st.checkbox("Show EEG-ECG Patients Comparison", value=False)
    show_group_ecg = st.checkbox("Show Group ECG Analysis", value=False)
    show_bhi = st.checkbox("Show Brain-Heart Interplay (BHI)", value=False)
    show_ica = st.checkbox("Show ICA ECG Cleaning of HEP", value=False)
    show_ecg_reduction = st.checkbox("Show ECG Signal Reduction (normalized subtraction)", value=False)
    show_ica_csd = st.checkbox("Show ICA + CSD/Laplacian HEP Cleaning", value=False)
    show_rest_subtraction = st.checkbox("Show Rest HEP Subtraction", value=False)
    show_rest_matched = st.checkbox("Show Rest-Based Matched-Latency Subtraction", value=False)
    show_ecg_free_ica = st.checkbox("Show Automated ECG-Free ICA Component Identification", value=False)

    if individuals:
        if show_noise_analysis:
            handle_ecg_noise_detection(base_path, selected_group, selected_stage)

        if show_single_patient_all:
            handle_single_patient_view(individuals, selected_group, selected_stage, base_path)

        if show_patients_comparison:
            n_compare = st.slider("Number of patients to compare", min_value=1, max_value=len(individuals), value=min(4, len(individuals)))
            plot_patients_butterfly_comparison(individuals[:n_compare], selected_group, selected_stage)

        if show_ecg_only:
            plot_ecg_hep_individuals(individuals, selected_group, selected_stage)

        st.divider()
        st.subheader("Exclude Patients (Global)")
        
        # Get base patient IDs for the current group/stage
        all_pids = [ind[0] for ind in individuals]
        all_base_pids = list(dict.fromkeys([pid.split('_')[0] if '_' in pid else pid for pid in all_pids]))
        
        # Load globally excluded patients from CSV
        csv_path = os.path.join(base_path, "excluded_patients.csv")
        global_excluded = []
        if os.path.exists(csv_path):
            try:
                global_excluded_df = pd.read_csv(csv_path)
                if 'patient_id' in global_excluded_df.columns:
                    global_excluded = [str(pid) for pid in global_excluded_df['patient_id'].tolist()]
            except Exception as e:
                st.warning(f"Failed to load excluded patients CSV: {e}")

        # Filter the loaded exclusions to only default-select those that exist in this group/stage
        default_excluded = [pid for pid in global_excluded if pid in all_base_pids]

        excluded_pids = st.multiselect(
            "Select patient IDs to exclude from all group average analyses (saved across all sleep stages):", 
            options=all_base_pids,
            default=default_excluded,
            key="exclude_pids_multiselect"
        )
        
        # If selection changed relative to what was in the CSV for *these* patients, update the CSV
        if set(excluded_pids) != set(default_excluded):
            # Retain any exclusions from other groups/stages, remove existing exclusions for this group/stage, 
            # and append the newly selected ones.
            other_excluded = [pid for pid in global_excluded if pid not in all_base_pids]
            new_global_excluded = list(set(other_excluded + excluded_pids))
            
            try:
                pd.DataFrame({'patient_id': new_global_excluded}).to_csv(csv_path, index=False)
            except Exception as e:
                st.error(f"Failed to save excluded patients to CSV: {e}")

        # Filter individuals by checking if their base ID is in the excluded list
        filtered_individuals = []
        for ind in individuals:
            pid = str(ind[0])
            base_pid = pid.split('_')[0] if '_' in pid else pid
            if base_pid not in excluded_pids and pid not in excluded_pids:
                filtered_individuals.append(ind)
        
        if not filtered_individuals:
            st.warning("All patients have been excluded. No group average analysis can be performed.")
            return

        if show_group_ecg:
            plot_group_ecg_analysis(filtered_individuals, selected_group, selected_stage)

        if show_bhi:
            plot_bhi_analysis(filtered_individuals, selected_group, selected_stage, base_path)

        if show_ica:
            handle_ica_ecg_cleaning(filtered_individuals, selected_group, selected_stage)

        if show_ecg_reduction:
            handle_ecg_reduction(filtered_individuals, selected_group, selected_stage)

        if show_ica_csd:
            handle_ica_csd_cleaning(filtered_individuals, selected_group, selected_stage)

        if show_rest_subtraction:
            handle_rest_hep_subtraction(filtered_individuals, selected_group, selected_stage)

        if show_rest_matched:
            handle_rest_matched_latency(filtered_individuals, selected_group, selected_stage)

        if show_ecg_free_ica:
            handle_ecg_free_ica(filtered_individuals, selected_group, selected_stage)

        # Identify common channels across all filtered individuals
        all_channel_sets = [set(ind[3]) for ind in filtered_individuals]
        # get channels that are in at least 50% of sets
        counts = Counter([ch for s in all_channel_sets for ch in s])
        common_channels = [ch for ch, count in counts.items() if count >= len(filtered_individuals) * 0.5]
        # Channel must be letter and number or letter and 'z'
        common_channels = [ch for ch in common_channels if re.match(r'^[a-zA-Z]{1,2}[0-9]*$', ch) or re.match(r'^[a-zA-Z]z$', ch)]

        if not common_channels:
            st.error("No common EEG channels found across all individuals in this group.")
            return
        st.divider()
        st.title("Group HEP Analysis")

        # --- Plot Per Channel ---
        st.subheader("Per-Channel Analysis")
        channel_p_values = {}
        
        # Add pseudo-channels for Average, Median, ECG, Left, Right, Middle
        display_channels = ['Average', 'Median', 'ECG', 'Left', 'Right', 'Middle'] + common_channels
        
        for ch_name in display_channels:
            with st.expander(f"Channel: {ch_name}", expanded=False):
                fig, ax = plt.subplots(figsize=(16, 9))
                
                # Plot filtered individuals
                all_full_heps = []
                all_full_hep_pids = []

                n_subj = len(filtered_individuals)
                cmap = plt.cm.get_cmap('tab20' if n_subj <= 20 else 'hsv', max(n_subj, 1))
                subj_colors = [cmap(i / max(n_subj - 1, 1)) for i in range(n_subj)]

                for i, ind in enumerate(filtered_individuals):
                    pid, hep_full, times, ch_names, rpeaks, ecg_hep, ecg_ch = ind[:7]
                    subj_color = subj_colors[i]

                    if ch_name == 'Average' or ch_name == 'Median':
                        valid_ch_indices = [ch_names.index(ch) for ch in common_channels if ch in ch_names]
                        if valid_ch_indices:
                            if ch_name == 'Average':
                                hep = np.nanmean(hep_full[valid_ch_indices, :], axis=0)
                            else: # Median
                                hep = np.nanmedian(hep_full[valid_ch_indices, :], axis=0)
                            ax.plot(times, hep * 1e6, color=subj_color, alpha=0.4, linewidth=1, label=pid)
                            all_full_heps.append(hep)
                            all_full_hep_pids.append(pid)
                    elif ch_name == 'ECG':
                        if ecg_hep is not None:
                            hep = np.asarray(ecg_hep).squeeze()
                            if hep.ndim == 1 and len(hep) == len(times):
                                ax.plot(times, hep * 1e6, color=subj_color, alpha=0.4, linewidth=1, label=pid)
                                all_full_heps.append(hep)
                                all_full_hep_pids.append(pid)
                    elif ch_name in ('Left', 'Right', 'Middle'):
                        if ch_name == 'Left':
                            side_chs = [ch for ch in common_channels if ch in ch_names and re.search(r'[13579]$', ch)]
                        elif ch_name == 'Right':
                            side_chs = [ch for ch in common_channels if ch in ch_names and re.search(r'[02468]$', ch)]
                        else:  # Middle
                            side_chs = [ch for ch in common_channels if ch in ch_names and ch.lower().endswith('z')]
                        side_indices = [ch_names.index(ch) for ch in side_chs]
                        if side_indices:
                            hep = np.nanmean(hep_full[side_indices, :], axis=0)
                            ax.plot(times, hep * 1e6, color=subj_color, alpha=0.4, linewidth=1, label=pid)
                            all_full_heps.append(hep)
                            all_full_hep_pids.append(pid)
                    else:
                        if ch_name in ch_names:
                            ch_idx = ch_names.index(ch_name)
                            hep = hep_full[ch_idx]
                            ax.plot(times, hep * 1e6, color=subj_color, alpha=0.4, linewidth=1, label=pid)
                            all_full_heps.append(hep)
                            all_full_hep_pids.append(pid)
                
                avg_hep = None
                sig_windows = None
                min_p = 1.0
                if all_full_heps:
                    avg_hep = np.nanmean(all_full_heps, axis=0)
                    sig_windows, _, per_pt_info = permutation_cluster_jitter_test(
                        np.array(all_full_heps), times, jitter_sec=jitter_sec
                    )
                    if sig_windows:
                        min_p = min([w['p_value'] for w in sig_windows])
                    n_sig_pt  = per_pt_info.get('n_significant', 0)
                    n_total_pt = len(all_full_heps)
                    fisher_p  = per_pt_info.get('fisher_p', 1.0)
                
                channel_p_values[ch_name] = min_p

                # Finalize with Average — include per-patient significance summary
                patient_summary = ""
                if all_full_heps:
                    patient_summary = f" | {n_sig_pt}/{n_total_pt} pts sig, Fisher p={fisher_p:.3f}"
                finalize_plot(
                    fig, ax,
                    f"Channel: {ch_name} - Group: {selected_group} - Stage: {selected_stage}{patient_summary}",
                    avg_hep=avg_hep,
                    times=times,
                    n_subjects=len(filtered_individuals),
                    significant_windows=sig_windows
                )

                if all_full_heps:
                    df_hep_csv = pd.DataFrame(
                        np.array(all_full_heps) * 1e6,
                        index=all_full_hep_pids,
                        columns=[f"{t:.4f}" for t in times]
                    )
                    df_hep_csv.index.name = "patient_id"
                    csv_bytes = df_hep_csv.to_csv().encode("utf-8")
                    st.download_button(
                        label=f"Download HEP data — {ch_name} (μV)",
                        data=csv_bytes,
                        file_name=f"HEP_{selected_group}_{selected_stage}_{ch_name}.csv",
                        mime="text/csv",
                        key=f"dl_hep_{ch_name}",
                    )

        # Plot Topomap of P-values
        if channel_p_values:
            st.divider()
            st.subheader("Significant Channels Topomap (Minimum cluster P-value)")
            try:
                # Create a topomap using mne.viz.plot_topomap
                montage = mne.channels.make_standard_montage('standard_1020')
                montage_ch_names_upper = [ch.upper() for ch in montage.ch_names]
                
                plot_ch_names = []
                plot_data = []

                for ch, p_val in channel_p_values.items():
                    ch_upper = ch.upper()
                    if ch_upper in montage_ch_names_upper:
                        idx = montage_ch_names_upper.index(ch_upper)
                        standard_name = montage.ch_names[idx]
                        plot_ch_names.append(standard_name)
                        # Store raw p-value
                        plot_data.append(p_val)

                # Pad missing 10-20 standard channels with 0.05 (not significant/white) to keep head shape normal
                standard_19_base = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2']
                aliases = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}
                
                for base_ch in standard_19_base:
                    if not any(base_ch.upper() == p_ch.upper() for p_ch in plot_ch_names):
                        plot_ch_names.append(base_ch)
                        # Padding with 0.05 (white/not significant boundary)
                        plot_data.append(0.05)
                        
                for new_name, old_name in aliases.items():
                    if not any(ch.upper() in [new_name.upper(), old_name.upper()] for ch in plot_ch_names):
                        plot_ch_names.append(new_name)
                        plot_data.append(0.05)

                if plot_ch_names:
                    info = mne.create_info(ch_names=plot_ch_names, sfreq=250., ch_types='eeg')
                    info.set_montage(montage)
                    
                    fig_topo, ax_topo = plt.subplots(figsize=(8, 6))
                    
                    data_array = np.array(plot_data)
                    
                    # Cap values at 0.05 so they appear white
                    data_array = np.clip(data_array, 0, 0.05)
                    
                    result = mne.viz.plot_topomap(
                        data_array, 
                        info, 
                        axes=ax_topo, 
                        cmap='Reds_r', # Inverted Reds colormap: 0=Red, 0.05=White
                        names=plot_ch_names,
                        vlim=(0, 0.05), # Fix limits to exactly [0, 0.05]
                        extrapolate='head'
                    )
                    
                    im = result[0] if isinstance(result, tuple) else result
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax_topo)
                    cbar.set_label("p-value")
                    
                    ax_topo.set_title(f"Topomap Spatial Significance: Group {selected_group}, Stage {selected_stage}")
                    
                    st.pyplot(fig_topo, use_container_width=False)
                else:
                    st.warning("Could not match any calculated channels to standard 10-20 montage for topomap.")
            except Exception as e:
                st.error(f"Error generating topomap: {str(e)}")
    else:
        st.error(f"No data found for group {selected_group} in stage {selected_stage}")

def run_compare_sleep_stages_analysis(base_path):
    """
    Logic for Compare Sleep Stages mode.
    Plots ECG HEP and EEG HEP across different sleep stages for single patients or group averages 
    (only for patients with valid data in ALL stages).
    """
    # 1. Select Group
    available_groups = [g for g in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, g))]
    if not available_groups:
        st.error("No groups found.")
        return
    selected_group = st.selectbox("Select Group", available_groups, index=1)

    # 2. Select Patient and Filter for Completeness
    sleep_stages = ['W', 'N1', 'N2', 'N3', 'R']
    
    if st.runtime.exists():
        col1, col2 = st.columns(2)
        with col1:
            test_run = st.checkbox("Test Run (first 5 files only)", value=False, key="test_run_compare_stages")
        with col2:
            recompute_cache = st.button("Recompute Cache", key="recompute_cache_compare_stages")
    else:
        test_run = True
        recompute_cache = False
    st.info("Scanning for patients with valid data (passed R-peaks test) across ALL sleep stages... (this may take a moment if not cached)")
    progress_scan = st.progress(0)
    
    valid_patients_per_stage = []
    # Cache all loaded individuals per stage for quick access later
    all_stage_individuals = {}
    
    # ── Load globally excluded patients ─────────────────────────────────────
    csv_path = os.path.join(base_path, "excluded_patients.csv")
    global_excluded_pids = []
    if os.path.exists(csv_path):
        try:
            global_excluded_df = pd.read_csv(csv_path)
            if 'patient_id' in global_excluded_df.columns:
                global_excluded_pids = [str(pid) for pid in global_excluded_df['patient_id'].tolist()]
        except Exception as e:
            st.warning(f"Failed to load excluded patients CSV: {e}")
            
    unfiltered_patients_per_stage = []  # for display table (includes excluded)
    for idx, stage in enumerate(sleep_stages):
        # We use get_group_individuals to reliably find valid files that passed processing
        stage_individuals = get_group_individuals(selected_group, stage, base_path, test_run=test_run, recompute_cache=recompute_cache)

        # Track unfiltered patients (for the availability table)
        unfiltered_stage_patients = set()
        for ind in stage_individuals:
            pid_full = ind[0]
            pid_base = pid_full.split('_')[0]
            unfiltered_stage_patients.add(pid_base)
        unfiltered_patients_per_stage.append(unfiltered_stage_patients)

        # Filter out globally excluded patients
        inds_filtered = []
        for ind in stage_individuals:
            pid = str(ind[0])
            base_pid = pid.split('_')[0] if '_' in pid else pid
            if base_pid not in global_excluded_pids and pid not in global_excluded_pids:
                inds_filtered.append(ind)

        stage_individuals = inds_filtered
        all_stage_individuals[stage] = stage_individuals

        stage_patients = set()
        for ind in stage_individuals:
            pid_full = ind[0]
            pid_base = pid_full.split('_')[0]
            stage_patients.add(pid_base)
        valid_patients_per_stage.append(stage_patients)
        progress_scan.progress((idx + 1) / len(sleep_stages))
        
    progress_scan.empty()
    
    if valid_patients_per_stage:
        all_patients = set.intersection(*valid_patients_per_stage)
        union_all_patients = set.union(*valid_patients_per_stage)
    else:
        all_patients = set()
        union_all_patients = set()

    # Expandable container showing all patients and their available stages
    union_all_patients_unfiltered = set.union(*unfiltered_patients_per_stage) if unfiltered_patients_per_stage else set()
    if union_all_patients_unfiltered:
        excluded_set = set(global_excluded_pids)

        with st.expander("Patient Sleep Stage Availability", expanded=False):
            st.markdown("This table details which sleep stages are available for each patient in this group.")
            stage_avail_data = []
            for pid in sorted(list(union_all_patients_unfiltered)):
                row = {'Patient ID': pid}
                n_available = 0
                for stage, stage_patients in zip(sleep_stages, unfiltered_patients_per_stage):
                    has_stage = pid in stage_patients
                    row[stage] = "Y" if has_stage else "N"
                    if has_stage:
                        n_available += 1
                row['Total Stages'] = n_available
                row['Excluded'] = "Y" if pid in excluded_set else "N"
                stage_avail_data.append(row)

            df_availability = pd.DataFrame(stage_avail_data).set_index('Patient ID')

            def color_rows(row):
                total = row['Total Stages']
                if total == len(sleep_stages):
                    bg_color = 'background-color: rgba(0, 255, 0, 0.2)'  # Green for all
                elif total == 0:
                    bg_color = 'background-color: rgba(255, 0, 0, 0.4)'  # Red for none
                elif total <= 2:
                    bg_color = 'background-color: rgba(255, 100, 0, 0.3)' # Orange for few
                else:
                    bg_color = 'background-color: rgba(255, 255, 0, 0.2)' # Yellow for some
                styles = [bg_color] * len(row)
                if row.get('Excluded') == "Y":
                    excluded_col_idx = row.index.get_loc('Excluded')
                    styles[excluded_col_idx] = 'background-color: rgba(220, 38, 38, 0.5); color: white; font-weight: bold'
                return styles

            st.dataframe(df_availability.style.apply(color_rows, axis=1), use_container_width=True)

    if not all_patients:
        st.warning(f"No patients found in group {selected_group} with valid data in ALL stages.")
        return

    # 3. Choose Mode
    analysis_mode = st.radio("Analysis Type", ["Single Patient", "Group Average"], horizontal=True)

    if analysis_mode == "Single Patient":
        selected_pid = st.selectbox("Select Patient", sorted(list(all_patients)))
    else:
        selected_pid = "Group Average"

    st.subheader(f"Comparisons across Sleep Stages - {selected_pid} - {selected_group}")
    
    stage_data = {}
    
    # Define colors for stages
    stage_colors = {
        'W': 'orange',
        'N1': 'yellow',
        'N2': 'lightblue',
        'N3': 'blue',
        'R': 'red'
    }
    
    for stage in sleep_stages:
        stage_individuals = all_stage_individuals[stage]
        
        if analysis_mode == "Single Patient":
            for ind in stage_individuals:
                pid_full = ind[0]
                pid_base = pid_full.split('_')[0]
                if pid_base == selected_pid:
                    stage_data[stage] = {
                        'ecg_hep': ind[5],
                        'eeg_hep': ind[1],
                        'times': ind[2],
                        'ch_names': ind[3],
                        'n_epochs': len(ind[4]),
                        'n_subjects': 1
                    }
                    break
        else:
            # Group Average over all completely valid patients
            ecg_heps = []
            eeg_heps_list = []
            n_epochs_list = []
            times = None
            
            # Find common channels for this stage across all valid patients
            common_channels_stage = None
            valid_inds_dict = {}
            for ind in stage_individuals:
                pid_full = ind[0]
                pid_base = pid_full.split('_')[0]
                if pid_base in all_patients:
                    valid_inds_dict[pid_base] = ind
                    if common_channels_stage is None:
                        common_channels_stage = set(ind[3])
                    else:
                        common_channels_stage = common_channels_stage.intersection(set(ind[3]))
                        
            # Ensure valid_inds are ordered consistently by pid_base for pairwise tests
            valid_inds = [valid_inds_dict[p] for p in sorted(list(all_patients))]
                        
            if common_channels_stage:
                common_channels_stage = sorted(list(common_channels_stage))
                # Optional regex to filter EEG channels
                common_channels_stage = [ch for ch in common_channels_stage if re.match(r'^[a-zA-Z][0-9]*$', ch) or re.match(r'^[a-zA-Z]z$', ch)]
            else:
                common_channels_stage = []

            for ind in valid_inds:
                if ind[5] is not None:
                    ecg_heps.append(ind[5])
                if ind[1] is not None:
                    eeg_data = ind[1]
                    ch_names_ind = ind[3]
                    aligned_eeg = []
                    for ch in common_channels_stage:
                        ch_idx = ch_names_ind.index(ch)
                        aligned_eeg.append(eeg_data[ch_idx])
                    if aligned_eeg:
                        eeg_heps_list.append(aligned_eeg)
                n_epochs_list.append(len(ind[4]))
                if times is None:
                    times = ind[2]
                    
            if ecg_heps and eeg_heps_list:
                # Ensure all arrays in ecg_heps have the same shape
                # Sometimes different sampling rates cause slight length mismatches
                min_len_ecg = min(len(x[0]) if len(x.shape) > 1 else len(x) for x in ecg_heps)
                ecg_heps_aligned = [x[:, :min_len_ecg] if len(x.shape) > 1 else x[:min_len_ecg] for x in ecg_heps]
                
                min_len_eeg = min(len(x[0]) for x in eeg_heps_list)
                eeg_heps_aligned = [[ch_arr[:min_len_eeg] for ch_arr in patient_eeg] for patient_eeg in eeg_heps_list]
                
                times = times[:min_len_eeg]

                avg_ecg = np.nanmean(ecg_heps_aligned, axis=0)
                avg_eeg = np.nanmean(eeg_heps_aligned, axis=0)
                avg_epochs = int(np.mean(n_epochs_list))
                
                stage_data[stage] = {
                    'ecg_hep': avg_ecg,
                    'eeg_hep': avg_eeg,
                    'times': times,
                    'ch_names': common_channels_stage,
                    'n_epochs': avg_epochs,
                    'n_subjects': len(ecg_heps),
                    'ecg_heps_aligned': np.array(ecg_heps_aligned),
                    'eeg_heps_aligned': np.array(eeg_heps_aligned)
                }
                
    if not stage_data:
        st.warning("No data found for any stage.")
        return

    # --- Plot ECG ---
    fig_ecg, ax_ecg = plt.subplots(figsize=(10, 6))
    has_ecg_data = False
    
    for stage, data in stage_data.items():
        ecg_hep = data['ecg_hep']
        times = data['times']
        
        if ecg_hep is not None:
             # ecg_hep is (1, n_times)
             n_epochs = data['n_epochs']
             label_str = f"{stage} (n={n_epochs})" if analysis_mode == "Single Patient" else f"{stage} (N={data['n_subjects']}, avg epochs={n_epochs})"
             ax_ecg.plot(times, ecg_hep[0] * 1e6, label=label_str, color=stage_colors.get(stage, 'gray'), linewidth=2, alpha=0.8)
             has_ecg_data = True
             
    if has_ecg_data:
        title_str = f"ECG HEP across Sleep Stages - {selected_pid}" if analysis_mode == "Single Patient" else f"ECG HEP across Sleep Stages - Group Average (N={len(all_patients)} patients)"
        ax_ecg.set_title(title_str)
        ax_ecg.set_xlabel("Time (s)")
        ax_ecg.set_ylabel("Amplitude (μV)")
        ax_ecg.grid(True)
        ax_ecg.legend()
        ax_ecg.axvline(0, color='black', linestyle='--', alpha=0.5)
        st.pyplot(fig_ecg, use_container_width=True)

        # Add Pairwise Comparison Matrix for Group Average
        if analysis_mode == "Group Average" and len(stage_data) > 1:
            st.markdown("#### Pairwise Significance (ECG HEP)")
            stages_list = list(stage_data.keys())
            
            # Use a standard T-test + FDR for pairwise matrix to get exact p-values for all pairs
            try:
                from mne.stats import fdr_correction
                HAS_FDR = True
            except ImportError:
                HAS_FDR = False

            # Build p-value lookup dict for all pairs
            pair_pvals = {}  # (stage_a, stage_b) -> (min_p, is_significant, windows_str)
            pair_sig_clusters = {} # (stage_a, stage_b) -> [list of sub-arrays of sig_times]
            for i in range(len(stages_list)):
                for j in range(i + 1, len(stages_list)):
                    stage_a = stages_list[i]
                    stage_b = stages_list[j]
                    heps_a = stage_data[stage_a].get('ecg_heps_aligned')
                    heps_b = stage_data[stage_b].get('ecg_heps_aligned')
                    
                    if heps_a is not None and heps_b is not None and len(heps_a) > 1 and len(heps_b) > 1:
                        min_len = min(heps_a.shape[-1], heps_b.shape[-1])
                        val_a = heps_a[:, 0, :min_len] if len(heps_a.shape) == 3 else heps_a[:, :min_len]
                        val_b = heps_b[:, 0, :min_len] if len(heps_b.shape) == 3 else heps_b[:, :min_len]
                        
                        # We use a simple t-test at each time point
                        t_stat, p_vals = stats.ttest_ind(val_a, val_b, axis=0, equal_var=False)
                        
                        if HAS_FDR:
                            reject, p_vals_corrected_fdr = fdr_correction(p_vals, alpha=0.05, method='indep')
                            p_vals = p_vals_corrected_fdr
                            
                        min_p = np.nanmin(p_vals)
                        is_sig = min_p < 0.05
                        
                        win_strs = ""
                        clusters = []
                        if is_sig:
                            sig_times = stage_data[stage_a]['times'][:min_len][p_vals < 0.05] * 1000
                            if len(sig_times) > 0:
                                # Find continuous clusters of significance
                                split_indices = np.where(np.diff(sig_times) > 5)[0] + 1 
                                clusters = np.split(sig_times, split_indices)
                                win_strs = ", ".join([f"{c[0]:.0f}–{c[-1]:.0f}ms" for c in clusters if len(c) > 1])

                        pair_pvals[(stage_a, stage_b)] = (min_p, is_sig, win_strs)
                        pair_sig_clusters[(stage_a, stage_b)] = clusters
                    else:
                        pair_pvals[(stage_a, stage_b)] = (None, None, "")  # no data
                        pair_sig_clusters[(stage_a, stage_b)] = []

            # ── Render as HTML matrix table ──────────────────────────────────
            n = len(stages_list)
            cmap = plt.get_cmap('Reds_r')

            # Color scheme
            COLOR_NONSIG = "#f0f0f0"   # gray   – not significant
            COLOR_NODATA = "#ffffff"   # white  – no data (missing)
            COLOR_DIAG   = "#d9d9d9"   # darker gray – diagonal

            html = "<style>table.pairwise{border-collapse:collapse;font-family:monospace;font-size:13px}" \
                   "table.pairwise td,table.pairwise th{border:1px solid #bbb;padding:6px 10px;text-align:center;min-width:90px}" \
                   "table.pairwise th{background:#444;color:white;font-weight:bold}" \
                   "</style><table class='pairwise'>"

            # Header row
            html += "<tr><th></th>"
            for s in stages_list:
                html += f"<th>{s}</th>"
            html += "</tr>"

            for row_s in stages_list:
                html += f"<tr><th>{row_s}</th>"
                for col_s in stages_list:
                    if row_s == col_s:
                        html += f"<td style='background:{COLOR_DIAG};color:#000000;font-weight:bold'>{row_s}</td>"
                    else:
                        # Canonical key order (always smaller index first)
                        key = (row_s, col_s) if (row_s, col_s) in pair_pvals else (col_s, row_s)
                        if key in pair_pvals:
                            min_p, is_sig, wins = pair_pvals[key]
                            if is_sig is None:
                                bg = COLOR_NODATA
                                text_c = "#000000"
                                cell_text = "n/a"
                                title = "No data"
                            else:
                                rgba = cmap(min_p / 0.05) if min_p <= 0.05 else mcolors.to_rgba(COLOR_NONSIG)
                                bg = mcolors.to_hex(rgba)
                                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                                text_c = "#ffffff" if luminance < 0.5 else "#000000"
                                
                                # Show Exact P-Value instead of 'n.s.'
                                p_text = f"p = {min_p:.4f}" if min_p >= 0.0001 else "p < 1e-4"
                                cell_text = f"<b>{p_text}</b><br><small>{wins}</small>" if is_sig else f"{p_text}"
                                title = f"p={min_p:.4f}"
                        else:
                            bg = COLOR_NODATA
                            text_c = "#000000"
                            cell_text = "n/a"
                            title = "No data"
                        html += f"<td style='background:{bg};color:{text_c}' title='{title}'>{cell_text}</td>"
                html += "</tr>"
            html += "</table>"
            st.markdown(html, unsafe_allow_html=True)

            # --- Plot Significant Windows ---
            sig_pairs_to_plot = {k: v for k, v in pair_sig_clusters.items() if len(v) > 0 and any(len(c) > 1 for c in v)}
            if sig_pairs_to_plot:
                st.markdown("#### Significant Difference Windows (ECG HEP)")
                fig_win, ax_win = plt.subplots(figsize=(10, max(3, len(sig_pairs_to_plot) * 0.5)))
                
                y_labels = []
                y_ticks = []
                
                for idx, (pair, clusters) in enumerate(sig_pairs_to_plot.items()):
                    y_pos = len(sig_pairs_to_plot) - idx
                    y_labels.append(f"{pair[0]} vs {pair[1]}")
                    y_ticks.append(y_pos)
                    
                    for cluster in clusters:
                        if len(cluster) > 1:
                            start_time = cluster[0] / 1000.0  # convert back to seconds
                            end_time = cluster[-1] / 1000.0
                            ax_win.plot([start_time, end_time], [y_pos, y_pos], color='black', linewidth=4, solid_capstyle='butt')
                            
                ax_win.set_yticks(y_ticks)
                ax_win.set_yticklabels(y_labels)
                ax_win.set_xlabel("Time (s)")
                ax_win.set_title("Significant Difference Windows (p < 0.05)")
                ax_win.grid(True, axis='x', alpha=0.3)
                ax_win.axvline(0, color='r', linestyle='--', alpha=0.5)
                ax_win.set_xlim(ax_ecg.get_xlim())
                fig_win.tight_layout()
                st.pyplot(fig_win, use_container_width=True)
                plt.close(fig_win)
    else:
        st.write("No ECG data available.")

    # --- Plot EEG by Regions ---
    # Determine all unique channels available (use first stage as reference, or intersection?)
    # Usually montage is constant. Let's use the first available stage.
    first_stage = next(iter(stage_data))
    all_channels = stage_data[first_stage]['ch_names']
    
    if not all_channels:
        st.warning("No EEG channels found.")
        return

    regions = ['Average of All Electrodes', 'F', 'C', 'T', 'P']
    region_map = {r: [] for r in regions}
    
    # All channels belong to the average
    region_map['Average of All Electrodes'] = all_channels

    for ch in all_channels:
        name = ch.upper()
        # Simple categorization heuristic
        if name.startswith('F'):
            region_map['F'].append(ch)
        elif name.startswith('C'):
            region_map['C'].append(ch)
        elif name.startswith('T'):
            region_map['T'].append(ch)
        elif name.startswith('P'):
            region_map['P'].append(ch)
            
    for region in regions:
        channels = region_map[region]
        if not channels:
            continue
            
        st.subheader(f"Region: {region} ({len(channels)} channels)")
        
        # Calculate Average for the entire region if it's the 'Average of All Electrodes'
        if region == 'Average of All Electrodes':
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Determine symmetric limits for the average across all stages
            max_abs_val = 0
            for stage, data in stage_data.items():
                if data['eeg_hep'] is None:
                    continue
                try:
                    indices = [data['ch_names'].index(ch) for ch in channels if ch in data['ch_names']]
                    if indices:
                        # calculate average across these channels
                        region_avg = np.nanmean(data['eeg_hep'][indices], axis=0) * 1e6
                        curr_max = np.nanmax(np.abs(region_avg))
                        if curr_max > max_abs_val:
                            max_abs_val = curr_max
                except Exception:
                    pass
            ylim = (-max_abs_val * 1.1, max_abs_val * 1.1) if max_abs_val > 0 else None
            
            for stage, data in stage_data.items():
                if data['eeg_hep'] is None:
                    continue
                indices = [data['ch_names'].index(ch) for ch in channels if ch in data['ch_names']]
                if indices:
                    times = data['times']
                    if analysis_mode == "Group Average" and 'eeg_heps_aligned' in data and data['eeg_heps_aligned'] is not None:
                        subj_data = np.nanmean(data['eeg_heps_aligned'][:, indices, :], axis=1) * 1e6
                        mad = np.nanmedian(np.abs(subj_data - np.nanmedian(subj_data, axis=0)), axis=0)
                        color = stage_colors.get(stage, 'gray')
                        region_avg = np.nanmean(data['eeg_hep'][indices], axis=0) * 1e6
                        ax.fill_between(times, region_avg - mad, region_avg + mad, color=color, alpha=0.2)
                        ax.plot(times, region_avg, label=stage, color=color, linewidth=2, alpha=0.9)
                    else:
                        region_avg = np.nanmean(data['eeg_hep'][indices], axis=0) * 1e6
                        ax.plot(times, region_avg, label=stage, color=stage_colors.get(stage, 'gray'), linewidth=2, alpha=0.9)
            
            ax.set_title("Average of All Electrodes")
            ax.grid(True, alpha=0.3)
            ax.axvline(0, color='r', linestyle='--', alpha=0.5)
            if ylim:
                ax.set_ylim(ylim)
            ax.set_ylabel("Amplitude (µV)")
            ax.set_xlabel("Time (s)")
            ax.legend(fontsize='small')
            
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            continue

        
        # Create subplots
        n_channels = len(channels)
        n_cols = 4
        n_rows = int(np.ceil(n_channels / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), sharex=True, sharey=True)
        if n_channels == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        # Determine symmetric limits for this region across all stages
        max_abs_val = 0
        for stage, data in stage_data.items():
            if data['eeg_hep'] is None:
                continue
            
            # Get indices for these channels
            # Note: ch_names in stage_data might differ if montage changed? Assuming constant for now.
            try:
                indices = [data['ch_names'].index(ch) for ch in channels if ch in data['ch_names']]
                if indices:
                    region_data = data['eeg_hep'][indices]
                    curr_max = np.nanmax(np.abs(region_data * 1e6))
                    if curr_max > max_abs_val:
                        max_abs_val = curr_max
            except Exception:
                pass
                
        ylim = (-max_abs_val * 1.1, max_abs_val * 1.1) if max_abs_val > 0 else None

        for i, ch_name in enumerate(channels):
            ax = axes[i]
            
            for stage, data in stage_data.items():
                if data['eeg_hep'] is None:
                    continue
                
                if ch_name in data['ch_names']:
                    ch_idx = data['ch_names'].index(ch_name)
                    times = data['times']
                    
                    if analysis_mode == "Group Average" and 'eeg_heps_aligned' in data and data['eeg_heps_aligned'] is not None:
                        subj_data = data['eeg_heps_aligned'][:, ch_idx, :] * 1e6
                        mad = np.nanmedian(np.abs(subj_data - np.nanmedian(subj_data, axis=0)), axis=0)
                        color = stage_colors.get(stage, 'gray')
                        hep = data['eeg_hep'][ch_idx] * 1e6
                        ax.fill_between(times, hep - mad, hep + mad, color=color, alpha=0.2)
                        ax.plot(times, hep, label=stage, color=color, linewidth=1.5, alpha=0.8)
                    else:
                        hep = data['eeg_hep'][ch_idx]
                        ax.plot(times, hep * 1e6, label=stage, color=stage_colors.get(stage, 'gray'), linewidth=1.5, alpha=0.8)
            
            ax.set_title(ch_name)
            ax.grid(True, alpha=0.3)
            ax.axvline(0, color='r', linestyle='--', alpha=0.5)
            if ylim:
                ax.set_ylim(ylim)
            
            # Y-axis label on the leftmost column
            if i % n_cols == 0:
                ax.set_ylabel("Amplitude (µV)")
            # X-axis label on the bottom row
            if i >= n_channels - n_cols:
                ax.set_xlabel("Time (s)")
            
            # Only legend on first plot to avoid clutter
            if i == 0:
                ax.legend(fontsize='small')
                
        # Hide unused
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # --- Plot Spatial Topomaps ---
    st.divider()
    st.subheader(f"Spatial Topomaps")
    st.markdown("Select a time window to visualize the average HEP amplitude distribution, PSD, and (for Group Average) spatial significance across the scalp.")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        topo_tmin = st.number_input("Start Time (ms)", min_value=-500, max_value=1000, value=200, step=10, key="topo_tmin_compare")
    with col_t2:
        topo_tmax = st.number_input("End Time (ms)", min_value=-500, max_value=1000, value=400, step=10, key="topo_tmax_compare")
        
    if topo_tmin >= topo_tmax:
        st.warning("Start time must be less than end time.")
    else:
        try:
            from mne.time_frequency import psd_array_welch
            montage = mne.channels.make_standard_montage('standard_1020')
            montage_ch_names_upper = [ch.upper() for ch in montage.ch_names]
            
            # Pre-compute global min/max for the same color scale
            global_max_abs = 0.0
            global_psd_max = 0.0
            stage_plot_data = {}
            valid_times = False
            
            for stage, data in stage_data.items():
                if data['eeg_hep'] is None:
                    continue
                times_ms = data['times'] * 1000
                t_mask = (times_ms >= topo_tmin) & (times_ms <= topo_tmax)
                if not np.any(t_mask):
                    continue
                valid_times = True
                
                # 1. Mean Amplitude
                mean_amps = np.nanmean(data['eeg_hep'][:, t_mask], axis=1) * 1e6
                
                # 2. PSD (Total Power)
                hep_windowed = data['eeg_hep'][:, t_mask]
                sfreq = 1000.0 / np.mean(np.diff(times_ms)) if len(times_ms) > 1 else 250.0
                try:
                    n_fft = min(256, hep_windowed.shape[1])
                    if n_fft > 0:
                        psds, freqs = psd_array_welch(hep_windowed, sfreq=sfreq, fmin=0.5, fmax=40.0, n_fft=n_fft, verbose=False)
                        psd_total = np.sum(psds, axis=1)
                    else:
                        psd_total = np.zeros(hep_windowed.shape[0])
                except Exception:
                    psd_total = np.zeros(hep_windowed.shape[0])
                    
                # 3. Jitter P-value (Group Average only)
                p_values = None
                if analysis_mode == "Group Average" and data.get('eeg_heps_aligned') is not None:
                    p_values = []
                    eeg_aligned = data.get('eeg_heps_aligned')
                    eeg_win = eeg_aligned[:, :, t_mask]
                    times_win = data['times'][t_mask]
                    
                    for ch_idx in range(eeg_win.shape[1]):
                        channel_data = eeg_win[:, ch_idx, :]
                        sig_windows, _, _pt = permutation_cluster_jitter_test(channel_data, times_win, n_permutations=100, p_threshold=0.05, jitter_sec=0.1)
                        if sig_windows:
                            min_p = min([w['p_value'] for w in sig_windows])
                            p_values.append(min_p)
                        else:
                            p_values.append(1.0)  # Non-significant (will be plotted as white)

                plot_ch_names = []
                plot_data_amp = []
                plot_data_psd = []
                plot_data_pval = [] if p_values is not None else None
                
                for i, ch in enumerate(data['ch_names']):
                    ch_upper = ch.upper()
                    if ch_upper in montage_ch_names_upper:
                        m_idx = montage_ch_names_upper.index(ch_upper)
                        standard_name = montage.ch_names[m_idx]
                        plot_ch_names.append(standard_name)
                        plot_data_amp.append(mean_amps[i])
                        plot_data_psd.append(psd_total[i])
                        if plot_data_pval is not None:
                            plot_data_pval.append(p_values[i])
                        
                # Pad missing 10-20 standard channels to keep head shape normal
                standard_19_base = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2']
                aliases = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}
                
                def pad_channel_if_missing(ch_name, pad_amp, pad_psd, pad_pval):
                    if not any(ch_name.upper() == p_ch.upper() for p_ch in plot_ch_names):
                        plot_ch_names.append(ch_name)
                        plot_data_amp.append(pad_amp)
                        plot_data_psd.append(pad_psd)
                        if plot_data_pval is not None:
                            plot_data_pval.append(pad_pval)
                
                for base_ch in standard_19_base:
                    pad_channel_if_missing(base_ch, 0.0, 0.0, 0.05)
                        
                for new_name, old_name in aliases.items():
                    if not any(ch.upper() in [new_name.upper(), old_name.upper()] for ch in plot_ch_names):
                        pad_channel_if_missing(new_name, 0.0, 0.0, 0.05)

                if plot_data_amp:
                    curr_max = np.max(np.abs(plot_data_amp))
                    if curr_max > global_max_abs:
                        global_max_abs = curr_max
                if plot_data_psd:
                    curr_psd_max = np.max(plot_data_psd)
                    if curr_psd_max > global_psd_max:
                        global_psd_max = curr_psd_max
                
                stage_plot_data[stage] = {
                    'names': plot_ch_names,
                    'amp': plot_data_amp,
                    'psd': plot_data_psd,
                    'pval': plot_data_pval
                }
                
            if not valid_times:
                st.warning("Invalid time window selected (no data points).")
            elif not stage_plot_data:
                st.warning("No channels matched the standard montage.")
            else:
                n_stages = len(stage_plot_data)
                stages_list = list(stage_plot_data.keys())
                
                vmin_amp, vmax_amp = -global_max_abs, global_max_abs
                if vmax_amp == 0: vmax_amp, vmin_amp = 1, -1
                
                vmin_psd, vmax_psd = 0, global_psd_max
                if vmax_psd == 0: vmax_psd = 1

                # Helper: build info object for a channel list
                def make_info(names):
                    info = mne.create_info(ch_names=names, sfreq=250., ch_types='eeg')
                    info.set_montage(montage)
                    return info

                # ── Figure 1: Mean Amplitude per stage ──────────────────────
                st.markdown("#### Mean HEP Amplitude")
                fig_amp, axes_amp = plt.subplots(1, n_stages, figsize=(4 * n_stages + 1, 4))
                if n_stages == 1: axes_amp = [axes_amp]
                im_amp = None
                
                # Prepare summary stats
                amp_stats = []
                
                for idx, stage in enumerate(stages_list):
                    p = stage_plot_data[stage]
                    info_ = make_info(p['names'])
                    r = mne.viz.plot_topomap(np.array(p['amp']), info_, axes=axes_amp[idx],
                                             cmap='RdBu_r', names=p['names'],
                                             vlim=(vmin_amp, vmax_amp), extrapolate='head', show=False)
                    im_amp = r[0] if isinstance(r, tuple) else r
                    axes_amp[idx].set_title(stage)
                    
                    # Compute stats
                    stage_n = stage_data[stage]['n_subjects']
                    amp_vals = np.array(p['amp'])
                    amp_stats.append({
                        'Stage': stage,
                        'N': stage_n,
                        'Min (µV)': np.min(amp_vals),
                        'Max (µV)': np.max(amp_vals),
                        'Std (µV)': np.std(amp_vals)
                    })
                    
                fig_amp.subplots_adjust(right=0.88)
                if im_amp is not None:
                    cbar_ax = fig_amp.add_axes([0.91, 0.15, 0.02, 0.7])
                    fig_amp.colorbar(im_amp, cax=cbar_ax).set_label("Mean Amplitude (µV)")
                st.pyplot(fig_amp, use_container_width=False)
                plt.close(fig_amp)
                
                # Display Stats & Expandable Table for Amp
                st.markdown("**Summary Statistics (Amplitude)**")
                for stat in amp_stats:
                    st.write(f"- **{stat['Stage']}** (N={stat['N']}): Min={stat['Min (µV)']:.3f}, Max={stat['Max (µV)']:.3f}, SD={stat['Std (µV)']:.3f}")
                with st.expander("Raw Amplitude Values by Channel"):
                    amp_df = pd.DataFrame({stage: stage_plot_data[stage]['amp'] for stage in stages_list}, index=stage_plot_data[stages_list[0]]['names'])
                    st.dataframe(amp_df.style.format("{:.3f}"))

                # ── Figure 2: P-value per stage (Group Average only) ─────────
                if analysis_mode == "Group Average":
                    has_pvals = any(stage_plot_data[s]['pval'] is not None for s in stages_list)
                    if has_pvals:
                        st.markdown("#### Spatial Significance (Min. Cluster P-value per Stage)")
                        fig_pval, axes_pval = plt.subplots(1, n_stages, figsize=(4 * n_stages + 1, 4))
                        if n_stages == 1: axes_pval = [axes_pval]
                        im_pval = None
                        
                        pval_stats = []
                        
                        for idx, stage in enumerate(stages_list):
                            p = stage_plot_data[stage]
                            if p['pval'] is not None:
                                info_ = make_info(p['names'])
                                data_pv = np.clip(np.array(p['pval']), 0, 0.05)
                                r = mne.viz.plot_topomap(data_pv, info_, axes=axes_pval[idx],
                                                         cmap='Reds_r', names=p['names'],
                                                         vlim=(0, 0.05), extrapolate='head', show=False)
                                im_pval = r[0] if isinstance(r, tuple) else r
                                
                                pval_vals = np.array(p['pval'])
                                pval_stats.append({
                                    'Stage': stage,
                                    'N': stage_data[stage]['n_subjects'],
                                    'Min P': np.min(pval_vals),
                                    'Max P': np.max(pval_vals),
                                    'Sig Channels Count': np.sum(pval_vals < 0.05)
                                })
                            else:
                                axes_pval[idx].axis('off')
                            axes_pval[idx].set_title(stage)
                        fig_pval.subplots_adjust(right=0.88)
                        if im_pval is not None:
                            cbar_ax = fig_pval.add_axes([0.91, 0.15, 0.02, 0.7])
                            fig_pval.colorbar(im_pval, cax=cbar_ax).set_label("p-value")
                        st.pyplot(fig_pval, use_container_width=False)
                        plt.close(fig_pval)
                        
                        # Display Stats & Expandable Table for P-vals
                        if pval_stats:
                            st.markdown("**Summary Statistics (P-values)**")
                            for stat in pval_stats:
                                st.write(f"- **{stat['Stage']}** (N={stat['N']}): Min={stat['Min P']:.4f}, Max={stat['Max P']:.4f}, Significant Channels (p<0.05)={stat['Sig Channels Count']}")
                            with st.expander("Raw P-Values by Channel"):
                                p_columns = {stage: stage_plot_data[stage]['pval'] for stage in stages_list if stage_plot_data[stage]['pval'] is not None}
                                if p_columns:
                                    pval_df = pd.DataFrame(p_columns, index=stage_plot_data[stages_list[0]]['names'])
                                    st.dataframe(pval_df.style.format("{:.4f}"))

                # ── Figure 3: Pairwise stage differences ─────────────────────
                pairs = [(stages_list[i], stages_list[j])
                         for i in range(len(stages_list))
                         for j in range(i + 1, len(stages_list))]

                if pairs:
                    st.markdown("#### Pairwise Stage Difference (Mean Amplitude, µV)")
                    n_pairs = len(pairs)
                    n_cols = min(3, n_pairs)
                    n_rows = int(np.ceil(n_pairs / n_cols))
                    fig_diff, axes_diff = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols + 1, 4 * n_rows))
                    if n_pairs == 1: axes_diff = [axes_diff]
                    else: axes_diff = axes_diff.flatten()

                    # Compute all diff arrays to find global max for a shared scale
                    diff_data_list = []
                    pair_names_list = []
                    for (s_a, s_b) in pairs:
                        p_a = stage_plot_data[s_a]
                        p_b = stage_plot_data[s_b]
                        # Match channels — use the union of names (pad missing with 0)
                        all_names_pair = list(dict.fromkeys(p_a['names'] + p_b['names']))
                        amp_a_map = dict(zip(p_a['names'], p_a['amp']))
                        amp_b_map = dict(zip(p_b['names'], p_b['amp']))
                        diff_arr = np.array([
                            amp_a_map.get(ch, 0.0) - amp_b_map.get(ch, 0.0)
                            for ch in all_names_pair
                        ])
                        diff_data_list.append((all_names_pair, diff_arr))
                        pair_names_list.append((s_a, s_b))

                    global_diff_max = max(np.max(np.abs(d)) for _, d in diff_data_list) or 1.0

                    im_diff = None
                    diff_stats = []
                    diff_df_dict = {}
                    
                    for idx, ((s_a, s_b), (pair_names, diff_arr)) in enumerate(
                            zip(pair_names_list, diff_data_list)):
                        info_ = make_info(pair_names)
                        r = mne.viz.plot_topomap(diff_arr, info_, axes=axes_diff[idx],
                                                 cmap='RdBu_r', names=pair_names,
                                                 vlim=(-global_diff_max, global_diff_max),
                                                 extrapolate='head', show=False)
                        im_diff = r[0] if isinstance(r, tuple) else r
                        label_ab = f"{s_a} − {s_b}"
                        axes_diff[idx].set_title(label_ab)
                        
                        diff_stats.append({
                            'Pair': label_ab,
                            'Min Diff': np.min(diff_arr),
                            'Max Diff': np.max(diff_arr),
                            'Std Diff': np.std(diff_arr)
                        })
                        diff_df_dict[label_ab] = dict(zip(pair_names, diff_arr))

                    # Hide unused axes
                    for j in range(len(pair_names_list), len(axes_diff)):
                        axes_diff[j].axis('off')

                    fig_diff.subplots_adjust(right=0.88)
                    if im_diff is not None:
                        cbar_ax = fig_diff.add_axes([0.91, 0.15, 0.02, 0.7])
                        fig_diff.colorbar(im_diff, cax=cbar_ax).set_label("Difference (µV)")
                    st.pyplot(fig_diff, use_container_width=False)
                    plt.close(fig_diff)
                    
                    # Display Stats & Expandable Table for Pairwise Diff
                    st.markdown("**Summary Statistics (Pairwise Differences)**")
                    for stat in diff_stats:
                        st.write(f"- **{stat['Pair']}**: Min={stat['Min Diff']:.3f}, Max={stat['Max Diff']:.3f}, SD={stat['Std Diff']:.3f}")
                    with st.expander("Raw Pairwise Differences by Channel"):
                        # Ensure we build a dataframe across all possible channel names
                        diff_df = pd.DataFrame(diff_df_dict)
                        st.dataframe(diff_df.style.format("{:.3f}"))
                
        except Exception as e:
            st.error(f"Error generating topomaps: {str(e)}")

def run_edf_viewer_mode(base_path_edf):
    """
    Logic for EDF Viewer mode.
    Allows user to select a folder and view aggregate statistics about EDF files.
    """
    st.subheader("EDF Viewer Mode")

    # 1. Folder Selection
    if not os.path.exists(base_path_edf):
        st.error(f"EDF Base Path not found: {base_path_edf}")
        return

    # specific logic for 'EDF_Format' structure: it might have subfolders
    # We want to list directories in base_path_edf
    try:
        subdirs = [d for d in os.listdir(base_path_edf) if os.path.isdir(os.path.join(base_path_edf, d))]
        subdirs.sort()
    except Exception as e:
        st.error(f"Error listing directories in {base_path_edf}: {e}")
        return

    if not subdirs:
        st.warning(f"No subdirectories found in {base_path_edf}")
        return

    selected_folder = st.selectbox("Select Dataset Folder", subdirs)
    folder_path = os.path.join(base_path_edf, selected_folder)

    # 2. Scanning Files
    st.write(f"Scanning files in `{folder_path}`...")
    
    # Find all .edf files (case insensitive)
    edf_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.edf'):
                edf_files.append(os.path.join(root, file))
    
    if not edf_files:
        st.warning("No .edf files found in selected folder.")
        return

    st.write(f"Found {len(edf_files)} EDF files. Extracting metadata...")

    # 3. Extract Metadata
    metadata_list = []
    
    # Use a progress bar
    progress_bar = st.progress(0)
    
    for k, file_path in enumerate(edf_files):
        try:
            # Update progress
            progress_bar.progress((k + 1) / len(edf_files))
            
            # Read header only (preload=False)
            # Suppress MNE warnings for speed and cleanliness
            with mne.utils.use_log_level('ERROR'):
                 raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
            
            # basic info
            f_name = os.path.basename(file_path)
            # Try to guess patient ID: default to filename, or split by _/-
            # Heuristic: often "SubjectID_Condition.edf" or "ID-Date.edf"
            # We'll just take the first part before _ or -
            pid_guess = re.split(r'[_-]', f_name)[0]
            
            n_ch = len(raw.ch_names)
            sfreq = raw.info['sfreq']
            # raw.times is not reliable without loading data sometimes, or might be empty
            # raw.n_times / raw.info['sfreq'] is safer if n_times is populated from header
            dur = raw.n_times / sfreq if raw.n_times > 0 else 0
            
            meas_date = raw.info['meas_date']
            
            metadata_list.append({
                'Filename': f_name,
                'Path': file_path,
                'PatientID_Guess': pid_guess,
                'Duration_sec': dur,
                'Duration_min': dur / 60,
                'N_Channels': n_ch,
                'Sf_Hz': sfreq,
                'Meas_Date': meas_date,
                'Channel_Names': raw.ch_names
            })
            
            raw.close()
            
        except Exception as e:
            # Store error info?
            metadata_list.append({
                'Filename': os.path.basename(file_path),
                'Path': file_path,
                'Error': str(e)
            })
            
    progress_bar.empty()
    
    # Convert to DataFrame
    df_meta = pd.DataFrame(metadata_list)
    
    # Separate successful reads vs errors
    if 'Error' in df_meta.columns:
        df_errors = df_meta[df_meta['Error'].notna()]
        df_success = df_meta[df_meta['Error'].isna()]
    else:
        df_errors = pd.DataFrame()
        df_success = df_meta

    # 4. Display Statistics
    if not df_success.empty:
        st.success(f"Successfully read headers for {len(df_success)} files.")
        
        # --- Overview Metrics ---
        col1, col2, col3, col4 = st.columns(4)
        
        total_dur_hours = df_success['Duration_sec'].sum() / 3600
        avg_dur_min = df_success['Duration_min'].mean()
        n_unique_pids = df_success['PatientID_Guess'].nunique()
        n_total_files = len(df_success)
        
        col1.metric("Total Files", n_total_files)
        col2.metric("Unique Patients (Est)", n_unique_pids)
        col3.metric("Total Duration (hrs)", f"{total_dur_hours:.1f}")
        col4.metric("Avg Duration (min)", f"{avg_dur_min:.1f}")
        
        # --- Channels Analysis ---
        st.markdown("### Channel Analysis")
        
        # Collect all channels
        all_channels = []
        for ch_list in df_success['Channel_Names']:
            all_channels.extend(ch_list)
            
        unique_channels = sorted(list(set(all_channels)))
        
        # Count frequency of each channel
        from collections import Counter
        ch_counts = Counter(all_channels)
        
        # Prepare DataFrame for Channel Frequency
        df_ch_freq = pd.DataFrame.from_dict(ch_counts, orient='index', columns=['Count']).reset_index()
        df_ch_freq.rename(columns={'index': 'Channel'}, inplace=True)
        df_ch_freq['Percentage'] = (df_ch_freq['Count'] / n_total_files) * 100
        df_ch_freq.sort_values(by='Count', ascending=False, inplace=True)
        
        # Display
        st.write(f"**Total Unique Channels Found:** {len(unique_channels)}")
        
        # Tabs for different channel views
        tab1, tab2 = st.tabs(["Channel Frequency", "Common Channels (>90%)"])
        
        with tab1:
            st.dataframe(df_ch_freq, use_container_width=True)
            
        with tab2:
            common_chs = df_ch_freq[df_ch_freq['Percentage'] > 90]['Channel'].tolist()
            st.write(f"**Common Channels (present in >90% of files):** {len(common_chs)}")
            st.write(", ".join(common_chs))
            
        # --- Detailed Table ---
        st.markdown("### Detailed File Info")
        
        # Select columns to display
        disp_cols = ['Filename', 'PatientID_Guess', 'Duration_min', 'N_Channels', 'Sf_Hz', 'Meas_Date']
        st.dataframe(df_success[disp_cols], use_container_width=True)
        
        # Allow download of metadata
        csv = df_success.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Metadata CSV",
            csv,
            "edf_metadata.csv",
            "text/csv",
            key='download-csv'
        )

    # 5. Show Errors if any
    if not df_errors.empty:
        st.warning(f"Failed to read headers for {len(df_errors)} files.")
        st.dataframe(df_errors[['Filename', 'Path', 'Error']], use_container_width=True)

    # 6. Patient Histogram View
    st.divider()
    st.subheader("Patient Data Inspection (Histograms)")
    
    if not df_success.empty:
        # File selector
        selected_file_name = st.selectbox("Select Patient File", df_success['Filename'].tolist())
        
        # Get path for selected file
        selected_file_path = df_success[df_success['Filename'] == selected_file_name]['Path'].iloc[0]
        
        if st.button("Load and Plot Histograms"):
            try:
                with st.spinner(f"Loading data for {selected_file_name}..."):
                    # Load raw data
                    # Suppress warnings
                    with mne.utils.use_log_level('ERROR'):
                        raw = mne.io.read_raw_edf(selected_file_path, preload=True, verbose=False)
                    
                    data = raw.get_data() # (n_channels, n_times)
                    ch_names = raw.ch_names
                    n_ch = len(ch_names)
                    
                    # Layout for histograms
                    n_cols = 4
                    n_rows = int(np.ceil(n_ch / n_cols))
                    
                    fig_hist, axes_hist = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
                    if n_ch == 1:
                        axes_hist = [axes_hist]
                    else:
                        axes_hist = axes_hist.flatten()
                        
                    st.write(f"Plotting histograms for {n_ch} channels...")
                    
                    # Plot histograms
                    for i, ch_name in enumerate(ch_names):
                        ax = axes_hist[i]
                        ch_data = data[i, :]
                        
                        # Plot histogram
                        # Use a reasonable number of bins, e.g., 50 or 100
                        ax.hist(ch_data * 1e6, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
                        
                        ax.set_title(ch_name)
                        ax.set_xlabel("Amplitude (μV)")
                        ax.set_ylabel("Count")
                        ax.grid(True, alpha=0.3)
                        
                        # Add basic stats to title
                        mean_val = np.mean(ch_data * 1e6)
                        std_val = np.std(ch_data * 1e6)
                        ax.set_title(f"{ch_name}\nμ={mean_val:.1f}, σ={std_val:.1f}")

                    # Hide unused axes
                    for j in range(n_ch, len(axes_hist)):
                        axes_hist[j].axis('off')
                        
                    fig_hist.tight_layout()
                    st.pyplot(fig_hist, use_container_width=True)
                    
                    raw.close()

            except Exception as e:
                st.error(f"Error loading or plotting data: {e}")


def main():
    st.title("HEP Group Comparison Dashboard")
    st.write("Comparing Amplitude vs Time (Heartbeat Evoked Potential).")

    st.sidebar.header("Export Tools")
    if 'pptx_figures_data' in st.session_state and len(st.session_state.pptx_figures_data) > 0:
        st.sidebar.write(f"Figures saved: {len(st.session_state.pptx_figures_data)}")
        if st.sidebar.button("Prepare PowerPoint Report"):
            with st.spinner("Generating PowerPoint..."):
                pptx_io = generate_pptx()
                if pptx_io:
                    st.sidebar.download_button(
                        label="Download PPTX",
                        data=pptx_io,
                        file_name="HEP_Report.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    )
                else:
                    st.sidebar.error("python-pptx is not installed.")
        if st.sidebar.button("Clear Saved Figures"):
            st.session_state.pptx_figures_data = []
            st.rerun()
    else:
        st.sidebar.write("No figures saved yet.")


    base_path = "/storage/pblab_shared_data/Nir/Cobrad/pickles_sleep_stage"
    # Define base path for EDF Viewer
    base_path_edf = "/storage/pblab_shared_data/Nir/Cobrad/EDF_Format"

    # Select Sleep Stage
    sleep_stages = ['N1', 'N2', 'N3', 'R', 'W']
    selected_stage = st.selectbox("Select Sleep Stage", sleep_stages, index=2)
    
    # Analysis Mode Selection
    mode = st.radio("Analysis Mode", ["Single Group Analysis", "Compare Groups", "Compare Sleep Stages", "EDF Viewer"], index=0)

    if mode == "Compare Groups":
        run_compare_groups_analysis(base_path, selected_stage)
    elif mode == "Compare Sleep Stages":
        run_compare_sleep_stages_analysis(base_path)
    elif mode == "EDF Viewer":
        run_edf_viewer_mode(base_path_edf)
    else: # Single Group Analysis
        run_single_group_analysis(base_path, selected_stage)

if __name__ == "__main__":
    main()