"""
HEP (Heartbeat Evoked Potential) Comparison: Systole vs Diastole

This script loads pickle files from pickles_sleep_stage/EDF/{sleep_stage},
extracts systole and diastole segments, computes HEP for each phase,
and compares them statistically and visually.
"""

import os
import pickle
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks, medfilt
from scipy.stats import entropy
from scipy.ndimage import median_filter
import mne
import neurokit2 as nk
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
import warnings
import streamlit as st
from io import BytesIO
try:
    import pynapple as nap
    PYNAPPLE_AVAILABLE = True
except ImportError:
    PYNAPPLE_AVAILABLE = False
    # Don't call st.warning() here - it must be called after st.set_page_config()

# Try to import PDF generation libraries
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Try to import PPTX generation libraries
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.enum.text import PP_ALIGN
    from pptx.dml.color import RGBColor
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

warnings.filterwarnings('ignore')

# Fix for MNE pickle loading
import sys, mne.io.array
sys.modules['mne.io.array.array'] = mne.io.array

# Import _plot_metric_vs_hrv from utils
from utils.eeg_utils import _plot_metric_vs_hrv


def load_pickle_files(sleep_stage='N1', base_dir='pickles_sleep_stage/EDF'):
    """
    Load all pickle files from the specified sleep stage directory.
    
    Parameters
    ----------
    sleep_stage : str
        Sleep stage to load (N1, N2, N3, R, W, or 'All' to load from all stages)
    base_dir : str
        Base directory containing sleep stage subdirectories
    
    Returns
    -------
    list : List of tuples (file_path, patient_id, raw_object, sleep_stage)
    """
    loaded_files = []
    
    # If "All" is selected, load from all sleep stages
    if sleep_stage == 'All':
        sleep_stages = ['N1', 'N2', 'N3', 'R', 'W']
        total_files = 0
        
        for stage in sleep_stages:
            pickle_dir = os.path.join(base_dir, stage)
            
            if not os.path.exists(pickle_dir):
                continue
            
            pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pkl')]
            
            if not pickle_files:
                continue
            
            for pickle_file in pickle_files:
                file_path = os.path.join(pickle_dir, pickle_file)
                
                # Extract patient ID from filename
                patient_id = pickle_file.replace('.pkl', '').split('.edf')[0]
                
                try:
                    with open(file_path, 'rb') as f:
                        raw = pickle.load(f)
                    
                    # Add sleep stage to the tuple for tracking
                    loaded_files.append((file_path, patient_id, raw, stage))
                    total_files += 1
                except Exception as e:
                    if 'st' in globals():
                        st.warning(f"Error loading {pickle_file} from {stage}: {e}")
                    else:
                        print(f"Error loading {pickle_file} from {stage}: {e}")
        
        if 'st' in globals():
            st.info(f"Found {total_files} pickle files across all sleep stages")
        else:
            print(f"Found {total_files} pickle files across all sleep stages")
        
        return loaded_files
    
    # Single sleep stage loading (original behavior)
    pickle_dir = os.path.join(base_dir, sleep_stage)
    
    if not os.path.exists(pickle_dir):
        if 'st' in globals():
            st.error(f"Directory not found: {pickle_dir}")
        else:
            print(f"Directory not found: {pickle_dir}")
        return []
    
    pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pkl')]
    
    if not pickle_files:
        if 'st' in globals():
            st.warning(f"No pickle files found in {pickle_dir}")
        else:
            print(f"No pickle files found in {pickle_dir}")
        return []
    
    if 'st' in globals():
        st.info(f"Found {len(pickle_files)} pickle files in {pickle_dir}")
    else:
        print(f"Found {len(pickle_files)} pickle files in {pickle_dir}")
    
    for pickle_file in pickle_files:
        file_path = os.path.join(pickle_dir, pickle_file)
        
        # Extract patient ID from filename
        patient_id = pickle_file.replace('.pkl', '').split('.edf')[0]
        
        try:
            with open(file_path, 'rb') as f:
                raw = pickle.load(f)
            loaded_files.append((file_path, patient_id, raw))
        except Exception as e:
            if 'st' in globals():
                st.warning(f"Error loading {pickle_file}: {e}")
            else:
                print(f"Error loading {pickle_file}: {e}")
            continue
    
    if 'st' in globals():
        st.success(f"Successfully loaded {len(loaded_files)} files")
    else:
        print(f"Successfully loaded {len(loaded_files)} files")
    return loaded_files


def calculate_total_avg_bpm(loaded_files):
    """
    Calculate the total average BPM (beats per minute) across all loaded files.
    
    Parameters
    ----------
    loaded_files : list
        List of tuples (file_path, patient_id, raw_object)
    
    Returns
    -------
    float : Total average BPM across all files
    dict : Dictionary with per-file BPM values and summary statistics
    """
    all_bpms = []
    file_bpms = {}
    
    for file_path, patient_id, raw in loaded_files:
        try:
            # Get sampling frequency
            sfreq = raw.info['sfreq']
            
            # Get channel names and data
            ch_names = raw.ch_names
            data = raw.get_data()
            
            # Find ECG channel
            ch_lower = [ch.lower() for ch in ch_names]
            ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
            
            if not ecg_indices:
                continue
            
            ecg_ch_idx = ecg_indices[0]
            ecg_signal = data[ecg_ch_idx, :]
            
            # Clean ECG signal
            try:
                # bandpass 0.5 - 5
                # ecg_signal = nk.ecg_bandpass(ecg_signal, sampling_rate=sfreq, low_cut=0.5, high_cut=20)
                # 
                ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=sfreq)
            except:
                ecg_clean = ecg_signal
            
            # Detect R-peaks
            _, rpk = nk.ecg_peaks(ecg_clean, sampling_rate=sfreq)
            rpeaks = rpk['ECG_R_Peaks']
            
            if len(rpeaks) < 2:
                continue
            # Calculate BPM from R-R intervals by number of beats / total time
            hr_bpm = (len(rpeaks) / (len(ecg_signal) / sfreq)) * 60 # beats per minute
            # Calculate average BPM for this file
            all_bpms.append(hr_bpm)
            file_bpms[patient_id] = {
                'avg_bpm': hr_bpm,
                'num_beats': len(rpeaks),
                'duration_sec': len(ecg_signal) / sfreq
            }
            
        except Exception as e:
            if 'st' in globals():
                st.warning(f"Error calculating BPM for {patient_id}: {e}")
            else:
                print(f"Error calculating BPM for {patient_id}: {e}")
            continue
    
    # Calculate total average BPM
    if all_bpms:
        total_avg_bpm = np.mean(all_bpms)
        total_std_bpm = np.std(all_bpms)
        
        results = {
            'total_avg_bpm': total_avg_bpm,
            'total_std_bpm': total_std_bpm,
            'total_min_bpm': np.min(all_bpms),
            'total_max_bpm': np.max(all_bpms),
            'num_files': len(file_bpms),
            'file_bpms': file_bpms
        }
        return total_avg_bpm, results
    else:
        return None, {'total_avg_bpm': None, 'num_files': 0, 'file_bpms': {}}


def extract_spike_trigger_events(ecg_data, sfreq, before_event_sec, after_event_sec):
    """
    Extract spike-triggered averaging event windows from ECG data.
    
    For each R-peak detected in the ECG data, extracts a time window
    from [R-peak - before_event_sec] to [R-peak + after_event_sec],
    and returns a 2D array where each row represents one event.
    
    Parameters
    ----------
    ecg_data : np.ndarray
        1D array of ECG signal data
    sfreq : float
        Sampling frequency in Hz
    before_event_sec : float
        Time in seconds to sample before the R-peak event
    after_event_sec : float
        Time in seconds to sample after the R-peak event
    
    Returns
    -------
    np.ndarray
        2D array of shape (n_events, n_samples_per_event) where:
        - n_events: number of detected R-peaks with valid windows
        - n_samples_per_event: number of samples in each window
        Each row contains the ECG data for one event window
    np.ndarray
        Array of R-peak indices (sample indices) for each event
    dict
        Metadata including:
        - n_events: number of events extracted
        - n_samples_per_event: number of samples per event
        - valid_rpeaks: array of valid R-peak indices
        - skipped_events: number of events skipped (out of bounds)
    """
    # Convert time windows to sample indices
    before_event_samples = int(np.round(before_event_sec * sfreq))
    after_event_samples = int(np.round(after_event_sec * sfreq))
    n_samples_per_event = before_event_samples + after_event_samples + 1  # +1 for the R-peak itself
    
    _, rpk = nk.ecg_peaks(ecg_data, sampling_rate=sfreq)
    rpeaks = rpk['ECG_R_Peaks']
    
    if len(rpeaks) == 0:
        return np.array([]), np.array([]), {
            'n_events': 0,
            'n_samples_per_event': n_samples_per_event,
            'valid_rpeaks': np.array([]),
            'skipped_events': 0
        }
    
    # Filter R-peaks to ensure they're within valid range
    valid_rpeaks = []
    events = []
    skipped = 0
    
    for rpeak_idx in rpeaks:
        # Calculate window boundaries
        start_idx = rpeak_idx - before_event_samples
        end_idx = rpeak_idx + after_event_samples + 1  # +1 to include the end sample
        
        # Check if window is within data bounds
        if start_idx >= 0 and end_idx <= len(ecg_data):
            # Extract event window
            event_window = ecg_data[start_idx:end_idx]
            
            # Ensure all events have the same length (handle edge cases)
            if len(event_window) == n_samples_per_event:
                events.append(event_window)
                valid_rpeaks.append(rpeak_idx)
            else:
                skipped += 1
        else:
            skipped += 1
    
    if len(events) == 0:
        return np.array([]), np.array([]), {
            'n_events': 0,
            'n_samples_per_event': n_samples_per_event,
            'valid_rpeaks': np.array([]),
            'skipped_events': skipped
        }
    
    # Convert to 2D numpy array
    events_array = np.array(events)
    valid_rpeaks_array = np.array(valid_rpeaks)
    
    metadata = {
        'n_events': len(events_array),
        'n_samples_per_event': n_samples_per_event,
        'valid_rpeaks': valid_rpeaks_array,
        'skipped_events': skipped,
        'before_event_samples': before_event_samples,
        'after_event_samples': after_event_samples,
        'before_event_sec': before_event_sec,
        'after_event_sec': after_event_sec
    }
    
    return events_array, valid_rpeaks_array, metadata


def extract_systole_diastole_segments(raw, sfreq):
    """
    Extract systole (RST) and diastole (PQR) segments from ECG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object with EEG/ECG data
    sfreq : float
        Sampling frequency
    
    Returns
    -------
    dict : Dictionary with 'systole' and 'diastole' keys, each containing:
        - 'eeg_segments': List of EEG data arrays during that phase
        - 'ecg_segments': List of ECG data arrays during that phase
        - 'indices': List of (start, end) sample index tuples
    """
    # Get channel names and data
    ch_names = raw.ch_names
    data = raw.get_data()
    
    # Find ECG channel
    ch_lower = [ch.lower() for ch in ch_names]
    ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
    
    if not ecg_indices:
        return None
    
    ecg_ch_idx = ecg_indices[0]
    ecg_signal = data[ecg_ch_idx, :]
    
    # Detect R-peaks
    _, rpk = nk.ecg_peaks(ecg_signal, sampling_rate=sfreq)
    rpeaks = rpk['ECG_R_Peaks']
    
    if len(rpeaks) < 2:
        return None
    
    # Find EEG channels
    eeg_ch_idx = [i for i, ch in enumerate(ch_names) 
                  if 'eeg' in ch.lower() or any(elec in ch.upper() for elec in ['FP', 'F', 'C', 'P', 'O', 'T', 'A'])]
    
    if not eeg_ch_idx:
        return None
    
    eeg_data = data[eeg_ch_idx, :]
    
    # Extract segments
    systole_segments_eeg = []
    systole_segments_ecg = []
    systole_indices = []
    
    diastole_segments_eeg = []
    diastole_segments_ecg = []
    diastole_indices = []
    
    # For each R-R interval
    for i in range(len(rpeaks) - 1):
        r_start = rpeaks[i]
        r_end = rpeaks[i + 1]
        rr_length = r_end - r_start
        
        if rr_length < 10:  # Skip very short intervals
            continue
        
        # Systole: R to T (RST) - approximately first 40% of R-R interval (R peak to T wave)
        systole_end = r_start + int(0.4 * rr_length)
        if systole_end > r_start and systole_end < len(ecg_signal):
            systole_segments_eeg.append(eeg_data[:, r_start:systole_end])
            systole_segments_ecg.append(ecg_signal[r_start:systole_end])
            systole_indices.append((r_start, systole_end))
        
        # Diastole: P to R (PQR) - approximately last 30% of previous R-R interval (P wave to R peak)
        if i > 0:
            prev_r_start = rpeaks[i - 1]
            prev_rr_length = r_start - prev_r_start
            # P wave typically starts around 70-80% into the previous R-R interval
            diastole_start = prev_r_start + int(0.7 * prev_rr_length)
            diastole_end = r_start
            if diastole_start < diastole_end and diastole_end < len(ecg_signal) and diastole_start >= 0:
                diastole_segments_eeg.append(eeg_data[:, diastole_start:diastole_end])
                diastole_segments_ecg.append(ecg_signal[diastole_start:diastole_end])
                diastole_indices.append((diastole_start, diastole_end))
    
    # Return segments
    if systole_segments_eeg and diastole_segments_eeg:
        return {
            'systole': {
                'eeg_segments': systole_segments_eeg,
                'ecg_segments': systole_segments_ecg,
                'indices': systole_indices
            },
            'diastole': {
                'eeg_segments': diastole_segments_eeg,
                'ecg_segments': diastole_segments_ecg,
                'indices': diastole_indices
            }
        }
    
    return None


def compute_hep_from_segments(segments_dict, ch_names, sfreq):
    """
    Compute HEP (Heartbeat Evoked Potential) by averaging EEG segments.
    
    Parameters
    ----------
    segments_dict : dict
        Dictionary with 'systole' and 'diastole' keys containing segments
    ch_names : list
        List of EEG channel names
    sfreq : float
        Sampling frequency
    
    Returns
    -------
    dict : Dictionary with averaged HEP for systole and diastole
    """
    hep_results = {}
    
    for phase in ['systole', 'diastole']:
        if phase not in segments_dict:
            continue
        
        eeg_segments = segments_dict[phase]['eeg_segments']
        
        if not eeg_segments:
            continue
        
        # Find maximum length for alignment
        max_len = max(seg.shape[1] for seg in eeg_segments)
        
        # Align and average segments
        aligned_segments = []
        for seg in eeg_segments:
            # Pad shorter segments with NaN
            if seg.shape[1] < max_len:
                padded = np.full((seg.shape[0], max_len), np.nan)
                padded[:, :seg.shape[1]] = seg
                aligned_segments.append(padded)
            else:
                aligned_segments.append(seg)
        
        # Stack and compute mean (ignoring NaNs)
        stacked = np.stack(aligned_segments, axis=0)  # (n_segments, n_channels, n_timepoints)
        hep_mean = np.nanmean(stacked, axis=0)  # (n_channels, n_timepoints)
        hep_std = np.nanstd(stacked, axis=0)  # (n_channels, n_timepoints)
        
        # Create time vector (in seconds)
        time_vector = np.arange(hep_mean.shape[1]) / sfreq
        
        hep_results[phase] = {
            'mean': hep_mean,
            'std': hep_std,
            'time': time_vector,
            'n_segments': len(eeg_segments),
            'ch_names': ch_names[:hep_mean.shape[0]]
        }
    
    return hep_results


def plot_hr_with_segments(segments, raw, patient_id, sfreq, save_path=None, max_duration_sec=60):
    """
    Plot heart rate vs time with systole and diastole segments highlighted.
    
    Parameters
    ----------
    segments : dict
        Dictionary with 'systole' and 'diastole' segments containing indices
    raw : mne.io.Raw
        MNE Raw object with ECG data
    patient_id : str
        Patient identifier
    sfreq : float
        Sampling frequency
    save_path : str, optional
        Path to save the figure
    max_duration_sec : float
        Maximum duration to plot in seconds (default: 60)
    """
    # Get ECG channel
    ch_names = raw.ch_names
    ch_lower = [ch.lower() for ch in ch_names]
    ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
    
    if not ecg_indices:
        print(f"No ECG channel found for {patient_id}")
        return
    
    ecg_ch_idx = ecg_indices[0]
    data = raw.get_data()
    ecg_signal = data[ecg_ch_idx, :]
    
    # Limit to max_duration_sec
    max_samples = int(max_duration_sec * sfreq)
    if len(ecg_signal) > max_samples:
        ecg_signal = ecg_signal[:max_samples]
    
    # Create time vector
    time_vector = np.arange(len(ecg_signal)) / sfreq
    
    # Calculate heart rate from R-peaks
    try:
        ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=sfreq)
    except:
        ecg_clean = ecg_signal
    
    try:
        _, rpk = nk.ecg_peaks(ecg_clean, sampling_rate=sfreq)
        rpeaks = rpk['ECG_R_Peaks']
    except:
        peaks, _ = find_peaks(ecg_clean, distance=int(sfreq * 0.3))
        rpeaks = peaks
    
    # Filter R-peaks within the time window
    rpeaks = rpeaks[rpeaks < len(ecg_signal)]
    r_times = rpeaks / sfreq
    
    # Calculate heart rate (beats per minute) from R-R intervals
    if len(rpeaks) > 1:
        rr_intervals = np.diff(rpeaks) / sfreq  # in seconds
        hr = 60.0 / rr_intervals  # beats per minute
        
        # Create HR time series (assign HR to midpoint between R-peaks)
        hr_times = (r_times[:-1] + r_times[1:]) / 2
    else:
        hr = np.array([])
        hr_times = np.array([])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f'Heart Rate and Cardiac Phases: {patient_id}', fontsize=14, fontweight='bold')
    
    # Plot ECG signal
    ax1.plot(time_vector, ecg_signal, 'k-', linewidth=0.5, alpha=0.7, label='ECG Signal')
    ax1.scatter(r_times, ecg_signal[rpeaks], color='red', s=30, zorder=5, label='R-peaks')
    ax1.set_ylabel('ECG Amplitude (V)', fontsize=11)
    ax1.set_title('ECG Signal with R-peaks', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot heart rate
    if len(hr) > 0:
        ax2.plot(hr_times, hr, 'b-', linewidth=1.5, alpha=0.8, label='Heart Rate')
        ax2.set_ylabel('Heart Rate (BPM)', fontsize=11)
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_title('Heart Rate vs Time with Systole/Diastole Phases', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Highlight systole segments
        systole_indices = segments.get('systole', {}).get('indices', [])
        systole_plotted = False
        for start_idx, end_idx in systole_indices:
            if end_idx < len(ecg_signal):
                start_time = start_idx / sfreq
                end_time = end_idx / sfreq
                if end_time <= max_duration_sec:
                    ax2.axvspan(start_time, end_time, alpha=0.3, color='red', 
                              label='Systole' if not systole_plotted else '')
                    systole_plotted = True
        
        # Highlight diastole segments
        diastole_indices = segments.get('diastole', {}).get('indices', [])
        diastole_plotted = False
        for start_idx, end_idx in diastole_indices:
            if end_idx < len(ecg_signal):
                start_time = start_idx / sfreq
                end_time = end_idx / sfreq
                if end_time <= max_duration_sec:
                    ax2.axvspan(start_time, end_time, alpha=0.3, color='green', 
                              label='Diastole' if not diastole_plotted else '')
                    diastole_plotted = True
        
        # Update legend to include phases
        ax2.legend(loc='upper right')
        
        # Set x-axis limit
        ax2.set_xlim([0, min(max_duration_sec, time_vector[-1])])
    else:
        ax2.text(0.5, 0.5, 'No R-peaks detected', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_ylabel('Heart Rate (BPM)', fontsize=11)
        ax2.set_xlabel('Time (s)', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if 'st' in globals():
            st.info(f"HR plot saved to {save_path}")
        else:
            print(f"HR plot saved to {save_path}")
    
    # Display in Streamlit if available
    if 'st' in globals():
        st.pyplot(fig)
    else:
        plt.show()
    
    plt.close(fig)


def process_all_files(loaded_files):
    """
    Process all loaded files to extract HEP for systole and diastole.
    
    Parameters
    ----------
    loaded_files : list
        List of tuples (file_path, patient_id, raw_object)
    
    Returns
    -------
    pd.DataFrame : DataFrame with HEP results for each patient
    """
    results = []
    
    for file_path, patient_id, raw in loaded_files:
        try:
            sfreq = raw.info['sfreq']
            
            # Extract systole and diastole segments
            segments = extract_systole_diastole_segments(raw, sfreq)
            
            if segments is None:
                if 'st' in globals():
                    st.warning(f"Could not extract segments for {patient_id}")
                else:
                    print(f"Could not extract segments for {patient_id}")
                continue
            
            # Plot HR with segments for the first patient as a sample
            if len(results) == 0:
                plot_hr_with_segments(segments, raw, patient_id, sfreq, 
                                     save_path=None)  # Don't save, display in Streamlit
            
            # Get EEG channel names
            ch_names = raw.ch_names
            eeg_ch_idx = [i for i, ch in enumerate(ch_names) 
                         if 'eeg' in ch.lower() or any(elec in ch.upper() for elec in ['FP', 'F', 'C', 'P', 'O', 'T', 'A'])]
            eeg_ch_names = [ch_names[i] for i in eeg_ch_idx]
            
            # Compute HEP
            hep_results = compute_hep_from_segments(segments, eeg_ch_names, sfreq)
            
            if not hep_results:
                continue
            
            # Extract features for comparison
            for phase in ['systole', 'diastole']:
                if phase not in hep_results:
                    continue
                
                hep_data = hep_results[phase]
                mean_hep = hep_data['mean']
                
                # Compute summary statistics across channels and time
                # Mean amplitude
                mean_amplitude = np.nanmean(mean_hep)
                
                # Peak amplitude (max absolute value)
                peak_amplitude = np.nanmax(np.abs(mean_hep))
                
                # RMS (root mean square)
                rms = np.sqrt(np.nanmean(mean_hep**2))
                
                # Standard deviation
                std_amplitude = np.nanstd(mean_hep)
                
                # Number of segments
                n_segments = hep_data['n_segments']
                
                results.append({
                    'patient_id': patient_id,
                    'phase': phase,
                    'mean_amplitude': mean_amplitude,
                    'peak_amplitude': peak_amplitude,
                    'rms': rms,
                    'std_amplitude': std_amplitude,
                    'n_segments': n_segments,
                    'n_channels': mean_hep.shape[0]
                })
        
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            continue
    
    return pd.DataFrame(results)


def calculate_modulation_index(segments, raw, sfreq, n_phase_bins=2):
    """
    Calculate correlation and modulation index (MI) between systole/diastole phases 
    and EEG amplitude time series using Tort et al. (2010) method.
    
    Parameters
    ----------
    segments : dict
        Dictionary with 'systole' and 'diastole' segments containing indices
    raw : mne.io.Raw
        MNE Raw object with EEG data
    sfreq : float
        Sampling frequency
    n_phase_bins : int
        Number of phase bins (default: 2 for systole/diastole)
    
    Returns
    -------
    dict : Dictionary containing:
        - 'correlation': Pearson correlation between phase and amplitude
        - 'modulation_index': MI value (Tort et al. 2010)
        - 'mean_amplitude_systole': Mean EEG amplitude during systole
        - 'mean_amplitude_diastole': Mean EEG amplitude during diastole
        - 'amplitude_distribution': Distribution of amplitudes across phases
        - 'entropy': Entropy of amplitude distribution
        - 'max_entropy': Maximum possible entropy
    """
    # Get EEG channels
    ch_names = raw.ch_names
    eeg_ch_idx = [i for i, ch in enumerate(ch_names) 
                 if 'eeg' in ch.lower() or any(elec in ch.upper() for elec in ['FP', 'F', 'C', 'P', 'O', 'T', 'A'])]
    
    if not eeg_ch_idx:
        return None
    
    data = raw.get_data()
    eeg_data = data[eeg_ch_idx, :]  # (n_channels, n_timepoints)
    
    # Calculate amplitude (envelope) using Hilbert transform
    from scipy.signal import hilbert
    amplitude_time_series = []
    
    for ch_idx in range(eeg_data.shape[0]):
        # Get analytic signal using Hilbert transform
        analytic_signal = hilbert(eeg_data[ch_idx, :])
        # Get amplitude envelope
        amplitude = np.abs(analytic_signal)
        amplitude_time_series.append(amplitude)
    
    # Average across channels
    amplitude_ts = np.mean(amplitude_time_series, axis=0)
    
    # Create phase assignment array (0 = systole, 1 = diastole, -1 = neither)
    phase_assignment = np.full(len(amplitude_ts), -1, dtype=int)
    
    # Assign systole segments
    systole_indices = segments.get('systole', {}).get('indices', [])
    for start_idx, end_idx in systole_indices:
        if end_idx < len(phase_assignment):
            phase_assignment[start_idx:end_idx] = 0  # Systole
    
    # Assign diastole segments
    diastole_indices = segments.get('diastole', {}).get('indices', [])
    for start_idx, end_idx in diastole_indices:
        if end_idx < len(phase_assignment):
            # Only assign if not already assigned (systole takes precedence if overlap)
            mask = (phase_assignment[start_idx:end_idx] == -1)
            phase_assignment[start_idx:end_idx][mask] = 1  # Diastole
    
    # Extract amplitudes for each phase
    systole_mask = phase_assignment == 0
    diastole_mask = phase_assignment == 1
    
    systole_amplitudes = amplitude_ts[systole_mask]
    diastole_amplitudes = amplitude_ts[diastole_mask]
    
    if len(systole_amplitudes) == 0 or len(diastole_amplitudes) == 0:
        return None
    
    # Calculate mean amplitudes
    mean_amplitude_systole = np.mean(systole_amplitudes)
    mean_amplitude_diastole = np.mean(diastole_amplitudes)
    
    # Calculate correlation between phase and amplitude
    # Create arrays: phase (0 or 1) and corresponding amplitude
    phase_values = np.concatenate([
        np.zeros(len(systole_amplitudes)),
        np.ones(len(diastole_amplitudes))
    ])
    amplitude_values = np.concatenate([systole_amplitudes, diastole_amplitudes])
    
    if len(phase_values) > 1 and len(amplitude_values) > 1:
        correlation, p_value = stats.pearsonr(phase_values, amplitude_values)
    else:
        correlation, p_value = np.nan, np.nan
    
    # Calculate Modulation Index (MI) using Tort et al. (2010) method
    # MI = (H_max - H) / H_max
    # where H is the entropy of the amplitude distribution across phase bins
    # and H_max is the maximum entropy (uniform distribution)
    
    # Create amplitude distribution across phase bins
    # For systole/diastole, we have 2 bins
    amplitude_distribution = np.array([
        np.mean(systole_amplitudes),
        np.mean(diastole_amplitudes)
    ])
    
    # Normalize to get probability distribution
    amplitude_distribution_sum = np.sum(amplitude_distribution)
    if amplitude_distribution_sum > 0:
        p_distribution = amplitude_distribution / amplitude_distribution_sum
    else:
        p_distribution = np.array([0.5, 0.5])  # Uniform if no signal
    
    # Calculate entropy
    # Remove zeros to avoid log(0)
    p_distribution_clean = p_distribution[p_distribution > 0]
    if len(p_distribution_clean) > 0:
        H = entropy(p_distribution_clean, base=2)  # Base 2 for bits
    else:
        H = 0
    
    # Maximum entropy (uniform distribution)
    H_max = np.log2(n_phase_bins)
    
    # Modulation Index
    if H_max > 0:
        modulation_index = (H_max - H) / H_max
    else:
        modulation_index = 0
    
    return {
        'correlation': correlation,
        'correlation_p_value': p_value,
        'modulation_index': modulation_index,
        'mean_amplitude_systole': mean_amplitude_systole,
        'mean_amplitude_diastole': mean_amplitude_diastole,
        'amplitude_distribution': amplitude_distribution,
        'entropy': H,
        'max_entropy': H_max,
        'n_systole_samples': len(systole_amplitudes),
        'n_diastole_samples': len(diastole_amplitudes)
    }


def process_all_files_with_mi(loaded_files):
    """
    Process all loaded files to extract HEP and calculate modulation index.
    
    Parameters
    ----------
    loaded_files : list
        List of tuples (file_path, patient_id, raw_object)
    
    Returns
    -------
    tuple : (pd.DataFrame, pd.DataFrame)
        First DataFrame: HEP results
        Second DataFrame: Modulation index results
    """
    results = []
    mi_results = []
    
    for file_path, patient_id, raw in loaded_files:
        try:
            sfreq = raw.info['sfreq']
            
            # Extract systole and diastole segments
            segments = extract_systole_diastole_segments(raw, sfreq)
            
            if segments is None:
                print(f"Could not extract segments for {patient_id}")
                continue
            
            # Calculate modulation index
            mi_data = calculate_modulation_index(segments, raw, sfreq)
            
            if mi_data:
                mi_results.append({
                    'patient_id': patient_id,
                    **mi_data
                })
            
            # Get EEG channel names
            ch_names = raw.ch_names
            eeg_ch_idx = [i for i, ch in enumerate(ch_names) 
                         if 'eeg' in ch.lower() or any(elec in ch.upper() for elec in ['FP', 'F', 'C', 'P', 'O', 'T', 'A'])]
            eeg_ch_names = [ch_names[i] for i in eeg_ch_idx]
            
            # Compute HEP
            hep_results = compute_hep_from_segments(segments, eeg_ch_names, sfreq)
            
            if not hep_results:
                continue
            
            # Extract features for comparison
            for phase in ['systole', 'diastole']:
                if phase not in hep_results:
                    continue
                
                hep_data = hep_results[phase]
                mean_hep = hep_data['mean']
                
                # Compute summary statistics across channels and time
                mean_amplitude = np.nanmean(mean_hep)
                peak_amplitude = np.nanmax(np.abs(mean_hep))
                rms = np.sqrt(np.nanmean(mean_hep**2))
                std_amplitude = np.nanstd(mean_hep)
                n_segments = hep_data['n_segments']
                
                results.append({
                    'patient_id': patient_id,
                    'phase': phase,
                    'mean_amplitude': mean_amplitude,
                    'peak_amplitude': peak_amplitude,
                    'rms': rms,
                    'std_amplitude': std_amplitude,
                    'n_segments': n_segments,
                    'n_channels': mean_hep.shape[0]
                })
        
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            continue
    
    return pd.DataFrame(results), pd.DataFrame(mi_results)


def compare_systole_diastole(df):
    """
    Perform statistical comparison between systole and diastole HEP.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with HEP results
    
    Returns
    -------
    dict : Dictionary with statistical test results
    """
    if df.empty:
        return None
    
    # Separate systole and diastole data
    systole_df = df[df['phase'] == 'systole']
    diastole_df = df[df['phase'] == 'diastole']
    
    # Get common patients (paired comparison)
    common_patients = set(systole_df['patient_id']).intersection(set(diastole_df['patient_id']))
    
    if len(common_patients) < 2:
        if 'st' in globals():
            st.warning("Not enough paired data for comparison")
        else:
            print("Not enough paired data for comparison")
        return None
    
    # Prepare paired data
    systole_paired = systole_df[systole_df['patient_id'].isin(common_patients)].sort_values('patient_id')
    diastole_paired = diastole_df[diastole_df['patient_id'].isin(common_patients)].sort_values('patient_id')
    
    results = {}
    
    # Compare each metric
    metrics = ['mean_amplitude', 'peak_amplitude', 'rms', 'std_amplitude']
    
    for metric in metrics:
        systole_vals = systole_paired[metric].values
        diastole_vals = diastole_paired[metric].values
        
        # Remove NaN values
        valid_mask = ~(np.isnan(systole_vals) | np.isnan(diastole_vals))
        systole_vals = systole_vals[valid_mask]
        diastole_vals = diastole_vals[valid_mask]
        
        if len(systole_vals) < 2:
            continue
        
        # Paired t-test
        t_stat, p_value_ttest = stats.ttest_rel(diastole_vals, systole_vals)
        
        # Wilcoxon signed-rank test (non-parametric)
        try:
            w_stat, p_value_wilcoxon = stats.wilcoxon(diastole_vals, systole_vals)
        except:
            p_value_wilcoxon = np.nan
        
        # Effect size (Cohen's d for paired samples)
        differences = diastole_vals - systole_vals
        cohens_d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences, ddof=1) > 0 else 0
        
        results[metric] = {
            'systole_mean': np.mean(systole_vals),
            'systole_std': np.std(systole_vals),
            'diastole_mean': np.mean(diastole_vals),
            'diastole_std': np.std(diastole_vals),
            'mean_difference': np.mean(differences),
            'p_value_ttest': p_value_ttest,
            'p_value_wilcoxon': p_value_wilcoxon,
            'cohens_d': cohens_d,
            'n_patients': len(systole_vals)
        }
    
    return results


def plot_comparison(df, stats_results, save_path=None):
    """
    Create visualization comparing systole and diastole HEP.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with HEP results
    stats_results : dict
        Statistical test results
    save_path : str, optional
        Path to save the figure
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Set style
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('HEP Comparison: Systole vs Diastole', fontsize=16, fontweight='bold')
    
    metrics = ['mean_amplitude', 'peak_amplitude', 'rms', 'std_amplitude']
    metric_labels = ['Mean Amplitude (μV)', 'Peak Amplitude (μV)', 'RMS (μV)', 'Std Amplitude (μV)']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        
        # Prepare data for this metric
        systole_vals = df[df['phase'] == 'systole'][metric].dropna()
        diastole_vals = df[df['phase'] == 'diastole'][metric].dropna()
        
        # Box plot
        data_to_plot = [systole_vals, diastole_vals]
        bp = ax.boxplot(data_to_plot, labels=['Systole', 'Diastole'], patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel(label)
        ax.set_title(f'{label.replace(" (μV)", "")}')
        ax.grid(True, alpha=0.3)
        
        # Add statistical annotation if available
        if stats_results and metric in stats_results:
            stats_data = stats_results[metric]
            p_val = stats_data['p_value_wilcoxon']
            if not np.isnan(p_val):
                if p_val < 0.001:
                    sig_text = '***'
                elif p_val < 0.01:
                    sig_text = '**'
                elif p_val < 0.05:
                    sig_text = '*'
                else:
                    sig_text = 'ns'
                
                # Add significance line
                y_max = max(systole_vals.max(), diastole_vals.max())
                y_min = min(systole_vals.min(), diastole_vals.min())
                y_range = y_max - y_min
                
                ax.plot([1, 2], [y_max + 0.05 * y_range, y_max + 0.05 * y_range], 
                       'k-', linewidth=1.5)
                ax.text(1.5, y_max + 0.08 * y_range, sig_text, 
                       ha='center', fontsize=12, fontweight='bold')
                
                # Add p-value text
                ax.text(1.5, y_max + 0.12 * y_range, f'p = {p_val:.3e}', 
                       ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if 'st' in globals():
            st.info(f"Figure saved to {save_path}")
        else:
            print(f"Figure saved to {save_path}")
    
    # Display in Streamlit if available
    if 'st' in globals():
        st.pyplot(fig)
    else:
        plt.show()
    
    plt.close(fig)


def print_statistics(stats_results):
    """
    Print statistical comparison results.
    
    Parameters
    ----------
    stats_results : dict
        Statistical test results
    """
    if not stats_results:
        if 'st' in globals():
            st.warning("No statistical results to display")
        else:
            print("No statistical results to display")
        return
    
    if 'st' in globals():
        st.markdown("## 📊 Statistical Comparison: Systole vs Diastole HEP")
        st.markdown("---")
    else:
        print("\n" + "="*80)
        print("STATISTICAL COMPARISON: Systole vs Diastole HEP")
        print("="*80)
    
    for metric, results in stats_results.items():
        metric_name = metric.upper().replace('_', ' ')
        
        if 'st' in globals():
            st.markdown(f"### {metric_name}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Systole", f"{results['systole_mean']:.4f} ± {results['systole_std']:.4f} μV")
            with col2:
                st.metric("Diastole", f"{results['diastole_mean']:.4f} ± {results['diastole_std']:.4f} μV")
            with col3:
                st.metric("Difference", f"{results['mean_difference']:.4f} μV")
            
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("Cohen's d", f"{results['cohens_d']:.4f}")
            with col5:
                p_val = results['p_value_wilcoxon']
                st.metric("Wilcoxon p-value", f"{p_val:.3e}")
            with col6:
                st.metric("N patients", f"{results['n_patients']}")
            
            # Significance interpretation
            p_val = results['p_value_wilcoxon']
            if not np.isnan(p_val):
                if p_val < 0.001:
                    sig = "*** (highly significant)"
                    st.success(f"Significance: {sig}")
                elif p_val < 0.01:
                    sig = "** (very significant)"
                    st.success(f"Significance: {sig}")
                elif p_val < 0.05:
                    sig = "* (significant)"
                    st.success(f"Significance: {sig}")
                else:
                    sig = "ns (not significant)"
                    st.info(f"Significance: {sig}")
            st.markdown("---")
        else:
            print(f"\n{metric_name}:")
            print(f"  Systole:   {results['systole_mean']:.4f} ± {results['systole_std']:.4f} μV")
            print(f"  Diastole:  {results['diastole_mean']:.4f} ± {results['diastole_std']:.4f} μV")
            print(f"  Difference: {results['mean_difference']:.4f} μV")
            print(f"  Cohen's d:  {results['cohens_d']:.4f}")
            print(f"  Paired t-test p-value:     {results['p_value_ttest']:.3e}")
            print(f"  Wilcoxon test p-value:     {results['p_value_wilcoxon']:.3e}")
            print(f"  N patients: {results['n_patients']}")
            
            # Significance interpretation
            p_val = results['p_value_wilcoxon']
            if not np.isnan(p_val):
                if p_val < 0.001:
                    sig = "*** (highly significant)"
                elif p_val < 0.01:
                    sig = "** (very significant)"
                elif p_val < 0.05:
                    sig = "* (significant)"
                else:
                    sig = "ns (not significant)"
                print(f"  Significance: {sig}")
    
    if 'st' not in globals():
        print("\n" + "="*80)


def generate_pdf_report(loaded_files, sleep_stage, output_dir, save_results):
    """
    Generate a PDF report of the HEP analysis.
    
    Parameters
    ----------
    loaded_files : list
        List of loaded files
    sleep_stage : str
        Sleep stage being analyzed
    output_dir : str
        Output directory for saving PDF
    save_results : bool
        Whether to save results
    
    Returns
    -------
    bytes or None
        PDF file as bytes if successful, None otherwise
    """
    if not REPORTLAB_AVAILABLE:
        st.error("reportlab is not available. Install with: `pip install reportlab`")
        return None
    
    try:
        # Create PDF in memory
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        elements.append(Paragraph("HEP Comparison: Systole vs Diastole", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Summary information
        elements.append(Paragraph("Analysis Summary", heading_style))
        elements.append(Paragraph(f"<b>Sleep Stage:</b> {sleep_stage}", styles['Normal']))
        num_patients = len(loaded_files) if loaded_files else 0
        elements.append(Paragraph(f"<b>Number of Patients:</b> {num_patients}", styles['Normal']))
        elements.append(Paragraph(f"<b>Output Directory:</b> {output_dir}", styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Add information about available data
        if loaded_files and len(loaded_files) > 0:
            elements.append(Paragraph("Patient IDs:", heading_style))
            patient_ids = [f[1] if len(f) > 1 else f"Patient_{i}" for i, f in enumerate(loaded_files)]
            for pid in patient_ids[:10]:  # Limit to first 10
                elements.append(Paragraph(f"• {pid}", styles['Normal']))
            if len(patient_ids) > 10:
                elements.append(Paragraph(f"... and {len(patient_ids) - 10} more", styles['Normal']))
            elements.append(Spacer(1, 0.2*inch))
        
        # Add note about plots
        elements.append(Paragraph("Note:", heading_style))
        elements.append(Paragraph(
            "This PDF contains a summary of the analysis. For detailed plots and visualizations, "
            "please refer to the interactive Streamlit interface or check the saved plot files in the output directory.",
            styles['Normal']
        ))
        elements.append(Spacer(1, 0.2*inch))
        
        # Add information about saved files if save_results is True
        if save_results:
            elements.append(Paragraph("Saved Files:", heading_style))
            elements.append(Paragraph(
                f"All analysis results and plots have been saved to: <b>{output_dir}</b>",
                styles['Normal']
            ))
            elements.append(Spacer(1, 0.2*inch))
        
        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph(f"<i>Report generated on: {timestamp}</i>", 
                                 ParagraphStyle('Timestamp', parent=styles['Normal'], 
                                               fontSize=9, textColor=colors.grey, alignment=TA_CENTER)))
        
        # Build PDF
        doc.build(elements)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
        
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_pptx_report(loaded_files, sleep_stage, output_dir, save_results):
    """
    Generate a PPTX report of the HEP analysis.
    
    Parameters
    ----------
    loaded_files : list
        List of loaded files
    sleep_stage : str
        Sleep stage being analyzed
    output_dir : str
        Output directory for saving PPTX
    save_results : bool
        Whether to save results
    
    Returns
    -------
    bytes or None
        PPTX file as bytes if successful, None otherwise
    """
    if not PPTX_AVAILABLE:
        st.error("python-pptx is not available. Install with: `pip install python-pptx`")
        return None
    
    try:
        # Create PowerPoint presentation
        prs = Presentation()
        prs.slide_width = Inches(10)
        prs.slide_height = Inches(7.5)
        
        # Title slide
        title_slide_layout = prs.slide_layouts[0]
        slide = prs.slides.add_slide(title_slide_layout)
        title = slide.shapes.title
        subtitle = slide.placeholders[1]
        
        title.text = "HEP Comparison: Systole vs Diastole"
        subtitle.text = f"Sleep Stage: {sleep_stage}\nAnalysis Report"
        
        # Summary slide
        blank_slide_layout = prs.slide_layouts[6]  # Blank layout
        slide = prs.slides.add_slide(blank_slide_layout)
        
        # Add title text box
        left = Inches(0.5)
        top = Inches(0.5)
        width = Inches(9)
        height = Inches(0.8)
        title_box = slide.shapes.add_textbox(left, top, width, height)
        title_frame = title_box.text_frame
        title_frame.text = "Analysis Summary"
        title_para = title_frame.paragraphs[0]
        title_para.font.size = Pt(24)
        title_para.font.bold = True
        title_para.font.color.rgb = RGBColor(31, 119, 180)  # Blue color
        
        # Add content text box
        content_top = Inches(1.5)
        content_height = Inches(5)
        content_box = slide.shapes.add_textbox(left, content_top, width, content_height)
        content_frame = content_box.text_frame
        content_frame.word_wrap = True
        
        # Add summary information
        num_patients = len(loaded_files) if loaded_files else 0
        summary_text = f"Sleep Stage: {sleep_stage}\n"
        summary_text += f"Number of Patients: {num_patients}\n"
        summary_text += f"Output Directory: {output_dir}\n\n"
        
        # Add patient IDs
        if loaded_files and len(loaded_files) > 0:
            summary_text += "Patient IDs:\n"
            patient_ids = [f[1] if len(f) > 1 else f"Patient_{i}" for i, f in enumerate(loaded_files)]
            for pid in patient_ids[:10]:  # Limit to first 10
                summary_text += f"• {pid}\n"
            if len(patient_ids) > 10:
                summary_text += f"... and {len(patient_ids) - 10} more\n"
            summary_text += "\n"
        
        # Add note about plots
        summary_text += "Note:\n"
        summary_text += "This presentation contains a summary of the analysis. "
        summary_text += "For detailed plots and visualizations, please refer to the interactive Streamlit interface "
        summary_text += "or check the saved plot files in the output directory.\n\n"
        
        # Add information about saved files
        if save_results:
            summary_text += "Saved Files:\n"
            summary_text += f"All analysis results and plots have been saved to: {output_dir}\n\n"
        
        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_text += f"Report generated on: {timestamp}"
        
        # Set text content
        para = content_frame.paragraphs[0]
        para.text = summary_text
        para.font.size = Pt(12)
        para.font.color.rgb = RGBColor(0, 0, 0)
        
        # Save to BytesIO
        buffer = BytesIO()
        prs.save(buffer)
        pptx_bytes = buffer.getvalue()
        buffer.close()
        
        return pptx_bytes
        
    except Exception as e:
        st.error(f"Error generating PPTX: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Main function to run HEP comparison analysis with Streamlit.
    """
    st.set_page_config(page_title="HEP Systole/Diastole Comparison", layout="wide")
    
    # Check and warn about pynapple availability after set_page_config()
    if not PYNAPPLE_AVAILABLE:
        st.warning("⚠️ pynapple is not available. PETH raster plot functionality will be disabled. Install with: `pip install pynapple`")
    
    st.title("🧠 HEP Comparison: Systole vs Diastole")
    st.markdown("Heartbeat Evoked Potential (HEP) analysis comparing systole and diastole phases")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    sleep_stage = st.sidebar.selectbox(
        "Sleep Stage",
        options=['N1', 'N2', 'N3', 'R', 'W', 'All'],
        index=0,
        help="Select the sleep stage to analyze (or 'All' to analyze all stages combined)"
    )
    
    base_dir = st.sidebar.text_input(
        "Base Directory",
        value='pickles_sleep_stage/EDF',
        help="Base directory containing sleep stage subdirectories"
    )
    
    output_dir = st.sidebar.text_input(
        "Output Directory",
        value='hep_comparison_results',
        help="Directory to save results"
    )
    
    save_results = st.sidebar.checkbox("Save Results to Files", value=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Display sleep stage info
    if sleep_stage == 'All':
        st.markdown(f"**Sleep Stage:** All (N1, N2, N3, R, W combined)")
    else:
        st.markdown(f"**Sleep Stage:** {sleep_stage}")
    st.markdown(f"**Base Directory:** {base_dir}")
    st.markdown("---")
    
    # Load pickle files
    with st.spinner("Loading pickle files..."):
        loaded_files_raw = load_pickle_files(sleep_stage=sleep_stage, base_dir=base_dir)
    
    if not loaded_files_raw:
        st.error("No files loaded. Please check the directory path.")
        return
    
    # Handle tuple format - convert 4-element tuples (with sleep_stage) to 3-element for compatibility
    # Check if files include sleep_stage (4 elements) or old format (3 elements)
    if loaded_files_raw and len(loaded_files_raw[0]) == 4:
        # New format: (file_path, patient_id, raw, sleep_stage)
        loaded_files = [(f[0], f[1], f[2]) for f in loaded_files_raw]
        # Get unique sleep stages for display
        sleep_stages_in_data = list(set([f[3] for f in loaded_files_raw]))
        if sleep_stage == 'All' and len(sleep_stages_in_data) > 1:
            st.info(f"📊 Loaded {len(loaded_files)} files from {len(sleep_stages_in_data)} sleep stages: {', '.join(sorted(sleep_stages_in_data))}")
    else:
        # Old format: (file_path, patient_id, raw) - keep as is
        loaded_files = loaded_files_raw
    
    # PDF Download button in sidebar (after files are loaded)
    st.sidebar.markdown("---")
    st.sidebar.header("Export")
    
    # Generate PDF when button is clicked
    if st.sidebar.button("📥 Generate PDF Report", help="Generate and download a PDF report of the analysis"):
        with st.sidebar:
            with st.spinner("Generating PDF report..."):
                pdf_bytes = generate_pdf_report(loaded_files=loaded_files, sleep_stage=sleep_stage, 
                                                output_dir=output_dir, save_results=save_results)
                if pdf_bytes:
                    # Store PDF in session state
                    st.session_state['pdf_bytes'] = pdf_bytes
                    st.session_state['pdf_generated'] = True
                    st.success("PDF report generated!")
                else:
                    st.error("Failed to generate PDF report.")
    
    # Show download button if PDF is generated
    if 'pdf_generated' in st.session_state and st.session_state.get('pdf_generated', False):
        if 'pdf_bytes' in st.session_state:
            st.sidebar.download_button(
                label="⬇️ Download PDF",
                data=st.session_state['pdf_bytes'],
                file_name=f"HEP_Analysis_{sleep_stage}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                key="pdf_download_sidebar"
            )
    
    # Generate PPTX when button is clicked
    if st.sidebar.button("📊 Generate PPTX Report", help="Generate and download a PPTX report of the analysis"):
        with st.sidebar:
            with st.spinner("Generating PPTX report..."):
                pptx_bytes = generate_pptx_report(loaded_files=loaded_files, sleep_stage=sleep_stage, 
                                                  output_dir=output_dir, save_results=save_results)
                if pptx_bytes:
                    # Store PPTX in session state
                    st.session_state['pptx_bytes'] = pptx_bytes
                    st.session_state['pptx_generated'] = True
                    st.success("PPTX report generated!")
                else:
                    st.error("Failed to generate PPTX report.")
    
    # Show download button if PPTX is generated
    if 'pptx_generated' in st.session_state and st.session_state.get('pptx_generated', False):
        if 'pptx_bytes' in st.session_state:
            st.sidebar.download_button(
                label="⬇️ Download PPTX",
                data=st.session_state['pptx_bytes'],
                file_name=f"HEP_Analysis_{sleep_stage}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                key="pptx_download_sidebar"
            )
    
    # Run BPM peri-event analysis (PETH raster plots)
    st.markdown("---")
    st.markdown("## 📊 BPM Peri-Event Analysis")
    BPM_peri_event_analysis(loaded_files, sleep_stage, output_dir, save_results)
    
    # Run HEP systole/diastole comparison analysis
    st.markdown("---")
    st.markdown("## 🧠 HEP Systole/Diastole Comparison Analysis")
    MI_systole_diastole_comparison(loaded_files, sleep_stage, output_dir, save_results)


def BPM_peri_event_analysis(loaded_files, sleep_stage, output_dir, save_results):
    """
    Run BPM peri-event analysis and create PETH raster plots.
    """
    # Create PETH raster plot for the first file
    if loaded_files:
        plot_peth_raster(loaded_files, file_index=0, minmax=(-0.5, 1.0), 
                        binsize=0.01, save_plot=save_results, save_dir=output_dir)


def clean_ecg_advanced(ecg_signal, sampling_rate, median_window_ms=300, 
                       wavelet='db4', wavelet_levels=6, zscore_threshold=3.0):
    """
    Advanced ECG signal cleaning using multiple techniques:
    1. Median filter to suppress spikes
    2. Wavelet denoising
    3. Z-score thresholding to remove extreme values
    
    Parameters
    ----------
    ecg_signal : np.ndarray
        1D array of ECG signal data
    sampling_rate : float
        Sampling frequency in Hz
    median_window_ms : float, optional
        Median filter window size in milliseconds (default: 300, range: 200-400)
    wavelet : str, optional
        Wavelet type for denoising: 'db4' or 'sym4' (default: 'db4')
    wavelet_levels : int, optional
        Number of decomposition levels for wavelet (default: 6, range: 5-8)
    zscore_threshold : float, optional
        Z-score threshold for removing extreme values (default: 3.0)
    
    Returns
    -------
    np.ndarray
        Cleaned ECG signal
    dict
        Dictionary with cleaning information including:
        - 'median_filtered': signal after median filter
        - 'wavelet_denoised': signal after wavelet denoising
        - 'n_extreme_samples': number of samples removed by z-score thresholding
        - 'methods_applied': list of methods successfully applied
    """
    if len(ecg_signal) == 0:
        return ecg_signal, {'methods_applied': [], 'n_extreme_samples': 0}
    
    methods_applied = []
    info = {}
    cleaned_signal = ecg_signal.copy()
    
    # Step 1: Median filter to suppress spikes
    # Convert window size from ms to samples
    median_window_samples = int(np.round(median_window_ms * sampling_rate / 1000.0))
    # Ensure window size is odd (required for median filter)
    if median_window_samples % 2 == 0:
        median_window_samples += 1
    # Ensure minimum window size of 3
    median_window_samples = max(3, median_window_samples)
    # Use scipy.ndimage.median_filter for better performance
    median_filtered = median_filter(cleaned_signal, size=median_window_samples)
    cleaned_signal = median_filtered
    methods_applied.append('median_filter')
    info['median_filtered'] = median_filtered
    info['median_window_samples'] = median_window_samples
    
    # Step 2: Wavelet denoising
    if PYWT_AVAILABLE:
        try:
            # Validate wavelet
            if wavelet not in ['db4', 'sym4']:
                wavelet = 'db4'  # Default to db4 if invalid
            
            # Validate levels
            max_level = pywt.dwt_max_level(len(cleaned_signal), wavelet)
            wavelet_levels = min(max(wavelet_levels, 5), min(8, max_level))
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(cleaned_signal, wavelet, level=wavelet_levels)
            
            # Estimate noise standard deviation using median absolute deviation (MAD)
            # of the finest detail coefficients
            detail_coeffs = coeffs[-1]
            if len(detail_coeffs) > 0:
                sigma = np.median(np.abs(detail_coeffs)) / 0.6745  # MAD estimator
            else:
                sigma = np.std(detail_coeffs) if len(detail_coeffs) > 0 else 1.0
            
            # Apply soft thresholding to detail coefficients
            threshold = sigma * np.sqrt(2 * np.log(len(cleaned_signal)))
            coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients
            for i in range(1, len(coeffs)):
                coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
            
            # Reconstruct signal
            wavelet_denoised = pywt.waverec(coeffs_thresh, wavelet)
            
            # Ensure same length (wavelet reconstruction might add samples)
            if len(wavelet_denoised) > len(cleaned_signal):
                wavelet_denoised = wavelet_denoised[:len(cleaned_signal)]
            elif len(wavelet_denoised) < len(cleaned_signal):
                # Pad with last value if shorter
                padding = np.full(len(cleaned_signal) - len(wavelet_denoised), 
                                wavelet_denoised[-1])
                wavelet_denoised = np.concatenate([wavelet_denoised, padding])
            
            cleaned_signal = wavelet_denoised
            methods_applied.append('wavelet_denoising')
            info['wavelet_denoised'] = wavelet_denoised
            info['wavelet_used'] = wavelet
            info['wavelet_levels'] = wavelet_levels
            info['threshold'] = threshold
        except Exception as e:
            if 'st' in globals():
                st.warning(f"Wavelet denoising failed: {e}")
            else:
                print(f"Wavelet denoising failed: {e}")
    else:
        if 'st' in globals():
            st.warning("PyWavelets not available. Skipping wavelet denoising.")
        else:
            print("PyWavelets not available. Skipping wavelet denoising.")
    
    # Step 3: Z-score thresholding to remove extreme values
    try:
        # Calculate z-scores
        mean_signal = np.mean(cleaned_signal)
        std_signal = np.std(cleaned_signal)
        
        if std_signal > 0:
            z_scores = np.abs((cleaned_signal - mean_signal) / std_signal)
            
            # Find extreme values
            extreme_mask = z_scores > zscore_threshold
            n_extreme = np.sum(extreme_mask)
            
            if n_extreme > 0:
                # Replace extreme values with median of surrounding samples
                # or interpolate
                extreme_indices = np.where(extreme_mask)[0]
                
                # Create a copy for interpolation
                signal_interp = cleaned_signal.copy()
                
                # For each extreme value, replace with median of neighbors
                for idx in extreme_indices:
                    # Get neighborhood (avoid edges)
                    start_idx = max(0, idx - 5)
                    end_idx = min(len(cleaned_signal), idx + 6)
                    neighborhood = cleaned_signal[start_idx:end_idx]
                    # Exclude the extreme value itself
                    neighborhood = neighborhood[neighborhood != cleaned_signal[idx]]
                    if len(neighborhood) > 0:
                        signal_interp[idx] = np.median(neighborhood)
                    else:
                        # Fallback: use mean
                        signal_interp[idx] = mean_signal
                
                cleaned_signal = signal_interp
                methods_applied.append('zscore_thresholding')
                info['n_extreme_samples'] = n_extreme
                info['zscore_threshold'] = zscore_threshold
            else:
                info['n_extreme_samples'] = 0
        else:
            info['n_extreme_samples'] = 0
    except Exception as e:
        if 'st' in globals():
            st.warning(f"Z-score thresholding failed: {e}")
        else:
            print(f"Z-score thresholding failed: {e}")
        info['n_extreme_samples'] = 0
    
    info['methods_applied'] = methods_applied
    
    return cleaned_signal, info


def plot_averaged_eeg_peth_raster(loaded_files, minmax=(-0.5, 1.0), binsize=0.01, save_plot=False, save_dir=None):
    """
    Create averaged PETH and raster plots for EEG channels across all patients.
    
    Parameters
    ----------
    loaded_files : list
        List of tuples (file_path, patient_id, raw_object)
    minmax : tuple, optional
        Time window around events in seconds (default: (-0.5, 1.0))
    binsize : float, optional
        Bin size for rate calculation in seconds (default: 0.01)
    save_plot : bool, optional
        Whether to save the plot (default: False)
    save_dir : str, optional
        Directory to save the plot (default: None)
    
    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if plot was created, None otherwise
    """
    if not PYNAPPLE_AVAILABLE:
        if 'st' in globals():
            st.error("pynapple is not available. Please install it to use averaged PETH raster plots.")
        else:
            print("pynapple is not available. Please install it to use averaged PETH raster plots.")
        return None
    
    if not loaded_files:
        return None
    
    # Collect data from all patients
    all_channel_data = {}  # {channel_name: {'rates_list': [], 'events_list': []}}
    common_channels = None
    
    if 'st' in globals():
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    for file_idx, (file_path, patient_id, raw) in enumerate(loaded_files):
        try:
            if 'st' in globals():
                status_text.text(f"Processing {patient_id} ({file_idx + 1}/{len(loaded_files)})...")
                progress_bar.progress((file_idx + 1) / len(loaded_files))
            
            # Get sampling frequency
            sfreq = raw.info['sfreq']
            
            # Get channel names and data
            ch_names = raw.ch_names
            data = raw.get_data()
            
            # Find ECG channel
            ch_lower = [ch.lower() for ch in ch_names]
            ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
            
            if not ecg_indices:
                continue
            
            ecg_ch_idx = ecg_indices[0]
            ecg_signal = data[ecg_ch_idx, :]
            
            # Clean ECG and detect R-peaks
            try:
                ecg_signal_clean, _ = clean_ecg_advanced(
                    ecg_signal, 
                    sampling_rate=sfreq,
                    median_window_ms=300,
                    wavelet='db4',
                    wavelet_levels=5,
                    zscore_threshold=3.0
                )
                _, rpk = nk.ecg_peaks(ecg_signal_clean, sampling_rate=sfreq)
                rpeaks = rpk['ECG_R_Peaks']
            except:
                continue
            
            if len(rpeaks) < 2:
                continue
            
            # Convert R-peak indices to timestamps
            rpeak_times = rpeaks / sfreq
            rpeak_ts = nap.Ts(t=rpeak_times, time_units="s")
            
            # Find EEG channels
            eeg_ch_indices = [i for i, ch in enumerate(ch_names) 
                            if 'eeg' in ch.lower() or any(elec in ch.upper() for elec in ['FP', 'F', 'C', 'P', 'O', 'T', 'A'])]
            
            if not eeg_ch_indices:
                continue
            
            # Limit to first 10 channels for consistency
            eeg_ch_indices = eeg_ch_indices[:10]
            eeg_ch_names = [ch_names[i] for i in eeg_ch_indices]
            
            # Update common channels (intersection across all patients)
            if common_channels is None:
                common_channels = set(eeg_ch_names)
            else:
                common_channels = common_channels.intersection(set(eeg_ch_names))
            
            # Get EEG data
            eeg_data = data[eeg_ch_indices, :]
            
            # Process each EEG channel
            for ch_idx, ch_name in enumerate(eeg_ch_names):
                if ch_name not in common_channels:
                    continue
                
                try:
                    eeg_signal = eeg_data[ch_idx, :]
                    
                    # Detect events in EEG
                    eeg_mean = np.mean(eeg_signal)
                    eeg_std = np.std(eeg_signal)
                    threshold = eeg_mean + 2 * eeg_std
                    
                    # Find peaks above threshold
                    peaks, _ = find_peaks(eeg_signal, height=threshold, distance=int(sfreq * 0.1))
                    
                    if len(peaks) > 0:
                        # Convert peak indices to timestamps
                        peak_times = peaks / sfreq
                        eeg_peak_ts = nap.Ts(t=peak_times, time_units="s")
                        
                        # Compute perievent alignment around R-peaks
                        peth_eeg = nap.compute_perievent(
                            timestamps=eeg_peak_ts,
                            tref=rpeak_ts,
                            minmax=minmax,
                            time_unit="s"
                        )
                        
                        if len(peth_eeg) > 0:
                            # Initialize channel data structure
                            if ch_name not in all_channel_data:
                                all_channel_data[ch_name] = {'rates_list': [], 'events_list': []}
                            
                            # Calculate rates for this patient
                            try:
                                peth_eeg_count = peth_eeg.count(binsize)
                                if hasattr(peth_eeg_count, 'values'):
                                    count_vals = peth_eeg_count.values
                                    if count_vals.ndim == 2:
                                        rates = np.mean(count_vals, axis=1) / binsize
                                    else:
                                        rates = count_vals / binsize
                                    
                                    if hasattr(peth_eeg_count, 'index'):
                                        time_axis = peth_eeg_count.index.values
                                    else:
                                        time_axis = np.arange(len(rates)) * binsize + minmax[0]
                                    
                                    all_channel_data[ch_name]['rates_list'].append((time_axis, rates))
                            except:
                                pass
                            
                            # Collect events for raster plot
                            try:
                                peth_eeg_tsd = peth_eeg.to_tsd()
                                all_channel_data[ch_name]['events_list'].append({
                                    'times': peth_eeg_tsd.index.values,
                                    'events': peth_eeg_tsd.values
                                })
                            except:
                                pass
                
                except Exception as e:
                    continue
        
        except Exception as e:
            if 'st' in globals():
                st.warning(f"Error processing {patient_id}: {e}")
            else:
                print(f"Error processing {patient_id}: {e}")
            continue
    
    if 'st' in globals():
        progress_bar.empty()
        status_text.empty()
    
    # Check if we have data
    if not all_channel_data or not common_channels:
        if 'st' in globals():
            st.warning("No common EEG channels found across patients or no data collected.")
        else:
            print("No common EEG channels found across patients or no data collected.")
        return None
    
    # Filter to only common channels
    channels_to_plot = sorted([ch for ch in all_channel_data.keys() if ch in common_channels])
    
    if not channels_to_plot:
        if 'st' in globals():
            st.warning("No channels to plot after filtering.")
        else:
            print("No channels to plot after filtering.")
        return None
    
    # Create plots
    n_channels = len(channels_to_plot)
    n_cols = 2  # PETH and Raster side by side
    n_rows = n_channels
    
    fig_avg, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), sharex='col')
    if n_channels == 1:
        axes = axes.reshape(1, -1)
    
    for ch_idx, ch_name in enumerate(channels_to_plot):
        ax_peth = axes[ch_idx, 0]
        ax_raster = axes[ch_idx, 1]
        
        channel_data = all_channel_data[ch_name]
        
        # Plot 1: Average PETH
        if channel_data['rates_list']:
            try:
                # Interpolate all rates to common time axis
                all_time_axes = [t for t, _ in channel_data['rates_list']]
                all_rates = [r for _, r in channel_data['rates_list']]
                
                # Find common time range
                min_time = max([t[0] for t in all_time_axes])
                max_time = min([t[-1] for t in all_time_axes])
                common_time = np.arange(min_time, max_time + binsize, binsize)
                
                # Interpolate each patient's rates to common time axis
                interpolated_rates = []
                for time_axis, rates in zip(all_time_axes, all_rates):
                    if len(time_axis) > 1 and len(rates) > 1:
                        interp_rates = np.interp(common_time, time_axis, rates)
                        interpolated_rates.append(interp_rates)
                
                if interpolated_rates:
                    # Average across patients
                    avg_rates = np.mean(interpolated_rates, axis=0)
                    std_rates = np.std(interpolated_rates, axis=0)
                    
                    ax_peth.plot(common_time, avg_rates, linewidth=2, color='blue', label='Mean')
                    ax_peth.fill_between(common_time, avg_rates - std_rates, avg_rates + std_rates, 
                                        alpha=0.3, color='blue', label='±1 SD')
                    ax_peth.set_ylabel("Rate (events/sec)", fontsize=10)
                    ax_peth.set_title(f"Avg PETH - {ch_name} (n={len(interpolated_rates)} patients)", 
                                     fontsize=11, fontweight='bold')
                    ax_peth.axvline(0.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
                    ax_peth.legend(fontsize=8)
                    ax_peth.grid(True, alpha=0.3)
                    ax_peth.set_xlim(minmax)
                    ax_peth.tick_params(axis='x', labelsize=8, bottom=True)
            except Exception as e:
                ax_peth.text(0.5, 0.5, f'PETH Error: {str(e)[:40]}', 
                           ha='center', va='center', transform=ax_peth.transAxes, fontsize=8, color='red')
        else:
            ax_peth.text(0.5, 0.5, 'No rate data', 
                       ha='center', va='center', transform=ax_peth.transAxes, fontsize=9)
        
        # Plot 2: Average Raster Plot (density of events)
        if channel_data['events_list']:
            try:
                # Collect all event times from all patients
                all_times = []
                for event_data in channel_data['events_list']:
                    times = event_data['times']
                    all_times.extend(times)
                
                if all_times and len(all_times) > 0:
                    # Create histogram/density of event times
                    # Use binsize for consistency with PETH
                    time_bins = np.arange(minmax[0], minmax[1] + binsize, binsize)
                    hist_counts, bin_edges = np.histogram(all_times, bins=time_bins)
                    
                    # Normalize by number of patients to get average event count per patient
                    n_patients = len(channel_data['events_list'])
                    avg_counts = hist_counts / n_patients
                    
                    # Create time axis for histogram (bin centers)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    # Plot as a bar plot or line plot showing average event density
                    ax_raster.bar(bin_centers, avg_counts, width=binsize*0.8, 
                                 color='blue', alpha=0.6, edgecolor='darkblue', linewidth=0.5)
                    ax_raster.set_ylabel("Avg Events/Bin", fontsize=10)
                    ax_raster.set_title(f"Avg Event Density - {ch_name} (n={n_patients} patients)", 
                                       fontsize=11, fontweight='bold')
                    ax_raster.axvline(0.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
                    ax_raster.set_xlim(minmax)
                    ax_raster.grid(True, alpha=0.3, axis='y')
                    ax_raster.tick_params(axis='x', labelsize=8, bottom=True)
                    ax_raster.tick_params(axis='y', labelsize=8)
            except Exception as e:
                ax_raster.text(0.5, 0.5, f'Raster Error: {str(e)[:40]}', 
                             ha='center', va='center', transform=ax_raster.transAxes, fontsize=8, color='red')
        else:
            ax_raster.text(0.5, 0.5, 'No event data', 
                         ha='center', va='center', transform=ax_raster.transAxes, fontsize=9)
        
        # Set x-label and ticks for all subplots
        ax_peth.set_xlabel("Time from R-peak (s)", fontsize=10)
        ax_raster.set_xlabel("Time from R-peak (s)", fontsize=10)
        
        # Ensure x-axis ticks are visible
        ax_peth.tick_params(labelsize=8, axis='both', which='major')
        ax_raster.tick_params(labelsize=8, axis='both', which='major')
        ax_peth.tick_params(axis='x', labelsize=8, bottom=True)
        ax_raster.tick_params(axis='x', labelsize=8, bottom=True)
    
    plt.suptitle(f"Average EEG Channel PETH & Raster Plots (Across All Patients)", 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save plot if requested
    if save_plot and save_dir:
        save_path_avg = os.path.join(save_dir, 'eeg_peth_raster_averaged.png')
        plt.savefig(save_path_avg, dpi=300, bbox_inches='tight')
        if 'st' in globals():
            st.success(f"Averaged EEG PETH & Raster plots saved to {save_path_avg}")
        else:
            print(f"Averaged EEG PETH & Raster plots saved to {save_path_avg}")
    
    # Display in streamlit if available
    if 'st' in globals():
        st.pyplot(fig_avg)
    else:
        plt.show()
    
    plt.close(fig_avg)
    
    return fig_avg


def plot_regional_averaged_eeg_peth_raster(loaded_files, minmax=(-0.5, 1.0), binsize=0.01, save_plot=False, save_dir=None):
    """
    Create averaged PETH and raster plots for regional EEG groups (F, C, T, P, O) across all patients.
    Only uses the first 21 electrodes and groups them by prefix.
    
    Parameters
    ----------
    loaded_files : list
        List of tuples (file_path, patient_id, raw_object)
    minmax : tuple, optional
        Time window around events in seconds (default: (-0.5, 1.0))
    binsize : float, optional
        Bin size for rate calculation in seconds (default: 0.01)
    save_plot : bool, optional
        Whether to save the plot (default: False)
    save_dir : str, optional
        Directory to save the plot (default: None)
    
    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if plot was created, None otherwise
    """
    if not PYNAPPLE_AVAILABLE:
        if 'st' in globals():
            st.error("pynapple is not available. Please install it to use regional PETH raster plots.")
        else:
            print("pynapple is not available. Please install it to use regional PETH raster plots.")
        return None
    
    if not loaded_files:
        return None
    
    # Regional prefixes to group
    regional_prefixes = ['F', 'C', 'T', 'P', 'O']
    
    # Collect data from all patients, grouped by region
    regional_data = {prefix: {'rates_list': [], 'events_list': []} for prefix in regional_prefixes}
    
    # Track unique electrodes per region across all patients
    regional_electrodes = {prefix: set() for prefix in regional_prefixes}
    
    if 'st' in globals():
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    for file_idx, (file_path, patient_id, raw) in enumerate(loaded_files):
        try:
            if 'st' in globals():
                status_text.text(f"Processing {patient_id} for regional plots ({file_idx + 1}/{len(loaded_files)})...")
                progress_bar.progress((file_idx + 1) / len(loaded_files))
            
            # Get sampling frequency
            sfreq = raw.info['sfreq']
            
            # Get channel names and data
            ch_names = raw.ch_names
            data = raw.get_data()
            
            # Find ECG channel
            ch_lower = [ch.lower() for ch in ch_names]
            ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
            
            if not ecg_indices:
                continue
            
            ecg_ch_idx = ecg_indices[0]
            ecg_signal = data[ecg_ch_idx, :]
            
            # Clean ECG and detect R-peaks
            try:
                ecg_signal_clean, _ = clean_ecg_advanced(
                    ecg_signal, 
                    sampling_rate=sfreq,
                    median_window_ms=300,
                    wavelet='db4',
                    wavelet_levels=5,
                    zscore_threshold=3.0
                )
                _, rpk = nk.ecg_peaks(ecg_signal_clean, sampling_rate=sfreq)
                rpeaks = rpk['ECG_R_Peaks']
            except:
                continue
            
            if len(rpeaks) < 2:
                continue
            
            # Convert R-peak indices to timestamps
            rpeak_times = rpeaks / sfreq
            rpeak_ts = nap.Ts(t=rpeak_times, time_units="s")
            
            # Find EEG channels (first 21 only)
            eeg_ch_indices = []
            eeg_ch_names = []
            for i, ch in enumerate(ch_names):
                ch_upper = ch.upper()
                # Check if it's an EEG channel with regional prefix
                if any(ch_upper.startswith(prefix) for prefix in regional_prefixes):
                    eeg_ch_indices.append(i)
                    eeg_ch_names.append(ch)
                    if len(eeg_ch_indices) >= 21:  # Only first 21 electrodes
                        break
            
            if not eeg_ch_indices:
                continue
            
            # Group channels by regional prefix
            regional_channels = {prefix: [] for prefix in regional_prefixes}
            for ch_idx, ch_name in zip(eeg_ch_indices, eeg_ch_names):
                ch_upper = ch_name.upper()
                for prefix in regional_prefixes:
                    if ch_upper.startswith(prefix):
                        regional_channels[prefix].append((ch_idx, ch_name))
                        regional_electrodes[prefix].add(ch_name.upper())  # Track unique electrodes
                        break
            
            # Process each region
            for region_prefix in regional_prefixes:
                if not regional_channels[region_prefix]:
                    continue
                
                # Collect events from all channels in this region
                all_region_events = []
                
                for ch_idx, ch_name in regional_channels[region_prefix]:
                    try:
                        eeg_signal = data[ch_idx, :]
                        
                        # Detect events in EEG
                        eeg_mean = np.mean(eeg_signal)
                        eeg_std = np.std(eeg_signal)
                        threshold = eeg_mean + 2 * eeg_std
                        
                        # Find peaks above threshold
                        peaks, _ = find_peaks(eeg_signal, height=threshold, distance=int(sfreq * 0.1))
                        
                        if len(peaks) > 0:
                            # Convert peak indices to timestamps
                            peak_times = peaks / sfreq
                            all_region_events.extend(peak_times)
                    
                    except Exception as e:
                        continue
                
                if len(all_region_events) > 0:
                    # Create pynapple Ts object for all events in this region
                    region_peak_ts = nap.Ts(t=np.array(all_region_events), time_units="s")
                    
                    # Compute perievent alignment around R-peaks
                    peth_region = nap.compute_perievent(
                        timestamps=region_peak_ts,
                        tref=rpeak_ts,
                        minmax=minmax,
                        time_unit="s"
                    )
                    
                    if len(peth_region) > 0:
                        # Calculate rates for this patient and region
                        try:
                            peth_region_count = peth_region.count(binsize)
                            if hasattr(peth_region_count, 'values'):
                                count_vals = peth_region_count.values
                                if count_vals.ndim == 2:
                                    rates = np.mean(count_vals, axis=1) / binsize
                                else:
                                    rates = count_vals / binsize
                                
                                if hasattr(peth_region_count, 'index'):
                                    time_axis = peth_region_count.index.values
                                else:
                                    time_axis = np.arange(len(rates)) * binsize + minmax[0]
                                
                                regional_data[region_prefix]['rates_list'].append((time_axis, rates))
                        except:
                            pass
                        
                        # Collect events for raster plot
                        try:
                            peth_region_tsd = peth_region.to_tsd()
                            regional_data[region_prefix]['events_list'].append({
                                'times': peth_region_tsd.index.values,
                                'events': peth_region_tsd.values
                            })
                        except:
                            pass
                    
        except Exception as e:
            if 'st' in globals():
                st.warning(f"Error processing {patient_id} for regional plots: {e}")
            else:
                print(f"Error processing {patient_id} for regional plots: {e}")
            continue
        
        if 'st' in globals():
            progress_bar.empty()
            status_text.empty()
    
    # Check if we have data
    regions_with_data = [prefix for prefix in regional_prefixes if regional_data[prefix]['rates_list'] or regional_data[prefix]['events_list']]
    
    if not regions_with_data:
        if 'st' in globals():
            st.warning("No regional data found across patients.")
        else:
            print("No regional data found across patients.")
        return None
    
    # Count electrodes per region
    region_electrode_counts = {prefix: len(regional_electrodes[prefix]) for prefix in regional_prefixes}
    
    # Create plots
    n_regions = len(regions_with_data)
    n_cols = 2  # PETH and Raster side by side
    n_rows = n_regions
    
    fig_regional, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), sharex='col')
    if n_regions == 1:
        axes = axes.reshape(1, -1)
    
    for reg_idx, region_prefix in enumerate(regions_with_data):
        ax_peth = axes[reg_idx, 0]
        ax_raster = axes[reg_idx, 1]
        
        region_data = regional_data[region_prefix]
        
        # Plot 1: Average PETH
        if region_data['rates_list']:
            try:
                # Interpolate all rates to common time axis
                all_time_axes = [t for t, _ in region_data['rates_list']]
                all_rates = [r for _, r in region_data['rates_list']]
                
                # Find common time range
                min_time = max([t[0] for t in all_time_axes])
                max_time = min([t[-1] for t in all_time_axes])
                common_time = np.arange(min_time, max_time + binsize, binsize)
                
                # Interpolate each patient's rates to common time axis
                interpolated_rates = []
                for time_axis, rates in zip(all_time_axes, all_rates):
                    if len(time_axis) > 1 and len(rates) > 1:
                        interp_rates = np.interp(common_time, time_axis, rates)
                        interpolated_rates.append(interp_rates)
                
                if interpolated_rates:
                    # Average across patients
                    avg_rates = np.mean(interpolated_rates, axis=0)
                    std_rates = np.std(interpolated_rates, axis=0)
                    
                    ax_peth.plot(common_time, avg_rates, linewidth=2, color='blue', label='Mean')
                    ax_peth.fill_between(common_time, avg_rates - std_rates, avg_rates + std_rates, 
                                        alpha=0.3, color='blue', label='±1 SD')
                    ax_peth.set_ylabel("Rate (events/sec)", fontsize=10)
                    n_electrodes = region_electrode_counts.get(region_prefix, 0)
                    ax_peth.set_title(f"Avg PETH - {region_prefix} Region (n={len(interpolated_rates)} patients, {n_electrodes} electrodes)", 
                                     fontsize=11, fontweight='bold')
                    ax_peth.axvline(0.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
                    ax_peth.legend(fontsize=8)
                    ax_peth.grid(True, alpha=0.3)
                    ax_peth.set_xlim(minmax)
                    ax_peth.tick_params(axis='x', labelsize=8, bottom=True)
            except Exception as e:
                ax_peth.text(0.5, 0.5, f'PETH Error: {str(e)[:40]}', 
                           ha='center', va='center', transform=ax_peth.transAxes, fontsize=8, color='red')
        else:
            ax_peth.text(0.5, 0.5, 'No rate data', 
                       ha='center', va='center', transform=ax_peth.transAxes, fontsize=9)
        
        # Plot 2: Average Event Density
        if region_data['events_list']:
            try:
                # Collect all event times from all patients
                all_times = []
                for event_data in region_data['events_list']:
                    times = event_data['times']
                    all_times.extend(times)
                
                if all_times and len(all_times) > 0:
                    # Create histogram/density of event times
                    time_bins = np.arange(minmax[0], minmax[1] + binsize, binsize)
                    hist_counts, bin_edges = np.histogram(all_times, bins=time_bins)
                    
                    # Normalize by number of patients
                    n_patients = len(region_data['events_list'])
                    avg_counts = hist_counts / n_patients
                    
                    # Create time axis for histogram (bin centers)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    # Plot as bar plot showing average event density
                    ax_raster.bar(bin_centers, avg_counts, width=binsize*0.8, 
                                 color='#0C7BDC', alpha=0.6, edgecolor='#0A5FA8', linewidth=0.5)
                    ax_raster.set_ylabel("Avg Events/Bin", fontsize=10)
                    n_electrodes = region_electrode_counts.get(region_prefix, 0)
                    ax_raster.set_title(f"Avg Event Density - {region_prefix} Region (n={n_patients} patients, {n_electrodes} electrodes)", 
                                       fontsize=11, fontweight='bold')
                    ax_raster.axvline(0.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
                    ax_raster.set_xlim(minmax)
                    ax_raster.grid(True, alpha=0.3, axis='y')
                    ax_raster.tick_params(axis='x', labelsize=8, bottom=True)
                    ax_raster.tick_params(axis='y', labelsize=8)
            except Exception as e:
                ax_raster.text(0.5, 0.5, f'Raster Error: {str(e)[:40]}', 
                             ha='center', va='center', transform=ax_raster.transAxes, fontsize=8, color='red')
        else:
            ax_raster.text(0.5, 0.5, 'No event data', 
                         ha='center', va='center', transform=ax_raster.transAxes, fontsize=9)
        
        # Set x-label and ticks for all subplots
        ax_peth.set_xlabel("Time from R-peak (s)", fontsize=10)
        ax_raster.set_xlabel("Time from R-peak (s)", fontsize=10)
        
        # Ensure x-axis ticks are visible
        ax_peth.tick_params(labelsize=8, axis='both', which='major')
        ax_raster.tick_params(labelsize=8, axis='both', which='major')
        ax_peth.tick_params(axis='x', labelsize=8, bottom=True)
        ax_raster.tick_params(axis='x', labelsize=8, bottom=True)
    
    # Create title with electrode counts for each region
    region_counts_str = ", ".join([f"{r}({region_electrode_counts.get(r, 0)})" for r in regions_with_data])
    plt.suptitle(f"Regional Average EEG PETH & Raster Plots ({region_counts_str} electrodes)", 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save plot if requested
    if save_plot and save_dir:
        save_path_regional = os.path.join(save_dir, 'eeg_peth_raster_regional_averaged.png')
        plt.savefig(save_path_regional, dpi=300, bbox_inches='tight')
        if 'st' in globals():
            st.success(f"Regional averaged EEG PETH & Raster plots saved to {save_path_regional}")
        else:
            print(f"Regional averaged EEG PETH & Raster plots saved to {save_path_regional}")
    
    # Display in streamlit if available
    if 'st' in globals():
        st.pyplot(fig_regional)
    else:
        plt.show()
    
    plt.close(fig_regional)
    
    return fig_regional


def bootstrap_pvalue_test(rates_list, test_window=(-0.1, 0.1), baseline_window=(-0.5, -0.5), n_bootstrap=100):
    """
    Perform bootstrapping test to compare event rates in test window vs baseline window.
    
    Parameters
    ----------
    rates_list : list
        List of (time_axis, rates) tuples from different patients
    test_window : tuple
        Time window around R-peak to test (default: (-0.1, 0.1))
    baseline_window : tuple
        Baseline time window before R-peak (default: (-0.5, -0.2))
    n_bootstrap : int
        Number of bootstrap iterations (default: 10000)
    
    Returns
    -------
    float : p-value from bootstrapping test
    float : observed mean difference (test - baseline)
    """
    if not rates_list or len(rates_list) < 2:
        return np.nan, np.nan
    
    # Collect differences for each patient
    differences = []
    
    for time_axis, rates in rates_list:
        if len(time_axis) < 2 or len(rates) < 2:
            continue
        
        # Find indices for test and baseline windows
        test_mask = (time_axis >= test_window[0]) & (time_axis <= test_window[1])
        baseline_mask = (time_axis >= baseline_window[0]) & (time_axis <= baseline_window[1])
        
        if np.sum(test_mask) == 0 or np.sum(baseline_mask) == 0:
            continue
        
        # Calculate mean rates in each window
        test_rate = np.mean(rates[test_mask])
        baseline_rate = np.mean(rates[baseline_mask])
        
        # Calculate difference
        diff = test_rate - baseline_rate
        differences.append(diff)
    
    if len(differences) < 2:
        return np.nan, np.nan
    
    differences = np.array(differences)
    observed_mean_diff = np.mean(differences)
    
    # Bootstrapping test: test if mean difference is significantly different from zero
    # Null hypothesis: mean difference = 0
    n_samples = len(differences)
    
    # Center the differences to create null distribution (mean = 0)
    centered_differences = differences - observed_mean_diff
    
    # Bootstrap: resample from centered differences and calculate means using pynapple Randomization
    if not PYNAPPLE_AVAILABLE:
        # Fallback to numpy if pynapple is not available
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(centered_differences, size=n_samples, replace=True)
            bootstrap_mean = np.mean(bootstrap_sample)
            bootstrap_means.append(bootstrap_mean)
        bootstrap_means = np.array(bootstrap_means)
    else:
        # Use pynapple Randomization for bootstrap resampling
        # Create a Ts object with indices as timestamps for resampling
        indices_ts = nap.Ts(t=np.arange(n_samples, dtype=float), time_units="s")
        bootstrap_means = []
        for _ in range(n_bootstrap):
            # Resample timestamps (indices) using pynapple's resample_timestamps
            resampled_indices_ts = nap.resample_timestamps(indices_ts)
            # Extract timestamps and convert to integer indices
            # resample_timestamps returns timestamps within [0, n_samples) range
            resampled_timestamps = resampled_indices_ts.index.values
            # Convert to integer indices, ensuring they're within valid range
            resampled_indices = np.clip(np.round(resampled_timestamps).astype(int), 0, n_samples - 1)
            # Use indices to sample from centered_differences
            bootstrap_sample = centered_differences[resampled_indices]
            bootstrap_mean = np.mean(bootstrap_sample)
            bootstrap_means.append(bootstrap_mean)
        bootstrap_means = np.array(bootstrap_means)
    
    # Calculate p-value: proportion of bootstrap means that are as extreme or more extreme
    # Two-tailed test: how often do we get a mean as extreme as observed under null hypothesis?
    p_value = np.mean(np.abs(bootstrap_means) >= np.abs(observed_mean_diff))
    
    # Ensure p-value is at least 1/n_bootstrap (minimum resolution)
    p_value = max(p_value, 1.0 / n_bootstrap)
    
    return p_value, observed_mean_diff


def test_electrode_significance_per_patient(loaded_files, minmax=(-0.5, 1.0), binsize=0.01, 
                                            test_window=(-0.1, 0.1), baseline_window=(-0.5, -0.2), 
                                            n_bootstrap=10000, significance_threshold=0.05):
    """
    Test significance of RR to channel peri-event analysis for each patient individually.
    For each patient, tests which electrodes show significant differences between test and baseline windows.
    
    Parameters
    ----------
    loaded_files : list
        List of tuples (file_path, patient_id, raw_object)
    minmax : tuple, optional
        Time window around events in seconds (default: (-0.5, 1.0))
    binsize : float, optional
        Bin size for rate calculation in seconds (default: 0.01)
    test_window : tuple, optional
        Time window around R-peak to test (default: (-0.1, 0.1))
    baseline_window : tuple, optional
        Baseline time window before R-peak (default: (-0.5, -0.2))
    n_bootstrap : int, optional
        Number of bootstrap iterations (default: 10000)
    significance_threshold : float, optional
        P-value threshold for significance (default: 0.05)
    
    Returns
    -------
    dict : Dictionary with patient_id as keys and lists of significant electrodes as values
    """
    if not PYNAPPLE_AVAILABLE:
        if 'st' in globals():
            st.error("pynapple is not available. Please install it to use electrode significance testing.")
        else:
            print("pynapple is not available. Please install it to use electrode significance testing.")
        return {}
    
    if not loaded_files:
        return {}
    
    results = {}  # {patient_id: [(electrode, p_value, mean_diff), ...]}
    
    if 'st' in globals():
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    for file_idx, (file_path, patient_id, raw) in enumerate(loaded_files):
        try:
            if 'st' in globals():
                status_text.text(f"Testing significance for {patient_id} ({file_idx + 1}/{len(loaded_files)})...")
                progress_bar.progress((file_idx + 1) / len(loaded_files))
            
            # Get sampling frequency
            sfreq = raw.info['sfreq']
            
            # Get channel names and data
            ch_names = raw.ch_names
            data = raw.get_data()
            
            # Find ECG channel
            ch_lower = [ch.lower() for ch in ch_names]
            ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
            
            if not ecg_indices:
                continue
            
            ecg_ch_idx = ecg_indices[0]
            ecg_signal = data[ecg_ch_idx, :]
            
            # Clean ECG and detect R-peaks
            try:
                ecg_signal_clean, _ = clean_ecg_advanced(
                    ecg_signal, 
                    sampling_rate=sfreq,
                    median_window_ms=300,
                    wavelet='db4',
                    wavelet_levels=5,
                    zscore_threshold=3.0
                )
                _, rpk = nk.ecg_peaks(ecg_signal_clean, sampling_rate=sfreq)
                rpeaks = rpk['ECG_R_Peaks']
            except Exception as e:
                continue
            
            if len(rpeaks) < 2:
                continue
            
            # Convert R-peak indices to timestamps
            rpeak_times = rpeaks / sfreq
            rpeak_ts = nap.Ts(t=rpeak_times, time_units="s")
            
            # Find EEG channels
            eeg_ch_indices = [i for i, ch in enumerate(ch_names) 
                            if 'eeg' in ch.lower() or any(elec in ch.upper() for elec in ['FP', 'F', 'C', 'P', 'O', 'T', 'A'])]
            
            if not eeg_ch_indices:
                continue
            
            # Get EEG data
            eeg_data = data[eeg_ch_indices, :]
            eeg_ch_names = [ch_names[i] for i in eeg_ch_indices]
            
            # Store results for this patient
            patient_results = []
            
            # Process each EEG channel
            for ch_idx, ch_name in enumerate(eeg_ch_names):
                try:
                    eeg_signal = eeg_data[ch_idx, :]
                    
                    # Detect events in EEG
                    eeg_mean = np.mean(eeg_signal)
                    eeg_std = np.std(eeg_signal)
                    threshold = eeg_mean + 2 * eeg_std
                    
                    # Find peaks above threshold
                    peaks, _ = find_peaks(eeg_signal, height=threshold, distance=int(sfreq * 0.1))
                    
                    if len(peaks) < 10:  # Need sufficient events for bootstrap
                        continue
                    
                    # Convert peak indices to timestamps
                    peak_times = peaks / sfreq
                    eeg_peak_ts = nap.Ts(t=peak_times, time_units="s")
                    
                    # Compute perievent alignment around R-peaks
                    peth_eeg = nap.compute_perievent(
                        timestamps=eeg_peak_ts,
                        tref=rpeak_ts,
                        minmax=minmax,
                        time_unit="s"
                    )
                    
                    if len(peth_eeg) == 0:
                        continue
                    
                    # Calculate rates for this electrode
                    # For single-patient bootstrap, we need multiple samples
                    # Split R-peaks into groups and compute separate PETHs for each group
                    try:
                        # Split R-peaks into groups (e.g., 5 groups for better statistics)
                        n_groups = min(10, len(rpeaks) // 5)  # At least 5 R-peaks per group
                        if n_groups < 2:
                            continue
                        
                        group_size = len(rpeaks) // n_groups
                        rates_list = []
                        
                        for group_idx in range(n_groups):
                            start_idx = group_idx * group_size
                            end_idx = (group_idx + 1) * group_size if group_idx < n_groups - 1 else len(rpeaks)
                            
                            # Get R-peaks for this group
                            group_rpeaks = rpeaks[start_idx:end_idx]
                            if len(group_rpeaks) < 2:
                                continue
                            
                            # Create Ts object for this group's R-peaks
                            group_rpeak_times = group_rpeaks / sfreq
                            group_rpeak_ts = nap.Ts(t=group_rpeak_times, time_units="s")
                            
                            # Compute perievent alignment for this group
                            group_peth_eeg = nap.compute_perievent(
                                timestamps=eeg_peak_ts,
                                tref=group_rpeak_ts,
                                minmax=minmax,
                                time_unit="s"
                            )
                            
                            if len(group_peth_eeg) == 0:
                                continue
                            
                            # Calculate rates for this group
                            group_peth_eeg_count = group_peth_eeg.count(binsize)
                            if hasattr(group_peth_eeg_count, 'values'):
                                group_count_vals = group_peth_eeg_count.values
                                if group_count_vals.ndim == 2:
                                    group_rates = np.mean(group_count_vals, axis=1) / binsize
                                else:
                                    group_rates = group_count_vals / binsize
                                
                                if hasattr(group_peth_eeg_count, 'index'):
                                    group_time_axis = group_peth_eeg_count.index.values
                                else:
                                    group_time_axis = np.arange(len(group_rates)) * binsize + minmax[0]
                                
                                rates_list.append((group_time_axis, group_rates))
                        
                        # Test significance using bootstrap with multiple groups
                        if len(rates_list) >= 2:
                            p_value, mean_diff = bootstrap_pvalue_test(
                                rates_list, 
                                test_window=test_window,
                                baseline_window=baseline_window,
                                n_bootstrap=n_bootstrap
                            )
                            
                            if not np.isnan(p_value):
                                patient_results.append({
                                    'electrode': ch_name,
                                    'p_value': p_value,
                                    'mean_diff': mean_diff,
                                    'significant': p_value < significance_threshold
                                })
                    
                    except Exception as e:
                        continue
                
                except Exception as e:
                    continue
            
            # Store results for this patient
            if patient_results:
                results[patient_id] = patient_results
        
        except Exception as e:
            continue
    
    if 'st' in globals():
        progress_bar.empty()
        status_text.empty()
    
    return results


def plot_significance_percentage_topomap(significance_results, save_plot=False, save_dir=None):
    """
    Plot an MNE topomap showing the percentage of patients with significant electrodes.
    
    Parameters
    ----------
    significance_results : dict
        Dictionary returned from test_electrode_significance_per_patient
        Format: {patient_id: [{'electrode': str, 'p_value': float, 'mean_diff': float, 'significant': bool}, ...]}
    save_plot : bool, optional
        Whether to save the plot (default: False)
    save_dir : str, optional
        Directory to save the plot (default: None)
    
    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if plot was created, None otherwise
    """
    if not significance_results:
        if 'st' in globals():
            st.warning("No significance results available for topomap.")
        else:
            print("No significance results available for topomap.")
        return None
    
    # Count significant and total occurrences for each electrode
    electrode_counts = {}  # {electrode: {'significant': count, 'total': count}}
    
    for patient_id, patient_results in significance_results.items():
        for result in patient_results:
            electrode = result['electrode']
            if electrode not in electrode_counts:
                electrode_counts[electrode] = {'significant': 0, 'total': 0}
            
            electrode_counts[electrode]['total'] += 1
            if result['significant']:
                electrode_counts[electrode]['significant'] += 1
    
    # Calculate percentages
    electrode_percentages = {}
    for electrode, counts in electrode_counts.items():
        if counts['total'] > 0:
            percentage = (counts['significant'] / counts['total']) * 100
            electrode_percentages[electrode] = percentage
    
    if not electrode_percentages:
        if 'st' in globals():
            st.warning("No electrode data available for topomap.")
        else:
            print("No electrode data available for topomap.")
        return None
    
    # Need at least 3 electrodes for a meaningful topomap
    if len(electrode_percentages) < 3:
        if 'st' in globals():
            st.warning(f"Insufficient electrodes ({len(electrode_percentages)}) for topomap. Need at least 3.")
        else:
            print(f"Insufficient electrodes ({len(electrode_percentages)}) for topomap. Need at least 3.")
        return None
    
    try:
        # Create montage (standard 10-20 system)
        montage = mne.channels.make_standard_montage('standard_1020')
        
        # Get available channels and values
        available_channels = []
        values = []
        used_electrodes = set()  # Track which electrodes we've matched
        
        # Normalize montage channel names (uppercase)
        montage_ch_names_upper = {ch.upper(): ch for ch in montage.ch_names}
        
        for electrode, percentage in electrode_percentages.items():
            if electrode in used_electrodes:
                continue
                
            electrode_upper = electrode.upper()
            matched = False
            
            # Strategy 1: Exact match (case-insensitive)
            if electrode_upper in montage_ch_names_upper:
                available_channels.append(montage_ch_names_upper[electrode_upper])
                values.append(percentage)
                used_electrodes.add(electrode)
                matched = True
                continue
            
            # Strategy 2: Try common variations
            # Remove common prefixes/suffixes and try again
            electrode_clean = electrode_upper
            # Remove common suffixes like '_EEG', '-EEG', etc.
            for suffix in ['_EEG', '-EEG', '_E', '-E', ' EEG', ' E']:
                if electrode_clean.endswith(suffix):
                    electrode_clean = electrode_clean[:-len(suffix)]
                    break
            
            if electrode_clean in montage_ch_names_upper:
                available_channels.append(montage_ch_names_upper[electrode_clean])
                values.append(percentage)
                used_electrodes.add(electrode)
                matched = True
                continue
            
            # Strategy 3: Try matching by first 2-3 characters (for regional matching)
            # This handles cases like 'Fp1' matching 'FP1', 'F3' matching 'F3', etc.
            if len(electrode_clean) >= 2:
                prefix = electrode_clean[:2]
                # Try to find montage channels that start with the same prefix
                for montage_ch_upper, montage_ch_orig in montage_ch_names_upper.items():
                    if montage_ch_upper.startswith(prefix) and montage_ch_orig not in available_channels:
                        # Check if the numbers match (e.g., Fp1 should match FP1, not FP2)
                        # Extract numbers from both
                        elec_num = ''.join([c for c in electrode_clean if c.isdigit()])
                        montage_num = ''.join([c for c in montage_ch_upper if c.isdigit()])
                        
                        # If numbers match or both have no numbers, it's a good match
                        if elec_num == montage_num or (not elec_num and not montage_num):
                            available_channels.append(montage_ch_orig)
                            values.append(percentage)
                            used_electrodes.add(electrode)
                            matched = True
                            break
                    
                    if matched:
                        break
        
        if len(available_channels) < 3:
            if 'st' in globals():
                st.warning(f"Could not match enough electrodes to montage. Found {len(available_channels)} matches.")
            else:
                print(f"Could not match enough electrodes to montage. Found {len(available_channels)} matches.")
            return None
        
        # Create MNE info object
        info = mne.create_info(ch_names=available_channels, sfreq=256, ch_types='eeg')
        info.set_montage(montage)
        
        # Create EvokedArray with percentage values
        values_array = np.array(values).reshape(-1, 1)
        evoked = mne.EvokedArray(values_array, info)
        
        # Calculate min and max for colorbar limits
        vmin = np.min(values) if len(values) > 0 else 0
        vmax = np.max(values) if len(values) > 0 else 100
        
        # Plot topomap
        fig, ax = plt.subplots(figsize=(10, 8))
        im, _ = mne.viz.plot_topomap(
            evoked.data[:, 0], 
            evoked.info, 
            axes=ax, 
            show=False,
            cmap='Reds',
            vlim=(vmin, vmax)  # Use actual min and max of percentage values
        )
        fig.colorbar(im, ax=ax, label='% of Patients with Significant Electrode')
        ax.set_title('Percentage of Patients with Significant RR-to-Channel Coupling\n(Per Electrode)', 
                    fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'electrode_significance_percentage_topomap.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if 'st' in globals():
                st.success(f"Topomap saved to {save_path}")
            else:
                print(f"Topomap saved to {save_path}")
        
        # Display in streamlit if available
        if 'st' in globals():
            st.pyplot(fig)
        else:
            plt.show()
        
        return fig
        
    except Exception as e:
        if 'st' in globals():
            st.error(f"Error creating topomap: {e}")
        else:
            print(f"Error creating topomap: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_regional_electrode_averaged_eeg(loaded_files, minmax=(-0.5, 1.0), binsize=0.01, save_plot=False, save_dir=None):
    """
    Create averaged PETH and raster plots for each electrode within each region (F, P, T, O, C).
    Each region gets its own figure with subplots for each electrode, averaged across all patients.
    
    Parameters
    ----------
    loaded_files : list
        List of tuples (file_path, patient_id, raw_object)
    minmax : tuple, optional
        Time window around events in seconds (default: (-0.5, 1.0))
    binsize : float, optional
        Bin size for rate calculation in seconds (default: 0.01)
    save_plot : bool, optional
        Whether to save the plot (default: False)
    save_dir : str, optional
        Directory to save the plot (default: None)
    
    Returns
    -------
    list : List of matplotlib.figure.Figure objects (one per region)
    """
    if not PYNAPPLE_AVAILABLE:
        if 'st' in globals():
            st.error("pynapple is not available. Please install it to use regional electrode plots.")
        else:
            print("pynapple is not available. Please install it to use regional electrode plots.")
        return []
    
    if not loaded_files:
        return []
    
    # Define electrode groupings by region
    regional_electrodes = {
        'F': ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
        'P': ['P3', 'Pz', 'P4'],
        'T': ['T3', 'T4', 'T5', 'T6'],
        'O': ['O1', 'Oz', 'O2'],
        'C': ['C3', 'Cz', 'C4']
    }
    
    # Collect data from all patients, grouped by region and electrode
    # Structure: {region: {electrode: {'rates_list': [], 'events_list': []}}}
    all_electrode_data = {}
    for region in regional_electrodes.keys():
        all_electrode_data[region] = {}
        for electrode in regional_electrodes[region]:
            all_electrode_data[region][electrode] = {'rates_list': [], 'events_list': []}
    
    if 'st' in globals():
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    for file_idx, (file_path, patient_id, raw) in enumerate(loaded_files):
        try:
            if 'st' in globals():
                status_text.text(f"Processing {patient_id} for regional electrode plots ({file_idx + 1}/{len(loaded_files)})...")
                progress_bar.progress((file_idx + 1) / len(loaded_files))
            
            # Get sampling frequency
            sfreq = raw.info['sfreq']
            
            # Get channel names and data
            ch_names = raw.ch_names
            data = raw.get_data()
            
            # Find ECG channel
            ch_lower = [ch.lower() for ch in ch_names]
            ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
            
            if not ecg_indices:
                continue
            
            ecg_ch_idx = ecg_indices[0]
            ecg_signal = data[ecg_ch_idx, :]
            
            # Clean ECG and detect R-peaks
            try:
                ecg_signal_clean, _ = clean_ecg_advanced(
                    ecg_signal, 
                    sampling_rate=sfreq,
                    median_window_ms=300,
                    wavelet='db4',
                    wavelet_levels=5,
                    zscore_threshold=3.0
                )
                _, rpk = nk.ecg_peaks(ecg_signal_clean, sampling_rate=sfreq)
                rpeaks = rpk['ECG_R_Peaks']
            except:
                continue
            
            if len(rpeaks) < 2:
                continue
            
            # Convert R-peak indices to timestamps
            rpeak_times = rpeaks / sfreq
            rpeak_ts = nap.Ts(t=rpeak_times, time_units="s")
            
            # Create mapping of channel names to indices
            ch_name_to_idx = {ch.upper(): i for i, ch in enumerate(ch_names)}
            
            # Process each region and electrode
            for region, electrodes in regional_electrodes.items():
                for electrode in electrodes:
                    electrode_upper = electrode.upper()
                    
                    # Find channel index
                    if electrode_upper not in ch_name_to_idx:
                        continue
                    
                    ch_idx = ch_name_to_idx[electrode_upper]
                    
                    try:
                        eeg_signal = data[ch_idx, :]
                        
                        # Detect events in EEG
                        eeg_mean = np.mean(eeg_signal)
                        eeg_std = np.std(eeg_signal)
                        threshold = eeg_mean + 2 * eeg_std
                        
                        # Find peaks above threshold
                        peaks, _ = find_peaks(eeg_signal, height=threshold, distance=int(sfreq * 0.1))
                        
                        if len(peaks) > 0:
                            # Convert peak indices to timestamps
                            peak_times = peaks / sfreq
                            eeg_peak_ts = nap.Ts(t=peak_times, time_units="s")
                            
                            # Compute perievent alignment around R-peaks
                            peth_eeg = nap.compute_perievent(
                                timestamps=eeg_peak_ts,
                                tref=rpeak_ts,
                                minmax=minmax,
                                time_unit="s"
                            )
                            
                            if len(peth_eeg) > 0:
                                # Calculate rates for this patient and electrode
                                try:
                                    peth_eeg_count = peth_eeg.count(binsize)
                                    if hasattr(peth_eeg_count, 'values'):
                                        count_vals = peth_eeg_count.values
                                        if count_vals.ndim == 2:
                                            rates = np.mean(count_vals, axis=1) / binsize
                                        else:
                                            rates = count_vals / binsize
                                        
                                        if hasattr(peth_eeg_count, 'index'):
                                            time_axis = peth_eeg_count.index.values
                                        else:
                                            time_axis = np.arange(len(rates)) * binsize + minmax[0]
                                        
                                        all_electrode_data[region][electrode]['rates_list'].append((time_axis, rates))
                                except:
                                    pass
                                
                                # Collect events for raster plot
                                try:
                                    peth_eeg_tsd = peth_eeg.to_tsd()
                                    all_electrode_data[region][electrode]['events_list'].append({
                                        'times': peth_eeg_tsd.index.values,
                                        'events': peth_eeg_tsd.values
                                    })
                                except:
                                    pass
                    
                    except Exception as e:
                        continue
        
        except Exception as e:
            if 'st' in globals():
                st.warning(f"Error processing {patient_id} for regional electrode plots: {e}")
            else:
                print(f"Error processing {patient_id} for regional electrode plots: {e}")
            continue
    
    if 'st' in globals():
        progress_bar.empty()
        status_text.empty()
    
    # Create plots for each region
    figures = []
    
    for region, electrodes in regional_electrodes.items():
        # Filter electrodes that have data
        electrodes_with_data = [e for e in electrodes if all_electrode_data[region][e]['rates_list'] or all_electrode_data[region][e]['events_list']]
        
        if not electrodes_with_data:
            continue
        
        # Create figure for this region
        n_electrodes = len(electrodes_with_data)
        n_cols = 2  # PETH and Raster side by side
        n_rows = n_electrodes
        
        fig_region, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), sharex='col')
        if n_electrodes == 1:
            axes = axes.reshape(1, -1)
        
        for elec_idx, electrode in enumerate(electrodes_with_data):
            ax_peth = axes[elec_idx, 0]
            ax_raster = axes[elec_idx, 1]
            
            electrode_data = all_electrode_data[region][electrode]
            
            # Plot 1: Average PETH
            if electrode_data['rates_list']:
                try:
                    # Interpolate all rates to common time axis
                    all_time_axes = [t for t, _ in electrode_data['rates_list']]
                    all_rates = [r for _, r in electrode_data['rates_list']]
                    
                    # Find common time range
                    min_time = max([t[0] for t in all_time_axes])
                    max_time = min([t[-1] for t in all_time_axes])
                    common_time = np.arange(min_time, max_time + binsize, binsize)
                    
                    # Interpolate each patient's rates to common time axis
                    interpolated_rates = []
                    for time_axis, rates in zip(all_time_axes, all_rates):
                        if len(time_axis) > 1 and len(rates) > 1:
                            interp_rates = np.interp(common_time, time_axis, rates)
                            interpolated_rates.append(interp_rates)
                    
                    if interpolated_rates:
                        # Average across patients
                        avg_rates = np.mean(interpolated_rates, axis=0)
                        std_rates = np.std(interpolated_rates, axis=0)
                        
                        # Perform bootstrapping test
                        p_value, mean_diff = bootstrap_pvalue_test(
                            electrode_data['rates_list'],
                            test_window=(-0.1, 0.1),  # Window around R-peak
                            baseline_window=(-0.5, -0.2),  # Baseline window
                            n_bootstrap=10000
                        )
                        
                        # Format p-value for display
                        if not np.isnan(p_value):
                            if p_value < 0.001:
                                p_str = "p < 0.001"
                            elif p_value < 0.01:
                                p_str = f"p = {p_value:.3f}"
                            else:
                                p_str = f"p = {p_value:.3f}"
                        else:
                            p_str = "p = N/A"
                        
                        ax_peth.plot(common_time, avg_rates, linewidth=2, color='blue', label='Mean')
                        ax_peth.fill_between(common_time, avg_rates - std_rates, avg_rates + std_rates, 
                                            alpha=0.3, color='blue', label='±1 SD')
                        ax_peth.set_ylabel("Rate (events/sec)", fontsize=10)
                        ax_peth.set_title(f"Avg PETH - {electrode} (n={len(interpolated_rates)} patients, {p_str})", 
                                         fontsize=11, fontweight='bold')
                        ax_peth.axvline(0.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
                        ax_peth.legend(fontsize=8)
                        ax_peth.grid(True, alpha=0.3)
                        ax_peth.set_xlim(minmax)
                        ax_peth.tick_params(axis='x', labelsize=8, bottom=True)
                except Exception as e:
                    ax_peth.text(0.5, 0.5, f'PETH Error: {str(e)[:40]}', 
                               ha='center', va='center', transform=ax_peth.transAxes, fontsize=8, color='red')
            else:
                ax_peth.text(0.5, 0.5, 'No rate data', 
                           ha='center', va='center', transform=ax_peth.transAxes, fontsize=9)
            
            # Plot 2: Average Event Density
            if electrode_data['events_list']:
                try:
                    # Collect all event times from all patients
                    all_times = []
                    for event_data in electrode_data['events_list']:
                        times = event_data['times']
                        all_times.extend(times)
                    
                    if all_times and len(all_times) > 0:
                        # Create histogram/density of event times
                        time_bins = np.arange(minmax[0], minmax[1] + binsize, binsize)
                        hist_counts, bin_edges = np.histogram(all_times, bins=time_bins)
                        
                        # Normalize by number of patients
                        n_patients = len(electrode_data['events_list'])
                        avg_counts = hist_counts / n_patients
                        
                        # Create time axis for histogram (bin centers)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        
                        # Plot as bar plot showing average event density
                        ax_raster.bar(bin_centers, avg_counts, width=binsize*0.8, 
                                     color='#0C7BDC', alpha=0.6, edgecolor='#0A5FA8', linewidth=0.5)
                        ax_raster.set_ylabel("Avg Events/Bin", fontsize=10)
                        ax_raster.set_title(f"Avg Event Density - {electrode} (n={n_patients} patients)", 
                                           fontsize=11, fontweight='bold')
                        ax_raster.axvline(0.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
                        ax_raster.set_xlim(minmax)
                        ax_raster.grid(True, alpha=0.3, axis='y')
                        ax_raster.tick_params(axis='x', labelsize=8, bottom=True)
                        ax_raster.tick_params(axis='y', labelsize=8)
                except Exception as e:
                    ax_raster.text(0.5, 0.5, f'Raster Error: {str(e)[:40]}', 
                                 ha='center', va='center', transform=ax_raster.transAxes, fontsize=8, color='red')
            else:
                ax_raster.text(0.5, 0.5, 'No event data', 
                             ha='center', va='center', transform=ax_raster.transAxes, fontsize=9)
            
            # Set x-label and ticks for all subplots
            ax_peth.set_xlabel("Time from R-peak (s)", fontsize=10)
            ax_raster.set_xlabel("Time from R-peak (s)", fontsize=10)
            
            # Ensure x-axis ticks are visible
            ax_peth.tick_params(labelsize=8, axis='both', which='major')
            ax_raster.tick_params(labelsize=8, axis='both', which='major')
            ax_peth.tick_params(axis='x', labelsize=8, bottom=True)
            ax_raster.tick_params(axis='x', labelsize=8, bottom=True)
        
        plt.suptitle(f"{region} Region - Average PETH & Raster Plots (All Patients)", 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save plot if requested
        if save_plot and save_dir:
            save_path_region = os.path.join(save_dir, f'eeg_peth_raster_{region}_region_averaged.png')
            plt.savefig(save_path_region, dpi=300, bbox_inches='tight')
            if 'st' in globals():
                st.success(f"{region} region averaged plots saved to {save_path_region}")
            else:
                print(f"{region} region averaged plots saved to {save_path_region}")
        
        # Display in streamlit if available
        if 'st' in globals():
            st.pyplot(fig_region)
        else:
            plt.show()
        
        figures.append(fig_region)
        plt.close(fig_region)
    
    return figures


def plot_significant_patients_electrode_peth(loaded_files, significance_results, minmax=(-0.5, 1.0), binsize=0.01, save_plot=False, save_dir=None):
    """
    Create averaged PETH plots for each electrode, showing only patients where that electrode is significant.
    
    Parameters
    ----------
    loaded_files : list
        List of tuples (file_path, patient_id, raw_object)
    significance_results : dict
        Dictionary returned from test_electrode_significance_per_patient
        Format: {patient_id: [{'electrode': str, 'p_value': float, 'mean_diff': float, 'significant': bool}, ...]}
    minmax : tuple, optional
        Time window around events in seconds (default: (-0.5, 1.0))
    binsize : float, optional
        Bin size for rate calculation in seconds (default: 0.01)
    save_plot : bool, optional
        Whether to save the plot (default: False)
    save_dir : str, optional
        Directory to save the plot (default: None)
    
    Returns
    -------
    list : List of matplotlib.figure.Figure objects
    """
    if not PYNAPPLE_AVAILABLE:
        if 'st' in globals():
            st.error("pynapple is not available. Please install it to use significant patients PETH plots.")
        else:
            print("pynapple is not available. Please install it to use significant patients PETH plots.")
        return []
    
    if not loaded_files or not significance_results:
        return []
    
    # Build mapping: {electrode: [list of patient_ids that are significant]}
    electrode_significant_patients = {}  # {electrode: [patient_id, ...]}
    
    for patient_id, patient_results in significance_results.items():
        for result in patient_results:
            if result.get('significant', False):
                electrode = result['electrode']
                if electrode not in electrode_significant_patients:
                    electrode_significant_patients[electrode] = []
                electrode_significant_patients[electrode].append(patient_id)
    
    if not electrode_significant_patients:
        if 'st' in globals():
            st.warning("No significant electrodes found for any patient.")
        else:
            print("No significant electrodes found for any patient.")
        return []
    
    # Create mapping from patient_id to file index
    patient_to_file_idx = {}
    for idx, (file_path, patient_id, raw) in enumerate(loaded_files):
        patient_to_file_idx[patient_id] = idx
    
    # Collect data from significant patients only, grouped by electrode
    # Structure: {electrode: {'rates_list': [], 'events_list': []}}
    all_electrode_data = {}
    for electrode in electrode_significant_patients.keys():
        all_electrode_data[electrode] = {'rates_list': [], 'events_list': []}
    
    if 'st' in globals():
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    total_patients = sum(len(patients) for patients in electrode_significant_patients.values())
    processed = 0
    
    # Process each electrode's significant patients
    for electrode, significant_patient_ids in electrode_significant_patients.items():
        for patient_id in significant_patient_ids:
            try:
                if patient_id not in patient_to_file_idx:
                    continue
                
                file_idx = patient_to_file_idx[patient_id]
                file_path, patient_id_check, raw = loaded_files[file_idx]
                
                if 'st' in globals():
                    processed += 1
                    status_text.text(f"Processing {electrode} - {patient_id} ({processed}/{total_patients})...")
                    progress_bar.progress(processed / total_patients)
                
                # Get sampling frequency
                sfreq = raw.info['sfreq']
                
                # Get channel names and data
                ch_names = raw.ch_names
                data = raw.get_data()
                
                # Find ECG channel
                ch_lower = [ch.lower() for ch in ch_names]
                ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
                
                if not ecg_indices:
                    continue
                
                ecg_ch_idx = ecg_indices[0]
                ecg_signal = data[ecg_ch_idx, :]
                
                # Clean ECG and detect R-peaks
                try:
                    ecg_signal_clean, _ = clean_ecg_advanced(
                        ecg_signal, 
                        sampling_rate=sfreq,
                        median_window_ms=300,
                        wavelet='db4',
                        wavelet_levels=5,
                        zscore_threshold=3.0
                    )
                    _, rpk = nk.ecg_peaks(ecg_signal_clean, sampling_rate=sfreq)
                    rpeaks = rpk['ECG_R_Peaks']
                except:
                    continue
                
                if len(rpeaks) < 2:
                    continue
                
                # Convert R-peak indices to timestamps
                rpeak_times = rpeaks / sfreq
                rpeak_ts = nap.Ts(t=rpeak_times, time_units="s")
                
                # Find electrode channel
                ch_name_to_idx = {ch.upper(): i for i, ch in enumerate(ch_names)}
                electrode_upper = electrode.upper()
                
                if electrode_upper not in ch_name_to_idx:
                    continue
                
                ch_idx = ch_name_to_idx[electrode_upper]
                
                try:
                    eeg_signal = data[ch_idx, :]
                    
                    # Detect events in EEG
                    eeg_mean = np.mean(eeg_signal)
                    eeg_std = np.std(eeg_signal)
                    threshold = eeg_mean + 2 * eeg_std
                    
                    # Find peaks above threshold
                    peaks, _ = find_peaks(eeg_signal, height=threshold, distance=int(sfreq * 0.1))
                    
                    if len(peaks) > 0:
                        # Convert peak indices to timestamps
                        peak_times = peaks / sfreq
                        eeg_peak_ts = nap.Ts(t=peak_times, time_units="s")
                        
                        # Compute perievent alignment around R-peaks
                        peth_eeg = nap.compute_perievent(
                            timestamps=eeg_peak_ts,
                            tref=rpeak_ts,
                            minmax=minmax,
                            time_unit="s"
                        )
                        
                        if len(peth_eeg) > 0:
                            # Calculate rates for this patient and electrode
                            try:
                                peth_eeg_count = peth_eeg.count(binsize)
                                if hasattr(peth_eeg_count, 'values'):
                                    count_vals = peth_eeg_count.values
                                    if count_vals.ndim == 2:
                                        rates = np.mean(count_vals, axis=1) / binsize
                                    else:
                                        rates = count_vals / binsize
                                    
                                    if hasattr(peth_eeg_count, 'index'):
                                        time_axis = peth_eeg_count.index.values
                                    else:
                                        time_axis = np.arange(len(rates)) * binsize + minmax[0]
                                    
                                    all_electrode_data[electrode]['rates_list'].append((time_axis, rates))
                            except:
                                pass
                            
                            # Collect events for raster plot
                            try:
                                peth_eeg_tsd = peth_eeg.to_tsd()
                                all_electrode_data[electrode]['events_list'].append({
                                    'times': peth_eeg_tsd.index.values,
                                    'events': peth_eeg_tsd.values
                                })
                            except:
                                pass
                
                except Exception as e:
                    continue
            
            except Exception as e:
                continue
    
    if 'st' in globals():
        progress_bar.empty()
        status_text.empty()
    
    # Filter electrodes that have data
    electrodes_with_data = [e for e in electrode_significant_patients.keys() 
                           if all_electrode_data[e]['rates_list'] or all_electrode_data[e]['events_list']]
    
    if not electrodes_with_data:
        if 'st' in globals():
            st.warning("No data available for significant electrodes.")
        else:
            print("No data available for significant electrodes.")
        return []
    
    # Create plots - group by region for better organization
    regional_electrodes = {
        'F': ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
        'P': ['P3', 'Pz', 'P4'],
        'T': ['T3', 'T4', 'T5', 'T6'],
        'O': ['O1', 'Oz', 'O2'],
        'C': ['C3', 'Cz', 'C4']
    }
    
    # Group electrodes by region
    electrodes_by_region = {}
    for region, region_elecs in regional_electrodes.items():
        electrodes_by_region[region] = [e for e in electrodes_with_data if e.upper() in [re.upper() for re in region_elecs]]
    
    figures = []
    
    # Create plots for each region that has significant electrodes
    for region, region_elecs in electrodes_by_region.items():
        if not region_elecs:
            continue
        
        # Create figure for this region
        n_electrodes = len(region_elecs)
        n_cols = 2  # PETH and Raster side by side
        n_rows = n_electrodes
        
        fig_region, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), sharex='col')
        if n_electrodes == 1:
            axes = axes.reshape(1, -1)
        
        for elec_idx, electrode in enumerate(region_elecs):
            ax_peth = axes[elec_idx, 0]
            ax_raster = axes[elec_idx, 1]
            
            electrode_data = all_electrode_data[electrode]
            n_significant_patients = len(electrode_significant_patients[electrode])
            
            # Plot 1: Average PETH
            if electrode_data['rates_list']:
                try:
                    # Interpolate all rates to common time axis
                    all_time_axes = [t for t, _ in electrode_data['rates_list']]
                    all_rates = [r for _, r in electrode_data['rates_list']]
                    
                    # Find common time range
                    min_time = max([t[0] for t in all_time_axes])
                    max_time = min([t[-1] for t in all_time_axes])
                    common_time = np.arange(min_time, max_time + binsize, binsize)
                    
                    # Interpolate each patient's rates to common time axis
                    interpolated_rates = []
                    for time_axis, rates in zip(all_time_axes, all_rates):
                        if len(time_axis) > 1 and len(rates) > 1:
                            interp_rates = np.interp(common_time, time_axis, rates)
                            interpolated_rates.append(interp_rates)
                    
                    if interpolated_rates:
                        # Average across significant patients only
                        avg_rates = np.mean(interpolated_rates, axis=0)
                        std_rates = np.std(interpolated_rates, axis=0)
                        
                        # Perform bootstrapping test to get p-value
                        p_value, mean_diff = bootstrap_pvalue_test(
                            electrode_data['rates_list'],
                            test_window=(-0.1, 0.1),  # Window around R-peak
                            baseline_window=(-0.5, -0.2),  # Baseline window
                            n_bootstrap=10000
                        )
                        
                        # Format p-value for display
                        if not np.isnan(p_value):
                            if p_value < 0.001:
                                p_str = "p < 0.001"
                            elif p_value < 0.01:
                                p_str = f"p = {p_value:.3f}"
                            else:
                                p_str = f"p = {p_value:.3f}"
                        else:
                            p_str = "p = N/A"
                        
                        ax_peth.plot(common_time, avg_rates, linewidth=2, color='blue', label='Mean')
                        ax_peth.fill_between(common_time, avg_rates - std_rates, avg_rates + std_rates, 
                                            alpha=0.3, color='blue', label='±1 SD')
                        ax_peth.set_ylabel("Rate (events/sec)", fontsize=10)
                        ax_peth.set_title(f"Avg PETH - {electrode} (n={len(interpolated_rates)} significant patients, {p_str})", 
                                         fontsize=11, fontweight='bold')
                        ax_peth.axvline(0.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
                        ax_peth.legend(fontsize=8)
                        ax_peth.grid(True, alpha=0.3)
                        ax_peth.set_xlim(minmax)
                        ax_peth.tick_params(axis='x', labelsize=8, bottom=True)
                except Exception as e:
                    ax_peth.text(0.5, 0.5, f'PETH Error: {str(e)[:40]}', 
                               ha='center', va='center', transform=ax_peth.transAxes, fontsize=8, color='red')
            else:
                ax_peth.text(0.5, 0.5, 'No rate data', 
                           ha='center', va='center', transform=ax_peth.transAxes, fontsize=9)
            
            # Plot 2: Average Event Density
            if electrode_data['events_list']:
                try:
                    # Collect all event times from significant patients
                    all_times = []
                    for event_data in electrode_data['events_list']:
                        times = event_data['times']
                        all_times.extend(times)
                    
                    if all_times and len(all_times) > 0:
                        # Create histogram/density of event times
                        time_bins = np.arange(minmax[0], minmax[1] + binsize, binsize)
                        hist_counts, bin_edges = np.histogram(all_times, bins=time_bins)
                        
                        # Normalize by number of significant patients
                        n_patients = len(electrode_data['events_list'])
                        avg_counts = hist_counts / n_patients
                        
                        # Create time axis for histogram (bin centers)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        
                        # Plot as bar plot showing average event density
                        ax_raster.bar(bin_centers, avg_counts, width=binsize*0.8, 
                                     color='blue', alpha=0.6, edgecolor='darkblue', linewidth=0.5)
                        ax_raster.set_ylabel("Avg Events/Bin", fontsize=10)
                        ax_raster.set_title(f"Avg Event Density - {electrode} (n={n_patients} significant patients)", 
                                           fontsize=11, fontweight='bold')
                        ax_raster.axvline(0.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
                        ax_raster.set_xlim(minmax)
                        ax_raster.grid(True, alpha=0.3, axis='y')
                        ax_raster.tick_params(axis='x', labelsize=8, bottom=True)
                        ax_raster.tick_params(axis='y', labelsize=8)
                except Exception as e:
                    ax_raster.text(0.5, 0.5, f'Raster Error: {str(e)[:40]}', 
                                 ha='center', va='center', transform=ax_raster.transAxes, fontsize=8, color='red')
            else:
                ax_raster.text(0.5, 0.5, 'No event data', 
                             ha='center', va='center', transform=ax_raster.transAxes, fontsize=9)
            
            # Set x-label and ticks for all subplots
            ax_peth.set_xlabel("Time from R-peak (s)", fontsize=10)
            ax_raster.set_xlabel("Time from R-peak (s)", fontsize=10)
            
            # Ensure x-axis ticks are visible
            ax_peth.tick_params(labelsize=8, axis='both', which='major')
            ax_raster.tick_params(labelsize=8, axis='both', which='major')
            ax_peth.tick_params(axis='x', labelsize=8, bottom=True)
            ax_raster.tick_params(axis='x', labelsize=8, bottom=True)
        
        plt.suptitle(f"{region} Region - Average PETH & Raster Plots (Significant Patients Only)", 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save plot if requested
        if save_plot and save_dir:
            save_path_region = os.path.join(save_dir, f'eeg_peth_raster_{region}_region_significant_only.png')
            plt.savefig(save_path_region, dpi=300, bbox_inches='tight')
            if 'st' in globals():
                st.success(f"{region} region significant patients plots saved to {save_path_region}")
            else:
                print(f"{region} region significant patients plots saved to {save_path_region}")
        
        # Display in streamlit if available
        if 'st' in globals():
            st.pyplot(fig_region)
        else:
            plt.show()
        
        figures.append(fig_region)
        plt.close(fig_region)
    
    return figures


def plot_peth_raster(loaded_files, file_index=0, minmax=(-0.5, 1.0), binsize=0.01, save_plot=False, save_dir=None):
    """
    Create a Peri-Event Time Histogram (PETH) raster plot using pynapple.
    
    This function loads a file from loaded_files, extracts R-peaks from ECG,
    and creates a raster plot showing the timing of events around R-peaks.
    
    Parameters
    ----------
    loaded_files : list
        List of tuples (file_path, patient_id, raw_object)
    file_index : int, optional
        Index of the file to process (default: 0)
    minmax : tuple, optional
        Time window around events in seconds (default: (-0.5, 1.0))
    binsize : float, optional
        Bin size for rate calculation in seconds (default: 0.01)
    save_plot : bool, optional
        Whether to save the plot (default: False)
    save_dir : str, optional
        Directory to save the plot (default: None)
    
    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if plot was created, None otherwise
    """
    if not PYNAPPLE_AVAILABLE:
        if 'st' in globals():
            st.error("pynapple is not available. Please install it to use PETH raster plots.")
        else:
            print("pynapple is not available. Please install it to use PETH raster plots.")
        return None
    
    if not loaded_files or file_index >= len(loaded_files):
        if 'st' in globals():
            st.error(f"Invalid file index {file_index}. Only {len(loaded_files)} files available.")
        else:
            print(f"Invalid file index {file_index}. Only {len(loaded_files)} files available.")
        return None
    
    # Initialize patient_id to handle cases where unpacking might fail
    patient_id = f"file_{file_index}"
    try:
        file_path, patient_id, raw = loaded_files[file_index]
    except (IndexError, ValueError, TypeError) as unpack_error:
        if 'st' in globals():
            st.error(f"Error unpacking file data at index {file_index}: {unpack_error}")
        else:
            print(f"Error unpacking file data at index {file_index}: {unpack_error}")
        return None
    
    # Get sampling frequency
    sfreq = raw.info['sfreq']
    
    # Get channel names and data
    ch_names = raw.ch_names
    data = raw.get_data()
    
    # Find ECG channel
    ch_lower = [ch.lower() for ch in ch_names]
    ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
    
    if not ecg_indices:
        if 'st' in globals():
            st.warning(f"No ECG channel found in {patient_id}")
        else:
            print(f"No ECG channel found in {patient_id}")
        return None
    
    ecg_ch_idx = ecg_indices[0]
    ecg_signal2 = data[ecg_ch_idx, :]
    # Advanced ECG cleaning: median filter, wavelet denoising, z-score thresholding
    ecg_signal_clean, cleaning_info = clean_ecg_advanced(
        ecg_signal2, 
        sampling_rate=sfreq,
        median_window_ms=300,  # 200-400 ms window
        wavelet='db4',  # or 'sym4'
        wavelet_levels=5,  # 5-8 levels
        zscore_threshold=3.0
    )
    # Clean ECG signal and detect R-peaks
    _, rpk = nk.ecg_peaks(ecg_signal_clean, sampling_rate=sfreq)
    rpeaks = rpk['ECG_R_Peaks']
    #print the bpm
    print(f"BPM: {len(rpeaks) / (len(ecg_signal_clean) / sfreq) * 60}")
    # Convert R-peak sample indices to timestamps (in seconds)
    rpeak_times = rpeaks / sfreq
    
    # Create pynapple Ts object for R-peaks (these are the reference events)
    rpeak_ts = nap.Ts(t=rpeak_times, time_units="s")
    
    # For raster plot, we'll use the R-peaks themselves as the events to align
    # This creates a "self-aligned" plot showing the cardiac rhythm
    # Alternatively, you could use other timestamps (e.g., EEG spikes)
    event_ts = nap.Ts(t=rpeak_times, time_units="s")
    
    # Compute perievent alignment
    peth = nap.compute_perievent(
        timestamps=event_ts,
        tref=rpeak_ts,
        minmax=minmax,
        time_unit="s"
    )
    
    # Initialize fig to None
    fig = None
    
    # Checkbox to show/hide PETH and Raster plots
    if 'st' in globals():
        show_peth_raster = st.checkbox("Show PETH & Raster Plot", value=False, key=f"peth_raster_{file_index}")
    else:
        show_peth_raster = True  # Always show if not in Streamlit
    
    if show_peth_raster:
        with st.expander("📊 Peri-Event Time Histogram (PETH) Raster Plot", expanded=True) if 'st' in globals() else contextlib.nullcontext():
            # Create figure with two subplots: rate histogram and raster plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Plot 1: Rate histogram (PETH)
            if len(peth) > 0:
                # Count events in bins and calculate mean rate
                # Following pynapple documentation example: np.mean(peth.count(binsize), 1) / binsize
                # Note: axis=1 means along columns (across events)
                try:
                    peth_count = peth.count(binsize)
                    # Calculate mean rate across events, following pynapple example
                    if hasattr(peth_count, 'values'):
                        count_values = peth_count.values
                    else:
                        count_values = peth_count
                    
                    # Take mean across events (axis 1 for 2D, or just use values for 1D)
                    if count_values.ndim == 2:
                        rates = np.mean(count_values, axis=1) / binsize
                    else:
                        rates = count_values / binsize
                    
                    # Get time axis
                    if hasattr(peth_count, 'index'):
                        time_axis = peth_count.index.values
                    else:
                        time_axis = np.arange(len(rates)) * binsize + minmax[0]
                except Exception as e:
                    # Fallback: use rate column if available
                    if hasattr(peth, 'rate'):
                        rates = peth.rate.values
                        time_axis = peth.index.values
                    else:
                        if 'st' in globals():
                            st.warning(f"Could not calculate PETH rate: {e}")
                        else:
                            print(f"Could not calculate PETH rate: {e}")
                        return None
                
                ax1.plot(time_axis, rates, linewidth=2, color='red')
                ax1.set_ylabel("Rate (events/sec)", fontsize=12)
                ax1.set_xlabel("Time from R-peak (s)", fontsize=12)
                ax1.set_title(f"Peri-Event Time Histogram (PETH) - {patient_id}", fontsize=14, fontweight='bold')
                ax1.axvline(0.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='R-peak')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis='x', labelsize=10, bottom=True)
                ax1.tick_params(axis='y', labelsize=10)
            
            # Plot 2: Raster plot
            if len(peth) > 0:
                # Convert TsGroup to flattened timestamps for raster plot
                peth_tsd = peth.to_tsd()
                ax2.plot(peth_tsd.index.values, peth_tsd.values, "|", markersize=15, color='red', mew=2)
                ax2.set_xlabel("Time from R-peak (s)", fontsize=12)
                ax2.set_ylabel("Event", fontsize=12)
                ax2.set_title("Raster Plot", fontsize=14, fontweight='bold')
                ax2.axvline(0.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
                ax2.set_xlim(minmax)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='x', labelsize=10, bottom=True)
                ax2.tick_params(axis='y', labelsize=10)
            
            plt.tight_layout()
            
            # Save plot if requested
            if save_plot and save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'peth_raster_{patient_id}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                if 'st' in globals():
                    st.success(f"Plot saved to {save_path}")
                else:
                    print(f"Plot saved to {save_path}")
            
            # Display in streamlit if available
            if 'st' in globals():
                st.pyplot(fig)
            else:
                plt.show()
    
    # Create raster plots for each EEG channel (only for first patient)
    if file_index == 0:
        # Find EEG channels
        eeg_ch_indices = [i for i, ch in enumerate(ch_names) 
                        if 'eeg' in ch.lower() or any(elec in ch.upper() for elec in ['FP', 'F', 'C', 'P', 'O', 'T', 'A'])]
        
        if eeg_ch_indices and len(rpeaks) >= 2:
            # Checkbox to show/hide EEG Channel PETH & Raster plots
            if 'st' in globals():
                show_eeg_peth_raster = st.checkbox("Show EEG Channel PETH & Raster Plots", value=False, key=f"eeg_peth_raster_{file_index}")
            else:
                show_eeg_peth_raster = True  # Always show if not in Streamlit
            
            if show_eeg_peth_raster:
                with st.expander("📊 EEG Channel PETH & Raster Plots (Aligned to R-peaks)", expanded=True) if 'st' in globals() else contextlib.nullcontext():
                    # Limit number of channels to plot (to avoid too many plots)
                    max_channels_to_plot = 10
                    eeg_ch_indices_plot = eeg_ch_indices[:max_channels_to_plot]
                    
                    # Get EEG data
                    eeg_data = data[eeg_ch_indices_plot, :]
                    eeg_ch_names = [ch_names[i] for i in eeg_ch_indices_plot]
                    
                    # Create PETH and raster plots for each EEG channel
                    # Each channel gets 2 subplots: PETH (top) and Raster (bottom)
                    n_channels = len(eeg_ch_indices_plot)
                    n_cols = 2  # PETH and Raster side by side
                    n_rows = n_channels  # One row per channel
                    
                    fig_eeg, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), sharex='col')
                    if n_channels == 1:
                        axes = axes.reshape(1, -1)
                    
                    for ch_idx, (eeg_idx, ch_name) in enumerate(zip(eeg_ch_indices_plot, eeg_ch_names)):
                        ax_peth = axes[ch_idx, 0]  # PETH plot (left)
                        ax_raster = axes[ch_idx, 1]  # Raster plot (right)
                        
                        try:
                            # Get EEG signal for this channel
                            eeg_signal = eeg_data[ch_idx, :]
                            
                            # Detect events in EEG (using threshold crossings or peaks)
                            eeg_mean = np.mean(eeg_signal)
                            eeg_std = np.std(eeg_signal)
                            threshold = eeg_mean + 2 * eeg_std  # 2 std above mean
                            
                            # Find peaks above threshold
                            peaks, _ = find_peaks(eeg_signal, height=threshold, distance=int(sfreq * 0.1))
                            
                            if len(peaks) > 0:
                                # Convert peak indices to timestamps
                                peak_times = peaks / sfreq
                                
                                # Create pynapple Ts object for EEG peaks
                                eeg_peak_ts = nap.Ts(t=peak_times, time_units="s")
                                
                                # Compute perievent alignment around R-peaks
                                # Use -200ms to 600ms for EEG plots
                                eeg_minmax = (-0.2, 0.6)
                                peth_eeg = nap.compute_perievent(
                                    timestamps=eeg_peak_ts,
                                    tref=rpeak_ts,
                                    minmax=eeg_minmax,
                                    time_unit="s"
                                )
                                
                                if len(peth_eeg) > 0:
                                    # Plot 1: PETH (Rate Histogram)
                                    try:
                                        peth_eeg_count = peth_eeg.count(binsize)
                                        if hasattr(peth_eeg_count, 'values'):
                                            count_vals = peth_eeg_count.values
                                            if count_vals.ndim == 2:
                                                rates_eeg = np.mean(count_vals, axis=1) / binsize
                                            else:
                                                rates_eeg = count_vals / binsize
                                            
                                            if hasattr(peth_eeg_count, 'index'):
                                                time_axis_eeg = peth_eeg_count.index.values
                                            else:
                                                time_axis_eeg = np.arange(len(rates_eeg)) * binsize + eeg_minmax[0]
                                            
                                            ax_peth.plot(time_axis_eeg, rates_eeg, linewidth=2, color='blue')
                                            ax_peth.set_ylabel("Rate (events/sec)", fontsize=10)
                                            ax_peth.set_title(f"PETH - {ch_name}", fontsize=11, fontweight='bold')
                                            ax_peth.axvline(0.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='R-peak')
                                            ax_peth.legend(fontsize=8)
                                            ax_peth.grid(True, alpha=0.3)
                                            ax_peth.set_xlim(eeg_minmax)
                                            ax_peth.tick_params(axis='x', labelsize=8, bottom=True)
                                        else:
                                            ax_peth.text(0.5, 0.5, 'Could not calculate rate', 
                                                        ha='center', va='center', transform=ax_peth.transAxes, fontsize=9)
                                    except Exception as e:
                                        ax_peth.text(0.5, 0.5, f'PETH Error: {str(e)[:40]}', 
                                                    ha='center', va='center', transform=ax_peth.transAxes, fontsize=8, color='red')
                                    
                                    # Plot 2: Raster Plot
                                    try:
                                        peth_eeg_tsd = peth_eeg.to_tsd()
                                        ax_raster.plot(peth_eeg_tsd.index.values, peth_eeg_tsd.values, "|", 
                                                        markersize=12, color='blue', mew=1.5, alpha=0.7)
                                        ax_raster.set_ylabel("Event", fontsize=10)
                                        ax_raster.set_title(f"Raster - {ch_name}", fontsize=11, fontweight='bold')
                                        ax_raster.axvline(0.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
                                        ax_raster.set_xlim(eeg_minmax)
                                        ax_raster.grid(True, alpha=0.3)
                                        ax_raster.tick_params(axis='x', labelsize=8, bottom=True)
                                    except Exception as e:
                                        ax_raster.text(0.5, 0.5, f'Raster Error: {str(e)[:40]}', 
                                                        ha='center', va='center', transform=ax_raster.transAxes, fontsize=8, color='red')
                                else:
                                    ax_peth.text(0.5, 0.5, 'No events detected', 
                                                ha='center', va='center', transform=ax_peth.transAxes, fontsize=9)
                                    ax_raster.text(0.5, 0.5, 'No events detected', 
                                                    ha='center', va='center', transform=ax_raster.transAxes, fontsize=9)
                            else:
                                ax_peth.text(0.5, 0.5, 'No peaks detected', 
                                            ha='center', va='center', transform=ax_peth.transAxes, fontsize=9)
                                ax_raster.text(0.5, 0.5, 'No peaks detected', 
                                                ha='center', va='center', transform=ax_raster.transAxes, fontsize=9)
                            
                            # Set x-label and ticks for all subplots
                            ax_peth.set_xlabel("Time from R-peak (s)", fontsize=10)
                            ax_raster.set_xlabel("Time from R-peak (s)", fontsize=10)
                            
                            # Ensure x-axis ticks are visible
                            ax_peth.tick_params(labelsize=8, axis='both', which='major')
                            ax_raster.tick_params(labelsize=8, axis='both', which='major')
                            ax_peth.tick_params(axis='x', labelsize=8, bottom=True)
                            ax_raster.tick_params(axis='x', labelsize=8, bottom=True)
                        
                        except Exception as e:
                            error_msg = f'Error: {str(e)[:40]}'
                            ax_peth.text(0.5, 0.5, error_msg, 
                                        ha='center', va='center', transform=ax_peth.transAxes, fontsize=8, color='red')
                            ax_raster.text(0.5, 0.5, error_msg, 
                                            ha='center', va='center', transform=ax_raster.transAxes, fontsize=8, color='red')
                            ax_peth.set_title(f"PETH - {ch_name} (Error)", fontsize=11)
                            ax_raster.set_title(f"Raster - {ch_name} (Error)", fontsize=11)
                    
                    plt.suptitle(f"EEG Channel PETH & Raster Plots - {patient_id}", fontsize=14, fontweight='bold', y=0.995)
                    plt.tight_layout(rect=[0, 0, 1, 0.99])
                    
                    # Save plot if requested
                    if save_plot and save_dir:
                        save_path_eeg = os.path.join(save_dir, f'eeg_peth_raster_{patient_id}.png')
                        plt.savefig(save_path_eeg, dpi=300, bbox_inches='tight')
                        if 'st' in globals():
                            st.success(f"EEG PETH & Raster plots saved to {save_path_eeg}")
                        else:
                            print(f"EEG PETH & Raster plots saved to {save_path_eeg}")
                    
                    # Display in streamlit if available
                    if 'st' in globals():
                        st.pyplot(fig_eeg)
                    else:
                        plt.show()
                    
                    plt.close(fig_eeg)
    
    
    significance_results = test_electrode_significance_per_patient(loaded_files, minmax=minmax, binsize=binsize,
                                            test_window=(-0.1, 0.1), baseline_window=(-0.5, -0.2),
                                            n_bootstrap=100, significance_threshold=0.05)
    
    # Plot topomap of significance percentages
    if significance_results:
        if 'st' in globals():
            st.markdown("---")
            st.markdown("## 🗺️ Electrode Significance Topomap (Percentage of Patients)")
        
        plot_significance_percentage_topomap(significance_results, save_plot=save_plot, save_dir=save_dir)
        
        # Create averaged PETH plots for significant patients only
        if len(loaded_files) > 1:
            if 'st' in globals():
                st.markdown("---")
                st.markdown("## 📊 Average PETH Plots (Significant Patients Only)")
            
            plot_significant_patients_electrode_peth(loaded_files, significance_results, 
                                                    minmax=minmax, binsize=binsize,
                                                    save_plot=save_plot, save_dir=save_dir)
    
    # Create averaged PETH and raster plots across all patients
    if len(loaded_files) > 1:
        if 'st' in globals():
            st.markdown("---")
            st.markdown("## 📊 Average EEG Channel PETH & Raster Plots (Across All Patients)")
        
        plot_averaged_eeg_peth_raster(loaded_files, minmax=minmax, binsize=binsize,
                                        save_plot=save_plot, save_dir=save_dir)
        
        # Create regional averaged plots
        if 'st' in globals():
            st.markdown("---")
            st.markdown("## 📊 Regional Average EEG PETH & Raster Plots (F, C, T, P, O - First 21 Electrodes)")
        
        plot_regional_averaged_eeg_peth_raster(loaded_files, minmax=minmax, binsize=binsize,
                                                save_plot=save_plot, save_dir=save_dir)
        
        # Create regional electrode plots (one plot per region with subplots for each electrode)
        if 'st' in globals():
            st.markdown("---")
            st.markdown("## 📊 Regional Electrode Plots (F, P, T, O, C - One Plot Per Region)")
        
        plot_regional_electrode_averaged_eeg(loaded_files, minmax=minmax, binsize=binsize,
                                            save_plot=save_plot, save_dir=save_dir)
    
    return fig
        
    

def MI_systole_diastole_comparison(loaded_files, sleep_stage, output_dir, save_results):

    # Process all files (with MI calculation)
    with st.spinner("Processing files to extract HEP and calculate modulation index..."):
        df, mi_df = process_all_files_with_mi(loaded_files)
    
    if df.empty:
        st.error("No data extracted. Exiting.")
        return
    
    # Save results to CSV
    if save_results:
        csv_path = os.path.join(output_dir, f'hep_results_{sleep_stage}.csv')
        df.to_csv(csv_path, index=False)
        st.success(f"Results saved to {csv_path}")
    
    # Display data summary
    st.markdown("## 📋 Data Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", len(df['patient_id'].unique()))
    with col2:
        st.metric("Systole Records", len(df[df['phase'] == 'systole']))
    with col3:
        st.metric("Diastole Records", len(df[df['phase'] == 'diastole']))
    
    # Save modulation index results
    if not mi_df.empty:
        if save_results:
            mi_csv_path = os.path.join(output_dir, f'modulation_index_{sleep_stage}.csv')
            mi_df.to_csv(mi_csv_path, index=False)
            st.success(f"Modulation index results saved to {mi_csv_path}")
        
        # Display summary statistics for MI
        st.markdown("## 📊 Modulation Index (Tort et al. 2010) Summary")
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean MI", f"{mi_df['modulation_index'].mean():.4f} ± {mi_df['modulation_index'].std():.4f}")
        with col2:
            st.metric("Mean Correlation", f"{mi_df['correlation'].mean():.4f} ± {mi_df['correlation'].std():.4f}")
        with col3:
            st.metric("Mean Amplitude Systole", f"{mi_df['mean_amplitude_systole'].mean():.4f} ± {mi_df['mean_amplitude_systole'].std():.4f}")
        with col4:
            st.metric("Mean Amplitude Diastole", f"{mi_df['mean_amplitude_diastole'].mean():.4f} ± {mi_df['mean_amplitude_diastole'].std():.4f}")
        
        # Display MI dataframe
        st.markdown("### Modulation Index Data")
        st.dataframe(mi_df, use_container_width=True)
    else:
        st.warning("No modulation index data calculated.")
    
    # Perform statistical comparison
    st.markdown("---")
    with st.spinner("Performing statistical comparison..."):
        stats_results = compare_systole_diastole(df)
    
    if stats_results:
        # Display statistics
        print_statistics(stats_results)
        
        # Save statistics to CSV
        if save_results:
            stats_df = pd.DataFrame(stats_results).T
            stats_csv_path = os.path.join(output_dir, f'hep_statistics_{sleep_stage}.csv')
            stats_df.to_csv(stats_csv_path)
            st.success(f"Statistics saved to {stats_csv_path}")
        
        # Create plots
        st.markdown("---")
        st.markdown("## 📈 Comparison Plots")
        with st.spinner("Creating comparison plots..."):
            plot_comparison(df, stats_results, save_path=None)
    else:
        st.warning("Could not perform statistical comparison")
        # Still create plots without statistics
        st.markdown("## 📈 Comparison Plots")
        plot_comparison(df, None, save_path=None)
    
    # Create PairGrid plot with systole/diastole as hue
    st.markdown("---")
    st.markdown("## 🔄 PairGrid Plot")
    with st.spinner("Creating PairGrid plot..."):
        plot_hep_pairgrid(df, sleep_stage=sleep_stage, save_plot=save_results, save_dir=output_dir)
    
    st.success("✅ Analysis complete!")


def plot_hep_pairgrid(df, sleep_stage='N1', save_plot=True, save_dir='hep_comparison_results'):
    """
    Plot HEP metrics using PairGrid with systole/diastole as hue.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with HEP results containing 'phase', 'mean_amplitude', 
        'peak_amplitude', 'rms', 'std_amplitude' columns
    sleep_stage : str
        Sleep stage identifier for naming
    save_plot : bool
        Whether to save the plot
    save_dir : str
        Directory to save the plot
    
    Returns
    -------
    None
    """
    if df.empty:
        if 'st' in globals():
            st.warning("No data to plot in PairGrid")
        else:
            print("No data to plot in PairGrid")
        return
    
    # Check if we have both systole and diastole data
    phases = df['phase'].unique()
    if 'systole' not in phases or 'diastole' not in phases:
        if 'st' in globals():
            st.warning("Need both systole and diastole data for PairGrid plot")
        else:
            print("Need both systole and diastole data for PairGrid plot")
        return
    
    # Get common patients (paired data)
    systole_df = df[df['phase'] == 'systole'].sort_values('patient_id')
    diastole_df = df[df['phase'] == 'diastole'].sort_values('patient_id')
    
    common_patients = set(systole_df['patient_id']).intersection(set(diastole_df['patient_id']))
    
    if len(common_patients) < 2:
        if 'st' in globals():
            st.warning("Not enough paired data for PairGrid plot")
        else:
            print("Not enough paired data for PairGrid plot")
        return
    
    # Filter to common patients and ensure same order
    systole_paired = systole_df[systole_df['patient_id'].isin(common_patients)].sort_values('patient_id')
    diastole_paired = diastole_df[diastole_df['patient_id'].isin(common_patients)].sort_values('patient_id')
    
    # Verify patient order matches
    if not np.array_equal(systole_paired['patient_id'].values, diastole_paired['patient_id'].values):
        if 'st' in globals():
            st.info("Patient order mismatch, sorting...")
        else:
            print("Patient order mismatch, sorting...")
        systole_paired = systole_paired.sort_values('patient_id')
        diastole_paired = diastole_paired.sort_values('patient_id')
    
    # Define metrics to plot
    metrics = ['mean_amplitude', 'peak_amplitude', 'rms', 'std_amplitude']
    metric_labels = ['Mean Amplitude', 'Peak Amplitude', 'RMS', 'Std Amplitude']
    
    # Find common valid indices across all metrics
    n_patients = len(systole_paired)
    
    # Start with all indices as potentially valid
    valid_mask = np.ones(n_patients, dtype=bool)
    
    # Check each metric and find common valid indices
    for metric in metrics:
        if metric in systole_paired.columns and metric in diastole_paired.columns:
            systole_vals = systole_paired[metric].values
            diastole_vals = diastole_paired[metric].values
            metric_valid = ~(np.isnan(systole_vals) | np.isnan(diastole_vals))
            valid_mask = valid_mask & metric_valid
    
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) < 2:
        if 'st' in globals():
            st.warning("Not enough valid data across all metrics for PairGrid plot")
        else:
            print("Not enough valid data across all metrics for PairGrid plot")
        return
    
    # Extract arrays for each metric using common valid indices
    arrs = []
    labels = []
    for metric, label in zip(metrics, metric_labels):
        if metric in systole_paired.columns and metric in diastole_paired.columns:
            systole_vals = systole_paired[metric].values[valid_indices]
            diastole_vals = diastole_paired[metric].values[valid_indices]
            
            # Combine systole and diastole values
            combined_vals = np.concatenate([systole_vals, diastole_vals])
            arrs.append(combined_vals)
            labels.append(label)
    
    if len(arrs) < 2:
        if 'st' in globals():
            st.warning("Not enough valid metrics for PairGrid plot")
        else:
            print("Not enough valid metrics for PairGrid plot")
        return
    
    # Create hue array (systole or diastole) matching the combined arrays
    n_valid = len(valid_indices)
    hue = pd.Series(['Systole'] * n_valid + ['Diastole'] * n_valid)
    
    # Create time array (patient index)
    t = np.arange(len(arrs[0]))
    
    # Create name for the plot
    plot_name = f'hep_pairgrid_{sleep_stage}'
    
    # Call _plot_metric_vs_hrv
    if 'st' in globals():
        st.info("Creating PairGrid plot for HEP metrics (hue: Systole/Diastole)...")
    else:
        print(f"\nCreating PairGrid plot for HEP metrics (hue: Systole/Diastole)...")
    
    _plot_metric_vs_hrv(
        t=t,
        arrs=arrs,
        labels=labels,
        save_plot=save_plot,
        name=plot_name,
        save_dir=save_dir,
        is_streamlit=('st' in globals()),
        hue=hue,
        size_values=None
    )
    
    if save_plot and 'st' not in globals():
        print(f"PairGrid plot saved to {save_dir}/pairgrid_all_{plot_name}.png")


if __name__ == "__main__":
    main()

