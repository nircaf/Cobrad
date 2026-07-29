import os
import gc
import glob
import pickle
import argparse
import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import numpy as np
import pandas as pd
import mne
import yasa
import csv

# Import utilities from HEP_parquet_generation
from HEP_parquet_generation import (
    rename_edfs_to_upper,
    get_project_name,
    RunStatusTracker,
    group_edf_files_by_patient_edf,
    _YASA_EEG_PRIORITY,
    prep_for_yasa,
    find_stage_epochs,
    smooth_sleep_stages,
    clean_duplicate_pickles,
    _extract_patient_id_from_name,
    _parse_sleep_stage_pickle_path,
    VALID_SLEEP_STAGES,
    DEFAULT_SLEEP_STAGE,
)
from utils.eeg_utils import (
    rename_channels, 
    eeg_channels, 
    detect_line_frequency
)
import edf_cleaning


MIN_COMPLETE_PATIENT_PICKLE_BYTES = 64 * 1024


def patient_pickle_is_complete(path):
    """Reject tiny files left behind when a pickle write was interrupted."""
    try:
        return os.path.getsize(path) >= MIN_COMPLETE_PATIENT_PICKLE_BYTES
    except OSError:
        return False


def remove_incomplete_patient_pickle(path, logger=None):
    if not os.path.exists(path) or patient_pickle_is_complete(path):
        return False
    try:
        size = os.path.getsize(path)
        os.remove(path)
        message = f"Removed incomplete patient pickle ({size} bytes): {path}"
        if logger is not None:
            logger.warning(message)
        else:
            logging.warning(message)
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        if logger is not None:
            logger.warning(f"Could not remove incomplete pickle {path}: {exc}")
        return False


def save_patient_pickle_atomic(raw, pickle_path):
    """Write completely, then atomically publish without replacing a peer's file."""
    pickle_dir = os.path.dirname(pickle_path)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='wb', dir=pickle_dir, prefix='.patient_pickle_', suffix='.tmp', delete=False
        ) as temp_file:
            temp_path = temp_file.name
            pickle.dump(raw, temp_file)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        # Hard-link publication is atomic and fails if another worker won.
        os.link(temp_path, pickle_path)
        return True
    except FileExistsError:
        return False
    finally:
        if temp_path is not None:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass


def patient_stage_pickle_exists(project_name, patient_id, stage):
    """Return True when this patient already has one saved scan for this stage."""
    pickle_dir = os.path.join("pickles_sleep_stage", project_name, stage)
    patient_key = _extract_patient_id_from_name(patient_id)

    for pickle_path in glob.glob(os.path.join(pickle_dir, "*.pkl")):
        parsed = _parse_sleep_stage_pickle_path(pickle_path)
        if parsed is None or parsed['stage'] != stage:
            continue
        if not patient_pickle_is_complete(pickle_path):
            remove_incomplete_patient_pickle(pickle_path)
            continue
        if _extract_patient_id_from_name(parsed['patient_id']) == patient_key:
            return True

    return False


def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    return logger

def find_longest_stage_streak(stage_epochs, max_gap=3):
    """Longest continuous streak of stage epochs (gaps up to max_gap allowed)."""
    if len(stage_epochs) == 0:
        return None, 0
    longest_start = None
    longest_length = 0
    current_start = stage_epochs[0]
    current_length = 1
    current_end = stage_epochs[0]
    for i in range(1, len(stage_epochs)):
        gap = stage_epochs[i] - current_end
        if gap <= max_gap + 1:
            current_length += 1
            current_end = stage_epochs[i]
        else:
            if current_length > longest_length:
                longest_start = current_start
                longest_length = current_length
            current_start = stage_epochs[i]
            current_length = 1
            current_end = stage_epochs[i]
    if current_length > longest_length:
        longest_start = current_start
        longest_length = current_length
    return longest_start, longest_length


def process_single_patient(args_tuple):
    """
    Worker function to process a single patient.
    Extracts 4 sleep stages from the same EDF files to save computation.
    """
    patient_id, files, edf_root, step_sec, n1_duration_min, project_name = args_tuple

    # Configure logging for this worker process — ProcessPoolExecutor child
    # processes do not inherit the parent's log handlers.
    log_dir = "logs/workers"
    os.makedirs(log_dir, exist_ok=True)
    worker_log_file = os.path.join(log_dir, f"patient_{patient_id}.log")
    worker_logger = logging.getLogger(f"Patient_{patient_id}")
    if not worker_logger.handlers:
        _fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        _fh = logging.FileHandler(worker_log_file, mode='w')
        _fh.setFormatter(_fmt)
        _sh = logging.StreamHandler()
        _sh.setFormatter(_fmt)
        worker_logger.addHandler(_fh)
        worker_logger.addHandler(_sh)
        worker_logger.setLevel(logging.INFO)
    logger = worker_logger

    stages_to_extract = VALID_SLEEP_STAGES
    patient_results = {stage: {'successful': 0, 'errors': 0, 'skipped': 0} for stage in stages_to_extract}
    
    for file_path in files:
        logger.info(f"Processing file: {file_path}")
        try:
            # 1. Load EDF and preprocess
            import signal as _signal
            def _edf_timeout(signum, frame):
                raise TimeoutError("read_raw_edf timed out after 10 min")
            _signal.signal(_signal.SIGALRM, _edf_timeout)
            _signal.alarm(1200)
            try:
                raw = mne.io.read_raw_edf(file_path, preload=True, encoding='latin1')
                line_freq = detect_line_frequency(raw)
                raw.notch_filter(np.arange(line_freq, raw.info['sfreq']/2, line_freq), verbose=False)
                raw.filter(l_freq=0.5, h_freq=raw.info['sfreq']/2 - 0.1, verbose=False)
                raw.resample(256)
                raw = rename_channels(raw)
            finally:
                _signal.alarm(0)

            ch_names = raw.ch_names
            ch_lower = [ch.lower() for ch in ch_names]                
            ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
            if not ecg_indices:
                logger.warning(f"No ECG channel found. Skipping file {file_path}.")
                continue

            available_eeg_channels = [ch for ch in raw.ch_names if ch in eeg_channels]
            if not available_eeg_channels:
                # Fallback for bipolar-referenced naming (e.g. Harvard: C3-M2, C4-M1).
                # Strip the reference suffix and rename to canonical monopolar name.
                bipolar_rename = {}
                for ch in raw.ch_names:
                    prefix = ch.split('-')[0]
                    if prefix in eeg_channels and ch != prefix:
                        bipolar_rename[ch] = prefix
                if bipolar_rename:
                    raw.rename_channels(bipolar_rename)
                    raw.set_channel_types({v: 'eeg' for v in bipolar_rename.values()})
                    available_eeg_channels = [v for v in bipolar_rename.values() if v in raw.ch_names]
                    logger.info(f"Bipolar fallback: renamed {list(bipolar_rename.items())[:5]}")
            if not available_eeg_channels:
                logger.warning(f"No suitable EEG channels found. Skipping file {file_path}.")
                continue

            # Resolve EOG channel
            eog_name = None
            if 'EOG+' in raw.ch_names:
                eog_name = 'EOG+'
            else:
                loc = 'LOC' if 'LOC' in raw.ch_names else None
                roc = 'ROC' if 'ROC' in raw.ch_names else None
                if loc and roc:
                    raw = mne.set_bipolar_reference(raw, anode=[loc], cathode=[roc], ch_name=['LOC-ROC'], drop_refs=False)
                    eog_name = 'LOC-ROC'
                elif loc: eog_name = loc
                elif roc: eog_name = roc

            # Resolve EMG channel
            emg_name = None
            if 'EMG1+' in raw.ch_names: emg_name = 'EMG1+'
            elif 'EMG2+' in raw.ch_names: emg_name = 'EMG2+'

            # Find best YASA electrode
            eeg_candidates = [ch for ch in _YASA_EEG_PRIORITY if ch in available_eeg_channels]
            best_candidate_result = None
            best_score = -1

            # We evaluate up to 3 candidates to save time, picking the one with most valid sleep stages.
            for candidate in eeg_candidates[:3]:
                logger.info(f"Trying EEG electrode '{candidate}' for staging on {file_path}")
                try:
                    raw_try = prep_for_yasa(raw=raw.copy(), eeg_name=candidate)
                    ss = yasa.SleepStaging(
                        raw_try,
                        eeg_name=candidate,
                        eog_name=eog_name,
                        emg_name=emg_name,
                    )
                    preds = ss.predict()
                    
                    # Score is number of non-wake epochs
                    non_w_epochs = sum(np.where((preds != 'W') & (preds != 'UN'), 1, 0))
                    
                    if non_w_epochs > best_score:
                        best_score = non_w_epochs
                        best_candidate_result = {
                            "candidate": candidate,
                            "raw_try": raw_try,
                            "preds": preds
                        }
                except Exception as exc:
                    logger.warning(f"YASA failed with electrode '{candidate}': {exc}")

            if not best_candidate_result:
                logger.warning(f"All evaluated electrodes failed YASA staging for {file_path}")
                continue

            eeg_name = best_candidate_result["candidate"]
            raw_staged = best_candidate_result["raw_try"]
            preds = best_candidate_result["preds"]
            smooth_stages = smooth_sleep_stages(preds)
            epoch_duration_sec = 30
            
            logger.info(f"Selected best electrode '{eeg_name}' for file {file_path}. Proceeding to extract stages.")
            
            # Extract and process each target stage
            for stage in stages_to_extract:
                logger.info(f"  Extracting stage: {stage}")
                if (
                    patient_results[stage]['successful'] > 0
                    or patient_stage_pickle_exists(project_name, patient_id, stage)
                ):
                    logger.info(
                        f"    Existing {stage} pickle found for {patient_id}; "
                        "leaving it untouched and skipping extra scans for this stage."
                    )
                    continue

                target_stages = ['N1', 'N2'] if stage == 'light_sleep' else [stage]
                stage_segments_in_file = []
                
                if stage == 'W':
                    # Special Wake logic
                    min_non_w_epochs = int(np.ceil(5 * 60 / epoch_duration_sec))
                    runs = []
                    if len(smooth_stages) > 0:
                        cur_label = smooth_stages[0]
                        cur_start, cur_len = 0, 1
                        for i in range(1, len(smooth_stages)):
                            if smooth_stages[i] == cur_label:
                                cur_len += 1
                            else:
                                runs.append((cur_label, cur_start, cur_len))
                                cur_label = smooth_stages[i]
                                cur_start = i
                                cur_len = 1
                        runs.append((cur_label, cur_start, cur_len))
                    
                    runs = [run for run in runs if run[2] > min_non_w_epochs]
                    stage_segments_in_file = [(run[1] * epoch_duration_sec, (run[1] + run[2]) * epoch_duration_sec) for run in runs if run[0] == 'W']
                    if not stage_segments_in_file:
                        logger.warning(f"    No W-stage runs found (total runs after filter: {len(runs)})")
                else:
                    # Generic stage logic
                    s_epochs = find_stage_epochs(preds, target_stages, bridge_gap=6)
                    lon_start, lon_len = find_longest_stage_streak(s_epochs)
                    
                    if lon_len and lon_len > 15: # minimum_usable_streak = 15
                        seg_start = lon_start * epoch_duration_sec
                        seg_end = (lon_start + lon_len) * epoch_duration_sec
                        stage_segments_in_file = [(seg_start, seg_end)]
                
                if not stage_segments_in_file:
                    lon_len_val = lon_len if stage != 'W' else 0
                    logger.warning(f"    No valid continuous segment found for {stage} (lon_len={lon_len_val})")
                    patient_results[stage]['skipped'] += 1
                    continue
                
                # Take the longest segment, cap at 1 hour (1200 seconds)
                longest = max(stage_segments_in_file, key=lambda s: s[1] - s[0])
                start_sec_orig, end_sec_orig = longest
                end_sec = min(end_sec_orig, start_sec_orig + 1200)
                
                logger.info(f"    Processing {stage} segment {start_sec_orig:.1f}s-{end_sec:.1f}s")
                
                try:
                    raw_stage = raw.copy().crop(tmin=start_sec_orig, tmax=end_sec)
                    raw_stage_cleaned = edf_cleaning.clean_mne_raw(raw_stage, file_path)
                    
                    if raw_stage_cleaned is not None:
                        duration = int(end_sec - start_sec_orig)
                        pickle_dir = os.path.join("pickles_sleep_stage", project_name, stage)
                        os.makedirs(pickle_dir, exist_ok=True)
                        pickle_filename = f"{patient_id}_{stage}_{duration}_{step_sec}.pkl"
                        pickle_path = os.path.join(pickle_dir, pickle_filename)

                        remove_incomplete_patient_pickle(pickle_path, logger)
                        if os.path.exists(pickle_path):
                            logger.info(
                                f"    Pickle already exists for {patient_id} at {pickle_path}; "
                                "leaving existing file untouched."
                            )
                            continue
                        
                        if not save_patient_pickle_atomic(raw_stage_cleaned, pickle_path):
                            logger.info(
                                f"    Pickle was created by another process at {pickle_path}; "
                                "leaving existing file untouched."
                            )
                            continue
                            
                        logger.info(f"    Saved {stage} segment for {patient_id} to {pickle_path}")
                        patient_results[stage]['successful'] += 1
                    else:
                        logger.warning(f"    clean_mne_raw returned None for {stage} segment.")
                        patient_results[stage]['errors'] += 1
                except Exception as e:
                    logger.error(f"    Error processing {stage} segment: {e}")
                    patient_results[stage]['errors'] += 1

        except Exception as e:
            logger.error(f"CRASHED file {file_path}: {e}")

        finally:
            # Aggressive Memory Cleanup
            if 'raw' in locals(): del raw
            if 'raw_staged' in locals(): del raw_staged
            if 'raw_try' in locals(): del raw_try
            if 'raw_stage' in locals(): del raw_stage
            if 'raw_stage_cleaned' in locals(): del raw_stage_cleaned
            gc.collect()

    logger.info(f"SUMMARY for {patient_id}: {patient_results}")
    return patient_id, patient_results

def run_parallel_processing(edf_root="EDF_Format/EDF", step_sec=5, n1_duration_min=10, workers=4, reverse=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = get_project_name(edf_root)
    log_file = f"logs/parallel_run_{project_name}_{timestamp}.log"
    logger = setup_logger('parallel_processing', log_file)
    
    logger.info(f"Starting parallel processing. edf_root={edf_root}, workers={workers}")
    
    # 1. Rename EDFs
    edf_root = rename_edfs_to_upper(edf_root)
    
    # 2. Find EDF files and group by patient
    patient_to_files = group_edf_files_by_patient_edf(edf_root=edf_root)
    
    # Filter out excluded patients
    excluded_csv = os.path.join("pickles_sleep_stage", "excluded_patients.csv")
    excluded_ids = set()
    if os.path.exists(excluded_csv):
        with open(excluded_csv, newline='') as _f:
            _reader = csv.DictReader(_f)
            excluded_ids = {row['patient_id'].strip() for row in _reader if row.get('patient_id', '').strip()}
    if excluded_ids:
        patient_to_files = {pid: files for pid, files in patient_to_files.items() if pid not in excluded_ids}
        
    if not patient_to_files:
        logger.error(f"No valid EDF files found under {edf_root}")
        return

    # Sort
    patient_to_files = dict(sorted(patient_to_files.items(), key=lambda x: x[0], reverse=reverse))
    
    logger.info(f"Found {len(patient_to_files)} patients to process.")
    
    # 3. Create arguments for the worker pool
    args_list = []
    for pid, files in patient_to_files.items():
        args_list.append((pid, files, edf_root, step_sec, n1_duration_min, project_name))
        
    # 4. Run ProcessPoolExecutor
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_pid = {executor.submit(process_single_patient, args): args[0] for args in args_list}
        
        for future in as_completed(future_to_pid):
            pid = future_to_pid[future]
            try:
                patient_id, results = future.result()
                completed += 1
                logger.info(f"[{completed}/{len(args_list)}] Completed patient {patient_id}. Results: {results}")
            except Exception as exc:
                logger.error(f"Patient {pid} generated an exception: {exc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process EDF files for multiple sleep stages in parallel')
    parser.add_argument('-c', '--edf_root', type=str, default="EDF_Format/The_Human_Sleep_Project",
                        help='Root directory containing EDF files')
    parser.add_argument('-s', '--step_sec', type=int, default=5,
                        help='Step size in seconds for sliding window (default: 5)')
    parser.add_argument('--n1_duration_min', type=int, default=10,
                        help='Duration of stage segments to extract in minutes (default: 10)')
    import multiprocessing
    default_workers = max(1, multiprocessing.cpu_count() // 3)
    
    parser.add_argument('-w', '--workers', type=int, default=default_workers,
                        help=f'Number of parallel workers (processes). Default: {default_workers} (1/3 of cores)')
    parser.add_argument('--reverse', action='store_true', default=False,
                        help='Process patients in reverse order (Z→A)')

    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    project_name = get_project_name(args.edf_root)
    clean_duplicate_pickles(os.path.join("pickles_sleep_stage", project_name))

    run_parallel_processing(
        edf_root=args.edf_root,
        step_sec=args.step_sec,
        n1_duration_min=args.n1_duration_min,
        workers=args.workers,
        reverse=args.reverse
    )
