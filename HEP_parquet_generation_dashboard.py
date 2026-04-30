#!/usr/bin/env python3
"""
HEP Processing Dashboard
Serves a live HTML dashboard showing patient processing progress per sleep stage.
Usage: python dashboard.py [--port 8765]
"""

import os
import glob
import json
import argparse
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUETS_HEP_DIR = os.path.join(BASE_DIR, "parquets_HEP")
EDF_ROOT = os.path.join(BASE_DIR, "pickles_sleep_stage", "Berkeley_data")

STAGES = ["R", "W", "light_sleep", "N3"]
STAGE_COLORS = {
    "R":  "#f87171",   # red   – REM
    "W":  "#60a5fa",   # blue  – Wake
    "N1": "#a78bfa",   # violet
    "N2": "#34d399",   # green
    "N3": "#fb923c",   # orange
}


# ── Data collection ─────────────────────────────────────────────────────────────
def get_all_patients(edf_root):
    """Find all unique patients in the EDF/Pickles directory."""
    patient_ids = set()
    for root, dirs, files in os.walk(edf_root):
        for file in files:
            if not (file.lower().endswith(".edf") or file.lower().endswith(".pkl")):
                continue
            m = re.search(r'(\d{4}-\d{3})', file)
            if m:
                patient_ids.add(m.group(1))
            else:
                if file.lower().endswith('.pkl'):
                    # e.g. SSS11_SR_W_6030_5.pkl -> SSS11_SR
                    # e.g. ASSY16_CHECKIFEXP_EKG_W_6150_5.pkl -> ASSY16_CHECKIFEXP_EKG
                    match = re.search(r'^(.*?)_(W|N1|N2|N3|R)_', file)
                    if match:
                        patient_ids.add(match.group(1))
                    else:
                        patient_ids.add(file.split('.')[0])
                else:
                    patient_ids.add(file.split('.')[0])
    return sorted(list(patient_ids))

def count_total_patients(edf_root):
    """Count unique patient IDs in the EDF directory (one entry per folder/file)."""
    return len(get_all_patients(edf_root))

def parse_logs(edf_root_name="Berkeley_data"):
    """Parse the log files to get the latest status of each patient per stage."""
    log_dir = os.path.join(BASE_DIR, "logs")
    status_dict = {s: {} for s in STAGES}
    
    if not os.path.exists(log_dir):
        return status_dict
        
    # Find all matching logs for this project
    log_files = glob.glob(os.path.join(log_dir, f"run_{edf_root_name}_*.log"))
    # Fallback to old log if specific logs don't exist
    if not log_files:
        old_log = os.path.join(log_dir, "hep_parquet_generation.log")
        if os.path.exists(old_log):
            log_files = [old_log]
            
    # Sort by mtime ascending so newer run logs overwrite older run states
    log_files.sort(key=os.path.getmtime)
        
    # Match: 2026-02-28 23:59:59,999 - ... - INFO - [N1] Patient 1234-567: started processing
    # Group 1: Timestamp
    # Group 2: Level (INFO|ERROR|WARNING)
    # Group 3: Stage (e.g. N1)
    # Group 4: Patient ID
    # Group 5: Message text
    
    # regex matches:
    # 2026-02-28 23:03:36,442 - HEP_parquet_generation - INFO - [N1] Patient 0345-010: started processing
    
    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                for line in f:
                    # Basic string manipulation is often faster/more reliable than strict regex if formats change slightly
                    if " - [" in line and "] Patient " in line:
                        try:
                            parts = line.split(" - [", 1)
                            if len(parts) > 1:
                                stage_part = parts[1].split("] Patient ", 1)
                                if len(stage_part) > 1:
                                    stage = stage_part[0]
                                    patient_part = stage_part[1].split(":", 1)
                                    if len(patient_part) > 1:
                                        patient = patient_part[0]
                                        msg = patient_part[1].strip()
                                        
                                        if stage not in status_dict:
                                            status_dict[stage] = {}
                                            
                                        status = None
                                        if "started processing" in msg:
                                            status = "RUNNING"
                                        elif "successfully finished" in msg:
                                            status = "SUCCESS"
                                        elif "CRASHED" in msg:
                                            status = "FAILED"
                                        elif "Skipping" in msg:
                                            status = "SKIPPED"
                                            
                                        if status:
                                            status_dict[stage][patient] = {
                                                "status": status,
                                                "msg": msg
                                            }
                        except Exception:
                            pass
        except Exception as e:
            print(f"Error parsing logs {log_file}: {e}")
            
    return status_dict


def parse_failures(edf_root_name="Berkeley_data"):
    """
    Scan log files for this project and collect the most recent failure entry
    per patient per stage.

    Returns:
        Dict[stage, Dict[patient_id, {reason, detail, logfile, timestamp}]]
    """
    log_dir = os.path.join(BASE_DIR, "logs")
    failures = {}  # stage -> patient -> info dict

    if not os.path.exists(log_dir):
        return failures

    # Collect all log files that belong to this project
    log_files = []
    for log_path in glob.glob(os.path.join(log_dir, "run_*.log")):
        parsed = parse_run_log_filename(log_path)
        if parsed and parsed.get("project") == edf_root_name:
            log_files.append(log_path)
    # Also check old-style fallback log
    old_log = os.path.join(log_dir, "hep_parquet_generation.log")
    if not log_files and os.path.exists(old_log):
        log_files = [old_log]

    # Sort ascending by mtime so newer files overwrite older ones
    log_files.sort(key=os.path.getmtime)

    # Pattern: [stage] Patient <id>: CRASHED during processing. Error: <reason>
    crashed_re = re.compile(
        r"\[(?P<stage>[^\]]+)\] Patient (?P<patient>[^:]+): CRASHED during processing\. Error: (?P<reason>.+)"
    )
    # Pattern: [patient=<id>] Patient previously failed (reason: <reason>)
    prev_failed_re = re.compile(
        r"\[patient=(?P<patient>[^\]]+)\] Patient previously failed \(reason: (?P<reason>[^)]+)\)"
    )

    for log_path in log_files:
        logfile_name = os.path.basename(log_path)
        # Determine stage from filename if possible
        parsed_meta = parse_run_log_filename(log_path)
        file_stage = parsed_meta.get("stage") if parsed_meta else None

        try:
            with open(log_path, "r", errors="replace") as f:
                for line in f:
                    line = line.rstrip("\n")
                    # Extract timestamp prefix if present
                    ts_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                    timestamp = ts_match.group(1) if ts_match else "—"

                    # Check for CRASHED line
                    m = crashed_re.search(line)
                    if m:
                        stage = m.group("stage")
                        patient = m.group("patient").strip()
                        reason_text = m.group("reason").strip()
                        # Categorise reason
                        r_lower = reason_text.lower()
                        if "no_ecg" in r_lower or "no ecg" in r_lower:
                            reason = "no_ecg"
                        elif "no_eeg" in r_lower or "no eeg" in r_lower:
                            reason = "no_eeg"
                        else:
                            reason = "crashed"
                        if stage not in failures:
                            failures[stage] = {}
                        failures[stage][patient] = {
                            "reason": reason,
                            "detail": reason_text[:500],
                            "logfile": logfile_name,
                            "timestamp": timestamp,
                        }
                        continue

                    # Check for previously-failed skip line
                    m2 = prev_failed_re.search(line)
                    if m2:
                        patient = m2.group("patient").strip()
                        reason_text = m2.group("reason").strip()
                        r_lower = reason_text.lower()
                        if "no_ecg" in r_lower or "no ecg" in r_lower:
                            reason = "no_ecg"
                        elif "no_eeg" in r_lower or "no eeg" in r_lower:
                            reason = "no_eeg"
                        elif "processing_error" in r_lower:
                            reason = "processing_error"
                        else:
                            reason = reason_text if reason_text else "unknown"
                        # Use file-level stage if we can determine it
                        stage = file_stage or "unknown"
                        if stage not in failures:
                            failures[stage] = {}
                        # Only update if not already recorded by a CRASHED line
                        # (newer file wins because we iterate ascending mtime)
                        failures[stage][patient] = {
                            "reason": reason,
                            "detail": reason_text[:500],
                            "logfile": logfile_name,
                            "timestamp": timestamp,
                        }
        except Exception as e:
            print(f"parse_failures: error reading {log_path}: {e}")

    return failures


def get_stage_stats(stage, edf_root_name="Berkeley_data"):
    """Return (processed_count, list_of_patient_ids) for a given stage."""
    stage_dir = os.path.join(BASE_DIR, "pickles_sleep_stage", edf_root_name, stage)
    if not os.path.isdir(stage_dir):
        return 0, []
    pkl_files = glob.glob(os.path.join(stage_dir, f"*_{stage}_*.pkl"))
    patient_ids = set()
    for fpath in pkl_files:
        fname = os.path.basename(fpath)
        pid = fname.split(f"_{stage}")[0]
        patient_ids.add(pid)
    return len(patient_ids), sorted(list(patient_ids))


def get_parquet_stats(stage, edf_root_name="Berkeley_data"):
    """Return (parquet_count, list_of_patient_ids) for a given stage."""
    # The directory is parquets_HEP/{edf_root_name}_{stage}
    stage_dir = os.path.join(PARQUETS_HEP_DIR, f"{edf_root_name}_{stage}")
    if not os.path.isdir(stage_dir):
        return 0, []
    
    # We look for any results parquets for this stage
    parquet_files = glob.glob(os.path.join(stage_dir, f"*_{stage}_results_*.parquet"))
    patient_ids = set()
    for fpath in parquet_files:
        fname = os.path.basename(fpath)
        # e.g. REW609_REST_NOFILT_N1_results_alpha.parquet
        # pid = REW609_REST_NOFILT
        pid = fname.split(f"_{stage}")[0]
        patient_ids.add(pid)
    return len(patient_ids), sorted(list(patient_ids))


def get_recently_modified(stage, edf_root_name="Berkeley_data", n=5):
    """Return the n most recently modified alpha parquets for a stage."""
    stage_dir = os.path.join(BASE_DIR, "pickles_sleep_stage", edf_root_name, stage)
    if not os.path.isdir(stage_dir):
        return []
    pkl_files = glob.glob(os.path.join(stage_dir, f"*_{stage}_*.pkl"))
    pkl_files.sort(key=os.path.getmtime, reverse=True)
    result = []
    seen = set()
    for fpath in pkl_files:
        if len(result) >= n: break
        fname = os.path.basename(fpath)
        pid = fname.split(f"_{stage}")[0]
        if pid in seen: continue
        seen.add(pid)
        mtime = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M:%S")
        result.append({"patient_id": pid, "modified": mtime})
    return result


STAGE_WINDOW_NAMES = {"light_sleep", "W", "N3", "R", "N1", "N2"}
RUN_LOG_RE = re.compile(
    r"^run_(?P<project>.+?)_(?:(?P<mode>random|stage)_)?(?P<stage>light_sleep|N1|N2|N3|R|W)_(?P<timestamp>\d{8}_\d{6})\.log$"
)


def parse_run_log_filename(log_path):
    """Parse a run log filename into project/mode/stage/timestamp metadata."""
    filename = os.path.basename(log_path)
    match = RUN_LOG_RE.match(filename)
    if not match:
        return None

    parsed = match.groupdict()
    try:
        parsed["timestamp_dt"] = datetime.strptime(parsed["timestamp"], "%Y%m%d_%H%M%S")
    except ValueError:
        parsed["timestamp_dt"] = None
    parsed["filename"] = filename
    return parsed


def find_latest_run_log(stage, session=None):
    """Return the latest run log path for a given stage and optional exact project/session."""
    log_dir = os.path.join(BASE_DIR, "logs")
    if not os.path.isdir(log_dir):
        return None

    candidates = []
    for log_path in glob.glob(os.path.join(log_dir, "run_*.log")):
        parsed = parse_run_log_filename(log_path)
        if not parsed:
            continue
        if parsed["stage"] != stage:
            continue
        if session and parsed["project"] != session:
            continue
        candidates.append((log_path, parsed))

    if not candidates:
        return None

    def sort_key(item):
        log_path, parsed = item
        ts = parsed.get("timestamp_dt")
        ts_epoch = ts.timestamp() if ts else 0
        return (ts_epoch, os.path.getmtime(log_path), parsed["filename"])

    return max(candidates, key=sort_key)[0]


def get_tmux_sessions():
    """Return active tmux sessions that contain HEP stage windows."""
    import subprocess

    result = {}
    try:
        sessions_raw = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            capture_output=True, text=True
        )
        if sessions_raw.returncode != 0:
            return result

        for session_name in sessions_raw.stdout.strip().splitlines():
            if not session_name:
                continue
            windows_raw = subprocess.run(
                ["tmux", "list-windows", "-t", session_name,
                 "-F", "#{window_index}:#{window_name}:#{pane_current_command}"],
                capture_output=True, text=True
            )
            windows = []
            for win_line in windows_raw.stdout.strip().splitlines():
                parts = win_line.split(":", 2)
                if len(parts) < 2:
                    continue
                win_name = parts[1]
                if win_name not in STAGE_WINDOW_NAMES:
                    continue
                pane_cmd = parts[2] if len(parts) > 2 else ""
                is_running = "python" in pane_cmd or "python3" in pane_cmd

                # Capture last 8 lines of the pane
                pane_out = subprocess.run(
                    ["tmux", "capture-pane", "-t", f"{session_name}:{win_name}", "-p", "-S", "-8"],
                    capture_output=True, text=True
                )
                pane_lines = [l for l in pane_out.stdout.splitlines() if l.strip()][-5:]
                log_stats = get_log_stats(win_name, session=session_name)

                windows.append({
                    "stage": win_name,
                    "running": is_running,
                    "pane_cmd": pane_cmd,
                    "pane_lines": pane_lines,
                    "log_stats": log_stats,
                })
            if windows:
                result[session_name] = windows
    except Exception:
        pass
    return result


def get_log_stats(stage, session=None):
    """Return stats parsed from the most recent log file for a stage."""
    logfile = find_latest_run_log(stage, session=session)

    if not logfile:
        return None

    finished = 0
    current_patient = None
    last_info_line = ""
    last_ts_str = None

    try:
        with open(logfile, "r", errors="replace") as f:
            for line in f:
                if "successfully finished processing" in line:
                    finished += 1
                if "started processing" in line:
                    m = re.search(r"\] Patient (.+?): started processing", line)
                    if m:
                        current_patient = m.group(1)
                if " - INFO - " in line:
                    last_info_line = line
                    ts_m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                    if ts_m:
                        last_ts_str = ts_m.group(1)
    except Exception:
        pass

    last_msg = None
    if last_info_line:
        last_msg = last_info_line.split(" - ")[-1].strip()[:150]

    last_ts_epoch = 0
    if last_ts_str:
        try:
            last_ts_epoch = datetime.strptime(last_ts_str, "%Y-%m-%d %H:%M:%S").timestamp()
        except ValueError:
            pass

    return {
        "logfile": os.path.basename(logfile),
        "finished": finished,
        "current_patient": current_patient or "—",
        "last_msg": last_msg or "—",
        "last_ts": last_ts_str or "—",
        "last_ts_epoch": last_ts_epoch,
    }



def collect_data(edf_root_name="Berkeley_data"):
    target_edf_root = os.path.join(BASE_DIR, "EDF_Format", edf_root_name)
    all_patients = get_all_patients(target_edf_root)
    total = len(all_patients)
    log_status = parse_logs(edf_root_name)

    stages_data = {}
    most_active_stage = None
    most_active_epoch = 0

    for stage in STAGES:
        count, processed_patients = get_stage_stats(stage, edf_root_name)
        parquet_count, _ = get_parquet_stats(stage, edf_root_name)
        recent = get_recently_modified(stage, edf_root_name, n=5)
        stage_log_status = log_status.get(stage, {})

        patients_meta = []
        currently_running = []
        failed_patients = []

        for p in all_patients:
            if p in processed_patients:
                state = "SUCCESS"
            elif p in stage_log_status:
                state = stage_log_status[p]["status"]
            else:
                state = "PENDING"

            if state == "RUNNING":
                currently_running.append(p)
            elif state == "FAILED":
                failed_patients.append(p)

            if state != "PENDING":
                patients_meta.append({"id": p, "state": state})

        def sort_key(x):
            weight = {"RUNNING": 0, "FAILED": 1, "SUCCESS": 2, "SKIPPED": 3}
            return (weight.get(x["state"], 99), x["id"])

        patients_meta.sort(key=sort_key)

        log_stats = get_log_stats(stage, session=edf_root_name)
        if log_stats and log_stats["last_ts_epoch"] > most_active_epoch:
            most_active_epoch = log_stats["last_ts_epoch"]
            most_active_stage = stage

        stages_data[stage] = {
            "count": count,
            "parquet_count": parquet_count,
            "total": total,
            "pct": round(count / total * 100, 1) if total > 0 else 0,
            "parquet_pct": round(parquet_count / total * 100, 1) if total > 0 else 0,
            "patients_meta": patients_meta,
            "running": currently_running,
            "failed": failed_patients,
            "recent": recent,
            "color": STAGE_COLORS.get(stage, "#aaa"),
            "log_stats": log_stats,
        }

    return {
        "total": total,
        "stages": stages_data,
        "tmux_sessions": get_tmux_sessions(),
        "most_active_stage": most_active_stage,
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── HTML template ───────────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HEP Processing Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #22263a;
    --border: #2e3350;
    --text: #e2e8f0;
    --muted: #8892b0;
    --accent: #6366f1;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Inter', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 24px;
  }

  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 32px;
  }

  header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .meta {
    font-size: 0.8rem;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 8px #22c55e;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%,100% { opacity:1; }
    50% { opacity:0.4; }
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-bottom: 32px;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    transition: transform 0.2s, box-shadow 0.2s;
  }

  .card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
  }

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 18px;
  }

  .stage-badge {
    font-size: 1.1rem;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 99px;
    color: #0f1117;
    letter-spacing: 0.05em;
  }

  .count-big {
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
  }

  .count-label {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 2px;
  }

  .prog-wrap {
    background: var(--surface2);
    border-radius: 99px;
    height: 10px;
    margin: 16px 0 8px;
    overflow: hidden;
  }

  .prog-bar {
    height: 100%;
    border-radius: 99px;
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
    position: relative;
  }

  .prog-bar::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.25) 50%, transparent 100%);
    animation: shimmer 2s infinite;
  }

  @keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }

  .pct {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--muted);
  }

  .recent-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: var(--muted);
    text-transform: uppercase;
    margin-top: 18px;
    margin-bottom: 8px;
  }

  .recent-list {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .recent-list li {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--surface2);
    border-radius: 8px;
    padding: 7px 12px;
    font-size: 0.78rem;
  }

  .recent-list .pid {
    font-weight: 500;
    font-family: 'Courier New', monospace;
    color: var(--text);
  }

  .recent-list .ts {
    color: var(--muted);
    font-size: 0.72rem;
  }

  .summary-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    display: flex;
    gap: 32px;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 32px;
  }

  .summary-item { text-align: center; }
  .summary-item .num {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .summary-item .lbl {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 2px;
  }

  .all-patients-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
  }

  .all-patients-section h2 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--muted);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 16px;
  }

  .patient-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    max-height: 300px;
    overflow-y: auto;
  }

  .patient-grid::-webkit-scrollbar { width: 4px; }
  .patient-grid::-webkit-scrollbar-track { background: var(--surface2); border-radius: 4px; }
  .patient-grid::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  /* Stage tabs */
  .tabs { display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }
  .tab-btn {
    border: none; border-radius: 8px; padding: 6px 14px;
    font-size: 0.8rem; font-weight: 600; cursor: pointer;
    transition: opacity 0.2s; font-family: inherit;
    color: #0f1117;
  }
  .tab-btn.inactive { opacity: 0.4; }

  .patient-chip {
    font-size: 0.72rem;
    font-family: 'Courier New', monospace;
    padding: 4px 10px;
    border-radius: 6px;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 6px;
  }
  
  .patient-chip .status-dot {
    width: 6px; height: 6px; border-radius: 50%;
  }
  
  .chip-SUCCESS { border-color: #22c55e; background: rgba(34, 197, 94, 0.1); }
  .chip-SUCCESS .status-dot { background: #22c55e; }
  
  .chip-RUNNING { border-color: #3b82f6; background: rgba(59, 130, 246, 0.1); animation: pulsechip 2s infinite; }
  .chip-RUNNING .status-dot { background: #3b82f6; box-shadow: 0 0 6px #3b82f6; }
  
  @keyframes pulsechip {
    0%,100% { border-color: #3b82f6; }
    50% { border-color: rgba(59, 130, 246, 0.3); }
  }
  
  .chip-SKIPPED { border-color: #6b7280; background: rgba(107, 114, 128, 0.1); color: #9ca3af; }
  .chip-SKIPPED .status-dot { background: #6b7280; }
  
  .chip-FAILED { border-color: #ef4444; background: rgba(239, 68, 68, 0.1); }
  .chip-FAILED .status-dot { background: #ef4444; box-shadow: 0 0 6px #ef4444; }

  .running-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(59, 130, 246, 0.1);
    border-left: 3px solid #3b82f6;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 0.75rem;
    margin-bottom: 12px;
    color: #93c5fd;
  }
  
  .running-banner.failed {
    background: rgba(239, 68, 68, 0.1);
    border-left: 3px solid #ef4444;
    color: #fca5a5;
  }

  footer {
    margin-top: 24px;
    text-align: center;
    font-size: 0.75rem;
    color: var(--muted);
  }

  /* ── Log viewer modal ── */
  .log-link {
    cursor: pointer;
    text-decoration: underline dotted;
    transition: color 0.15s;
  }
  .log-link:hover { color: #818cf8 !important; }

  #log-modal {
    display: none;
    position: fixed; inset: 0;
    background: rgba(0,0,0,0.75);
    z-index: 1000;
    align-items: center;
    justify-content: center;
  }
  #log-modal.open { display: flex; }

  #log-box {
    background: #0d0f18;
    border: 1px solid var(--border);
    border-radius: 14px;
    width: min(96vw, 1100px);
    height: 80vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 24px 80px rgba(0,0,0,0.7);
  }

  #log-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 18px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    gap: 10px;
  }

  #log-filename {
    font-family: monospace;
    font-size: 0.8rem;
    color: #94a3b8;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  #log-controls {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-shrink: 0;
  }

  .log-btn {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 7px;
    padding: 4px 12px;
    font-size: 0.75rem;
    cursor: pointer;
    font-family: inherit;
    transition: background 0.15s;
    white-space: nowrap;
  }
  .log-btn:hover { background: var(--accent); border-color: var(--accent); }
  .log-btn:disabled { opacity: 0.35; cursor: default; }

  #log-body {
    flex: 1;
    overflow-y: scroll;
    padding: 10px 0;
    scroll-behavior: smooth;
  }

  #log-lines {
    font-family: 'Courier New', monospace;
    font-size: 0.72rem;
    line-height: 1.55;
    padding: 0 16px;
    white-space: pre-wrap;
    word-break: break-all;
  }

  .ll { display: flex; gap: 10px; padding: 1px 0; border-radius: 3px; }
  .ll:hover { background: rgba(255,255,255,0.04); }
  .ln { color: #374151; min-width: 52px; text-align: right; user-select: none; flex-shrink: 0; }
  .lt { flex: 1; }
  .lt.info    { color: #93c5fd; }
  .lt.warning { color: #fbbf24; }
  .lt.error   { color: #f87171; }
  .lt.default { color: #64748b; }

  #log-status {
    font-size: 0.7rem;
    color: var(--muted);
    padding: 6px 18px;
    border-top: 1px solid var(--border);
    flex-shrink: 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .nav-tab {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--muted);
    border-radius: 8px;
    padding: 8px 20px;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    font-family: inherit;
    transition: all 0.2s;
  }
  .nav-tab:hover { border-color: var(--accent); color: var(--text); }
  .active-tab { background: var(--accent); border-color: var(--accent); color: #fff; }

  .failure-stage-section { margin-bottom: 24px; }
  .failure-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  .failure-table th { text-align: left; padding: 8px 12px; color: var(--muted); font-weight: 600; border-bottom: 1px solid var(--border); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }
  .failure-table td { padding: 8px 12px; border-bottom: 1px solid rgba(46,51,80,0.5); vertical-align: top; }
  .failure-table tr:hover td { background: var(--surface2); }
  .failure-reason-badge { display: inline-block; padding: 2px 8px; border-radius: 99px; font-size: 0.7rem; font-weight: 600; }
  .reason-no_ecg { background: rgba(251,146,60,0.2); color: #fb923c; border: 1px solid rgba(251,146,60,0.3); }
  .reason-no_eeg { background: rgba(167,139,250,0.2); color: #a78bfa; border: 1px solid rgba(167,139,250,0.3); }
  .reason-processing_error { background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
  .reason-crashed { background: rgba(239,68,68,0.3); color: #fca5a5; border: 1px solid rgba(239,68,68,0.5); }
  .reason-unknown { background: rgba(107,114,128,0.2); color: #9ca3af; border: 1px solid rgba(107,114,128,0.3); }
  .open-log-btn { background: var(--surface2); border: 1px solid var(--border); color: var(--text); border-radius: 5px; padding: 3px 10px; font-size: 0.7rem; cursor: pointer; font-family: inherit; transition: background 0.15s; white-space: nowrap; }
  .open-log-btn:hover { background: var(--accent); border-color: var(--accent); }
  .detail-text { color: #64748b; font-family: monospace; font-size: 0.68rem; word-break: break-word; max-width: 350px; }
</style>
</head>
<body>

<header>
  <div style="display: flex; align-items: center; gap: 20px;">
      <h1>🧠 HEP Processing Dashboard</h1>
      <select id="dir-select" style="background: var(--surface2); border: 1px solid var(--border); color: var(--text); padding: 8px 12px; border-radius: 8px; font-size: 0.9rem; font-family: 'Inter', sans-serif; cursor: pointer; outline: none;">
          <option value="Berkeley_data">Berkeley_data (Loading...)</option>
      </select>
  </div>
  <div class="meta">
    <div class="dot"></div>
    <span id="updated">Loading…</span>
    &nbsp;|&nbsp;
    <span id="countdown">Refreshes in <b id="sec">30</b>s</span>
  </div>
</header>

<!-- Page navigation -->
<div id="page-nav" style="display:flex;gap:8px;margin-bottom:20px;">
  <button id="nav-overview" class="nav-tab active-tab" onclick="showPage('overview')">Overview</button>
  <button id="nav-failures" class="nav-tab" onclick="showPage('failures')">&#9888; Failures</button>
</div>

<div id="overview-page">
<!-- Summary row -->
<div class="summary-card" id="summary-card">
  <div class="summary-item">
    <div class="num" id="total-patients">—</div>
    <div class="lbl">Total Patients</div>
  </div>
  <!-- per-stage summary injected by JS -->
</div>

<!-- Active Tmux Sessions -->
<div class="summary-card" style="flex-direction: column; align-items: stretch;" id="tmux-container">
  <h2 style="font-size:1rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:16px;">Active Tmux Sessions</h2>
  <div id="tmux-grid" style="display: flex; flex-direction: column; gap: 20px;">
      <!-- tmux session entries injected here -->
  </div>
</div>

<!-- Stage cards -->
<div class="grid" id="cards-grid"></div>

<!-- Patient list -->
<div class="all-patients-section">
  <h2>Processed Patients</h2>
  <div class="tabs" id="stage-tabs"></div>
  <div class="patient-grid" id="patient-grid"></div>
</div>

<footer>Auto-refreshes every 30 seconds &nbsp;·&nbsp; Cobrad HEP Pipeline</footer>

</div>
<!-- end overview-page -->

<!-- Failures page -->
<div id="failures-page" style="display:none;">
  <div id="failures-content">
    <p style="color:var(--muted);font-size:0.85rem;">Loading failure data...</p>
  </div>
</div>

<!-- Log viewer modal -->
<div id="log-modal" role="dialog" aria-modal="true">
  <div id="log-box">
    <div id="log-header">
      <span id="log-filename">—</span>
      <div id="log-controls">
        <button class="log-btn" id="log-older-btn" onclick="logLoadOlder()" title="Load 500 older lines (PgUp)">⬆ Load older</button>
        <button class="log-btn" id="log-newer-btn" onclick="logLoadNewer()" title="Load newer lines (PgDn)">⬇ Load newer</button>
        <button class="log-btn" onclick="logScrollBottom()" title="Jump to bottom (End)">↓ Bottom</button>
        <button class="log-btn" onclick="logScrollTop()" title="Jump to top (Home)">↑ Top</button>
        <button class="log-btn" onclick="closeLog()" style="background:#1e1e2e;" title="Close (Esc)">✕ Close</button>
      </div>
    </div>
    <div id="log-body" tabindex="0">
      <div id="log-lines"></div>
    </div>
    <div id="log-status">
      <span id="log-info">—</span>
      <span style="color:#374151;font-size:0.65rem;">↑↓ scroll &nbsp;·&nbsp; PgUp/PgDn load more &nbsp;·&nbsp; Esc close</span>
    </div>
  </div>
</div>

<script>
const STAGE_COLORS = {R:"#f87171",W:"#60a5fa",N1:"#a78bfa",N2:"#34d399",N3:"#fb923c",light_sleep:"#34d399"};
const STAGES = ["R","W","light_sleep","N3"];
let currentTab = "light_sleep";
let lastData = null;

function escHtml(s) {
  if (!s) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

async function fetchData(){
  try {
    const dir = document.getElementById('dir-select').value;
    const r = await fetch('/api/data?dir=' + encodeURIComponent(dir));
    lastData = await r.json();
    render(lastData);
    if (currentPage === 'failures') fetchFailures();
  } catch(e){ console.error(e); }
}

async function fetchDirectories() {
    try {
        const r = await fetch('/api/directories');
        const dirs = await r.json();
        const select = document.getElementById('dir-select');
        
        // Save currently selected option
        const currentSelection = select.value;
        select.innerHTML = '';
        
        dirs.forEach(d => {
            const opt = document.createElement('option');
            opt.value = d;
            opt.textContent = "📁 " + d;
            select.appendChild(opt);
        });
        
        // Restore selection or default to first
        if (dirs.includes(currentSelection)) {
            select.value = currentSelection;
        } else if (dirs.includes('Berkeley_data')) {
            select.value = 'Berkeley_data';
        } else if (dirs.length > 0) {
            select.value = dirs[0];
        }
        
    } catch(e) { console.error("Failed to fetch directories:", e); }
}

// Add event listener to dir selector
document.getElementById('dir-select').addEventListener('change', () => {
    secs = 30; // reset timer
    document.getElementById('cards-grid').innerHTML = '<div style="color:var(--muted);">Loading new directory data...</div>';
    fetchData();
});

function relativeTime(tsStr) {
  if (!tsStr || tsStr === '—') return '—';
  const d = new Date(tsStr.replace(',', '.'));
  if (isNaN(d)) return tsStr;
  const diffS = Math.floor((Date.now() - d) / 1000);
  if (diffS < 60) return diffS + 's ago';
  if (diffS < 3600) return Math.floor(diffS/60) + 'm ago';
  if (diffS < 86400) return Math.floor(diffS/3600) + 'h ago';
  return Math.floor(diffS/86400) + 'd ago';
}

function render(data){
  document.getElementById('updated').textContent = 'Updated: ' + data.updated;
  document.getElementById('total-patients').textContent = data.total;

  // Tmux Sessions
  const tg = document.getElementById('tmux-grid');
  tg.innerHTML = '';
  const tmux = data.tmux_sessions || {};
  if (Object.keys(tmux).length === 0) {
    tg.innerHTML = '<span style="color:var(--muted);font-size:0.8rem;">No active tmux sessions detected.</span>';
  } else {
    for (const [sessName, windows] of Object.entries(tmux)) {
      const sdiv = document.createElement('div');
      sdiv.style.cssText = 'background:var(--surface2);border-radius:8px;padding:16px;';
      let html = `<h3 style="font-size:0.9rem;margin-bottom:12px;color:var(--accent);">📺 Session: ${sessName}</h3>`;
      html += `<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;">`;
      windows.forEach(w => {
        const color = STAGE_COLORS[w.stage] || '#aaa';
        const statusDot = w.running
          ? `<div class="dot" style="width:7px;height:7px;background:#22c55e;"></div><span style="color:#22c55e;font-size:0.75rem;">RUNNING</span>`
          : `<div class="dot" style="width:7px;height:7px;background:#ef4444;box-shadow:none;animation:none;"></div><span style="color:#ef4444;font-size:0.75rem;">STOPPED</span>`;
        const paneHtml = w.pane_lines && w.pane_lines.length
          ? w.pane_lines.map(l=>`<div style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${escHtml(l)}</div>`).join('')
          : '<span style="color:var(--muted);">—</span>';
        const stageLogFile = w.log_stats ? w.log_stats.logfile : null;
        const logFileLine = stageLogFile
          ? `<div class="log-link" style="font-size:0.63rem;font-family:monospace;color:#475569;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="Click to open log" onclick="openLog('${escHtml(stageLogFile)}')">📄 ${escHtml(stageLogFile)}</div>`
          : '';
        html += `
          <div style="background:#0f1117;padding:12px;border-radius:8px;border-left:3px solid ${color};">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
              <span style="font-weight:700;font-size:0.85rem;color:${color};">${w.stage}</span>
              <div style="display:flex;align-items:center;gap:5px;">${statusDot}</div>
            </div>
            ${logFileLine}
            <div style="font-size:0.68rem;font-family:monospace;color:#9ca3af;background:#0a0c14;padding:6px 8px;border-radius:4px;max-height:80px;overflow:hidden;margin-top:6px;">${paneHtml}</div>
          </div>`;
      });
      html += `</div>`;
      sdiv.innerHTML = html;
      tg.appendChild(sdiv);
    }
  }

  // Summary extra items
  const sc = document.getElementById('summary-card');
  // remove old stage summaries
  sc.querySelectorAll('.stage-summary').forEach(el=>el.remove());
  STAGES.forEach(s=>{
    const sd = data.stages[s];
    const div = document.createElement('div');
    div.className = 'summary-item stage-summary';
    div.innerHTML = `
      <div class="num" style="background:linear-gradient(135deg,${sd.color},${sd.color}aa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">${sd.count} <small style="font-size:0.5em;color:var(--muted)">pkl</small> | ${sd.parquet_count} <small style="font-size:0.5em;color:var(--muted)">pq</small></div>
      <div class="lbl">Stage ${s}</div>
    `;
    sc.appendChild(div);
  });

  // Cards
  const grid = document.getElementById('cards-grid');
  grid.innerHTML = '';
  STAGES.forEach(s=>{
    const sd = data.stages[s];
    const card = document.createElement('div');
    card.className = 'card';
    const recentHtml = sd.recent.length ? sd.recent.map(r=>
      `<li><span class="pid">${r.patient_id}</span><span class="ts">${r.modified}</span></li>`
    ).join('') : '<li style="color:var(--muted);font-size:0.78rem;">No parquets generated yet</li>';
    
    let activeStatusHtml = '';
    if (sd.running.length > 0) {
        activeStatusHtml += `<div class="running-banner">
            <div class="dot" style="width:6px;height:6px;background:#3b82f6;box-shadow:none;"></div>
            Currently running: <b>${sd.running.join(", ")}</b>
        </div>`;
    } else {
        activeStatusHtml += `<div class="running-banner" style="border-left-color: #6b7280; background: rgba(107, 114, 128, 0.1); color: #9ca3af;">
            <div class="dot" style="width:6px;height:6px;background:#6b7280;box-shadow:none;animation:none;"></div>
            Idle (No patient running)
        </div>`;
    }
    
    if (sd.failed.length > 0) {
        activeStatusHtml += `<div class="running-banner failed">
            <div class="dot" style="width:6px;height:6px;background:#ef4444;box-shadow:none;"></div>
            Crashed: <b>${sd.failed.join(", ")}</b>
        </div>`;
    }
    
    const ls = sd.log_stats;
    const isMostActive = s === data.most_active_stage;
    const mostActiveBadge = isMostActive
      ? `<span style="font-size:0.65rem;background:#6366f1;color:#fff;padding:2px 7px;border-radius:99px;margin-left:8px;">⚡ most active</span>`
      : '';

    let logStatsHtml = '';
    if (ls) {
      logStatsHtml = `
        <div class="recent-title">Live Log Stats</div>
        <div style="background:var(--surface2);border-radius:8px;padding:10px 12px;font-size:0.75rem;display:flex;flex-direction:column;gap:5px;">
          <div style="display:flex;justify-content:space-between;gap:8px;border-bottom:1px solid var(--border);padding-bottom:5px;margin-bottom:2px;">
            <span style="color:var(--muted);white-space:nowrap;">Log file</span>
            <span class="log-link" style="font-family:monospace;font-size:0.68rem;color:#64748b;text-align:right;word-break:break-all;" title="Click to open log" onclick="openLog('${escHtml(ls.logfile)}')">${escHtml(ls.logfile)}</span>
          </div>
          <div style="display:flex;justify-content:space-between;">
            <span style="color:var(--muted);">Finished</span>
            <span style="color:#22c55e;font-weight:600;">${ls.finished}</span>
          </div>
          <div style="display:flex;justify-content:space-between;gap:8px;">
            <span style="color:var(--muted);white-space:nowrap;">Current patient</span>
            <span style="font-family:monospace;color:#e2e8f0;text-align:right;word-break:break-all;">${escHtml(ls.current_patient)}</span>
          </div>
          <div style="border-top:1px solid var(--border);padding-top:5px;margin-top:2px;">
            <div style="color:var(--muted);margin-bottom:3px;">Last step</div>
            <div style="font-family:monospace;font-size:0.68rem;color:#93c5fd;word-break:break-word;">${escHtml(ls.last_msg)}</div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:2px;">
            <span style="color:var(--muted);">Last activity</span>
            <span style="color:${isMostActive?'#a78bfa':'var(--muted)'};">${relativeTime(ls.last_ts)} <span style="font-size:0.65rem;opacity:0.6;">(${ls.last_ts})</span></span>
          </div>
        </div>`;
    }

    card.innerHTML = `
      <div class="card-header">
        <div>
          <div class="count-big" style="color:${sd.color}">${sd.count} <small style="font-size:0.5em;opacity:0.6;">pkl</small> | ${sd.parquet_count} <small style="font-size:0.5em;opacity:0.6;">pq</small></div>
          <div class="count-label">/ ${sd.total} total patients</div>
        </div>
        <div style="display:flex;flex-direction:column;align-items:flex-end;gap:4px;">
          <span class="stage-badge" style="background:${sd.color}">${s}</span>
          ${mostActiveBadge}
        </div>
      </div>
      <div class="prog-wrap">
        <div class="prog-bar" style="width:${sd.pct}%;background:${sd.color}"></div>
      </div>
      <div class="pct">${sd.pct}% pickles generated</div>
      <div class="prog-wrap" style="height:6px; margin-top:12px;">
        <div class="prog-bar" style="width:${sd.parquet_pct}%;background:${sd.color};opacity:0.7;"></div>
      </div>
      <div class="pct">${sd.parquet_pct}% parquets generated</div>
      <div style="margin-top:20px;">${activeStatusHtml}</div>
      ${logStatsHtml}
      <div class="recent-title">Recently Generated Pickles</div>
      <ul class="recent-list">${recentHtml}</ul>
    `;
    grid.appendChild(card);
  });

  // Tabs
  const tabs = document.getElementById('stage-tabs');
  tabs.innerHTML = '';
  STAGES.forEach(s=>{
    const btn = document.createElement('button');
    btn.className = 'tab-btn' + (s===currentTab?'':' inactive');
    btn.style.background = STAGE_COLORS[s];
    btn.textContent = `${s} (pkl:${data.stages[s].count} | pq:${data.stages[s].parquet_count})`;
    btn.onclick = ()=>{ currentTab=s; renderPatients(data); updateTabStyles(); };
    tabs.appendChild(btn);
  });
  renderPatients(data);
}

function updateTabStyles(){
  document.querySelectorAll('.tab-btn').forEach((btn,i)=>{
    btn.classList.toggle('inactive', STAGES[i]!==currentTab);
  });
}

function renderPatients(data){
  const pg = document.getElementById('patient-grid');
  pg.innerHTML = '';
  const meta = data.stages[currentTab].patients_meta;
  if(!meta || meta.length===0){
    pg.innerHTML = '<span style="color:var(--muted);font-size:0.8rem;">No records logged or processed yet for this stage.</span>';
    return;
  }
  meta.forEach(p=>{
    const chip = document.createElement('div');
    chip.className = `patient-chip chip-${p.state}`;
    // map hover text
    chip.title = `Patient: ${p.id}\nStatus: ${p.state}`;
    chip.innerHTML = `<div class="status-dot"></div> ${p.id}`;
    pg.appendChild(chip);
  });
}

// ── Page navigation ────────────────────────────────────────────────────────────
let currentPage = 'overview';
let lastFailureData = null;

function showPage(page) {
  currentPage = page;
  document.getElementById('overview-page').style.display = page === 'overview' ? '' : 'none';
  document.getElementById('failures-page').style.display = page === 'failures' ? '' : 'none';
  document.getElementById('nav-overview').classList.toggle('active-tab', page === 'overview');
  document.getElementById('nav-failures').classList.toggle('active-tab', page === 'failures');
  if (page === 'failures' && !lastFailureData) fetchFailures();
}

async function fetchFailures() {
  try {
    const dir = document.getElementById('dir-select').value;
    const r = await fetch('/api/failures?dir=' + encodeURIComponent(dir));
    lastFailureData = await r.json();
    renderFailures(lastFailureData);
  } catch(e) { console.error('fetchFailures error:', e); }
}

function getReasonClass(reason) {
  if (!reason) return 'reason-unknown';
  const r = reason.toLowerCase();
  if (r === 'no_ecg') return 'reason-no_ecg';
  if (r === 'no_eeg') return 'reason-no_eeg';
  if (r === 'processing_error') return 'reason-processing_error';
  if (r === 'crashed') return 'reason-crashed';
  return 'reason-unknown';
}

function renderFailures(data) {
  const container = document.getElementById('failures-content');
  if (!data || !data.stages) { container.innerHTML = '<p style="color:var(--muted);">No failure data available.</p>'; return; }

  const stages = data.stages;
  let totalFailed = 0;
  Object.values(stages).forEach(arr => totalFailed += arr.length);

  let html = `<div style="display:flex;align-items:center;gap:16px;margin-bottom:24px;flex-wrap:wrap;">
    <h2 style="font-size:1.1rem;font-weight:700;color:var(--text);">Failure Report</h2>
    <span style="font-size:0.8rem;color:var(--muted);">Updated: ${escHtml(data.updated || '—')}</span>
    <span style="background:rgba(239,68,68,0.2);color:#f87171;border:1px solid rgba(239,68,68,0.3);border-radius:99px;padding:3px 12px;font-size:0.78rem;font-weight:600;">${totalFailed} total failures</span>
  </div>`;

  if (totalFailed === 0) {
    html += '<div class="card" style="text-align:center;padding:40px;"><span style="font-size:2rem;">&#10003;</span><p style="color:#22c55e;margin-top:8px;font-weight:600;">No failures recorded!</p></div>';
  } else {
    STAGES.forEach(stage => {
      const failures = stages[stage] || [];
      if (failures.length === 0) return;
      const color = STAGE_COLORS[stage] || '#aaa';
      html += `<div class="failure-stage-section">
        <div class="card">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
            <span class="stage-badge" style="background:${color}">${stage}</span>
            <span style="color:var(--muted);font-size:0.8rem;">${failures.length} failure${failures.length !== 1 ? 's' : ''}</span>
          </div>
          <table class="failure-table">
            <thead><tr>
              <th>Patient ID</th><th>Reason</th><th>Detail</th><th>Timestamp</th><th>Log</th>
            </tr></thead>
            <tbody>`;
      failures.forEach(f => {
        const reasonClass = getReasonClass(f.reason);
        const logBtn = f.logfile && f.logfile !== '—'
          ? `<button class="open-log-btn" onclick="openLog('${escHtml(f.logfile)}')">&#128196; View Log</button>`
          : '<span style="color:var(--muted);font-size:0.7rem;">—</span>';
        html += `<tr>
          <td style="font-family:monospace;font-weight:500;">${escHtml(f.patient)}</td>
          <td><span class="failure-reason-badge ${reasonClass}">${escHtml(f.reason || 'unknown')}</span></td>
          <td><span class="detail-text">${escHtml((f.detail || '').substring(0, 200))}</span></td>
          <td style="color:var(--muted);white-space:nowrap;font-size:0.72rem;">${escHtml(f.timestamp || '—')}</td>
          <td>${logBtn}</td>
        </tr>`;
      });
      html += `</tbody></table></div></div>`;
    });
  }
  container.innerHTML = html;
}

// ── Log viewer ────────────────────────────────────────────────────────────────
let logState = { file: null, total: 0, start: 0, end: 0, loading: false };

function openLog(filename) {
  if (!filename || filename === '—') return;
  logState = { file: filename, total: 0, start: 0, end: 0, loading: false };
  document.getElementById('log-filename').textContent = filename;
  document.getElementById('log-lines').innerHTML = '';
  document.getElementById('log-info').textContent = 'Loading…';
  document.getElementById('log-modal').classList.add('open');
  document.getElementById('log-body').focus();
  // Load last 500 lines (offset=0 → tail)
  logFetch(0, true);
}

function closeLog() {
  document.getElementById('log-modal').classList.remove('open');
}

async function logFetch(offset, isInitial) {
  if (logState.loading) return;
  logState.loading = true;
  updateLogButtons();
  try {
    const r = await fetch(`/api/log?file=${encodeURIComponent(logState.file)}&offset=${offset}&lines=500`);
    if (!r.ok) { document.getElementById('log-info').textContent = 'Error loading file.'; return; }
    const d = await r.json();
    logState.total = d.total;
    logState.start = d.start;
    logState.end   = d.end;
    renderLogLines(d.lines, d.start);
    if (isInitial) logScrollBottom();
    updateLogInfo();
  } catch(e) {
    document.getElementById('log-info').textContent = 'Fetch error: ' + e;
  } finally {
    logState.loading = false;
    updateLogButtons();
  }
}

function renderLogLines(lines, startLineNum) {
  const container = document.getElementById('log-lines');
  container.innerHTML = '';
  lines.forEach((line, i) => {
    const lineNum = startLineNum + i + 1;
    let cls = 'default';
    if (line.includes(' - INFO - '))    cls = 'info';
    else if (line.includes(' - WARNING - ') || line.includes(' - WARN - ')) cls = 'warning';
    else if (line.includes(' - ERROR - ') || line.includes(' - CRITICAL - ')) cls = 'error';
    const row = document.createElement('div');
    row.className = 'll';
    row.innerHTML = `<span class="ln">${lineNum}</span><span class="lt ${cls}">${escHtml(line)}</span>`;
    container.appendChild(row);
  });
}

function logLoadOlder() {
  // Load 500 lines before current start
  const newOffset = logState.total - logState.start;
  if (newOffset >= logState.total) return;
  logFetch(newOffset, false).then(() => logScrollTop());
}

function logLoadNewer() {
  // Load 500 lines after current end (offset from end = total - end - 500)
  const newOffset = Math.max(0, logState.total - logState.end - 500);
  logFetch(newOffset, false).then(() => {
    if (newOffset === 0) logScrollBottom();
  });
}

function logScrollBottom() {
  const body = document.getElementById('log-body');
  body.scrollTop = body.scrollHeight;
}
function logScrollTop() {
  document.getElementById('log-body').scrollTop = 0;
}

function updateLogInfo() {
  const { start, end, total } = logState;
  document.getElementById('log-info').textContent =
    `Lines ${start + 1}–${end} of ${total}`;
}

function updateLogButtons() {
  const atTop = logState.start === 0;
  const atBottom = logState.end >= logState.total;
  document.getElementById('log-older-btn').disabled = atTop || logState.loading;
  document.getElementById('log-newer-btn').disabled = atBottom || logState.loading;
}

// Click outside modal to close
document.getElementById('log-modal').addEventListener('click', e => {
  if (e.target === document.getElementById('log-modal')) closeLog();
});

// Keyboard navigation
document.addEventListener('keydown', e => {
  if (!document.getElementById('log-modal').classList.contains('open')) return;
  const body = document.getElementById('log-body');
  if (e.key === 'Escape')    { closeLog(); return; }
  if (e.key === 'End')       { logScrollBottom(); e.preventDefault(); return; }
  if (e.key === 'Home')      { logScrollTop(); e.preventDefault(); return; }
  if (e.key === 'ArrowDown') { body.scrollTop += 40; e.preventDefault(); return; }
  if (e.key === 'ArrowUp')   { body.scrollTop -= 40; e.preventDefault(); return; }
  if (e.key === 'PageDown')  { logLoadNewer(); e.preventDefault(); return; }
  if (e.key === 'PageUp')    { logLoadOlder(); e.preventDefault(); return; }
});

// ── Countdown timer ────────────────────────────────────────────────────────────
let secs = 30;
function tick(){
  document.getElementById('sec').textContent = secs;
  if(secs===0){ secs=30; fetchData(); } else { secs--; }
}
setInterval(tick, 1000);

// Initialize
fetchDirectories().then(() => {
    fetchData();
});
</script>
</body>
</html>
"""


from urllib.parse import urlparse, parse_qs

class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence default access log

    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == "/" or parsed_path.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())

        elif parsed_path.path == "/api/directories":
            edf_format_root = os.path.join(BASE_DIR, "EDF_Format")
            try:
                # get subdirectories, ignoring weird files
                directories = [d for d in os.listdir(edf_format_root) if os.path.isdir(os.path.join(edf_format_root, d))]
                directories.sort()
            except Exception:
                directories = ["Berkeley_data"]
                
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(directories).encode())

        elif parsed_path.path == "/api/data":
            # Extract dir parameter or default to Berkeley_data
            query_params = parse_qs(parsed_path.query)
            edf_root_name = query_params.get('dir', ['Berkeley_data'])[0]
            
            data = collect_data(edf_root_name)
            payload = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(payload)

        elif parsed_path.path == "/api/failures":
            query_params = parse_qs(parsed_path.query)
            edf_root_name = query_params.get('dir', ['Berkeley_data'])[0]
            failures = parse_failures(edf_root_name)
            # Convert nested dict to JSON-serialisable structure
            stages_out = {}
            for stage, patients in failures.items():
                entries = []
                for patient_id, info in patients.items():
                    entries.append({
                        "patient": patient_id,
                        "reason": info.get("reason", "unknown"),
                        "detail": info.get("detail", ""),
                        "logfile": info.get("logfile", "—"),
                        "timestamp": info.get("timestamp", "—"),
                    })
                # Sort by patient id for deterministic output
                entries.sort(key=lambda x: x["patient"])
                stages_out[stage] = entries
            payload = json.dumps({
                "stages": stages_out,
                "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(payload)

        elif parsed_path.path == "/api/log":
            query_params = parse_qs(parsed_path.query)
            filename = os.path.basename(query_params.get('file', [''])[0])
            offset   = max(0, int(query_params.get('offset', [0])[0]))
            n_lines  = min(2000, max(1, int(query_params.get('lines', [500])[0])))

            log_path = os.path.join(BASE_DIR, "logs", filename)
            if not filename or not os.path.isfile(log_path):
                self.send_response(404)
                self.end_headers()
                return

            with open(log_path, "r", errors="replace") as f:
                all_lines = f.readlines()

            total = len(all_lines)
            # offset counts from the end; offset=0 → last n_lines
            end   = max(0, total - offset)
            start = max(0, end - n_lines)
            chunk = [l.rstrip("\n") for l in all_lines[start:end]]

            payload = json.dumps({
                "filename": filename,
                "lines":    chunk,
                "total":    total,
                "start":    start,
                "end":      end,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(payload)

        else:
            self.send_response(404)
            self.end_headers()


def find_free_port(start=8765):
    import socket
    port = start
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                port += 1


def run(port=8765):
    port = find_free_port(port)
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"✅  Dashboard running at  http://localhost:{port}")
    print(f"    Serving data from:     {PARQUETS_HEP_DIR}")
    print(f"    EDF root:              {EDF_ROOT}")
    print("    Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HEP Processing Dashboard")
    parser.add_argument("--port", type=int, default=8765, help="Port to serve on (default: 8765)")
    args = parser.parse_args()
    run(port=args.port)
