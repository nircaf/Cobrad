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

STAGES = ["R", "W", "N1", "N2", "N3"]
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


def get_zellij_sessions():
    """Scrapes running python processes to extract active dashboard data per-project."""
    import subprocess
    sessions_by_project = {}
    
    try:
        # ps format: pid,etimes,command
        ps_output = subprocess.check_output(
            ["ps", "-eo", "pid,etimes,command"], 
            text=True, stderr=subprocess.DEVNULL
        )
        
        worker_id = 1
        for line in ps_output.split("\n"):
            if "HEP_parquet_generation.py" in line and "python" in line and not "grep" in line:
                parts = line.split(None, 2)
                if len(parts) < 3:
                    continue
                
                pid = parts[0]
                etimes = int(parts[1])
                cmd = parts[2]
                
                # Format uptime
                h = etimes // 3600
                m = (etimes % 3600) // 60
                if h > 0:
                    uptime_str = f"{h}h {m}m"
                else:
                    uptime_str = f"{m}m"
                
                # Extract --stage
                stage = "Unknown"
                stage_match = re.search(r'--stage\s+(\w+)', cmd)
                if stage_match:
                    stage = stage_match.group(1)
                    
                # Extract --edf_root
                project = "Unknown Project"
                root_match = re.search(r'--edf_root\s+([^\s;]+)', cmd)
                if root_match:
                    project = root_match.group(1)
                    project = project.strip('\'"')
                
                # Extract patient from command path
                patient = "Unknown"
                m_pat = re.search(r'(\d{4}-\d{3})', cmd)
                if m_pat:
                    patient = m_pat.group(1)
                else:
                    patient = os.path.basename(project).split('.')[0]
                
                # We can't easily capture Zellij pane screen for tqdm progress, so we just say processing
                status_text = "Processing..."
                
                # For UI display logic
                if project not in sessions_by_project:
                    sessions_by_project[project] = []
                    
                sessions_by_project[project].append({
                    "session_id": f"worker_{worker_id}",
                    "stage": stage,
                    "patient": patient,
                    "progress": status_text,
                    "uptime": uptime_str
                })
                worker_id += 1
                
    except subprocess.CalledProcessError:
        pass
        
    return sessions_by_project



def collect_data(edf_root_name="Berkeley_data"):
    target_edf_root = os.path.join(BASE_DIR, "EDF_Format", edf_root_name)
    all_patients = get_all_patients(target_edf_root)
    total = len(all_patients)
    log_status = parse_logs(edf_root_name)
    
    stages_data = {}
    for stage in STAGES:
        count, processed_patients = get_stage_stats(stage, edf_root_name)
        parquet_count, _ = get_parquet_stats(stage, edf_root_name)
        recent = get_recently_modified(stage, edf_root_name, n=5)
        
        # Combine processed items with log status
        stage_log_status = log_status.get(stage, {})
        
        # build combined patient list with statuses
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
                
            # If not pending, or if we just want to track everything:
            # We'll just append them all or only ones that have started/finished
            if state != "PENDING":
                patients_meta.append({
                    "id": p,
                    "state": state
                })
        
        # Sort so running/failed are usually visible, then successful
        def sort_key(x):
            weight = {"RUNNING": 0, "FAILED": 1, "SUCCESS": 2, "SKIPPED": 3}
            return (weight.get(x["state"], 99), x["id"])
            
        patients_meta.sort(key=sort_key)
        
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
            "color": STAGE_COLORS[stage],
        }
    return {
        "total": total,
        "stages": stages_data,
        "zellij_sessions": get_zellij_sessions(),
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

<!-- Summary row -->
<div class="summary-card" id="summary-card">
  <div class="summary-item">
    <div class="num" id="total-patients">—</div>
    <div class="lbl">Total Patients</div>
  </div>
  <!-- per-stage summary injected by JS -->
</div>

<!-- Active Processing Zellij Sessions -->
<div class="summary-card" style="flex-direction: column; align-items: stretch;" id="zellij-container">
  <h2 style="font-size:1rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:16px;">Active Processing Tabs (Zellij)</h2>
  <div id="zellij-grid" style="display: flex; flex-direction: column; gap: 20px;">
      <!-- project entries injected here -->
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

<script>
const STAGE_COLORS = {R:"#f87171",W:"#60a5fa",N1:"#a78bfa",N2:"#34d399",N3:"#fb923c"};
const STAGES = ["R","W","N1","N2","N3"];
let currentTab = "N1";
let lastData = null;

async function fetchData(){
  try {
    const dir = document.getElementById('dir-select').value;
    const r = await fetch('/api/data?dir=' + encodeURIComponent(dir));
    lastData = await r.json();
    render(lastData);
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

function render(data){
  document.getElementById('updated').textContent = 'Updated: ' + data.updated;
  document.getElementById('total-patients').textContent = data.total;

  // Zellij Container Update
  const tc = document.getElementById('zellij-grid');
  tc.innerHTML = '';
  if (!data.zellij_sessions || Object.keys(data.zellij_sessions).length === 0) {
      tc.innerHTML = '<span style="color:var(--muted);font-size:0.8rem;">No active processing workers detected.</span>';
  } else {
      for (const [project, sessions] of Object.entries(data.zellij_sessions)) {
          const pdiv = document.createElement('div');
          pdiv.style.background = "var(--surface2)";
          pdiv.style.borderRadius = "8px";
          pdiv.style.padding = "16px";
          
          let sessHtml = `<h3 style="font-size:0.9rem;margin-bottom:12px;color:var(--accent);">📁 Project: ${project}</h3>`;
          sessHtml += `<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(250px, 1fr));gap:12px;">`;
          
          sessions.forEach(s => {
              const color = STAGE_COLORS[s.stage] || '#fff';
              let statHtml = '';
              
              if (s.patient.includes("Idle")) {
                  statHtml = `
                    <div style="font-size:0.75rem; color:var(--muted); display:flex; align-items:center; gap:6px;">
                        <div class="dot" style="width:6px;height:6px;background:#6b7280;box-shadow:none;animation:none;"></div>
                        ${s.patient}
                    </div>`;
              } else {
                  statHtml = `
                    <div style="font-size:0.8rem; font-family:monospace; margin-bottom:6px; color:#e2e8f0; display:flex; align-items:center; gap:6px;">
                        <div class="dot" style="width:6px;height:6px;background:#3b82f6;"></div>
                        ${s.patient}
                    </div>
                    <div style="font-size:0.7rem; color:#9ca3af; background:#111; padding:4px 8px; border-radius:4px; font-family:monospace; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                        > ${s.progress}
                    </div>`;
              }
              
              sessHtml += `
              <div style="background:#0f1117; padding:12px; border-radius:8px; border-left:3px solid ${color};">
                  <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                      <span style="font-weight:600;font-size:0.8rem;color:${color};">Stage ${s.stage}</span>
                      <span style="font-size:0.7rem;color:var(--muted);">zellij: ${s.session_id} &nbsp;·&nbsp; ⏱ ${s.uptime || '?'}</span>
                  </div>
                  ${statHtml}
              </div>`;
          });
          sessHtml += `</div>`;
          pdiv.innerHTML = sessHtml;
          tc.appendChild(pdiv);
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
    
    card.innerHTML = `
      <div class="card-header">
        <div>
          <div class="count-big" style="color:${sd.color}">${sd.count} <small style="font-size:0.5em;opacity:0.6;">pkl</small> | ${sd.parquet_count} <small style="font-size:0.5em;opacity:0.6;">pq</small></div>
          <div class="count-label">/ ${sd.total} total patients</div>
        </div>
        <span class="stage-badge" style="background:${sd.color}">${s}</span>
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
      <div class="recent-title">Recently Generated Parquets</div>
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

// Countdown timer
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

        else:
            self.send_response(404)
            self.end_headers()


def run(port=8765):
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
