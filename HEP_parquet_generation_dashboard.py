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
import threading
import time
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUETS_HEP_DIR = os.path.join(BASE_DIR, "parquets_HEP")
EDF_ROOT = os.path.join(BASE_DIR, "EDF_Format", "Berkeley_data")

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
    """Find all unique patients in the EDF directory."""
    patient_ids = set()
    for root, dirs, files in os.walk(edf_root):
        for file in files:
            if not file.lower().endswith(".edf"):
                continue
            m = re.search(r'(\d{4}-\d{3})', file)
            if m:
                patient_ids.add(m.group(1))
            else:
                patient_ids.add(file.split('.')[0])
    return sorted(list(patient_ids))

def count_total_patients(edf_root):
    """Count unique patient IDs in the EDF directory (one entry per folder/file)."""
    return len(get_all_patients(edf_root))

def parse_logs():
    """Parse the log file to get the latest status of each patient per stage."""
    log_file = os.path.join(BASE_DIR, "logs", "hep_parquet_generation.log")
    
    status_dict = {s: {} for s in STAGES}
    if not os.path.exists(log_file):
        return status_dict
        
    # Match: 2026-02-28 23:59:59,999 - ... - INFO - [N1] Patient 1234-567: started processing
    # Group 1: Timestamp
    # Group 2: Level (INFO|ERROR|WARNING)
    # Group 3: Stage (e.g. N1)
    # Group 4: Patient ID
    # Group 5: Message text
    
    # regex matches:
    # 2026-02-28 23:03:36,442 - HEP_parquet_generation - INFO - [N1] Patient 0345-010: started processing
    
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
        print(f"Error parsing logs: {e}")
        
    return status_dict


def get_stage_stats(stage):
    """Return (processed_count, list_of_patient_ids) for a given stage."""
    stage_dir = os.path.join(PARQUETS_HEP_DIR, f"Berkeley_data_{stage}")
    if not os.path.isdir(stage_dir):
        return 0, []
    alpha_files = glob.glob(os.path.join(stage_dir, f"*_{stage}_results_alpha.parquet"))
    patient_ids = []
    for fpath in sorted(alpha_files):
        fname = os.path.basename(fpath)
        # filename: {patient_id}_{stage}_results_alpha.parquet
        pid = fname.replace(f"_{stage}_results_alpha.parquet", "")
        patient_ids.append(pid)
    return len(patient_ids), patient_ids


def get_recently_modified(stage, n=5):
    """Return the n most recently modified alpha parquets for a stage."""
    stage_dir = os.path.join(PARQUETS_HEP_DIR, f"Berkeley_data_{stage}")
    if not os.path.isdir(stage_dir):
        return []
    alpha_files = glob.glob(os.path.join(stage_dir, f"*_{stage}_results_alpha.parquet"))
    alpha_files.sort(key=os.path.getmtime, reverse=True)
    result = []
    for fpath in alpha_files[:n]:
        fname = os.path.basename(fpath)
        pid = fname.replace(f"_{stage}_results_alpha.parquet", "")
        mtime = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M:%S")
        result.append({"patient_id": pid, "modified": mtime})
    return result


def collect_data():
    all_patients = get_all_patients(EDF_ROOT)
    total = len(all_patients)
    log_status = parse_logs()
    
    stages_data = {}
    for stage in STAGES:
        count, processed_patients = get_stage_stats(stage)
        recent = get_recently_modified(stage, n=5)
        
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
            "total": total,
            "pct": round(count / total * 100, 1) if total > 0 else 0,
            "patients_meta": patients_meta,
            "running": currently_running,
            "failed": failed_patients,
            "recent": recent,
            "color": STAGE_COLORS[stage],
        }
    return {
        "total": total,
        "stages": stages_data,
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
  <h1>🧠 HEP Processing Dashboard</h1>
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
    const r = await fetch('/api/data');
    lastData = await r.json();
    render(lastData);
  } catch(e){ console.error(e); }
}

function render(data){
  document.getElementById('updated').textContent = 'Updated: ' + data.updated;
  document.getElementById('total-patients').textContent = data.total;

  // Summary extra items
  const sc = document.getElementById('summary-card');
  // remove old stage summaries
  sc.querySelectorAll('.stage-summary').forEach(el=>el.remove());
  STAGES.forEach(s=>{
    const sd = data.stages[s];
    const div = document.createElement('div');
    div.className = 'summary-item stage-summary';
    div.innerHTML = `
      <div class="num" style="background:linear-gradient(135deg,${sd.color},${sd.color}aa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">${sd.count}</div>
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
          <div class="count-big" style="color:${sd.color}">${sd.count}</div>
          <div class="count-label">/ ${sd.total} total patients</div>
        </div>
        <span class="stage-badge" style="background:${sd.color}">${s}</span>
      </div>
      <div class="prog-wrap">
        <div class="prog-bar" style="width:${sd.pct}%;background:${sd.color}"></div>
      </div>
      <div class="pct">${sd.pct}% parquets generated</div>
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
    btn.textContent = `${s} (${data.stages[s].count})`;
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

fetchData();
</script>
</body>
</html>
"""


# ── HTTP Server ─────────────────────────────────────────────────────────────────
class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence default access log

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())

        elif self.path == "/api/data":
            data = collect_data()
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
