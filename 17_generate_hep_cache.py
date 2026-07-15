"""
HEP Cache Generator — standalone (no Streamlit).

Pre-builds the per-group/stage `individuals_cache.pkl` (and `individuals_cache_ica.pkl`)
files used by the dashboards, by calling the tested, memory-safe pipeline in
6_hep_group_comparison.py directly.

Key design:
  - No custom pickle-loading of raw EEG here. get_group_individuals() in
    6_hep_group_comparison.py does the real work: it processes one patient
    file at a time in a ProcessPoolExecutor worker (R-peak detection + epoch
    averaging), discards the raw ~50-75MB signal after producing a small
    per-patient result, and writes/reads a small disk cache
    (individuals_cache.pkl / individuals_cache_ica.pkl) inside each
    <group>/<stage>/ directory. There is no single giant combined cache file
    anymore — caching is per group/stage, handled entirely by that function.
  - Workers are capped: max_workers = min(--workers, n_files_in_group_stage,
    cpu_count()). A direct Python invocation defaults to one quarter of the
    CPUs available to this process. The shell launcher divides that same
    quarter-machine budget between its two cache subprocesses.
  - 6_hep_group_comparison.py is imported as a plain module by mocking
    `streamlit` before import (it's a Streamlit dashboard file but has no
    top-level Streamlit calls outside functions), same pattern as
    generate_sleep_stage_cache.py.

Run:
  python 17_generate_hep_cache.py [--groups G1 G2 ...] [--stages S1 S2 ...]
                                   [--workers N] [--force] [--ica] [--test-run]
                                   [--min-eeg-channels N]
"""

from __future__ import annotations

import argparse
import fcntl
import multiprocessing
import os
import sys
import time
import types
import importlib.util
import traceback
from typing import List, Optional

BASE_PATH = "/storage/pblab_shared_data/Nir/Cobrad/pickles_sleep_stage"
HEP_MODULE_PATH = "/storage/pblab_shared_data/Nir/Cobrad/6_hep_group_comparison.py"

STAGE_ORDER = ["W", "light_sleep", "N3", "R"]
NON_PATIENT_CACHE_PREFIXES = ("individuals_cache", "non_eeg_individuals_cache")
DEFAULT_HEP_GROUPS = [
    "The_Human_Sleep_Project",
    "Young_The_Human_Sleep_Project",
    "Harvard_Electroencephalography",
]


# ── Mock streamlit + module loader (same pattern as generate_sleep_stage_cache.py) ──
def _make_mock_streamlit():
    class _ConsoleProgress:
        def __init__(self, value=0, text=None):
            self.last_value = -1.0
            self.last_print = 0.0
            if text:
                print(f"[patient progress] {text}", flush=True)

        def progress(self, value, text=None):
            now = time.monotonic()
            numeric = float(value)
            if numeric >= 1.0 or numeric - self.last_value >= 0.01 or now - self.last_print >= 10:
                width = 30
                filled = min(width, max(0, int(round(numeric * width))))
                bar = "#" * filled + "-" * (width - filled)
                print(
                    f"[patient progress] [{bar}] {numeric:6.1%} {text or ''}",
                    flush=True,
                )
                self.last_value = numeric
                self.last_print = now

        def empty(self):
            pass

    class _ConsoleStatus:
        def __init__(self):
            self.last_print = 0.0

        def text(self, message):
            now = time.monotonic()
            if message.startswith("Starting") or now - self.last_print >= 10:
                print(f"[patient status] {message}", flush=True)
                self.last_print = now

        def empty(self):
            pass

    class _NoOp:
        def __init__(self, *a, **kw):
            pass
        def __call__(self, *a, **kw):
            return _NoOp()
        def progress(self, *a, **kw):
            pass
        def empty(self, *a, **kw):
            pass
        def text(self, *a, **kw):
            pass
        def warning(self, *a, **kw):
            pass
        def error(self, *a, **kw):
            pass
        def info(self, *a, **kw):
            pass
        def __getattr__(self, name):
            return _NoOp()

    def _cache_data(func=None, **kwargs):
        if func is not None:
            return func
        def decorator(f):
            return f
        return decorator

    def _cache_resource(func=None, **kwargs):
        if func is not None:
            return func
        def decorator(f):
            return f
        return decorator

    st = types.ModuleType("streamlit")
    st.cache_data = _cache_data
    st.cache_resource = _cache_resource
    st.session_state = {}
    st.progress = lambda value=0, text=None, **kw: _ConsoleProgress(value, text)
    st.empty = lambda *a, **kw: _ConsoleStatus()
    st.warning = lambda msg, *a, **kw: print(f"[WARN] {msg}", file=sys.stderr)
    st.error = lambda msg, *a, **kw: print(f"[ERROR] {msg}", file=sys.stderr)
    st.info = lambda msg, *a, **kw: print(f"[INFO] {msg}", file=sys.stderr)
    st.write = lambda *a, **kw: None
    st.title = lambda *a, **kw: None
    st.header = lambda *a, **kw: None
    st.subheader = lambda *a, **kw: None
    st.markdown = lambda *a, **kw: None
    st.text = lambda *a, **kw: None
    st.pyplot = lambda *a, **kw: None
    st.sidebar = _NoOp()
    st.columns = lambda *a, **kw: [_NoOp() for _ in range(a[0] if a else 2)]
    st.tabs = lambda labels: [_NoOp() for _ in labels]
    st.expander = lambda *a, **kw: _NoOp()
    st.spinner = lambda *a, **kw: _NoOp()
    st.set_page_config = lambda *a, **kw: None
    st.stop = lambda: None
    st.experimental_rerun = lambda: None
    st.rerun = lambda: None

    st.__enter__ = lambda s: s
    st.__exit__ = lambda s, *a: False

    components = types.ModuleType("streamlit.components")
    components.v1 = _NoOp()
    st.components = components

    return st


def load_hep_module():
    """Import 6_hep_group_comparison.py as a module, with streamlit mocked out."""
    mock_st = _make_mock_streamlit()
    sys.modules["streamlit"] = mock_st
    for submod in ["streamlit.components", "streamlit.components.v1"]:
        if submod not in sys.modules:
            sys.modules[submod] = types.ModuleType(submod)

    spec = importlib.util.spec_from_file_location("hep_module", HEP_MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hep_module"] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Group/stage discovery ────────────────────────────────────────────────────
def list_groups(base_path: str) -> List[str]:
    if not os.path.isdir(base_path):
        return []
    return sorted(d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)))


def list_stages(base_path: str, groups: List[str]) -> List[str]:
    stages: set = set()
    for group in groups:
        gdir = os.path.join(base_path, group)
        if os.path.isdir(gdir):
            stages.update(d for d in os.listdir(gdir) if os.path.isdir(os.path.join(gdir, d)))
    return sorted(stages, key=lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else 99)


def count_patient_files(base_path: str, group: str, stage: str) -> int:
    stage_dir = os.path.join(base_path, group, stage)
    if not os.path.isdir(stage_dir):
        return 0
    return sum(
        1 for f in os.listdir(stage_dir)
        if f.endswith(".pkl") and not f.startswith(NON_PATIENT_CACHE_PREFIXES)
    )


class _StageGenerationLock:
    """Serialize writers for one group/stage across generator subprocesses."""

    def __init__(self, base_path: str, group: str, stage: str, label: str):
        self.path = os.path.join(base_path, group, stage, ".hep_cache_generation.lock")
        self.label = label
        self.handle = None

    def __enter__(self):
        self.handle = open(self.path, "a+")
        started = time.monotonic()
        timeout = int(os.environ.get("HEP_CACHE_LOCK_TIMEOUT_SECONDS", 14400))
        while True:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                elapsed = int(time.monotonic() - started)
                if elapsed >= timeout:
                    self.handle.close()
                    self.handle = None
                    raise TimeoutError(
                        f"Stage cache lock timed out after {elapsed}s: {self.path}"
                    )
                self.handle.seek(0)
                owner = self.handle.read().strip() or "another cache subprocess"
                print(
                    f"[lock wait] {self.label}: {elapsed}s elapsed; owner: {owner}",
                    flush=True,
                )
                time.sleep(10)
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"PID {os.getpid()} · {self.label}")
        self.handle.flush()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


# ── Main generation ───────────────────────────────────────────────────────────
def generate_cache(
    mod,
    groups: List[str],
    stages: List[str],
    n_workers: int,
    force: bool,
    apply_ica: bool,
    test_run: bool,
    min_eeg_channels: Optional[int] = None,
) -> int:
    """Runs get_group_individuals() for every (group, stage) pair. Returns failure count."""
    combos = [(g, s) for g in groups for s in stages]
    print(f"Workers  : {n_workers}", flush=True)
    print(f"Groups   : {groups}", flush=True)
    print(f"Stages   : {stages}", flush=True)
    print(f"ICA      : {apply_ica}", flush=True)
    print(
        f"Cohort   : at least {min_eeg_channels} canonical 10-20 EEG channels"
        if min_eeg_channels else "Cohort   : all patients",
        flush=True,
    )
    print(f"Test run : {test_run} (first 10 files only)" if test_run else "Test run : False", flush=True)
    print(f"Force    : {force}", flush=True)
    print(f"Combos   : {len(combos)}", flush=True)

    failed = 0
    for i, (group, stage) in enumerate(combos, 1):
        label = f"{group}/{stage} ({i}/{len(combos)})"
        completed_width = int(round((i - 1) / len(combos) * 30))
        print(
            f"[overall progress] [{'#' * completed_width}{'-' * (30 - completed_width)}] "
            f"{i - 1}/{len(combos)} complete",
            flush=True,
        )
        n_files = count_patient_files(BASE_PATH, group, stage)
        if n_files == 0:
            print(f"--- {label}: no patient files, skipping ---", flush=True)
            continue

        print(f"--- {label}: {n_files} files found ---", flush=True)
        try:
            print(f"--- {label}: waiting for stage cache lock ---", flush=True)
            cohort_label = f"{label} · {'min EEG ' + str(min_eeg_channels) if min_eeg_channels else 'all patients'}"
            with _StageGenerationLock(BASE_PATH, group, stage, cohort_label):
                print(f"--- {label}: generating cache ---", flush=True)
                individuals = mod.get_group_individuals(
                    group, stage, BASE_PATH,
                    test_run=test_run,
                    recompute_cache=force,
                    apply_ica=apply_ica,
                    parallel_workers=n_workers,
                    min_eeg_channels=min_eeg_channels,
                )
            n = len(individuals) if individuals else 0
            print(f"--- {label}: done, {n} patients returned ---", flush=True)
        except Exception as e:
            print(f"--- {label}: FAILED: {e} ---", file=sys.stderr, flush=True)
            traceback.print_exc()
            failed += 1

    print(
        f"[overall progress] [{'#' * 30}] {len(combos)}/{len(combos)} complete",
        flush=True,
    )

    return failed


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Required for ProcessPoolExecutor on some platforms
    multiprocessing.freeze_support()

    print("Loading HEP module...", flush=True)
    hep_mod = load_hep_module()
    print("HEP module loaded.", flush=True)

    available_workers = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    default_workers = max(1, available_workers // 4)

    parser = argparse.ArgumentParser(description="Pre-build per-group/stage HEP disk caches via get_group_individuals().")
    parser.add_argument("--groups",  nargs="+", default=DEFAULT_HEP_GROUPS)
    parser.add_argument("--stages",  nargs="+", default=None,
                        help="Default: all stages found for selected groups")
    parser.add_argument("--workers", type=int, default=default_workers,
                        help=f"Worker processes, capped at cpu_count (default: {default_workers})")
    parser.add_argument("--force",   action="store_true",
                        help="Ignore existing per-group/stage cache; recompute (recompute_cache=True)")
    parser.add_argument("--ica",     action="store_true",
                        help="Generate the ICA-applied cache (individuals_cache_ica.pkl) instead of the plain one")
    parser.add_argument("--test-run", action="store_true",
                        help="Process only the first ~10 files per group/stage (smoke test)")
    parser.add_argument("--min-eeg-channels", type=int, default=None,
                        help="Generate a separate cache requiring at least N channels from the canonical 10-20 list")
    args = parser.parse_args()

    groups = [g for g in args.groups if os.path.isdir(os.path.join(BASE_PATH, g))]
    if not groups:
        print(f"ERROR: none of the requested groups exist under {BASE_PATH}", file=sys.stderr)
        sys.exit(1)

    stages = args.stages or list_stages(BASE_PATH, groups)

    n_failed = generate_cache(
        hep_mod,
        groups=groups,
        stages=stages,
        n_workers=args.workers,
        force=args.force,
        apply_ica=args.ica,
        test_run=args.test_run,
        min_eeg_channels=args.min_eeg_channels,
    )

    sys.exit(1 if n_failed > 0 else 0)
