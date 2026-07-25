"""
Generate individuals_cache.pkl and individuals_cache_ica.pkl for COBRAD sleep stage data.

Usage:
  python generate_sleep_stage_cache.py                          # all projects, all stages
  python generate_sleep_stage_cache.py --project Berkeley_data  # one project, all stages
  python generate_sleep_stage_cache.py --stage light_sleep       # all projects, one stage
  python generate_sleep_stage_cache.py --project EDF --stage W  # specific combo
  python generate_sleep_stage_cache.py --no-ica                 # skip ICA cache
"""

import sys
import os
import argparse
import types
import importlib.util
import pickle
import traceback
from datetime import datetime

BASE_PATH = "/storage/pblab_shared_data2/Nir/Cobrad/pickles_sleep_stage"
HEP_MODULE_PATH = "/storage/pblab_shared_data2/Nir/Cobrad/6_hep_group_comparison.py"

VALID_STAGES = ['light_sleep', 'N3', 'R', 'W']


def _make_mock_streamlit():
    """Build a mock streamlit module with no-op UI calls."""

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
        """Transparent pass-through — no caching."""
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
    st.progress = lambda *a, **kw: _NoOp()
    st.empty = lambda *a, **kw: _NoOp()
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

    # Make st itself context-manager-friendly
    st.__enter__ = lambda s: s
    st.__exit__ = lambda s, *a: False

    # Patch submodules that might be accessed
    components = types.ModuleType("streamlit.components")
    components.v1 = _NoOp()
    st.components = components

    return st


def load_hep_module():
    """Import 6_hep_group_comparison with mocked streamlit."""
    # Inject mock BEFORE importing the module
    mock_st = _make_mock_streamlit()
    sys.modules["streamlit"] = mock_st

    # Also mock streamlit submodules that may get imported
    for submod in ["streamlit.components", "streamlit.components.v1"]:
        if submod not in sys.modules:
            sys.modules[submod] = types.ModuleType(submod)

    spec = importlib.util.spec_from_file_location("hep_module", HEP_MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hep_module"] = mod
    spec.loader.exec_module(mod)
    return mod


def discover_projects():
    return sorted([
        d for d in os.listdir(BASE_PATH)
        if os.path.isdir(os.path.join(BASE_PATH, d))
        and not d.startswith('.')
    ])


def discover_stages(project):
    project_path = os.path.join(BASE_PATH, project)
    return sorted([
        d for d in os.listdir(project_path)
        if os.path.isdir(os.path.join(project_path, d))
        and not d.startswith('.')
    ])


def generate_cache(mod, project, stage, apply_ica, label):
    ica_label = "ICA" if apply_ica else "non-ICA"
    print(f"  [{label}] Generating {ica_label} cache...", flush=True)
    try:
        result = mod.get_group_individuals(
            project, stage, BASE_PATH,
            test_run=False,
            recompute_cache=True,
            apply_ica=apply_ica,
        )
        n = len(result) if result else 0
        print(f"  [{label}] {ica_label} cache done — {n} patients", flush=True)
        return True
    except Exception as e:
        print(f"  [{label}] {ica_label} cache FAILED: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate individuals_cache.pkl and individuals_cache_ica.pkl for COBRAD sleep stage data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--project", default=None, help="Project name (e.g. Berkeley_data, EDF). Default: all.")
    parser.add_argument("--stage", default=None,
                        help=f"Sleep stage. Valid values: {VALID_STAGES}. Default: all.")
    parser.add_argument("--no-ica", action="store_true", help="Skip ICA cache generation.")
    args = parser.parse_args()

    print(f"=== COBRAD Sleep Stage Cache Generator ===", flush=True)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Base path: {BASE_PATH}", flush=True)

    # Validate inputs
    all_projects = discover_projects()
    if args.project:
        if args.project not in all_projects:
            print(f"ERROR: Project '{args.project}' not found. Available: {all_projects}", file=sys.stderr)
            sys.exit(1)
        projects = [args.project]
    else:
        projects = all_projects

    print(f"Projects: {projects}", flush=True)

    # Build work list
    work = []
    for proj in projects:
        all_stages = discover_stages(proj)
        if args.stage:
            if args.stage not in all_stages:
                print(f"  WARNING: Stage '{args.stage}' not found in project '{proj}'. Skipping.", file=sys.stderr)
                continue
            stages = [args.stage]
        else:
            stages = all_stages
        for stage in stages:
            work.append((proj, stage))

    print(f"Total combinations: {len(work)}", flush=True)
    print(f"Generate ICA cache: {not args.no_ica}", flush=True)
    print("", flush=True)

    if not work:
        print("Nothing to do.", flush=True)
        sys.exit(0)

    # Load module with mocked streamlit
    print("Loading HEP module...", flush=True)
    try:
        mod = load_hep_module()
        print("HEP module loaded.", flush=True)
    except Exception as e:
        print(f"FATAL: Failed to load HEP module: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # Process each combination
    succeeded = 0
    failed = 0
    skipped = 0

    for i, (proj, stage) in enumerate(work, 1):
        label = f"{proj}/{stage} ({i}/{len(work)})"
        print(f"\n--- {label} ---", flush=True)

        # Check if there are patient files
        group_dir = os.path.join(BASE_PATH, proj, stage)
        patient_files = [
            f for f in os.listdir(group_dir)
            if f.endswith('.pkl') and not f.startswith('individuals_cache')
        ]
        if not patient_files:
            print(f"  [{label}] No patient files found. Skipping.", flush=True)
            skipped += 1
            continue

        print(f"  [{label}] Found {len(patient_files)} patient files.", flush=True)

        ok_plain = generate_cache(mod, proj, stage, apply_ica=False, label=label)
        ok_ica = True
        if not args.no_ica:
            ok_ica = generate_cache(mod, proj, stage, apply_ica=True, label=label)

        if ok_plain and ok_ica:
            succeeded += 1
        else:
            failed += 1

    print(f"\n=== Summary ===", flush=True)
    print(f"Succeeded: {succeeded}", flush=True)
    print(f"Failed:    {failed}", flush=True)
    print(f"Skipped:   {skipped}", flush=True)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
