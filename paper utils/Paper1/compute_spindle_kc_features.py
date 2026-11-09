"""Per-patient spindle / K-complex features from the saved light-sleep EEG segments.
Spindles: YASA spindles_detect. K-complex proxy: YASA sw_detect restricted to 0.5-1.5 Hz-band
high-amplitude negative-then-positive waves (no dedicated KC detector in YASA).
Usage: python compute_spindle_kc_features.py [max_patients] ; writes spindle_kc_features.csv
"""
import sys, glob, os, pickle, warnings, logging
from multiprocessing import Pool
import numpy as np, pandas as pd, yasa
warnings.filterwarnings("ignore"); logging.getLogger("yasa").setLevel(logging.ERROR)
try:
    from loguru import logger; logger.remove()
except Exception: pass
ROOT = "/storage/pblab_shared_data2/Nir/Cobrad"
D = f"{ROOT}/pickles_sleep_stage/Harvard_Electroencephalography/light_sleep"
OUT = f"{ROOT}/paper utils/Paper1/spindle_kc_features.csv"

EEG10_20 = {"FP1","FP2","F7","F3","FZ","F4","F8","T3","T4","T5","T6","C3","CZ","C4","P3","PZ","P4","O1","O2","T7","T8","P7","P8"}

def feat(args):
    pid, fl = args
    try:
        for f in fl:  # first session with >=10 standard 10-20 EEG channels
            r = pickle.load(open(f, "rb"))
            if sum(c.upper() in EEG10_20 for c in r.ch_names) >= 10: break
        else: return None
        sf = r.info["sfreq"]
        chs = [c for c in r.ch_names if c.upper() in ("C3", "C4")]
        if not chs: return None
        X = r.copy().pick(chs).get_data() * 1e6
        mins = X.shape[1] / sf / 60; row = dict(patient_id=pid, minutes=mins)
        sp = yasa.spindles_detect(X, sf, ch_names=chs, freq_sp=(11, 16), thresh={"rel_pow": 0.2, "corr": 0.65, "rms": 1.5}, verbose=False)
        s = sp.summary() if sp is not None else None
        row["spindle_density"] = 0 if s is None else len(s) / len(chs) / mins
        row["spindle_amp"] = np.nan if s is None else s["Amplitude"].mean()
        # KC proxy: slow, high-amplitude biphasic waves; thresholds adaptive to epoch amplitude (robust SD)
        kcs = 0
        for x in X:
            sd = 1.4826 * np.median(np.abs(x - np.median(x)))
            sw = yasa.sw_detect(x, sf, freq_sw=(0.5, 1.5), dur_neg=(0.3, 1.5), dur_pos=(0.1, 1.0), amp_neg=(2.5 * sd, 1e4),
                                amp_pos=(0.5 * sd, 1e4), amp_ptp=(5 * sd, 1e4), coupling=False, verbose=False)
            if sw is not None: kcs += len(sw.summary())
        row["kc_density"] = kcs / len(chs) / mins
        return row
    except Exception as e:
        return None

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    avg = pickle.load(open(f"{ROOT}/paper utils/Paper1/hep_diagnosis_long_df.pkl", "rb"))["avg_df"]
    pids = sorted(avg.patient_id.unique())
    files = {}
    for f in os.listdir(D):
        if "_light_sleep_" in f and not f.endswith("meta.pkl") and f.startswith("SUB-"):
            files.setdefault(f.split("_SES")[0].split("_light")[0][4:], []).append(f"{D}/{f}")
    jobs = [(p, files[p]) for p in pids if p in files][:n]
    print(len(jobs), "jobs", flush=True)
    with Pool(48) as pool: res = [r for r in pool.imap_unordered(feat, jobs, chunksize=4) if r]
    pd.DataFrame(res).to_csv(OUT, index=False); print(len(res), "done")
