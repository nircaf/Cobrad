#!/usr/bin/env python3
"""Learned (deep) CFA removers vs CFAR, on the same held-out split as benchmark_ica_cfar.py.

A network sees only the 6 EEG channels and predicts their cardiac component; clean = EEG - prediction.
Training target = ECG-explained part of the EEG (EEG - ecg_regress(EEG)), so the ECG is used only to
build targets on the training half -- at test time the cleaners are ECG-free, like CFAR.
Architectures (each trained on raw EEG and, as "cfar+X", on CFAR-cleaned EEG):
  cnn    dilated residual temporal CNN (TCN)
  tf     patch transformer (100 ms tokens)
  ae     convolutional autoencoder with a bottleneck
  aetf   autoencoder whose latent sequence is mixed by a transformer (all three together)
Extra test pipelines: "X_cfar" = CFAR applied after X, "ens" = mean prediction of the raw-input models.
Scores come from benchmark_ica_cfar.run (CFA R^2, CFA power, synthetic HEP retained, PSD change).

Run: venv/bin/python "paper utils/Paper CFA/dl_cfar.py" [--arch cnn,tf] [--steps N]
"""
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import resample_poly

import benchmark_ica_cfar as bm
import make_cleaning_demo as cd

FS, WIN, BATCH, LR = 200, 800, 256, 1e-3  # nets run at 200 Hz on 4-s windows
CACHE = os.path.join(cd.HERE, "dl_cfar_cache")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
C = len(cd.SITES)


# ---------- data ----------
def records():
    """Same sample, load order and train/test halves as benchmark_ica_cfar.main()."""
    f = os.path.join(CACHE, "records.pkl")
    if os.path.exists(f):
        with open(f, "rb") as fh:
            return pickle.load(fh)
    cfa = pd.read_parquet(os.path.join(cd.HERE, "cfa_combined.parquet"),
                          columns=["patient_id", "edf_path", "window_start_s"]).drop_duplicates("patient_id")
    cfa = cfa[cfa.patient_id.str.startswith("I0") & (cfa.patient_id != cd.EXAMPLE)].sample(bm.N_LOAD, random_state=cd.SEED)
    with ProcessPoolExecutor(cd.N_WORKERS) as ex:
        recs = [r for r in ex.map(cd.load, [(r.patient_id, r.edf_path, float(r.window_start_s))
                                            for r in cfa.itertuples()], chunksize=4) if r]
    os.makedirs(CACHE, exist_ok=True)
    with open(f, "wb") as fh:
        pickle.dump(recs, fh, protocol=4)
    return recs


def rs(x, sf_from, sf_to, n=None):
    q = Fraction(int(sf_to), int(sf_from))
    y = resample_poly(x, q.numerator, q.denominator, axis=-1)
    if n is not None:
        y = y[..., :n] if y.shape[-1] >= n else np.pad(y, [(0, 0)] * (y.ndim - 1) + [(0, n - y.shape[-1])])
    return y


def scale_of(x):
    return float(np.median(np.abs(x)) / 0.6745) + 1e-6  # robust EEG scale, ECG-free


def train_tensors(recs, u=None):
    """(input, target) at FS, per-recording scaled. u given -> input is CFAR-cleaned EEG."""
    X, Y = [], []
    for pid, raw, ecg, peaks, sf in recs:
        x = raw if u is None else cd.project_out(raw, u)
        art = x - bm.ecg_regress(x, ecg, sf)
        s = scale_of(x)
        X.append(rs(x / s, sf, FS).astype(np.float32))
        Y.append(rs(art / s, sf, FS).astype(np.float32))
    n = min(a.shape[1] for a in X)
    return torch.tensor(np.stack([a[:, :n] for a in X])), torch.tensor(np.stack([a[:, :n] for a in Y]))


# ---------- models (in/out: batch x C x WIN) ----------
class TCN(nn.Module):
    def __init__(s, h=64, n=8):
        super().__init__()
        s.inp = nn.Conv1d(C, h, 7, padding=3)
        s.blocks = nn.ModuleList(nn.Sequential(nn.GELU(), nn.Conv1d(h, h, 3, padding=2 ** i, dilation=2 ** i),
                                               nn.GELU(), nn.Conv1d(h, h, 1)) for i in range(n))
        s.out = nn.Conv1d(h, C, 1)

    def forward(s, x):
        z = s.inp(x)
        for b in s.blocks:
            z = z + b(z)
        return s.out(z)


class PatchTF(nn.Module):
    def __init__(s, p=20, d=128, L=4):
        super().__init__()
        s.p = p
        s.emb = nn.Linear(C * p, d)
        s.pos = nn.Parameter(torch.zeros(1, WIN // p, d))
        s.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True), L)
        s.out = nn.Linear(d, C * p)

    def forward(s, x):
        B = x.shape[0]
        t = x.reshape(B, C, -1, s.p).permute(0, 2, 1, 3).reshape(B, -1, C * s.p)
        t = s.out(s.tf(s.emb(t) + s.pos))
        return t.reshape(B, -1, C, s.p).permute(0, 2, 1, 3).reshape(B, C, -1)


def enc(z=16):  # 6 x 800 -> z x 100
    return nn.Sequential(nn.Conv1d(C, 32, 7, 2, 3), nn.GELU(), nn.Conv1d(32, 64, 7, 2, 3), nn.GELU(),
                         nn.Conv1d(64, 128, 7, 2, 3), nn.GELU(), nn.Conv1d(128, z, 1))


def dec(z=16):
    return nn.Sequential(nn.Conv1d(z, 128, 1), nn.GELU(), nn.ConvTranspose1d(128, 64, 8, 2, 3), nn.GELU(),
                         nn.ConvTranspose1d(64, 32, 8, 2, 3), nn.GELU(), nn.ConvTranspose1d(32, C, 8, 2, 3))


class AE(nn.Module):
    def __init__(s, z=16):
        super().__init__()
        s.enc, s.dec = enc(z), dec(z)

    def forward(s, x):
        return s.dec(s.enc(x))


class AETF(nn.Module):
    def __init__(s, z=128, L=3):
        super().__init__()
        s.enc, s.dec = enc(z), dec(z)
        s.pos = nn.Parameter(torch.zeros(1, WIN // 8, z))
        s.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(z, 4, 4 * z, 0.1, batch_first=True, norm_first=True), L)

    def forward(s, x):
        h = s.enc(x).transpose(1, 2)
        return s.dec(s.tf(h + s.pos).transpose(1, 2))


ARCHS = {"cnn": TCN, "tf": PatchTF, "ae": AE, "aetf": AETF}


# ---------- training / inference ----------
def windows(X, Y, n):
    r = torch.randint(0, X.shape[0], (n,))
    t = torch.randint(0, X.shape[2] - WIN, (n,))
    idx = t[:, None] + torch.arange(WIN)
    g = lambda A: A[r[:, None], :, idx].transpose(1, 2)  # noqa: E731
    return g(X), g(Y)


def fit(arch, X, Y, steps):
    torch.manual_seed(cd.SEED)
    nv = max(1, X.shape[0] // 10)  # last 10% of training patients for model selection
    Xt, Yt, Xv, Yv = X[nv:].to(DEV), Y[nv:].to(DEV), X[:nv].to(DEV), Y[:nv].to(DEV)
    xv, yv = windows(Xv, Yv, 2048)
    m = ARCHS[arch]().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps)
    best, state = np.inf, None
    for it in range(steps):
        m.train()
        xb, yb = windows(Xt, Yt, BATCH)
        loss = nn.functional.mse_loss(m(xb), yb)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
        if (it + 1) % 500 == 0:
            m.eval()
            with torch.no_grad():
                v = float(sum(nn.functional.mse_loss(m(xv[i:i + 512]), yv[i:i + 512], reduction="sum")
                              for i in range(0, len(xv), 512)) / yv.numel())
            print(f"  {arch} step {it + 1}: train {loss.item():.4f}  val {v:.4f}  "
                  f"(val R2 of artifact {1 - v / float(yv.var()):.3f})", flush=True)
            if v < best:
                best, state = v, {k: t.clone() for k, t in m.state_dict().items()}
    m.load_state_dict(state)
    return m.eval()


@torch.no_grad()
def predict(m, x, sf):
    """Cardiac component of x (C x n, native sf): 50%-overlap Hann overlap-add of 4-s windows."""
    s, n = scale_of(x), x.shape[1]
    z = rs(x / s, sf, FS)
    N, hop = z.shape[1], WIN // 2
    pad = (-(N - WIN) % hop) + hop
    zp = np.pad(z, ((0, 0), (hop, pad)))
    starts = range(0, zp.shape[1] - WIN + 1, hop)
    w = torch.hann_window(WIN, periodic=True, device=DEV)
    xb = torch.tensor(np.stack([zp[:, a:a + WIN] for a in starts]), dtype=torch.float32, device=DEV)
    pb = torch.cat([m(xb[i:i + 512]) for i in range(0, len(xb), 512)]) * w
    out = torch.zeros(C, zp.shape[1], device=DEV)
    for k, a in enumerate(starts):
        out[:, a:a + WIN] += pb[k]
    return rs(out[:, hop:hop + N].cpu().numpy().astype(np.float64), FS, sf, n) * s


def main(archs, steps):
    recs = records()
    half = len(recs) // 2
    train, test = recs[:half], recs[half:]
    u = cd.mean_pattern([cd.cfa_pattern(r[1], r[3], r[4]) for r in train])
    print(f"n train {len(train)}, n test {len(test)}; u_raw {np.round(u, 2)}", flush=True)
    models = {}
    for tag, uu in [("", None), ("cfar+", u)]:
        X, Y = train_tensors(train, uu)
        print(f"{tag or 'raw'} input: target/input variance {float(Y.var() / X.var()):.3f}", flush=True)
        for a in archs:
            models[tag + a] = fit(a, X, Y, steps)
            torch.save(models[tag + a].state_dict(), os.path.join(CACHE, f"{tag}{a}.pt"))

    P = cd.project_out
    raw_models = [models[a] for a in archs]

    def pipes():
        d = {}
        for a in archs:
            d[a] = lambda x, sf, m=models[a]: x - predict(m, x, sf)
            d[a + "_cfar"] = lambda x, sf, m=models[a]: P(x - predict(m, x, sf), u)
            d["cfar+" + a] = lambda x, sf, m=models["cfar+" + a]: P(x, u) - predict(m, P(x, u), sf)
        d["ens"] = lambda x, sf: x - np.mean([predict(m, x, sf) for m in raw_models], 0)
        d["ens_cfar"] = lambda x, sf: P(d["ens"](x, sf), u)
        return d

    rows = []
    for i, rec in enumerate(test):
        sf = rec[4]
        extra = {k: (lambda x, f=f: f(x, sf)) for k, f in pipes().items()}
        rows.append(bm.run((rec, u, None, ["raw", "cfar", "own"]), extra=extra))
        if (i + 1) % 50 == 0:
            print(f"  scored {i + 1}/{len(test)}", flush=True)
    res = pd.DataFrame(rows)
    old = pd.read_parquet(os.path.join(cd.HERE, "benchmark_ica_cfar.parquet"), columns=["patient_id"])
    if set(res.patient_id) != set(old.patient_id):
        print("WARNING: test split differs from benchmark_ica_cfar.parquet")
    res.to_parquet(os.path.join(cd.HERE, "dl_cfar.parquet"))
    names = [c[3:] for c in res.columns if c.startswith("r2_")]
    print(f"n test = {len(res)}")
    print(pd.DataFrame({
        "mean_r2": [res[f"r2_{n}"].mean() for n in names],
        "pct_pow_vs_raw": [(res[f"pow_{n}"] / res.pow_raw).median() * 100 for n in names],
        "hep_retained": [res[f"hep_{n}"].median() for n in names],
        "psd_db": [res[f"psd_{n}"].median() for n in names],
    }, index=names).round(3).sort_values("pct_pow_vs_raw").to_string())


if __name__ == "__main__":
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d  # noqa: E731
    main(arg("--arch", ",".join(ARCHS)).split(","), int(arg("--steps", 6000)))
