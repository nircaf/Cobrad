"""
Per-electrode grid for the Susp. Epilepsy delta contrasts (panels A: REM-DS,
B: REM-LS in abstract_data.pkl): one figure per contrast, 19 subplots (one
per electrode), each showing that electrode's mean delta HEP +/- SEM with
its own cluster-permutation significant window in red -- the electrode-level
detail behind each panel's single topomap in the abstract figure.

Run: source venv/bin/activate && python3 Paper1/plot_electrode_grid.py
"""
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
OUT_DIR = os.path.join(REPO, "Paper1")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
DATA_PKL = os.path.join(OUT_DIR, "abstract_data.pkl")

COLOR_NS = "#4C72B0"
COLOR_SIG = "#C0392B"
COLOR_QRS = "#BBBBBB"

# Electrode layout approximating the 10-20 montage on a 5x4 grid so
# neighbouring subplots stay roughly neighbouring on the scalp.
GRID_LAYOUT = [
    [None,  "Fp1", None,  "Fp2", None],
    ["F7",  "F3",  "Fz",  "F4",  "F8"],
    ["T3",  "C3",  "Cz",  "C4",  "T4"],
    ["T5",  "P3",  "Pz",  "P4",  "T6"],
    [None,  "O1",  None,  "O2",  None],
]

DISPLAY_TITLE = {"A": "REM − DS", "B": "REM − LS"}


def delta_and_sem(channel_result):
    a, b = channel_result["matrix_a"], channel_result["matrix_b"]
    if channel_result["paired"]:
        delta = a - b
        return np.nanmean(delta, axis=0), np.nanstd(delta, axis=0) / np.sqrt(delta.shape[0])
    mean_a, mean_b = np.nanmean(a, axis=0), np.nanmean(b, axis=0)
    sem_a = np.nanstd(a, axis=0) / np.sqrt(a.shape[0])
    sem_b = np.nanstd(b, axis=0) / np.sqrt(b.shape[0])
    return mean_a - mean_b, np.sqrt(sem_a ** 2 + sem_b ** 2)


def plot_electrode(ax, channel_result, qrs_window, channel_name):
    times = channel_result["times"]
    delta_mean, delta_sem = delta_and_sem(channel_result)

    t0, t1 = qrs_window
    ax.axvspan(t0, t1, color=COLOR_QRS, alpha=0.5, zorder=0, linewidth=0)
    ax.axhline(0, color="#999999", lw=0.6, zorder=1)
    ax.axvline(0, color="black", lw=0.7, linestyle="--", zorder=1)

    ax.fill_between(times, delta_mean - delta_sem, delta_mean + delta_sem,
                     color=COLOR_NS, alpha=0.20, zorder=2, linewidth=0)
    ax.plot(times, delta_mean - delta_sem, color=COLOR_NS, lw=0.4, alpha=0.55, zorder=2)
    ax.plot(times, delta_mean + delta_sem, color=COLOR_NS, lw=0.4, alpha=0.55, zorder=2)
    ax.plot(times, delta_mean, color=COLOR_NS, lw=1.1, zorder=3)

    sig_mask = np.zeros_like(times, dtype=bool)
    for c in channel_result["clusters"]:
        if c["significant"]:
            sig_mask |= (times >= c["start"]) & (times <= c["end"])
    if sig_mask.any():
        idx = np.flatnonzero(sig_mask)
        runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for run in runs:
            if len(run) < 2:
                continue
            ax.plot(times[run], delta_mean[run], color=COLOR_SIG, lw=1.8,
                    zorder=4, solid_capstyle="round")

    is_fdr_sig = channel_result.get("fdr_significant", False)
    title_color = COLOR_SIG if is_fdr_sig else "#333333"
    marker = " *" if is_fdr_sig else ""
    ax.set_title(f"{channel_name}{marker}", fontsize=10, fontweight="bold",
                 color=title_color, pad=2)
    ax.set_xlim(times.min(), times.max())
    ax.tick_params(axis="both", labelsize=6.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_contrast_grid(panel_key, panel, qrs_window):
    channel_results = panel["channel_results"]
    fig, axes = plt.subplots(len(GRID_LAYOUT), len(GRID_LAYOUT[0]),
                              figsize=(14, 12), sharex=True)
    for row, row_channels in enumerate(GRID_LAYOUT):
        for col, channel_name in enumerate(row_channels):
            ax = axes[row][col]
            if channel_name is None or channel_name not in channel_results:
                ax.axis("off")
                continue
            plot_electrode(ax, channel_results[channel_name], qrs_window, channel_name)

    fig.suptitle(
        f"{DISPLAY_TITLE[panel_key]} (Susp. Epilepsy, paired) — per-electrode delta HEP\n"
        f"* = FDR-significant (q < 0.05)",
        fontsize=15, fontweight="bold", y=0.995,
    )
    fig.supxlabel("Time from R-peak (s)", fontsize=11)
    fig.supylabel("Δ HEP amplitude (µV)", fontsize=11)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))

    out_png = os.path.join(FIG_DIR, f"electrode_grid_{panel_key}.png")
    out_pdf = os.path.join(FIG_DIR, f"electrode_grid_{panel_key}.pdf")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


def main():
    with open(DATA_PKL, "rb") as f:
        data = pickle.load(f)
    qrs_window = data["hep_artifact_exclude_s"]
    for key in ["A", "B"]:
        plot_contrast_grid(key, data["panels"][key], qrs_window)


if __name__ == "__main__":
    main()
