#!/usr/bin/env python3
"""Generate the five prespecified figures for the registered report.

Figures 2--5 use simulated values solely to explain estimands and decision rules;
they are not empirical results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11})
NAVY, BLUE, TEAL, GOLD, RED, GREY = "#17233c", "#4472c4", "#2a9d8f", "#e9c46a", "#d95f59", "#667085"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, xy, wh, text, color=BLUE, fs=8):  # ponytail: 9pt overflowed the box width
    p = FancyBboxPatch(xy, *wh, boxstyle="round,pad=0.025,rounding_size=0.025",
                       facecolor=color, edgecolor="white", linewidth=1.5)
    ax.add_patch(p)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center",
            color="white", weight="bold", fontsize=fs)


# Figure 1: analysis pipeline
fig, ax = plt.subplots(figsize=(10.5, 3.7)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
items = [
    (0.02, "Synchronized PSG\nEEG + ECG + stages", NAVY),
    (0.22, "Non-overlapping\n5-minute windows", BLUE),
    (0.42, "Coupling per window\n+ quality controls", TEAL),
    (0.62, "Variance components\nsubject + within-person", GOLD),
    (0.82, "Repeat PSG\nmonths / years later", RED),
]
for x, label, c in items: box(ax, (x, .55), (.16, .25), label, c)
for x in [.18, .38, .58, .78]:
    ax.add_patch(FancyArrowPatch((x, .675), (x + .035, .675), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.6, color=GREY))
ax.text(.5, .94, "Two-timescale test of a brain–heart coupling fingerprint", ha="center", weight="bold", fontsize=14, color=NAVY)
ax.text(.70, .36, r"Within night:  $y_{ij}=\beta_0+u_i+\epsilon_{ij}$", ha="center", fontsize=11, color=NAVY)
ax.text(.70, .22, r"$ICC=\sigma^2_{subject}/(\sigma^2_{subject}+\sigma^2_{within})$", ha="center", fontsize=11, color=NAVY)
ax.text(.98, .06, "Long term: absolute agreement + fingerprint identification", ha="right", fontsize=9, color=RED)
save(fig, "figure1_pipeline")


# Figure 2: simulated trajectories and variance decomposition
rng = np.random.default_rng(20260820)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
for ax, sb, sw, title in zip(axes[:2], [1.2, .35], [.35, 1.2], ["Trait-dominant", "State/noise-dominant"]):
    for s in range(8):
        y = rng.normal(0, sb) + rng.normal(0, sw, 60)
        ax.plot(np.arange(60) * 5 / 60, y, alpha=.72, lw=1)
    icc = sb**2 / (sb**2 + sw**2)
    ax.set_title(f"{title}\nICC = {icc:.2f}")
    ax.set_xlabel("Hours from lights-off"); ax.set_ylabel("Coupling (standardized)")
    ax.axhline(0, color="#cccccc", lw=.8); ax.spines[["top", "right"]].set_visible(False)
v = np.array([[1.2**2, .35**2], [.35**2, 1.2**2]])
axes[2].bar([-.18, .82], v[:, 0], width=.36, color=BLUE, label=r"Between: $\sigma^2_{subject}$")
axes[2].bar([.18, 1.18], v[:, 1], width=.36, color=GOLD, label=r"Within: $\sigma^2_{within}$")
axes[2].set_xticks([0, 1], ["Trait-\ndominant", "State/noise-\ndominant"]); axes[2].set_ylabel("Variance")
axes[2].set_title("Same total variance,\ndifferent biological meaning"); axes[2].legend(frameon=False, fontsize=8)
axes[2].spines[["top", "right"]].set_visible(False)
fig.suptitle("Illustration of the primary estimand (simulated data; not results)", weight="bold", color=NAVY, y=1.03)
fig.tight_layout(); save(fig, "figure2_icc_estimand")


# Figure 3: stage confounding and adjustment
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
stages = ["Wake", "N1", "N2", "N3", "REM"]
colors = ["#8d99ae", "#a8dadc", "#457b9d", "#1d3557", "#e76f51"]
for i, (stage, c) in enumerate(zip(stages, colors)):
    x = np.linspace(i * 1.25, i * 1.25 + 1, 12)
    axes[0].scatter(x, rng.normal(i * .28, .16, len(x)), s=18, color=c, label=stage)
axes[0].set_xlabel("Night sequence (schematic)"); axes[0].set_ylabel("Coupling")
axes[0].set_title("Sleep state creates within-person shifts"); axes[0].legend(ncol=3, frameon=False, fontsize=7)
axes[0].spines[["top", "right"]].set_visible(False)
axes[1].axis("off")
box(axes[1], (.05, .63), (.25, .18), "Subject\ntrait", BLUE)
box(axes[1], (.38, .63), (.25, .18), "Sleep stage +\ntime of night", TEAL)
box(axes[1], (.71, .63), (.25, .18), "Signal quality +\ncardiac field", RED)
box(axes[1], (.36, .20), (.30, .18), "Observed 5-minute\ncoupling", NAVY)
for x in [.175, .505, .835]:
    axes[1].add_patch(FancyArrowPatch((x, .61), (.51, .40), arrowstyle="-|>", mutation_scale=12, color=GREY))
axes[1].text(.5, .04, "Report unadjusted and adjusted variance components", ha="center", color=NAVY, weight="bold")
fig.suptitle("State and artifact must not masquerade as identity", weight="bold", color=NAVY, y=1.02)
fig.tight_layout(); save(fig, "figure3_state_adjustment")


# Figure 4: reliability gained by aggregation
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
k = np.arange(1, 25)
for rho, c in zip([.15, .35, .60, .80], ["#9b5de5", BLUE, TEAL, RED]):
    rel = k * rho / (1 + (k - 1) * rho)
    axes[0].plot(k, rel, lw=2, color=c, label=f"single-window ICC={rho:.2f}")
axes[0].set(xlabel="Number of 5-minute windows averaged", ylabel="Expected reliability of the mean", ylim=(0, 1.02))
axes[0].axhline(.75, color=GREY, ls="--", lw=1); axes[0].legend(frameon=False, fontsize=7)
axes[0].spines[["top", "right"]].set_visible(False); axes[0].set_title("Aggregation can rescue a noisy window metric")
intervals = ["Same\nnight", "<6 mo", "6–24 mo", ">24 mo"]
vals = [.82, .72, .60, .46]
axes[1].plot(range(4), vals, marker="o", lw=2.5, color=RED)
axes[1].fill_between(range(4), np.array(vals)-.10, np.array(vals)+.10, alpha=.16, color=RED)
axes[1].set_xticks(range(4), intervals); axes[1].set_ylim(0, 1); axes[1].set_ylabel("Absolute-agreement ICC")
axes[1].set_title("Prespecified test for long-term decay"); axes[1].spines[["top", "right"]].set_visible(False)
fig.suptitle("Reliability depends on both aggregation and elapsed time (theoretical illustration)", weight="bold", color=NAVY, y=1.02)
fig.tight_layout(); save(fig, "figure4_reliability_design")


# Figure 5: fingerprints and inference matrix
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
n = 14
mat = rng.normal(.22, .08, (n, n)); np.fill_diagonal(mat, rng.normal(.74, .06, n)); mat = np.clip(mat, 0, 1)
im = axes[0].imshow(mat, cmap="magma", vmin=0, vmax=1, aspect="auto")
axes[0].set(xlabel="Baseline subject", ylabel="Repeat PSG subject", title="Fingerprint similarity matrix\n(simulated; true matches on diagonal)")
fig.colorbar(im, ax=axes[0], fraction=.046, pad=.04, label="Similarity")
axes[1].set_xlim(0, 2); axes[1].set_ylim(0, 2); axes[1].set_xticks([.5, 1.5], ["Low", "High"])
axes[1].set_yticks([.5, 1.5], ["Low", "High"]); axes[1].set_xlabel("Between-night reliability"); axes[1].set_ylabel("Within-night ICC")
labels = {(0,1):("Session signature",GOLD),(1,1):("Persistent trait",TEAL),(0,0):("Unreliable metric",RED),(1,0):("Stable only after\naggregation",BLUE)}
for (x,y),(lab,c) in labels.items():
    axes[1].add_patch(Rectangle((x,y),1,1,facecolor=c,alpha=.82,edgecolor="white",lw=2))
    axes[1].text(x+.5,y+.5,lab,ha="center",va="center",color="white",weight="bold")
axes[1].set_title("Decision matrix"); axes[1].tick_params(length=0)
fig.suptitle("The fingerprint must return in an independently acquired PSG", weight="bold", color=NAVY, y=1.01)
fig.tight_layout(); save(fig, "figure5_repeat_fingerprint")

print(f"Wrote figures to {OUT}")
