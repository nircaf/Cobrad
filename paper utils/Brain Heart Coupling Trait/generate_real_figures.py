#!/usr/bin/env python3
"""Generate empirical paper figures from real_psg_windows.parquet."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
D = pd.read_parquet(HERE / "real_psg_windows.parquet")
S = json.loads((HERE / "real_psg_stats.json").read_text())
D = D[(D.stage != "Other") & (D.stage_purity >= .8)].copy()
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11})
NAVY, BLUE, TEAL, GOLD, RED, GREY = "#17233c", "#4472c4", "#2a9d8f", "#e9c46a", "#d95f59", "#667085"
STAGE_COLORS = {"Wake":"#8d99ae", "N1":"#a8dadc", "N2":"#457b9d", "N3":"#1d3557", "REM":"#e76f51"}
EEG = ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1"]

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

# Figure 2: actual trajectories and raw variance partition.
first_session = D.sort_values("start_date").groupby("subject").head(1)[["subject", "session"]]
keys = set(map(tuple, first_session.values))
night = D[D[["subject", "session"]].apply(tuple, axis=1).isin(keys)]
subject_means = night.groupby("subject").coupling.mean().sort_values()
idx = np.linspace(0, len(subject_means)-1, 8).astype(int)
chosen = list(subject_means.index[idx])
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.55), gridspec_kw={"width_ratios":[1.65,1]})
for offset, subject in enumerate(chosen):
    x = night[night.subject == subject].sort_values("window")
    axes[0].plot(x.hour, x.coupling + offset*.12, color=BLUE, alpha=.45, lw=.8)
    for stage, g in x.groupby("stage"):
        axes[0].scatter(g.hour, g.coupling + offset*.12, s=9, color=STAGE_COLORS[stage], zorder=3)
axes[0].set(xlabel="Hours from recording start", ylabel="Coupling (vertically offset by subject)", title="Real 5-minute trajectories across eight PSGs")
axes[0].spines[["top","right"]].set_visible(False)
handles=[plt.Line2D([],[],marker='o',ls='',color=c,label=s,markersize=5) for s,c in STAGE_COLORS.items()]
# ponytail: headroom so the inline stage legend clears the topmost trajectory
_lo,_hi = axes[0].get_ylim(); axes[0].set_ylim(_lo, _hi + .10*(_hi-_lo))
axes[0].legend(handles=handles,ncol=5,frameon=False,fontsize=7,loc="upper center")
u=S["unadjusted"]
axes[1].bar([0,1],[u["between"],u["within"]],color=[BLUE,GOLD],width=.66)
axes[1].set_xticks([0,1],["Between\nsubjects","Within\nsubject"]); axes[1].set_ylabel("Variance (coherence² units)")
axes[1].set_title(f"Within-night decomposition\nICC={u['icc']:.3f} ({u['ci'][0]:.3f}–{u['ci'][1]:.3f})")
axes[1].spines[["top","right"]].set_visible(False)
fig.suptitle("Most 5-minute variation occurred within people",weight="bold",color=NAVY,y=1.01)
fig.tight_layout(); save(fig,"figure2_icc_estimand")

# Figure 3: overall adjusted/unadjusted and stage-specific ICCs.
fig, axes = plt.subplots(1,2,figsize=(10.5,3.5),gridspec_kw={"width_ratios":[.8,1.5]})
for i,key in enumerate(["unadjusted","adjusted"]):
    r=S[key]; axes[0].errorbar(r["icc"],i,xerr=[[r["icc"]-r["ci"][0]],[r["ci"][1]-r["icc"]]],fmt='o',ms=8,color=[BLUE,TEAL][i],capsize=4)
axes[0].set_yticks([0,1],["Unadjusted","Stage/time adjusted"]); axes[0].set_xlim(0,.62); axes[0].set_xlabel("ICC (95% bootstrap CI)")
axes[0].axvline(0,color="#bbb",lw=.8); axes[0].spines[["top","right"]].set_visible(False); axes[0].set_title("Overall")
order=[x for x in ["Wake","N1","N2","N3","REM"] if x in S["stages"]]
for i,stage in enumerate(order):
    r=S["stages"][stage]; axes[1].errorbar(r["icc"],i,xerr=[[r["icc"]-r["ci"][0]],[r["ci"][1]-r["icc"]]],fmt='o',ms=8,color=STAGE_COLORS[stage],capsize=4)
    axes[1].text(.61,i,f"n={r['n_subjects']}, windows={r['n_windows']:,}",va="center",fontsize=7,color=GREY)
axes[1].set_yticks(range(len(order)),order); axes[1].set_xlim(0,.82); axes[1].set_xlabel("ICC (95% bootstrap CI)")
axes[1].spines[["top","right"]].set_visible(False); axes[1].set_title("Sleep-stage-specific")
fig.suptitle("Within-night reliability depended strongly on sleep stage",weight="bold",color=NAVY,y=1.02)
fig.tight_layout(); save(fig,"figure3_state_adjustment")

# Visit means and dates.
V=D.groupby(["subject","session"],as_index=False).agg(coupling=("coupling","mean"),start_date=("start_date","first"))
V=V.sort_values(["subject","start_date"]).groupby("subject").head(2)
V["visit"]=V.groupby("subject").cumcount()
W=V.pivot(index="subject",columns="visit",values="coupling").dropna()
a,b=W[0].values,W[1].values

# Figure 4: repeat scalar agreement.
fig,axes=plt.subplots(1,2,figsize=(10.5,3.5))
axes[0].scatter(a,b,s=24,color=BLUE,alpha=.72,edgecolor="white",linewidth=.3)
lo=min(a.min(),b.min()); hi=max(a.max(),b.max()); axes[0].plot([lo,hi],[lo,hi],ls="--",color=GREY,lw=1)
coef=np.polyfit(a,b,1); axes[0].plot([lo,hi],np.polyval(coef,[lo,hi]),color=RED,lw=1.8)
axes[0].set(xlabel="First PSG mean coupling",ylabel="Repeat PSG mean coupling",title=f"n={len(W)}; r={S['repeat']['pearson_r']:.3f}; ICC(A,1)={S['repeat']['icc_a1']:.3f}")
mean=(a+b)/2; diff=b-a; md=diff.mean(); sd=diff.std(ddof=1)
axes[1].scatter(mean,diff,s=24,color=TEAL,alpha=.72,edgecolor="white",linewidth=.3)
axes[1].axhline(md,color=NAVY,lw=1.5); axes[1].axhline(md+1.96*sd,color=RED,ls="--"); axes[1].axhline(md-1.96*sd,color=RED,ls="--")
axes[1].set(xlabel="Mean of two PSGs",ylabel="Repeat − first PSG",title="Bland–Altman agreement")
for ax in axes: ax.spines[["top","right"]].set_visible(False)
fig.suptitle(f"The aggregated nightly coupling level returned after a median {S['repeat']['median_interval_years']:.2f} years",weight="bold",color=NAVY,y=1.02)
fig.tight_layout();save(fig,"figure4_reliability_design")

# Figure 5: actual fingerprint similarities and identification.
cols=[f"coupling_{x}" for x in EEG]
FP=D.groupby(["subject","session","stage"])[cols].mean().unstack("stage").sort_index(axis=1)
first=[];second=[];subjects=[]
for s,g in FP.groupby(level=0):
    if len(g)>=2:
        x,y=g.iloc[0].values.astype(float),g.iloc[1].values.astype(float); good=np.isfinite(x)&np.isfinite(y)
        if good.sum()>=12: first.append(x);second.append(y);subjects.append(s)
sim=np.full((len(subjects),len(subjects)),np.nan)
for i,x in enumerate(second):
    for j,y in enumerate(first):
        good=np.isfinite(x)&np.isfinite(y)
        if good.sum()>=12: sim[i,j]=np.corrcoef(x[good],y[good])[0,1]
within=np.diag(sim); between=sim[~np.eye(len(sim),dtype=bool)]
fig,axes=plt.subplots(1,2,figsize=(10.5,3.6),gridspec_kw={"width_ratios":[1.2,1]})
axes[0].hist(between,bins=30,density=True,alpha=.55,color=GREY,label=f"Different people (median {np.nanmedian(between):.2f})")
axes[0].hist(within,bins=18,density=True,alpha=.65,color=TEAL,label=f"Same person (median {np.nanmedian(within):.2f})")
axes[0].set(xlabel="Fingerprint correlation",ylabel="Density",title="Same-person fingerprints were only modestly closer");axes[0].legend(frameon=False,fontsize=8)
order=np.argsort(np.nanmax(sim,axis=1))[::-1][:30]; submat=sim[np.ix_(order,order)]
im=axes[1].imshow(submat,cmap="magma",vmin=-.5,vmax=1,aspect="auto")
axes[1].set(xlabel="First PSG",ylabel="Repeat PSG",title=f"Top-1 identification {S['repeat']['identification_accuracy']*100:.1f}%\n(chance {100/len(subjects):.1f}%)")
axes[1].set_xticks([]);axes[1].set_yticks([]);fig.colorbar(im,ax=axes[1],fraction=.046,pad=.04,label="Correlation")
fig.suptitle("The multichannel × stage fingerprint showed weak individual specificity",weight="bold",color=NAVY,y=1.02)
fig.tight_layout();save(fig,"figure5_repeat_fingerprint")

print("Generated empirical figures 2–5")
