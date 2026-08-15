# -*- coding: utf-8 -*-
"""
Stage 12 — MANIPULATION CHECK: did participants actually perform motor imagery?

Every result in this project is conditional on the paradigm having worked.
Nothing so far establishes that. Protocol *delivery* is verified (marker
structure and timing are identical across all 13 files), but delivery is not
performance.

Why the ERD figure in 01_diagnose.py does NOT answer this
---------------------------------------------------------
There, ERD is computed against the fixation screen. All three conditions —
including Rest, in which participants were instructed to imagine nothing —
show -1.3 to -2.9 dB. A power change present in the no-imagery condition
cannot be evidence of imagery. Fixation shows a cross; the task screen is
blank white. That contrast is visual/attentional, and it swamps everything.

The correct baseline is already in the design
---------------------------------------------
The Rest condition is visually IDENTICAL to the MI conditions: same 5 s, same
blank white screen, same onset beep, same preceding 2 s cue. The only thing
removed is the imagery. So

        ERD = 10*log10( P(Stand or Walk) / P(Rest) )

is matched for every non-motor factor the fixation baseline confounds. This
is the manipulation check, and the experiment was designed to support it.

Prediction if the paradigm worked: NEGATIVE values (desynchronisation) in mu
(8-13 Hz) and/or beta (13-30 Hz), maximal over central/centro-parietal
electrodes, since lower-limb imagery recruits the medial foot representation.

Output: results/manipulation_check.csv, fig_manipulation_check.png
"""

from __future__ import annotations

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import ttest_1samp, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import steam_io as sio        # noqa: E402

R_DIR = sio.RESULTS_DIR
CH = sio.EEG_CHANNELS
CENTRAL = ["C3", "C4", "Cz", "CP3", "CP4", "CPz"]
BANDS = {"mu 8-13": (8, 13), "beta 13-30": (13, 30),
         "low beta 13-20": (13, 20)}
INK, GRID = "#1f2328", "#e5e7eb"
SER = {"Stand": "#1f6feb", "Walk": "#bc4c00"}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#6b7280", "ytick.color": "#6b7280",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.dpi": 130,
})


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def subject_erd(date, subj):
    """dB power of Stand and Walk relative to the Rest condition, per ch x band."""
    ep = sio.get_epochs(date, subj, l_freq=1.0, h_freq=45.0,
                        tmin=0.5, tmax=4.5, ref="none")
    f, P = welch(ep["X"], fs=ep["fs"], nperseg=int(ep["fs"]), axis=-1)
    out = {}
    rest = ep["y"] == sio.CLASS_NAMES.index("Rest")
    for cls in ("Stand", "Walk"):
        m = ep["y"] == sio.CLASS_NAMES.index(cls)
        for bname, (lo, hi) in BANDS.items():
            sel = (f >= lo) & (f <= hi)
            # geometric mean across trials (average in log domain): robust to
            # the single high-amplitude trial that dominates a linear mean
            pc = np.log10(P[m][:, :, sel].mean(-1) + 1e-30).mean(0)
            pr = np.log10(P[rest][:, :, sel].mean(-1) + 1e-30).mean(0)
            out[(cls, bname)] = 10.0 * (pc - pr)         # (n_ch,) in dB
    return out


def main():
    subs = sio.list_subjects()
    rows = []
    for date, subj in subs:
        erd = subject_erd(date, subj)
        for (cls, band), v in erd.items():
            for ci, ch in enumerate(CH):
                rows.append(dict(subject=f"{date}/{subj}", cls=cls, band=band,
                                 ch=ch, db=float(v[ci])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(R_DIR, "manipulation_check.csv"), index=False)

    print("=" * 76)
    print("MANIPULATION CHECK — power during MI relative to the visually")
    print("matched Rest condition.  Negative dB = desynchronisation (ERD).")
    print("Prediction if imagery was performed: negative, central-maximal.")
    print("=" * 76)

    for band in BANDS:
        print(f"\n### {band}  " + "-" * 52)
        print(f"  {'electrode':10s} " + "".join(f"{c:>18s}" for c in
                                                ("Stand  dB (p)", "Walk  dB (p)")))
        pv_s, pv_w, means = [], [], {}
        for ch in CH:
            row = {}
            for cls, store in (("Stand", pv_s), ("Walk", pv_w)):
                v = df[(df.band == band) & (df.cls == cls) &
                       (df.ch == ch)]["db"].values
                t, p = ttest_1samp(v, 0.0)
                store.append(p)
                row[cls] = (v.mean(), p)
            means[ch] = row
        a_s, a_w = holm(pv_s), holm(pv_w)
        for i, ch in enumerate(CH):
            s, w = means[ch]["Stand"], means[ch]["Walk"]
            ms = "*" if a_s[i] < 0.05 else " "
            mw = "*" if a_w[i] < 0.05 else " "
            print(f"  {ch:10s} {s[0]:+11.2f} ({s[1]:.3f}){ms}"
                  f"{w[0]:+11.2f} ({w[1]:.3f}){mw}")

        # central-vs-frontal focus and how many participants show central ERD
        for cls in ("Stand", "Walk"):
            d = df[(df.band == band) & (df.cls == cls)]
            cen = (d[d.ch.isin(CENTRAL)].groupby("subject")["db"].mean())
            t, p = ttest_1samp(cen.values, 0.0)
            try:
                _, pw = wilcoxon(cen.values)
            except ValueError:
                pw = np.nan
            n_neg = int((cen < 0).sum())
            print(f"    {cls:5s} central mean = {cen.mean():+.2f} dB   "
                  f"t-test p={p:.3f}  wilcoxon p={pw:.3f}   "
                  f"{n_neg}/10 participants show ERD")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, len(BANDS), figsize=(4.8 * len(BANDS), 3.9),
                             constrained_layout=True, sharey=True)
    x = np.arange(len(CH))
    for ax, band in zip(np.atleast_1d(axes), BANDS):
        for cls in ("Stand", "Walk"):
            d = df[(df.band == band) & (df.cls == cls)]
            piv = d.pivot_table(index="subject", columns="ch", values="db")[CH]
            m, se = piv.mean(0).values, piv.sem(0).values
            off = -0.12 if cls == "Stand" else 0.12
            ax.errorbar(x + off, m, yerr=se, fmt="o-", ms=5, lw=2, capsize=3,
                        color=SER[cls], label=cls)
        ax.axhline(0, color=INK, lw=1.2)
        ax.set_xticks(x, CH, fontsize=8)
        ax.set_title(band, loc="left")
        ax.set_xlabel("electrode")
    np.atleast_1d(axes)[0].set_ylabel("power vs Rest condition (dB)\n"
                                      "negative = ERD")
    np.atleast_1d(axes)[0].legend(frameon=False, ncols=2, fontsize=9)
    fig.suptitle("Manipulation check: MI vs the visually matched Rest "
                 "condition (mean ± s.e.m., n = 10)\n"
                 "This is the contrast the fixation-baselined ERD figure "
                 "could not provide.", ha="left", x=0.01, fontsize=11)
    fig.savefig(os.path.join(R_DIR, "fig_manipulation_check.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("\nwrote", os.path.join(R_DIR, "fig_manipulation_check.png"))


if __name__ == "__main__":
    main()
