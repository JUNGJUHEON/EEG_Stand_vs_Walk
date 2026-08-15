# -*- coding: utf-8 -*-
"""
Stage 19 — the direct physiological statement: are the Stand and Walk scalp
topographies distinguishable at all?

Everything else in this project shows that Stand and Walk cannot be *decoded*.
This asks the prior question — do they even produce different patterns on the
scalp? — and answers it against the right reference.

WHY THE OBVIOUS ANALYSIS DOES NOT WORK
--------------------------------------
The tempting version is to correlate the Stand-minus-Rest topography with the
Walk-minus-Rest topography.  Both contain the term -Rest, so they are
correlated by construction.  Measured here: that comparison gives r = +0.67,
but re-referencing the same data to a different baseline moves it to +0.47
(Stand as baseline) and +0.19 (Walk as baseline).  The number is dominated by
the shared baseline, not by the classes, and it cannot support a conclusion.

THE TEST USED INSTEAD
---------------------
Compare the between-class topographic distance against that participant's own
NOISE FLOOR, measured with matched sample size:

    split Stand trials at random into two halves -> S1, S2
    split Walk  trials at random into two halves -> W1, W2

    within  = mean(||S1 - S2||, ||W1 - W2||)      pure noise, no class signal
    between = mean(||S1 - W1||, ||S2 - W2||)      class signal + the same noise

Both use n/2 trials per topography, so the noise contribution is identical by
construction and no baseline is shared.  Ratio between/within = 1.00 means the
two conditions differ no more than two random halves of one condition do.
200 random splits per participant.

Output: results/topography_floor.csv, fig_topography_floor.png
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
from scipy.stats import ttest_rel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import steam_io as sio        # noqa: E402

R_DIR = sio.RESULTS_DIR
CH = sio.EEG_CHANNELS
BANDS = {"mu 8-13": (8, 13), "beta 13-30": (13, 30)}
PAIRS = {"Stand vs Walk": (1, 2), "Rest vs Walk": (0, 2), "Rest vs Stand": (0, 1)}
N_SPLIT = 200
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
BLUE, ORANGE = "#1f6feb", "#bc4c00"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.dpi": 170,
})


def band_power(date, subj, lo, hi):
    ep = sio.get_epochs(date, subj, l_freq=1.0, h_freq=45.0,
                        tmin=0.5, tmax=4.5, ref="none")
    f, P = welch(ep["X"], fs=ep["fs"], nperseg=int(ep["fs"]), axis=-1)
    sel = (f >= lo) & (f <= hi)
    return np.log10(P[:, :, sel].mean(-1) + 1e-30), ep["y"]


def ratio_for(lp, y, ca, cb, rng):
    ia, ib = np.where(y == ca)[0], np.where(y == cb)[0]
    if min(len(ia), len(ib)) < 8:
        return np.nan, np.nan
    w, b = [], []
    for _ in range(N_SPLIT):
        a, c = rng.permutation(ia), rng.permutation(ib)
        A1, A2 = lp[a[:len(a) // 2]].mean(0), lp[a[len(a) // 2:]].mean(0)
        B1, B2 = lp[c[:len(c) // 2]].mean(0), lp[c[len(c) // 2:]].mean(0)
        w.append((np.linalg.norm(A1 - A2) + np.linalg.norm(B1 - B2)) / 2)
        b.append((np.linalg.norm(A1 - B1) + np.linalg.norm(A2 - B2)) / 2)
    return float(np.mean(w)), float(np.mean(b))


def main():
    rng = np.random.RandomState(0)
    subs = sio.list_subjects()
    rows = []

    print("=" * 78)
    print("TOPOGRAPHIC SEPARABILITY vs THE PARTICIPANT'S OWN NOISE FLOOR")
    print("  ratio = between-class distance / within-class split-half distance")
    print("  1.00 means the two conditions differ no more than two random")
    print("  halves of a single condition do.")
    print("=" * 78)

    for bname, (lo, hi) in BANDS.items():
        cache = {}
        print(f"\n### {bname}")
        for pname, (ca, cb) in PAIRS.items():
            W, B = [], []
            for date, subj in subs:
                key = (date, subj)
                if key not in cache:
                    cache[key] = band_power(date, subj, lo, hi)
                lp, y = cache[key]
                w, b = ratio_for(lp, y, ca, cb, rng)
                W.append(w)
                B.append(b)
                rows.append(dict(band=bname, pair=pname, subject=f"{date}/{subj}",
                                 within=w, between=b, ratio=b / w))
            W, B = np.array(W), np.array(B)
            r = B / W
            t, p = ttest_rel(B, W)
            verdict = ("distinguishable" if p < .05 and r.mean() > 1
                       else "NOT distinguishable from noise")
            print(f"  {pname:16s} ratio {r.mean():5.2f}   t={t:+5.2f}  "
                  f"p={p:.4f}   {verdict}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(R_DIR, "topography_floor.csv"), index=False)

    # --------------------------- figure ---------------------------
    fig, axes = plt.subplots(1, len(BANDS), figsize=(5.6 * len(BANDS), 4.0),
                             constrained_layout=True, sharey=True)
    for ax, bname in zip(np.atleast_1d(axes), BANDS):
        d = df[df.band == bname]
        names = list(PAIRS)
        for i, pname in enumerate(names):
            v = d[d.pair == pname]["ratio"].values
            ax.bar(i, v.mean(), 0.6,
                   color=BLUE if v.mean() > 1.25 else ORANGE, zorder=2)
            ax.scatter(np.full(len(v), i), v, s=14, color="white",
                       edgecolor=INK, linewidth=0.6, zorder=3)
        ax.axhline(1.0, color=INK, lw=1.6, ls="--")
        ax.text(-0.42, 1.03, "noise floor", fontsize=7.6, color=INK)
        ax.set_xticks(range(len(names)), [n.replace(" vs ", "\nvs ")
                                          for n in names], fontsize=8.2)
        ax.set_title(bname, loc="left")
    np.atleast_1d(axes)[0].set_ylabel(
        "between-class distance ÷\nwithin-class split-half distance")
    fig.suptitle("Do the conditions produce different scalp topographies at "
                 "all?  Each participant compared to their own noise floor",
                 ha="left", x=0.008, fontsize=11.5, fontweight="bold")
    fig.savefig(os.path.join(R_DIR, "fig_topography_floor.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("\nwrote", os.path.join(R_DIR, "fig_topography_floor.png"))


if __name__ == "__main__":
    main()
