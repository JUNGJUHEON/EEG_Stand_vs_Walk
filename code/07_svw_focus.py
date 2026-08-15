# -*- coding: utf-8 -*-
"""
Stage 7 — maximal honest attack on Stand vs Walk, the hard pair.

Hypothesis under test
---------------------
Stand imagery is a SINGLE postural transition; the subject is told to imagine
standing up, which takes ~1-2 s, leaving the rest of the 5 s cue idle.  Walk
imagery is TWO alternating steps spread across the whole cue.  If that is
true the classes should diverge late in the trial, and a decoder built on the
trial average (which is what every previous analysis used) throws that away.

So this script scans, for the true Stand-vs-Walk pair only:
  * 15 time windows, including short late windows and short early windows
  * 7 band-pass settings, including a >30 Hz option flagged as EMG-suspect
  * 3 references (none / CAR / small Laplacian)
  * the strongest pipelines, incl. crop ensembling

Everything is repeated stratified 5-fold within subject.  This is an
EXPLORATION grid: the winner of a 300-cell scan is optimistically biased, so
whatever it finds must be re-checked with 05_optimize.py's nested protocol
before it counts.  That caveat is the whole reason this file is separate.

Output: results/svw_focus.csv, fig_svw_window_scan.png
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import decoders as D          # noqa: E402
import steam_io as sio        # noqa: E402

R_DIR = sio.RESULTS_DIR
INK, GRID = "#1f2328", "#e5e7eb"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#6b7280", "ytick.color": "#6b7280",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.dpi": 130,
})

WINDOWS = [
    (0.0, 5.0), (0.5, 4.5), (0.0, 2.5), (2.5, 5.0),          # halves
    (0.0, 1.5), (1.0, 2.5), (2.0, 3.5), (3.0, 4.5), (3.5, 5.0),  # 1.5 s slices
    (0.0, 2.0), (2.0, 4.0), (3.0, 5.0),                      # 2 s slices
    (1.0, 5.0), (2.0, 5.0), (1.5, 4.5),
]
BANDS = [(8, 30), (8, 13), (13, 30), (4, 40), (8, 45), (16, 30), (4, 8)]
REFS = ["none", "car"]
PIPES = ["ts_lr", "acm_d5_10_ts_lr", "fgmdm", "csp_lda"]


def cell(date, subj, lo, hi, tmin, tmax, ref, repeats):
    ep = sio.get_epochs(date, subj, l_freq=lo, h_freq=hi,
                        tmin=tmin, tmax=tmax, ref=ref)
    X, y = sio.pair(ep, "Stand", "Walk")
    if len(np.unique(y)) < 2:
        return []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=42)
    out = []
    for name in PIPES:
        s = cross_val_score(D.get(name), X, y, cv=cv, scoring="accuracy", n_jobs=1)
        out.append(dict(subject=f"{date}/{subj}", pipe=name, lo=lo, hi=hi,
                        tmin=tmin, tmax=tmax, ref=ref, n=len(y),
                        acc=float(s.mean()), std=float(s.std())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()

    subs = sio.list_subjects()
    combos = list(itertools.product(BANDS, REFS))
    print(f"[svw] {len(subs)} subjects x {len(WINDOWS)} windows x {len(combos)} "
          f"band/ref x {len(PIPES)} pipes", flush=True)

    rows, t0 = [], time.time()
    for (lo, hi), ref in combos:
        jobs = [delayed(cell)(d, s, lo, hi, tmin, tmax, ref, a.repeats)
                for (d, s) in subs for (tmin, tmax) in WINDOWS]
        for r in Parallel(n_jobs=a.jobs)(jobs):
            rows.extend(r)
        print(f"  {lo}-{hi} Hz ref={ref:4s}  n={len(rows):6d}  "
              f"{time.time()-t0:5.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(R_DIR, "svw_focus.csv"), index=False)

    g = df.groupby(["tmin", "tmax", "lo", "hi", "ref", "pipe"])["acc"].mean()
    print("\nTOP 15 cells (mean over subjects) — EXPLORATION, biased:")
    print((g.sort_values(ascending=False).head(15) * 100).round(1).to_string())

    print("\nWindow marginal (mean over band/ref/pipe/subject):")
    w = df.groupby(["tmin", "tmax"])["acc"].mean().sort_values(ascending=False)
    print((w * 100).round(1).to_string())

    print("\nPer-subject best cell (upper bound, NOT a valid estimate):")
    for sid, d in df.groupby("subject"):
        r = d.loc[d["acc"].idxmax()]        # r["pipe"], not r.pipe — DataFrame.pipe
        print(f"  {sid:14s} {r['acc']*100:5.1f}%  {r['pipe']:16s} "
              f"{r['lo']:.0f}-{r['hi']:.0f}Hz t[{r['tmin']},{r['tmax']}] "
              f"ref={r['ref']}")

    # figure: window x band heatmap, best pipe/ref per cell
    piv = (df.groupby(["tmin", "tmax", "lo", "hi"])["acc"].max()
             .reset_index()
             .pivot_table(index=["tmin", "tmax"], columns=["lo", "hi"], values="acc"))
    fig, ax = plt.subplots(figsize=(1.5 + 0.85 * piv.shape[1],
                                    1.6 + 0.34 * piv.shape[0]),
                           constrained_layout=True)
    im = ax.imshow(piv.values * 100, cmap="Blues", aspect="auto", vmin=50,
                   vmax=max(60, np.nanmax(piv.values) * 100))
    ax.set_xticks(range(piv.shape[1]), [f"{a_}-{b_}" for a_, b_ in piv.columns],
                  rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0]), [f"[{a_},{b_}]" for a_, b_ in piv.index],
                  fontsize=8)
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v * 100 > 58 else INK)
    fig.colorbar(im, ax=ax, shrink=0.7, label="accuracy (%)")
    ax.set_xlabel("band (Hz)")
    ax.set_ylabel("window (s from imagery onset)")
    ax.set_title("Stand vs Walk — window x band scan (best pipeline/reference "
                 "per cell)\nchance 50 %; exploration grid, values are upper-biased",
                 loc="left", fontsize=10)
    fig.savefig(os.path.join(R_DIR, "fig_svw_window_scan.png"), bbox_inches="tight")
    plt.close(fig)
    print("\nwrote", os.path.join(R_DIR, "svw_focus.csv"))


if __name__ == "__main__":
    main()
