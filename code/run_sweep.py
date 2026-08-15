# -*- coding: utf-8 -*-
"""
Stage 1 — preprocessing sweep.

Question: which epoch window, band-pass and reference actually carry the
Stand-vs-Walk (and Rest-vs-*) information?

Also acts as a *control experiment*: the window (-2.0, 0.0) contains only the
cue screen (up-arrow / right-arrow / rest text) and NO motor imagery.  If that
window decodes above chance, the classifier is reading the visual stimulus,
not the imagery — which is exactly what the old tmin = -1.0 setting leaked in.

Output: results/sweep_<task>.csv
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import decoders as D          # noqa: E402
import steam_io as sio        # noqa: E402

# ------------------------------- grid -------------------------------
WINDOWS = [
    (-2.0, 0.0),   # CONTROL: cue screen only, no imagery
    (-1.0, 5.0),   # what the old pipeline used (cue leak included)
    (0.0, 5.0),    # full MI period
    (0.0, 4.0),
    (0.5, 4.5),
    (1.0, 5.0),
    (1.0, 4.0),
    (0.5, 2.5),    # early MI
    (2.5, 4.5),    # late MI
]
BANDS = [(8, 30), (4, 40), (8, 13), (13, 30), (4, 30)]
REFS = ["car", "none"]
PIPES = ["csp_lda", "ts_lr", "acm_d5_10_ts_lr", "fgmdm"]

TASKS = {
    "SvW": ("Stand", "Walk"),
    "RvS": ("Rest", "Stand"),
    "RvW": ("Rest", "Walk"),
    "3cls": None,
}


def eval_cell(date, subj, lo, hi, tmin, tmax, ref, task, n_repeats=5):
    """One (subject, band, window, ref, task) cell -> one row per pipeline."""
    ep = sio.get_epochs(date, subj, l_freq=lo, h_freq=hi,
                        tmin=tmin, tmax=tmax, ref=ref)
    if TASKS[task] is None:
        X, y = ep["X"], ep["y"]
    else:
        X, y = sio.pair(ep, *TASKS[task])
    if len(np.unique(y)) < 2:
        return []

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=42)
    rows = []
    for name in PIPES:
        s = cross_val_score(D.get(name), X, y, cv=cv, scoring="accuracy", n_jobs=1)
        rows.append(dict(subject=f"{date}/{subj}", task=task, pipe=name,
                         lo=lo, hi=hi, tmin=tmin, tmax=tmax, ref=ref,
                         n=len(y), acc=float(s.mean()), std=float(s.std())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="SvW", choices=list(TASKS))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=7)
    args = ap.parse_args()

    subjects = sio.list_subjects()
    grid = list(itertools.product(BANDS, WINDOWS, REFS))
    print(f"[sweep] task={args.task}  {len(subjects)} subjects x {len(grid)} configs "
          f"x {len(PIPES)} pipes = {len(subjects)*len(grid)*len(PIPES)} evals")

    t0 = time.time()
    all_rows = []
    # band/ref outermost so the continuous-filter cache is reused across windows
    for (lo, hi), ref in itertools.product(BANDS, REFS):
        wins = [w for w in WINDOWS]
        jobs = [delayed(eval_cell)(d, s, lo, hi, tmin, tmax, ref, args.task, args.repeats)
                for (d, s) in subjects for (tmin, tmax) in wins]
        out = Parallel(n_jobs=args.jobs, verbose=0)(jobs)
        for r in out:
            all_rows.extend(r)
        done = len(all_rows)
        print(f"  band {lo}-{hi} ref={ref:4s}  rows={done:6d}  "
              f"elapsed={time.time()-t0:6.0f}s", flush=True)

    df = pd.DataFrame(all_rows)
    out = os.path.join(sio.RESULTS_DIR, f"sweep_{args.task}.csv")
    df.to_csv(out, index=False)
    print(f"\n[sweep] wrote {out}  ({len(df)} rows, {time.time()-t0:.0f}s)")

    g = (df.groupby(["tmin", "tmax", "lo", "hi", "ref", "pipe"])["acc"]
           .mean().reset_index().sort_values("acc", ascending=False))
    print("\nTop 20 configs (mean over subjects):")
    print(g.head(20).to_string(index=False))
    print("\nCue-only control window (-2.0, 0.0):")
    print(g[(g.tmin == -2.0)].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
