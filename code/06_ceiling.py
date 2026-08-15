# -*- coding: utf-8 -*-
"""
Stage 6 — how much information is actually in an 8-channel montage, and what
would it take to reach a target accuracy?

Two curves, both estimated inside subject:

  TRIALS  acc(n) for n = 5 ... 40 trials PER CLASS, stratified subsamples,
          15 repeats, evaluated on a held-out 20 % test set.  Summarised by
          the slope in accuracy-points per decade of trials, with a
          subject-bootstrap CI.  A slope whose CI contains zero means more
          trials of the same paradigm will not help — the limit is signal,
          not sample size.  (The inverse-power fit acc = a - b*n^-c is also
          reported, but only when the slope is clearly positive: on a short
          noisy curve that 3-parameter fit will happily return a 100 %
          asymptote for pure noise.)

  CHANNELS acc(k) for k = 1 ... 8 electrodes, best k-subset chosen inside the
          training fold only.  The slope at k = 8 says whether spatial
          sampling is still the binding constraint (it usually is for
          lower-limb MI, where the foot representation sits in the
          interhemispheric fissure and needs dense central coverage).

Together these separate "we need more data" from "we need a better cap",
which is the question that decides what the next experiment looks like.

Outputs: results/ceiling_scores.csv, fig_ceiling_trials.png, fig_ceiling_channels.png
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from sklearn.model_selection import (StratifiedKFold, StratifiedShuffleSplit,
                                     cross_val_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import decoders as D          # noqa: E402
import steam_io as sio        # noqa: E402

R_DIR = sio.RESULTS_DIR
CLASSES = sio.CLASS_NAMES
CH = sio.EEG_CHANNELS
TASKS = {"3cls": None, "SvW": ("Stand", "Walk"), "RvS": ("Rest", "Stand")}
INK, GRID = "#1f2328", "#e5e7eb"
SER = ["#1f6feb", "#bc4c00", "#116329", "#8250df", "#6b7280"]
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#6b7280", "ytick.color": "#6b7280",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.dpi": 130,
})

PIPE = "fb_ts_lr"          # best block from stage 2
# Subsample sizes are specified PER CLASS, not as a total, so that every
# requested point is actually achievable given the pool a split leaves behind
# (50 trials/class * (1 - TEST_FRAC)).  Specifying totals silently skipped
# most of the grid and left too few points to fit a curve to.
PER_CLASS = [5, 8, 12, 16, 20, 26, 32, 40]
N_REP = 15
TEST_FRAC = 0.20


def _subset(task, Xp, Xb, y):
    if TASKS[task] is None:
        return Xp, Xb, y
    a, b = TASKS[task]
    ia, ib = CLASSES.index(a), CLASSES.index(b)
    m = np.isin(y, [ia, ib])
    return Xp[m], Xb[m], (y[m] == ib).astype(int)


def trial_curve(sid, Xb, y, seed=0):
    rows = []
    classes = np.unique(y)
    sss = StratifiedShuffleSplit(N_REP, test_size=TEST_FRAC, random_state=seed)
    for rep, (tr_all, te) in enumerate(sss.split(np.zeros(len(y)), y)):
        rng = np.random.RandomState(1000 + rep)
        pools = {c: tr_all[y[tr_all] == c] for c in classes}
        for per in PER_CLASS:
            if any(len(p) < per for p in pools.values()):
                continue
            idx = np.concatenate([rng.choice(pools[c], per, replace=False)
                                  for c in classes])
            mdl = D.get(PIPE).fit(Xb[idx], y[idx])
            rows.append(dict(subject=sid, per_class=per, n_train=len(idx), rep=rep,
                             acc=float((mdl.predict(Xb[te]) == y[te]).mean())))
    return rows


def channel_curve(sid, Xb, y, seed=0):
    """Greedy forward channel selection, refit inside each split."""
    rows = []
    sss = StratifiedShuffleSplit(4, test_size=TEST_FRAC, random_state=seed)
    for rep, (tr, te) in enumerate(sss.split(np.zeros(len(y)), y)):
        chosen, remaining = [], list(range(len(CH)))
        for k in range(1, len(CH) + 1):
            best, best_c = -1, None
            for c in remaining:
                sel = chosen + [c]
                # candidate scored by inner CV on the TRAINING split only
                v = cross_val_score(D.get(PIPE), Xb[tr][:, :, sel], y[tr],
                                    cv=StratifiedKFold(3, shuffle=True,
                                                       random_state=7),
                                    n_jobs=1).mean()
                if v > best:
                    best, best_c = v, c
            chosen.append(best_c)
            remaining.remove(best_c)
            mdl = D.get(PIPE).fit(Xb[tr][:, :, chosen], y[tr])
            rows.append(dict(subject=sid, k=k, rep=rep,
                             acc=float((mdl.predict(Xb[te][:, :, chosen]) == y[te]).mean()),
                             chans="+".join(CH[i] for i in chosen)))
    return rows


def inv_power(n, a, b, c):
    return a - b * np.power(n, -c)


def fit_curve(n, acc):
    try:
        p, _ = curve_fit(inv_power, n, acc, p0=[0.8, 1.0, 0.5],
                         bounds=([0.2, 0, 0.05], [1.0, 50, 3.0]), maxfev=20000)
        return p
    except Exception:                                            # noqa: BLE001
        return None


def log_slope(n, acc):
    """
    Accuracy gain per decade of training trials.

    More robust than the 3-parameter inverse-power fit when the curve is
    short and noisy: with 8 points a power law can 'explain' pure noise with
    an asymptote of 100 %, which is how this analysis produced a nonsense
    '527,927,427 trials' estimate before.  A slope indistinguishable from
    zero is the honest signal that more data of the same kind will not help.
    """
    return float(np.polyfit(np.log10(n), acc, 1)[0])


def boot_slope(df, n_boot=2000, seed=0):
    """Percentile CI on the group slope, resampling SUBJECTS (the unit of
    generalisation), not trials."""
    rng = np.random.RandomState(seed)
    sids = df.subject.unique()
    per = {s: g.groupby("per_class")["acc"].mean() for s, g in df.groupby("subject")}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(sids, len(sids), replace=True)
        m = pd.concat([per[s] for s in pick], axis=1).mean(axis=1)
        out.append(log_slope(m.index.values.astype(float), m.values))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def n_for_target(p, target):
    a, b, c = p
    if a <= target:
        return np.inf
    return float((b / (a - target)) ** (1 / c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="3cls,SvW")
    ap.add_argument("--target", type=float, default=0.85)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--skip-channels", action="store_true")
    a = ap.parse_args()
    cfg = dict(lo=8.0, hi=30.0, tmin=0.5, tmax=4.5, ref="none")

    all_t, all_c = [], []
    for task in a.tasks.split(","):
        chance = 1 / (3 if task == "3cls" else 2)
        data = []
        for date, subj in sio.list_subjects():
            Xp, Xb, y = sio.get_task_arrays(date, subj, **cfg)
            Xp, Xb, y = _subset(task, Xp, Xb, y)
            data.append((f"{date}/{subj}", Xb, y))

        print(f"\n===== {task} : trial curve =====", flush=True)
        out = Parallel(n_jobs=a.jobs)(delayed(trial_curve)(s, X, y) for s, X, y in data)
        rows = [dict(r, task=task) for rr in out for r in rr]
        all_t += rows
        df = pd.DataFrame(rows)

        print(f"  {'subject':14s} " + " ".join(f"{n:>5d}" for n in PER_CLASS)
              + "    slope/decade")
        for sid, g in df.groupby("subject"):
            m = g.groupby("per_class")["acc"].mean()
            sl = log_slope(m.index.values.astype(float), m.values)
            print(f"  {sid:14s} "
                  + " ".join(f"{m.get(n, np.nan)*100:5.1f}" for n in PER_CLASS)
                  + f"    {sl*100:+6.1f} pts")

        mm = df.groupby("per_class")["acc"].mean()
        x = mm.index.values.astype(float)
        sl = log_slope(x, mm.values)
        lo_s, hi_s = boot_slope(df)
        print(f"\n  GROUP curve: " + " ".join(f"{v*100:.1f}" for v in mm.values))
        print(f"  slope = {sl*100:+.1f} accuracy-points per 10x more trials "
              f"(95% CI {lo_s*100:+.1f} .. {hi_s*100:+.1f})")

        # Extrapolate log-linearly from the largest measured point.  The
        # 3-parameter power fit is NOT used here: on 8 noisy points it returns
        # a 100 % asymptote and correspondingly meaningless trial counts.
        n_max, acc_max = float(x[-1]), float(mm.values[-1])
        gap = a.target - acc_max
        if hi_s <= 0:
            print("  => curve is FLAT (slope CI includes no positive value). "
                  "More trials of this paradigm will not help.")
        else:
            decades = gap / sl
            need = n_max * 10 ** decades
            print(f"  => at {n_max:.0f} trials/class the curve is at "
                  f"{acc_max*100:.1f}%; closing the {gap*100:.1f}-point gap to "
                  f"{a.target*100:.0f}% at {sl*100:+.1f} pts/decade needs "
                  f"~{decades:.1f} decades = {need:,.0f} trials/class.")
            dbl = sl * np.log10(2)
            print(f"     practical reading: DOUBLING the session (40 -> 80 "
                  f"trials/class) buys about {dbl*100:+.1f} points.")
            if need > 1e4:
                print("     i.e. not reachable by collecting more of the same "
                      "data — the paradigm/montage has to change.")

        if not a.skip_channels:
            print(f"===== {task} : channel curve =====", flush=True)
            outc = Parallel(n_jobs=a.jobs)(
                delayed(channel_curve)(s, X, y) for s, X, y in data)
            rc = [dict(r, task=task) for rr in outc for r in rr]
            all_c += rc
            dc = pd.DataFrame(rc)
            g = dc.groupby("k")["acc"].mean()
            print("  k:      " + " ".join(f"{k:5d}" for k in g.index))
            print("  acc(%): " + " ".join(f"{v*100:5.1f}" for v in g.values))
            first = dc[dc.k == 1].groupby("chans").size().sort_values(ascending=False)
            print("  most-chosen first electrode:", first.head(3).to_dict())

    pd.DataFrame(all_t).to_csv(os.path.join(R_DIR, "ceiling_trials.csv"), index=False)
    if all_c:
        pd.DataFrame(all_c).to_csv(os.path.join(R_DIR, "ceiling_channels.csv"), index=False)

    # ---- figures ----
    dt = pd.DataFrame(all_t)
    tasks = dt.task.unique()
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 3.8),
                             constrained_layout=True, squeeze=False)
    for ax, task in zip(axes[0], tasks):
        d = dt[dt.task == task]
        chance = 1 / (3 if task == "3cls" else 2)
        for i, (sid, g) in enumerate(d.groupby("subject")):
            m = g.groupby("per_class")["acc"].mean()
            ax.plot(m.index, m.values * 100, lw=1, alpha=0.35, color="#9aa4b2")
        m = d.groupby("per_class")["acc"].mean()
        se = d.groupby("per_class")["acc"].sem()
        ax.errorbar(m.index, m.values * 100, yerr=se.values * 100, lw=2.5,
                    color=SER[0], marker="o", ms=6, capsize=3, label="group mean")
        sl = log_slope(m.index.values.astype(float), m.values)
        ax.plot([], [], " ", label=f"slope {sl*100:+.1f} pts / 10x trials")
        ax.axhline(chance * 100, color=INK, lw=1, ls=":")
        ax.axhline(85, color=SER[2], lw=1.2, ls="-.")
        ax.text(5.5, 86, "85 % target", fontsize=8, color=SER[2])
        ax.set_xlabel("training trials per class")
        ax.set_xscale("log")
        ax.set_ylabel("accuracy (%)")
        ax.set_title(f"{task}", loc="left")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Learning curve — does this paradigm just need more trials?",
                 ha="left", x=0.01, fontsize=11)
    fig.savefig(os.path.join(R_DIR, "fig_ceiling_trials.png"), bbox_inches="tight")
    plt.close(fig)

    if all_c:
        dc = pd.DataFrame(all_c)
        fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.6),
                                 constrained_layout=True, squeeze=False)
        for ax, task in zip(axes[0], tasks):
            d = dc[dc.task == task]
            chance = 1 / (3 if task == "3cls" else 2)
            m = d.groupby("k")["acc"].mean()
            se = d.groupby("k")["acc"].sem()
            ax.errorbar(m.index, m.values * 100, yerr=se.values * 100, lw=2.5,
                        color=SER[0], marker="o", ms=6, capsize=3)
            ax.axhline(chance * 100, color=INK, lw=1, ls=":")
            ax.set_xlabel("electrodes used (greedy forward selection)")
            ax.set_ylabel("accuracy (%)")
            ax.set_title(task, loc="left")
        fig.suptitle("Channel curve — is spatial sampling the binding constraint?",
                     ha="left", x=0.01, fontsize=11)
        fig.savefig(os.path.join(R_DIR, "fig_ceiling_channels.png"), bbox_inches="tight")
        plt.close(fig)
    print("\nfigures in", R_DIR)


if __name__ == "__main__":
    main()
