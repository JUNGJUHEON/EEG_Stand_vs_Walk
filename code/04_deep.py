# -*- coding: utf-8 -*-
"""
Stage 4 — EEGNet, same corrected preprocessing, two regimes.

  within  : per-subject 5-fold CV   (directly comparable with the lab's
            existing EEGNet scripts, which also trained per subject)
  loso+EA : pooled training on nine Euclidean-aligned subjects, tested on the
            tenth — the regime where a deep net actually has enough data

Outputs: results/deep_scores.csv, fig_deep.png, fig_confusion_deep_<task>.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import eegnet                                     # noqa: E402
import steam_io as sio                            # noqa: E402
from importlib import import_module               # noqa: E402

_tr = import_module("03_transfer")
euclidean_align, TASKS = _tr.euclidean_align, _tr.TASKS

R_DIR = sio.RESULTS_DIR
CLASSES = sio.CLASS_NAMES
INK, GRID = "#1f2328", "#e5e7eb"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#6b7280", "ytick.color": "#6b7280",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.dpi": 130,
})


def build(task, cfg, align):
    data = {}
    for date, subj in sio.list_subjects():
        X, _, y = sio.get_task_arrays(date, subj, **cfg)
        if TASKS[task] is not None:
            a, b = TASKS[task]
            ia, ib = CLASSES.index(a), CLASSES.index(b)
            m = np.isin(y, [ia, ib])
            X, y = X[m], (y[m] == ib).astype(int)
        if align:
            X = euclidean_align(X)
        data[f"{date}/{subj}"] = (X.astype(np.float32) * 1e6, y)  # V -> uV scale
    return data


def within(data, n_classes, seed=0):
    rows, preds = [], []
    for sid, (X, y) in data.items():
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        p = np.empty_like(y)
        for tr, te in cv.split(X, y):
            p[te], _ = eegnet.fit_predict(X[tr], y[tr], X[te],
                                          n_classes=n_classes, seed=seed)
        rows.append(dict(subject=sid, regime="within", acc=float((p == y).mean()),
                         n=len(y)))
        preds.append((sid, y, p))
        print(f"    within {sid}: {rows[-1]['acc']*100:.1f}%", flush=True)
    return rows, preds


def loso(data, n_classes, seed=0):
    rows, preds = [], []
    sids = list(data)
    for held in sids:
        Xte, yte = data[held]
        Xtr = np.concatenate([data[s][0] for s in sids if s != held])
        ytr = np.concatenate([data[s][1] for s in sids if s != held])
        p, _ = eegnet.fit_predict(Xtr, ytr, Xte, n_classes=n_classes,
                                  epochs=200, patience=30, bs=64, seed=seed)
        rows.append(dict(subject=held, regime="loso+EA", acc=float((p == yte).mean()),
                         n=len(yte)))
        preds.append((held, yte, p))
        print(f"    loso   {held}: {rows[-1]['acc']*100:.1f}%  "
              f"(train n={len(ytr)})", flush=True)
    return rows, preds


def plot_conf(preds, task, tag):
    labels = CLASSES if TASKS[task] is None else list(TASKS[task])
    n = len(labels)
    agg = np.zeros((n, n))
    for _, y, p in preds:
        agg += confusion_matrix(y, p, labels=range(n))
    cmn = agg / agg.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(2.4 + 0.5 * n, 2.4 + 0.5 * n), constrained_layout=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cmn[i,j]*100:.1f}", ha="center", va="center",
                    fontsize=11, color="white" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    ax.set_title(f"EEGNet · {task} · {tag}\npooled {np.trace(agg)/agg.sum()*100:.1f}%",
                 loc="left", fontsize=10)
    fig.savefig(os.path.join(R_DIR, f"fig_confusion_deep_{task}_{tag}.png"),
                bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=8.0)
    ap.add_argument("--hi", type=float, default=30.0)
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=4.5)
    ap.add_argument("--ref", default="car")
    ap.add_argument("--tasks", default="SvW,3cls")
    a = ap.parse_args()
    cfg = dict(lo=a.lo, hi=a.hi, tmin=a.tmin, tmax=a.tmax, ref=a.ref)
    print("device:", eegnet.device())

    all_rows = []
    for task in a.tasks.split(","):
        nc = 3 if task == "3cls" else 2
        print(f"\n===== EEGNet {task} =====", flush=True)
        t0 = time.time()
        d_raw = build(task, cfg, align=False)
        d_ali = build(task, cfg, align=True)
        rw, pw = within(d_raw, nc)
        rl, pl = loso(d_ali, nc)
        for r in rw + rl:
            r["task"] = task
        all_rows += rw + rl
        plot_conf(pw, task, "within")
        plot_conf(pl, task, "loso+EA")
        print(f"  within {np.mean([r['acc'] for r in rw])*100:.1f}%   "
              f"loso+EA {np.mean([r['acc'] for r in rl])*100:.1f}%   "
              f"({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(all_rows)
    for k, v in cfg.items():
        df[k] = v
    df.to_csv(os.path.join(R_DIR, "deep_scores.csv"), index=False)
    print("\nwrote", os.path.join(R_DIR, "deep_scores.csv"))


if __name__ == "__main__":
    main()
