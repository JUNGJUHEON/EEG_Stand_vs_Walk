# -*- coding: utf-8 -*-
"""
Stage 2 — honest evaluation at a fixed preprocessing configuration.

Three numbers are produced for every subject x task:

  1. per-pipeline accuracy under repeated stratified 5-fold CV
     (the usual number; comparable to the lab's existing scripts)

  2. NESTED accuracy — the outer fold never sees the pipeline choice.
     An inner 4-fold CV picks the pipeline on the training part only.
     This is the number to quote, because "best pipeline out of 17" chosen
     on the same CV you report is optimistically biased.

  3. CHRONOLOGICAL accuracy — train on the first 60 % of trials, test on the
     last 40 %.  Random k-fold interleaves early and late trials, so slow
     within-session drift (impedance, sweat, fatigue) can be learned as if it
     were class information.  If (3) collapses relative to (1), the random-CV
     number is partly drift.

Outputs under results/:
  final_scores.csv, fig_final_heatmap.png, fig_confusion_<task>.png,
  fig_nested_vs_chrono.png
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
from joblib import Parallel, delayed
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import decoders as D          # noqa: E402
import steam_io as sio        # noqa: E402

R = sio.RESULTS_DIR
CLASSES = sio.CLASS_NAMES
TASKS = {"SvW": ("Stand", "Walk"), "RvS": ("Rest", "Stand"),
         "RvW": ("Rest", "Walk"), "3cls": None}

INK, INK_MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "figure.dpi": 130,
})

# pipelines entering the nested model-selection pool (keep it small so the
# inner CV has enough data per candidate)
NESTED_POOL = ["csp_lda", "ts_lr", "acm_d5_10_ts_lr", "fgmdm",
               "fb_ts_lr", "fbcsp_mibif_lda"]


# ============================== helpers ==============================
def load_task(date, subj, task, cfg):
    """Return (X_plain, X_bank, y) for one subject/task, trial-aligned."""
    Xp, Xb, y = sio.get_task_arrays(date, subj, **cfg)
    if TASKS[task] is None:
        return Xp, Xb, y
    a, b = TASKS[task]
    ia, ib = CLASSES.index(a), CLASSES.index(b)
    m = np.isin(y, [ia, ib])
    return Xp[m], Xb[m], (y[m] == ib).astype(int)


def _pick(name, Xp, Xb):
    return Xb if D.is_bank(name) else Xp


def nested_score(Xp, Xb, y, pool, n_outer=5, seed=0):
    """Outer CV accuracy where the pipeline is chosen inside each training fold."""
    outer = StratifiedKFold(n_outer, shuffle=True, random_state=seed)
    accs, chosen = [], []
    for tr, te in outer.split(np.zeros(len(y)), y):
        inner = StratifiedKFold(4, shuffle=True, random_state=seed + 1)
        best, best_name = -np.inf, pool[0]
        for name in pool:
            X = _pick(name, Xp, Xb)
            s = cross_val_score(D.get(name), X[tr], y[tr], cv=inner,
                                scoring="accuracy", n_jobs=1).mean()
            if s > best:
                best, best_name = s, name
        X = _pick(best_name, Xp, Xb)
        mdl = D.get(best_name).fit(X[tr], y[tr])
        accs.append(float((mdl.predict(X[te]) == y[te]).mean()))
        chosen.append(best_name)
    return float(np.mean(accs)), chosen


def chrono_score(Xp, Xb, y, name, frac=0.6):
    """Train on the first `frac` of trials in acquisition order, test on the rest."""
    n = len(y)
    cut = int(n * frac)
    tr, te = np.arange(cut), np.arange(cut, n)
    if len(np.unique(y[tr])) < 2 or len(te) == 0:
        return np.nan
    X = _pick(name, Xp, Xb)
    mdl = D.get(name).fit(X[tr], y[tr])
    return float((mdl.predict(X[te]) == y[te]).mean())


def eval_subject(date, subj, task, cfg, pipes, repeats):
    Xp, Xb, y = load_task(date, subj, task, cfg)
    sid = f"{date}/{subj}"
    rows, preds = [], {}

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=42)
    for name in pipes:
        X = _pick(name, Xp, Xb)
        s = cross_val_score(D.get(name), X, y, cv=cv, scoring="accuracy", n_jobs=1)
        rows.append(dict(subject=sid, task=task, pipe=name, kind="kfold",
                         acc=float(s.mean()), std=float(s.std()), n=len(y)))
        preds[name] = cross_val_predict(
            D.get(name), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
            n_jobs=1)
        rows.append(dict(subject=sid, task=task, pipe=name, kind="chrono",
                         acc=chrono_score(Xp, Xb, y, name), std=np.nan, n=len(y)))

    acc, chosen = nested_score(Xp, Xb, y, NESTED_POOL)
    rows.append(dict(subject=sid, task=task, pipe="NESTED", kind="nested",
                     acc=acc, std=np.nan, n=len(y),
                     note="|".join(chosen)))
    return rows, (sid, task, y, preds)


# ============================== figures ==============================
def heatmap(df, cfg):
    for task in TASKS:
        d = df[(df.task == task) & (df.kind == "kfold")]
        if d.empty:
            continue
        piv = d.pivot_table(index="subject", columns="pipe", values="acc")
        order = piv.mean().sort_values(ascending=False).index
        piv = piv[order]
        nest = (df[(df.task == task) & (df.kind == "nested")]
                .set_index("subject")["acc"].reindex(piv.index))
        piv.insert(0, "NESTED", nest)

        chance = 1 / (3 if task == "3cls" else 2)
        fig, ax = plt.subplots(figsize=(1.05 + 0.62 * piv.shape[1],
                                        1.6 + 0.34 * piv.shape[0]),
                               constrained_layout=True)
        im = ax.imshow(piv.values * 100, cmap="Blues", aspect="auto",
                       vmin=chance * 100, vmax=max(85, np.nanmax(piv.values) * 100))
        ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=8)
        ax.grid(False)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if v > (chance + np.nanmax(piv.values)) / 2 else INK)
        fig.colorbar(im, ax=ax, shrink=0.7, label="accuracy (%)")
        ax.set_title(f"{task}  —  per-subject decoding accuracy   "
                     f"(chance {chance*100:.0f}%)\n"
                     f"{cfg['lo']}–{cfg['hi']} Hz, t∈[{cfg['tmin']},{cfg['tmax']}] s, "
                     f"ref={cfg['ref']}   ·   column 1 = nested (unbiased)",
                     loc="left", fontsize=10)
        fig.savefig(os.path.join(R, f"fig_final_heatmap_{task}.png"), bbox_inches="tight")
        plt.close(fig)


def confusions(store, best_pipe, cfg):
    for task in TASKS:
        items = [(sid, y, pr[best_pipe]) for (sid, t, y, pr) in store
                 if t == task and best_pipe in pr]
        if not items:
            continue
        labels = (CLASSES if task == "3cls" else list(TASKS[task]))
        n = len(labels)
        agg = np.zeros((n, n))
        k = len(items)
        cols = min(5, k + 1)
        rows_n = int(np.ceil((k + 1) / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(2.35 * cols, 2.55 * rows_n),
                                 constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, (sid, y, p) in zip(axes, items):
            cm = confusion_matrix(y, p, labels=range(n))
            agg += cm
            cmn = cm / cm.sum(1, keepdims=True)
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{cmn[i,j]*100:.0f}", ha="center", va="center",
                            fontsize=8, color="white" if cmn[i, j] > 0.55 else INK)
            ax.set_xticks(range(n), labels, fontsize=7)
            ax.set_yticks(range(n), labels, fontsize=7)
            ax.grid(False)
            ax.set_title(f"{sid}  {np.trace(cm)/cm.sum()*100:.0f}%", fontsize=8.5)
        ax = axes[k]
        agn = agg / agg.sum(1, keepdims=True)
        ax.imshow(agn, cmap="Blues", vmin=0, vmax=1)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{agn[i,j]*100:.0f}", ha="center", va="center",
                        fontsize=9, weight="bold",
                        color="white" if agn[i, j] > 0.55 else INK)
        ax.set_xticks(range(n), labels, fontsize=7)
        ax.set_yticks(range(n), labels, fontsize=7)
        ax.grid(False)
        ax.set_title(f"POOLED  {np.trace(agg)/agg.sum()*100:.1f}%", fontsize=9, weight="bold")
        for ax in axes[k + 1:]:
            ax.axis("off")
        fig.suptitle(f"{task} — confusion matrices, {best_pipe}, 5-fold out-of-fold "
                     f"predictions  (rows = true, % of row)", ha="left", x=0.01, fontsize=11)
        fig.savefig(os.path.join(R, f"fig_confusion_{task}.png"), bbox_inches="tight")
        plt.close(fig)


def nested_vs_chrono(df, best_pipe):
    """
    `best_pipe` is fixed globally (chosen from the k-fold table) and used for
    the chronological bars, so the comparison is not "best of 17 on the test
    split" against a nested estimate.
    """
    fig, axes = plt.subplots(1, len(TASKS), figsize=(3.1 * len(TASKS), 3.4),
                             constrained_layout=True, sharey=True)
    for ax, task in zip(np.atleast_1d(axes), TASKS):
        chance = 1 / (3 if task == "3cls" else 2)
        nest = df[(df.task == task) & (df.kind == "nested")].set_index("subject")["acc"]
        ch = (df[(df.task == task) & (df.kind == "chrono") & (df.pipe == best_pipe)]
              .set_index("subject")["acc"])
        idx = nest.index
        x = np.arange(len(idx))
        ax.bar(x - 0.2, nest.values * 100, 0.4, label="nested k-fold", color="#1f6feb")
        ax.bar(x + 0.2, ch.reindex(idx).values * 100, 0.4,
               label=f"chronological 60/40 ({best_pipe})", color="#bc4c00")
        ax.axhline(chance * 100, color=INK, lw=1, ls="--")
        ax.set_xticks(x, [s.split("/")[1] for s in idx], rotation=60, ha="right", fontsize=7)
        ax.set_title(task, loc="left")
        ax.set_ylim(0, 100)
    np.atleast_1d(axes)[0].set_ylabel("accuracy (%)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Unbiased nested CV vs a chronological split "
                 "(dashed = chance)", ha="left", x=0.01, fontsize=11)
    fig.savefig(os.path.join(R, "fig_nested_vs_chrono.png"), bbox_inches="tight")
    plt.close(fig)


# ================================ main ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=8.0)
    ap.add_argument("--hi", type=float, default=30.0)
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=4.5)
    ap.add_argument("--ref", default="car")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--tasks", default="SvW,RvS,RvW,3cls")
    a = ap.parse_args()
    cfg = dict(lo=a.lo, hi=a.hi, tmin=a.tmin, tmax=a.tmax, ref=a.ref)

    pipes = list(D.ALL)
    subs = sio.list_subjects()
    tasks = a.tasks.split(",")
    print(f"[final] cfg={cfg}  {len(subs)} subjects x {len(tasks)} tasks "
          f"x {len(pipes)} pipes")

    t0 = time.time()
    out = Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(eval_subject)(d, s, t, cfg, pipes, a.repeats)
        for t in tasks for (d, s) in subs)

    rows = [r for rr, _ in out for r in rr]
    store = [st for _, st in out]
    df = pd.DataFrame(rows)
    for k, v in cfg.items():
        df[k] = v
    df.to_csv(os.path.join(R, "final_scores.csv"), index=False)

    print(f"\n[final] {time.time()-t0:.0f}s\n")
    kf = df[df.kind == "kfold"].groupby(["task", "pipe"])["acc"].mean().unstack(0)
    print("mean k-fold accuracy across subjects:")
    print((kf * 100).round(1).sort_values(kf.columns[0], ascending=False).to_string())

    print("\nnested (unbiased) per task:")
    for t in tasks:
        d = df[(df.task == t) & (df.kind == "nested")]
        v = d["acc"].to_numpy()
        chance = 1 / (3 if t == "3cls" else 2)
        # exact pooled binomial over the true per-subject trial counts
        n_tot = int(d["n"].sum())
        k = int(round((d["acc"] * d["n"]).sum()))
        p = binomtest(k, n_tot, chance, alternative="greater").pvalue
        print(f"  {t:5s} {v.mean()*100:5.1f} % ± {v.std()*100:4.1f} "
              f"(range {v.min()*100:.0f}-{v.max()*100:.0f}, chance {chance*100:.0f}, "
              f"pooled {k}/{n_tot}, binomial p={p:.2g})")

    heatmap(df, cfg)
    best = kf.mean(axis=1).idxmax()
    print(f"\nbest overall pipeline: {best}")
    confusions(store, best, cfg)
    nested_vs_chrono(df, best)
    print("figures in", R)


if __name__ == "__main__":
    main()
