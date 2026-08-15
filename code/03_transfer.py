# -*- coding: utf-8 -*-
"""
Stage 3 — cross-subject transfer with Euclidean Alignment.

Motivation: each subject only contributes ~50 trials per class, which is the
binding constraint on this dataset — not the model.  Pooling ten subjects
gives ~500 trials per class, but raw EEG covariances are not comparable
across subjects (different impedances, cap placement, head geometry).

Euclidean Alignment (He & Wu, IEEE TBME 2020) fixes that in one line:

    R = mean_i( X_i X_i^T )        # over ALL trials of one subject
    X_i <- R^{-1/2} X_i            # every subject's mean covariance -> I

It is fully unsupervised, so it may legitimately be applied to a held-out
subject's data before that subject's labels are known.

Three regimes are evaluated:
  within   — train and test inside one subject (the baseline from stage 2)
  loso     — leave one subject out, zero calibration trials for the new user
  loso+ft  — leave one subject out, then add the target's first `n_cal`
             trials to the training set (realistic short calibration)

Outputs: results/transfer_scores.csv, fig_transfer.png,
         fig_confusion_transfer_<task>.png
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
from scipy.linalg import fractional_matrix_power
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import decoders as D          # noqa: E402
import steam_io as sio        # noqa: E402

R_DIR = sio.RESULTS_DIR
CLASSES = sio.CLASS_NAMES
TASKS = {"SvW": ("Stand", "Walk"), "RvS": ("Rest", "Stand"),
         "RvW": ("Rest", "Walk"), "3cls": None}

INK, GRID = "#1f2328", "#e5e7eb"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#6b7280", "ytick.color": "#6b7280",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "figure.dpi": 130,
})


def euclidean_align(X: np.ndarray) -> np.ndarray:
    """
    X : (n_trials, n_ch, n_times) for ONE subject -> aligned copy.
    Unsupervised: uses every trial of the subject regardless of label.
    """
    C = np.einsum("nct,ndt->cd", X, X) / (X.shape[0] * X.shape[2])
    C = C + np.eye(C.shape[0]) * 1e-12 * np.trace(C)
    W = np.real(fractional_matrix_power(C, -0.5))
    return np.einsum("cd,ndt->nct", W, X)


def align_bank(Xb: np.ndarray) -> np.ndarray:
    """Same, applied band-by-band to (n, n_bands, n_ch, n_times)."""
    return np.stack([euclidean_align(Xb[:, b]) for b in range(Xb.shape[1])], axis=1)


# ============================== data ==============================
def build(task, cfg, align: bool):
    """Load every subject, optionally EA-align, return dict of arrays."""
    out = {}
    for date, subj in sio.list_subjects():
        Xp, Xb, y = sio.get_task_arrays(date, subj, **cfg)
        if TASKS[task] is not None:
            a, b = TASKS[task]
            ia, ib = CLASSES.index(a), CLASSES.index(b)
            m = np.isin(y, [ia, ib])
            Xp, Xb, y = Xp[m], Xb[m], (y[m] == ib).astype(int)
        if align:
            Xp, Xb = euclidean_align(Xp), align_bank(Xb)
        out[f"{date}/{subj}"] = (Xp, Xb, y)
    return out


def _pick(name, Xp, Xb):
    return Xb if D.is_bank(name) else Xp


# ============================ experiments ==========================
def run_loso(data, pipes, n_cal=0, seed=0):
    """Leave-one-subject-out; optionally seed with n_cal target trials."""
    sids = list(data)
    rows, preds = [], {}
    for held in sids:
        Xp_te, Xb_te, y_te = data[held]
        src = [s for s in sids if s != held]
        Xp_tr = np.concatenate([data[s][0] for s in src])
        Xb_tr = np.concatenate([data[s][1] for s in src])
        y_tr = np.concatenate([data[s][2] for s in src])

        if n_cal > 0:
            rng = np.random.RandomState(seed)
            cal = []
            for c in np.unique(y_te):                    # class-balanced, earliest
                cal += list(np.where(y_te == c)[0][:n_cal // len(np.unique(y_te))])
            cal = np.array(sorted(cal))
            rest = np.setdiff1d(np.arange(len(y_te)), cal)
            Xp_tr = np.concatenate([Xp_tr, Xp_te[cal]])
            Xb_tr = np.concatenate([Xb_tr, Xb_te[cal]])
            y_tr = np.concatenate([y_tr, y_te[cal]])
            Xp_te, Xb_te, y_te = Xp_te[rest], Xb_te[rest], y_te[rest]

        for name in pipes:
            Xtr, Xte = _pick(name, Xp_tr, Xb_tr), _pick(name, Xp_te, Xb_te)
            mdl = D.get(name).fit(Xtr, y_tr)
            p = mdl.predict(Xte)
            rows.append(dict(subject=held, pipe=name, acc=float((p == y_te).mean()),
                             n_test=len(y_te), n_train=len(y_tr)))
            preds.setdefault(name, []).append((held, y_te, p))
    return rows, preds


def run_within(data, pipes, seed=42):
    rows, preds = [], {}
    for sid, (Xp, Xb, y) in data.items():
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        for name in pipes:
            X = _pick(name, Xp, Xb)
            p = np.empty_like(y)
            for tr, te in cv.split(X, y):
                p[te] = D.get(name).fit(X[tr], y[tr]).predict(X[te])
            rows.append(dict(subject=sid, pipe=name, acc=float((p == y).mean()),
                             n_test=len(y), n_train=int(len(y) * 0.8)))
            preds.setdefault(name, []).append((sid, y, p))
    return rows, preds


# ============================== figures ============================
def plot_transfer(df, task, chance):
    piv = df.pivot_table(index="subject", columns="regime", values="acc")
    order = ["within", "loso", "loso+cal20", "loso+cal40"]
    piv = piv[[c for c in order if c in piv.columns]]
    x = np.arange(len(piv))
    w = 0.8 / piv.shape[1]
    colors = ["#6b7280", "#1f6feb", "#bc4c00", "#116329"]
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(piv), 3.8), constrained_layout=True)
    for j, col in enumerate(piv.columns):
        ax.bar(x + (j - (piv.shape[1] - 1) / 2) * w, piv[col] * 100, w,
               label=col, color=colors[j % len(colors)])
    ax.axhline(chance * 100, color=INK, lw=1, ls="--")
    ax.set_xticks(x, [s.split("/")[1] for s in piv.index], rotation=0, fontsize=8)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncols=4, fontsize=8, loc="upper left")
    means = "   ".join(f"{c}: {piv[c].mean()*100:.1f}%" for c in piv.columns)
    ax.set_title(f"{task} — cross-subject transfer with Euclidean Alignment "
                 f"(best pipeline per regime)\nmeans → {means}", loc="left", fontsize=10)
    fig.savefig(os.path.join(R_DIR, f"fig_transfer_{task}.png"), bbox_inches="tight")
    plt.close(fig)


def plot_conf(preds, task, name, tag):
    labels = CLASSES if TASKS[task] is None else list(TASKS[task])
    n = len(labels)
    agg = np.zeros((n, n))
    for _, y, p in preds[name]:
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
    ax.set_title(f"{task} · {tag} · {name}\npooled accuracy "
                 f"{np.trace(agg)/agg.sum()*100:.1f}%", loc="left", fontsize=10)
    fig.savefig(os.path.join(R_DIR, f"fig_confusion_transfer_{task}_{tag}.png"),
                bbox_inches="tight")
    plt.close(fig)


# ================================ main =============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=8.0)
    ap.add_argument("--hi", type=float, default=30.0)
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=4.5)
    ap.add_argument("--ref", default="car")
    ap.add_argument("--tasks", default="SvW,RvS,RvW,3cls")
    ap.add_argument("--pipes", default="csp_lda,ts_lr,acm_d5_10_ts_lr,fb_ts_lr,fgmdm")
    a = ap.parse_args()
    cfg = dict(lo=a.lo, hi=a.hi, tmin=a.tmin, tmax=a.tmax, ref=a.ref)
    pipes = a.pipes.split(",")

    all_rows = []
    for task in a.tasks.split(","):
        chance = 1 / (3 if task == "3cls" else 2)
        print(f"\n===== {task} (chance {chance*100:.0f}%) =====", flush=True)
        t0 = time.time()

        raw = build(task, cfg, align=False)
        ali = build(task, cfg, align=True)

        regimes = {}
        r, p_w = run_within(raw, pipes)
        regimes["within"] = (r, p_w)
        r, _ = run_within(ali, pipes)
        regimes["within+EA"] = (r, None)
        r, p_l = run_loso(raw, pipes)
        regimes["loso_noEA"] = (r, None)
        r, p_lea = run_loso(ali, pipes)
        regimes["loso"] = (r, p_lea)
        r, _ = run_loso(ali, pipes, n_cal=20)
        regimes["loso+cal20"] = (r, None)
        r, p_c40 = run_loso(ali, pipes, n_cal=40)
        regimes["loso+cal40"] = (r, p_c40)

        rows = []
        for reg, (rr, _) in regimes.items():
            for d in rr:
                rows.append({**d, "regime": reg, "task": task})
        df = pd.DataFrame(rows)
        all_rows.extend(rows)

        summ = df.groupby(["regime", "pipe"])["acc"].mean().unstack()
        print((summ * 100).round(1).to_string())
        best = {reg: summ.loc[reg].idxmax() for reg in summ.index}
        print("best pipe per regime:", best)

        # figure uses, per regime, that regime's own best pipeline
        sel = pd.concat([df[(df.regime == reg) & (df.pipe == best[reg])]
                         for reg in ["within", "loso", "loso+cal20", "loso+cal40"]])
        plot_transfer(sel, task, chance)
        plot_conf(p_w, task, best["within"], "within")
        plot_conf(p_lea, task, best["loso"], "loso")
        plot_conf(p_c40, task, best["loso+cal40"], "loso+cal40")
        print(f"({time.time()-t0:.0f}s)", flush=True)

    out = pd.DataFrame(all_rows)
    for k, v in cfg.items():
        out[k] = v
    out.to_csv(os.path.join(R_DIR, "transfer_scores.csv"), index=False)
    print("\nwrote", os.path.join(R_DIR, "transfer_scores.csv"))


if __name__ == "__main__":
    main()
