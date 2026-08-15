# -*- coding: utf-8 -*-
"""
Stage 9 — the two statistics the write-up actually needs.

(A) PER-SUBJECT SIGNIFICANCE.  The group means hide a wide spread (Rest vs
    Stand ranges 48-88 %).  A pooled binomial over all trials can be carried
    by one or two strong participants, so it cannot answer "how many of the
    ten actually worked?".  This runs a 1000-permutation label null per
    subject per task and reports the count that survive Holm correction.

(B) PIPELINE COMPARISON.  The claim "Riemannian beats the CSP/FBCSP family"
    rests on a 3-4 point gap in mean k-fold accuracy.  Fold accuracies are
    correlated (training sets overlap), so a naive paired t-test over folds
    is anticonservative and will manufacture significance.  This uses the
    Nadeau-Bengio corrected resampled t-test, which inflates the variance by
    (1/k + test_frac/train_frac) to account for that overlap, applied over
    subjects.

Output: results/stats_permutation.csv, results/stats_pipecmp.csv,
        fig_subject_significance.png
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
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import StratifiedKFold, permutation_test_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import decoders as D          # noqa: E402
import steam_io as sio        # noqa: E402

R_DIR = sio.RESULTS_DIR
CFG = dict(lo=8.0, hi=30.0, tmin=0.5, tmax=4.5, ref="none")
PIPE = "fb_ts_lr"
N_PERM = 1000
TASKS = {"RvS": ("Rest", "Stand"), "RvW": ("Rest", "Walk"),
         "SvW": ("Stand", "Walk"), "3cls": None}
INK, GRID = "#1f2328", "#e5e7eb"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#6b7280", "ytick.color": "#6b7280",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.dpi": 130,
})


def holm(p):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def perm_cell(sid, task, Xb, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    score, null, p = permutation_test_score(
        D.get(PIPE), Xb, y, scoring="accuracy", cv=cv,
        n_permutations=N_PERM, random_state=0, n_jobs=1)
    return dict(subject=sid, task=task, acc=float(score), p=float(p),
                null_mean=float(np.mean(null)),
                null_p95=float(np.percentile(null, 95)), n=len(y))


def nadeau_bengio(diff, n_train, n_test):
    """
    Corrected resampled t-test (Nadeau & Bengio 2003; Bouckaert & Frank 2004).

    Standard paired t on resampled folds ignores that training sets overlap,
    understating the variance.  The correction multiplies the variance by
    (1/k + n_test/n_train).
    """
    d = np.asarray(diff, float)
    k = len(d)
    if k < 2:
        return np.nan, np.nan
    var = d.var(ddof=1)
    if var <= 0:
        return np.nan, np.nan
    t = d.mean() / np.sqrt(var * (1.0 / k + n_test / n_train))
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))


def main():
    subs = sio.list_subjects()

    # ---------------- (A) per-subject permutation ----------------
    print(f"== (A) per-subject permutation, {N_PERM} perms, pipeline={PIPE} ==")
    jobs = []
    for date, subj in subs:
        _, Xb, y = sio.get_task_arrays(date, subj, **CFG)
        sid = f"{date}/{subj}"
        for task, pair in TASKS.items():
            if pair is None:
                jobs.append((sid, task, Xb, y))
            else:
                ia = sio.CLASS_NAMES.index(pair[0])
                ib = sio.CLASS_NAMES.index(pair[1])
                m = np.isin(y, [ia, ib])
                jobs.append((sid, task, Xb[m], (y[m] == ib).astype(int)))

    rows = Parallel(n_jobs=6)(delayed(perm_cell)(*j) for j in jobs)
    df = pd.DataFrame(rows)
    # correct within task, across the ten subjects
    df["p_holm"] = np.nan
    for task, g in df.groupby("task"):
        df.loc[g.index, "p_holm"] = holm(g["p"].values)
    df.to_csv(os.path.join(R_DIR, "stats_permutation.csv"), index=False)

    for task in TASKS:
        g = df[df.task == task].sort_values("acc", ascending=False)
        chance = 1 / (3 if task == "3cls" else 2)
        n_sig = int((g["p_holm"] < 0.05).sum())
        n_raw = int((g["p"] < 0.05).sum())
        print(f"\n  --- {task} (chance {chance*100:.0f} %) : "
              f"{n_sig}/10 significant after Holm, {n_raw}/10 uncorrected ---")
        for _, r in g.iterrows():
            mark = " *" if r["p_holm"] < 0.05 else ("  ." if r["p"] < 0.05 else "")
            print(f"    {r['subject']:14s} {r['acc']*100:5.1f}%  "
                  f"null95={r['null_p95']*100:4.1f}%  p={r['p']:.4f}  "
                  f"p_holm={r['p_holm']:.3f}{mark}")

    print("\n  summary (* = Holm-significant, . = uncorrected only):")
    summ = (df.assign(sig=df.p_holm < 0.05, raw=df.p < 0.05)
              .groupby("task")[["sig", "raw"]].sum().astype(int))
    print(summ.to_string())

    # ---------------- (B) pipeline comparison ----------------
    print("\n== (B) Riemannian vs CSP/FBCSP, corrected resampled t-test ==")
    fs = pd.read_csv(os.path.join(R_DIR, "final_scores.csv"))
    kf = fs[fs.kind == "kfold"]
    n_test, n_train = 0.2, 0.8            # 5-fold
    out = []
    contrasts = [("fb_ts_lr", "fbcsp_mibif_lda"), ("fb_ts_lr", "csp_lda"),
                 ("ts_lr", "csp_lda"), ("fb_ts_lr", "fb_logvar_lda")]
    for task in ["RvS", "RvW", "SvW", "3cls"]:
        for a, b in contrasts:
            pa = kf[(kf.task == task) & (kf["pipe"] == a)].set_index("subject")["acc"]
            pb = kf[(kf.task == task) & (kf["pipe"] == b)].set_index("subject")["acc"]
            common = pa.index.intersection(pb.index)
            d = (pa.loc[common] - pb.loc[common]).values
            t, p = nadeau_bengio(d, n_train, n_test)
            tn, pn = stats.ttest_rel(pa.loc[common], pb.loc[common])
            out.append(dict(task=task, a=a, b=b, mean_diff=float(d.mean()),
                            t_corrected=t, p_corrected=p,
                            p_naive=float(pn), n_subjects=len(common)))
    cmp = pd.DataFrame(out)
    cmp.to_csv(os.path.join(R_DIR, "stats_pipecmp.csv"), index=False)
    print(f"  {'task':6s} {'A':16s} vs {'B':16s} {'diff':>7s} "
          f"{'p_corr':>8s} {'p_naive':>8s}")
    for _, r in cmp.iterrows():
        flag = " *" if r["p_corrected"] < 0.05 else ""
        print(f"  {r['task']:6s} {r['a']:16s} vs {r['b']:16s} "
              f"{r['mean_diff']*100:+6.1f}pt {r['p_corrected']:8.3f} "
              f"{r['p_naive']:8.4f}{flag}")
    print("\n  note: p_naive is the UNCORRECTED paired t-test, shown only to "
          "make the size of the anticonservatism visible. Quote p_corr.")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), constrained_layout=True,
                             sharey=True)
    for ax, task in zip(axes, ["RvS", "RvW", "SvW", "3cls"]):
        g = df[df.task == task].sort_values("acc")
        chance = 1 / (3 if task == "3cls" else 2)
        x = np.arange(len(g))
        cols = ["#1f6feb" if p < 0.05 else "#c9d1d9" for p in g["p_holm"]]
        ax.barh(x, g["acc"] * 100, color=cols)
        ax.plot(g["null_p95"] * 100, x, "k|", ms=9)
        ax.axvline(chance * 100, color=INK, lw=1, ls=":")
        ax.set_yticks(x, [s.split("/")[1] for s in g["subject"]], fontsize=8)
        ax.set_xlabel("accuracy (%)")
        n_sig = int((g["p_holm"] < 0.05).sum())
        ax.set_title(f"{task}  —  {n_sig}/10 significant", loc="left")
    fig.suptitle("Per-participant significance vs a 1000-permutation label null "
                 "(filled = Holm-corrected p < .05; tick = that participant's "
                 "own null 95th percentile)", ha="left", x=0.01, fontsize=11)
    fig.savefig(os.path.join(R_DIR, "fig_subject_significance.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("\nwrote", os.path.join(R_DIR, "fig_subject_significance.png"))


if __name__ == "__main__":
    main()
