# -*- coding: utf-8 -*-
"""
Stage 8 — is Stand-vs-Walk decodable in a *trained* subject?

S06 reached 70.0 % nested on Stand vs Walk while the group mean is 53.1 %,
and S06 is reported to have had prior MI experience.  That suggests the
contrast becomes decodable with training.  It is a good hypothesis, but the
current data can only support it if three things hold, and this script tests
all three.

  (1) EXISTENCE.  Is S06's accuracy individually above chance, or is it just
      the maximum of ten noisy draws?  With ten subjects at chance, the
      expected best-of-ten is well above 50 %, so "S06 is highest" is not
      evidence of anything.  Tested with a label-permutation null (the whole
      CV is re-run on shuffled labels), giving a per-subject empirical p.

  (2) SPECIFICITY.  Does S06 lead only on Stand vs Walk, or on every task and
      also on plain data-quality measures?  A subject who simply produced the
      cleanest recording will top every leaderboard without any training
      effect.  S06 has the lowest channel amplitude and zero saturation in
      the whole cohort, so this is a live alternative.

  (3) CONFOUND.  S06 is one of three subjects whose 150 trials are stitched
      from two runs (MI + MI2) via KEEP_RULES.  Class balance is exact within
      each run, so a run-vs-class confound is ruled out, but the check is
      re-printed here so the claim is not taken on trust.

None of this can establish a training effect from n = 1 selected *because*
it was the best — that is selection on the outcome.  What it can establish is
whether the contrast is decodable in at least one person, which is the
existence proof a follow-up training study would be built on.

Output: results/s06_hypothesis.csv, fig_s06_permutation.png
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
from sklearn.model_selection import StratifiedKFold, permutation_test_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import decoders as D          # noqa: E402
import steam_io as sio        # noqa: E402

R_DIR = sio.RESULTS_DIR
CFG = dict(lo=8.0, hi=30.0, tmin=0.5, tmax=4.5, ref="none")
PIPE = "fb_ts_lr"
N_PERM = 1000
INK, GRID = "#1f2328", "#e5e7eb"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#6b7280", "ytick.color": "#6b7280",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.dpi": 130,
})


def perm_one(sid, Xb, y, n_perm=N_PERM, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    score, perm_scores, p = permutation_test_score(
        D.get(PIPE), Xb, y, scoring="accuracy", cv=cv,
        n_permutations=n_perm, random_state=seed, n_jobs=1)
    return dict(subject=sid, acc=float(score), p=float(p),
                null_mean=float(np.mean(perm_scores)),
                null_p95=float(np.percentile(perm_scores, 95)),
                n=len(y)), perm_scores


def main():
    subs = sio.list_subjects()

    # ---------- (3) run x class balance, re-verified ----------
    print("== (3) run x class balance for the multi-run subjects ==")
    for date, subj in subs:
        ep = sio.get_epochs(date, subj, l_freq=8, h_freq=30,
                            tmin=0.5, tmax=4.5, ref="none", reject=False)
        runs = np.unique(ep["run"])
        if len(runs) == 1:
            continue
        for r in runs:
            m = ep["run"] == r
            cnt = {n: int((ep["y"][m] == i).sum())
                   for i, n in enumerate(sio.CLASS_NAMES)}
            print(f"  {date}/{subj} {r}: {cnt}")

    # ---------- (1) permutation test, Stand vs Walk ----------
    print(f"\n== (1) label-permutation test, Stand vs Walk, {N_PERM} perms ==")
    data = []
    for date, subj in subs:
        _, Xb, y = sio.get_task_arrays(date, subj, **CFG)
        ia, ib = sio.CLASS_NAMES.index("Stand"), sio.CLASS_NAMES.index("Walk")
        m = np.isin(y, [ia, ib])
        data.append((f"{date}/{subj}", Xb[m], (y[m] == ib).astype(int)))

    out = Parallel(n_jobs=6)(delayed(perm_one)(s, X, yy) for s, X, yy in data)
    rows = [r for r, _ in out]
    nulls = {r["subject"]: ns for (r, ns) in out}
    df = pd.DataFrame(rows).sort_values("acc", ascending=False)

    # Holm-Bonferroni across the ten subjects
    df = df.reset_index(drop=True)
    order = np.argsort(df["p"].values)
    adj = np.empty(len(df))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(df) - rank) * df["p"].values[i])
        adj[i] = min(1.0, running)
    df["p_holm"] = adj

    print(f"  {'subject':14s} {'acc':>6s} {'null mean':>10s} {'null p95':>9s} "
          f"{'p':>7s} {'p_holm':>7s}")
    for _, r in df.iterrows():
        star = "  *" if r["p_holm"] < 0.05 else ""
        print(f"  {r['subject']:14s} {r['acc']*100:5.1f}% {r['null_mean']*100:9.1f}% "
              f"{r['null_p95']*100:8.1f}% {r['p']:7.4f} {r['p_holm']:7.3f}{star}")
    n_sig = int((df["p_holm"] < 0.05).sum())
    print(f"  -> {n_sig}/10 subjects individually significant after "
          f"Holm correction for 10 tests")

    # ---------- (2) specificity: is S06 best at everything? ----------
    print("\n== (2) is S06's lead specific to Stand vs Walk? ==")
    fs = pd.read_csv(os.path.join(R_DIR, "final_scores.csv"))
    nested = fs[fs.kind == "nested"]
    ranks = {}
    for task in ["SvW", "RvS", "RvW", "3cls"]:
        t = nested[nested.task == task].sort_values("acc", ascending=False)
        t = t.reset_index(drop=True)
        for i, r in t.iterrows():
            ranks.setdefault(r["subject"], {})[task] = i + 1
    q = pd.read_csv(os.path.join(R_DIR, "quality.csv"))
    qual = q.groupby("subject").agg(max_std_uv=("std_uv", "max"),
                                    max_clip=("clip_pct", "max"))
    print(f"  {'subject':14s} " + " ".join(f"{t:>5s}" for t in
                                           ["SvW", "RvS", "RvW", "3cls"])
          + f" {'maxStd(uV)':>11s} {'clip%':>6s}")
    for sid in sorted(ranks, key=lambda s: ranks[s]["SvW"]):
        rr = ranks[sid]
        print(f"  {sid:14s} " + " ".join(f"{rr[t]:5d}" for t in
                                         ["SvW", "RvS", "RvW", "3cls"])
              + f" {qual.loc[sid,'max_std_uv']:11.0f} "
                f"{qual.loc[sid,'max_clip']:6.1f}")

    # correlation of SvW accuracy with recording cleanliness
    acc = nested[nested.task == "SvW"].set_index("subject")["acc"]
    common = acc.index.intersection(qual.index)
    from scipy.stats import spearmanr
    for col in ("max_std_uv", "max_clip"):
        rho, pv = spearmanr(acc.loc[common], qual.loc[common, col])
        print(f"  Spearman(SvW nested acc, {col}) = {rho:+.2f}  p={pv:.3f}")

    df.to_csv(os.path.join(R_DIR, "s06_hypothesis.csv"), index=False)

    # ---------- figure ----------
    top = df.iloc[0]["subject"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True,
                             width_ratios=[1, 1.3])
    ns = nulls[top]
    axes[0].hist(np.asarray(ns) * 100, bins=30, color="#c9d1d9",
                 edgecolor="white", linewidth=0.5)
    axes[0].axvline(df.iloc[0]["acc"] * 100, color="#bc4c00", lw=2.5,
                    label=f"{top}  {df.iloc[0]['acc']*100:.1f}%")
    axes[0].axvline(np.percentile(ns, 95) * 100, color=INK, lw=1, ls="--",
                    label="null 95th pct")
    axes[0].set_xlabel("accuracy under permuted labels (%)")
    axes[0].set_ylabel("permutations")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title(f"{top} — permutation null, Stand vs Walk", loc="left")

    d2 = df.sort_values("acc")
    x = np.arange(len(d2))
    cols = ["#1f6feb" if p < 0.05 else "#c9d1d9" for p in d2["p_holm"]]
    axes[1].barh(x, d2["acc"] * 100, color=cols)
    axes[1].plot(d2["null_p95"] * 100, x, "k|", ms=10, lw=1)
    axes[1].axvline(50, color=INK, lw=1, ls=":")
    axes[1].set_yticks(x, [s.split("/")[1] for s in d2["subject"]], fontsize=8)
    axes[1].set_xlabel("Stand vs Walk accuracy (%)")
    axes[1].set_title("filled = significant after Holm correction;\n"
                      "tick = that subject's own null 95th percentile",
                      loc="left", fontsize=9)
    fig.savefig(os.path.join(R_DIR, "fig_s06_permutation.png"), bbox_inches="tight")
    plt.close(fig)
    print("\nwrote", os.path.join(R_DIR, "fig_s06_permutation.png"))


if __name__ == "__main__":
    main()
