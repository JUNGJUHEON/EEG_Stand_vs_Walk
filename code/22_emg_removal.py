# -*- coding: utf-8 -*-
"""
Stage 22 — does removing EMG change anything?

WHAT WAS AND WAS NOT DONE BEFORE THIS
--------------------------------------
The pipeline so far handles muscle contamination three ways: a 8-30 Hz
band-pass (which excludes EMG above 40 Hz but NOT the 20-30 Hz component that
Goncharova et al. call the EMG beta rhythm), amplitude/variance trial
rejection (which catches bursts, not tonic muscle tone), and a surface
Laplacian tested in stage 18 (a spatial high-pass that suppresses the widely
distributed peripheral sources; it gave no improvement, all p > .37).

ICA was not attempted and should not be: stage 19 measures the effective
dimensionality of this montage at 1.49, and a 1.5-dimensional space cannot be
decomposed into separable "brain" and "muscle" subspaces.  Standard practice
requires >= 32 channels.

What was NOT done, without a good reason, is a decomposition that works at low
channel counts.  This stage does two.

METHOD 1 — NUISANCE REGRESSION
    Per trial, compute an EMG index (30-45 Hz log power, mean over channels).
    Project that direction out of the tangent-space feature vector:

        s_clean = s - b * emg          with b fit inside the training fold

    Crude - it cannot touch the 20-30 Hz component - but it fights no
    dimensionality problem and the removed direction is interpretable.

METHOD 2 — BSS-CCA  (De Clercq et al., 2006)
    EMG is close to temporally white; EEG is autocorrelated.  Canonical
    correlation between X(t) and X(t-1) yields components ordered by
    autocorrelation.  The least autocorrelated are muscle; zero them and
    reconstruct.  Needs no higher-order statistics, so it degrades more
    gracefully than ICA at eight channels - though eight is still marginal,
    and over-removal would destroy real signal.  Both risks are measured
    below by what happens to the contrasts that DO carry signal.

PRE-REGISTERED READING OF THE OUTCOME  (fixed before running; all reported)

    Stand-vs-Walk UNCHANGED
        -> the null is information-limited, not noise-limited.  Strengthens
           Sections III-D and III-H.
    Stand-vs-Walk RISES materially (> ~3 points, and topographic ratio moves
    off the noise floor)
        -> the null was partly an SNR problem.  Sections III-D and III-H must
           be revised, and this becomes a positive methodological result.
    P06 FALLS
        -> confirms Section III-E: that participant's 67% was muscle.
    Rest-vs-Walk and Rest-vs-Stand FALL substantially
        -> over-removal; the method is destroying signal and its output on
           Stand-vs-Walk cannot be interpreted.

Output: results/emg_removal.csv, fig_emg_removal.png
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
from scipy.linalg import eigh
from scipy.signal import welch
from scipy.stats import ttest_rel

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import steam_io as sio        # noqa: E402

R_DIR = sio.RESULTS_DIR
PAIRS = {"SvW": (1, 2), "RvW": (0, 2), "RvS": (0, 1)}
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
BLUE, ORANGE, RED = "#1f6feb", "#bc4c00", "#b3261e"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.dpi": 190,
})


# ------------------------------ method 2 --------------------------------
def bss_cca_clean(X, n_remove=2):
    """
    Remove the least autocorrelated components of one trial.

    X : (channels, time).  Returns the reconstruction with the n_remove
    lowest-autocorrelation CCA components zeroed.
    """
    C, T = X.shape
    Xc = X - X.mean(1, keepdims=True)
    Y = np.hstack([Xc[:, 1:], Xc[:, -1:]])          # lag-1 copy
    Cxx = Xc @ Xc.T / T
    Cyy = Y @ Y.T / T
    Cxy = Xc @ Y.T / T
    eps = 1e-10 * np.trace(Cxx) / C
    Cxx += eps * np.eye(C)
    Cyy += eps * np.eye(C)
    # generalised eigenproblem for the CCA directions on X
    M = np.linalg.solve(Cxx, Cxy @ np.linalg.solve(Cyy, Cxy.T))
    w, V = eigh((M + M.T) / 2)                      # ascending: w = rho^2
    order = np.argsort(w)[::-1]                     # most autocorrelated first
    V = V[:, order]
    S = V.T @ Xc                                    # components
    S[-n_remove:] = 0                               # drop the whitest
    return np.linalg.pinv(V.T) @ S


def emg_index(X_gamma):
    """Per-trial EMG proxy: log variance of the 30-45 Hz band, mean over channels.

    Taken from the SAME filter bank as the analysis band, so the rejection mask
    is shared and the rows stay aligned - loading a second epoch set with its
    own rejection desynchronises the trials (150 vs 144 in the first attempt).
    """
    return np.log10(X_gamma.var(axis=2) + 1e-30).mean(1)


# ------------------------------ evaluation ------------------------------
def cv_accuracy(X, y, emg=None, mode="raw"):
    """5-fold CV.  mode in {'raw', 'regress', 'cca'}; all fitting inside folds."""
    if mode == "cca":
        X = np.stack([bss_cca_clean(x) for x in X])
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    accs = []
    for tr, te in cv.split(X, y):
        cov = Covariances("oas")
        Ctr, Cte = cov.fit_transform(X[tr]), cov.transform(X[te])
        ts = TangentSpace().fit(Ctr)
        Str, Ste = ts.transform(Ctr), ts.transform(Cte)
        if mode == "regress":
            r = LinearRegression().fit(emg[tr].reshape(-1, 1), Str)
            Str = Str - r.predict(emg[tr].reshape(-1, 1))
            Ste = Ste - r.predict(emg[te].reshape(-1, 1))
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(Str, y[tr])
        accs.append(clf.score(Ste, y[te]))
    return float(np.mean(accs))


def topo_ratio(ep, ca, cb, rng, n_split=200, mode="raw"):
    """Section III-H.1 ratio, optionally on EMG-cleaned data."""
    X = ep["X"]
    if mode == "cca":
        X = np.stack([bss_cca_clean(x) for x in X])
    f, P = welch(X, fs=ep["fs"], nperseg=int(ep["fs"]), axis=-1)
    sel = (f >= 8) & (f <= 13)
    lp = np.log10(P[:, :, sel].mean(-1) + 1e-30)
    y = ep["y"]
    ia, ib = np.where(y == ca)[0], np.where(y == cb)[0]
    if min(len(ia), len(ib)) < 8:
        return np.nan
    w, b = [], []
    for _ in range(n_split):
        a, c = rng.permutation(ia), rng.permutation(ib)
        A1, A2 = lp[a[:len(a)//2]].mean(0), lp[a[len(a)//2:]].mean(0)
        B1, B2 = lp[c[:len(c)//2]].mean(0), lp[c[len(c)//2:]].mean(0)
        w.append((np.linalg.norm(A1-A2) + np.linalg.norm(B1-B2)) / 2)
        b.append((np.linalg.norm(A1-B1) + np.linalg.norm(A2-B2)) / 2)
    return float(np.mean(b) / np.mean(w))


def main():
    rng = np.random.RandomState(0)
    rows = []
    print("=" * 78)
    print("EMG REMOVAL — does cleaning change the conclusions?")
    print("  raw      : pipeline as used throughout")
    print("  regress  : 30-45 Hz nuisance direction projected out (in-fold)")
    print("  cca      : BSS-CCA, two least autocorrelated components removed")
    print("=" * 78)

    for date, subj in sio.list_subjects():
        sid = f"{date}/{subj}"
        bk = sio.get_bank(date, subj, bands=((8.0, 30.0), (30.0, 45.0)),
                          tmin=0.5, tmax=4.5, ref="none", broad=(1.0, 45.0))
        ep = dict(X=bk["Xb"][0], y=bk["y"], fs=bk["fs"],
                  ch_names=bk["ch_names"])
        emg = emg_index(bk["Xb"][1])
        y_all = bk["y"]
        line = f"  {sid:14s}"
        for pname, (ca, cb) in PAIRS.items():
            m = np.isin(y_all, [ca, cb])
            X, y = ep["X"][m], y_all[m]
            e = emg[m]
            if len(np.unique(y)) < 2:
                continue
            accs = {md: cv_accuracy(X, y, e, md)
                    for md in ("raw", "regress", "cca")}
            tr = {md: topo_ratio(ep, ca, cb, np.random.RandomState(0), mode=md)
                  for md in ("raw", "cca")}
            rows.append(dict(subject=sid, pair=pname,
                             acc_raw=accs["raw"], acc_reg=accs["regress"],
                             acc_cca=accs["cca"],
                             ratio_raw=tr["raw"], ratio_cca=tr["cca"]))
            if pname == "SvW":
                line += (f"  SvW {accs['raw']*100:5.1f} → reg {accs['regress']*100:5.1f}"
                         f" / cca {accs['cca']*100:5.1f}")
        print(line)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(R_DIR, "emg_removal.csv"), index=False)

    print("\n" + "=" * 78)
    print("GROUP RESULT  (paired t-test vs raw, n = 10)")
    print("=" * 78)
    summary = []
    for pname in PAIRS:
        g = df[df.pair == pname]
        base = g.acc_raw.values * 100
        print(f"\n  {pname}   raw {base.mean():5.1f} %")
        for md, col in (("regress", "acc_reg"), ("cca", "acc_cca")):
            v = g[col].values * 100
            t, p = ttest_rel(v, base)
            print(f"    {md:8s} {v.mean():5.1f} %   Δ {v.mean()-base.mean():+5.2f}"
                  f"   t={t:+5.2f}  p={p:.4f}")
            summary.append(dict(pair=pname, method=md, mean=v.mean(),
                                delta=v.mean()-base.mean(), t=t, p=p))
        rr, rc = g.ratio_raw.values, g.ratio_cca.values
        t, p = ttest_rel(rc, rr)
        print(f"    topo ratio  {rr.mean():.2f} → {rc.mean():.2f}"
              f"   t={t:+5.2f}  p={p:.4f}")
    pd.DataFrame(summary).to_csv(
        os.path.join(R_DIR, "emg_removal_tests.csv"), index=False)

    # ------------------------- P06 specifically -----------------------
    p06 = df[(df.subject.str.contains("S06")) & (df.pair == "SvW")]
    if len(p06):
        r = p06.iloc[0]
        print("\n" + "=" * 78)
        print("PARTICIPANT P06 (the one Section III-E rejects as muscle)")
        print(f"  Stand vs Walk :  raw {r.acc_raw*100:.1f} %"
              f"  →  regress {r.acc_reg*100:.1f} %"
              f"  →  cca {r.acc_cca*100:.1f} %")
        print("  prediction if that 67 % was muscle: it should FALL")

    # ------------------------------ figure ----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9),
                             constrained_layout=True, sharey=True)
    for ax, pname in zip(axes, PAIRS):
        g = df[df.pair == pname]
        for i, (lab, col) in enumerate([("raw", "acc_raw"),
                                        ("regress", "acc_reg"),
                                        ("BSS-CCA", "acc_cca")]):
            v = g[col].values * 100
            ax.bar(i, v.mean(), .6, color=BLUE if i == 0 else ORANGE, zorder=2)
            ax.scatter(np.full(len(v), i), v, s=15, color="white",
                       edgecolor=INK, linewidth=.6, zorder=3)
        ax.axhline(50, color=INK, lw=1.4, ls="--")
        ax.set_xticks(range(3), ["raw", "regress", "BSS-CCA"], fontsize=8.4)
        ax.set_title(pname, loc="left")
    axes[0].set_ylabel("accuracy (%)")
    fig.suptitle("Does removing EMG change the result?  "
                 "(dashed line = chance)",
                 ha="left", x=.006, fontsize=11.4, fontweight="bold")
    fig.savefig(os.path.join(R_DIR, "fig_emg_removal.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("\nwrote", os.path.join(R_DIR, "fig_emg_removal.png"))


if __name__ == "__main__":
    main()
