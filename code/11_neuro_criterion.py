# -*- coding: utf-8 -*-
"""
Stage 11 — does a PRE-REGISTERED neurophysiological criterion predict
Stand-vs-Walk decodability?

WHY THIS IS STRUCTURED BACKWARDS ON PURPOSE
-------------------------------------------
The tempting analysis is: take the participants who decode best, look at their
spectra, and describe what you find.  That is selection on the outcome, and it
is exactly how stage 8's "S06 is an existence proof" claim was produced and
then withdrawn in stage 10.  With ten participants there is always a top three,
and there is always something in their spectra to point at.

So the criterion below is defined from motor-imagery physiology ALONE and is
computed without ever looking at an accuracy number.  Only afterwards is it
correlated with decoding performance.  That makes it falsifiable: if the
broadband/global participants turn out to decode best, the criterion fails and
the honest conclusion is that the separability is not neural.

THE CRITERION
-------------
Motor imagery modulates sensorimotor rhythm amplitude (ERD/ERS).  That
phenomenon has three signatures an artifact does not share:

  M1  SPATIAL GRADIENT.  Cortical sources project unevenly onto the scalp.
      grad = (max|r| - min|r|) / max|r| over the eight electrodes, in 8-30 Hz.
      Cortical -> high (uneven).  Global arousal / scalp EMG -> near 0.

  M2  RHYTHM SPECIFICITY.  ERD is confined to mu/beta.  There is no
      sensorimotor rhythm at 2-8 Hz and none in the EMG range.
      rspec = mean r^2 in 8-30  /  mean r^2 in (2-8 and 30-45).
      Rhythm-driven -> > 1.  Broadband state change -> ~ 1.

  M3  CENTRAL FOCUS.  Lower-limb (foot) representation is medial.
      cfoc = mean|r| over {Cz, CPz, C3, C4, CP3, CP4} / mean|r| over {FC3, FC4}
      in 8-30 Hz.  Foot-area MI -> > 1.

  M4  EMG CLEANLINESS (a rejection flag, not a strength).
      emg = mean|r| in 30-45 / mean|r| in 8-30.  High -> muscle-contaminated.

PRE-REGISTERED PREDICTION, stated before any correlation is computed:
  if Stand-vs-Walk separability is neural, accuracy should correlate
  POSITIVELY with M1, M2 and M3, and NEGATIVELY with M4.

Output: results/neuro_criterion.csv, fig_neuro_criterion.png
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
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import steam_io as sio        # noqa: E402

R_DIR = sio.RESULTS_DIR
CH = sio.EEG_CHANNELS
CENTRAL = ["Cz", "CPz", "C3", "C4", "CP3", "CP4"]
FRONTAL = ["FC3", "FC4"]
INK, GRID = "#1f2328", "#e5e7eb"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#6b7280", "ytick.color": "#6b7280",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.dpi": 130,
})


def r_map(date, subj, ia, ib):
    """Point-biserial r between log spectral power and class, per ch x freq."""
    ep = sio.get_epochs(date, subj, l_freq=1.0, h_freq=45.0,
                        tmin=0.5, tmax=4.5, ref="none")
    m = np.isin(ep["y"], [ia, ib])
    y = (ep["y"][m] == ib).astype(float)
    f, P = welch(ep["X"][m], fs=ep["fs"], nperseg=int(ep["fs"]), axis=-1)
    sel = (f >= 2) & (f <= 45)
    f, Z = f[sel], np.log(P[:, :, sel] + 1e-30)
    zc, yc = Z - Z.mean(0), y - y.mean()
    r = (zc * yc[:, None, None]).sum(0) / (
        np.sqrt((zc ** 2).sum(0)) * np.sqrt((yc ** 2).sum()) + 1e-30)
    return f, r


def metrics(f, r):
    smr = (f >= 8) & (f <= 30)
    low = (f >= 2) & (f < 8)
    gam = (f > 30) & (f <= 45)

    prof = np.abs(r[:, smr]).mean(1)                       # |r| per electrode
    grad = float((prof.max() - prof.min()) / (prof.max() + 1e-12))

    out_mass = np.concatenate([r[:, low], r[:, gam]], axis=1) ** 2
    rspec = float((r[:, smr] ** 2).mean() / (out_mass.mean() + 1e-12))

    ci = [CH.index(c) for c in CENTRAL]
    fi = [CH.index(c) for c in FRONTAL]
    cfoc = float(prof[ci].mean() / (prof[fi].mean() + 1e-12))

    emg = float(np.abs(r[:, gam]).mean() / (np.abs(r[:, smr]).mean() + 1e-12))
    return dict(M1_gradient=grad, M2_rhythm_spec=rspec,
                M3_central_focus=cfoc, M4_emg_ratio=emg)


def main():
    pairs = {"SvW": (1, 2), "RvS": (0, 1), "RvW": (0, 2)}
    rows = []
    for date, subj in sio.list_subjects():
        sid = f"{date}/{subj}"
        for task, (ia, ib) in pairs.items():
            f, r = r_map(date, subj, ia, ib)
            rows.append(dict(subject=sid, task=task, **metrics(f, r)))
    met = pd.DataFrame(rows)

    # accuracies computed in earlier stages — joined only NOW
    fs = pd.read_csv(os.path.join(R_DIR, "final_scores.csv"))
    acc = (fs[fs.kind == "nested"][["subject", "task", "acc"]]
           .rename(columns={"acc": "nested_acc"}))
    df = met.merge(acc, on=["subject", "task"], how="left")
    df.to_csv(os.path.join(R_DIR, "neuro_criterion.csv"), index=False)

    print("=" * 78)
    print("PRE-REGISTERED PREDICTION: accuracy correlates + with M1, M2, M3")
    print("                           and - with M4, if the signal is neural.")
    print("=" * 78)

    for task in ["SvW", "RvS", "RvW"]:
        d = df[df.task == task].sort_values("nested_acc", ascending=False)
        print(f"\n### {task} " + "-" * 60)
        print(f"  {'subject':14s} {'acc':>6s} {'M1 grad':>8s} {'M2 rspec':>9s} "
              f"{'M3 cfoc':>8s} {'M4 emg':>7s}")
        for _, r in d.iterrows():
            print(f"  {r['subject']:14s} {r['nested_acc']*100:5.1f}% "
                  f"{r['M1_gradient']:8.2f} {r['M2_rhythm_spec']:9.2f} "
                  f"{r['M3_central_focus']:8.2f} {r['M4_emg_ratio']:7.2f}")
        print("  --- correlation with nested accuracy (Spearman, n=10) ---")
        for m, expect in [("M1_gradient", "+"), ("M2_rhythm_spec", "+"),
                          ("M3_central_focus", "+"), ("M4_emg_ratio", "-")]:
            rho, p = spearmanr(d["nested_acc"], d[m])
            ok = ("as predicted" if (rho > 0) == (expect == "+") else
                  "AGAINST prediction")
            print(f"    {m:18s} rho={rho:+.2f} p={p:.3f}   (expected {expect}) "
                  f"{ok}")

    # ---------------- figure ----------------
    d = df[df.task == "SvW"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), constrained_layout=True)
    specs = [("M1_gradient", "spatial gradient", "+"),
             ("M2_rhythm_spec", "rhythm specificity (8-30 vs rest)", "+"),
             ("M3_central_focus", "central focus", "+"),
             ("M4_emg_ratio", "EMG ratio (30-45 / 8-30)", "-")]
    for ax, (m, lab, exp) in zip(axes, specs):
        ax.scatter(d[m], d["nested_acc"] * 100, s=45, color="#1f6feb", zorder=3)
        for _, r in d.iterrows():
            ax.annotate(r["subject"].split("/")[1], (r[m], r["nested_acc"] * 100),
                        fontsize=6.5, xytext=(3, 3), textcoords="offset points",
                        color="#6b7280")
        rho, p = spearmanr(d["nested_acc"], d[m])
        ax.axhline(50, color=INK, lw=1, ls=":")
        ax.set_xlabel(lab, fontsize=8)
        ax.set_title(f"rho={rho:+.2f}, p={p:.2f}  (expected {exp})",
                     loc="left", fontsize=9)
    axes[0].set_ylabel("Stand vs Walk nested acc (%)")
    fig.suptitle("Does a pre-registered neurophysiological criterion predict "
                 "Stand-vs-Walk decodability?", ha="left", x=0.01, fontsize=11)
    fig.savefig(os.path.join(R_DIR, "fig_neuro_criterion.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("\nwrote", os.path.join(R_DIR, "fig_neuro_criterion.png"))


if __name__ == "__main__":
    main()
