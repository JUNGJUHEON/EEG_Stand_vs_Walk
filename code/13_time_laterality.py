# -*- coding: utf-8 -*-
"""
Stage 13 — does laterality ALTERNATE within a Walk trial?

Why this is a different question from stage 4c
----------------------------------------------
Stage 4c measured a laterality index averaged over the whole 4 s analysis
window and found +0.026 dB (p = .73) between Walk and Stand, and concluded
that the compound-limbs asymmetry is absent.  That conclusion was too strong,
because the measurement was blind to the structure the paradigm actually has.

The instruction for Walk is: imagine walking "starting with a right-foot step
and a left-arm swing ... right foot, left foot, two steps total".  So a single
Walk trial contains BOTH compound configurations in sequence:

    step 1   right leg + left arm    -> right-hemisphere dominant
    step 2   left leg  + right arm   -> left-hemisphere  dominant

Ma et al. put those two configurations in *separate classes*, so their
contrast survives trial averaging.  Here they are inside the same trial, so a
trial-averaged laterality index cancels them by construction.  +0.026 dB is
therefore consistent with "the asymmetry alternates and cancels", not only
with "there is no asymmetry".

Stand, by contrast, is a bilateral movement with no alternation.

Three tests, in increasing robustness to timing jitter
------------------------------------------------------
T1  PHASE-LOCKED.  Two steps in 5 s means the laterality index should be one
    sign in the first half of the trial and the other sign in the second.
    Statistic: LI(first half) - LI(second half), Walk vs Stand.
    Strongest test, but assumes participants followed the instructed timing.

T2  PHASE-FREE MAGNITUDE.  Even if step timing varies trial to trial, an
    alternating trial has a laterality index that MOVES more than a bilateral
    one.  Statistic: within-trial SD of LI(t).  Needs no alignment.

T3  DECODING.  Can Stand vs Walk be classified from the LI(t) trajectory
    alone (8 time bins)?  If the alternation carries information the previous
    covariance features destroyed, this is where it shows up.

T1 fixes the switch at 2.5 s and T3 needs the bins to mean the same thing in
every trial, so both assume the participant followed the instructed pace.
Three further tests drop that assumption entirely and allow a switch at ANY
time, and any number of switches:

T4  BEST SPLIT POINT.  For each trial take max over tau of
    |mean LI before tau - mean LI after tau|, tau free in [0.5, 4.5] s.
    This is the largest step-like transition anywhere in the trial.  The
    max-over-tau is biased upward by noise, but identically so for every
    class, so the Walk-minus-Stand comparison stays valid.

T5  SIGN CROSSINGS.  After 0.5 s smoothing, how often does LI(t) cross its
    own trial mean?  Detects several alternations at unknown times.

T6  LI MODULATION SPECTRUM.  Power of the LI(t) fluctuation in 0.2-1.0 Hz
    (one to five switches across a 5 s trial).  Completely phase-free: a
    periodic alternation shows up here whatever its timing.

Output: results/time_laterality.csv, fig_time_laterality.png
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
from scipy.signal import hilbert
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import steam_io as sio        # noqa: E402

R_DIR = sio.RESULTS_DIR
CH = sio.EEG_CHANNELS
RIGHT = ["FC4", "C4", "CP4"]
LEFT = ["FC3", "C3", "CP3"]
BANDS = {"mu 8-13": (8, 13), "beta 13-30": (13, 30)}
TMIN, TMAX = 0.0, 5.0            # the FULL imagery period, not the 0.5-4.5 crop
SMOOTH_S = 0.25
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
SER = {"Rest": "#6b7280", "Stand": "#1f6feb", "Walk": "#bc4c00"}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.dpi": 160,
})


def li_timecourse(date, subj, lo, hi):
    """Laterality index time course per trial: (n_trials, n_times), in dB."""
    ep = sio.get_epochs(date, subj, l_freq=lo, h_freq=hi,
                        tmin=TMIN, tmax=TMAX, ref="none")
    fs = ep["fs"]
    env = np.abs(hilbert(ep["X"], axis=-1)) ** 2          # power envelope
    k = max(1, int(round(SMOOTH_S * fs)))
    kern = np.ones(k) / k
    env = np.apply_along_axis(lambda v: np.convolve(v, kern, mode="same"),
                              -1, env)
    ri = [CH.index(c) for c in RIGHT]
    li = [CH.index(c) for c in LEFT]
    Rp = env[:, ri].mean(axis=1)
    Lp = env[:, li].mean(axis=1)
    LI = 10.0 * np.log10((Rp + 1e-30) / (Lp + 1e-30))
    t = np.arange(LI.shape[-1]) / fs + TMIN
    return LI, ep["y"], t


def free_structure(LI, fs):
    """
    Phase-free descriptors of LI(t), one value per trial.
    No assumption about WHEN a switch happens, or how many there are.
    """
    from scipy.signal import welch

    n, T = LI.shape
    dem = LI - LI.mean(axis=1, keepdims=True)

    # --- T4: largest step-like transition at any split point ---
    lo, hi = int(0.5 * fs), int(T - 0.5 * fs)
    csum = np.cumsum(LI, axis=1)
    taus = np.arange(lo, hi)
    left = csum[:, taus - 1] / taus                                # mean before
    right = (csum[:, -1][:, None] - csum[:, taus - 1]) / (T - taus)  # mean after
    best_split = np.abs(left - right).max(axis=1)

    # --- T5: crossings of the trial mean, after 0.5 s smoothing ---
    k = max(1, int(round(0.5 * fs)))
    kern = np.ones(k) / k
    sm = np.apply_along_axis(lambda v: np.convolve(v, kern, mode="same"), -1, dem)
    core = sm[:, k:-k] if T > 2 * k else sm
    crossings = (np.diff(np.sign(core), axis=1) != 0).sum(axis=1)

    # --- T6: LI modulation power, 0.2-1.0 Hz ---
    f, P = welch(dem, fs=fs, nperseg=min(T, int(5 * fs)), axis=-1)
    band = (f >= 0.2) & (f <= 1.0)
    mod_power = np.log10(P[:, band].mean(axis=1) + 1e-30)

    return best_split, crossings.astype(float), mod_power


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    rows, curves = [], {}
    tests = []          # (band, test, direction_predicted, diff, p)
    subs = sio.list_subjects()

    for bname, (lo, hi) in BANDS.items():
        print(f"\n{'='*70}\n### {bname}\n{'='*70}")
        per_sub = {}
        for date, subj in subs:
            LI, y, t = li_timecourse(date, subj, lo, hi)
            sid = f"{date}/{subj}"
            per_sub[sid] = {}
            half = len(t) // 2
            fs_hz = 1.0 / float(np.median(np.diff(t)))
            bsplit, ncross, modp = free_structure(LI, fs_hz)
            for ci, cname in enumerate(sio.CLASS_NAMES):
                m = y == ci
                per_sub[sid][cname] = dict(
                    mean_tc=LI[m].mean(axis=0),
                    # T1 phase-locked: first half minus second half
                    halfdiff=float((LI[m][:, :half].mean(axis=1)
                                    - LI[m][:, half:].mean(axis=1)).mean()),
                    # T2 phase-free: how much LI moves inside a trial
                    within_sd=float(LI[m].std(axis=1).mean()),
                    # T4-T6: switch at any time, any number of times
                    best_split=float(bsplit[m].mean()),
                    crossings=float(ncross[m].mean()),
                    mod_power=float(modp[m].mean()),
                )
            # T3 decoding from the LI trajectory alone (8 equal bins)
            nb = 8
            bins = np.array_split(np.arange(LI.shape[-1]), nb)
            F = np.stack([LI[:, b].mean(axis=1) for b in bins], axis=1)
            ia, ib = 1, 2
            sel = np.isin(y, [ia, ib])
            yy = (y[sel] == ib).astype(int)
            clf = make_pipeline(StandardScaler(),
                                LDA(solver="lsqr", shrinkage="auto"))
            acc = cross_val_score(clf, F[sel], yy,
                                  cv=StratifiedKFold(5, shuffle=True,
                                                     random_state=42),
                                  n_jobs=1).mean()
            per_sub[sid]["decode_acc"] = float(acc)
            rows.append(dict(subject=sid, band=bname, metric="LI_decode_SvW",
                             value=float(acc)))
            for cname in sio.CLASS_NAMES:
                rows.append(dict(subject=sid, band=bname,
                                 metric=f"halfdiff_{cname}",
                                 value=per_sub[sid][cname]["halfdiff"]))
                rows.append(dict(subject=sid, band=bname,
                                 metric=f"withinSD_{cname}",
                                 value=per_sub[sid][cname]["within_sd"]))
        curves[bname] = (per_sub, t)

        # ---------------- T1 ----------------
        hd = {c: np.array([per_sub[s][c]["halfdiff"] for s in per_sub])
              for c in sio.CLASS_NAMES}
        print("\nT1  phase-locked: LI(first half) - LI(second half), dB")
        for c in sio.CLASS_NAMES:
            tt, pp = ttest_1samp(hd[c], 0.0)
            print(f"    {c:6s} {hd[c].mean():+6.3f}  (t={tt:+5.2f}, p={pp:.3f})")
        d = hd["Walk"] - hd["Stand"]
        tt, pp = ttest_rel(hd["Walk"], hd["Stand"])
        print(f"    Walk - Stand = {d.mean():+.3f} dB   t={tt:+.2f}  p={pp:.3f}")
        tests.append((bname, "T1 half-difference", "+", float(d.mean()), float(pp)))

        # ---------------- T2 ----------------
        wsd = {c: np.array([per_sub[s][c]["within_sd"] for s in per_sub])
               for c in sio.CLASS_NAMES}
        print("\nT2  phase-free: within-trial SD of LI(t), dB")
        for c in sio.CLASS_NAMES:
            print(f"    {c:6s} {wsd[c].mean():6.3f}")
        tt, pp = ttest_rel(wsd["Walk"], wsd["Stand"])
        dd = float((wsd["Walk"] - wsd["Stand"]).mean())
        print(f"    Walk - Stand = {dd:+.3f} dB   t={tt:+.2f}  p={pp:.3f}")
        tests.append((bname, "T2 within-trial SD", "+", dd, float(pp)))

        # ---------------- T3 ----------------
        acc = np.array([per_sub[s]["decode_acc"] for s in per_sub])
        tt, pp = ttest_1samp(acc, 0.5)
        print(f"\nT3  decode Stand vs Walk from LI(t) alone: "
              f"{acc.mean()*100:.1f}% ± {acc.std()*100:.1f}"
              f"  (vs 50%, p={pp:.3f})")
        print("    per participant: " +
              " ".join(f"{v*100:.0f}" for v in acc))
        tests.append((bname, "T3 decode from LI(t)", "+",
                      float(acc.mean() - 0.5), float(pp)))

        # ------------- T4/T5/T6: switch at ANY time, ANY count -------------
        print("\nT4-T6  no assumption about when or how often the switch occurs")
        for key, label, unit in (
                ("best_split", "T4 largest step at the best split point", "dB"),
                ("crossings", "T5 crossings of the trial mean", "count"),
                ("mod_power", "T6 LI modulation power 0.2-1 Hz", "log10")):
            vals = {c: np.array([per_sub[s][c][key] for s in per_sub])
                    for c in sio.CLASS_NAMES}
            tt, pp = ttest_rel(vals["Walk"], vals["Stand"])
            d = (vals["Walk"] - vals["Stand"]).mean()
            print(f"    {label:42s} "
                  f"Rest {vals['Rest'].mean():7.3f} | "
                  f"Stand {vals['Stand'].mean():7.3f} | "
                  f"Walk {vals['Walk'].mean():7.3f}   "
                  f"W-S {d:+.3f} {unit}  p={pp:.3f}")
            tests.append((bname, label[:34], "+", float(d), float(pp)))
            for c in sio.CLASS_NAMES:
                rows.append(dict(subject="GROUP", band=bname,
                                 metric=f"{key}_{c}", value=float(vals[c].mean())))

    # ---------------- multiple-comparison correction ----------------
    # Twelve tests were run (6 statistics x 2 bands).  Every one of them
    # predicts a POSITIVE Walk-minus-Stand difference if the instructed
    # alternation drives the EEG, so direction is as informative as p.
    print("\n" + "=" * 78)
    print("ALL TESTS, Holm-corrected across the 12 comparisons")
    print("=" * 78)
    ps = [t[4] for t in tests]
    adj = holm(ps)
    print(f"  {'band':11s} {'test':36s} {'W-S':>9s} {'p':>7s} {'p_holm':>8s}  dir")
    n_sig_right, n_sig_wrong = 0, 0
    for (band, name, want, diff, pv), pa in zip(tests, adj):
        right = diff > 0
        mark = ""
        if pa < 0.05:
            mark = "  SIGNIF, predicted dir" if right else "  SIGNIF, WRONG dir"
            n_sig_right += right
            n_sig_wrong += not right
        print(f"  {band:11s} {name:36s} {diff:+9.3f} {pv:7.3f} {pa:8.3f}"
              f"  {'+' if right else '-'}{mark}")
    print(f"\n  after correction: {n_sig_right} significant in the predicted "
          f"direction, {n_sig_wrong} in the opposite direction.")
    if n_sig_right == 0:
        print("  => no support for an alternating-laterality signature.")

    pd.DataFrame(tests, columns=["band", "test", "predicted", "walk_minus_stand",
                                 "p"]).assign(p_holm=adj).to_csv(
        os.path.join(R_DIR, "time_laterality_tests.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(R_DIR, "time_laterality.csv"),
                              index=False)

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, len(BANDS), figsize=(6.2 * len(BANDS), 4.0),
                             constrained_layout=True, sharey=True)
    for ax, bname in zip(np.atleast_1d(axes), BANDS):
        per_sub, t = curves[bname]
        for cname in sio.CLASS_NAMES:
            M = np.stack([per_sub[s][cname]["mean_tc"] for s in per_sub])
            m, se = M.mean(0), M.std(0) / np.sqrt(len(M))
            ax.plot(t, m, color=SER[cname], lw=2, label=cname)
            ax.fill_between(t, m - se, m + se, color=SER[cname], alpha=.18,
                            linewidth=0)
        ax.axhline(0, color=INK, lw=1)
        ax.axvline(2.5, color=MUTED, lw=1, ls="--")
        ax.text(2.52, ax.get_ylim()[1], " expected step 1 → step 2",
                fontsize=7.4, color=MUTED, va="top")
        ax.set_xlabel("time from imagery onset (s)")
        ax.set_title(bname, loc="left")
    np.atleast_1d(axes)[0].set_ylabel(
        "laterality index (dB)\nright hemisphere ← → left")
    np.atleast_1d(axes)[0].legend(frameon=False, ncols=3, fontsize=8.5)
    fig.suptitle("Does laterality alternate within a Walk trial?  "
                 "(mean ± s.e.m., n = 10)\n"
                 "Prediction if the two instructed steps drive the EEG: Walk "
                 "crosses zero near 2.5 s, Stand stays flat.",
                 ha="left", x=0.01, fontsize=11)
    fig.savefig(os.path.join(R_DIR, "fig_time_laterality.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("\nwrote", os.path.join(R_DIR, "fig_time_laterality.png"))


if __name__ == "__main__":
    main()
