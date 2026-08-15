# -*- coding: utf-8 -*-
"""
Stage 17 — INDEPENDENT VALIDATION of the P1b peak-laterality finding on the
motor-EXECUTION recordings.

=============================  PRE-SPECIFICATION  =============================
Written before this script was run for the first time.  The ME recordings have
never been touched by any analysis in this project; every result up to stage 16
comes from the MI files only.

WHAT WAS FOUND IN MI (stages 14-16, beta 13-30 Hz only; mu was dropped at
stage 16 because it failed the jackknife):

    At each trial's instant of maximum laterality, the Walk condition sits
    further toward the RIGHT hemisphere than Stand: +0.72 dB combined,
    positive in all three electrode pairs with a posterior gradient
    (FC +0.32, C +0.66, CP +1.03), and the derived decoder survives removal
    of any single participant (p = .003-.021).

WHY ME IS A VALID TEST SET:
    separate session, separate cap placement and impedances, never used here,
    and — decisively — the arm swing is ACTUALLY PERFORMED rather than
    imagined.  A real unilateral arm movement produces the largest and most
    reliable lateralised ERD in all of EEG.  If an 8-channel montage can see
    compound-limb lateralisation at all, it must see it here.

PREDICTIONS, in order of how much they would change the conclusion:

    V1  (primary, label-free)  The two ME task conditions differ in beta
        peak-laterality, and by MORE than the two MI conditions do.  Real
        movement > imagined movement.  If V1 fails, the montage cannot
        resolve compound-limb lateralisation even during overt movement, and
        the MI finding is almost certainly noise.

    V2  (direction)  The compound condition is right-hemisphere dominant.
        Which ME marker is the compound condition is NOT established by the
        timing, so both readings are reported and neither is assumed.

    V3  (per-participant consistency)  Participants with a larger MI effect
        should show a larger ME effect.  Positive Spearman across the ten.
        This one is label-free in magnitude and is the strongest evidence
        available without new participants.

FAILURE IS INFORMATIVE.  If V1 fails, the honest conclusion is that P1b was a
product of searching, and it should be dropped rather than reported.
===============================================================================

Output: results/me_validation.csv, fig_me_validation.png
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
from scipy.stats import spearmanr, ttest_1samp, ttest_rel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import steam_io as sio        # noqa: E402

R_DIR = sio.RESULTS_DIR
CH = sio.EEG_CHANNELS
PAIRS = [("FC3", "FC4"), ("C3", "C4"), ("CP3", "CP4")]
LEFT = [p[0] for p in PAIRS]
RIGHT = [p[1] for p in PAIRS]
BAND = (13.0, 30.0)               # beta only — mu was dropped at stage 16
HALF_S = 1.0
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
BLUE, ORANGE, GREEN = "#1f6feb", "#bc4c00", "#116329"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.dpi": 170,
})


# Absolute artifact threshold, in microvolts peak-to-peak within the analysis
# band.  This is NOT optional for ME.
#
# The default rejection is MAD-based, i.e. relative to each participant's own
# median trial.  That cannot work when an entire session is contaminated,
# because the median is inflated too — and ME is exactly that case, since the
# participant is really standing up and really stepping.  Measured: ME median
# peak-to-peak is 1.5-4x the MI value and the 99th percentile up to 1000x
# (S05 reaches 60 mV, essentially the amplifier rail), yet MAD rejection kept
# 149/150 and 150/150 trials.
#
# The threshold is calibrated on the CLEAN condition: the largest single MI
# trial in the whole dataset is 141.9 uV, so 150 uV passes ~100 % of MI while
# removing the ME movement artifacts.  Applied identically to both.
P2P_MAX_UV = 150.0
MIN_TRIALS_PER_CLASS = 15


def peak_laterality(date, subj, kind, strict=True):
    """
    Signed laterality (dB) at each trial's own instant of maximum |LI|,
    identical procedure to stages 14-16.  Returns (values, labels, names).
    """
    bk = sio.get_bank(date, subj, bands=(BAND,), tmin=0.0, tmax=5.0,
                      ref="none", kind=kind)
    X, y, fs = bk["Xb"][0], bk["y"], bk["fs"]
    if strict:
        p2p = (X.max(-1) - X.min(-1)).max(axis=1) * 1e6
        good = p2p <= P2P_MAX_UV
        X, y = X[good], y[good]
    env = np.abs(hilbert(X, axis=-1)) ** 2
    k = max(1, int(round(0.25 * fs)))
    kern = np.ones(k) / k
    env = np.apply_along_axis(lambda v: np.convolve(v, kern, mode="same"), -1, env)
    LI = 10.0 * np.log10(
        (env[:, [CH.index(c) for c in RIGHT]].mean(1) + 1e-30) /
        (env[:, [CH.index(c) for c in LEFT]].mean(1) + 1e-30))
    h = int(round(HALF_S * fs))
    pk = np.abs(LI[:, h:LI.shape[1] - h]).argmax(axis=1) + h
    sp = LI[np.arange(len(LI)), pk]

    # per electrode pair too
    per_pair = {}
    for l, r in PAIRS:
        Lp = env[:, CH.index(l)]
        Rp = env[:, CH.index(r)]
        li = 10.0 * np.log10((Rp + 1e-30) / (Lp + 1e-30))
        pk2 = np.abs(li[:, h:li.shape[1] - h]).argmax(axis=1) + h
        per_pair[f"{l}/{r}"] = li[np.arange(len(li)), pk2]
    return sp, y, bk["class_names"], per_pair


def main():
    subs = sio.list_subjects()
    rows = []
    mi_d, me_d = [], []
    me_pair = {f"{l}/{r}": [] for l, r in PAIRS}
    labels = None

    print("=" * 78)
    print("ME VALIDATION — beta 13-30 Hz, peak-laterality, procedure frozen "
          "from stage 16")
    print("=" * 78)
    print(f"\n  {'participant':14s} {'MI Walk-Stand':>14s} {'ME TaskB-TaskA':>15s}")

    kept_subs = []
    print(f"  (absolute artifact threshold {P2P_MAX_UV:.0f} uV p2p, applied to "
          f"BOTH conditions; participants need >= {MIN_TRIALS_PER_CLASS} "
          f"surviving trials per condition)")
    for date, subj in subs:
        sp_mi, y_mi, _, _ = peak_laterality(date, subj, "MI")
        sp_me, y_me, names, pp = peak_laterality(date, subj, "ME")
        labels = names
        n_mi = min((y_mi == 1).sum(), (y_mi == 2).sum())
        n_me = min((y_me == 1).sum(), (y_me == 2).sum())
        if n_mi < MIN_TRIALS_PER_CLASS or n_me < MIN_TRIALS_PER_CLASS:
            print(f"  {date}/{subj:4s} EXCLUDED — only {n_me} ME trials/class "
                  f"survive the artifact threshold")
            rows.append(dict(subject=f"{date}/{subj}", mi_diff=np.nan,
                             me_diff=np.nan, n_me_per_class=int(n_me)))
            continue
        d_mi = float(sp_mi[y_mi == 2].mean() - sp_mi[y_mi == 1].mean())
        d_me = float(sp_me[y_me == 2].mean() - sp_me[y_me == 1].mean())
        for k_, v_ in pp.items():
            me_pair[k_].append(float(v_[y_me == 2].mean() - v_[y_me == 1].mean()))
        mi_d.append(d_mi)
        me_d.append(d_me)
        kept_subs.append((date, subj))
        print(f"  {date}/{subj:4s} {d_mi:+14.3f} {d_me:+15.3f}   "
              f"(ME n/class = {n_me})")
        rows.append(dict(subject=f"{date}/{subj}", mi_diff=d_mi, me_diff=d_me,
                         n_me_per_class=int(n_me)))

    mi_d, me_d = np.array(mi_d), np.array(me_d)
    subs = kept_subs
    print(f"\n  {len(subs)}/10 participants usable after artifact rejection")
    if len(subs) < 5:
        print("  WARNING: too few participants left for the validation to "
              "mean anything.")

    # ------------------------- V1 -------------------------
    print("\n" + "-" * 78)
    print("V1  PRIMARY (label-free): do the two ME conditions differ, and by "
          "more than MI?")
    print("-" * 78)
    t_me, p_me = ttest_1samp(me_d, 0.0)
    t_mi, p_mi = ttest_1samp(mi_d, 0.0)
    t_c, p_c = ttest_rel(np.abs(me_d), np.abs(mi_d))
    print(f"    ME condition difference   {me_d.mean():+.3f} dB   "
          f"t={t_me:+5.2f}  p={p_me:.3f}")
    print(f"    MI condition difference   {mi_d.mean():+.3f} dB   "
          f"t={t_mi:+5.2f}  p={p_mi:.3f}   (the stage-14 finding)")
    print(f"    |ME| - |MI| magnitude     {(np.abs(me_d)-np.abs(mi_d)).mean():+.3f} dB   "
          f"t={t_c:+5.2f}  p={p_c:.3f}")
    v1 = p_me < 0.05
    print(f"    => V1 {'PASSES' if v1 else 'FAILS'}: executed movement "
          f"{'does' if v1 else 'does NOT'} produce a significant "
          f"peak-laterality difference between the two conditions.")
    rows.append(dict(subject="GROUP", mi_diff=float(mi_d.mean()),
                     me_diff=float(me_d.mean())))

    # ------------------------- V2 -------------------------
    print("\n" + "-" * 78)
    print("V2  DIRECTION (both readings; the timing does not say which ME "
          "marker is the compound task)")
    print("-" * 78)
    print(f"    if {labels[2]} is the compound (arm-swing) task: "
          f"right-dominance = {me_d.mean():+.3f} dB "
          f"({'predicted sign' if me_d.mean() > 0 else 'WRONG sign'})")
    print(f"    if {labels[1]} is the compound task: "
          f"right-dominance = {-me_d.mean():+.3f} dB "
          f"({'predicted sign' if -me_d.mean() > 0 else 'WRONG sign'})")
    print("    per electrode pair (ME):")
    for k_, v_ in me_pair.items():
        v_ = np.array(v_)
        t_, p_ = ttest_1samp(v_, 0.0)
        print(f"      {k_:9s} {v_.mean():+7.3f} dB  t={t_:+5.2f}  p={p_:.3f}")
        rows.append(dict(subject=f"PAIR_{k_}", mi_diff=np.nan,
                         me_diff=float(v_.mean())))

    # ------------------------- V3 -------------------------
    print("\n" + "-" * 78)
    print("V3  PER-PARTICIPANT CONSISTENCY between MI and ME")
    print("-" * 78)
    rho, p_rho = spearmanr(mi_d, me_d)
    r_p, p_p = spearmanr(np.abs(mi_d), np.abs(me_d))
    print(f"    signed   Spearman rho = {rho:+.2f}  p={p_rho:.3f}")
    print(f"    absolute Spearman rho = {r_p:+.2f}  p={p_p:.3f}")
    v3 = rho > 0 and p_rho < 0.05
    print(f"    => V3 {'PASSES' if v3 else 'FAILS'}")

    # ------------------------- verdict -------------------------
    print("\n" + "=" * 78)
    if v1 and v3:
        print("VERDICT: P1b replicates on independent execution data.")
    elif v1:
        print("VERDICT: ME shows a condition difference, but it does not track "
              "MI participant-by-participant.  Partial support only.")
    else:
        print("VERDICT: NO replication.  Executed movement — where the arm "
              "swing is real and the lateralised ERD should be at its "
              "largest — does not show the effect.  P1b should be dropped, "
              "not reported.")
    print("=" * 78)

    pd.DataFrame(rows).to_csv(os.path.join(R_DIR, "me_validation.csv"),
                              index=False)

    # ------------------------- figure -------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(subs))
    ax.bar(x - 0.2, mi_d, 0.4, color=BLUE, label="MI  (Walk − Stand)")
    ax.bar(x + 0.2, me_d, 0.4, color=ORANGE, label="ME  (TaskB − TaskA)")
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xticks(x, [s for _, s in subs], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("peak laterality difference (dB)")
    ax.legend(frameon=False, fontsize=7.6)
    ax.set_title("per participant", loc="left", fontsize=9.5)

    ax = axes[1]
    ax.scatter(mi_d, me_d, s=46, color=BLUE, zorder=3)
    for (d_, s_), xm, ym in zip(subs, mi_d, me_d):
        ax.annotate(s_, (xm, ym), fontsize=6.6, xytext=(4, 3),
                    textcoords="offset points", color=MUTED)
    ax.axhline(0, color=INK, lw=1)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("MI effect (dB)")
    ax.set_ylabel("ME effect (dB)")
    ax.set_title(f"V3 consistency   rho = {rho:+.2f}, p = {p_rho:.3f}",
                 loc="left", fontsize=9.5)

    ax = axes[2]
    names = list(me_pair)
    for i, k_ in enumerate(names):
        v_ = np.array(me_pair[k_])
        ax.bar(i, v_.mean(), 0.58, color=BLUE if v_.mean() > 0 else ORANGE)
        ax.scatter(np.full(len(v_), i), v_, s=13, color="white",
                   edgecolor=INK, linewidth=0.6, zorder=3)
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xticks(range(len(names)), names, fontsize=8)
    ax.set_ylabel("ME peak laterality diff (dB)")
    ax.set_title("V2 per electrode pair (ME)", loc="left", fontsize=9.5)

    fig.suptitle("Independent validation on motor EXECUTION — procedure frozen "
                 "from the MI analysis, predictions pre-specified",
                 ha="left", x=0.008, fontsize=11.5, fontweight="bold")
    fig.savefig(os.path.join(R_DIR, "fig_me_validation.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("\nwrote", os.path.join(R_DIR, "fig_me_validation.png"))


if __name__ == "__main__":
    main()
