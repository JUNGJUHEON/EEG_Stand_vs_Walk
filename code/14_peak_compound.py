# -*- coding: utf-8 -*-
"""
Stage 14 — is there a MOMENT of compound-limb lateralisation in a Walk trial?

Different question again
------------------------
Stage 13 asked whether laterality *alternates* and found nothing.  But a Walk
trial contains, by instruction, at least one compound configuration:

    right foot + left arm      (or, in the second step, left foot + right arm)

Even if the alternation is not resolvable, that configuration should produce a
lateralised MOMENT somewhere in the trial, which Stand — a bilateral movement
— should not have.  Stage 13's statistics all averaged or counted over the
trial; none of them asked "how lateralised does this trial ever get?".

Two tests
---------
P1  PEAK LATERALITY.  Per trial, the largest excursion of the laterality index
    away from that participant's own Rest baseline:  max_t |LI(t) - LI_rest|.
    If a compound configuration is expressed at any instant, Walk should reach
    further than Stand.  Direction is also checked: step 1 (right leg + left
    arm) predicts a right-hemisphere-dominant excursion.

P2  ADAPTIVE-WINDOW DECODING.  For every trial, independently and WITHOUT
    using its label, find the 2 s window where |LI| is largest — the trial's
    own most-lateralised moment — and decode Stand vs Walk from covariance
    features taken there.  Compared against the same decoder on a fixed
    window.  Because the window is chosen per trial from that trial's own
    signal and never from the labels, this adds no leakage: the identical
    transform is applied to Stand and Walk alike.

If a compound-limb moment exists but is smeared by trial averaging, P2 is the
analysis that recovers it.

Output: results/peak_compound.csv, fig_peak_compound.png
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
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import decoders as D          # noqa: E402
import steam_io as sio        # noqa: E402

R_DIR = sio.RESULTS_DIR
CH = sio.EEG_CHANNELS
RIGHT, LEFT = ["FC4", "C4", "CP4"], ["FC3", "C3", "CP3"]
BANDS = {"mu 8-13": (8, 13), "beta 13-30": (13, 30)}
TMIN, TMAX = 0.0, 5.0
WIN_S = 2.0                       # length of the adaptive window
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


def li_and_raw(date, subj, lo, hi):
    ep = sio.get_epochs(date, subj, l_freq=lo, h_freq=hi,
                        tmin=TMIN, tmax=TMAX, ref="none")
    fs = ep["fs"]
    env = np.abs(hilbert(ep["X"], axis=-1)) ** 2
    k = max(1, int(round(0.25 * fs)))
    kern = np.ones(k) / k
    env = np.apply_along_axis(lambda v: np.convolve(v, kern, mode="same"), -1, env)
    Rp = env[:, [CH.index(c) for c in RIGHT]].mean(axis=1)
    Lp = env[:, [CH.index(c) for c in LEFT]].mean(axis=1)
    LI = 10.0 * np.log10((Rp + 1e-30) / (Lp + 1e-30))
    return LI, ep["X"], ep["y"], fs


def adaptive_windows(LI, fs, win_s=WIN_S):
    """Per trial: start index of the 2 s window with the largest |mean LI|.
    Uses only that trial's own signal — no labels involved."""
    w = int(round(win_s * fs))
    csum = np.cumsum(np.concatenate([np.zeros((LI.shape[0], 1)), LI], axis=1),
                     axis=1)
    starts = np.arange(0, LI.shape[1] - w + 1)
    means = (csum[:, starts + w] - csum[:, starts]) / w
    return starts[np.abs(means).argmax(axis=1)], w


def main():
    rows, store = [], {}
    subs = sio.list_subjects()

    for bname, (lo, hi) in BANDS.items():
        print(f"\n{'='*72}\n### {bname}\n{'='*72}")
        peak, speak, acc_fix, acc_ada = {c: [] for c in sio.CLASS_NAMES}, \
            {c: [] for c in sio.CLASS_NAMES}, [], []

        for date, subj in subs:
            LI, X, y, fs = li_and_raw(date, subj, lo, hi)
            rest_ref = LI[y == 0].mean()                 # this subject's baseline
            LIr = LI - rest_ref

            for ci, cname in enumerate(sio.CLASS_NAMES):
                m = y == ci
                idx = np.abs(LIr[m]).argmax(axis=1)
                peak[cname].append(float(np.abs(LIr[m]).max(axis=1).mean()))
                speak[cname].append(float(
                    LIr[m][np.arange(m.sum()), idx].mean()))

            # ---- P2: decode Stand vs Walk, fixed vs adaptive window ----
            starts, w = adaptive_windows(LI, fs)
            Xa = np.stack([X[i, :, s:s + w] for i, s in enumerate(starts)])
            c0 = int(round((TMAX - TMIN - WIN_S) / 2 * fs))
            Xf = X[:, :, c0:c0 + w]

            sel = np.isin(y, [1, 2])
            yy = (y[sel] == 2).astype(int)
            cv = StratifiedKFold(5, shuffle=True, random_state=42)
            acc_fix.append(cross_val_score(D.get("ts_lr"), Xf[sel], yy,
                                           cv=cv, n_jobs=1).mean())
            acc_ada.append(cross_val_score(D.get("ts_lr"), Xa[sel], yy,
                                           cv=cv, n_jobs=1).mean())

        # ---------------- P1 ----------------
        print("\nP1  peak |LI - Rest baseline| reached anywhere in the trial (dB)")
        for c in sio.CLASS_NAMES:
            print(f"    {c:6s} peak {np.mean(peak[c]):6.3f}   "
                  f"signed at peak {np.mean(speak[c]):+6.3f}")
        pw, ps = np.array(peak["Walk"]), np.array(peak["Stand"])
        tt, pp = ttest_rel(pw, ps)
        print(f"    Walk - Stand peak MAGNITUDE = {(pw-ps).mean():+.3f} dB   "
              f"t={tt:+.2f}  p={pp:.3f}")
        rows.append(dict(band=bname, test="P1_peak_LI_magnitude",
                         walk_minus_stand=float((pw - ps).mean()), p=float(pp)))

        # P1b: the DIRECTION at that peak.  A priori prediction — Walk step 1
        # is right leg + LEFT arm, so its excursion should be more
        # right-hemisphere dominant (more positive) than Stand's.
        sw, ss = np.array(speak["Walk"]), np.array(speak["Stand"])
        tt2, pp2 = ttest_rel(sw, ss)
        print(f"    Walk - Stand peak DIRECTION  = {(sw-ss).mean():+.3f} dB   "
              f"t={tt2:+.2f}  p={pp2:.3f}"
              f"{'   <-- predicted sign' if (sw-ss).mean() > 0 and pp2 < .05 else ''}")
        rows.append(dict(band=bname, test="P1b_peak_LI_direction",
                         walk_minus_stand=float((sw - ss).mean()), p=float(pp2)))
        # internal control: if peak |LI| were task-driven, Rest should be the
        # SMALLEST.  Report it so the reader can see whether it is.
        print(f"    control — Rest peak magnitude {np.mean(peak['Rest']):.3f} dB "
              f"({'largest of the three: peak |LI| is noise-driven' if np.mean(peak['Rest']) > max(pw.mean(), ps.mean()) else 'not the largest'})")

        # ---------------- P2 ----------------
        af, aa = np.array(acc_fix), np.array(acc_ada)
        tt, pp = ttest_rel(aa, af)
        print(f"\nP2  decode Stand vs Walk from a 2 s window")
        print(f"    fixed centre window     {af.mean()*100:5.1f}% ± {af.std()*100:4.1f}")
        print(f"    per-trial most-lateralised window "
              f"{aa.mean()*100:5.1f}% ± {aa.std()*100:4.1f}")
        print(f"    adaptive - fixed = {(aa-af).mean()*100:+.1f} pts  "
              f"t={tt:+.2f}  p={pp:.3f}")
        print("    per participant (adaptive): " +
              " ".join(f"{v*100:.0f}" for v in aa))
        rows.append(dict(band=bname, test="P2_adaptive_minus_fixed",
                         walk_minus_stand=float((aa - af).mean()), p=float(pp)))
        store[bname] = dict(peak=peak, acc_fix=af, acc_ada=aa)

    pd.DataFrame(rows).to_csv(os.path.join(R_DIR, "peak_compound.csv"),
                              index=False)

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 2 * len(BANDS),
                             figsize=(3.4 * 2 * len(BANDS), 3.8),
                             constrained_layout=True)
    for j, bname in enumerate(BANDS):
        s = store[bname]
        ax = axes[2 * j]
        for i, c in enumerate(sio.CLASS_NAMES):
            v = np.array(s["peak"][c])
            ax.bar(i, v.mean(), 0.6, color=SER[c])
            ax.scatter(np.full(len(v), i), v, s=12, color="white",
                       edgecolor=INK, linewidth=0.6, zorder=3)
        ax.set_xticks(range(3), sio.CLASS_NAMES, fontsize=8)
        ax.set_ylabel("peak |LI − Rest| (dB)")
        ax.set_title(f"P1  {bname}", loc="left", fontsize=9.5)

        ax = axes[2 * j + 1]
        x = np.arange(len(s["acc_fix"]))
        ax.plot(x, s["acc_fix"] * 100, "o-", color=MUTED, ms=4,
                label="fixed window")
        ax.plot(x, s["acc_ada"] * 100, "o-", color=SER["Walk"], ms=4,
                label="most-lateralised window")
        ax.axhline(50, color=INK, lw=1, ls=":")
        ax.set_xticks(x, [s_.split("/")[1] for s_ in
                          [f"{d}/{p}" for d, p in sio.list_subjects()]],
                      rotation=60, ha="right", fontsize=6.6)
        ax.set_ylabel("Stand vs Walk (%)")
        ax.legend(frameon=False, fontsize=7.2)
        ax.set_title(f"P2  {bname}", loc="left", fontsize=9.5)
    fig.suptitle("Is there a compound-limb MOMENT in a Walk trial that Stand "
                 "lacks?", ha="left", x=0.01, fontsize=11)
    fig.savefig(os.path.join(R_DIR, "fig_peak_compound.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("\nwrote", os.path.join(R_DIR, "fig_peak_compound.png"))


if __name__ == "__main__":
    main()
