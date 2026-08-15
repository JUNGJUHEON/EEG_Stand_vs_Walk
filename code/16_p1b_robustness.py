# -*- coding: utf-8 -*-
"""
Stage 16 — cheap robustness checks on the P1b / peak-laterality finding,
BEFORE spending an independent validation set on it.

P1b is the one directionally-predicted effect in this dataset: at the instant
of maximum laterality, Walk sits further toward the right hemisphere than
Stand.  It is also the most likely thing in this project to be an artifact of
searching, because it was noticed in a printout and tested afterwards.

Two checks that cost minutes and can kill it outright:

R1  INTERNAL REPLICATION ACROSS ELECTRODE PAIRS.
    The montage has three symmetric pairs — FC3/FC4, C3/C4, CP3/CP4.  A real
    lateralised cortical source projects onto all three.  Each pair here gets
    its OWN peak selection and its own measurement, so the three are close to
    independent replications of the same claim.  Noise scatters; signal does
    not.

R2  JACKKNIFE.  Drop one participant at a time and recompute.  Two specific
    worries: (a) the effect rests on one or two participants; (b) it rests on
    S06, whose Stand-vs-Walk effect was shown in stage 10 to be arousal and
    muscle tone rather than sensorimotor rhythm, yet who is among the four
    participants with an uncorrected-significant peak-feature decode.

If either check fails there is no point running the ME validation.

Output: results/p1b_robustness.csv, fig_p1b_robustness.png
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
from scipy.stats import ttest_rel, ttest_1samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import steam_io as sio        # noqa: E402

R_DIR = sio.RESULTS_DIR
CH = sio.EEG_CHANNELS
PAIRS = {"FC3/FC4": ("FC3", "FC4"), "C3/C4": ("C3", "C4"),
         "CP3/CP4": ("CP3", "CP4"),
         "all three (combined)": (("FC3", "C3", "CP3"), ("FC4", "C4", "CP4"))}
BANDS = {"mu 8-13": (8, 13), "beta 13-30": (13, 30)}
HALF_S = 1.0
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
BLUE, ORANGE = "#1f6feb", "#bc4c00"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.dpi": 170,
})


def env_of(X, fs):
    env = np.abs(hilbert(X, axis=-1)) ** 2
    k = max(1, int(round(0.25 * fs)))
    kern = np.ones(k) / k
    return np.apply_along_axis(lambda v: np.convolve(v, kern, mode="same"), -1, env)


def li_from(env, left, right):
    li = [CH.index(c) for c in (left if isinstance(left, tuple) else (left,))]
    ri = [CH.index(c) for c in (right if isinstance(right, tuple) else (right,))]
    return 10.0 * np.log10((env[:, ri].mean(1) + 1e-30) /
                           (env[:, li].mean(1) + 1e-30))


def signed_peak(LI, fs, half_s=HALF_S):
    h = int(round(half_s * fs))
    seg = np.abs(LI[:, h:LI.shape[1] - h])
    pk = seg.argmax(axis=1) + h
    return LI[np.arange(len(LI)), pk], pk


def main():
    subs = sio.list_subjects()
    rows = []
    # per band -> per pair -> per subject Walk-minus-Stand signed peak LI
    R1 = {b: {p: [] for p in PAIRS} for b in BANDS}
    dec = {b: [] for b in BANDS}          # per-subject decode accuracy (combined)

    for bname, (lo, hi) in BANDS.items():
        for date, subj in subs:
            bk = sio.get_bank(date, subj, bands=((lo, hi),),
                              tmin=0.0, tmax=5.0, ref="none")
            X, y, fs = bk["Xb"][0], bk["y"], bk["fs"]
            env = env_of(X, fs)

            for pname, (lch, rch) in PAIRS.items():
                LI = li_from(env, lch, rch)
                sp, pk = signed_peak(LI, fs)          # own peak per pair
                d = sp[y == 2].mean() - sp[y == 1].mean()
                R1[bname][pname].append(float(d))

            # decoding, combined montage — same features as stage 15
            LI = li_from(env, PAIRS["all three (combined)"][0],
                         PAIRS["all three (combined)"][1])
            sp, pk = signed_peak(LI, fs)
            h = int(round(HALF_S * fs))
            n = len(y)
            idx = pk[:, None] + np.arange(-h, h + 1)[None, :]
            LIl = np.take_along_axis(LI, idx, axis=1)
            cp = np.log10(np.stack([env[i, :, pk[i] - h:pk[i] + h + 1]
                                    for i in range(n)]).mean(-1) + 1e-30)
            F = np.column_stack([sp, np.abs(sp), pk / fs, LIl.mean(1), cp])
            m = np.isin(y, [1, 2])
            yy = (y[m] == 2).astype(int)
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=5000, C=0.3))
            dec[bname].append(cross_val_score(
                clf, F[m], yy,
                cv=StratifiedKFold(5, shuffle=True, random_state=42)).mean())

    # ================= R1 =================
    print("=" * 78)
    print("R1  INTERNAL REPLICATION — each electrode pair, own peak selection")
    print("    Walk minus Stand, signed laterality at the peak (dB).")
    print("    A real right-lateralised source is POSITIVE in all three pairs.")
    print("=" * 78)
    for bname in BANDS:
        print(f"\n  {bname}")
        n_pos = 0
        for pname in PAIRS:
            v = np.array(R1[bname][pname])
            t, p = ttest_1samp(v, 0.0)
            pos = v.mean() > 0
            n_pos += pos and pname != "all three (combined)"
            print(f"    {pname:22s} {v.mean():+7.3f} dB  t={t:+5.2f}  p={p:.3f}"
                  f"  {'+' if pos else '-'}")
            rows.append(dict(check="R1", band=bname, pair=pname,
                             value=float(v.mean()), p=float(p)))
        print(f"    -> {n_pos}/3 independent pairs in the predicted direction")

    # ================= R2 =================
    print("\n" + "=" * 78)
    print("R2  JACKKNIFE — drop one participant, recompute")
    print("=" * 78)
    for bname in BANDS:
        v = np.array(R1[bname]["all three (combined)"])
        a = np.array(dec[bname])
        full_t, full_p = ttest_1samp(v, 0.0)
        full_at, full_ap = ttest_1samp(a, 0.5)
        print(f"\n  {bname}   full sample: LI diff {v.mean():+.3f} dB "
              f"(p={full_p:.3f}) | decode {a.mean()*100:.1f}% (p={full_ap:.4f})")
        print(f"    {'dropped':14s} {'LI diff':>9s} {'p':>7s} "
              f"{'decode':>8s} {'p':>8s}")
        worst_p, worst_s = 0.0, ""
        for i, (d_, s_) in enumerate(subs):
            keep = np.ones(len(v), bool)
            keep[i] = False
            t1, p1 = ttest_1samp(v[keep], 0.0)
            t2, p2 = ttest_1samp(a[keep], 0.5)
            flag = "  <-- S06" if sio.anon(d_) == "S06" else ""
            print(f"    {sio.anon(d_):4s} {v[keep].mean():+9.3f} {p1:7.3f} "
                  f"{a[keep].mean()*100:7.1f}% {p2:8.4f}{flag}")
            rows.append(dict(check="R2", band=bname, pair=f"drop_{sio.anon(d_)}",
                             value=float(a[keep].mean()), p=float(p2)))
            if p2 > worst_p:
                worst_p, worst_s = p2, s_
        print(f"    worst case: dropping {worst_s} gives decode p={worst_p:.4f}"
              f"  {'-> effect survives every single-participant removal'
                 if worst_p < .05 else '-> effect DOES NOT survive'}")

    pd.DataFrame(rows).to_csv(os.path.join(R_DIR, "p1b_robustness.csv"),
                              index=False)

    # ================= figure =================
    fig, axes = plt.subplots(1, 2 * len(BANDS), figsize=(4.0 * 2 * len(BANDS), 3.8),
                             constrained_layout=True)
    for j, bname in enumerate(BANDS):
        ax = axes[2 * j]
        names = [p for p in PAIRS if p != "all three (combined)"]
        for i, pname in enumerate(names):
            v = np.array(R1[bname][pname])
            ax.bar(i, v.mean(), 0.58,
                   color=BLUE if v.mean() > 0 else ORANGE, zorder=2)
            ax.scatter(np.full(len(v), i), v, s=13, color="white",
                       edgecolor=INK, linewidth=0.6, zorder=3)
        ax.axhline(0, color=INK, lw=1.2)
        ax.set_xticks(range(len(names)), names, fontsize=8)
        ax.set_ylabel("Walk − Stand peak LI (dB)")
        ax.set_title(f"R1  electrode pairs — {bname}\npositive in all three = "
                     "internal replication", loc="left", fontsize=8.8,
                     linespacing=1.35)

        ax = axes[2 * j + 1]
        a = np.array(dec[bname])
        jk = np.array([a[np.arange(len(a)) != i].mean() for i in range(len(a))])
        x = np.arange(len(a))
        ax.axhline(a.mean() * 100, color=INK, lw=1.2, ls="--",
                   label=f"full sample {a.mean()*100:.1f}%")
        ax.axhline(50, color=MUTED, lw=1, ls=":")
        cols = [ORANGE if sio.anon(d_) == "S06" else BLUE for d_, _ in subs]
        ax.bar(x, jk * 100, 0.6, color=cols, zorder=2)
        ax.set_ylim(45, max(60, jk.max() * 100 + 2))
        ax.set_xticks(x, [sio.anon(d_) for d_, _ in subs], rotation=60, ha="right",
                      fontsize=6.6)
        ax.set_ylabel("decode accuracy, that\nparticipant removed (%)")
        ax.legend(frameon=False, fontsize=7.4)
        ax.set_title(f"R2  jackknife — {bname}\norange = S06 (known artifact)",
                     loc="left", fontsize=8.8, linespacing=1.35)
    fig.suptitle("Robustness of the peak-laterality (P1b) finding before "
                 "spending a validation set on it",
                 ha="left", x=0.008, fontsize=11.5, fontweight="bold")
    fig.savefig(os.path.join(R_DIR, "fig_p1b_robustness.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("\nwrote", os.path.join(R_DIR, "fig_p1b_robustness.png"))


if __name__ == "__main__":
    main()
