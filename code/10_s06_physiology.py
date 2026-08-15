# -*- coding: utf-8 -*-
"""
Stage 10 — is S06's Stand-vs-Walk effect neurophysiology or an artifact?

Stage 8 showed S06 decodes Stand vs Walk at 67.0 % (permutation p = .004,
Holm .040) while nine other participants sit at chance.  That is a
*statistical* result: it proves something separates the two conditions.  It
does not prove that something is sensorimotor cortex.  Scalp EMG, eye
movement, electrode drift and respiration all produce decodable differences,
and S06 is specifically the participant whose strongest Rest-vs-Stand feature
sat at 36 Hz with a FLAT topography — the muscle-tension signature.

Four checks, each of which an artifact would be expected to fail:

  1. SPECTRUM.  Where in frequency does the discriminability live?  Genuine
     sensorimotor rhythm sits in mu (8-13) / beta (13-30).  EMG rises
     monotonically into the 30 Hz+ range; drift and respiration live below
     ~2 Hz; blinks below ~4 Hz and frontally.

  2. TOPOGRAPHY.  Cortical mu/beta is spatially GRADED with a central or
     centro-parietal maximum.  Scalp EMG loads roughly uniformly across a
     small montage, or maximally at the most peripheral electrodes.

  3. CROSS-RUN GENERALISATION.  S06's 150 trials come from two separate
     recordings (MI: 40/class, MI2: 10/class) with their own cap placement
     and impedances.  Training on MI and testing on MI2 is a much harder test
     than k-fold: a drift or impedance artifact is run-specific and should
     not transfer, whereas a real neural pattern should.

  4. BAND ABLATION.  Rebuild the decoder from mu only, beta only, and >30 Hz
     only.  If accuracy survives on mu/beta and collapses above 30 Hz, the
     effect is not muscle.

Output: results/fig_s06_physiology.png, results/s06_physiology.csv
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
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import welch
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import decoders as D          # noqa: E402
import steam_io as sio        # noqa: E402

R_DIR = sio.RESULTS_DIR
CH = sio.EEG_CHANNELS
DATE = "250922"          # the individually significant participant (S06)
REF_DATE = "250918"      # a non-significant participant, for contrast
INK, GRID = "#1f2328", "#e5e7eb"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#6b7280", "ytick.color": "#6b7280",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.dpi": 130,
})


def sw(date, subj, lo=1.0, hi=45.0, ref="none", reject=True):
    ep = sio.get_epochs(date, subj, l_freq=lo, h_freq=hi, tmin=0.5, tmax=4.5,
                        ref=ref, reject=reject)
    ia, ib = sio.CLASS_NAMES.index("Stand"), sio.CLASS_NAMES.index("Walk")
    m = np.isin(ep["y"], [ia, ib])
    return ep["X"][m], (ep["y"][m] == ib).astype(int), ep["run"][m], ep["fs"]


def rsq_map(X, y, fs):
    f, P = welch(X, fs=fs, nperseg=int(fs), axis=-1)
    sel = (f >= 2) & (f <= 45)
    f, Z = f[sel], np.log(P[:, :, sel] + 1e-30)
    zc = Z - Z.mean(0)
    yc = y - y.mean()
    r = (zc * yc[:, None, None]).sum(0) / (
        np.sqrt((zc ** 2).sum(0)) * np.sqrt((yc ** 2).sum()) + 1e-30)
    return f, r


def main():
    rows = []
    date, subj = DATE, sio.subject_for(DATE)
    X, y, run, fs = sw(date, subj)
    print(f"S06 Stand vs Walk: n={len(y)} ({np.bincount(y)}), runs={dict(zip(*np.unique(run, return_counts=True)))}")

    # ---------------- 1 + 2 : spectrum and topography ----------------
    f, r = rsq_map(X, y, fs)
    sig = np.sign(r) * r ** 2
    pk_ch, pk_f = np.unravel_index(np.abs(r).argmax(), r.shape)
    print(f"\n[1/2] peak |r| = {abs(r[pk_ch, pk_f]):.2f} (r^2={r[pk_ch,pk_f]**2:.3f}) "
          f"at {CH[pk_ch]} {f[pk_f]:.0f} Hz")

    # spatial profile at the peak frequency: graded or flat?
    prof = np.abs(r[:, pk_f])
    ratio = prof.max() / prof.min()
    print(f"      topography at {f[pk_f]:.0f} Hz: " +
          " ".join(f"{c}={v:+.2f}" for c, v in zip(CH, r[:, pk_f])))
    print(f"      max/min |r| across electrodes = {ratio:.2f}  "
          f"({'GRADED (cortical-like)' if ratio > 2 else 'FLAT (EMG-like)'})")

    # where is the mass, by band?
    for name, (lo, hi) in [("delta/theta 2-8", (2, 8)), ("mu 8-13", (8, 13)),
                           ("beta 13-30", (13, 30)), ("gamma 30-45", (30, 45))]:
        m = (f >= lo) & (f < hi)
        print(f"      mean r^2 in {name:16s} = {(r[:, m] ** 2).mean():.4f}")
        rows.append(dict(check="rsq_by_band", band=name,
                         value=float((r[:, m] ** 2).mean())))

    # ---------------- 3 : cross-run generalisation ----------------
    print("\n[3] cross-run generalisation (train MI -> test MI2)")
    Xb_all, y_all, run_all = [], [], []
    Xp, Xb, yy = sio.get_task_arrays(date, subj, lo=8.0, hi=30.0,
                                     tmin=0.5, tmax=4.5, ref="none")
    ep = sio.get_epochs(date, subj, l_freq=8, h_freq=30, tmin=0.5, tmax=4.5,
                        ref="none")
    ia, ib = sio.CLASS_NAMES.index("Stand"), sio.CLASS_NAMES.index("Walk")
    m = np.isin(yy, [ia, ib])
    Xb2, y2, run2 = Xb[m], (yy[m] == ib).astype(int), ep["run"][m]
    tr, te = run2 == "MI", run2 == "MI2"
    print(f"    train n={tr.sum()} (MI), test n={te.sum()} (MI2)")
    if te.sum() >= 8 and len(np.unique(y2[te])) == 2:
        mdl = D.get("fb_ts_lr").fit(Xb2[tr], y2[tr])
        acc = float((mdl.predict(Xb2[te]) == y2[te]).mean())
        # chance band for this small test set
        from scipy.stats import binomtest
        p = binomtest(int(round(acc * te.sum())), int(te.sum()), 0.5,
                      alternative="greater").pvalue
        print(f"    cross-run accuracy = {acc*100:.1f}%  (binomial p={p:.3f})")
        rows.append(dict(check="cross_run", band="8-30", value=acc))
    else:
        acc = np.nan
        print("    too few MI2 trials for a meaningful test")

    # within-run baseline for comparison
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    within = cross_val_score(D.get("fb_ts_lr"), Xb2[tr], y2[tr], cv=cv,
                             n_jobs=1).mean()
    print(f"    within-MI 5-fold baseline = {within*100:.1f}%")
    rows.append(dict(check="within_run", band="8-30", value=float(within)))

    # ---------------- 4 : band ablation ----------------
    print("\n[4] band ablation (5-fold, ts_lr on a single band)")
    bands = [("mu 8-13", 8, 13), ("beta 13-30", 13, 30),
             ("low beta 13-20", 13, 20), ("high beta 20-30", 20, 30),
             ("gamma 30-45 (EMG)", 30, 45), ("delta/theta 2-8", 2, 8)]
    for name, lo, hi in bands:
        Xa, ya, _, _ = sw(date, subj, lo=lo, hi=hi)
        s = cross_val_score(D.get("ts_lr"), Xa, ya, cv=cv, n_jobs=1)
        print(f"    {name:20s} {s.mean()*100:5.1f}% ± {s.std()*100:4.1f}")
        rows.append(dict(check="band_ablation", band=name, value=float(s.mean())))

    pd.DataFrame(rows).to_csv(os.path.join(R_DIR, "s06_physiology.csv"),
                              index=False)

    # ---------------- figure ----------------
    dref, sref = REF_DATE, sio.subject_for(REF_DATE)
    Xr, yr, _, fsr = sw(dref, sref)
    fr, rr = rsq_map(Xr, yr, fsr)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.9), constrained_layout=True,
                             width_ratios=[1.35, 1.35, 1])
    v = float(max(np.abs(r).max(), np.abs(rr).max()))
    for ax, mat, ff, ttl in ((axes[0], r, f, f"S06 (significant, 67.0 %)"),
                             (axes[1], rr, fr, f"{sref} (n.s., 56.0 %)")):
        im = ax.imshow(np.sign(mat) * mat ** 2, cmap="RdBu_r", aspect="auto",
                       extent=[ff[0], ff[-1], len(CH) - 0.5, -0.5],
                       norm=TwoSlopeNorm(vmin=-v ** 2, vcenter=0, vmax=v ** 2),
                       interpolation="nearest")
        ax.set_yticks(range(len(CH)), CH)
        ax.set_xlabel("frequency (Hz)")
        ax.axvspan(30, 45, color="#bc4c00", alpha=0.07)
        ax.text(37, -0.9, "EMG range", fontsize=7, color="#bc4c00", ha="center")
        ax.grid(False)
        ax.set_title(f"signed r²  Stand ←→ Walk — {ttl}", loc="left", fontsize=10)
    fig.colorbar(im, ax=axes[1], shrink=0.9, label="← Stand      Walk →")

    ab = [x for x in rows if x["check"] == "band_ablation"]
    names = [x["band"] for x in ab]
    vals = [x["value"] * 100 for x in ab]
    cols = ["#bc4c00" if "EMG" in n else "#1f6feb" for n in names]
    axes[2].barh(np.arange(len(ab)), vals, color=cols)
    axes[2].axvline(50, color=INK, lw=1, ls=":")
    axes[2].set_yticks(np.arange(len(ab)), names, fontsize=8)
    axes[2].set_xlabel("S06 accuracy from that band alone (%)")
    axes[2].set_title("band ablation", loc="left", fontsize=10)
    fig.suptitle("Is S06's Stand-vs-Walk effect cortical or muscular?",
                 ha="left", x=0.01, fontsize=11)
    fig.savefig(os.path.join(R_DIR, "fig_s06_physiology.png"), bbox_inches="tight")
    plt.close(fig)
    print("\nwrote", os.path.join(R_DIR, "fig_s06_physiology.png"))


if __name__ == "__main__":
    main()
