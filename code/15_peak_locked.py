# -*- coding: utf-8 -*-
"""
Stage 15 — peak-locked analysis of the compound-limb moment, and whether a
classifier can use it.

Motivation
----------
Stage 14 found the only directionally-predicted effect in this dataset: at the
instant of maximum laterality, Walk sits further toward the right hemisphere
than Stand (+0.63 dB mu, p = .16; +1.02 dB beta, p = .083).  Consistent with
step 1 being right leg + LEFT arm.  Not significant at n = 10, but the sign is
the predicted one in both bands.

If that moment exists but occurs at a different time in every trial, then
stimulus-locked averaging smears it.  Realigning each trial to its own peak
before averaging — response-locked averaging — is the standard fix.

THE CIRCULARITY GUARD (read before trusting any of this)
--------------------------------------------------------
Selecting a peak by |LI| and then plotting LI around that peak produces a peak
BY CONSTRUCTION, even in pure noise.  So nothing here is interpreted from a
single class.  Every quantity is computed with the identical procedure for
Rest, Stand and Walk, and only DIFFERENCES BETWEEN CLASSES are read.  Rest
acts as the noise reference: whatever peak-locking does to noise, it does to
Rest too.

Peak selection never sees the trial label, so it adds no leakage to the
decoding in Part B.

Parts
-----
A  Peak-locked band-power time courses, per class, with Rest as reference
B  Topography at the peak moment: Walk, Stand, and Walk - Stand
C  PSD at the peak moment, split by hemisphere
D  Decoding: can peak-derived features separate Stand from Walk, and do they
   add anything to the covariance pipeline?

Output: results/peak_locked.csv, fig_peak_locked.png
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
from scipy.signal import hilbert, welch
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import decoders as D          # noqa: E402
import steam_io as sio        # noqa: E402

R_DIR = sio.RESULTS_DIR
CH = sio.EEG_CHANNELS
RIGHT, LEFT = ["FC4", "C4", "CP4"], ["FC3", "C3", "CP3"]
BANDS = {"mu 8-13": (8, 13), "beta 13-30": (13, 30)}
TMIN, TMAX = 0.0, 5.0
HALF_S = 1.0                  # +-1 s around each trial's own peak
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
SER = {"Rest": "#6b7280", "Stand": "#1f6feb", "Walk": "#bc4c00"}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.dpi": 170,
})


def mne_info():
    import mne
    info = mne.create_info(CH, 256.0, "eeg", verbose=False)
    info.set_montage("standard_1020", on_missing="ignore", verbose=False)
    return info


def load(date, subj, lo, hi):
    """
    Band-filtered AND broadband views of the SAME trials.

    Loading them with two separate get_epochs calls silently desynchronises
    them: artifact rejection is variance-based and therefore band-dependent,
    so the two calls can keep different trials.  get_bank makes one rejection
    decision for every band it is given.
    """
    bk = sio.get_bank(date, subj, bands=((lo, hi), (1.0, 45.0)),
                      tmin=TMIN, tmax=TMAX, ref="none")
    Xband, Xbroad, y = bk["Xb"][0], bk["Xb"][1], bk["y"]
    fs = bk["fs"]
    env = np.abs(hilbert(Xband, axis=-1)) ** 2
    k = max(1, int(round(0.25 * fs)))
    kern = np.ones(k) / k
    env = np.apply_along_axis(lambda v: np.convolve(v, kern, mode="same"), -1, env)
    Rp = env[:, [CH.index(c) for c in RIGHT]].mean(axis=1)
    Lp = env[:, [CH.index(c) for c in LEFT]].mean(axis=1)
    LI = 10.0 * np.log10((Rp + 1e-30) / (Lp + 1e-30))
    return Xband, Xbroad, env, LI, y, fs


def peak_index(LI, fs, half_s=HALF_S):
    """Per trial, the index of max |LI| — restricted so a full +-half_s window
    fits inside the trial.  Uses no labels."""
    h = int(round(half_s * fs))
    seg = np.abs(LI[:, h:LI.shape[1] - h])
    return seg.argmax(axis=1) + h, h


def main():
    subs = sio.list_subjects()
    rows = []
    store = {}

    for bname, (lo, hi) in BANDS.items():
        print(f"\n{'='*74}\n### {bname}\n{'='*74}")
        locked = {c: [] for c in sio.CLASS_NAMES}      # peak-locked LI courses
        topo = {c: [] for c in sio.CLASS_NAMES}        # per-channel power at peak
        psd_h = {c: {"L": [], "R": []} for c in sio.CLASS_NAMES}
        acc_peak, acc_cov, acc_both = [], [], []

        for date, subj in subs:
            X, Xb, env, LI, y, fs = load(date, subj, lo, hi)
            pk, h = peak_index(LI, fs)
            n = len(y)

            # ---- peak-locked LI and per-channel power ----
            idx = pk[:, None] + np.arange(-h, h + 1)[None, :]
            LIl = np.take_along_axis(LI, idx, axis=1)
            envw = np.stack([env[i, :, pk[i] - h:pk[i] + h + 1] for i in range(n)])
            chan_pow = np.log10(envw.mean(axis=-1) + 1e-30)          # (n, 8)

            for ci, cname in enumerate(sio.CLASS_NAMES):
                m = y == ci
                locked[cname].append(LIl[m].mean(axis=0))
                topo[cname].append(chan_pow[m].mean(axis=0))

            # ---- PSD in the peak window, by hemisphere (broadband) ----
            Xw = np.stack([Xb[i, :, pk[i] - h:pk[i] + h + 1] for i in range(n)])
            f, P = welch(Xw, fs=fs, nperseg=min(Xw.shape[-1], int(fs)), axis=-1)
            fs_sel = (f >= 4) & (f <= 40)
            for ci, cname in enumerate(sio.CLASS_NAMES):
                m = y == ci
                for side, chs in (("L", LEFT), ("R", RIGHT)):
                    idxc = [CH.index(c) for c in chs]
                    psd_h[cname][side].append(
                        np.log10(P[m][:, idxc][:, :, fs_sel].mean(axis=1) + 1e-30
                                 ).mean(axis=0))

            # ---- D: decoding ----
            sel = np.isin(y, [1, 2])
            yy = (y[sel] == 2).astype(int)
            cv = StratifiedKFold(5, shuffle=True, random_state=42)

            # peak-derived scalar features (no labels used to build them)
            feat = np.column_stack([
                LI[np.arange(n), pk],                       # signed LI at peak
                np.abs(LI[np.arange(n), pk]),               # magnitude
                pk / fs,                                    # latency
                LIl.mean(axis=1),                           # mean LI in window
                chan_pow,                                   # 8 channel powers
            ])
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=5000, C=0.3))
            acc_peak.append(cross_val_score(clf, feat[sel], yy, cv=cv).mean())

            Xw_band = np.stack([X[i, :, pk[i] - h:pk[i] + h + 1] for i in range(n)])
            acc_cov.append(cross_val_score(D.get("ts_lr"), Xw_band[sel], yy,
                                           cv=cv).mean())

            from pyriemann.estimation import Covariances
            from pyriemann.tangentspace import TangentSpace
            ts = make_pipeline(Covariances("oas"), TangentSpace("riemann"))
            both = np.column_stack([ts.fit_transform(Xw_band), feat])
            acc_both.append(cross_val_score(clf, both[sel], yy, cv=cv).mean())

        # ---------------- report ----------------
        pk_lat = {c: np.stack(locked[c]) for c in sio.CLASS_NAMES}
        print("\nA  peak-locked LI at the peak instant (dB, mean over "
              "participants)")
        mid = pk_lat["Rest"].shape[1] // 2
        for c in sio.CLASS_NAMES:
            print(f"    {c:6s} {pk_lat[c][:, mid].mean():+6.3f}")
        dw = pk_lat["Walk"][:, mid] - pk_lat["Stand"][:, mid]
        tt, pp = ttest_rel(pk_lat["Walk"][:, mid], pk_lat["Stand"][:, mid])
        print(f"    Walk - Stand at peak = {dw.mean():+.3f} dB  "
              f"t={tt:+.2f}  p={pp:.3f}")
        rows.append(dict(band=bname, test="peak_LI_WalkMinusStand",
                         value=float(dw.mean()), p=float(pp)))

        print("\nB  topography at the peak window, Walk - Stand (log10 power)")
        TW, TS = np.stack(topo["Walk"]), np.stack(topo["Stand"])
        d = (TW - TS).mean(axis=0)
        for c, v in zip(CH, d):
            print(f"    {c:5s} {v:+.4f}")
        rows += [dict(band=bname, test=f"topo_diff_{c}", value=float(v), p=np.nan)
                 for c, v in zip(CH, d)]

        print("\nD  decoding Stand vs Walk from the peak window")
        for nm, a in (("peak scalar features", acc_peak),
                      ("covariance in peak window", acc_cov),
                      ("both combined", acc_both)):
            a = np.array(a)
            print(f"    {nm:28s} {a.mean()*100:5.1f}% ± {a.std()*100:4.1f}"
                  f"   per-subj " + " ".join(f"{v*100:.0f}" for v in a))
            rows.append(dict(band=bname, test=f"decode_{nm.replace(' ','_')}",
                             value=float(a.mean()), p=np.nan))

        store[bname] = dict(locked=pk_lat, topo=(TW, TS, np.stack(topo["Rest"])),
                            psd=psd_h,
                            f=f[fs_sel], h=h, fs=fs,
                            acc=(np.array(acc_peak), np.array(acc_cov),
                                 np.array(acc_both)))

    pd.DataFrame(rows).to_csv(os.path.join(R_DIR, "peak_locked.csv"), index=False)
    make_figure(store)


def make_figure(store):
    import mne
    info = mne_info()
    bands = list(store)
    fig = plt.figure(figsize=(15.5, 4.3 * len(bands)))
    gs = fig.add_gridspec(len(bands), 5, width_ratios=[1.35, .75, .75, .75, 1.25],
                          hspace=0.55, wspace=0.35)

    for r, bname in enumerate(bands):
        s = store[bname]
        h, fs = s["h"], s["fs"]
        t = (np.arange(2 * h + 1) - h) / fs

        # --- A: peak-locked LI ---
        ax = fig.add_subplot(gs[r, 0])
        for c in sio.CLASS_NAMES:
            M = s["locked"][c]
            m, se = M.mean(0), M.std(0) / np.sqrt(len(M))
            ax.plot(t, m, color=SER[c], lw=2, label=c)
            ax.fill_between(t, m - se, m + se, color=SER[c], alpha=.18, lw=0)
        ax.axvline(0, color=INK, lw=1.2)
        ax.set_xlabel("time from that trial's own peak (s)")
        ax.set_ylabel("laterality index (dB)")
        ax.set_title(f"A  peak-locked LI — {bname}\nRest is the noise reference: "
                     "peak-locking creates a bump\nin any signal, so read only "
                     "the class differences",
                     loc="left", fontsize=8.6, linespacing=1.35)
        if r == 0:
            ax.legend(frameon=False, ncols=3, fontsize=7.6)

        # --- B: topographies ---
        # All three are DIFFERENCES on one shared symmetric scale.  Plotting
        # raw log-power for Walk and Stand auto-scales each map to its own
        # absolute level, which saturates both and shows nothing.
        TW, TS, TR = s["topo"]
        maps = [("Walk − Rest", (TW - TR).mean(0)),
                ("Stand − Rest", (TS - TR).mean(0)),
                ("Walk − Stand", (TW - TS).mean(0))]
        lim = max(np.abs(v).max() for _, v in maps)
        for k, (ttl, vals) in enumerate(maps):
            ax = fig.add_subplot(gs[r, 1 + k])
            im, _ = mne.viz.plot_topomap(
                vals, info, axes=ax, show=False, cmap="RdBu_r",
                vlim=(-lim, lim), contours=0, sensors=True)
            ax.set_title(ttl, fontsize=8.6)
            if k == 2:
                cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.04)
                cb.ax.tick_params(labelsize=6)
                cb.set_label("Δ log10 power", fontsize=6.5)

        # --- D: decoding ---
        ax = fig.add_subplot(gs[r, 4])
        ap, ac, ab = s["acc"]
        x = np.arange(len(ap))
        for a, lab, col in ((ap, "peak features", SER["Walk"]),
                            (ac, "covariance", SER["Stand"]),
                            (ab, "both", "#116329")):
            ax.plot(x, a * 100, "o-", ms=3.6, lw=1.4, color=col,
                    label=f"{lab}  ({a.mean()*100:.1f}%)")
        ax.axhline(50, color=INK, lw=1, ls=":")
        ax.set_xticks(x, [f"S{i+1}" for i in x], fontsize=6.4)
        ax.set_ylabel("Stand vs Walk (%)")
        ax.legend(frameon=False, fontsize=6.8)
        ax.set_title(f"D  decoding from the peak window — {bname}",
                     loc="left", fontsize=8.6)

    fig.suptitle("Peak-locked analysis of the compound-limb moment  "
                 "(each trial realigned to its own maximum-laterality instant)",
                 ha="left", x=0.008, fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(R_DIR, "fig_peak_locked.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("\nwrote", os.path.join(R_DIR, "fig_peak_locked.png"))


if __name__ == "__main__":
    main()
