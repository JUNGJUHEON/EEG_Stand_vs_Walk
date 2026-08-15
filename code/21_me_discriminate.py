# -*- coding: utf-8 -*-
"""
Stage 21 — discriminating the two readings of the ME positive control.

Stage 20 found a large, highly significant task-vs-rest power DECREASE during
real lower-limb movement (mu -2.8 dB, p = .0015) that is neither band-limited
nor medially focused.  Two readings survive that result:

  A  the montage detects genuine cortical activity but cannot localise it
     (spatial resolution is the binding constraint, as Section III-H.4 argues)

  B  the M1-M4 screens are too strict — they would reject real movement, so
     the rejection of P06 in Section III-E was over-confident

Three tests separate them.  Each was chosen because A and B predict different
outcomes, and all three are reported whatever they show.

  T1  SPECTRAL SHAPE.  A cortical ERD peaks near the individual mu frequency.
      A global/multiplicative change is flat in dB across 8-45 Hz.
      A predicts a peak; B is agnostic; a flat profile favours neither cortex
      nor the criteria and would point to a global state change.

  T2  POST-MOVEMENT BETA REBOUND.  Beta ERS after movement offset is a
      specific cortical signature that arousal and EMG do not reproduce —
      EMG stops when movement stops, it does not overshoot upward.
      A predicts rebound; a global state change does not.

  T3  TASK-A vs TASK-B WITHIN ME.  The direct analogue of Stand-vs-Walk, but
      with real movement and therefore real EMG present.  If even executed
      movements cannot be told apart on this montage, the imagery null is not
      about imagery.

Output: results/me_discriminate.csv, fig_me_discriminate.png
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
from scipy.stats import ttest_1samp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import steam_io as sio        # noqa: E402

R_DIR = sio.RESULTS_DIR
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
BLUE, ORANGE, RED, GREEN = "#1f6feb", "#bc4c00", "#b3261e", "#1a7f37"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.dpi": 190,
})


def usable(y, gate=0.70):
    n = {c: int((y == c).sum()) for c in np.unique(y)}
    return len(n) >= 2 and min(n.values()) / max(n.values()) >= gate


# ----------------------------- T1: spectral shape -----------------------
def t1_spectrum():
    prof, freqs = [], None
    for date, s in sio.list_subjects():
        try:
            ep = sio.get_epochs(date, s, l_freq=1.0, h_freq=48.0,
                                tmin=0.5, tmax=4.5, ref="none", kind="ME")
        except Exception:                            # noqa: BLE001
            continue
        y = ep["y"]
        if not usable(y):
            continue
        f, P = welch(ep["X"], fs=ep["fs"], nperseg=int(ep["fs"] * 2), axis=-1)
        sel = (f >= 5) & (f <= 45)
        lp = 10 * np.log10(P[:, :, sel] + 1e-30).mean(1)      # trials x freq
        prof.append(lp[y != 0].mean(0) - lp[y == 0].mean(0))
        freqs = f[sel]
    return np.array(prof), freqs


# ------------------------- T2: post-movement rebound --------------------
def t2_timecourse():
    """Beta power over time, task trials, referenced to the pre-cue baseline."""
    curves, tc = [], None
    for date, s in sio.list_subjects():
        try:
            ep = sio.get_epochs(date, s, l_freq=13.0, h_freq=30.0,
                                tmin=-1.0, tmax=7.0, ref="none", kind="ME",
                                reject=False)
        except Exception:                            # noqa: BLE001
            continue
        y = ep["y"]
        if not usable(y):
            continue
        fs = ep["fs"]
        X = ep["X"][y != 0]                          # task trials only
        env = X ** 2                                 # power envelope
        win = int(fs * 0.4)
        k = np.ones(win) / win
        sm = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"),
                                 -1, env).mean(1)    # trials x time
        t = np.arange(sm.shape[-1]) / fs - 1.0
        base = sm[:, (t >= -1.0) & (t < -0.2)].mean(-1, keepdims=True)
        curves.append((10 * np.log10(sm / (base + 1e-30) + 1e-30)).mean(0))
        tc = t
    return np.array(curves), tc


# ---------------------- T3: TaskA vs TaskB within ME --------------------
def t3_taskAB():
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    rows = []
    for date, s in sio.list_subjects():
        try:
            ep = sio.get_epochs(date, s, l_freq=8.0, h_freq=30.0,
                                tmin=0.5, tmax=4.5, ref="none", kind="ME")
        except Exception:                            # noqa: BLE001
            continue
        y = ep["y"]
        m = y != 0                                   # the two executed tasks
        yy, XX = y[m], ep["X"][m]
        if len(np.unique(yy)) < 2 or not usable(yy):
            continue
        pipe = make_pipeline(Covariances("oas"), TangentSpace(),
                             StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        acc = cross_val_score(pipe, XX, yy, cv=cv, scoring="accuracy").mean()
        rows.append(dict(subject=f"{date}/{s}", acc=acc, n=len(yy)))
        print(f"    {date}/{s:5s}  TaskA vs TaskB  {acc*100:5.1f}%  (n={len(yy)})")
    return pd.DataFrame(rows)


def main():
    print("=" * 78)
    print("T1 — SPECTRAL SHAPE of the ME task-minus-rest change")
    print("=" * 78)
    prof, freqs = t1_spectrum()
    m = prof.mean(0)
    imu = (freqs >= 8) & (freqs <= 13)
    igam = (freqs >= 30) & (freqs <= 45)
    peak_f = freqs[np.argmin(m)]
    depth = m[imu].min() - m[igam].mean()
    print(f"  n = {len(prof)}   deepest point at {peak_f:.1f} Hz")
    print(f"  mu minimum {m[imu].min():+5.2f} dB   gamma mean {m[igam].mean():+5.2f} dB")
    print(f"  mu-relative-to-gamma depth = {depth:+.2f} dB")
    d_sub = np.array([p[imu].min() - p[igam].mean() for p in prof])
    t, p = ttest_1samp(d_sub, 0)
    print(f"  across participants: {d_sub.mean():+.2f} dB, t={t:+.2f}, p={p:.4f}")
    print("  -> " + ("a mu-band trough exists on top of the broadband shift"
                     if p < .05 and d_sub.mean() < 0 else
                     "no mu-specific trough: the change is spectrally flat"))

    print()
    print("=" * 78)
    print("T2 — POST-MOVEMENT BETA REBOUND (task trials, pre-cue baseline)")
    print("=" * 78)
    cur, tt = t2_timecourse()
    if len(cur):
        dur = np.array([c[(tt >= 1.0) & (tt <= 4.0)].mean() for c in cur])
        reb = np.array([c[(tt >= 5.2) & (tt <= 6.5)].mean() for c in cur])
        t1_, p1_ = ttest_1samp(dur, 0)
        t2_, p2_ = ttest_1samp(reb, 0)
        t3_, p3_ = ttest_1samp(reb - dur, 0)
        print(f"  n = {len(cur)}")
        print(f"  during movement  [1.0, 4.0] s : {dur.mean():+5.2f} dB  p={p1_:.4f}")
        print(f"  after  movement  [5.2, 6.5] s : {reb.mean():+5.2f} dB  p={p2_:.4f}")
        print(f"  rebound (after − during)      : {(reb-dur).mean():+5.2f} dB  p={p3_:.4f}")
        print("  -> " + ("beta REBOUND present — a cortical signature EMG cannot mimic"
                         if p3_ < .05 and (reb - dur).mean() > 0 else
                         "no significant rebound"))
    else:
        dur = reb = np.array([])

    print()
    print("=" * 78)
    print("T3 — TaskA vs TaskB WITHIN ME (real movement, real EMG present)")
    print("=" * 78)
    ab = t3_taskAB()
    if len(ab):
        t4_, p4_ = ttest_1samp(ab.acc.values, 0.5)
        print(f"\n  mean {ab.acc.mean()*100:.1f}%  (n={len(ab)})  "
              f"t={t4_:+.2f}  p={p4_:.4f}")
        print("  -> " + ("executed movements ARE separable"
                         if p4_ < .05 and ab.acc.mean() > .5 else
                         "even EXECUTED movements are not separable on this montage"))
        ab.to_csv(os.path.join(R_DIR, "me_taskAB.csv"), index=False)

    # ------------------------------ figure ---------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.9), constrained_layout=True)
    ax = axes[0]
    for p_ in prof:
        ax.plot(freqs, p_, color=MUTED, lw=.5, alpha=.4)
    ax.plot(freqs, m, color=BLUE, lw=2.0)
    ax.axhline(0, color=INK, lw=1.2)
    ax.axvspan(8, 13, color=ORANGE, alpha=.10)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("task − rest (dB)")
    ax.set_title("T1  spectral shape", loc="left")

    ax = axes[1]
    if len(cur):
        for c in cur:
            ax.plot(tt, c, color=MUTED, lw=.5, alpha=.4)
        ax.plot(tt, cur.mean(0), color=BLUE, lw=2.0)
        ax.axhline(0, color=INK, lw=1.2)
        ax.axvspan(0, 5, color=GRID, alpha=.55, zorder=0)
        ax.text(2.5, ax.get_ylim()[1]*.92, "movement", ha="center",
                fontsize=8, color=MUTED)
        ax.set_xlim(-1, 7)
    ax.set_xlabel("time from task onset (s)"); ax.set_ylabel("beta power (dB)")
    ax.set_title("T2  post-movement rebound", loc="left")

    ax = axes[2]
    if len(ab):
        ax.bar(0, ab.acc.mean()*100, .55,
               color=BLUE if ab.acc.mean() > .58 else RED, zorder=2)
        ax.scatter(np.zeros(len(ab)), ab.acc.values*100, s=18, color="white",
                   edgecolor=INK, linewidth=.7, zorder=3)
        ax.axhline(50, color=INK, lw=1.4, ls="--")
        ax.set_xticks([0], ["TaskA vs TaskB\n(executed)"], fontsize=8.4)
        ax.set_ylim(35, 85); ax.set_ylabel("accuracy (%)")
    ax.set_title("T3  executed-movement separability", loc="left")
    fig.suptitle("Discriminating the two readings of the ME positive control",
                 ha="left", x=.006, fontsize=11.4, fontweight="bold")
    fig.savefig(os.path.join(R_DIR, "fig_me_discriminate.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("\nwrote", os.path.join(R_DIR, "fig_me_discriminate.png"))


if __name__ == "__main__":
    main()
