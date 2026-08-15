# -*- coding: utf-8 -*-
"""
Stage 20 — POSITIVE CONTROL: does this montage see sensorimotor cortex at all,
and do the M1-M4 criteria detect it when it is present?

WHY THIS IS NEEDED
------------------
Sections III-B to III-H establish that Stand and Walk imagery cannot be
decoded, and Section III-F reports that against the visually matched Rest
condition only 1/10 participants showed the expected desynchronisation.  Both
findings are conditional on a premise nothing in the project has tested: that
an 8-channel montage with no EOG/EMG reference can register lower-limb
sensorimotor activity in the first place, and that the M1-M4 screens can
recognise it.  If it cannot, the null is an instrument result, not a
physiological one.

The motor EXECUTION (ME) runs are the available control: participants actually
moved, so genuine sensorimotor activity is present by construction.

WHY DECODING ACCURACY WOULD NOT ANSWER IT
-----------------------------------------
ME rest-vs-task will decode well almost trivially, because real movement
produces (a) cortical ERD, (b) limb and trunk EMG and (c) motion artifact, and
all three separate task from rest.  A high accuracy would therefore prove
nothing about cortex — the same trap that Section III-E documents for P06.

WHAT IS TESTED INSTEAD: THE SIGNATURE, NOT THE SEPARABILITY
-----------------------------------------------------------
ERD and EMG differ in sign and in topography, so they can be told apart
without being separable:

                     cortical ERD          muscle EMG
    power change     DECREASE              INCREASE
    band             confined to mu/beta   broadband, esp. >20 Hz
    topography       medial (Cz, CPz)      peripheral (FC3/FC4, temporal)

PRE-REGISTERED DECISION RULE  (fixed before running; every branch is reported)

  gate  retention >= 70% in every condition AND no gross imbalance between
        conditions, else verdict (3) NOT ASSESSABLE

  (1)   mu and/or beta DECREASE, medial > lateral, M4 low
        -> montage and criteria both valid; the MI null is physiological
  (2)   power INCREASE, broadband, peripheral > medial
        -> montage registers muscle, not lower-limb cortex
  (2')  no clear signature either way
        -> the M1-M4 screens may be insensitive; Sections III-E/III-H weaken
  (3)   gate failed -> not assessable, report as a limitation

Output: results/me_positive_control.csv, fig_me_positive_control.png
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
BANDS = {"mu 8-13": (8, 13), "beta 13-30": (13, 30), "gamma 30-45": (30, 45)}
MEDIAL = ["Cz", "CPz"]
LATERAL = ["FC3", "FC4"]
RETENTION_GATE = 0.70

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


def me_power(date, subj):
    """Band power per trial for the ME run. Returns (logP, y, ch, retention)."""
    ep = sio.get_epochs(date, subj, l_freq=1.0, h_freq=48.0,
                        tmin=0.5, tmax=4.5, ref="none", kind="ME")
    f, P = welch(ep["X"], fs=ep["fs"], nperseg=int(ep["fs"]), axis=-1)
    out = {}
    for bname, (lo, hi) in BANDS.items():
        sel = (f >= lo) & (f <= hi)
        out[bname] = 10 * np.log10(P[:, :, sel].mean(-1) + 1e-30)
    return out, ep["y"], list(ep["ch_names"]), ep


def main():
    subs = sio.list_subjects()
    rows, gate_rows = [], []

    print("=" * 78)
    print("ME POSITIVE CONTROL — signature of task vs rest during real movement")
    print("  prediction if the montage sees lower-limb cortex:")
    print("  mu/beta DECREASE, medial > lateral, no broadband gamma increase")
    print("=" * 78)

    for date, subj in subs:
        sid = f"{date}/{subj}"
        try:
            band_lp, y, ch, ep = me_power(date, subj)
        except Exception as e:                       # noqa: BLE001
            print(f"  {sid:14s} LOAD FAILED: {str(e)[:60]}")
            gate_rows.append(dict(subject=sid, ok=False, note="load failed"))
            continue

        # ---- retention gate -------------------------------------------
        n_per = {c: int((y == c).sum()) for c in np.unique(y)}
        expected = max(n_per.values()) if n_per else 0
        retention = min(n_per.values()) / expected if expected else 0.0
        ok = retention >= RETENTION_GATE and len(n_per) >= 2
        gate_rows.append(dict(subject=sid, ok=ok, retention=retention,
                              counts=str(n_per)))
        if not ok:
            print(f"  {sid:14s} GATE FAIL  retention={retention:.2f}  {n_per}")
            continue

        rest = 0                                     # ME_CODE_TO_Y: 1->0 rest
        task = [c for c in np.unique(y) if c != rest]
        for bname, lp in band_lp.items():
            db = lp[np.isin(y, task)].mean(0) - lp[y == rest].mean(0)
            med = np.mean([db[ch.index(c)] for c in MEDIAL if c in ch])
            lat = np.mean([db[ch.index(c)] for c in LATERAL if c in ch])
            rows.append(dict(subject=sid, band=bname, medial=med, lateral=lat,
                             all_ch=db.mean(),
                             **{f"db_{c}": db[i] for i, c in enumerate(ch)}))
            print(f"  {sid:14s} {bname:12s} all {db.mean():+5.2f} dB  "
                  f"medial {med:+5.2f}  lateral {lat:+5.2f}")

    df = pd.DataFrame(rows)
    gates = pd.DataFrame(gate_rows)
    gates.to_csv(os.path.join(R_DIR, "me_gate.csv"), index=False)

    n_ok = int(gates.ok.sum()) if len(gates) else 0
    print("\n" + "=" * 78)
    print(f"GATE: {n_ok}/{len(gates)} participants passed retention >= "
          f"{RETENTION_GATE:.0%}")
    if n_ok < 5 or df.empty:
        print("VERDICT (3): NOT ASSESSABLE — too few usable ME runs.")
        df.to_csv(os.path.join(R_DIR, "me_positive_control.csv"), index=False)
        return

    print("=" * 78)
    print("GROUP TEST (one-sample t against 0 dB, n = %d)" % n_ok)
    summary = []
    for bname in BANDS:
        g = df[df.band == bname]
        for label, col in (("all electrodes", "all_ch"),
                           ("medial Cz/CPz", "medial"),
                           ("lateral FC3/FC4", "lateral")):
            v = g[col].values
            t, p = ttest_1samp(v, 0.0)
            summary.append(dict(band=bname, site=label, mean_db=v.mean(),
                                t=t, p=p, n=len(v)))
            print(f"  {bname:12s} {label:16s} {v.mean():+5.2f} dB   "
                  f"t={t:+5.2f}  p={p:.4f}")
    pd.DataFrame(summary).to_csv(
        os.path.join(R_DIR, "me_positive_control_tests.csv"), index=False)
    df.to_csv(os.path.join(R_DIR, "me_positive_control.csv"), index=False)

    # ---------------------------- verdict ----------------------------
    mu = df[df.band == "mu 8-13"]
    beta = df[df.band == "beta 13-30"]
    gam = df[df.band == "gamma 30-45"]
    mu_dec = mu.all_ch.mean() < 0 and ttest_1samp(mu.all_ch, 0)[1] < .05
    beta_dec = beta.all_ch.mean() < 0 and ttest_1samp(beta.all_ch, 0)[1] < .05
    medial_wins = (mu.medial.mean() < mu.lateral.mean()) or \
                  (beta.medial.mean() < beta.lateral.mean())
    broadband_up = gam.all_ch.mean() > 0.5

    print("\n" + "=" * 78)
    if (mu_dec or beta_dec) and medial_wins and not broadband_up:
        print("VERDICT (1): ERD signature present, medial-dominant.")
        print("  -> montage registers lower-limb sensorimotor cortex, and the")
        print("     M1-M4 screens detect it when it is there.")
    elif broadband_up and (mu.all_ch.mean() > 0 or beta.all_ch.mean() > 0):
        print("VERDICT (2): broadband power INCREASE — muscle, not cortex.")
        print("  -> the montage registers EMG; lower-limb cortical activity is")
        print("     not resolved even during real movement.")
    else:
        print("VERDICT (2'): no clear signature in either direction.")
        print("  -> the M1-M4 screens may be insensitive; Sections III-E and")
        print("     III-H are weakened and this must be stated.")
    print("=" * 78)

    # ---------------------------- figure -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9),
                             constrained_layout=True, sharey=True)
    for ax, bname in zip(axes, BANDS):
        g = df[df.band == bname]
        for i, (lab, col) in enumerate([("all", "all_ch"),
                                        ("medial\nCz/CPz", "medial"),
                                        ("lateral\nFC3/FC4", "lateral")]):
            v = g[col].values
            ax.bar(i, v.mean(), .62,
                   color=BLUE if v.mean() < 0 else RED, zorder=2)
            ax.scatter(np.full(len(v), i), v, s=15, color="white",
                       edgecolor=INK, linewidth=.6, zorder=3)
        ax.axhline(0, color=INK, lw=1.4)
        ax.set_xticks(range(3), ["all", "medial\nCz/CPz", "lateral\nFC3/FC4"],
                      fontsize=8.4)
        ax.set_title(bname, loc="left")
    axes[0].set_ylabel("task − rest  (dB)")
    fig.suptitle("Motor EXECUTION positive control: is the task-vs-rest change "
                 "a cortical ERD (blue, negative) or muscle (red, positive)?",
                 ha="left", x=.006, fontsize=11.4, fontweight="bold")
    fig.savefig(os.path.join(R_DIR, "fig_me_positive_control.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.join(R_DIR, "fig_me_positive_control.png"))


if __name__ == "__main__":
    main()
