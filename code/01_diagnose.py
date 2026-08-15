# -*- coding: utf-8 -*-
"""
Stage 0 — is there any neurophysiological signal in here at all?

Produces, under results/:
  quality.csv            per-subject / per-channel amplitude + clipping table
  fig_quality.png        that table as a heatmap
  fig_erds_<subj>.png    ERD/ERS time-frequency maps, one row per class
  fig_erds_grand.png     grand average over subjects
  fig_erd_bands.png      mu / beta ERD per channel per class (grand average)
  fig_rsq_<pair>.png     channel x frequency discriminability (point-biserial r^2)

Baseline for ERD/ERS is the fixation screen, t in [-4.5, -2.5] relative to the
task marker (trial structure: fixation 3 s -> cue 2 s -> task 5 s, so t = -5
is fixation onset and t = -2 is cue onset).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import steam_io as sio  # noqa: E402

R = sio.RESULTS_DIR
CH = sio.EEG_CHANNELS
CLASSES = sio.CLASS_NAMES

# --- palette -------------------------------------------------------------
# ERD/ERS is polarity data (desync < 0 < sync) -> diverging, neutral midpoint.
CMAP_DIV = "RdBu_r"
# magnitude only -> single hue, light to dark.
CMAP_SEQ = "Blues"
INK, INK_MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
SERIES = {"Rest": "#6b7280", "Stand": "#1f6feb", "Walk": "#bc4c00"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "figure.dpi": 130,
})


# ========================== 1. quality table =========================
def quality_table() -> pd.DataFrame:
    rows = []
    for date, subj in sio.list_subjects():
        for tag, _ in sio.run_paths(date, subj, "MI"):
            r = sio.load_run(date, subj, tag)
            uv = r["data"] * 1e6
            for i, ch in enumerate(r["ch_names"]):
                rows.append(dict(subject=f"{date}/{subj}", run=tag, ch=ch,
                                 std_uv=float(uv[i].std()),
                                 p2p_uv=float(np.percentile(uv[i], 99.5)
                                              - np.percentile(uv[i], 0.5)),
                                 clip_pct=float(r["clip"][i].mean() * 100)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(R, "quality.csv"), index=False)

    piv_clip = df.pivot_table(index="subject", columns="ch", values="clip_pct",
                              aggfunc="mean").reindex(columns=CH)
    piv_std = df.pivot_table(index="subject", columns="ch", values="std_uv",
                             aggfunc="mean").reindex(columns=CH)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, piv, title, unit in (
            (axes[0], piv_std, "Channel amplitude (std)", "µV"),
            (axes[1], piv_clip, "Amplifier saturation", "% of recording")):
        im = ax.imshow(piv.values, cmap=CMAP_SEQ, aspect="auto",
                       vmin=0, vmax=np.nanpercentile(piv.values, 95))
        ax.set_xticks(range(len(CH)), CH)
        ax.set_yticks(range(len(piv)), piv.index, fontsize=8)
        ax.set_title(title, loc="left")
        ax.grid(False)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v) and v > np.nanpercentile(piv.values, 60):
                    ax.text(j, i, f"{v:.0f}" if unit == "µV" else f"{v:.1f}",
                            ha="center", va="center", fontsize=6.5, color="white")
        fig.colorbar(im, ax=ax, shrink=0.85, label=unit)
    fig.suptitle("Recording quality per channel  (saturation falls in the "
                 "between-block breaks, not in trials)", ha="left", x=0.01, fontsize=11)
    fig.savefig(os.path.join(R, "fig_quality.png"), bbox_inches="tight")
    plt.close(fig)
    return df


# ============================ 2. ERD / ERS ===========================
FREQS = np.arange(4, 41, 1.5)
BASE = (-4.5, -2.5)


def subject_tfr(date, subj):
    """
    ERD/ERS in dB, shape (n_class, n_ch, n_freq, n_time).

    Each trial is normalised by *its own* fixation-period power before
    averaging, and the average is taken in the log domain.  Averaging raw
    power first lets a single high-amplitude trial dominate the mean and
    paints vertical stripes across the map; the log/geometric mean does not.
    """
    from mne.time_frequency import tfr_array_morlet

    ep = sio.get_epochs(date, subj, l_freq=1.0, h_freq=45.0,
                        tmin=-5.0, tmax=5.0, ref="car")
    fs = ep["fs"]
    n_cycles = np.clip(FREQS / 2.0, 3, 12)
    power = tfr_array_morlet(ep["X"], fs, FREQS, n_cycles=n_cycles,
                             output="power", decim=8, n_jobs=1, verbose=False)
    times = np.arange(power.shape[-1]) * 8 / fs - 5.0
    bmask = (times >= BASE[0]) & (times <= BASE[1])

    base = power[:, :, :, bmask].mean(axis=-1, keepdims=True)      # per trial
    rel_db = 10.0 * np.log10(power / (base + 1e-30) + 1e-30)
    return (np.stack([rel_db[ep["y"] == ci].mean(axis=0) for ci in range(len(CLASSES))]),
            times, ep["ch_names"])


def plot_erds(erds, times, ch_names, title, path, vlim=3.0):
    key = ["C3", "Cz", "C4", "CPz"]
    idx = [ch_names.index(c) for c in key]
    fig, axes = plt.subplots(len(CLASSES), len(key),
                             figsize=(2.5 * len(key), 2.1 * len(CLASSES)),
                             sharex=True, sharey=True, constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    for ri, cname in enumerate(CLASSES):
        for cj, (ch, ii) in enumerate(zip(key, idx)):
            ax = axes[ri, cj]
            im = ax.pcolormesh(times, FREQS, erds[ri, ii], cmap=CMAP_DIV,
                               norm=norm, shading="gouraud", rasterized=True)
            ax.axvline(0, color=INK, lw=1.0)
            ax.axvline(-2, color=INK_MUTED, lw=0.8, ls="--")
            ax.grid(False)
            if ri == 0:
                ax.set_title(ch, loc="center")
            if cj == 0:
                ax.set_ylabel(f"{cname}\nHz")
            if ri == len(CLASSES) - 1:
                ax.set_xlabel("time (s)")
    fig.colorbar(im, ax=axes, shrink=0.6, label="ERD / ERS  (dB vs fixation)")
    fig.suptitle(f"{title}   —   dashed = cue onset, solid = imagery onset",
                 ha="left", x=0.01, fontsize=11)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def erd_band_summary(all_erds, times, ch_names):
    """mu (8-13) and beta (13-30) ERD averaged over the MI window, per channel."""
    tmask = (times >= 0.5) & (times <= 4.5)
    bands = {"mu 8–13 Hz": (8, 13), "beta 13–30 Hz": (13, 30)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    x = np.arange(len(CH))
    for ax, (bname, (lo, hi)) in zip(axes, bands.items()):
        fmask = (FREQS >= lo) & (FREQS <= hi)
        for k, cname in enumerate(CLASSES):
            v = all_erds[:, k][:, :, fmask][:, :, :, tmask].mean(axis=(2, 3))  # (subj, ch)
            m, se = v.mean(0), v.std(0) / np.sqrt(v.shape[0])
            order = [ch_names.index(c) for c in CH]
            ax.errorbar(x + (k - 1) * 0.18, m[order], yerr=se[order], fmt="o-",
                        ms=5, lw=2, capsize=3, color=SERIES[cname], label=cname)
        ax.axhline(0, color=INK, lw=1)
        ax.set_xticks(x, CH)
        ax.set_title(bname, loc="left")
        ax.set_ylabel("ERD / ERS  (dB)")
    axes[0].legend(frameon=False, ncols=3, loc="lower left")
    fig.suptitle("Band-limited ERD/ERS over the imagery window (mean ± s.e.m. "
                 f"across {all_erds.shape[0]} subjects)", ha="left", x=0.01, fontsize=11)
    fig.savefig(os.path.join(R, "fig_erd_bands.png"), bbox_inches="tight")
    plt.close(fig)


# ==================== 3. discriminability r^2 map ====================
def rsq_maps():
    """Point-biserial r^2 between log band-power and class, per channel x freq."""
    from scipy.signal import welch

    pairs = [("Stand", "Walk"), ("Rest", "Stand"), ("Rest", "Walk")]
    store = {p: [] for p in pairs}
    subs = sio.list_subjects()
    freqs = None
    for date, subj in subs:
        ep = sio.get_epochs(date, subj, l_freq=1.0, h_freq=45.0,
                            tmin=0.5, tmax=4.5, ref="car")
        f, P = welch(ep["X"], fs=ep["fs"], nperseg=int(ep["fs"]), axis=-1)
        sel = (f >= 4) & (f <= 40)
        f, P = f[sel], np.log(P[:, :, sel] + 1e-30)
        freqs = f
        for a, b in pairs:
            ia, ib = CLASSES.index(a), CLASSES.index(b)
            m = np.isin(ep["y"], [ia, ib])
            yy = (ep["y"][m] == ib).astype(float)
            Z = P[m]
            zc = Z - Z.mean(0)
            yc = yy - yy.mean()
            r = (zc * yc[:, None, None]).sum(0) / (
                np.sqrt((zc ** 2).sum(0)) * np.sqrt((yc ** 2).sum()) + 1e-30)
            store[(a, b)].append(np.sign(r) * r ** 2)

    for (a, b), mats in store.items():
        M = np.stack(mats)                              # (subj, ch, f)
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True,
                                 width_ratios=[1.6, 1])
        g = M.mean(0)
        v = float(np.abs(g).max())
        im = axes[0].imshow(g, cmap=CMAP_DIV, aspect="auto", origin="upper",
                            extent=[freqs[0], freqs[-1], len(CH) - 0.5, -0.5],
                            norm=TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v),
                            interpolation="nearest")
        axes[0].set_yticks(range(len(CH)), CH)
        axes[0].set_xlabel("frequency (Hz)")
        axes[0].grid(False)
        axes[0].set_title(f"signed r²  ({a} ← → {b}),  mean over subjects", loc="left")
        fig.colorbar(im, ax=axes[0], shrink=0.9, label=f"← {a}      {b} →")

        pk = np.abs(M).max(axis=(1, 2))
        axes[1].bar(range(len(subs)), pk, color="#1f6feb", width=0.68)
        axes[1].set_xticks(range(len(subs)),
                           [f"{d[2:]}/{s}" for d, s in subs], rotation=60,
                           ha="right", fontsize=7)
        axes[1].set_ylabel("peak |r²|")
        axes[1].set_title("strongest single channel×freq effect per subject", loc="left")
        fig.savefig(os.path.join(R, f"fig_rsq_{a}_vs_{b}.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"[rsq] {a} vs {b}: grand-mean peak |r²| = {np.abs(g).max():.4f} "
              f"at {CH[int(np.abs(g).max(1).argmax())]} "
              f"{freqs[int(np.abs(g).max(0).argmax())]:.0f} Hz")


# ================================ main ===============================
def main():
    print("== quality ==")
    q = quality_table()
    print(q.groupby("subject")[["std_uv", "clip_pct"]].max().round(1).to_string())

    print("\n== ERD/ERS ==")
    all_erds, times, ch_names = [], None, None
    for date, subj in sio.list_subjects():
        e, times, ch_names = subject_tfr(date, subj)
        all_erds.append(e)
        plot_erds(e, times, ch_names, f"ERD/ERS — {date}/{subj}",
                  os.path.join(R, f"fig_erds_{date}_{subj}.png"))
        print(f"  {date}/{subj} done")
    A = np.stack(all_erds)
    plot_erds(A.mean(0), times, ch_names,
              f"ERD/ERS — grand average, n={A.shape[0]} subjects",
              os.path.join(R, "fig_erds_grand.png"), vlim=2.0)
    erd_band_summary(A, times, ch_names)
    np.savez_compressed(os.path.join(R, "erds.npz"), erds=A, times=times,
                        freqs=FREQS, ch=np.array(ch_names))

    print("\n== discriminability ==")
    rsq_maps()
    print("\nfigures in", R)


if __name__ == "__main__":
    main()
