# -*- coding: utf-8 -*-
"""
Figure 3 — the whole result in one panel.

The individual analysis figures are each correct but none of them answers
"what happened?" on its own.  This assembles the four claims a reader needs:

  A  what decodes, and in how many participants
  B  the bound on the Stand-vs-Walk null — nested versus the deliberately
     optimistic best-of-840 search
  C  the ceiling — more trials, and more electrodes
  D  why the one significant participant does not count

Reads only CSVs written by earlier stages; runs in seconds.
Writes results/fig_summary.{png,pdf}.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steam_io as sio  # noqa: E402

R = sio.RESULTS_DIR
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
BLUE, ORANGE, GREEN, GREY = "#1f6feb", "#bc4c00", "#116329", "#9aa4b2"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.dpi": 200, "pdf.fonttype": 42, "ps.fonttype": 42,
})

NICE = {"RvS": "Rest\nvs Stand", "RvW": "Rest\nvs Walk",
        "SvW": "Stand\nvs Walk", "3cls": "3-class"}


def panel_a(ax):
    fs = pd.read_csv(os.path.join(R, "final_scores.csv"))
    perm = pd.read_csv(os.path.join(R, "stats_permutation.csv"))
    nested = fs[fs.kind == "nested"]
    order = ["RvS", "RvW", "SvW", "3cls"]
    for i, task in enumerate(order):
        v = nested[nested.task == task]["acc"].values * 100
        chance = 100 / (3 if task == "3cls" else 2)
        ax.bar(i, v.mean(), 0.58, color=BLUE, zorder=2)
        ax.scatter(np.full(len(v), i) + np.linspace(-.16, .16, len(v)), v,
                   s=13, color="white", edgecolor=INK, linewidth=0.6, zorder=3)
        ax.plot([i - .33, i + .33], [chance, chance], color=INK, lw=1.4,
                ls=":", zorder=4)
        n_sig = int((perm[perm.task == task]["p_holm"] < .05).sum())
        ax.text(i, 97, f"{n_sig}/10", ha="center", va="top", fontsize=9.4,
                fontweight="bold", color=BLUE if n_sig >= 5 else MUTED)
    ax.set_xticks(range(4), [NICE[t] for t in order], fontsize=8.2)
    ax.set_ylim(30, 100)
    ax.set_ylabel("nested accuracy (%)")
    ax.set_title("A   What decodes\nbar = group mean, dots = participants, "
                 "dotted = chance;\nn/10 = participants individually "
                 "significant", loc="left", fontsize=9.5, linespacing=1.4)


def panel_b(ax):
    fs = pd.read_csv(os.path.join(R, "final_scores.csv"))
    sw = pd.read_csv(os.path.join(R, "svw_focus.csv"))
    nested = (fs[(fs.kind == "nested") & (fs.task == "SvW")]
              .set_index("subject")["acc"] * 100)
    biased = sw.groupby("subject")["acc"].max() * 100
    idx = nested.sort_values().index
    y = np.arange(len(idx))
    ax.barh(y, biased.reindex(idx).values, 0.72, color=ORANGE, alpha=.35,
            label="best of 840 configs,\nscored on the reported folds", zorder=2)
    ax.barh(y, nested.reindex(idx).values, 0.42, color=BLUE,
            label="nested (unbiased)", zorder=3)
    ax.axvline(50, color=INK, lw=1.2, ls=":", zorder=4)
    ax.set_yticks(y, [s.split("/")[1] for s in idx], fontsize=7.6)
    ax.set_xlim(40, 80)
    ax.set_xlabel("Stand vs Walk accuracy (%)")
    ax.legend(frameon=False, fontsize=7.2, loc="lower right")
    ax.set_title("B   The null is bounded\neven the cherry-picked bound stays "
                 "near chance", loc="left", fontsize=9.5, linespacing=1.4)


def panel_c(ax):
    t = pd.read_csv(os.path.join(R, "ceiling_trials.csv"))
    c = pd.read_csv(os.path.join(R, "ceiling_channels.csv"))
    for task, col in (("SvW", BLUE), ("3cls", ORANGE)):
        m = t[t.task == task].groupby("per_class")["acc"].mean() * 100
        ax.plot(m.index, m.values, "o-", color=col, lw=2, ms=5,
                label=f"{task}  (trials)")
    ax.set_xscale("log")
    ax.set_xlabel("training trials per class")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(30, 80)
    ax.legend(frameon=False, fontsize=7.4, loc="lower right")

    ax2 = ax.inset_axes([0.56, 0.60, 0.41, 0.30])
    for task, col in (("SvW", BLUE), ("3cls", ORANGE)):
        m = c[c.task == task].groupby("k")["acc"].mean() * 100
        ax2.plot(m.index, m.values, "o-", color=col, lw=1.4, ms=3.4)
    ax2.set_xlabel("electrodes", fontsize=6.6, labelpad=1)
    ax2.tick_params(labelsize=6.2)
    ax2.set_title("flat past 3", fontsize=6.8, color=MUTED, pad=2)
    ax.set_title("C   The ceiling\n+1.7 pts for double the session",
                 loc="left", fontsize=9.5, linespacing=1.4)


def panel_d(ax):
    p = pd.read_csv(os.path.join(R, "s06_physiology.csv"))
    ab = p[p.check == "band_ablation"]
    names = [b.replace(" (EMG)", "") for b in ab["band"]]
    vals = ab["value"].values * 100
    cols = [ORANGE if ("gamma" in b or "delta" in b) else BLUE
            for b in ab["band"]]
    y = np.arange(len(names))
    ax.barh(y, vals, 0.68, color=cols, zorder=2)
    ax.axvline(50, color=INK, lw=1.2, ls=":", zorder=3)
    ax.set_yticks(y, names, fontsize=7.6)
    ax.set_xlim(40, 78)
    ax.set_xlabel("accuracy from that band alone (%)")
    ax.text(0.985, 0.06,
            "orange = bands with no sensorimotor rhythm.\n"
            "They decode as well as mu and beta, so the\n"
            "effect is broadband state, not SMR.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0,
            color=MUTED, linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f6f8fa",
                      edgecolor=GRID, linewidth=0.8))
    ax.set_title("D   The one 'significant' participant\nis not decoding "
                 "sensorimotor rhythm", loc="left", fontsize=9.5,
                 linespacing=1.4)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2), constrained_layout=True)
    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])
    fig.suptitle("Lower-limb motor imagery on an 8-channel wearable montage — "
                 "summary of results",
                 ha="left", x=0.008, fontsize=12.5, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(R, f"fig_summary.{ext}"),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.join(R, "fig_summary.{png,pdf}"))


if __name__ == "__main__":
    main()
