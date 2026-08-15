# -*- coding: utf-8 -*-
"""
Figure 1 — paradigm timing and marker structure.

Redrawn as vector from the PsychoPy builder flow in the handover deck
(1. Bitbrain Rest_Stand_Walk 관련 인수인계.pptx, slide 16), which shows
    Setup -> loop_setup -> [ Fixation_prepare (3.00 s)
                             -> Current_Img (2.00 s)
                             -> White (5.00 s) ] x trials_2 x trials
and the three cue images: Rest = filled circle, Stand = up arrow,
Walk = right arrow.

Two things worth noting, both verified against the recordings:

  * The builder's 3.00 / 2.00 / 5.00 s matches the marker timing measured in
    all thirteen XDF files to within +-0.02 s.  The prose on the same slide
    says "Fixation_prepare (5 sec)"; that text is stale, the builder is not.

  * The imagery screen is blank white for ALL THREE classes.  Only the 2 s
    cue differs.  That is what makes Rest a visually matched baseline and is
    the reason the manipulation check in Section III-F is possible at all.

Writes results/fig_paradigm.{png,pdf}.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, FancyArrowPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steam_io as sio  # noqa: E402

R_DIR = sio.RESULTS_DIR

INK, MUTED, LINE = "#1f2328", "#6b7280", "#c9d1d9"
GREEN = "#4caf50"
ACCENT = "#1f6feb"
WIN_BG = "#eef4fd"

plt.rcParams.update({"font.size": 9, "text.color": INK, "figure.dpi": 200,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# (label, duration s, marker note)
SEGS = [("Fixation_prepare", 3.0), ("Current_Img (cue)", 2.0), ("White (imagery)", 5.0)]
CLASSES = ["circle", "uparrow", "rightarrow"]   # cue images, in list order


def draw_screen(ax, x, y, w, h, kind):
    """Miniature of the actual PsychoPy screen."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=INK,
                           linewidth=1.0, zorder=3))
    cx, cy = x + w / 2, y + h / 2
    if kind == "cross":
        ax.plot([cx - w * 0.16, cx + w * 0.16], [cy, cy], color=INK, lw=2.6, zorder=4)
        ax.plot([cx, cx], [cy - h * 0.22, cy + h * 0.22], color=INK, lw=2.6, zorder=4)
    elif kind == "circle":
        ax.add_patch(Circle((cx, cy), min(w, h) * 0.22, facecolor=GREEN,
                            edgecolor="none", zorder=4))
    elif kind == "uparrow":
        ax.add_patch(FancyArrow(cx, cy - h * 0.26, 0, h * 0.44, width=w * 0.13,
                                head_width=w * 0.30, head_length=h * 0.22,
                                facecolor=GREEN, edgecolor="none",
                                length_includes_head=True, zorder=4))
    elif kind == "rightarrow":
        ax.add_patch(FancyArrow(cx - w * 0.20, cy, w * 0.38, 0, width=h * 0.16,
                                head_width=h * 0.34, head_length=w * 0.18,
                                facecolor=GREEN, edgecolor="none",
                                length_includes_head=True, zorder=4))
    # "white" -> blank


def main():
    fig, ax = plt.subplots(figsize=(11.4, 4.5))
    total = sum(d for _, d in SEGS)
    t_task = SEGS[0][1] + SEGS[1][1]          # x-position of imagery onset
    ax.set_xlim(-2.1, total + 0.7)
    ax.set_ylim(-1.60, 3.95)
    ax.axis("off")

    y_bar, bar_h = 0.95, 0.62
    y_win = 0.44                               # analysis-window bar
    y_axis = 0.06                              # time axis

    # ---- timeline bar ----
    t = 0.0
    for label, dur in SEGS:
        ax.add_patch(Rectangle((t, y_bar), dur, bar_h, facecolor="white",
                               edgecolor=INK, linewidth=1.2, zorder=2))
        ax.text(t + dur / 2, y_bar + bar_h / 2, f"{label}\n{dur:.2f} s",
                ha="center", va="center", fontsize=8.6, zorder=3,
                linespacing=1.35)
        t += dur

    # ---- analysis window, as its own bar directly under the timeline ----
    ax.add_patch(Rectangle((t_task + 0.5, y_win), 4.0, 0.26,
                           facecolor=WIN_BG, edgecolor=ACCENT, linewidth=1.1,
                           zorder=2))
    ax.text(t_task + 2.5, y_win + 0.13, "analysis window",
            ha="center", va="center", fontsize=8.0, color=ACCENT, zorder=3)

    # ---- time axis, referenced to IMAGERY ONSET (matches the epoch notation
    #      used throughout the paper; a trial-relative axis forces the reader
    #      to hold two reference frames at once) ----
    ax.annotate("", xy=(total + 0.45, y_axis), xytext=(0, y_axis),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))
    for xpos, lab in ((0, "−5"), (3, "−2"), (t_task, "0"),
                      (t_task + 0.5, "+0.5"), (t_task + 4.5, "+4.5"),
                      (total, "+5")):
        ax.plot([xpos, xpos], [y_axis - 0.10, y_axis + 0.08],
                color=MUTED, lw=1.0)
        bold = lab == "0"
        ax.text(xpos, y_axis - 0.18, lab, ha="center", va="top", fontsize=7.6,
                color=INK if bold else MUTED,
                fontweight="bold" if bold else "normal")
    ax.text(total / 2, y_axis - 0.52, "time relative to imagery onset (s)",
            ha="center", va="top", fontsize=8, color=MUTED)

    # imagery-onset rule
    ax.plot([t_task, t_task], [y_axis - 0.10, y_bar + bar_h + 0.06],
            color=INK, lw=1.4, zorder=4)
    ax.text(t_task - 0.12, y_bar + bar_h + 0.14, "imagery onset  t = 0",
            ha="right", va="bottom", fontsize=8.6, fontweight="bold")

    # ---- screen miniatures above the bar ----
    sw, sh = 1.05, 0.78
    y_scr = y_bar + bar_h + 0.52
    draw_screen(ax, 1.5 - sw / 2, y_scr, sw, sh, "cross")
    draw_screen(ax, 4.0 - sw / 2, y_scr, sw, sh, "uparrow")
    ax.text(4.0, y_scr + sh + 0.08, "class-specific cue", ha="center",
            va="bottom", fontsize=7.6, color=MUTED)
    draw_screen(ax, 7.5 - sw / 2, y_scr, sw, sh, "white")
    ax.text(7.5, y_scr + sh + 0.08,
            "blank — identical for all three classes", ha="center",
            va="bottom", fontsize=7.6, color=MUTED)

    # (LSL marker codes intentionally omitted: implementation detail, not
    # something a reader of the paper needs.)
    ax.text(-2.05, y_axis - 1.05,
            "50 trials per class (150 total), classes interleaved in random "
            "order.  Timing verified identical in all 13 recordings (±0.02 s).\n"
            "Only the 2 s cue differs between classes; the 5 s imagery screen "
            "is the same blank white for all three, which is what makes Rest "
            "a visually matched baseline.",
            ha="left", va="top", fontsize=8.2, color=MUTED, linespacing=1.55)

    ax.text(-2.05, 3.88, "Trial structure", ha="left", va="top", fontsize=13,
            fontweight="bold")
    ax.text(-2.05, 3.50,
            "Redrawn from the PsychoPy builder flow; durations confirmed "
            "against the recorded marker stream.",
            ha="left", va="top", fontsize=8.6, color=MUTED)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(R_DIR, f"fig_paradigm.{ext}"),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.join(R_DIR, "fig_paradigm.{png,pdf}"))


if __name__ == "__main__":
    main()
