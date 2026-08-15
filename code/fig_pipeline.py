# -*- coding: utf-8 -*-
"""
Figure 1b — the decoding pipeline, drawn so that each stage shows BOTH what it
does and why it is there.

The paper's argument is that the pipeline is derived from properties of the
recording rather than selected by benchmark score, so the figure carries the
justification inline; a plain box-and-arrow flow would lose the point.

Writes results/fig_pipeline.{png,pdf} — pdf is vector, use that for submission.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steam_io as sio  # noqa: E402

R_DIR = sio.RESULTS_DIR

INK = "#1f2328"
MUTED = "#6b7280"
LINE = "#c9d1d9"
ACCENT = "#1f6feb"
FOLD_BG = "#eef4fd"
FOLD_EDGE = "#a9c7f5"
BOX_BG = "#ffffff"

plt.rcParams.update({
    "font.size": 9, "text.color": INK, "figure.dpi": 200,
    "pdf.fonttype": 42, "ps.fonttype": 42,      # editable text in the PDF
})

# stage: (title, detail, shape, rationale)
# Titles are kept to ONE line so the title/detail offset is uniform across
# boxes; a two-line title silently overruns the detail text below it.
STAGES = [
    ("Raw XDF", "Bitbrain Versatile\nA1 reference",
     "8 ch × 256 Hz", "as recorded"),
    ("Notch", "60 / 120 Hz",
     "8 × n", "mains"),
    ("Epoch", "+0.5 → +4.5 s\nfrom imagery onset",
     "N × 8 × 1024", "starts AFTER the cue:\nthe cue window alone\ndecodes at 52–56 %"),
    ("Reject", "rail clipping +\nlog-var MAD > 5",
     "N′ × 8 × 1024", "0–8 trials/participant;\nno clipped sample falls\ninside a task epoch"),
    ("Filter bank", "10 bands, 4 Hz wide\n2 Hz hop, 8–30 Hz",
     "N′ × 10 × 8 × 1024", "the discriminative ridge is\n≈2 Hz wide and moves\n9–16 Hz across participants"),
    ("Covariance", "per band,\nOAS shrinkage",
     "N′ × 10 × 8 × 8", "MI modulates rhythm amplitude,\nso information is in how power\ndistributes over electrodes"),
    ("Tangent space", "at the Riemannian\nmean, per band",
     "N′ × 360", "covariances are SPD on a curved\nmanifold; Euclidean LDA distorts\nthem when ill-conditioned"),
    ("Classifier", "L2 logistic regression,\nstandardised",
     "N′ × n_class", "360 features vs ~100 trials;\nlinear keeps weights traceable\nto electrode × frequency"),
]

FOLD_FROM = 4          # first stage fitted inside the training fold


def main():
    n = len(STAGES)
    # --- vertical layout, in data units; ylim is set tight to what is used ---
    Y_TITLE = 5.80          # figure title
    Y_SUB = 5.44            # subtitle
    Y_FOLD = 4.90           # fold caption (2 lines, grows upward)
    Y_BOX = 3.05            # box bottom
    BH = 1.30               # box height
    Y_SHAPE = 2.86          # data shape (below box)
    Y_WHY = 2.48            # rationale (up to 3 lines, grows downward)
    Y_EVAL = 1.02           # evaluation note (grows downward)

    fig, ax = plt.subplots(figsize=(2.05 * n, 4.35))
    ax.set_xlim(0, n)
    ax.set_ylim(0.30, 6.05)
    ax.axis("off")

    bw = 0.86

    # shaded region: everything fitted inside the CV training fold
    x0 = FOLD_FROM + (1 - bw) / 2 - 0.14
    x1 = n - (1 - bw) / 2 + 0.14
    ax.add_patch(Rectangle((x0, Y_SHAPE - 0.34), x1 - x0,
                           (Y_BOX + BH + 0.22) - (Y_SHAPE - 0.34),
                           facecolor=FOLD_BG, edgecolor=FOLD_EDGE,
                           linewidth=1.1, linestyle=(0, (5, 3)), zorder=0))
    ax.text((x0 + x1) / 2, Y_FOLD,
            "fitted inside the training fold only — band weighting, covariance "
            "shrinkage,\ntangent-space reference, scaler and classifier all "
            "refit per fold",
            ha="center", va="bottom", fontsize=8.2, color=ACCENT,
            style="italic", linespacing=1.4)

    for i, (title, detail, shape, why) in enumerate(STAGES):
        cx = i + 0.5
        inside = i >= FOLD_FROM
        ax.add_patch(FancyBboxPatch(
            (cx - bw / 2, Y_BOX), bw, BH,
            boxstyle="round,pad=0.03,rounding_size=0.05",
            facecolor=BOX_BG, edgecolor=ACCENT if inside else LINE,
            linewidth=1.5 if inside else 1.1, zorder=3))
        ax.text(cx, Y_BOX + BH - 0.30, title, ha="center", va="center",
                fontsize=9.6, fontweight="bold", zorder=4)
        ax.text(cx, Y_BOX + BH - 0.62, detail, ha="center", va="top",
                fontsize=7.5, color=MUTED, zorder=4, linespacing=1.4)

        ax.text(cx, Y_SHAPE, shape, ha="center", va="top", fontsize=7.7,
                color=INK, family="monospace", zorder=4)
        ax.text(cx, Y_WHY, why, ha="center", va="top", fontsize=7.1,
                color=MUTED, linespacing=1.5, zorder=4)

        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (cx + bw / 2 + 0.012, Y_BOX + BH / 2),
                (cx + 1 - bw / 2 - 0.012, Y_BOX + BH / 2),
                arrowstyle="-|>", mutation_scale=11,
                color=MUTED, linewidth=1.1, zorder=2))

    ax.text(0.02, Y_TITLE, "Decoding pipeline  (fb_ts_lr)",
            ha="left", va="center", fontsize=13, fontweight="bold")
    ax.text(0.02, Y_SUB,
            "Each stage follows from a property of the recording, not from a "
            "benchmark score.  Data shape in monospace; justification below.",
            ha="left", va="center", fontsize=8.6, color=MUTED)

    ax.text(n / 2, Y_EVAL,
            "Evaluation: repeated stratified 5-fold CV, plus NESTED accuracy "
            "(inner 4-fold selects the pipeline within each outer training "
            "fold), plus a\nchronological 60/40 split.  Per-participant "
            "significance from 1000-permutation label nulls, Holm-corrected "
            "across participants.",
            ha="center", va="top", fontsize=8.1, color=INK, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f6f8fa",
                      edgecolor=LINE, linewidth=0.9))

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(R_DIR, f"fig_pipeline.{ext}"),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.join(R_DIR, "fig_pipeline.{png,pdf}"))


if __name__ == "__main__":
    main()
