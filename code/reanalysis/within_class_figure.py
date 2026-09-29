"""Plot all planned within-class windows without selecting significant bins."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


WINDOWS = ("cue1", "task1", "task2", "task3", "task4", "task5")
X = (-1.5, 0.5, 1.5, 2.5, 3.5, 4.5)
COLORS = {"Rest": "#606a70", "Stand": "#087e83", "Walk": "#bc5735"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.results / "within_class_group.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True,
                             constrained_layout=True)
    for i, reference in enumerate(("earlobe", "car")):
        for j, band in enumerate(("mu_8_13", "beta_13_30")):
            ax = axes[i, j]
            for condition in ("Rest", "Stand", "Walk"):
                sub = data[(data.reference == reference)
                           & (data.roi == "midline")
                           & (data.band == band)
                           & (data.condition == condition)].set_index("window").loc[list(WINDOWS)]
                color = COLORS[condition]
                ax.plot(X, sub.mean_db, marker="o", lw=1.9, ms=4.5,
                        color=color, label=condition)
                ax.fill_between(X, sub.ci95_low_db.to_numpy(),
                                sub.ci95_high_db.to_numpy(), color=color, alpha=.12)
            ax.axhline(0, color="#9aa4a8", lw=.9)
            ax.axvline(0, color="#c2c8ca", ls="--", lw=.9)
            ax.set_title(f"{band.replace('_', ' ')}  |  {reference}", loc="left", fontsize=11)
            ax.set_ylabel("Change from fixation (dB)")
            ax.set_xticks(X, ("Cue 1", "Task 1", "Task 2", "Task 3", "Task 4", "Task 5"))
            ax.grid(axis="y", color="#e7ebec", lw=.7)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper left")
    fig.suptitle("Within-condition central EEG power  |  10 participants",
                 fontsize=14, x=.02, ha="left")
    fig.savefig(args.results / "within_class_timecourse.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
