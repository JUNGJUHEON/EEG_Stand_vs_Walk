"""Transparent post-hoc average of the four task bins after the first second."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


LATE = {"task2", "task3", "task4", "task5"}


def summarize(values: pd.DataFrame, group_columns: list[str],
              value_column: str, first_column: str, last_column: str) -> pd.DataFrame:
    rows = []
    for keys, group in values.groupby(group_columns):
        sample = group[value_column].to_numpy()
        width = (t.ppf(.975, len(sample) - 1) * sample.std(ddof=1)
                 / np.sqrt(len(sample)))
        rows.append(dict(zip(group_columns, keys), n_participants=len(sample),
                         mean_db=float(sample.mean()),
                         ci95_low_db=float(sample.mean() - width),
                         ci95_high_db=float(sample.mean() + width),
                         n_positive=int((sample > 0).sum()),
                         n_both_halves_positive=int(((group[first_column] > 0)
                                                     & (group[last_column] > 0)).sum()),
                         first_half_mean_db=float(group[first_column].mean()),
                         last_half_mean_db=float(group[last_column].mean())))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    p = args.results
    own = pd.read_csv(p / "within_class_participant.csv")
    own = (own[own.window.isin(LATE)]
           .groupby(["participant", "condition", "band", "roi", "reference"])
           [["mean_db", "first", "last"]].mean().reset_index())
    own["comparison"] = own.condition + "-fixation"
    own = own.drop(columns="condition")
    own = own.rename(columns={"first": "first_half_db", "last": "last_half_db"})

    contrast = pd.read_csv(p / "within_class_vs_rest.csv")
    contrast = (contrast[contrast.window.isin(LATE)]
                .groupby(["participant", "condition_minus_rest", "band", "roi", "reference"])
                [["delta_db", "first_delta_db", "last_delta_db"]].mean().reset_index())
    contrast["comparison"] = contrast.condition_minus_rest + "-Rest"
    contrast = contrast.drop(columns="condition_minus_rest")
    contrast = contrast.rename(columns={"delta_db": "mean_db",
                                        "first_delta_db": "first_half_db",
                                        "last_delta_db": "last_half_db"})
    participant = pd.concat([own, contrast], ignore_index=True)
    participant.to_csv(p / "within_class_late_participant.csv", index=False)
    group = summarize(participant, ["comparison", "band", "roi", "reference"],
                      "mean_db", "first_half_db", "last_half_db")
    group.to_csv(p / "within_class_late_group.csv", index=False)
    print("Wrote late-window participant and group summaries")


if __name__ == "__main__":
    main()
