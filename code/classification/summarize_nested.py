"""Export the three saved binary nested-CV results without fitting a model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import t

SUBJECTS = [f"S{i:02d}" for i in range(1, 11)]
TASKS = {"RvS": "Rest-Stand", "RvW": "Rest-Walk", "SvW": "Stand-Walk"}
POOL = {"csp_lda", "ts_lr", "acm_d5_10_ts_lr", "fgmdm", "fb_ts_lr", "fbcsp_mibif_lda"}


def read_scores(path):
    with Path(path).open(newline="") as stream:
        original = list(csv.DictReader(stream))
    records = []
    for row in original:
        if row.get("kind") != "nested" or row.get("task") not in TASKS:
            continue
        score = float(row["acc"])
        count = int(row["n"])
        choices = row["note"].split("|")
        if not np.isfinite(score) or not 0 <= score <= 1 or not 94 <= count <= 100:
            raise ValueError("Unexpected saved score or retained-trial count")
        if len(choices) != 5 or not set(choices).issubset(POOL):
            raise ValueError("Expected five choices from the documented nested pool")
        if (float(row["lo"]), float(row["hi"]), float(row["tmin"]),
                float(row["tmax"]), row["ref"]) != (8., 30., .5, 4.5, "none"):
            raise ValueError("Unexpected stored preprocessing profile")
        records.append(dict(subject=row["subject"], contrast=TASKS[row["task"]],
                            accuracy=score, n_retained_trials=count,
                            outer_pipeline_choices=row["note"]))
    keys = [(r["subject"], r["contrast"]) for r in records]
    expected = {(s, c) for s in SUBJECTS for c in TASKS.values()}
    if len(keys) != 30 or len(set(keys)) != 30 or set(keys) != expected:
        raise ValueError("Require all ten participants and all three binary comparisons")
    return sorted(records, key=lambda r: (r["subject"], list(TASKS.values()).index(r["contrast"])))


def summarize(records):
    rows = []
    for contrast in TASKS.values():
        values = np.array([r["accuracy"] for r in records if r["contrast"] == contrast])
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        half = float(t.ppf(.975, len(values)-1)*sd/np.sqrt(len(values)))
        rows.append(dict(contrast=contrast, n_participants=len(values),
                         mean_accuracy=mean, sd_between_participants=sd,
                         t_ci95_low=mean-half, t_ci95_high=mean+half, chance=.5))
    return rows


def participant_table(records):
    lookup = {(r["subject"], r["contrast"]): r["accuracy"] for r in records}
    lines = ["| Participant | Rest-Stand | Rest-Walk | Stand-Walk |",
             "|---|---:|---:|---:|"]
    for subject in SUBJECTS:
        cells = [f"{100*lookup[subject, contrast]:.1f}" for contrast in TASKS.values()]
        lines.append("| " + " | ".join([subject, *cells]) + " |")
    return "\n".join(lines)


def group_table(rows):
    lines = ["| Contrast | Mean accuracy | Participant-level 95% t interval | Chance |",
             "|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['contrast']} | {100*r['mean_accuracy']:.1f}% | "
                     f"{100*r['t_ci95_low']:.1f}-{100*r['t_ci95_high']:.1f}% | 50% |")
    return "\n".join(lines)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = read_scores(args.scores)
    summary = summarize(records)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "nested_participant_scores.csv", records)
    write_csv(args.out / "nested_group_summary.csv", summary)
    print(participant_table(records))
    print(group_table(summary))


if __name__ == "__main__":
    main()
