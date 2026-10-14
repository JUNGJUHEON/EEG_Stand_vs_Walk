"""Validate and summarize all recorded participant/contrast/training seeds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

SUBJECTS = [f"S{i:02d}" for i in range(1, 11)]
CONTRASTS = ["R_vs_S", "R_vs_W", "S_vs_W"]
SEEDS = [42, 43, 44]


def read_scores(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = []
    for row in rows:
        seed = int(row.get("training_seed", row.get("seed", -1)))
        accuracy, auc = float(row["accuracy"]), float(row["auc"])
        if not 0 <= accuracy <= 1 or not 0 <= auc <= 1:
            raise ValueError("Scores must be finite fractions in [0,1]")
        n_correct = round(20*accuracy)
        if abs(20*accuracy-n_correct) > 1e-8:
            raise ValueError("Accuracy is inconsistent with twenty test trials")
        if int(row.get("n_test", 20)) != 20 or int(row.get("n_correct", n_correct)) != n_correct:
            raise ValueError("Stored test count disagrees with the accuracy")
        if abs(float(row.get("balanced_accuracy", accuracy))-accuracy) > 1e-8:
            raise ValueError("Balanced accuracy must match accuracy for this balanced test set")
        records.append(dict(subject=row["subject"], contrast=row["contrast"],
                            training_seed=seed, accuracy=accuracy,
                            balanced_accuracy=accuracy, n_correct=n_correct, n_test=20, auc=auc))
    keys = [(r["subject"], r["contrast"], r["training_seed"]) for r in records]
    expected = {(s, c, seed) for s in SUBJECTS for c in CONTRASTS for seed in SEEDS}
    if len(keys) != 90 or len(set(keys)) != 90 or set(keys) != expected:
        raise ValueError("Require all ten participants, all three contrasts, all three seeds")
    return sorted(records, key=lambda r: (r["subject"], CONTRASTS.index(r["contrast"]), r["training_seed"]))


def summarize(records):
    participant, group = [], []
    for subject in SUBJECTS:
        for contrast in CONTRASTS:
            scores = [r["accuracy"] for r in records
                      if r["subject"] == subject and r["contrast"] == contrast]
            participant.append(dict(subject=subject, contrast=contrast, n_training_seeds=3,
                                    mean_accuracy=mean(scores), min_accuracy=min(scores),
                                    max_accuracy=max(scores)))
    for contrast in CONTRASTS:
        for seed in SEEDS:
            scores = [r["accuracy"] for r in records
                      if r["contrast"] == contrast and r["training_seed"] == seed]
            group.append(dict(contrast=contrast, training_seed=seed, n_subjects=10,
                              mean_accuracy=mean(scores), min_accuracy=min(scores),
                              max_accuracy=max(scores)))
    return participant, group


def participant_table(records):
    lookup = {(r["subject"], r["contrast"], r["training_seed"]): r["accuracy"] for r in records}
    lines = ["| Participant | Rest-Stand | Rest-Walk | Stand-Walk |",
             "|---|---:|---:|---:|"]
    for subject in SUBJECTS:
        cells = [" / ".join(f"{100*lookup[subject, contrast, seed]:.0f}" for seed in SEEDS)
                 for contrast in CONTRASTS]
        lines.append("| " + " | ".join([subject, *cells]) + " |")
    return "\n".join(lines)


def write_csv(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = read_scores(args.scores)
    participant, group = summarize(records)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "participant_scores.csv", records)
    write_csv(args.out / "participant_summary.csv", participant)
    write_csv(args.out / "group_summary.csv", group)
    print(participant_table(records))


if __name__ == "__main__":
    main()
