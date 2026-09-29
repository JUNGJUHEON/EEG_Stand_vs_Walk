"""Exploratory, fixed-model Rest/MI decoding across matched EEG stages.

See the repository README. Outputs contain participant labels
only, not raw signals or source filenames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reanalyse_cached_eeg import CODE_TO_Y, CUE_CODE, keep_position, participant_files


BANDS = {"mu": (8, 13), "beta": (13, 30)}
WINDOWS = {
    "base1": (-4.5, -3.5), "base2": (-3.5, -2.5),
    "cue1": (-2, -1), "cue2": (-1, 0),
    "task1": (0, 1), "task2": (1, 2), "task3": (2, 3),
    "task4": (3, 4), "task5": (4, 5),
}
STAGES = ("fixation", "cue", "task_first", "task_late",
          "late_change", "late_mu", "late_beta", "task_early2", "task_late2")


def read_stages(date: str, runs: list, reference: str) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict]]:
    features = {stage: [] for stage in STAGES}
    labels, quality = [], []
    for tag, _, path in runs:
        with np.load(path, allow_pickle=False) as z:
            data = z["data"].astype(np.float64)
            fs = float(z["fs"])
            n_samples = int(z["n_samples"])
            codes, samples = z["codes"], z["samples"]
            clip = np.unpackbits(z["clip"], axis=1, count=n_samples).astype(bool)
        if fs != 256 or data.shape != (8, n_samples):
            raise ValueError(f"Unexpected cache geometry: {path.name}")
        if reference == "car":
            data -= data.mean(axis=0, keepdims=True)

        order = np.argsort(samples, kind="stable")
        codes, samples = codes[order], samples[order]
        positions = np.flatnonzero(np.isin(codes, tuple(CODE_TO_Y)))
        filtered = {name: sosfiltfilt(
            butter(4, edges, btype="bandpass", fs=fs, output="sos"), data, axis=1)
            for name, edges in BANDS.items()}
        accepted, invalid, clipped = 0, 0, 0
        for trial, pos in enumerate(positions, start=1):
            if not keep_position(date, tag, trial):
                continue
            code, sample = int(codes[pos]), int(samples[pos])
            if (pos < 2 or int(codes[pos - 1]) != CUE_CODE[code]
                    or int(codes[pos - 2]) != 0
                    or abs((sample - int(samples[pos - 1])) / fs - 2) > 0.3
                    or abs((sample - int(samples[pos - 2])) / fs - 5) > 0.3):
                invalid += 1
                continue
            slices = {key: slice(sample + round(start * fs),
                                 sample + round(stop * fs))
                      for key, (start, stop) in WINDOWS.items()}
            if any(sl.start < 0 or sl.stop > n_samples for sl in slices.values()):
                invalid += 1
                continue
            if any(np.max(clip[:, sl].mean(axis=1)) > 0.005
                   for sl in slices.values()):
                clipped += 1
                continue

            by_band = {}
            for name, signal in filtered.items():
                power = {key: np.log(np.var(signal[:, sl], axis=1) + 1e-30)
                         for key, sl in slices.items()}
                fixation = (power["base1"] + power["base2"]) / 2
                cue = (power["cue1"] + power["cue2"]) / 2
                late = np.mean([power[f"task{i}"] for i in range(2, 6)], axis=0)
                early2 = (power["task1"] + power["task2"]) / 2
                late2 = (power["task4"] + power["task5"]) / 2
                by_band[name] = (fixation, cue, power["task1"], late,
                                 late - fixation, early2, late2)
            for idx, stage in enumerate(STAGES[:5]):
                features[stage].append(np.concatenate(
                    [by_band["mu"][idx], by_band["beta"][idx]]))
            features["late_mu"].append(by_band["mu"][3])
            features["late_beta"].append(by_band["beta"][3])
            for stage, idx in (("task_early2", 5), ("task_late2", 6)):
                features[stage].append(np.concatenate(
                    [by_band["mu"][idx], by_band["beta"][idx]]))
            labels.append(CODE_TO_Y[code])
            accepted += 1
        quality.append({"run": tag, "retained": accepted,
                        "rejected_event_or_window": invalid, "rejected_clipping": clipped})
    return {key: np.asarray(value) for key, value in features.items()}, np.asarray(labels), quality


def score_cv(x: np.ndarray, y: np.ndarray, split: str) -> float:
    if split == "random":
        splitters = [StratifiedKFold(5, shuffle=True, random_state=42 + seed)
                     for seed in range(5)]
        splits = [list(cv.split(x, y)) for cv in splitters]
    elif split == "contiguous":
        tests = np.array_split(np.arange(len(y)), 5)
        splits = [[(np.setdiff1d(np.arange(len(y)), test), test) for test in tests]]
    else:
        raise ValueError(split)

    scores = []
    for fold_set in splits:
        predicted = np.empty_like(y)
        for train, test in fold_set:
            if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
                raise ValueError("Fold has a missing class")
            model = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
            model.fit(x[train], y[train])
            predicted[test] = model.predict(x[test])
        scores.append(balanced_accuracy_score(y, predicted))
    return float(np.mean(scores))


def group_summary(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for (contrast, feature, split), group in rows.groupby(["contrast", "feature", "split"]):
        scores = group.balanced_accuracy.to_numpy() * 100
        ci = t.ppf(.975, len(scores) - 1) * scores.std(ddof=1) / np.sqrt(len(scores))
        output.append({"contrast": contrast, "feature": feature, "split": split,
                       "n_participants": len(scores), "mean_pct": scores.mean(),
                       "median_pct": np.median(scores), "ci95_low_pct": scores.mean() - ci,
                       "ci95_high_pct": scores.mean() + ci,
                       "n_above_50": int((scores > 50).sum())})
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", choices=("ear", "car"), default="ear")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, quality, input_hashes = [], [], {}
    for participant, date, runs in participant_files(args.cache):
        stages, labels, run_quality = read_stages(date, runs, args.reference)
        counts = np.bincount(labels, minlength=3)
        if counts.tolist() != [50, 50, 50]:
            raise ValueError(f"Unexpected class counts for {participant}: {counts}")
        for tag, _, path in runs:
            input_hashes[f"{participant}_{tag}.npz"] = hashlib.sha256(path.read_bytes()).hexdigest()
        quality.extend({"participant": participant, **entry} for entry in run_quality)
        for contrast, classes in (("Rest-Stand", (0, 1)),
                                  ("Rest-Walk", (0, 2)),
                                  ("Rest-AnyMI", (0, 1, 2))):
            subset = np.isin(labels, classes)
            y = (labels[subset] != 0).astype(int)
            for stage in STAGES:
                x = stages[stage][subset]
                for split in ("random", "contiguous"):
                    rows.append({"participant": participant, "contrast": contrast,
                                 "feature": stage, "split": split,
                                 "n_trials": len(y), "balanced_accuracy": score_cv(x, y, split)})
        print(participant, "retained", len(labels), flush=True)

    participant_table = pd.DataFrame(rows)
    prefix = "rest_mi_stage" if args.reference == "ear" else "rest_mi_stage_car"
    participant_table.to_csv(args.out / f"{prefix}_participant.csv", index=False)
    group_summary(participant_table).to_csv(args.out / f"{prefix}_group.csv", index=False)
    pd.DataFrame(quality).to_csv(args.out / f"{prefix}_quality.csv", index=False)
    with (args.out / f"{prefix}_provenance.json").open("w") as stream:
        json.dump({"analysis_type": "post_hoc_exploratory",
                   "question": "time and baseline source of Rest/MI label information",
                   "reference": args.reference,
                   "input_cache_sha256": input_hashes,
                   "filters": BANDS, "windows_seconds_from_task_marker": WINDOWS,
                   "model": "StandardScaler plus logistic regression C=0.3, fitted inside folds",
                   "evaluation": "five repetitions of stratified fivefold; one contiguous fivefold",
                   "limitations": ["visual cues and instructions differ by condition",
                                   "no direct EOG, EMG, gaze or task compliance",
                                   "same data used to formulate questions"]}, stream, indent=2)


if __name__ == "__main__":
    main()
