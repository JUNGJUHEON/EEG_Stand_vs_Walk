"""Exploratory checks separating task decoding from imagery physiology.

The existing pairwise classifiers do not estimate Rest vs pooled imagery.
This script adds that contrast and compares band-limited decoding with
within-trial fixation-to-task power changes. None identifies signal source.
"""

from __future__ import annotations

import argparse
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

from reanalyse_cached_eeg import (CENTRAL, CODE_TO_Y, CUE_CODE, CLASSES,
                                  keep_position, participant_files,
                                  read_participant)


BAND_FEATURES = ("task_8_30", "task_mu_8_13", "task_beta_13_30",
                 "task_30_40", "cue_8_30")
BANDS = {"mu_8_13": (8, 13), "beta_13_30": (13, 30)}
FIXATION = (-4.5, -2.5)
TASK = (0.5, 4.5)


def balanced_cv(x: np.ndarray, y: np.ndarray, repeats: int = 5) -> float:
    scores = []
    for repeat in range(repeats):
        cv = StratifiedKFold(5, shuffle=True, random_state=112 + repeat)
        prediction = np.empty_like(y)
        for train, test in cv.split(x, y):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000),
            )
            model.fit(x[train], y[train])
            prediction[test] = model.predict(x[test])
        scores.append(balanced_accuracy_score(y, prediction))
    return float(np.mean(scores))


def contrasts(features: dict[str, np.ndarray], labels: np.ndarray,
              participant: str) -> list[dict]:
    definitions = (
        ("Rest-Stand", np.isin(labels, (0, 1)), labels == 1),
        ("Rest-Walk", np.isin(labels, (0, 2)), labels == 2),
        ("Stand-Walk", np.isin(labels, (1, 2)), labels == 2),
        ("Rest-AnyMI", np.ones(len(labels), bool), labels != 0),
    )
    rows = []
    for name, mask, positive in definitions:
        y = positive[mask].astype(int)
        for feature in BAND_FEATURES:
            rows.append({
                "participant": participant,
                "contrast": name,
                "feature": feature,
                "n_negative": int((y == 0).sum()),
                "n_positive": int((y == 1).sum()),
                "balanced_accuracy": balanced_cv(features[feature][mask], y),
            })
    return rows


def fixation_to_task(date: str, runs: list[tuple[str, str, Path]],
                     participant: str) -> tuple[list[dict], list[dict]]:
    rows, quality = [], []
    for tag, _, path in runs:
        with np.load(path, allow_pickle=False) as z:
            data = z["data"].astype(np.float64)
            fs = float(z["fs"])
            codes = z["codes"]
            samples = z["samples"]
            n_samples = int(z["n_samples"])
            clip = np.unpackbits(z["clip"], axis=1, count=n_samples).astype(bool)
            channels = [str(c) for c in z["ch_names"]]
        if fs != 256 or data.shape != (8, n_samples):
            raise ValueError(f"Unexpected cache geometry: {path.name}")
        center = [channels.index(ch) for ch in CENTRAL]
        order = np.argsort(samples, kind="stable")
        codes, samples = codes[order], samples[order]
        task_positions = np.flatnonzero(np.isin(codes, tuple(CODE_TO_Y)))

        filtered = {}
        for band, edges in BANDS.items():
            sos = butter(4, edges, btype="bandpass", fs=fs, output="sos")
            filtered[band] = sosfiltfilt(sos, data, axis=1)

        excluded = 0
        for trial, position in enumerate(task_positions, start=1):
            if not keep_position(date, tag, trial):
                continue
            code = int(codes[position])
            sample = int(samples[position])
            if (position < 2 or int(codes[position - 1]) != CUE_CODE[code]
                    or int(codes[position - 2]) != 0):
                excluded += 1
                continue
            if (abs((sample - int(samples[position - 1])) / fs - 2) > 0.3
                    or abs((sample - int(samples[position - 2])) / fs - 5) > 0.3):
                excluded += 1
                continue
            windows = {}
            for name, (start, stop) in (("fixation", FIXATION), ("task", TASK)):
                a, b = sample + round(start * fs), sample + round(stop * fs)
                if a < 0 or b > n_samples:
                    break
                windows[name] = slice(a, b)
            if len(windows) != 2 or any(
                np.max(clip[:, sl].mean(axis=1)) > 0.005 for sl in windows.values()
            ):
                excluded += 1
                continue
            for band, original in filtered.items():
                for reference in ("earlobe", "car"):
                    signal = (original if reference == "earlobe" else
                              original - original.mean(axis=0, keepdims=True))
                    base = np.log(np.var(signal[center, windows["fixation"]], axis=1) + 1e-30)
                    task = np.log(np.var(signal[center, windows["task"]], axis=1) + 1e-30)
                    scale = 10 / np.log(10)
                    rows.append({
                        "participant": participant,
                        "run": tag,
                        "trial": trial,
                        "condition": CLASSES[CODE_TO_Y[code]],
                        "band": band,
                        "reference": reference,
                        "fixation_db": float(base.mean() * scale),
                        "task_db": float(task.mean() * scale),
                        "task_minus_fixation_db": float((task - base).mean() * scale),
                    })
        quality.append({"participant": participant, "run": tag,
                        "retained_task_markers": sum(keep_position(date, tag, j)
                                                     for j in range(1, len(task_positions) + 1)),
                        "excluded_from_paired_windows": excluded})
    return rows, quality


def summarize_cv(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (contrast, feature), group in scores.groupby(["contrast", "feature"]):
        values = group["balanced_accuracy"].to_numpy() * 100
        n = len(values)
        mean = values.mean()
        width = t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n)
        rows.append({"contrast": contrast, "feature": feature,
                     "n_participants": n, "mean_balanced_accuracy_pct": mean,
                     "ci95_low_pct": mean - width, "ci95_high_pct": mean + width})
    return pd.DataFrame(rows)


def summarize_power(trials: pd.DataFrame) -> pd.DataFrame:
    participant = (trials.groupby(["participant", "condition", "band", "reference"])
                   [["fixation_db", "task_db", "task_minus_fixation_db"]]
                   .mean().reset_index())
    baseline = participant.pivot(index=["participant", "band", "reference"],
                                 columns="condition", values="task_minus_fixation_db")
    for name, left, right in (("Stand-minus-Rest", "Stand", "Rest"),
                              ("Walk-minus-Rest", "Walk", "Rest"),
                              ("Stand-minus-Walk", "Stand", "Walk")):
        delta = (baseline[left] - baseline[right]).rename(name)
        participant = participant.merge(delta.reset_index(),
                                        on=["participant", "band", "reference"], how="left")
    return participant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cv_rows, power_rows, quality_rows = [], [], []
    for participant, date, runs in participant_files(args.cache):
        features, labels, _, _, _ = read_participant(date, runs)
        cv_rows.extend(contrasts(features, labels, participant))
        power, quality = fixation_to_task(date, runs, participant)
        power_rows.extend(power)
        quality_rows.extend(quality)
        print(participant, "cv=20", "paired_trials=", len(power) // (2 * len(BANDS)), flush=True)
    scores = pd.DataFrame(cv_rows)
    power_trials = pd.DataFrame(power_rows)
    scores.to_csv(args.out / "contrast_logic_scores.csv", index=False)
    summarize_cv(scores).to_csv(args.out / "contrast_logic_summary.csv", index=False)
    summarize_power(power_trials).to_csv(args.out / "fixation_task_participant_power.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(args.out / "fixation_task_quality.csv", index=False)
    print("Wrote four contrast-logic CSV files to", args.out)


if __name__ == "__main__":
    main()
