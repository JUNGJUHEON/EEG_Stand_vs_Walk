"""Independent, exploratory checks from continuous notch-filtered EEG caches.

The caches contain volts, event markers, and raw clipping masks. This uses a
single fixed log-bandpower classifier and does not reproduce the older MNE FIR
filter, filter-bank model search, or nested-CV estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CODE_TO_Y = {2: 0, 5: 1, 8: 2}
CUE_CODE = {2: 1, 5: 4, 8: 7}
CLASSES = ("Rest", "Stand", "Walk")
CENTRAL = ("C3", "C4", "Cz", "CP3", "CP4", "CPz")
KEEP = {
    "250911": {"MI": ((1, 30), (61, 120)), "MI2": ((1, 60),)},
    "250922": {"MI": ((1, 30), (61, 150)), "MI2": ((1, 30),)},
    "250924": {"MI": ((1, 30), (61, 150)), "MI2": ((1, 30),)},
}
WINDOWS = {"task": (0.5, 4.5), "cue": (-1.75, -0.25)}
BANDS = {"task_8_30": (8, 30), "cue_8_30": (8, 30),
         "task_30_40": (30, 40), "task_mu_8_13": (8, 13),
         "task_beta_13_30": (13, 30)}


def participant_files(cache):
    files = sorted(p for p in cache.glob("*_MI*.npz")
                   if p.stem.rsplit("_", 1)[-1] in ("MI", "MI2"))
    by_date = {}
    for path in files:
        date, name, tag = path.stem.split("_")
        by_date.setdefault(date, []).append((tag, name, path))
    if len(by_date) != 10:
        raise ValueError(f"Expected ten MI participants; found {len(by_date)}")
    return [(f"S{i:02d}", date, sorted(by_date[date]))
            for i, date in enumerate(sorted(by_date), start=1)]


def keep_position(date, tag, position_one_based):
    ranges = KEEP.get(date, {}).get(tag)
    return ranges is None or any(a <= position_one_based <= b for a, b in ranges)


def log_power(x):
    return np.log(np.var(x, axis=-1) + 1e-30)


def read_participant(date, runs):
    features = {name: [] for name in BANDS}
    labels, run_labels, quality = [], [], []
    ch_ref = None
    for tag, _, path in runs:
        with np.load(path, allow_pickle=False) as z:
            data = z["data"].astype(np.float64)
            n_samples, fs = int(z["n_samples"]), float(z["fs"])
            codes, samples = z["codes"], z["samples"]
            clip = np.unpackbits(z["clip"], axis=1, count=n_samples).astype(bool)
            ch_names = [str(c) for c in z["ch_names"]]
        if ch_ref is not None and ch_ref != ch_names:
            raise ValueError(f"Channel order changed: {path.name}")
        ch_ref = ch_names
        if data.shape != (8, n_samples) or clip.shape != data.shape or fs != 256:
            raise ValueError(f"Unexpected cache geometry: {path.name}")

        order = np.argsort(samples, kind="stable")
        codes, samples = codes[order], samples[order]
        task_pos = np.flatnonzero(np.isin(codes, list(CODE_TO_Y)))
        selected = []
        invalid_cue = 0
        n_kept_by_rule = 0
        n_clipped = 0
        for trial_number, pos in enumerate(task_pos, start=1):
            if not keep_position(date, tag, trial_number):
                continue
            n_kept_by_rule += 1
            code, sample = int(codes[pos]), int(samples[pos])
            if pos == 0 or int(codes[pos - 1]) != CUE_CODE[code]:
                invalid_cue += 1
                continue
            cue_duration = (sample - int(samples[pos - 1])) / fs
            if abs(cue_duration - 2.0) > 0.30:
                invalid_cue += 1
                continue
            slices = {}
            for name, (start, stop) in WINDOWS.items():
                a = sample + round(start * fs)
                b = sample + round(stop * fs)
                slices[name] = slice(a, b)
                if a < 0 or b > n_samples:
                    invalid_cue += 1
                    break
            else:
                task = slices["task"]
                if np.max(clip[:, task].mean(axis=1)) > 0.005:
                    n_clipped += 1
                    continue
                selected.append((CODE_TO_Y[code], slices))
        if not selected:
            raise ValueError(f"No valid epochs: {path.name}")

        # Each band uses one fixed filter for the whole run, before cropping.
        # No statistic is estimated from labels or from held-out trials.
        filtered = {}
        for band in set(BANDS.values()):
            sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
            filtered[band] = sosfiltfilt(sos, data, axis=1)
        for label, slices in selected:
            labels.append(label)
            run_labels.append(tag)
            for name, band in BANDS.items():
                window = "cue" if name.startswith("cue") else "task"
                features[name].append(log_power(filtered[band][:, slices[window]]))
        quality.append({
            "run": tag, "n_task_markers": len(task_pos),
            "n_kept_by_rule": n_kept_by_rule,
            "n_rejected_invalid_cue_or_window": invalid_cue,
            "n_rejected_clipping": n_clipped,
            "n_analysed": len(selected),
            "n_rest": sum(v[0] == 0 for v in selected),
            "n_stand": sum(v[0] == 1 for v in selected),
            "n_walk": sum(v[0] == 2 for v in selected),
        })
    return ({k: np.asarray(v) for k, v in features.items()},
            np.asarray(labels), np.asarray(run_labels), ch_ref, quality)


def fit_predict(x, y, train, test):
    if len(np.unique(y[train])) != 2:
        raise ValueError("Training fold lacks a class")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
    model.fit(x[train], y[train])
    return model.predict(x[test])


def cv_scores(x, y, indices, mode, repeats=5):
    x, y = x[indices], y[indices]
    if mode == "random":
        splitters = [list(StratifiedKFold(5, shuffle=True, random_state=42 + r)
                          .split(x, y)) for r in range(repeats)]
    elif mode == "contiguous":
        splitters = []
        folds = np.array_split(np.arange(len(y)), 5)
        splitters.append([(np.setdiff1d(np.arange(len(y)), te), te) for te in folds])
    else:
        raise ValueError(mode)
    accuracies, balanced, first_pred = [], [], None
    for splits in splitters:
        pred = np.empty_like(y)
        for train, test in splits:
            pred[test] = fit_predict(x, y, train, test)
        if first_pred is None:
            first_pred = pred.copy()
        accuracies.append(accuracy_score(y, pred))
        balanced.append(balanced_accuracy_score(y, pred))
    return np.mean(accuracies), np.mean(balanced), np.std(balanced, ddof=1) if repeats > 1 and mode == "random" else np.nan, first_pred


def run_transfer(x, y, run_labels, indices):
    names = sorted(set(run_labels[indices]))
    if len(names) != 2:
        return []
    output = []
    for train_name, test_name in (names, names[::-1]):
        train = indices[run_labels[indices] == train_name]
        test = indices[run_labels[indices] == test_name]
        if len(train) < 15 or len(test) < 15 or len(np.unique(y[train])) != 2:
            continue
        pred = fit_predict(x, y, train, test)
        output.append({"train_run": train_name, "test_run": test_name,
                       "n_train": len(train), "n_test": len(test),
                       "accuracy": accuracy_score(y[test], pred),
                       "balanced_accuracy": balanced_accuracy_score(y[test], pred)})
    return output


def topography_ratio(x, y, a, b, centered, rng, repetitions=200):
    xa, xb = x[y == a], x[y == b]
    if centered:
        xa = xa - xa.mean(axis=1, keepdims=True)
        xb = xb - xb.mean(axis=1, keepdims=True)
    n = min(len(xa), len(xb))
    if n < 10:
        return np.nan, np.nan, np.nan
    within, between = [], []
    for _ in range(repetitions):
        ia, ib = rng.permutation(len(xa))[:n], rng.permutation(len(xb))[:n]
        half = n // 2
        a1, a2 = xa[ia[:half]].mean(axis=0), xa[ia[half:]].mean(axis=0)
        b1, b2 = xb[ib[:half]].mean(axis=0), xb[ib[half:]].mean(axis=0)
        within.append((np.linalg.norm(a1 - a2) + np.linalg.norm(b1 - b2)) / 2)
        between.append((np.linalg.norm(a1 - b1) + np.linalg.norm(a2 - b2)) / 2)
    w, b = np.mean(within), np.mean(between)
    return w, b, b / w if w > 0 else np.nan


def make_figure(scores, output):
    df = scores.loc[scores["contrast"].eq("Stand-Walk")]
    wanted = [
        ("task_8_30", "random", "Task 8-30 Hz"),
        ("cue_8_30", "random", "Cue 8-30 Hz"),
        ("task_30_40", "random", "Task 30-40 Hz"),
        ("task_8_30", "contiguous", "Task, contiguous folds"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    colors = ("#176c78", "#b26330", "#6a7389", "#546a36")
    rng = np.random.default_rng(841)
    for i, (feature, split, label) in enumerate(wanted):
        values = df.loc[df["feature"].eq(feature) & df["split"].eq(split)] \
                   .sort_values("participant")["balanced_accuracy"].to_numpy()
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(values)), values * 100,
                   color=colors[i], s=24, alpha=0.75, zorder=2)
        ax.plot([i - 0.20, i + 0.20], [values.mean() * 100] * 2,
                color="#202a2d", linewidth=2.5, zorder=3)
    ax.axhline(50, linestyle="--", linewidth=1, color="#696969")
    ax.set_xticks(range(len(wanted)), [v[2] for v in wanted])
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_ylim(25, 100)
    ax.set_title("Stand vs Walk imagery: fixed log-power model (n=10)", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    score_rows, quality_rows, mu_rows, topo_rows, transfer_rows, confusion_rows = [], [], [], [], [], []
    cache_files = []
    for i, (participant, date, runs) in enumerate(participant_files(args.cache), start=1):
        features, y, run_labels, ch, quality = read_participant(date, runs)
        cache_files.extend((f"{participant}_{tag}.npz", path) for tag, _, path in runs)
        for row in quality:
            quality_rows.append({"participant": participant, **row})
        if set(y) != {0, 1, 2}:
            raise ValueError(f"Missing MI class: {participant}")

        for contrast, pair in (("Stand-Walk", (1, 2)),
                               ("Rest-Stand", (0, 1)),
                               ("Rest-Walk", (0, 2))):
            indices = np.flatnonzero(np.isin(y, pair))
            yy = (y == pair[1]).astype(int)
            if min(Counter(yy[indices]).values()) < 10:
                raise ValueError(f"Too few trials: {participant} {contrast}")
            for feature in ("task_8_30", "cue_8_30", "task_30_40"):
                for split in ("random", "contiguous") if feature == "task_8_30" else ("random",):
                    acc, bal, split_sd, pred = cv_scores(features[feature], yy, indices, split)
                    score_rows.append({"participant": participant, "contrast": contrast,
                                       "feature": feature, "split": split, "n_trials": len(indices),
                                       "accuracy": acc, "balanced_accuracy": bal,
                                       "random_split_sd": split_sd})
                    if feature == "task_8_30" and split == "random":
                        cm = confusion_matrix(yy[indices], pred, labels=(0, 1))
                        for j, name in enumerate(pair):
                            confusion_rows.append({"participant": participant, "contrast": contrast,
                                                   "true_class": CLASSES[name],
                                                   "n_test": int(cm[j].sum()),
                                                   "n_correct": int(cm[j, j]),
                                                   "recall": float(cm[j, j] / cm[j].sum())})
            for result in run_transfer(features["task_8_30"], yy, run_labels, indices):
                transfer_rows.append({"participant": participant, "contrast": contrast, **result})

        central_idx = [ch.index(c) for c in CENTRAL]
        for band in ("task_mu_8_13", "task_beta_13_30"):
            logvar = features[band][:, central_idx].mean(axis=1)
            for cls in (1, 2):
                mu_rows.append({"participant": participant, "contrast": f"{CLASSES[cls]}-Rest",
                                "band": band, "mean_db": (logvar[y == cls].mean() -
                                logvar[y == 0].mean()) * 10 / np.log(10)})

        rng = np.random.default_rng(9000 + i)
        for band in ("task_mu_8_13", "task_beta_13_30"):
            for contrast, a, b in (("Stand-Walk", 1, 2), ("Rest-Stand", 0, 1),
                                   ("Rest-Walk", 0, 2)):
                for centered in (False, True):
                    w, bt, ratio = topography_ratio(features[band], y, a, b, centered, rng)
                    topo_rows.append({"participant": participant, "contrast": contrast,
                                      "band": band, "global_power_removed": centered,
                                      "within": w, "between": bt, "ratio": ratio})
        print(participant, "kept", len(y), "classes", np.bincount(y, minlength=3), flush=True)

    output_files = {
        "fixed_model_scores.csv": score_rows,
        "epoch_quality.csv": quality_rows,
        "mi_vs_rest_power.csv": mu_rows,
        "topography_split_half.csv": topo_rows,
        "cross_run_transfer.csv": transfer_rows,
        "fixed_model_class_recall.csv": confusion_rows,
    }
    for name, rows in output_files.items():
        pd.DataFrame(rows).to_csv(args.out / name, index=False)
    make_figure(pd.DataFrame(score_rows), args.out / "fig_fixed_model_controls.png")
    manifest = {
        "analysis_type": "exploratory_reanalysis_of_preprocessed_continuous_cache",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_cache_sha256": {label: hashlib.sha256(path.read_bytes()).hexdigest()
                               for label, path in cache_files},
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__, "pandas": pd.__version__},
        "preprocessing": "Original XDF-to-cache notch/DC steps; new order-4 Butterworth zero-phase bands",
        "inclusion": "Original keep ranges, valid preceding cue, no clipping >0.5% inside task; no variance MAD exclusion",
        "model": "fixed log channel variance, StandardScaler, logistic regression C=0.3",
        "evaluation": "5x stratified shuffled fivefold CV; one contiguous fivefold CV; balanced and raw accuracy",
        "limitations": [
            "Original filtered/nested model results are not reproduced by this independent pipeline.",
            "Same ten participants and data-informed questions; all inferential results are exploratory.",
            "Cached XDF conversion and notch processing are trusted pending raw validation.",
            "No direct EOG/EMG measurement, imagery performance report or independent cohort.",
            "Frequency bands do not isolate distinct signal generators.",
        ],
    }
    (args.out / "cache_reanalysis_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Wrote", *output_files, "and fig_fixed_model_controls.png", sep="\n")


if __name__ == "__main__":
    main()
