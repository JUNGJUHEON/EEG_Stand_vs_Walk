"""Evaluate the compact binary pool and optionally compare all six candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from importlib.metadata import version
from pathlib import Path
from statistics import mean
import sys

import numpy as np
from sklearn.metrics import (auc, average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import decoders
import steam_io


TASKS = {"Rest-Stand": (0, 1), "Rest-Walk": (0, 2), "Stand-Walk": (1, 2)}
FULL_POOL = ("csp_lda", "ts_lr", "acm_d5_10_ts_lr", "fgmdm", "fb_ts_lr", "fbcsp_mibif_lda")
COMPACT_POOL = ("fb_ts_lr", "fbcsp_mibif_lda")
INPUT = dict(lo=8.0, hi=30.0, tmin=0.5, tmax=4.5, ref="none")
COUNT_FIELDS = ("tn", "fp", "fn", "tp")


def score_metrics(y, predicted, score):
    """Use model predictions for counts and oriented scores for curve metrics."""
    tn, fp, fn, tp = map(int, confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    recall_first = tn / (tn + fp)
    recall_second = tp / (tp + fn)
    metrics = dict(
        tn=tn, fp=fp, fn=fn, tp=tp,
        accuracy=(tn + tp) / len(y),
        balanced_accuracy=(recall_first + recall_second) / 2,
        recall_first=recall_first, recall_second=recall_second,
        precision_first=tn / (tn + fn) if tn + fn else None,
        precision_second=tp / (tp + fp) if tp + fp else None,
        macro_f1=float(f1_score(y, predicted, average="macro", zero_division=0)),
        mcc=float(matthews_corrcoef(y, predicted)),
        roc_auc=float(roc_auc_score(y, score)),
    )
    for label, suffix, oriented_score in ((0, "first", -score), (1, "second", score)):
        binary = y == label
        precision, recall, _ = precision_recall_curve(binary, oriented_score)
        metrics[f"pr_auc_{suffix}"] = float(auc(recall, precision))
        metrics[f"ap_{suffix}"] = float(average_precision_score(binary, oriented_score))
    metrics["macro_pr_auc"] = (metrics["pr_auc_first"] + metrics["pr_auc_second"]) / 2
    metrics["macro_ap"] = (metrics["ap_first"] + metrics["ap_second"]) / 2
    return metrics


def compare_folds(plain, bank, labels, subject, contrast, candidates):
    first, second = TASKS[contrast]
    mask = np.isin(labels, [first, second])
    inputs = {False: plain[mask], True: bank[mask]}
    y = (labels[mask] == second).astype(int)
    outer = StratifiedKFold(5, shuffle=True, random_state=0)
    inner = StratifiedKFold(4, shuffle=True, random_state=1)
    rows = []
    for fold, (train, test) in enumerate(outer.split(inputs[False], y), 1):
        validation = {}
        for model in candidates:
            x = inputs[decoders.is_bank(model)]
            np.random.seed(1000 * fold + FULL_POOL.index(model))
            validation[model] = float(cross_val_score(
                decoders.get(model), x[train], y[train], cv=inner, scoring="accuracy",
                n_jobs=1, error_score="raise",
            ).mean())

        ranking = sorted(candidates, key=validation.__getitem__, reverse=True)
        compact_pick = next(model for model in ranking if model in COMPACT_POOL)
        evaluate = candidates if candidates == FULL_POOL else (compact_pick,)
        outputs = {}
        for model in evaluate:
            x = inputs[decoders.is_bank(model)]
            np.random.seed(1000 * fold + 100 + FULL_POOL.index(model))
            fitted = decoders.get(model).fit(x[train], y[train])
            test_x = x[test]
            predicted = fitted.predict(test_x)
            if hasattr(fitted, "decision_function"):
                score = fitted.decision_function(test_x)
            else:
                positive = np.flatnonzero(fitted.classes_ == 1).item()
                score = fitted.predict_proba(test_x)[:, positive]
            outputs[model] = (predicted, score)

        for model, (predicted, score) in outputs.items():
            rows.append(dict(
                subject=subject, contrast=contrast, fold=fold, model=model,
                n_train=len(train), n_test=len(test), inner_accuracy=validation[model],
                selected_compact=int(model == compact_pick), **score_metrics(y[test], predicted, score),
            ))
    return rows


def configure_loader(data, cache):
    steam_io.BASE_DIR = str(data)
    if cache is not None:
        steam_io.CACHE_DIR = str(cache)
        cache.mkdir(parents=True, exist_ok=True)


def work(params):
    data, cache, number, date, identifier, candidates = params
    configure_loader(data, cache)
    plain, bank, labels = steam_io.get_task_arrays(date, identifier, **INPUT)
    subject = f"S{number:02d}"
    return [r for contrast in TASKS
            for r in compare_folds(plain, bank, labels, subject, contrast, candidates)]


def compare_pools(rows, candidates):
    """Match pre-existing model scores using the recorded training-fold rankings."""
    groups = defaultdict(dict)
    for r in rows:
        groups[r["subject"], r["contrast"], r["fold"]][r["model"]] = r
    selected = []
    for by_model in groups.values():
        if candidates != FULL_POOL:
            selected.append(("compact", next(iter(by_model.values()))))
            continue
        ranking = sorted(candidates, key=lambda model: by_model[model]["inner_accuracy"], reverse=True)
        compact = next(model for model in ranking if model in COMPACT_POOL)
        selected.extend((strategy, by_model[model])
                        for strategy, model in (("compact", compact), ("full", ranking[0])))
        for model in candidates:
            alternate = next(pick for pick in ranking if pick != model)
            selected.extend(((f"drop:{model}", by_model[alternate]),
                             (f"fixed:{model}", by_model[model])))
    return selected


def summarize(rows, candidates):
    selected = compare_pools(rows, candidates)
    fields = list(rows[0])
    metrics = fields[fields.index("accuracy"):]
    by_person = defaultdict(list)
    by_group = defaultdict(list)
    for strategy, r in selected:
        by_person[strategy, r["contrast"], r["subject"]].append(r)

    person_rows = []
    for (strategy, contrast, subject), person in by_person.items():
        result = dict(strategy=strategy, contrast=contrast, subject=subject,
                      n_folds=len(person), n_test=sum(r["n_test"] for r in person))
        result.update({c: sum(r[c] for r in person) for c in COUNT_FIELDS})
        for metric in metrics:
            values = [r[metric] for r in person if r[metric] is not None]
            result[metric] = mean(values) if values else None
        person_rows.append(result)
        by_group[strategy, contrast].append(result)

    group_rows = []
    for (strategy, contrast), people in by_group.items():
        result = dict(strategy=strategy, contrast=contrast, n_participants=len(people),
                      n_folds=sum(r["n_folds"] for r in people),
                      n_test=sum(r["n_test"] for r in people))
        result.update({c: sum(r[c] for r in people) for c in COUNT_FIELDS})
        for metric in metrics:
            values = [r[metric] for r in people if r[metric] is not None]
            result[metric] = mean(values) if values else None
        group_rows.append(result)
    return person_rows, group_rows


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing recording dates")
    parser.add_argument("--cache", type=Path, help="Directory for private EEG caches")
    parser.add_argument("--out", type=Path, required=True, help="Directory for aggregate tables")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--full-comparison", action="store_true",
                        help="Also score all six candidates and leave-one-candidate-out pools")
    args = parser.parse_args()
    configure_loader(args.data, args.cache)
    candidates = FULL_POOL if args.full_comparison else COMPACT_POOL
    subjects = steam_io.list_subjects()
    if not subjects:
        parser.error("no recordings were found in --data")
    tasks = [(args.data, args.cache, i, date, key, candidates)
             for i, (date, key) in enumerate(subjects, 1)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for result in pool.map(work, tasks):
            rows.extend(result)
            print(f'{result[0]["subject"]}: {len(result)} fold/model evaluations', flush=True)
    people, groups = summarize(rows, candidates)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, records in (("binary_fold_metrics.csv", rows), ("binary_participant_metrics.csv", people),
                          ("binary_group_metrics.csv", groups)):
        write_csv(args.out / name, records)
    print({name: version(name) for name in ("numpy", "scipy", "scikit-learn", "pyriemann", "mne")})


if __name__ == "__main__":
    main()
