"""Additional LOSO checks with explicit, label-free session calibration.

Session whitening uses all 150 unlabeled EEG trials of each participant,
including the held-out participant. It is transductive offline calibration,
not prediction for a completely unseen, uncalibrated participant.
"""

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from recheck_stand_walk import HERE, PerBandTangent, splits, summarize
from pyriemann.preprocessing import Whitening


def main():
    people, labels, features = [], [], {"trace_normalized": [], "session_whitened": []}
    for group, path in enumerate(sorted((HERE / "private_features").glob("S*.npz"))):
        with np.load(path, allow_pickle=False) as z:
            cov = z["ear_cov"]
            keep = z["y"] > 0
            labels.extend(z["y"][keep] - 1)
        people.extend([group] * int(keep.sum()))
        trace = np.trace(cov, axis1=-2, axis2=-1)
        features["trace_normalized"].append((cov / trace[..., None, None])[keep])
        aligned = np.stack([Whitening(metric="euclid").fit(cov[:, b]).transform(cov[:, b])
                            for b in range(cov.shape[1])], axis=1)
        features["session_whitened"].append(aligned[keep])
    features = {key: np.concatenate(value) for key, value in features.items()}
    y, groups = np.asarray(labels), np.asarray(people)
    rows, choices = [], []
    for view, x in features.items():
        for g in np.unique(groups):
            train, test = np.flatnonzero(groups != g), np.flatnonzero(groups == g)
            inner = splits(y[train], "random", 3, groups[train])
            candidates, scores = [], []
            for c in (.01, .1, 1):
                estimator = make_pipeline(PerBandTangent(), StandardScaler(), LogisticRegression(C=c, max_iter=1000))
                candidates.append(estimator)
                fold_scores = []
                for tr, te in inner:
                    prediction = clone(estimator).fit(x[train[tr]], y[train[tr]]).predict(x[train[te]])
                    fold_scores.extend(accuracy_score(y[train[te]][groups[train[te]] == h],
                                                      prediction[groups[train[te]] == h])
                                       for h in np.unique(groups[train[te]]))
                scores.append(np.mean(fold_scores))
            best = int(np.argmax(scores))
            prediction = clone(candidates[best]).fit(x[train], y[train]).predict(x[test])
            rows.append(dict(subject=f"S{g+1:02d}", model=view, accuracy=accuracy_score(y[test], prediction),
                             test_session_statistics_used=view == "session_whitened"))
            choices.append(dict(subject=f"S{g+1:02d}", model=view, selected_C=(.01, .1, 1)[best],
                                inner_accuracy=scores[best]))
    out = HERE / "results"
    pd.DataFrame(rows).to_csv(out / "alignment_scores.csv", index=False)
    pd.DataFrame(choices).to_csv(out / "alignment_choices.csv", index=False)
    summary = summarize(pd.DataFrame(rows), ["model"], "accuracy")
    summary.to_csv(out / "alignment_summary.csv", index=False)
    (out / "alignment_scope.json").write_text(json.dumps(dict(
        stage="post-hoc additional sensitivity, after initial LOSO result",
        trace_normalized="trial-specific trace scaling; no test-person cohort statistics",
        session_whitened="Euclidean covariance whitening fitted on 150 unlabeled trials per person, including held-out person",
        hyperparameter_selection="C selected by 3-fold participant-exclusive CV within outer training subjects",
        interpretation="descriptive transfer evaluation, no best-view winner accuracy or MI-source inference"), indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
