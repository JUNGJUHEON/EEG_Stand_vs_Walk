"""Matched, post-hoc sensitivity checks, not another winner search."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from recheck_stand_walk import HERE, CACHE_DIR, PerBandTangent, participant_files, splits, summarize


def main():
    rows, clipping, diagnostics, hashes = [], [], [], []
    out = HERE / "results"
    for path in sorted((HERE / "private_features").glob("S*.npz")):
        with np.load(path, allow_pickle=False) as z:
            selected = z["y"] > 0
            y = z["y"][selected] - 1
            power = z["ear_power"][selected].reshape(-1, 4, 8)
            data = {
                "power_all8": power.reshape(-1, 32),
                "power_midline2": power[:, :, [4, 7]].reshape(-1, 8),
                "power_lateral6": power[:, :, [0, 1, 2, 3, 5, 6]].reshape(-1, 24),
                "uncapped_cov": z["ear_broad_cov"][selected],
                "winsorized_cov": z["ear_robust_cov"][selected],
            }
            for ref in ("ear", "car"):
                clipping.append(dict(subject=path.stem, reference=ref,
                                     mean_capped_fraction=float(z[ref+"_winsor_fraction"].mean()),
                                     max_capped_fraction=float(z[ref+"_winsor_fraction"].max())))
        for mode in ("random", "contiguous"):
            for name, x in data.items():
                if "cov" in name:
                    estimator = make_pipeline(PerBandTangent(), StandardScaler(),
                                              LogisticRegression(C=.1, max_iter=1000))
                else:
                    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000))
                pred, training = np.full(len(y), -1), []
                for train, test in splits(y, mode):
                    model = clone(estimator).fit(x[train], y[train])
                    pred[test] = model.predict(x[test])
                    training.append(accuracy_score(y[train], model.predict(x[train])))
                rows.append(dict(subject=path.stem, evaluation=mode, model=name,
                                 accuracy=accuracy_score(y, pred), training_accuracy=np.mean(training)))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "matched_controls.csv", index=False)
    summarize(frame, ["evaluation", "model"], "accuracy").to_csv(out / "matched_controls_summary.csv", index=False)
    pd.DataFrame(clipping).to_csv(out / "winsorization_fraction.csv", index=False)
    for mode in ("random", "contiguous"):
        scores = frame[frame.evaluation == mode].pivot(index="subject", columns="model", values="accuracy")
        for a, b in (("uncapped_cov", "winsorized_cov"), ("power_midline2", "power_all8")):
            difference = scores[b] - scores[a]
            half = stats.t.ppf(.975, 9) * difference.std(ddof=1) / np.sqrt(10)
            diagnostics.append(dict(evaluation=mode, comparison=b+"_minus_"+a,
                                    difference=float(difference.mean()), ci_low=float(difference.mean()-half),
                                    ci_high=float(difference.mean()+half), scope="post-hoc descriptive matched sensitivity"))
    pd.DataFrame(diagnostics).to_csv(out / "matched_control_differences.csv", index=False)
    normality = []
    within = pd.read_csv(out / "within_person_scores.csv")
    for mode in ("random", "contiguous"):
        values = within[(within.model == "nested_selection") & (within.evaluation == mode)].accuracy
        result = stats.shapiro(values - .5)
        normality.append(dict(evaluation=mode, n=len(values), shapiro_w=result.statistic,
                              shapiro_p=result.pvalue,
                              interpretation="Small-sample diagnostic, not proof of normality."))
    pd.DataFrame(normality).to_csv(out / "participant_score_diagnostics.csv", index=False)
    cache = CACHE_DIR
    for sid, _, runs in participant_files(cache):
        for tag, _, path in runs:
            hashes.append(dict(subject=sid, run=tag, bytes=path.stat().st_size,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (out / "input_checksums.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(pd.read_csv(out / "matched_controls_summary.csv").to_string(index=False))
    print(pd.DataFrame(diagnostics).to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
