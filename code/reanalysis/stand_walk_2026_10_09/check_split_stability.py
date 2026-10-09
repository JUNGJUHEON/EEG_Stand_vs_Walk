"""Post-hoc stability across four extra CV seeds, without choosing the best seed."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits

import recheck_stand_walk as analysis


def run_repeat(repeat):
    analysis.SEED = 20261009 + repeat
    rows = []
    with threadpool_limits(limits=1):
        for path in sorted((analysis.HERE / "private_features").glob("S*.npz")):
            with np.load(path, allow_pickle=False) as z:
                take = z["y"] > 0
                y = z["y"][take] - 1
                views = {view for _, view, _ in analysis.candidates()}
                data = {key: z[key][take] for key in views}
            pred, fixed, _ = analysis.nested_predict(data, y, analysis.splits(y, "random"), "random")
            rows.append(dict(repeat=repeat, seed=analysis.SEED, subject=path.stem,
                             model="nested_selection", accuracy=accuracy_score(y, pred)))
            for ci, (name, _, _) in enumerate(analysis.candidates()):
                rows.append(dict(repeat=repeat, seed=analysis.SEED, subject=path.stem,
                                 model=name, accuracy=accuracy_score(y, fixed[ci])))
    return rows


def main():
    out = analysis.HERE / "results"
    initial = pd.read_csv(out / "fixed_candidate_scores.csv")
    initial = initial[initial.evaluation == "random"][["subject", "model", "accuracy"]]
    nested = pd.read_csv(out / "within_person_scores.csv")
    nested = nested[(nested.evaluation == "random") & (nested.model == "nested_selection")][["subject", "model", "accuracy"]]
    combined = pd.concat((initial, nested), ignore_index=True)
    combined["repeat"], combined["seed"] = 0, 20261009
    records = combined.to_dict("records")
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as executor:
        for repeat, rows in zip(range(1, 5), executor.map(run_repeat, range(1, 5))):
            records.extend(rows)
            print(f"repeat {repeat} finished", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / "split_stability_scores.csv", index=False)
    participant = frame.groupby(["subject", "model"], as_index=False).accuracy.mean()
    participant.to_csv(out / "split_stability_participant_means.csv", index=False)
    summary = analysis.summarize(participant, ["model"], "accuracy")
    summary.to_csv(out / "split_stability_summary.csv", index=False)
    print(summary[summary.model.isin(["nested_selection", "ear_trajectory_lr0.01"])].to_string(index=False))
    print(frame[frame.model == "nested_selection"].groupby("repeat").accuracy.mean().to_string())


if __name__ == "__main__":
    main()
