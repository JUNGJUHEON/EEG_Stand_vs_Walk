"""Post-hoc blocked-label permutation of the complete fixed candidate bank.

This tests the maximum group-mean held-out score across all fixed candidates and
both evaluation splits. It does not validate the nested-selection algorithm or
turn the winning outer-CV accuracy into an independent performance estimate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits

from recheck_stand_walk import HERE, SEED, candidates, splits

DATASET = None


def initialize():
    global DATASET
    DATASET = {}
    for path in sorted((HERE / "private_features").glob("S*.npz")):
        with np.load(path, allow_pickle=False) as z:
            take = z["y"] > 0
            views = {view for _, view, _ in candidates()}
            DATASET[path.stem] = {key: z[key][take] for key in views | {"y", "run"}}


def permuted_maximum(index):
    rng = np.random.default_rng(SEED + 10000 + index)
    bank = candidates()
    scores = np.zeros((2, len(bank), len(DATASET)))
    with threadpool_limits(limits=1):
        for si, (sid, data) in enumerate(DATASET.items()):
            y = data["y"].copy() - 1
            # Preserve local class counts and run boundaries to retain slow drift.
            for run in np.unique(data["run"]):
                order = np.flatnonzero(data["run"] == run)
                for start in range(0, len(order), 10):
                    block = order[start:start+10]
                    y[block] = rng.permutation(y[block])
            for mi, mode in enumerate(("random", "contiguous")):
                outer = splits(y, mode)
                for ci, (name, view, estimator) in enumerate(bank):
                    pred = np.full(len(y), -1, int)
                    for train, test in outer:
                        model = clone(estimator).fit(data[view][train], y[train])
                        pred[test] = model.predict(data[view][test])
                    scores[mi, ci, si] = accuracy_score(y, pred)
    mean = scores.mean(axis=-1)
    return dict(permutation=index, maximum_both_modes=mean.max(),
                maximum_random=mean[0].max(), maximum_contiguous=mean[1].max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = HERE / "results"
    observed = pd.read_csv(root / "fixed_candidate_summary.csv")
    observed_max = float(observed["mean"].max())
    started = time.monotonic()
    rows = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                             initializer=initialize) as executor:
        for row in executor.map(permuted_maximum, range(args.permutations)):
            rows.append(row)
            if len(rows) % 10 == 0 or len(rows) == args.permutations:
                pd.DataFrame(rows).to_csv(root / "blocked_search_permutations.csv", index=False)
                print(f"{len(rows)}/{args.permutations} permutations; {time.monotonic()-started:.1f}s", flush=True)
    null = np.asarray([row["maximum_both_modes"] for row in rows])
    result = dict(observed_maximum=observed_max, n_permutations=len(rows),
                  p_familywise=(1 + int(np.sum(null >= observed_max - 1e-12))) / (len(rows) + 1),
                  null_mean=float(null.mean()), null_95_percentile=float(np.quantile(null, .95)),
                  family="23 fixed candidates x 2 evaluation modes; group mean across 10 participants",
                  exchangeability="labels shuffled in chronological blocks of at most 10 Stand/Walk trials within each run",
                  scope="exploratory global existence test; not nested-algorithm test or independent winner accuracy")
    (root / "blocked_search_test.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
