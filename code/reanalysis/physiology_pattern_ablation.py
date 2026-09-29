"""Post-hoc separation of global band power and relative scalp pattern."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from contrast_logic_checks import balanced_cv
from reanalyse_cached_eeg import participant_files, read_participant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for participant, date, runs in participant_files(args.cache):
        features, labels, _, _, _ = read_participant(date, runs)
        mu = features["task_mu_8_13"]
        beta = features["task_beta_13_30"]
        global_power = np.column_stack([mu.mean(axis=1), beta.mean(axis=1)])
        spatial_shape = np.column_stack([
            mu - mu.mean(axis=1, keepdims=True),
            beta - beta.mean(axis=1, keepdims=True),
        ])
        definitions = (("global_mu_beta", global_power),
                       ("spatial_shape_mu_beta", spatial_shape),
                       ("full_mu_beta", np.column_stack([global_power, spatial_shape])))
        for contrast, mask, y in (
            ("Rest-AnyMI", np.ones(len(labels), dtype=bool), labels != 0),
            ("Stand-Walk", labels != 0, labels == 2),
        ):
            for feature, x in definitions:
                rows.append({"participant": participant, "contrast": contrast,
                             "feature": feature, "n_features": x.shape[1],
                             "n_trials": int(mask.sum()),
                             "balanced_accuracy": balanced_cv(x[mask], y[mask].astype(int))})
    participant = pd.DataFrame(rows)
    participant.to_csv(args.out / "physiology_pattern_participant.csv", index=False)
    summary = []
    for (contrast, feature), group in participant.groupby(["contrast", "feature"]):
        values = group.balanced_accuracy.to_numpy() * 100
        width = t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
        summary.append({"contrast": contrast, "feature": feature,
                        "n_participants": len(values), "mean_balanced_accuracy_pct": values.mean(),
                        "ci95_low_pct": values.mean() - width,
                        "ci95_high_pct": values.mean() + width})
    pd.DataFrame(summary).to_csv(args.out / "physiology_pattern_group.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
