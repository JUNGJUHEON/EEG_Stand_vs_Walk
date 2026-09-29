"""Recalculate summaries of stored scores, without fitting EEG models.

These are retrospective checks conditional on the original model and data
selection. Participant-level intervals do not account for that selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.minimum(1, np.maximum.accumulate(
        values[order] * np.arange(len(values), 0, -1)))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    scores = pd.read_csv(source / "results/final_scores.csv")
    nested = scores.loc[scores["kind"].eq("nested")].copy()
    expected = {"SvW", "RvS", "RvW", "3cls"}
    if set(nested["task"]) != expected:
        raise ValueError("Unexpected task set")
    if nested.duplicated(["subject", "task"]).any():
        raise ValueError("Repeated participant/task rows")
    if not nested["subject"].str.fullmatch(r"S\d{2}").all():
        raise ValueError("Expected anonymized participant identifiers")
    if not nested["acc"].between(0, 1).all():
        raise ValueError("Invalid accuracy")

    rows = []
    for task in ("SvW", "RvS", "RvW", "3cls"):
        group = nested.loc[nested["task"].eq(task)].sort_values("subject")
        values = group["acc"].to_numpy()
        if len(values) != 10 or set(group["subject"]) != {f"S{i:02}" for i in range(1, 11)}:
            raise ValueError("Expected the same ten participants in each task")
        chance = 1 / 3 if task == "3cls" else 0.5
        mean, sd = values.mean(), values.std(ddof=1)
        se = sd / np.sqrt(len(values))
        half = stats.t.ppf(0.975, len(values) - 1) * se
        test = stats.ttest_1samp(values, chance)
        leave_one_out = (values.sum() - values) / (len(values) - 1)
        rows.append({
            "task": task, "n_participants": len(values), "chance": chance,
            "mean_accuracy": mean, "sd_between_participants": sd,
            "t_ci95_low": mean - half, "t_ci95_high": mean + half,
            "median_accuracy": np.median(values),
            "minimum_accuracy": values.min(), "maximum_accuracy": values.max(),
            "excess_over_chance": mean - chance,
            "t_vs_chance": test.statistic, "df": len(values) - 1,
            "p_two_sided_original_test": test.pvalue,
            "shapiro_p_diagnostic_only": stats.shapiro(values).pvalue,
            "leave_one_participant_out_mean_min": leave_one_out.min(),
            "leave_one_participant_out_mean_max": leave_one_out.max(),
        })
    summary = pd.DataFrame(rows)
    summary["p_holm_four_tasks"] = holm(summary["p_two_sided_original_test"])
    summary.to_csv(args.out / "nested_score_summary.csv", index=False)
    nested[["subject", "task", "acc", "n", "note"]].to_csv(
        args.out / "participant_nested_scores.csv", index=False)

    permutation = pd.read_csv(source / "results/stats_permutation.csv")
    if permutation.duplicated(["subject", "task"]).any():
        raise ValueError("Repeated permutation rows")
    p_rows = []
    for task in ("SvW", "RvS", "RvW", "3cls"):
        group = permutation.loc[permutation["task"].eq(task)]
        if len(group) != 10:
            raise ValueError("Unexpected permutation sample size")
        if not np.allclose(holm(group["p"]), group["p_holm"]):
            raise ValueError("Stored Holm correction differs from recalculation")
        p_rows.append({
            "task": task, "pipeline": "fb_ts_lr",
            "n_participants": len(group),
            "mean_fixed_pipeline_accuracy": group["acc"].mean(),
            "n_p_raw_below_005": int(group["p"].lt(0.05).sum()),
            "n_p_holm_below_005": int(group["p_holm"].lt(0.05).sum()),
            "correction_family": "ten_participants_within_this_task",
            "tests_nested_pipeline": False,
        })
    pd.DataFrame(p_rows).to_csv(args.out / "fixed_pipeline_permutation_summary.csv", index=False)

    inputs = [
        "results/final_scores.csv", "results/stats_permutation.csv",
        "code/02_final_eval.py", "code/09_stats.py", "code/steam_io.py",
        "code/decoders.py", "code/15_peak_locked.py", "code/07_svw_focus.py",
    ]
    manifest = {
        "analysis_type": "retrospective_stored_score_audit_not_raw_reproduction",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {p: hashlib.sha256((source / p).read_bytes()).hexdigest() for p in inputs},
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy")},
        "unit_of_analysis": "participant",
        "limits": [
            "Original preprocessing and model search used these same data.",
            "Intervals and p-values are conditional on stored model estimates.",
            "Normality diagnostic is reported without selecting another test.",
            "Ten participants give limited precision; leave-one-out is descriptive.",
            "Fixed-pipeline permutation results are not tests of nested selection.",
            "No universal device-use threshold is assumed.",
        ],
    }
    (args.out / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary[["task", "mean_accuracy", "t_ci95_low", "t_ci95_high",
                   "p_two_sided_original_test", "p_holm_four_tasks"]].to_string(index=False))
    print("Stored Holm corrections verified; raw EEG models were not rerun.")


if __name__ == "__main__":
    main()
