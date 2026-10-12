"""Audit historical MD2DEEGNet grid logs without loading or publishing EEG.

The primary statistic is a mean of participant-specific test-selected maxima,
NOT an independently evaluated accuracy. All three binary contrasts use the
same seven-normalization, eighteen-configuration scope.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


NORMS = ("C", "CT", "N", "NC", "NCT", "NT", "T")
CONTRASTS = ("R_vs_S", "R_vs_W", "S_vs_W")
CONFIG = ("F1", "D", "F2", "DROPOUT", "LR")
SCORES = ("val_acc_fused", "test_acc_fused", "test_f1_fused")


def participant_map(cache: Path) -> dict[str, str]:
    # Use the same chronological participant order as the other reanalyses.
    keys = set()
    for path in cache.glob("*_MI*.npz"):
        parts = path.stem.split("_")
        if len(parts) == 3 and parts[2] in ("MI", "MI2"):
            keys.add("_".join(parts[:2]))
    if len(keys) != 10:
        raise ValueError(f"Expected ten participants, found {len(keys)}")
    return {key: f"S{i:02d}" for i, key in enumerate(sorted(keys), 1)}


def read_grid(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(CONFIG + SCORES).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing grid columns in {path.name}")
        rows = [{key: float(row[key]) for key in CONFIG + SCORES} for row in reader]
    if len(rows) != 18:
        raise ValueError(f"Expected eighteen configurations in {path.name}")
    if len({tuple(row[key] for key in CONFIG) for row in rows}) != 18:
        raise ValueError(f"Duplicate configuration in {path.name}")
    for row in rows:
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError(f"Nonfinite value in {path.name}")
        if not all(0 <= row[key] <= 1 for key in SCORES):
            raise ValueError(f"Score outside [0, 1] in {path.name}")
    return rows


def select_test_max(rows: list[dict]) -> dict:
    # The test maximum is the estimand. Validation accuracy breaks exact ties;
    # remaining ties retain input order. No generalization claim follows.
    return max(rows, key=lambda row: (row["test_acc_fused"], row["val_acc_fused"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit(logs: Path, cache: Path, out: Path) -> dict:
    mapping = participant_map(cache)
    actual_norms = {p.name.removeprefix("Norm_") for p in logs.glob("Norm_*") if p.is_dir()}
    if actual_norms != set(NORMS):
        raise ValueError(f"Unexpected normalization scope: {sorted(actual_norms)}")
    all_rows, selected, summary = [], [], []
    expected_configs = None
    for contrast in CONTRASTS:
        contrast_rows = []
        for norm in NORMS:
            files = sorted((logs / f"Norm_{norm}").glob(f"*_{contrast}_gridlog.csv"))
            stems = {p.name.removesuffix(f"_{contrast}_gridlog.csv") for p in files}
            if stems != set(mapping) or len(files) != 10:
                raise ValueError(f"Incomplete participant scope: {norm}/{contrast}")
            for path in files:
                subject = mapping[path.name.removesuffix(f"_{contrast}_gridlog.csv")]
                grid = read_grid(path)
                signatures = {tuple(row[key] for key in CONFIG) for row in grid}
                if expected_configs is None:
                    expected_configs = signatures
                if signatures != expected_configs:
                    raise ValueError(f"Configuration grid differs in {path.name}")
                for index, row in enumerate(grid, 1):
                    contrast_rows.append(dict(subject=subject, contrast=contrast, norm=norm,
                                              config_index=index, **row))
        for subject in sorted(mapping.values()):
            person = [row for row in contrast_rows if row["subject"] == subject]
            if len(person) != 126:
                raise ValueError("Participant does not have exactly 126 candidates")
            selected.append(dict(select_test_max(person), n_candidates=126,
                                 selection_rule="maximum_test_accuracy"))
        winners = [row["test_acc_fused"] for row in selected if row["contrast"] == contrast]
        ct = [row for row in contrast_rows if row["norm"] == "CT"]
        ct_best = [select_test_max([row for row in ct if row["subject"] == subject])["test_acc_fused"]
                   for subject in sorted(mapping.values())]
        summary.append(dict(contrast=contrast, participants=10, norm_count=7,
                            configs_per_norm=18, rows=1260,
                            mean_participant_test_max=mean(winners),
                            min_participant_test_max=min(winners),
                            max_participant_test_max=max(winners),
                            mean_participant_CT_test_max=mean(ct_best)))
        all_rows.extend(sorted(contrast_rows, key=lambda row: (row["subject"], row["norm"], row["config_index"])))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "historical_grid_anonymized.csv", all_rows)
    write_csv(out / "historical_selected.csv", selected)
    write_csv(out / "historical_summary.csv", summary)
    scope = dict(
        audit_date="2026-10-12", operation="reaggregate_saved_logs_not_retraining",
        contrasts=list(CONTRASTS), participants=10, normalizations=list(NORMS),
        configurations_per_normalization=18, total_binary_log_files=210,
        total_binary_configuration_rows=len(all_rows),
        selection_rule="participant-specific maximum of 126 recorded test accuracies",
        interpretation="descriptive post-test-selected maximum; not independent generalization accuracy",
        provenance_limit="historical source version, preprocessing and RNG parity not fully reconstructed",
        publication_scope="pseudonyms and model scores only; no EEG, direct identifiers, local paths or input hashes",
        author_approval="PENDING",
    )
    (out / "audit_scope.json").write_text(json.dumps(scope, indent=2) + "\n", encoding="utf-8")
    return scope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True, help="Authorized historical grid-log root")
    parser.add_argument("--cache", type=Path, required=True, help="Authorized cache, used for participant ordering only")
    parser.add_argument("--out", type=Path, default=Path("cache/md2d_grid_audit"))
    args = parser.parse_args()
    scope = audit(args.logs, args.cache, args.out)
    print(f"Audited {scope['total_binary_configuration_rows']} rows across all three binary contrasts.")


if __name__ == "__main__":
    main()
