"""Synthetic checks for aggregation and fail-closed historical-log auditing."""

import csv
from pathlib import Path
import tempfile
import unittest

from audit_md2d_grid import CONTRASTS, NORMS, audit, participant_map, read_grid, select_test_max


class GridAuditTests(unittest.TestCase):
    def test_max_is_per_participant_not_all_row_mean(self):
        rows = [{"test_acc_fused": .5, "val_acc_fused": .9},
                {"test_acc_fused": .8, "val_acc_fused": .4},
                {"test_acc_fused": .8, "val_acc_fused": .6}]
        self.assertEqual(select_test_max(rows), rows[2])

    def test_incomplete_log_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "grid.csv"
            path.write_text("F1,D,F2,DROPOUT,LR,val_acc_fused,test_acc_fused,test_f1_fused\n"
                            "8,2,16,.5,.001,.5,.8,.7\n")
            with self.assertRaises(ValueError):
                read_grid(path)

    def test_full_scope_and_anonymized_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache, logs, out = root / "cache", root / "logs", root / "out"
            cache.mkdir()
            keys = [f"2509{i:02d}_PRIVATE{i}" for i in range(1, 11)]
            for key in keys:
                (cache / f"{key}_MI.npz").touch()
            self.assertEqual(participant_map(cache)[keys[0]], "S01")
            for norm in NORMS:
                (logs / f"Norm_{norm}").mkdir(parents=True)
                for contrast in CONTRASTS:
                    for person, key in enumerate(keys):
                        with (logs / f"Norm_{norm}" / f"{key}_{contrast}_gridlog.csv").open("w", newline="") as stream:
                            writer = csv.writer(stream)
                            writer.writerow(("F1", "D", "F2", "DROPOUT", "LR", "val_acc_fused", "test_acc_fused", "test_f1_fused"))
                            for cfg in range(18):
                                score = .6 + .01 * person if cfg == 17 and norm == "CT" else .5
                                writer.writerow((cfg + 1, 2, 16, .5, .001, .5, score, .5))
            scope = audit(logs, cache, out)
            self.assertEqual(scope["total_binary_configuration_rows"], 3780)
            with (out / "historical_summary.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertAlmostEqual(float(row["mean_participant_test_max"]), .645)
            for path in out.iterdir():
                self.assertNotIn("PRIVATE", path.read_text())
                self.assertNotIn(b"\r", path.read_bytes())
            (logs / "Norm_CT" / f"{keys[0]}_S_vs_W_gridlog.csv").unlink()
            with self.assertRaises(ValueError):
                audit(logs, cache, out)


if __name__ == "__main__":
    unittest.main()
