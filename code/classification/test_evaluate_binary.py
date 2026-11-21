import ast
from pathlib import Path
import unittest

import numpy as np

from evaluate_binary import COMPACT_POOL, FULL_POOL, compare_pools, score_metrics


class BinaryEvaluationTests(unittest.TestCase):
    def test_perfect_metrics_for_both_score_directions(self):
        labels = np.array([0, 1, 0, 1])
        metrics = score_metrics(labels, labels, np.array([-2, 4, -1, 2]))
        for name in ("accuracy", "balanced_accuracy", "roc_auc", "macro_f1", "mcc",
                     "pr_auc_first", "pr_auc_second", "ap_first", "ap_second"):
            self.assertEqual(metrics[name], 1.0)

    def test_constant_predictor_leaves_precision_undefined(self):
        labels = np.array([0, 0, 0, 1])
        metrics = score_metrics(labels, np.zeros(4), np.zeros(4))
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["roc_auc"], 0.5)
        self.assertEqual(metrics["ap_first"], 0.75)
        self.assertEqual(metrics["ap_second"], 0.25)
        self.assertIsNone(metrics["precision_second"])

    def test_ablation_and_compact_selection_use_inner_scores(self):
        rows = [dict(subject="S01", contrast="Stand-Walk", fold=1, model=name,
                     inner_accuracy=1 - (i / 10), accuracy=i / 10)
                for i, name in enumerate(FULL_POOL)]
        comparison = dict(compare_pools(rows, FULL_POOL))
        self.assertEqual(comparison["full"]["model"], FULL_POOL[0])
        self.assertEqual(comparison[f"drop:{FULL_POOL[0]}"]["model"], FULL_POOL[1])
        self.assertEqual(comparison["compact"]["model"], COMPACT_POOL[0])
        self.assertEqual(dict(compare_pools([rows[4]], COMPACT_POOL))["compact"], rows[4])

    def test_test_labels_are_only_passed_to_the_scoring_function(self):
        path = Path(__file__).with_name("evaluate_binary.py")
        module = ast.parse(path.read_text())
        compare = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "compare_folds")
        calls = [n for n in ast.walk(compare) if isinstance(n, ast.Call)]
        for callee, expected in (("cross_val_score", ["x[train]", "y[train]"]),
                                 ("fit", ["x[train]", "y[train]"]),
                                 ("score_metrics", ["y[test]", "predicted", "score"])):
            call = next(n for n in calls if ast.unparse(n.func).split(".")[-1] == callee)
            args = call.args[1:] if callee == "cross_val_score" else call.args
            self.assertEqual([ast.unparse(arg) for arg in args], expected)
        self.assertEqual(sum(ast.unparse(n) == "y[test]" for n in ast.walk(compare)), 1)


if __name__ == "__main__":
    unittest.main()
