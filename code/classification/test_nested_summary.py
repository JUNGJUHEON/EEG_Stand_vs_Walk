import ast
import csv
from pathlib import Path
import tempfile
import unittest

from summarize_nested import group_table, participant_table, read_scores, summarize

ROOT = Path(__file__).resolve().parents[2]


class NestedSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = read_scores(ROOT / "results/final_scores.csv")

    def test_complete_binary_records(self):
        self.assertEqual(len(self.records), 30)
        self.assertEqual({r["contrast"] for r in self.records},
                         {"Rest-Stand", "Rest-Walk", "Stand-Walk"})
        self.assertEqual(min(r["n_retained_trials"] for r in self.records), 94)
        self.assertEqual(max(r["n_retained_trials"] for r in self.records), 100)

    def test_saved_participant_export(self):
        with (ROOT / "results/classification/nested_participant_scores.csv").open(newline="") as stream:
            published = list(csv.DictReader(stream))
        self.assertEqual(len(published), 30)
        for source, output in zip(self.records, published):
            for key, value in source.items():
                self.assertEqual(str(value), output[key])

    def test_group_values_match_original_audit(self):
        with (ROOT / "results/reanalysis/nested_score_summary.csv").open(newline="") as stream:
            original = {r["task"]: r for r in csv.DictReader(stream)}
        tasks = {"Rest-Stand": "RvS", "Rest-Walk": "RvW", "Stand-Walk": "SvW"}
        for row in summarize(self.records):
            source = original[tasks[row["contrast"]]]
            for key in ("mean_accuracy", "sd_between_participants", "t_ci95_low", "t_ci95_high"):
                self.assertAlmostEqual(row[key], float(source[key]), places=13)

    def test_both_readme_tables_match(self):
        readme = (ROOT / "README.md").read_text()
        self.assertIn(participant_table(self.records), readme)
        self.assertIn(group_table(summarize(self.records)), readme)

    def test_nested_source_uses_training_fold_for_choice_and_fit(self):
        module = ast.parse((ROOT / "code/02_final_eval.py").read_text())
        function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "nested_score")
        calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)]
        inner = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "cross_val_score")
        self.assertEqual([ast.unparse(n) for n in inner.args[1:]], ["X[tr]", "y[tr]"])
        fitting = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "fit"]
        self.assertEqual(len(fitting), 1)
        self.assertEqual([ast.unparse(n) for n in fitting[0].args], ["X[tr]", "y[tr]"])
        prediction = next(n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "predict")
        self.assertEqual(ast.unparse(prediction.args[0]), "X[te]")

    def test_no_fitting_in_summary_module(self):
        module = ast.parse(Path(__file__).with_name("summarize_nested.py").read_text())
        fitted = [n for n in ast.walk(module) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr in ("fit", "fit_transform")]
        self.assertEqual(fitted, [])

    def test_missing_binary_record_is_rejected(self):
        with (ROOT / "results/final_scores.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = [r for r in reader if r["kind"] == "nested" and r["task"] in ("RvS", "RvW", "SvW")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows[:-1])
            with self.assertRaises(ValueError):
                read_scores(path)


if __name__ == "__main__":
    unittest.main()
