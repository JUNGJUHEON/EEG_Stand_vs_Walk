import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from model import feature_arrays, make_model
from run import predict, seed_all, split_indices, tensors, train_model, validate_input
from summarize import read_scores, summarize, participant_table

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/reanalysis/filterbank_fixed_2026_10_14"


class FilterbankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fixed_disjoint_original_trials(self):
        y = np.tile([0, 1, 2], 50)
        for pair in ((0, 1), (0, 2), (1, 2)):
            train, val, test = split_indices(y, pair)
            self.assertEqual(tuple(map(len, (train, val, test))), (60, 20, 20))
            for first, second in ((train, val), (train, test), (val, test)):
                self.assertEqual(len(np.intersect1d(first, second)), 0)
            for ids, n in ((train, 30), (val, 10), (test, 10)):
                self.assertEqual([np.count_nonzero(y[ids] == c) for c in pair], [n, n])

    def test_features_do_not_depend_on_other_trials(self):
        x = np.random.default_rng(1).normal(size=(4, 8, 1281))
        combined = feature_arrays(x)
        single = feature_arrays(x[1:2])
        for all_trials, one in zip(combined, single):
            np.testing.assert_array_equal(all_trials[1:2], one)
            self.assertEqual(all_trials.dtype, np.float32)

    def test_training_and_prediction_interfaces_exclude_test_truth(self):
        self.assertEqual(list(inspect.signature(train_model).parameters),
                         ["train_features", "y_train", "val_features", "y_val", "device", "max_epochs"])
        self.assertEqual(list(inspect.signature(predict).parameters), ["model", "features", "batch_size"])

    def test_frozen_prediction_shape_and_no_bn_updates(self):
        seed_all(42)
        model = make_model()
        x = feature_arrays(np.random.default_rng(3).normal(size=(2, 8, 1281)))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        probabilities = predict(model, tensors(x, "cpu"))
        self.assertEqual(probabilities.shape, (2, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1., atol=1e-6)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_bounded_synthetic_training_restores_validation_checkpoint(self):
        rng = np.random.default_rng(4)
        train, val = feature_arrays(rng.normal(size=(8, 8, 1281))), feature_arrays(rng.normal(size=(4, 8, 1281)))
        seed_all(42)
        model, state, history, epoch = train_model(train, np.tile([0, 1], 4),
                                                  val, np.tile([0, 1], 2), "cpu", max_epochs=2)
        best_loss, expected_epoch = float("inf"), 0
        for row in history:
            if row["val_loss"] < best_loss-.0001:
                best_loss, expected_epoch = row["val_loss"], row["epoch"]
        self.assertEqual(epoch, expected_epoch)
        self.assertFalse(model.training)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)

    def test_input_contract(self):
        validate_input(np.zeros((150, 8, 1281)), np.repeat([0, 1, 2], 50))
        with self.assertRaises(ValueError):
            validate_input(np.zeros((150, 8, 1281)), np.zeros(150, dtype=int))

    def test_all_recorded_results_and_readme_table(self):
        rows = read_scores(RESULTS / "participant_scores.csv")
        self.assertEqual(len(rows), 90)
        participant, group = summarize(rows)
        self.assertEqual((len(participant), len(group)), (30, 9))
        self.assertIn(participant_table(rows), (ROOT / "README.md").read_text())
        for row in rows:
            self.assertEqual(row["n_test"], 20)
            self.assertAlmostEqual(row["n_correct"]/20, row["accuracy"])
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "incomplete.csv"
            missing.write_text("subject,contrast,seed,accuracy,auc\nS03,R_vs_S,42,0.9,0.98\n")
            with self.assertRaises(ValueError):
                read_scores(missing)


if __name__ == "__main__":
    unittest.main()
