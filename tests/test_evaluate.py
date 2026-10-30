"""Contract tests for valid-support segmentation metrics."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np


def implementation():
    spec = importlib.util.spec_from_file_location("evaluate", Path(__file__).parents[1] / "src/cv_final/evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate_segmentation


class EvaluationTests(unittest.TestCase):
    def test_known_confusion_fixture(self):
        evaluate = implementation()
        pred = np.array([[True, True, False, False]])
        truth = np.array([[True, False, True, False]])
        result = evaluate(pred, truth, np.ones_like(pred))
        self.assertEqual({k: result[k] for k in ('tp', 'fp', 'fn', 'tn')}, dict(tp=1, fp=1, fn=1, tn=1))
        self.assertEqual(result['valid_pixels'], 4)
        self.assertAlmostEqual(result['iou'], 1 / 3)
        for name in ('precision', 'recall', 'f1'):
            self.assertAlmostEqual(result[name], .5)

    def test_zero_union_is_perfect(self):
        zeros = np.zeros((2, 2), dtype=bool)
        result = implementation()(zeros, zeros, ~zeros)
        for key in ('iou', 'precision', 'recall', 'f1'):
            self.assertEqual(result[key], 1.0)

    def test_empty_prediction_with_truth_scores_zero(self):
        pred = np.zeros((1, 2), dtype=bool)
        result = implementation()(pred, ~pred, ~pred)
        for key in ('iou', 'precision', 'recall', 'f1'):
            self.assertEqual(result[key], 0.0)

    def test_empty_truth_with_predictions_scores_zero(self):
        truth = np.zeros((1, 2), dtype=bool)
        result = implementation()(~truth, truth, ~truth)
        for key in ('iou', 'precision', 'recall', 'f1'):
            self.assertEqual(result[key], 0.0)

    def test_empty_support_is_rejected(self):
        zeros = np.zeros((2, 2), dtype=bool)
        with self.assertRaises(ValueError):
            implementation()(zeros, zeros, zeros)

    def test_boolean_2d_same_shape_inputs_required(self):
        good = np.ones((2, 2), dtype=bool)
        bad_arrays = [good.astype(np.uint8), good.astype(float), good[..., None],
                      np.ones((1, 2), dtype=bool), [[True, True], [True, True]], None]
        for position in range(3):
            for bad in bad_arrays:
                with self.subTest(position=position, bad=type(bad)):
                    args = [good, good, good]
                    args[position] = bad
                    with self.assertRaises(ValueError):
                        implementation()(*args)

    def test_ignore_region_does_not_change_metrics(self):
        pred = np.array([[True, True, False, False]])
        truth = np.array([[True, False, True, False]])
        valid = np.array([[True, False, False, True]])
        result = implementation()(pred, truth, valid)
        self.assertEqual(result, dict(tp=1, fp=0, fn=0, tn=1, iou=1.,
                                     precision=1., recall=1., f1=1., valid_pixels=2))


if __name__ == '__main__':
    unittest.main()
