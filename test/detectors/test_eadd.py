import unittest
import warnings

import numpy as np

from detectors.eadd import ExplainableAdversarialDriftDetector


def feed(detector, X):
    return [detector.update({str(j): float(v) for j, v in enumerate(x)}) for x in X]


class EADDTest(unittest.TestCase):
    def setUp(self) -> None:
        warnings.filterwarnings("ignore", category=UserWarning)
        self.rng = np.random.default_rng(0)
        self.detector = ExplainableAdversarialDriftDetector(
            n_reference_samples=200, n_current_samples=100, monitoring_frequency=50, seed=0
        )

    def test_aspt_early_reject_on_clear_shift(self):
        ref = self.rng.normal(0, 1, size=(200, 3))
        cur = self.rng.normal(3, 1, size=(100, 3))
        p_value = self.detector._aspt_test(ref, cur, actual_auc=0.99)
        # no permutation can reach AUC 0.99, so ASPT stops at the first i with 1/(i+1) < alpha
        self.assertEqual(20, self.detector.last_aspt_permutations)
        self.assertAlmostEqual(1 / 21, p_value)

    def test_aspt_early_accept_without_shift(self):
        ref = self.rng.normal(0, 1, size=(200, 3))
        cur = self.rng.normal(0, 1, size=(100, 3))
        p_value = self.detector._aspt_test(ref, cur, actual_auc=0.5)
        self.assertGreaterEqual(p_value, 0.05)
        self.assertLess(self.detector.last_aspt_permutations, 199)

    def test_no_detection_on_stable_stream(self):
        X = self.rng.normal(0, 1, size=(3000, 3))
        self.assertEqual(0, sum(feed(self.detector, X)))

    def test_detects_abrupt_shift_after_onset(self):
        X = np.vstack([self.rng.normal(0, 1, size=(1500, 3)), self.rng.normal(2, 1, size=(1500, 3))])
        drifts = feed(self.detector, X)
        self.assertEqual(0, sum(drifts[:1500]))
        self.assertGreaterEqual(sum(drifts[1500:1800]), 1)


if __name__ == "__main__":
    unittest.main()
