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

    def test_full_permutation_test_uses_all_permutations(self):
        detector = ExplainableAdversarialDriftDetector(
            n_reference_samples=200, n_current_samples=100, n_permutations=30,
            permutation_test="full", seed=0
        )
        ref = self.rng.normal(0, 1, size=(200, 3))
        cur = self.rng.normal(3, 1, size=(100, 3))
        p_value = detector._aspt_test(ref, cur, actual_auc=0.99)
        self.assertEqual(30, detector.last_aspt_permutations)
        self.assertAlmostEqual(1 / 31, p_value)
        self.assertEqual(0, detector.n_early_stops)

    def test_test_modes_give_same_detections_when_decisions_agree(self):
        X = np.vstack([self.rng.normal(0, 1, size=(1500, 3)), self.rng.normal(2, 1, size=(1500, 3))])
        runs = {}
        for mode in ("aspt", "none"):
            detector = ExplainableAdversarialDriftDetector(
                n_reference_samples=200, n_current_samples=100, monitoring_frequency=50,
                permutation_test=mode, seed=0
            )
            runs[mode] = feed(detector, X)
            if mode == "aspt":
                self.assertEqual(0, detector.n_rejections)
        # the permutations use their own generator, so the reservoir samples are the same
        self.assertEqual(runs["aspt"], runs["none"])

    def test_sliding_reference_holds_the_samples_before_the_current_window(self):
        detector = ExplainableAdversarialDriftDetector(
            n_reference_samples=5, n_current_samples=3, use_reservoir_sampling=False,
            sliding_reference=True, monitoring_frequency=10**6, seed=0
        )
        for i in range(20):
            detector.update({"x": float(i)})
        self.assertEqual([12, 13, 14, 15, 16], [int(v[0]) for v in detector.reference_window])
        self.assertEqual([17, 18, 19], [int(v[0]) for v in detector.current_window])


if __name__ == "__main__":
    unittest.main()
