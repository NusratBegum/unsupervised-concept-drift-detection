import unittest
import warnings

import numpy as np

from detectors.d3_lightgbm import DiscriminativeDriftDetectorLightGBM


class D3LightGBMTest(unittest.TestCase):
    def setUp(self) -> None:
        # LightGBM needs at least 20 samples per leaf, so the windows are larger than in test_d3
        self.detector = DiscriminativeDriftDetectorLightGBM(
            n_reference_samples=100, recent_samples_proportion=1, seed=0
        )

    def test_simple_detection(self):
        warnings.filterwarnings("ignore", category=UserWarning)
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 1, size=(300, 2)), rng.normal(3, 1, size=(300, 2))])
        drifts = [self.detector.update({"0": a, "1": b}) for a, b in X]
        self.assertEqual(0, sum(drifts[:300]))
        self.assertEqual(1, sum(drifts[300:500]))

    def test_step_sets_the_check_interval(self):
        from detectors.d3 import DiscriminativeDriftDetector2019

        def n_checks(**kw):
            d = DiscriminativeDriftDetector2019(n_reference_samples=50, recent_samples_proportion=0.4, seed=0, **kw)
            calls = []
            d._detect_drift = lambda: calls.append(1) or False
            for i in range(1070):
                d.update({"x": float(i)})
            return len(calls)

        # a check every step + 1 samples (the sample that triggers a check is dropped, as upstream)
        self.assertEqual(1000 // 21 + 1, n_checks())  # default step = 0.4 * 50 = 20
        self.assertEqual(1000 // 11 + 1, n_checks(step=10))
        self.assertEqual(5, DiscriminativeDriftDetectorLightGBM(100, 1, n_folds=5).kfold.get_n_splits())


if __name__ == "__main__":
    unittest.main()
