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


if __name__ == "__main__":
    unittest.main()
