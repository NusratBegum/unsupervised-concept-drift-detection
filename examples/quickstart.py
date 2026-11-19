# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Minimal EADD example: one synthetic stream, EADD with the settings of the final registered test,
the detections and the top TreeSHAP features at each alarm. Needs no dataset files.

  python examples/quickstart.py                 # SynthGradual, seed 0 (about a minute)
  python examples/quickstart.py SynthAbrupt 3
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.eadd import ExplainableAdversarialDriftDetector
from synthetic_streams import synthetic_stream

name = sys.argv[1] if len(sys.argv) > 1 else "SynthGradual"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
stream = synthetic_stream(name, seed)  # 20,000 samples x 10 features; x0-x2 drift at 5k, 10k, 15k

# experiment_final.EADD_KW: experiment_validity.FIXED plus the history null
detector = ExplainableAdversarialDriftDetector(
    n_reference_samples=500, n_current_samples=200, auc_threshold=0.7, monitoring_frequency=50,
    n_permutations=199, significance_level=0.05, permutation_test="aspt", seed=seed,
    reservoir_after_current=True, perm_n_folds=None, aspt_alpha_split=1 / 3,
    null="history", history_stride=10)

print(f"{name}, seed {seed}: true drift points {stream.drifts}, drifting features "
      f"{['x%d' % j for j in stream.shifted]}")
detections = []
for i, (x, _) in enumerate(stream):  # the label is never shown to the detector
    if detector.update(x):
        detections.append(i)
        top = list(detector.last_shap_importances.items())[:3]
        print(f"alarm at {i:>6}: AUC {detector.last_auc:.3f}, p {detector.last_p_value:.3f}, top features "
              + ", ".join(f"{f} ({v:.0f}%)" for f, v in top))
print(f"{len(detections)} alarms; {detector.n_tests} tests, {detector.n_permutations_used} null draws in all")
