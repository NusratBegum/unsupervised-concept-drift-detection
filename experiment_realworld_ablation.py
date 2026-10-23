# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Real-World Component Ablation (Thesis Review, 12 Oct 2026)
==========================================================
Reruns Experiment 2 (13 real-world streams from the Lukats et al. benchmark)
with the protocol of EADD_Thesis_Experiments.ipynb (up to 100,000 samples,
detections matched within 500 samples after a known drift), adding the
ablation variants used on the synthetic streams:

  D3 (logistic regression) -> D3-LightGBM -> EADD-NoASPT -> EADD

All detectors use the same window sizes (N_ref=500, N_cur=200) and three
seeds. EADD is also logged per monitoring step (AUC, H_pred, p-value,
permutations used by ASPT) to check the component analysis in the thesis.

Outputs:
  experiments/results_realworld_ablation/detections.csv
  experiments/results_realworld_ablation/eadd_steps.csv
"""
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detectors.eadd import ExplainableAdversarialDriftDetector
from detectors.d3 import DiscriminativeDriftDetector2019
from detectors.d3_lightgbm import DiscriminativeDriftDetectorLightGBM
from experiment_camera_ready import EADDNoASPT

SEEDS = [42, 142, 242]
MAX_SAMPLES = 100_000
TOLERANCE = 500
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "experiments", "results_realworld_ablation")

DATASETS = [
    "Elec2", "InsectsAbruptBalanced", "InsectsGradualBalanced",
    "InsectsIncrementalBalanced", "InsectsIncrementalAbruptBalanced",
    "InsectsIncrementalReoccurringBalanced", "Luxembourg", "NOAAWeather",
    "OutdoorObjects", "Ozone", "PokerHand", "Powersupply", "RialtoBridgeTimelapse",
]


def load_dataset(name):
    import datasets as ds
    cls = {"Elec2": ds.Electricity}.get(name) or getattr(ds, name)
    return cls()


# ──────────────────────────────────────────────────────────────
# EADD with per-step logging
# ──────────────────────────────────────────────────────────────
class LoggingEADD(ExplainableAdversarialDriftDetector):
    """EADD that records AUC, H_pred, p-value and ASPT permutations at every check."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.t = 0
        self.steps = []

    def _detect_drift(self):
        self.last_aspt_permutations = None
        drift = super()._detect_drift()
        self.steps.append(dict(t=self.t, auc=self.last_auc, hpred=self.last_hpred,
                               p_value=self.last_p_value,
                               aspt_permutations=self.last_aspt_permutations or 0,
                               drift=int(drift)))
        return drift


def make_detector(name, seed):
    eadd_kwargs = dict(n_reference_samples=500, n_current_samples=200, auc_threshold=0.7,
                       n_permutations=199, significance_level=0.05, monitoring_frequency=50,
                       seed=seed)
    d3_kwargs = dict(n_reference_samples=500, recent_samples_proportion=0.4, threshold=0.7, seed=seed)
    if name == "EADD":
        return LoggingEADD(**eadd_kwargs)
    if name == "EADD-NoASPT":
        return EADDNoASPT(**eadd_kwargs)
    if name == "D3-LightGBM":
        return DiscriminativeDriftDetectorLightGBM(**d3_kwargs)
    return DiscriminativeDriftDetector2019(**d3_kwargs)


# same matching rule as compute_detection_metrics in EADD_Thesis_Experiments.ipynb
def compute_detection_metrics(detections, true_drifts, tolerance=TOLERANCE):
    tp, delays, matched = 0, [], set()
    for d in sorted(detections):
        for td in true_drifts:
            if td not in matched and abs(d - td) <= tolerance and d >= td:
                tp += 1
                delays.append(d - td)
                matched.add(td)
                break
    fp = len(detections) - tp
    fn = len(true_drifts) - len(matched)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return dict(precision=precision, recall=recall,
                mean_delay=np.mean(delays) if delays else np.nan,
                true_positives=tp, false_positives=fp, false_negatives=fn)


def job(args):
    ds_name, det_name, seed = args
    dataset = load_dataset(ds_name)
    true_drifts = list(getattr(dataset, "drifts", []) or [])
    detector = make_detector(det_name, seed)
    detections, n = [], 0
    t0 = time.perf_counter()
    for i, (x, _) in enumerate(dataset):
        if i >= MAX_SAMPLES:
            break
        detector.t = i
        if detector.update(x):
            detections.append(i)
        n += 1
    elapsed = time.perf_counter() - t0
    row = dict(dataset=ds_name, detector=det_name, seed=seed, n_samples=n,
               n_detections=len(detections), n_true_drifts=len(true_drifts),
               runtime_sec=round(elapsed, 2))
    if true_drifts:
        row.update(compute_detection_metrics(detections, [d for d in true_drifts if d < n]))
    steps = [dict(dataset=ds_name, seed=seed, **s) for s in getattr(detector, "steps", [])]
    return row, steps


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    workers = max(1, (os.cpu_count() or 2) - 1)
    detectors = ["EADD", "EADD-NoASPT", "D3", "D3-LightGBM"]
    # slowest jobs first so the pool stays busy
    jobs = [(d, det, s) for det in detectors for d in DATASETS for s in SEEDS]
    jobs.sort(key=lambda j: (j[1] in ("D3", "D3-LightGBM"), j[0] not in (
        "InsectsIncrementalAbruptBalanced", "InsectsIncrementalReoccurringBalanced",
        "InsectsIncrementalBalanced", "InsectsAbruptBalanced", "RialtoBridgeTimelapse")))
    rows, steps = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row, st in pool.map(job, jobs):
            rows.append(row)
            steps.extend(st)
            print(f"  {row['dataset']:<40} {row['detector']:<12} seed={row['seed']:<4} "
                  f"dets={row['n_detections']:<4} {row['runtime_sec']:>7.1f}s", flush=True)
    df = pd.DataFrame(rows).sort_values(["dataset", "detector", "seed"])
    df.to_csv(os.path.join(OUT_DIR, "detections.csv"), index=False)
    pd.DataFrame(steps).to_csv(os.path.join(OUT_DIR, "eadd_steps.csv"), index=False)
    print(f"\nAll results saved to {OUT_DIR}", flush=True)
