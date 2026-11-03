# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Thesis Experiments by Hypothesis (Thesis Review, 28 Oct 2026)
=============================================================
All detectors are run on real-world streams from the Lukats et al. (2024)
benchmark with matched windows (500 reference / 200 current samples where
the detector has such a parameter) and several seeds. No synthetic data.

  annotated    4 INSECTS streams with known drift points, 5 seeds
               EADD, D3-LightGBM and six Lukats detectors   -> H1, H2
               EADD with ASPT, a full permutation test or no
               test, at tau = 0.3 ... 0.7                     -> H3
  unannotated  9 streams without drift points, 3 seeds, scored like
               Lukats et al.: prequential accuracy of a Hoeffding tree
               that is reset at every detection, and lift-per-drift
                                                              -> Experiment 4

Only detection positions and run statistics are stored, so the metrics can be
recomputed by analyze_hypotheses.py without rerunning the detectors.

Outputs:
  experiments/results_hypotheses/runs.jsonl
"""
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datasets as ds
from detectors import (
    BayesianNonparametricDetectionMethod,
    ClusteredStatisticalTestDriftDetectionMethod,
    ImageBasedDriftDetector,
    OneClassDriftDetector,
    SemiParametricLogLikelihood,
    UDetect,
)
from detectors.d3_lightgbm import DiscriminativeDriftDetectorLightGBM
from detectors.eadd import ExplainableAdversarialDriftDetector

MAX_SAMPLES = 100_000
N_TRAINING_SAMPLES = 1000  # base classifier of lift-per-drift, as in Lukats et al.
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "experiments", "results_hypotheses")
OUT_FILE = os.path.join(OUT_DIR, "runs.jsonl")

ANNOTATED = ["InsectsAbruptBalanced", "InsectsGradualBalanced",
             "InsectsIncrementalAbruptBalanced", "InsectsIncrementalReoccurringBalanced"]
UNANNOTATED = ["Electricity", "InsectsIncrementalBalanced", "Luxembourg", "NOAAWeather",
               "OutdoorObjects", "Ozone", "PokerHand", "Powersupply", "RialtoBridgeTimelapse"]
SEEDS_ANNOTATED = [42, 142, 242, 342, 442]
SEEDS_UNANNOTATED = [42, 142, 242]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def make_detector(name, seed, tau=0.7, test="aspt", freq=50):
    if name == "EADD":
        return ExplainableAdversarialDriftDetector(
            n_reference_samples=500, n_current_samples=200, auc_threshold=tau,
            n_permutations=199, significance_level=0.05, monitoring_frequency=freq,
            permutation_test=test, seed=seed)
    if name == "D3-LightGBM":
        return DiscriminativeDriftDetectorLightGBM(
            n_reference_samples=500, recent_samples_proportion=0.4, threshold=tau, seed=seed)
    if name == "BNDM":
        return BayesianNonparametricDetectionMethod(
            n_samples=500, const=1.0, threshold=0.5, max_depth=3, seed=seed)
    if name == "CSDDM":
        return ClusteredStatisticalTestDriftDetectionMethod(
            n_samples=500, n_clusters=2, confidence=0.01, feature_proportion=0.1, seed=seed)
    if name == "IBDD":
        return ImageBasedDriftDetector(n_samples=500, seed=seed)
    if name == "OCDD":
        return OneClassDriftDetector(
            n_samples=500, threshold=0.3,
            outlier_detector_kwargs={"nu": 0.5, "kernel": "rbf", "gamma": "auto"}, seed=seed)
    if name == "SPLL":
        return SemiParametricLogLikelihood(n_samples=500, n_clusters=2, threshold=0.05, seed=seed)
    if name == "UDetect":
        return UDetect(n_windows=25, n_samples=500, seed=seed)
    raise ValueError(name)


def job_key(job):
    return json.dumps(job, sort_keys=True)


def run_job(job):
    stream = getattr(ds, job["dataset"])()
    detector = None if job["detector"] == "NoDetection" else make_detector(
        job["detector"], job["seed"], job.get("tau", 0.7), job.get("test", "aspt"),
        job.get("freq", 50))
    with_classifier = job["part"] == "unannotated"
    if with_classifier:
        from river.tree import HoeffdingTreeClassifier
        base, assisted = HoeffdingTreeClassifier(), HoeffdingTreeClassifier()
        correct_base = correct_assisted = n_pred = 0
    detections, n = [], 0
    detector_time = 0.0
    t0 = time.perf_counter()
    for i, (x, y) in enumerate(stream):
        if i >= MAX_SAMPLES:
            break
        if with_classifier and i != 0:
            correct_base += base.predict_one(x) == y
            correct_assisted += assisted.predict_one(x) == y
            n_pred += 1
        if detector is not None:
            t1 = time.perf_counter()
            drift = detector.update(x)
            detector_time += time.perf_counter() - t1
            if drift:
                detections.append(i)
                if with_classifier:
                    assisted = HoeffdingTreeClassifier()
        if with_classifier:
            if i < N_TRAINING_SAMPLES:
                base.learn_one(x, y)
            assisted.learn_one(x, y)
        n += 1
    result = dict(job, key=job_key(job), n_samples=n, detections=detections,
                  true_drifts=[d for d in (getattr(stream, "drifts", None) or []) if d < n],
                  detector_time_sec=round(detector_time, 3),
                  total_time_sec=round(time.perf_counter() - t0, 3))
    if with_classifier:
        result.update(acc_base=correct_base / n_pred, acc_assisted=correct_assisted / n_pred)
    for attr in ("n_tests", "n_early_stops", "n_rejections", "n_permutations_used"):
        if hasattr(detector, attr):
            result[attr] = getattr(detector, attr)
    return result


def all_jobs():
    jobs = []
    lukats = ["BNDM", "CSDDM", "IBDD", "OCDD", "SPLL", "UDetect"]
    for d in ANNOTATED:
        for s in SEEDS_ANNOTATED:
            for det in ["D3-LightGBM"] + lukats:
                jobs.append(dict(part="annotated", dataset=d, detector=det, seed=s))
            for tau in THRESHOLDS:
                for test in ("aspt", "none"):
                    jobs.append(dict(part="annotated", dataset=d, detector="EADD", seed=s,
                                     tau=tau, test=test))
            jobs.append(dict(part="annotated", dataset=d, detector="EADD", seed=s, tau=0.7, test="full"))
            # EADD checking every 200 samples, the interval at which D3 checks
            jobs.append(dict(part="annotated", dataset=d, detector="EADD", seed=s, tau=0.7,
                             test="aspt", freq=200))
    for d in UNANNOTATED:
        jobs.append(dict(part="unannotated", dataset=d, detector="NoDetection", seed=0))
        for s in SEEDS_UNANNOTATED:
            for det in ["EADD", "D3-LightGBM"] + lukats:
                jobs.append(dict(part="unannotated", dataset=d, detector=det, seed=s))
    return jobs


def expected_cost(job):
    """Rough relative cost, so that the slowest jobs start first."""
    if job["detector"] == "EADD":
        cost = {"full": 6, "aspt": 3 - 3 * (job.get("tau", 0.7) - 0.3)}.get(job.get("test"), 1)
    else:
        cost = {"BNDM": 4, "SPLL": 1.5, "CSDDM": 0.8}.get(job["detector"], 0.2)
    return -cost


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    done = set()
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE) as f:
            done = {json.loads(line)["key"] for line in f if line.strip()}
    jobs = [j for j in all_jobs() if job_key(j) not in done]
    jobs.sort(key=expected_cost)
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"{len(jobs)} jobs to run ({len(done)} done) on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, open(OUT_FILE, "a") as out:
        futures = {pool.submit(run_job, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                r = fut.result()
            except Exception as e:  # e.g. a dataset file that cannot be read yet
                j = futures[fut]
                print(f"[{k}/{len(jobs)}] FAILED {j['dataset']} {j['detector']}: {e!r}", flush=True)
                continue
            out.write(json.dumps(r) + "\n")
            out.flush()
            print(f"[{k}/{len(jobs)}] {r['part']:<11} {r['dataset']:<38} {r['detector']:<12} "
                  f"tau={r.get('tau', '-'):<4} test={r.get('test', '-'):<4} seed={r['seed']:<4} "
                  f"dets={len(r['detections']):<4} {r['total_time_sec']:>8.1f}s", flush=True)
    print(f"Results in {OUT_FILE}", flush=True)
