# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
D3 with a LightGBM Discriminator (Thesis Review, 12 Oct 2026)
==============================================================
EADD uses LightGBM as its discriminator while the D3 baseline uses logistic
regression, so earlier EADD vs D3 comparisons mix two effects. This script
reruns the synthetic experiments with D3-LightGBM (D3 with only the
classifier swapped) so each component can be judged on its own:

  D3 (logistic regression) -> D3-LightGBM -> EADD-NoASPT -> EADD (full)

  Part A: drift patterns, Experiment 5 protocol (n=5,000, 5 seeds).
  Part B: false alarms on stable streams, Experiment 4 protocol
          (n=10,000, 5 runs), with D3's original small windows
          (N_ref=100, N_cur=50) and with windows matched to EADD
          (N_ref=500, N_cur=200).

Outputs:
  experiments/results_d3_lightgbm/drift_patterns.csv
  experiments/results_d3_lightgbm/false_alarms.csv
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detectors.eadd import ExplainableAdversarialDriftDetector
from detectors.d3 import DiscriminativeDriftDetector2019
from detectors.d3_lightgbm import DiscriminativeDriftDetectorLightGBM
from experiment_camera_ready import EADDNoASPT
import experiment4_false_alarms as e4
import experiment5_multidetector_synthetic as e5

SEEDS = [42, 142, 242, 342, 442]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "experiments", "results_d3_lightgbm")


# ──────────────────────────────────────────────────────────────
# Detector factories
# ──────────────────────────────────────────────────────────────
def make_eadd(seed, full=True, reservoir=True):
    cls = ExplainableAdversarialDriftDetector if full else EADDNoASPT
    return cls(n_reference_samples=500, n_current_samples=200, auc_threshold=0.7,
               n_permutations=199, significance_level=0.05, use_reservoir_sampling=reservoir,
               monitoring_frequency=50, seed=seed)


def make_d3(seed, classifier, n_ref, proportion, tau):
    cls = DiscriminativeDriftDetectorLightGBM if classifier == "lightgbm" else DiscriminativeDriftDetector2019
    return cls(n_reference_samples=n_ref, recent_samples_proportion=proportion, threshold=tau, seed=seed)


def build(spec, seed):
    kind = spec["kind"]
    if kind == "eadd":
        return make_eadd(seed, full=True)
    if kind == "eadd_noaspt":
        return make_eadd(seed, full=False, reservoir=spec.get("reservoir", True))
    return make_d3(seed, spec["classifier"], spec["n_ref"], spec["proportion"], spec["tau"])


def feed_stream(detector, X):
    detections = []
    t0 = time.perf_counter()
    for i in range(len(X)):
        feats = {f"F{j}": float(X[i, j]) for j in range(X.shape[1])}
        if detector.update(feats):
            detections.append(i)
    return detections, time.perf_counter() - t0


# ──────────────────────────────────────────────────────────────
# Part A: drift patterns (Experiment 5 protocol)
# ──────────────────────────────────────────────────────────────
DRIFT_GENERATORS = {
    "Abrupt": (e5.generate_abrupt_drift, 1000),
    "Gradual": (e5.generate_gradual_drift, 2000),
    "Incremental": (e5.generate_incremental_drift, 3000),
    "Recurring": (e5.generate_recurring_drift, 1000),
}

# D3 windows as in Experiment 5 (N_ref=300, N_cur=120). The last two rows
# separate window size from reference memory: D3-LightGBM with EADD's
# window sizes, and EADD-NoASPT with a fixed (non-reservoir) reference.
DRIFT_DETECTORS = {
    "EADD": dict(kind="eadd"),
    "EADD-NoASPT": dict(kind="eadd_noaspt"),
    "D3": dict(kind="d3", classifier="logreg", n_ref=300, proportion=0.4, tau=0.7),
    "D3-LightGBM": dict(kind="d3", classifier="lightgbm", n_ref=300, proportion=0.4, tau=0.7),
    "D3-LightGBM matched": dict(kind="d3", classifier="lightgbm", n_ref=500, proportion=0.4, tau=0.7),
    "EADD-NoASPT fixed-ref": dict(kind="eadd_noaspt", reservoir=False),
}


def drift_job(args):
    drift_type, seed, det_name = args
    gen, tolerance = DRIFT_GENERATORS[drift_type]
    X, drift_points = gen(shift=2.0, seed=seed)
    dets, elapsed = feed_stream(build(DRIFT_DETECTORS[det_name], seed), X)
    m = e5.compute_metrics(drift_points, dets, len(X), tolerance)
    return dict(drift_type=drift_type, seed=seed, detector=det_name,
                n_detections=m["n_detections"], mtd=m["mtd"], mdr=m["mdr"],
                false_alarms=m["false_alarms"], time_s=round(elapsed, 3))


# ──────────────────────────────────────────────────────────────
# Part B: false alarms on stable streams (Experiment 4 protocol)
# ──────────────────────────────────────────────────────────────
STABLE_STREAMS = {
    "Gaussian (i.i.d.)": e4.generate_stable_gaussian,
    "Autocorrelated": e4.generate_autocorrelated_stable,
    "Heteroscedastic": e4.generate_heteroscedastic_stable,
    "Correlated": e4.generate_correlated_stable,
}

FA_DETECTORS = {"EADD": dict(kind="eadd"), "EADD-NoASPT": dict(kind="eadd_noaspt")}
for clf_name, clf in [("D3", "logreg"), ("D3-LightGBM", "lightgbm")]:
    # D3's windows in Experiment 4 (N_ref=100, N_cur=50)
    for tau in (0.6, 0.7, 0.8):
        FA_DETECTORS[f"{clf_name} small tau={tau}"] = dict(
            kind="d3", classifier=clf, n_ref=100, proportion=0.5, tau=tau)
    # windows matched to EADD (N_ref=500, N_cur=200)
    FA_DETECTORS[f"{clf_name} matched tau=0.7"] = dict(
        kind="d3", classifier=clf, n_ref=500, proportion=0.4, tau=0.7)


def fa_job(args):
    stream, run, det_name = args
    seed = 42 + run * 100
    X = STABLE_STREAMS[stream](seed=seed)
    dets, elapsed = feed_stream(build(FA_DETECTORS[det_name], seed), X)
    return dict(stream=stream, run=run, detector=det_name,
                false_alarms=len(dets), time_s=round(elapsed, 3))


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    workers = max(1, (os.cpu_count() or 2) - 1)

    print("\n=== Part A: drift patterns (Experiment 5 protocol) ===", flush=True)
    jobs = [(t, s, d) for t in DRIFT_GENERATORS for s in SEEDS for d in DRIFT_DETECTORS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(drift_job, jobs))
    df_a = pd.DataFrame(rows)
    df_a.to_csv(os.path.join(OUT_DIR, "drift_patterns.csv"), index=False)
    summary_a = df_a.groupby(["drift_type", "detector"]).agg(
        detected_seeds=("mdr", lambda s: int((s < 1.0).sum())),
        mean_mtd=("mtd", "mean"),
        mean_false_alarms=("false_alarms", "mean"),
    )
    print(summary_a.round(1).to_string(), flush=True)

    print("\n=== Part B: false alarms on stable streams (Experiment 4 protocol) ===", flush=True)
    jobs = [(st, r, d) for st in STABLE_STREAMS for r in range(5) for d in FA_DETECTORS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(fa_job, jobs))
    df_b = pd.DataFrame(rows)
    df_b.to_csv(os.path.join(OUT_DIR, "false_alarms.csv"), index=False)
    summary_b = df_b.pivot_table(index="detector", columns="stream",
                                 values="false_alarms", aggfunc="mean")
    print(summary_b.round(1).to_string(), flush=True)

    print(f"\nAll results saved to {OUT_DIR}", flush=True)
