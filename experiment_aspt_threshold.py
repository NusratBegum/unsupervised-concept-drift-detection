# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Role of the AUC Threshold and ASPT
==================================
With the default threshold (tau = 0.7) and windows of 500/200 samples, ASPT
confirmed every alarm it was asked to test on the real-world streams. This
script lowers tau so that ASPT has to make the decision itself, and compares
EADD with and without ASPT on the synthetic drift streams (Experiment 1
protocol, 5 seeds) and stable streams (Experiment 3 protocol, 5 runs).

Outputs:
  experiments/results_aspt_threshold/runs.csv
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_camera_ready import EADDNoASPT
from experiment_realworld_ablation import LoggingEADD
import experiment4_false_alarms as e4
import experiment5_multidetector_synthetic as e5

SEEDS = [42, 142, 242, 342, 442]
TAUS = [0.5, 0.6, 0.7]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "experiments", "results_aspt_threshold")

DRIFT_GENERATORS = {
    "Abrupt": (e5.generate_abrupt_drift, 1000),
    "Gradual": (e5.generate_gradual_drift, 2000),
    "Incremental": (e5.generate_incremental_drift, 3000),
    "Recurring": (e5.generate_recurring_drift, 1000),
}
STABLE_STREAMS = {
    "Gaussian (i.i.d.)": e4.generate_stable_gaussian,
    "Autocorrelated": e4.generate_autocorrelated_stable,
    "Heteroscedastic": e4.generate_heteroscedastic_stable,
    "Correlated": e4.generate_correlated_stable,
}


def make(variant, tau, seed):
    kwargs = dict(n_reference_samples=500, n_current_samples=200, auc_threshold=tau,
                  n_permutations=199, significance_level=0.05, monitoring_frequency=50, seed=seed)
    return LoggingEADD(**kwargs) if variant == "EADD" else EADDNoASPT(**kwargs)


def run(detector, X):
    detections = []
    t0 = time.perf_counter()
    for i in range(len(X)):
        detector.t = i
        if detector.update({f"F{j}": float(X[i, j]) for j in range(X.shape[1])}):
            detections.append(i)
    return detections, time.perf_counter() - t0


def job(args):
    kind, stream, seed, variant, tau = args
    if kind == "drift":
        gen, tolerance = DRIFT_GENERATORS[stream]
        X, drift_points = gen(shift=2.0, seed=seed)
    else:
        X, drift_points = STABLE_STREAMS[stream](seed=seed), []
    detector = make(variant, tau, seed)
    dets, elapsed = run(detector, X)
    row = dict(kind=kind, stream=stream, seed=seed, variant=variant, tau=tau,
               n_detections=len(dets), time_s=round(elapsed, 2))
    if drift_points:
        m = e5.compute_metrics(drift_points, dets, len(X), tolerance)
        row.update(mtd=m["mtd"], mdr=m["mdr"], false_alarms=m["false_alarms"])
    else:
        row.update(mtd=np.nan, mdr=np.nan, false_alarms=len(dets))
    steps = getattr(detector, "steps", [])
    tested = [s for s in steps if s["aspt_permutations"] > 0]
    row.update(checks=len(steps) if steps else np.nan,
               aspt_invocations=len(tested) if steps else np.nan,
               aspt_rejections=sum(1 for s in tested if not s["drift"]) if steps else np.nan,
               aspt_mean_permutations=np.mean([s["aspt_permutations"] for s in tested]) if tested else np.nan)
    return row


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    workers = max(1, (os.cpu_count() or 2) - 1)
    jobs = [("drift", s, seed, v, tau) for s in DRIFT_GENERATORS for seed in SEEDS
            for v in ("EADD", "EADD-NoASPT") for tau in TAUS]
    jobs += [("stable", s, 42 + r * 100, v, tau) for s in STABLE_STREAMS for r in range(5)
             for v in ("EADD", "EADD-NoASPT") for tau in TAUS]
    jobs.sort(key=lambda j: (j[3] != "EADD", j[4]))  # slowest (EADD, low tau) first
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(job, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "runs.csv"), index=False)
    print(df.groupby(["kind", "variant", "tau"]).agg(
        false_alarms=("false_alarms", "mean"), mdr=("mdr", "mean"), mtd=("mtd", "mean"),
        invocations=("aspt_invocations", "mean"), rejections=("aspt_rejections", "mean"),
        perms=("aspt_mean_permutations", "mean"), time_s=("time_s", "mean")).round(2).to_string())
    print(f"\nAll results saved to {OUT_DIR}", flush=True)
