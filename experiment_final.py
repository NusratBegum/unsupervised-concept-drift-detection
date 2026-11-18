# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Final confirmatory test of H1-H4 with the corrected EADD (addendum 5 of
experiments/results_confirm/PREREGISTRATION.md). Fresh seeds 3000-3009, p-value reps 3000-3999.
Every detector checks every 50 samples and each seed starts at a random offset (as R1).

  taskset -c 0-9 python experiment_final.py --shard 0/2   # one machine: half the stream runs, then half the p-value tests
  python experiment_final.py --shard 1/2                  # the other machine
  python analyze_final.py
"""
import argparse
import os

import numpy as np

from experiment_confirm import ABRUPT, NO_DRIFT, SLOW
from experiment_h2 import F_STREAMS
from experiment_hypotheses import run_job
from experiment_validity import FIXED, p_job, run

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "results_final")
SEEDS = range(3000, 3010)
P_REPS = range(3000, 4000)
EADD_KW = dict(FIXED, null="history", history_stride=10)  # EADD exactly as in the confirmatory test
D3 = lambda det: dict(detector=det, tau=0.7, eadd_kwargs=dict(
    n_reference_samples=500, recent_samples_proportion=0.4, step=49, n_folds=5))  # step 49: a check every 50
DETECTORS = {
    "EADD": dict(detector="EADD", tau=0.7, test="aspt", eadd_kwargs=EADD_KW, record_shap=True),
    "EADD, full test": dict(detector="EADD", tau=0.7, test="full", eadd_kwargs=EADD_KW),
    "D3-LightGBM": D3("D3-LightGBM"),
    "D3": D3("D3"),
}
GROUPS = (("slow", SLOW), ("abrupt", ABRUPT), ("no drift", NO_DRIFT), ("autocorrelated", F_STREAMS))
P_SOURCES = ["iid", "real_elec", "real_poker", "ar0.3", "ar0.5"]
P_CONFIGS = ["history10", "warm31"]  # steady state; first test after an alarm (descriptive)


def offset(seed):
    return int(np.random.default_rng([seed, 21]).integers(0, 50))


def stream_jobs():
    jobs = [dict(part="annotated", group=g, dataset=s, seed=seed, variant=name, offset=offset(seed), **f)
            for name, f in DETECTORS.items() for g, streams in GROUPS for s in streams for seed in SEEDS
            if not (name == "EADD, full test" and g == "autocorrelated")]
    return sorted(jobs, key=lambda j: (j["detector"] != "EADD", j["group"] == "autocorrelated"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="*", default=["S", "P"])
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i (two machines)")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    i, n = map(int, a.shard.split("/"))
    if "S" in a.part:
        run(stream_jobs()[i::n], run_job, os.path.join(OUT_DIR, f"runs.{i}.jsonl"))
    if "P" in a.part:
        run([dict(source=s, config=c, rep=r) for s in P_SOURCES for c in P_CONFIGS for r in P_REPS][i::n],
            p_job, os.path.join(OUT_DIR, f"pvalues.{i}.jsonl"))
