# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Runtime of ASPT against a Full Permutation Test (H3b)
=====================================================
experiment_hypotheses.py runs many jobs in parallel, so its wall-clock times
depend on the load of the machine. This script times EADD with ASPT, with a
full permutation test and without a test one run at a time, on the four
annotated INSECTS streams with five seeds. The order of the variants
alternates between runs. The detections are stored as well, to confirm that
they match the parallel runs.

Outputs:
  experiments/results_hypotheses/aspt_runtime.csv
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datasets as ds
from experiment_hypotheses import ANNOTATED, OUT_DIR, SEEDS_ANNOTATED, make_detector

VARIANTS = ["aspt", "full", "none"]


def time_run(dataset, seed, test):
    detector = make_detector("EADD", seed, tau=0.7, test=test)
    detections, elapsed = [], 0.0
    for i, (x, _) in enumerate(getattr(ds, dataset)()):
        t0 = time.perf_counter()
        drift = detector.update(x)
        elapsed += time.perf_counter() - t0
        if drift:
            detections.append(i)
    return dict(dataset=dataset, seed=seed, test=test, detector_time_sec=round(elapsed, 3),
                n_tests=detector.n_tests, n_permutations_used=detector.n_permutations_used,
                n_early_stops=detector.n_early_stops, detections=json.dumps(detections))


if __name__ == "__main__":
    # optional arguments dataset:seed rerun only those runs, e.g. when another
    # job shared the machine, and replace them in the existing file
    redo = {(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[1:]}
    out = os.path.join(OUT_DIR, "aspt_runtime.csv")
    rows = []
    if redo:
        rows = [r for r in pd.read_csv(out).to_dict("records") if (r["dataset"], r["seed"]) not in redo]
    k = 0
    for dataset in ANNOTATED:
        for seed in SEEDS_ANNOTATED:
            order = VARIANTS if k % 2 == 0 else VARIANTS[::-1]
            k += 1
            if redo and (dataset, seed) not in redo:
                continue
            for test in order:
                r = time_run(dataset, seed, test)
                rows.append(r)
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"{dataset:<38} seed={seed:<4} {test:<4} {r['detector_time_sec']:>8.1f}s", flush=True)
    print(f"Results in {out}", flush=True)
