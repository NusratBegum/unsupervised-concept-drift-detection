# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Online drift detectors from Alibi Detect 0.13 (Cobb et al., AISTATS 2022) on the streams of
the final test (addendum 6 of experiments/results_confirm/PREREGISTRATION.md).

Runs in its own environment (Python 3.12, numpy 1.26; Alibi Detect does not install next to
EADD's pinned packages):  ~/eadd/venv312/bin/python baselines_online.py
Same protocol as the other detectors: the first 500 samples after a start or an alarm are the
reference, the detector is calibrated on them for the given expected run time (ERT), and after
an alarm it restarts with the next 500 samples. Output: experiments/results_final/online.jsonl
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from synthetic_streams import INJECT_BASES, synthetic_stream

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "results_final", "online.jsonl")
SEEDS = range(3000, 3003)  # addendum 6a: compute limit
N_REF, ERT, WINDOW, BOOT = 500, 2_500, 50, 25_000  # bootstraps at least 10 x ERT (Alibi Detect guidance)
# the streams of experiment_confirm and experiment_h2 (not imported: they need EADD's environment)
SLOW = ["SynthGradual", "SynthIncremental"] + [f"Inject{k}_{b}" for b in INJECT_BASES for k in ("Gradual", "Incremental")]
ABRUPT = ["SynthAbrupt"] + [f"InjectAbrupt_{b}" for b in INJECT_BASES]
NO_DRIFT = ["SynthStable", "AR0.5", "AR0.9", "ShuffledElectricity", "ShuffledNOAAWeather", "ShuffledPowersupply",
            "ShuffledPokerHand", "ShuffledRialtoBridgeTimelapse"]
F_STREAMS = ["AR0.8", "AR0.85", "AR0.9", "AR0.93", "AR0.95", "AR0.97"]
GROUPS = (("slow", SLOW), ("abrupt", ABRUPT), ("no drift", NO_DRIFT), ("autocorrelated", F_STREAMS))


def offset(seed):
    return int(np.random.default_rng([seed, 21]).integers(0, 50))


def make(name, x_ref, seed):
    import torch
    from alibi_detect.cd import CVMDriftOnline, LSDDDriftOnline, MMDDriftOnline
    torch.manual_seed(seed)
    np.random.seed(seed)
    if name == "MMD-Online":
        return MMDDriftOnline(x_ref, ERT, WINDOW, backend="pytorch", n_bootstraps=BOOT, verbose=False, device="cpu")
    if name == "LSDD-Online":
        return LSDDDriftOnline(x_ref, ERT, WINDOW, backend="pytorch", n_bootstraps=BOOT, verbose=False, device="cpu")
    if name == "CVM-Online":
        return CVMDriftOnline(x_ref, ERT, window_sizes=[WINDOW], n_bootstraps=BOOT, verbose=False)
    raise ValueError(name)


def job_key(j):
    return json.dumps(j, sort_keys=True)


def run(job):
    import torch
    torch.set_num_threads(1)
    stream = synthetic_stream(job["dataset"], job["seed"])
    X = stream.X.astype(np.float32)
    # standardise with the first reference, as the kernels assume comparable scales
    start, det, ref, detections, t0 = job["offset"], None, [], [], time.perf_counter()
    for i in range(start, len(X)):
        if det is None:
            ref.append(X[i])
            if len(ref) == N_REF:
                det, ref = make(job["variant"], np.array(ref), job["seed"] + i), []
            continue
        if det.predict(X[i], return_test_stat=False)["data"]["is_drift"]:
            detections.append(i)
            det = None
    return dict(job, key=job_key(job), n_samples=len(X), detections=detections,
                true_drifts=[d for d in (stream.drifts or []) if d < len(X)],
                total_time_sec=round(time.perf_counter() - t0, 1))


def jobs():
    return [dict(part="annotated", group=g, dataset=s, seed=seed, variant=v, offset=offset(seed),
                 ert=ERT, window=WINDOW, n_ref=N_REF)
            for v in ("MMD-Online", "LSDD-Online", "CVM-Online") for g, ss in GROUPS for s in ss for seed in SEEDS]


if __name__ == "__main__":
    todo = jobs()
    if os.path.exists(OUT):
        done = {json.loads(l)["key"] for l in open(OUT) if l.strip()}
        todo = [j for j in todo if job_key(j) not in done]
    print(f"{len(todo)} jobs", flush=True)
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 10) as pool, open(OUT, "a") as out:
        for k, f in enumerate(as_completed([pool.submit(run, j) for j in todo]), 1):
            try:
                out.write(json.dumps(f.result()) + "\n"); out.flush()
            except Exception as e:
                print("FAILED", repr(e), flush=True)
            if k % 20 == 0:
                print(f"[{k}/{len(todo)}]", flush=True)
