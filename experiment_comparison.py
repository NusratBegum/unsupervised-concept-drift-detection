# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
EADD vs D3: the authors' settings, an identical setup, and EADD's components
============================================================================
Answers the questions of the 10 Nov 2026 meeting:

  A  each detector with its own published settings: D3 as its authors ran it
     (logistic regression, w = 100, rho = 0.1, tau = 0.7, 2-fold CV, a check
     every 10 samples), D3-LightGBM and EADD as in the thesis;
  B  the same setup for every detector: windows 500/200, tau = 0.7, the same
     check interval and CV folds, the same seeds and stream start; run in two
     setups, D3's (check every 200, 2-fold) and EADD's (every 50, 5-fold);
  C  EADD's components one at a time inside setup B: reference memory
     (sliding / reservoir / reservoir without overlap) and decision rule
     (threshold only / ASPT);
  D  ASPT where it decides: tau = 0.5 and 0.6, ASPT vs full test vs no test;
  E  threshold sweep tau = 0.3-0.7 for D3 and D3-LightGBM in setup B;
  F  setup B again with a different stream start per seed (check-grid check).

Streams: the four annotated INSECTS streams and five synthetic streams
(synthetic_streams.py), five seeds each. Each run is one JSON line in
experiments/results_comparison/runs.jsonl; rerunning skips finished runs.

  taskset -c 0-9 python experiment_comparison.py          # run everything
  python experiment_comparison.py --list                  # list variants
  python experiment_comparison.py --only B                # one group
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiment_hypotheses import ANNOTATED, SEEDS_ANNOTATED, job_key, run_job
from synthetic_streams import KINDS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "results_comparison")
OUT_FILE = os.path.join(OUT_DIR, "runs.jsonl")
SYNTHETIC = ["Synth" + k for k in KINDS]
STREAMS = ANNOTATED + SYNTHETIC
OFFSETS = dict(zip(SEEDS_ANNOTATED, [0, 40, 80, 120, 160]))
SETUPS = {"S200": dict(freq=200, folds=2), "S50": dict(freq=50, folds=5)}  # D3's and EADD's
SLIDING = dict(use_reservoir_sampling=False, sliding_reference=True)
NO_OVERLAP = dict(reservoir_after_current=True)


def d3(det, freq, folds, tau=0.7):
    # windows 500 + 200 as EADD; step = check interval
    return dict(detector=det, tau=tau, eadd_kwargs=dict(
        n_reference_samples=500, recent_samples_proportion=0.4, step=freq, n_folds=folds))


def eadd(freq, folds, test="aspt", tau=0.7, **kw):
    return dict(detector="EADD", tau=tau, test=test, freq=freq, eadd_kwargs=dict(n_folds=folds, **kw))


def variants():
    v = {  # group, name: job fields
        ("A", "D3 (authors)"): dict(detector="D3", tau=0.7),
        ("A", "D3-LightGBM (thesis)"): dict(detector="D3-LightGBM"),
        ("A", "EADD (thesis)"): dict(detector="EADD", tau=0.7, test="aspt"),
    }
    for s, c in SETUPS.items():
        f, k = c["freq"], c["folds"]
        v[("B", f"{s} D3 (logistic regression)")] = d3("D3", f, k)
        v[("B", f"{s} D3-LightGBM")] = d3("D3-LightGBM", f, k)
        v[("B", f"{s} EADD")] = eadd(f, k)
        v[("B", f"{s} EADD, no overlap")] = eadd(f, k, **NO_OVERLAP)
        v[("C", f"{s} sliding, threshold only")] = eadd(f, k, "none", **SLIDING)
        v[("C", f"{s} sliding + ASPT")] = eadd(f, k, **SLIDING)
        v[("C", f"{s} reservoir, threshold only")] = eadd(f, k, "none")
        v[("C", f"{s} reservoir without overlap, threshold only")] = eadd(f, k, "none", **NO_OVERLAP)
    for tau in (0.5, 0.6):
        for test in ("aspt", "full", "none"):
            v[("D", f"S200 EADD tau {tau}, {test}")] = eadd(200, 2, test, tau)
    for tau in (0.3, 0.4, 0.5, 0.6):
        for det, label in (("D3", "D3 (logistic regression)"), ("D3-LightGBM", "D3-LightGBM")):
            v[("E", f"S200 {label} tau {tau}")] = d3(det, 200, 2, tau)
    for s in SETUPS:
        for name in ("D3 (logistic regression)", "D3-LightGBM", "EADD", "EADD, no overlap"):
            v[("F", f"{s} {name}, offsets")] = dict(v[("B", f"{s} {name}")], offset=True)
    return v


def jobs(only=None):
    out = []
    for (group, name), fields in variants().items():
        if only and group not in only:
            continue
        for dataset in (ANNOTATED if group == "F" else STREAMS):
            for seed in SEEDS_ANNOTATED:
                job = dict(part="annotated", dataset=dataset, seed=seed, **fields)
                if job.pop("offset", False):
                    job["offset"] = OFFSETS[seed]
                job["variant"] = name
                job["group"] = group
                out.append(job)
    return out


def cost(job):
    c = {"full": 8, "aspt": 2, "none": 1}.get(job.get("test"), 0.3)
    c *= 4 if job.get("freq", 50) == 50 else 1
    c *= 3 if job.get("tau", 0.7) <= 0.5 else 1
    return -c * (1 if job["dataset"].startswith("Synth") else 3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="groups to run, e.g. A B")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for (g, n), f in variants().items():
            print(g, n, json.dumps(f))
        raise SystemExit
    os.makedirs(OUT_DIR, exist_ok=True)
    done = set()
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE) as f:
            done = {json.loads(line)["key"] for line in f if line.strip()}
    todo = sorted((j for j in jobs(a.only) if job_key(j) not in done), key=cost)
    workers = max(1, (os.process_cpu_count() or 2) - 1)
    print(f"{len(todo)} jobs to run ({len(done)} done) on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, open(OUT_FILE, "a") as out:
        futures = {pool.submit(run_job, j): j for j in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            j = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                print(f"[{k}/{len(todo)}] FAILED {j['variant']} {j['dataset']} {j['seed']}: {e!r}", flush=True)
                continue
            out.write(json.dumps(r) + "\n")
            out.flush()
            print(f"[{k}/{len(todo)}] {r['group']} {r['variant']:<45} {r['dataset']:<38} seed={r['seed']:<4} "
                  f"dets={len(r['detections']):<4} {r['total_time_sec']:>7.1f}s", flush=True)
