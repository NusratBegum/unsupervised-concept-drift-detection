# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Validity checks of EADD's test (16 Nov 2026)
=============================================
Part P (p-values under no drift). One window pair at a time: a reference of
500 samples drawn at random from the 5,000 before the current window (as a
reservoir) and a current window of the last 200 samples. With no drift a
valid test gives p-values that are uniform, so P(p <= 0.05) <= 0.05.
Full permutation test, 99 permutations, tau not applied.

  sources:  iid      Gaussian, 10 features
            real     rows of a real stream in random order (no drift)
            ar0.9    stationary AR(1), rho 0.9: no drift but autocorrelated
  configs:  thesis   permuted AUCs with 2-fold CV, observed with 5-fold
            matched  both 5-fold
            blocked  both 5-fold, blocks of 20 consecutive samples

Part F (false alarms on streams without drift): every detector on
SynthStable, AR(0.5/0.9/0.99) and shuffled real streams, 5 seeds.

Outputs: experiments/results_validity/pvalues.jsonl, runs.jsonl
  taskset -c 0-9 python experiment_validity.py --part P F
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from detectors.eadd import ExplainableAdversarialDriftDetector
from experiment_hypotheses import SEEDS_ANNOTATED, job_key, run_job
from synthetic_streams import ar_stream, shuffled_stream

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "results_validity")
REPS = 300
CONFIGS = {"thesis": dict(perm_n_folds=2), "matched": dict(perm_n_folds=None),
           "blocked": dict(perm_n_folds=None, block_size=20)}
SOURCES = {"iid": ("iid", None), "real": ("real", "InsectsAbruptBalanced"), "ar0.9": ("ar", 0.9)}
NULL_STREAMS = ["SynthStable", "AR0.5", "AR0.9", "AR0.99",
                "ShuffledInsectsAbruptBalanced", "ShuffledElectricity", "ShuffledNOAAWeather",
                "ShuffledPowersupply", "ShuffledOutdoorObjects"]
# EADD with every fix: CV-matched permutations, exact-level ASPT, no window overlap
FIXED = dict(perm_n_folds=None, aspt_alpha_split=1 / 3, reservoir_after_current=True)


def window_pair(source, rep):
    kind, arg = SOURCES[source]
    rng = np.random.default_rng([rep, 11])
    if kind == "iid":
        X = rng.normal(size=(5200, 10))
    elif kind == "real":
        X = shuffled_stream(arg, rep, n=5200).X
    else:
        X = ar_stream(arg, rep, n=5200).X
    t_ref = np.sort(rng.choice(5000, 500, replace=False))
    t_cur = np.arange(5000, 5200)
    return X[t_ref], X[t_cur], t_ref.astype(float), t_cur.astype(float)


def p_job(job):
    ref, cur, t_ref, t_cur = window_pair(job["source"], job["rep"])
    det = ExplainableAdversarialDriftDetector(n_permutations=99, permutation_test="full",
                                              seed=job["rep"], **CONFIGS[job["config"]])
    groups = det._time_blocks(t_ref, t_cur)
    t0 = time.perf_counter()
    auc, _ = det._adversarial_auc_with_preds(ref, cur, groups)
    p = det._aspt_test(ref, cur, auc, groups)
    return dict(job, key=job_key(job), auc=auc, p=p, sec=round(time.perf_counter() - t0, 2))


def detectors():
    dets = {
        "EADD (thesis)": dict(detector="EADD", tau=0.7, test="aspt"),
        "EADD (fixed)": dict(detector="EADD", tau=0.7, test="aspt", eadd_kwargs=FIXED),
        "EADD (fixed, blocked)": dict(detector="EADD", tau=0.7, test="aspt",
                                      eadd_kwargs=dict(FIXED, block_size=20)),
        "D3 (authors)": dict(detector="D3", tau=0.7),
        "D3 (500/200, every 200)": dict(detector="D3", tau=0.7, eadd_kwargs=dict(
            n_reference_samples=500, recent_samples_proportion=0.4, step=200, n_folds=2)),
        "D3-LightGBM (500/200, every 200)": dict(detector="D3-LightGBM", tau=0.7, eadd_kwargs=dict(
            n_reference_samples=500, recent_samples_proportion=0.4, step=200, n_folds=2)),
    }
    for d in ["BNDM", "CSDDM", "IBDD", "OCDD", "SPLL", "UDetect"]:
        dets[d] = dict(detector=d)
    return dets


def f_jobs():
    return [dict(part="annotated", dataset=s, seed=seed, variant=name, **f)
            for name, f in detectors().items() for s in NULL_STREAMS for seed in SEEDS_ANNOTATED]


def run(jobs, fn, out_file):
    done = set()
    if os.path.exists(out_file):
        with open(out_file) as f:
            done = {json.loads(line)["key"] for line in f if line.strip()}
    todo = [j for j in jobs if job_key(j) not in done]
    workers = os.process_cpu_count() or 1  # the parent process only waits
    print(f"{len(todo)} jobs ({len(done)} done) on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, open(out_file, "a") as out:
        futures = {pool.submit(fn, j): j for j in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                r = fut.result()
            except Exception as e:
                print(f"[{k}/{len(todo)}] FAILED {futures[fut]}: {e!r}", flush=True)
                continue
            out.write(json.dumps(r) + "\n")
            out.flush()
            if k % 50 == 0 or k == len(todo):
                print(f"[{k}/{len(todo)}]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="*", default=["P", "F"])
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if "P" in a.part:
        jobs = [dict(source=s, config=c, rep=r) for s in SOURCES for c in CONFIGS for r in range(REPS)
                if not (s != "ar0.9" and c == "blocked")]
        run(jobs, p_job, os.path.join(OUT_DIR, "pvalues.jsonl"))
    if "F" in a.part:
        run(f_jobs(), run_job, os.path.join(OUT_DIR, "runs.jsonl"))
