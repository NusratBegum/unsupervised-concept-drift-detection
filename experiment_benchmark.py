# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Benchmark after the validity fixes (16 Nov 2026)
================================================
Every detector of experiment_validity.detectors() (EADD as in the thesis,
EADD with the fixes, D3 as its authors ran it, D3 and D3-LightGBM with
EADD's windows, six Lukats detectors) on

  annotated   4 INSECTS streams with known drift points
  synthetic   the five synthetic streams (synthetic_streams.py)
  injected    6 real streams, shuffled, with abrupt or incremental drifts
              injected at known points (12 streams); EADD was not tuned on
              them, but shuffled Electricity, NOAA and Powersupply rows were
              also in the no-drift checks that led to the fixes
  unannotated 9 real streams, scored like Lukats et al. (3 seeds)

5 seeds for streams with drift points. Output: experiments/results_benchmark/runs.jsonl
  taskset -c 0-9 python experiment_benchmark.py
"""
import os

from experiment_hypotheses import ANNOTATED, SEEDS_ANNOTATED, SEEDS_UNANNOTATED, UNANNOTATED
from experiment_hypotheses import run_job
from experiment_validity import detectors, run
from synthetic_streams import INJECT_BASES, KINDS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "results_benchmark")
SYNTHETIC = ["Synth" + k for k in KINDS]
INJECTED = [f"Inject{k}_{b}" for b in INJECT_BASES for k in ("Abrupt", "Incremental")]
GROUPS = {"annotated": ANNOTATED, "synthetic": SYNTHETIC, "injected": INJECTED}


def matched():
    """Same check interval and CV for EADD and D3: the main benchmark checks EADD every 50
    samples (5-fold) but D3 every 200 (2-fold), so delays there are not comparable."""
    from experiment_validity import FIXED
    d3 = lambda det, step, k: dict(detector=det, tau=0.7, eadd_kwargs=dict(
        n_reference_samples=500, recent_samples_proportion=0.4, step=step, n_folds=k))
    return {
        "EADD (fixed, every 200)": dict(detector="EADD", tau=0.7, test="aspt", freq=200,
                                        eadd_kwargs=dict(FIXED, n_folds=2)),
        "D3 (500/200, every 50)": d3("D3", 50, 5),
        "D3-LightGBM (500/200, every 50)": d3("D3-LightGBM", 50, 5),
        # components one at a time, all every 50 samples with 5-fold CV (EADD (fixed) = reservoir + ASPT).
        # With both switched off, EADD's code must behave like D3-LightGBM: a check of the pipeline.
        "EADD parts: sliding, threshold only": dict(detector="EADD", tau=0.7, test="none", eadd_kwargs=dict(
            FIXED, use_reservoir_sampling=False, sliding_reference=True, reservoir_after_current=False)),
        "EADD parts: sliding + ASPT": dict(detector="EADD", tau=0.7, test="aspt", eadd_kwargs=dict(
            FIXED, use_reservoir_sampling=False, sliding_reference=True, reservoir_after_current=False)),
        "EADD parts: reservoir, threshold only": dict(detector="EADD", tau=0.7, test="none", eadd_kwargs=FIXED),
    }


def jobs():
    out = []
    for name, f in {**detectors(), **matched()}.items():
        for group, streams in GROUPS.items():
            out += [dict(part="annotated", group=group, dataset=s, seed=seed, variant=name, **f)
                    for s in streams for seed in SEEDS_ANNOTATED]
        out += [dict(part="unannotated", group="unannotated", dataset=s, seed=seed, variant=name, **f)
                for s in UNANNOTATED for seed in SEEDS_UNANNOTATED]
    # EADD runs first, they take longest
    return sorted(out, key=lambda j: (j["detector"] != "EADD", j["detector"] != "BNDM"))


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    run(jobs(), run_job, os.path.join(OUT_DIR, "runs.jsonl"))
