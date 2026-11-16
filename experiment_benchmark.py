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
              injected at known points (12 streams); none of these was
              used to design or fix EADD, so they are held-out data
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


def jobs():
    out = []
    for name, f in detectors().items():
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
