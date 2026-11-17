# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Robustness check R1 (addendum 4 of experiments/results_confirm/PREREGISTRATION.md):
every detector checks every 50 samples and each seed starts at a random offset.

  python experiment_robust.py; python analyze_confirm.py    # robust.csv
"""
import os

import numpy as np

from experiment_confirm import ABRUPT, DETECTORS, OUT_DIR, SEEDS, SLOW
from experiment_hypotheses import run_job
from experiment_validity import run


def offset(seed):
    return int(np.random.default_rng([seed, 21]).integers(0, 50))


def detectors():
    out = {}
    for name, f in DETECTORS.items():
        f = dict(f, eadd_kwargs=dict(f["eadd_kwargs"]))
        if name != "EADD":
            f["eadd_kwargs"]["step"] = 49  # D3 skips the triggering sample: a check every 50
        out[name] = f
    return out


def jobs():
    return sorted([dict(part="annotated", group=g, dataset=s, seed=seed, variant=name, offset=offset(seed), **f)
                   for name, f in detectors().items()
                   for g, streams in (("slow", SLOW), ("abrupt", ABRUPT)) for s in streams for seed in SEEDS],
                  key=lambda j: j["detector"] != "EADD")


if __name__ == "__main__":
    run(jobs(), run_job, os.path.join(OUT_DIR, "robust_runs.jsonl"))
