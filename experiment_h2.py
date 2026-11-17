# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Confirmatory test of the new H2 (addendum 3 of experiments/results_confirm/PREREGISTRATION.md).

  python experiment_h2.py --part P      # 10,000 single tests without drift
  python experiment_h2.py --part F      # 180 runs on strongly autocorrelated streams
  python analyze_confirm.py             # h2_new.csv
"""
import argparse
import os

from experiment_confirm import DETECTORS, OUT_DIR
from experiment_hypotheses import run_job
from experiment_validity import p_job, run

P_SOURCES = ["iid", "real_elec", "real_poker", "ar0.3", "ar0.5"]
P_CONFIGS = ["history10", "history10_reset"]
P_REPS = range(2000, 3000)
F_STREAMS = ["AR0.8", "AR0.85", "AR0.9", "AR0.93", "AR0.95", "AR0.97"]
SEEDS = range(2000, 2010)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="*", default=["P", "F"])
    a = ap.parse_args()
    if "F" in a.part:
        run([dict(part="annotated", group="h2 autocorrelated", dataset=s, seed=seed, variant=name, **f)
             for name, f in DETECTORS.items() for s in F_STREAMS for seed in SEEDS],
            run_job, os.path.join(OUT_DIR, "h2_runs.jsonl"))
    if "P" in a.part:
        run([dict(source=s, config=c, rep=r) for s in P_SOURCES for c in P_CONFIGS for r in P_REPS],
            p_job, os.path.join(OUT_DIR, "h2_pvalues.jsonl"))
