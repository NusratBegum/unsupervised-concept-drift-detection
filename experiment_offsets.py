# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Annotated Streams with Different Start Positions, and the Revised EADD
======================================================================
experiment_root_cause.py showed that two implementations of the same D3
detector, whose check grids differ by a few samples, differ by 0.20 in
recall: the grid decides whether a detection falls just before an annotated
drift point (a false alarm and a miss) or just after it (a match). With the
stream always starting at sample 0, all seeds share the same grid.

Here every seed starts the detectors at a different sample (OFFSETS), so the
runs cover different grid positions. The variants were fixed before the runs:
D3-LightGBM, the D3-like EADD of experiment_root_cause.py, EADD with its
default settings, and the revised EADD suggested by the ablation (sliding
reference, a check every 200 samples, 5-fold CV) with ASPT at tau 0.5, 0.6 and
0.7 and without a test at tau 0.5 and 0.6. All variants are reported.

Outputs:
  experiments/results_hypotheses/runs_offsets.jsonl
  experiments/results_hypotheses/offsets_table.csv
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from analyze_hypotheses import detection_metrics
from experiment_hypotheses import ANNOTATED, OUT_DIR, SEEDS_ANNOTATED, job_key, run_job
from experiment_root_cause import table

OUT_FILE = os.path.join(OUT_DIR, "runs_offsets.jsonl")
OFFSETS = dict(zip(SEEDS_ANNOTATED, [0, 40, 80, 120, 160]))  # cover the 200-sample grid of D3
REVISED = dict(use_reservoir_sampling=False, sliding_reference=True)
VARIANTS = {  # name: (detector, tau, freq, test, EADD keyword arguments)
    "D3-LightGBM": ("D3-LightGBM", 0.7, 50, "aspt", {}),
    "D3 in EADD code (sliding, every 200, 2-fold, no test)": ("EADD", 0.7, 200, "none", {**REVISED, "n_folds": 2}),
    "EADD default": ("EADD", 0.7, 50, "aspt", {}),
    "Revised EADD tau 0.7": ("EADD", 0.7, 200, "aspt", REVISED),
    "Revised EADD tau 0.6": ("EADD", 0.6, 200, "aspt", REVISED),
    "Revised EADD tau 0.5": ("EADD", 0.5, 200, "aspt", REVISED),
    "Revised EADD tau 0.6, no test": ("EADD", 0.6, 200, "none", REVISED),
    "Revised EADD tau 0.5, no test": ("EADD", 0.5, 200, "none", REVISED),
}


def jobs():
    return [dict(part="annotated", dataset=d, detector=det, seed=s, tau=tau, freq=freq, test=test,
                 eadd_kwargs=kwargs, offset=OFFSETS[s], variant=name)
            for name, (det, tau, freq, test, kwargs) in VARIANTS.items()
            for d in ANNOTATED for s in SEEDS_ANNOTATED]


if __name__ == "__main__":
    done = set()
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE) as f:
            done = {json.loads(line)["key"] for line in f if line.strip()}
    todo = [j for j in jobs() if job_key(j) not in done]
    print(f"{len(todo)} jobs to run ({len(done)} done)", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool, open(OUT_FILE, "a") as out:
        for k, r in enumerate(pool.map(run_job, todo), 1):
            out.write(json.dumps(r) + "\n")
            out.flush()
            print(f"[{k}/{len(todo)}] {r['variant']:<55} {r['dataset']:<38} seed={r['seed']}", flush=True)
    with open(OUT_FILE) as f:
        df = pd.DataFrame([json.loads(line) for line in f if line.strip()])
    m = df.apply(lambda r: pd.Series(detection_metrics(r.detections, r.true_drifts)), axis=1)
    t = table(pd.concat([df[["variant", "dataset", "seed"]], m], axis=1))
    t.to_csv(os.path.join(OUT_DIR, "offsets_table.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t.round(3).to_string(index=False))
