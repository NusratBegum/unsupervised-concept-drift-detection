# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Why EADD Has Lower Recall and More False Alarms than D3-LightGBM
================================================================
EADD differs from D3-LightGBM in its reference window (reservoir sample vs.
the 500 samples just before the current window), its check interval (50 vs.
200 samples), its cross-validation (5 vs. 2 folds) and its permutation test.
This script changes one design choice at a time on the annotated streams:

  ladder        start from a D3-like EADD and add EADD's choices one by one
  leave-one-out remove one choice from EADD

Runs already in runs.jsonl (EADD, EADD without a test, EADD checking every
200 samples, D3-LightGBM) are reused.

Outputs:
  experiments/results_hypotheses/runs_root_cause.jsonl
  experiments/results_hypotheses/root_cause_table.csv
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from analyze_hypotheses import detection_metrics, one_sided
from experiment_hypotheses import ANNOTATED, OUT_DIR, SEEDS_ANNOTATED, job_key, run_job

OUT_FILE = os.path.join(OUT_DIR, "runs_root_cause.jsonl")
D3_LIKE = dict(use_reservoir_sampling=False, sliding_reference=True, n_folds=2)
NEW = {  # name: (freq, test, EADD keyword arguments)
    "L1 D3-like (sliding, every 200, 2-fold, no test)": (200, "none", D3_LIKE),
    "L2 + 5-fold": (200, "none", {**D3_LIKE, "n_folds": 5}),
    "L3 + reservoir": (200, "none", {}),
    "O1 EADD with sliding reference": (50, "aspt", {"use_reservoir_sampling": False, "sliding_reference": True}),
    "O2 EADD with 2-fold": (50, "aspt", {"n_folds": 2}),
}
REUSED = {  # name: filter on runs.jsonl
    "D3-LightGBM": dict(detector="D3-LightGBM"),
    "L4 + every 50 (EADD, no test)": dict(detector="EADD", test="none", freq=50),
    "L5 + ASPT (EADD)": dict(detector="EADD", test="aspt", freq=50),
    "O3 EADD every 200": dict(detector="EADD", test="aspt", freq=200),
}


def jobs():
    return [dict(part="annotated", dataset=d, detector="EADD", seed=s, tau=0.7, test=test, freq=freq,
                 eadd_kwargs=kwargs, variant=name)
            for name, (freq, test, kwargs) in NEW.items() for d in ANNOTATED for s in SEEDS_ANNOTATED]


def run_all():
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
            print(f"[{k}/{len(todo)}] {r['variant']:<50} {r['dataset']:<38} seed={r['seed']} dets={len(r['detections'])}",
                  flush=True)


def runs():
    with open(os.path.join(OUT_DIR, "runs.jsonl")) as f:
        base = pd.DataFrame([json.loads(line) for line in f if line.strip()])
    base = base[(base.part == "annotated") & (base.get("tau").fillna(0.7) == 0.7)]
    base["test"] = base.test.fillna("-")
    base["freq"] = base.freq.fillna(50)
    rows = []
    for name, cond in REUSED.items():
        sel = base
        for col, val in cond.items():
            sel = sel[sel[col] == val]
        rows.append(sel.assign(variant=name))
    with open(OUT_FILE) as f:
        rows.append(pd.DataFrame([json.loads(line) for line in f if line.strip()]))
    df = pd.concat(rows, ignore_index=True)
    m = df.apply(lambda r: pd.Series(detection_metrics(r.detections, r.true_drifts)), axis=1)
    return pd.concat([df[["variant", "dataset", "seed"]], m], axis=1)


def table(df):
    d3 = df[df.variant == "D3-LightGBM"].set_index(["dataset", "seed"])
    out = []
    for name, g in df.groupby("variant", sort=False):
        g = g.set_index(["dataset", "seed"]).loc[d3.index]
        row = dict(variant=name, runs=len(g), recall=g.recall.mean(), mtd=g.mtd.mean(), fa=g.fa.mean(),
                   precision=g.n_matched.sum() / max(g.n_detections.sum(), 1))
        if name != "D3-LightGBM":
            row["p_recall_greater"] = one_sided(g.recall, d3.recall, "greater")[1]
            row["p_mtd_less"] = one_sided(g.mtd, d3.mtd, "less")[1]
            row["p_fa_less"] = one_sided(g.fa, d3.fa, "less")[1]
        out.append(row)
    return pd.DataFrame(out).sort_values("variant")


if __name__ == "__main__":
    run_all()
    t = table(runs())
    t.to_csv(os.path.join(OUT_DIR, "root_cause_table.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t.round(3).to_string(index=False))
