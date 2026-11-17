# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Does EADD's TreeSHAP ranking name the features that drifted? (17 Nov 2026)

EADD (fixed, history null) on the synthetic and injected-drift streams, where
the drifting features are known (3 of 10 synthetic features; 30% of the real
features). For every detection that matches a drift point (0-500 samples
after it, 0-1,500 for incremental and gradual drift), the ranking is compared
with the true set: precision@k with k = number of drifting features, and
whether the top feature is a drifting one. Chance level is k / d.

Output: experiments/results_shap/runs.jsonl, summary.csv
  taskset -c 0-9 python experiment_shap.py; python experiment_shap.py --summary
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from analyze_benchmark import tolerance
from detectors.eadd import ExplainableAdversarialDriftDetector
from experiment_benchmark import INJECTED
from experiment_hypotheses import SEEDS_ANNOTATED, job_key
from experiment_validity import FIXED, run
from synthetic_streams import synthetic_stream

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "results_shap")
STREAMS = ["SynthAbrupt", "SynthGradual", "SynthIncremental", "SynthRecurring"] + INJECTED


def shap_job(job):
    stream = synthetic_stream(job["dataset"], job["seed"])
    det = ExplainableAdversarialDriftDetector(
        n_reference_samples=500, n_current_samples=200, auc_threshold=0.7, n_permutations=199,
        significance_level=0.05, monitoring_frequency=50, seed=job["seed"], **FIXED, null="history",
        history_stride=job.get("history_stride", 50))  # the 17 Nov SHAP runs used stride 50
    tol = tolerance(job["dataset"])
    out = []
    for i, (x, _) in enumerate(stream):
        if det.update(x):
            hit = [d for d in stream.drifts if d <= i <= d + tol]
            ranking = [int(name[1:]) for name in det.last_shap_importances]  # names are x0, x1, ...
            out.append(dict(at=i, drift=hit[0] if hit else None, ranking=ranking))
    return dict(job, key=job_key(job), shifted=stream.shifted, n_features=stream.X.shape[1], detections=out)


def summary():
    rows = []
    for line in open(os.path.join(OUT_DIR, "runs.jsonl")):
        r = json.loads(line)
        k, true = len(r["shifted"]), set(r["shifted"])
        seen = set()
        for d in r["detections"]:
            if d["drift"] is None or d["at"] - d["drift"] > tolerance(r["dataset"]) or d["drift"] in seen:
                continue  # only the first detection of each drift point
            seen.add(d["drift"])
            rows.append(dict(dataset=r["dataset"], seed=r["seed"], k=k, d=r["n_features"],
                             precision_at_k=len(true & set(d["ranking"][:k])) / k,
                             top1=d["ranking"][0] in true, chance=k / r["n_features"]))
    df = pd.DataFrame(rows)
    s = df.groupby("dataset")[["precision_at_k", "top1", "chance"]].mean()
    s["n_detections"] = df.groupby("dataset").size()
    s.loc["all"] = [df.precision_at_k.mean(), df.top1.mean(), df.chance.mean(), len(df)]
    s.round(3).to_csv(os.path.join(OUT_DIR, "summary.csv"))
    print(s.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.summary:
        summary()
    else:
        run([dict(dataset=s, seed=seed) for s in STREAMS for seed in SEEDS_ANNOTATED],
            shap_job, os.path.join(OUT_DIR, "runs.jsonl"))
