# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Summaries of experiment_validity.py.

  pvalues.csv        per source and config: P(p <= 0.05) with a 95% Wilson
                     interval (a valid test stays at or below 0.05), mean AUC
  false_alarms.csv   per detector and no-drift stream: alarms per 10,000 samples
"""
import json
import os

import numpy as np
import pandas as pd

OUT_DIR = os.path.join("experiments", "results_validity")


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def load(name):
    path = os.path.join(OUT_DIR, name)
    return pd.DataFrame([json.loads(l) for l in open(path) if l.strip()]) if os.path.exists(path) else None


def main():
    p = load("pvalues.jsonl")
    if p is not None:
        rows = []
        for (s, c), g in p.groupby(["source", "config"]):
            k, n = int((g.p <= 0.05).sum()), len(g)
            lo, hi = wilson(k, n)
            rows.append(dict(source=s, config=c, n=n, reject_rate=k / n, ci_low=lo, ci_high=hi,
                             mean_auc=g.auc.mean(), valid=lo <= 0.05))
        t = pd.DataFrame(rows)
        t.to_csv(os.path.join(OUT_DIR, "pvalues.csv"), index=False)
        print(t.round(3).to_string(index=False))
    r = load("runs.jsonl")
    if r is not None:
        r["fa_per_10k"] = r.detections.str.len() / r.n_samples * 1e4
        t = r.pivot_table(index="variant", columns="dataset", values="fa_per_10k", aggfunc="mean")
        t["mean"] = t.mean(axis=1)
        t.sort_values("mean").to_csv(os.path.join(OUT_DIR, "false_alarms.csv"))
        print(t.sort_values("mean").round(1).to_string())


if __name__ == "__main__":
    main()
