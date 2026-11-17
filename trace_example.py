# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Illustration only (not used in any test): every check of EADD, D3-LightGBM and D3
on one stream per drift pattern, with the AUC, EADD's p-value and the alarms.
Settings as in experiment_confirm.DETECTORS. Output: experiments/results_confirm/trace.jsonl

  python trace_example.py
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import detectors.d3 as d3mod
import detectors.d3_lightgbm as d3lmod
from experiment_confirm import DETECTORS, OUT_DIR
from experiment_hypotheses import make_detector
from synthetic_streams import synthetic_stream

STREAMS = ["SynthAbrupt", "SynthGradual", "SynthIncremental", "SynthRecurring", "AR0.9"]
SEED = 1000


def trace(args):
    dataset, name = args
    f = DETECTORS[name]
    det = make_detector(f["detector"], SEED, f["tau"], f.get("test", "aspt"), 50, **f["eadd_kwargs"])
    log, t = [], [0]
    if name == "EADD":
        inner = det._detect_drift

        def wrapped():
            det.last_p_value = None
            alarm = inner()
            log.append(dict(t=t[0], auc=det.last_auc, p=det.last_p_value, alarm=bool(alarm)))
            return alarm
        det._detect_drift = wrapped
    else:
        mod = d3lmod if name == "D3-LightGBM" else d3mod
        score = mod.roc_auc_score

        def rec(y, p):
            a = score(y, p)
            log.append(dict(t=t[0], auc=float(a), alarm=bool(a >= f["tau"])))
            return a
        mod.roc_auc_score = rec
    stream = synthetic_stream(dataset, SEED)
    for i, (x, _) in enumerate(stream):
        t[0] = i
        det.update(x)
    return dict(dataset=dataset, detector=name, seed=SEED, drifts=stream.drifts, checks=log)


if __name__ == "__main__":
    jobs = [(s, d) for s in STREAMS for d in DETECTORS]
    with ProcessPoolExecutor(len(jobs)) as pool, open(os.path.join(OUT_DIR, "trace.jsonl"), "w") as out:
        for r in pool.map(trace, jobs):
            out.write(json.dumps(r) + "\n")
            print(r["dataset"], r["detector"], len(r["checks"]), sum(c["alarm"] for c in r["checks"]), flush=True)
