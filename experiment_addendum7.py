# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Addendum 7 of experiments/results_confirm/PREREGISTRATION.md (secondary checks).
  7a: univariate KS / standardised-mean-difference rankings beside TreeSHAP at each EADD alarm (21 drift streams)
  7b: EADD with the label-shuffle null and EADD with a sliding reference + history null (6 AR streams)
  7c: ASPT vs full test wall-clock time with the history null (run alone: --timing)

  python experiment_addendum7.py            # 7a + 7b, parallel
  python experiment_addendum7.py --timing   # 7c, sequential, nothing else running
  python analyze_addendum7.py
"""
import os
import sys

import numpy as np
from scipy.stats import ks_2samp

from detectors.eadd import ExplainableAdversarialDriftDetector as EADD
from experiment_confirm import ABRUPT, SLOW
from experiment_final import DETECTORS, EADD_KW, offset
from experiment_h2 import F_STREAMS
from experiment_hypotheses import run_job
from experiment_validity import FIXED, run

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "results_addendum7")
UNI = []  # univariate rankings at the alarms of the current job (one job at a time per worker)
_shap = EADD._compute_shap


def _compute_shap_and_univariate(self, ref, cur):
    names = np.array([int(str(n).lstrip("x")) for n in self.last_feature_names])
    ks = [ks_2samp(ref[:, j], cur[:, j]).statistic for j in range(ref.shape[1])]
    smd = np.abs(cur.mean(0) - ref.mean(0)) / (ref.std(0) + 1e-12)
    # stable sort: ties keep feature order, so a constant feature never wins a tie by chance
    UNI.append(dict(ks=names[np.argsort(-np.array(ks), kind="stable")].tolist(),
                    smd=names[np.argsort(-smd, kind="stable")].tolist()))
    return _shap(self, ref, cur)  # no random numbers used above: detections are unchanged


EADD._compute_shap = _compute_shap_and_univariate

VARIANTS_7B = {
    "EADD, label-shuffle null": dict(detector="EADD", tau=0.7, test="aspt", eadd_kwargs=FIXED),
    "EADD, sliding + history null": dict(detector="EADD", tau=0.7, test="aspt", eadd_kwargs=dict(
        EADD_KW, use_reservoir_sampling=False, sliding_reference=True)),
}


def jobs():
    a = [dict(part="annotated", group=g, dataset=s, seed=seed, variant="EADD", offset=offset(seed), **DETECTORS["EADD"])
         for g, streams in (("slow", SLOW), ("abrupt", ABRUPT)) for s in streams for seed in range(3000, 3010)]
    b = [dict(part="annotated", group="autocorrelated", dataset=s, seed=seed, variant=v, offset=offset(seed), **f)
         for v, f in VARIANTS_7B.items() for s in F_STREAMS for seed in range(3000, 3005)]
    return b + a  # long AR runs first


def job(j):
    UNI.clear()
    r = run_job(j)
    if j.get("record_shap"):
        r["univariate"] = list(UNI)
    return r


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if "--timing" in sys.argv:
        EADD._compute_shap = _shap  # timing: the plain detector, no extra rankings
        import experiment_aspt_runtime as rt
        from experiment_hypotheses import ANNOTATED
        import pandas as pd
        rt.FIXED_KW.update(EADD_KW)
        rows, k = [], 0
        for dataset in ANNOTATED:
            for seed in (42, 142):
                for test in (("aspt", "full") if k % 2 == 0 else ("full", "aspt")):
                    rows.append(rt.time_run(dataset, seed, test))
                    pd.DataFrame(rows).to_csv(os.path.join(OUT, "timing.csv"), index=False)
                    print(dataset, seed, test, rows[-1]["detector_time_sec"], flush=True)
                k += 1
    else:
        run(jobs(), job, os.path.join(OUT, "runs.jsonl"))
