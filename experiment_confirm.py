# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Confirmatory test of the revised H1 and H2 (17 Nov 2026).
Design, hypotheses and tests: experiments/results_confirm/PREREGISTRATION.md

  taskset -c 0-9 python experiment_confirm.py        # streams (630 + 240 runs) and p-value tests (8,000)
  python analyze_confirm.py
"""
import os

from experiment_hypotheses import run_job
from experiment_shap import shap_job
from experiment_validity import FIXED, p_job, run
from synthetic_streams import INJECT_BASES

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "results_confirm")
SEEDS = list(range(1000, 1010))
SLOW = ["SynthGradual", "SynthIncremental"] + [f"Inject{k}_{b}" for b in INJECT_BASES for k in ("Gradual", "Incremental")]
ABRUPT = ["SynthAbrupt"] + [f"InjectAbrupt_{b}" for b in INJECT_BASES]
NO_DRIFT = ["SynthStable", "AR0.5", "AR0.9", "ShuffledElectricity", "ShuffledNOAAWeather", "ShuffledPowersupply",
            "ShuffledPokerHand", "ShuffledRialtoBridgeTimelapse"]
D3 = lambda det: dict(detector=det, tau=0.7, eadd_kwargs=dict(
    n_reference_samples=500, recent_samples_proportion=0.4, step=50, n_folds=5))
DETECTORS = {
    "EADD": dict(detector="EADD", tau=0.7, test="aspt", eadd_kwargs=dict(FIXED, null="history", history_stride=10)),
    "D3-LightGBM": D3("D3-LightGBM"),
    "D3": D3("D3"),
}
# H3: the same EADD with the full permutation test
EADD_FULL = dict(DETECTORS["EADD"], test="full")
P_SOURCES = ["iid", "real", "ar0.5", "ar0.9"]
P_CONFIGS = ["history10", "history10_reset"]
P_REPS = range(1000, 2000)


def stream_jobs():
    out = []
    for group, streams in (("slow", SLOW), ("abrupt", ABRUPT), ("no drift", NO_DRIFT)):
        for name, f in DETECTORS.items():
            out += [dict(part="annotated", group=group, dataset=s, seed=seed, variant=name, **f)
                    for s in streams for seed in SEEDS]
    return sorted(out, key=lambda j: j["detector"] != "EADD")  # EADD runs take longest


def full_test_jobs():
    return [dict(part="annotated", group=g, dataset=s, seed=seed, variant="EADD, full test", **EADD_FULL)
            for g, streams in (("slow", SLOW), ("abrupt", ABRUPT), ("no drift", NO_DRIFT))
            for s in streams for seed in SEEDS]


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    run(stream_jobs(), run_job, os.path.join(OUT_DIR, "runs.jsonl"))
    run([dict(source=s, config=c, rep=r) for s in P_SOURCES for c in P_CONFIGS for r in P_REPS],
        p_job, os.path.join(OUT_DIR, "pvalues.jsonl"))
    run([dict(dataset=s, seed=seed, history_stride=10) for s in SLOW + ABRUPT for seed in SEEDS], shap_job,
        os.path.join(OUT_DIR, "shap.jsonl"))  # H4
    run(full_test_jobs(), run_job, os.path.join(OUT_DIR, "runs.jsonl"))  # H3
