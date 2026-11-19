# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""Analysis of addendum 7 (experiments/results_addendum7/): h4_univariate.csv, h2_source.csv, timing_summary.csv.
  python analyze_addendum7.py"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from analyze_benchmark import holm, tolerance
from analyze_final import load as load_final, per_stream

D = os.path.join("experiments", "results_addendum7")


def h4(runs):
    hits = []
    for r in runs.itertuples():
        if not r.true_drifts or not r.shifted:
            continue
        true, k, seen = set(r.shifted), len(r.shifted), set()
        for at, shap, uni in zip(r.detections, r.rankings, r.univariate):
            d = [x for x in r.true_drifts if x <= at <= x + tolerance(r.dataset)]
            if not d or d[0] in seen:
                continue
            seen.add(d[0])
            hits.append(dict(dataset=r.dataset, treeshap=len(true & set(shap[:k])) / k,
                             ks=len(true & set(uni["ks"][:k])) / k, smd=len(true & set(uni["smd"][:k])) / k,
                             chance=k / r.n_features))
    per = pd.DataFrame(hits).groupby("dataset").mean()
    p = [stats.wilcoxon(per.treeshap, per[c], zero_method="zsplit").pvalue for c in ("ks", "smd")]
    tests = pd.DataFrame(dict(against=["KS", "SMD"], treeshap=per.treeshap.mean(), other=[per.ks.mean(), per.smd.mean()],
                              wins=[(per.treeshap > per[c]).sum() for c in ("ks", "smd")],
                              ties=[(per.treeshap == per[c]).sum() for c in ("ks", "smd")],
                              p_two_sided=p, p_holm=holm(p), n=len(per)))
    return per, tests


def main():
    runs = pd.DataFrame([json.loads(l) for l in open(os.path.join(D, "runs.jsonl"))])
    a = runs[runs.variant == "EADD"]
    # detections must equal those of the final test (same seeds; rankings use no random numbers)
    fin = load_final("runs")
    fin = fin[fin.variant == "EADD"].set_index(["dataset", "seed"]).detections
    same = sum(fin.loc[(r.dataset, r.seed)] == r.detections for r in a.itertuples())
    print(f"7a: identical detections in {same} of {len(a)} runs")
    per, tests = h4(a)
    per.round(4).to_csv(os.path.join(D, "h4_univariate_per_stream.csv"))
    tests.round(4).to_csv(os.path.join(D, "h4_univariate.csv"), index=False)
    print(tests.round(3).to_string(index=False))

    b = runs[runs.group == "autocorrelated"]
    fb = load_final("runs")
    fb = fb[(fb.group == "autocorrelated") & fb.seed.isin(range(3000, 3005)) & fb.variant.isin(["EADD", "D3-LightGBM"])]
    ps = per_stream(pd.concat([b, fb], ignore_index=True))
    t = ps.pivot(index="dataset", columns="variant", values="fa_per_10k")
    t.round(2).to_csv(os.path.join(D, "h2_source.csv"))
    p = stats.wilcoxon(t["EADD, sliding + history null"], t["EADD, label-shuffle null"], alternative="less",
                       zero_method="zsplit").pvalue
    print(t.round(2).to_string())
    print("mean", t.mean().round(2).to_dict(), f"one-sided p (sliding+history < label-shuffle) = {p:.4f}")
    with open(os.path.join(D, "h2_source_test.txt"), "w") as f:
        f.write(f"{t.mean().round(3).to_dict()}\none-sided Wilcoxon, sliding + history null < label-shuffle null: p = {p:.4f}\n")

    tf = os.path.join(D, "timing.csv")
    if os.path.exists(tf):
        tm = pd.read_csv(tf)
        w = tm.pivot_table(index=["dataset", "seed"], columns="test", values="detector_time_sec")
        s = tm.groupby("test")[["detector_time_sec", "n_tests", "n_permutations_used"]].sum()
        s["perms_per_test"] = s.n_permutations_used / s.n_tests
        s["sec_per_run"] = tm.groupby("test").detector_time_sec.mean()
        det = tm.pivot_table(index=["dataset", "seed"], columns="test", values="detections", aggfunc="first")
        s["identical_detections"] = int((det["aspt"] == det["full"]).sum())
        s["p_aspt_faster"] = stats.wilcoxon(w["aspt"], w["full"], alternative="less").pvalue
        s.round(3).to_csv(os.path.join(D, "timing_summary.csv"))
        print(s.round(2).to_string())


if __name__ == "__main__":
    main()
