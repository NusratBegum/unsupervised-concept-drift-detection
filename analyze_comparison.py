# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Tables and tests for experiment_comparison.py.

  python analyze_comparison.py

Reads experiments/results_comparison/runs.jsonl and writes, next to it:
  summary.csv      recall, MTD, false alarms, precision and runtime per variant and stream type
  per_stream.csv   the same per stream
  tests.csv        one-sided tests of EADD against each D3 inside the same setup
  aspt_tests.csv   group D: ASPT against the full test and against no test
  reproduced.txt   whether the thesis EADD and D3-LightGBM runs were reproduced exactly
Metrics are those of the thesis (analyze_hypotheses.detection_metrics): a
detection matches a drift point if it comes 0-500 samples after it. On the
synthetic gradual and incremental streams the change itself lasts 1,000
samples, so there the window is 0-1,500 (the change plus 500). On the
synthetic stream without drift only false alarms are defined. A missed drift
counts as the window length in the MTD (500, or 1,500 on those two streams);
"delay" is the mean delay of the detected drifts only.
"""
import json
import os

import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from analyze_hypotheses import detection_metrics
from experiment_comparison import OUT_DIR, OUT_FILE
from synthetic_streams import WIDTH

THESIS_RUNS = os.path.join(os.path.dirname(OUT_DIR), "results_hypotheses", "runs.jsonl")


def load():
    df = pd.DataFrame([json.loads(line) for line in open(OUT_FILE) if line.strip()])
    df = df.drop_duplicates("key", keep="last")
    slow = df.dataset.isin(["SynthGradual", "SynthIncremental"])
    m = [detection_metrics(d, t, 500 + WIDTH * s) if t else
         dict(recall=None, mtd=None, fa=len(d), n_detections=len(d), n_matched=0)
         for d, t, s in zip(df.detections, df.true_drifts, slow)]
    m = pd.DataFrame(m)[["recall", "mtd", "fa", "n_detections", "n_matched"]]
    tol = 500 + WIDTH * slow.reset_index(drop=True)
    n = df.true_drifts.map(len).reset_index(drop=True)
    # mean delay of the detected drifts: remove the penalty of the missed ones from the MTD
    m["delay"] = ((m.mtd * n - tol * (n - m.n_matched)) / m.n_matched).where(m.n_matched > 0)
    df = pd.concat([df.reset_index(drop=True), m], axis=1)
    df["kind"] = df.dataset.map(lambda d: "synthetic, no drift" if d == "SynthStable"
                                else "synthetic" if d.startswith("Synth") else "real (INSECTS)")
    return df


def summarise(g):
    return pd.Series(dict(runs=len(g), recall=g.recall.mean(), mtd=g.mtd.mean(), delay=g.delay.mean(), fa=g.fa.mean(),
                          precision=g.n_matched.sum() / max(g.n_detections.sum(), 1),
                          seconds=g.detector_time_sec.mean()))


def tests(df):
    out = []
    for setup in ("S200", "S50"):
        for kind in ("real (INSECTS)", "synthetic"):
            sub = df[df.kind == kind]
            e = sub[sub.variant == f"{setup} EADD"].set_index(["dataset", "seed"]).sort_index()
            for other in ("D3 (logistic regression)", "D3-LightGBM", "EADD, no overlap"):
                o = sub[sub.variant == f"{setup} {other}"].set_index(["dataset", "seed"]).sort_index()
                for metric, alt in (("recall", "greater"), ("mtd", "less"), ("fa", "less")):
                    a, b = e[metric].astype(float), o.loc[e.index, metric].astype(float)
                    p_mw = mannwhitneyu(a, b, alternative=alt).pvalue
                    p_w = wilcoxon(a, b, alternative=alt).pvalue if (a != b).any() else 1.0
                    out.append(dict(setup=setup, streams=kind, eadd=f"{setup} EADD", vs=other, metric=metric,
                                    hypothesis=f"EADD {'>' if alt == 'greater' else '<'} other", n=len(a),
                                    eadd_mean=a.mean(), other_mean=b.mean(), p_mannwhitney=p_mw, p_wilcoxon=p_w))
    return pd.DataFrame(out)


def aspt_tests(df):
    out = []
    for tau in (0.5, 0.6):
        for kind in ("real (INSECTS)", "synthetic"):
            sub = df[df.kind == kind]
            v = {t: sub[sub.variant == f"S200 EADD tau {tau}, {t}"].set_index(["dataset", "seed"]).sort_index()
                 for t in ("aspt", "full", "none")}
            a, f, n = v["aspt"], v["full"].loc[v["aspt"].index], v["none"].loc[v["aspt"].index]
            row = dict(tau=tau, streams=kind, n=len(a),
                       identical_to_full=sum(x == y for x, y in zip(a.detections, f.detections)),
                       seconds_aspt=a.detector_time_sec.mean(), seconds_full=f.detector_time_sec.mean(),
                       aspt_faster_runs=int((a.detector_time_sec < f.detector_time_sec).sum()),
                       fa_aspt=a.fa.mean(), fa_none=n.fa.mean(),
                       p_fa_aspt_less_than_none=wilcoxon(a.fa, n.fa, alternative="less").pvalue
                       if (a.fa != n.fa).any() else 1.0)
            for metric in ("recall", "mtd", "fa"):  # two-sided: does ASPT change the result of the full test?
                x, y = a[metric].astype(float), f[metric].astype(float)
                row[f"p_{metric}_aspt_vs_full"] = wilcoxon(x, y).pvalue if (x != y).any() else 1.0
            out.append(row)
    return pd.DataFrame(out)


def reproduced(df):
    old = {}
    for line in open(THESIS_RUNS):
        r = json.loads(line)
        if (r["part"] == "annotated" and r.get("tau", 0.7) == 0.7 and "freq" not in r and not r.get("eadd_kwargs")
                and r.get("test", "aspt") == "aspt" and r["detector"] in ("EADD", "D3-LightGBM")):
            old[(r["detector"], r["dataset"], r["seed"])] = r["detections"]
    new = df[(df.group == "A") & df.detector.isin(["EADD", "D3-LightGBM"])]
    pairs = [(old[k], d) for k, d in zip(zip(new.detector, new.dataset, new.seed), new.detections) if k in old]
    return sum(a == b for a, b in pairs), len(pairs)


if __name__ == "__main__":
    df = load()
    summary = df.groupby(["group", "variant", "kind"], sort=False).apply(summarise, include_groups=False).reset_index()
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    df.groupby(["group", "variant", "dataset"], sort=False).apply(summarise, include_groups=False).reset_index() \
        .to_csv(os.path.join(OUT_DIR, "per_stream.csv"), index=False)
    t = tests(df)
    t.to_csv(os.path.join(OUT_DIR, "tests.csv"), index=False)
    at = aspt_tests(df)
    at.to_csv(os.path.join(OUT_DIR, "aspt_tests.csv"), index=False)
    same, n = reproduced(df)
    with open(os.path.join(OUT_DIR, "reproduced.txt"), "w") as f:
        f.write(f"{same} of {n} thesis runs (EADD and D3-LightGBM, tau 0.7, annotated streams) reproduced exactly\n")
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 20):
        for kind in summary.kind.unique():
            print(f"\n=== {kind}")
            print(summary[summary.kind == kind].drop(columns="kind").round(3).to_string(index=False))
        print("\n=== tests (EADD vs other, same setup)")
        print(t.round(4).to_string(index=False))
        print("\n=== ASPT vs full test vs no test (group D)")
        print(at.round(4).to_string(index=False))
    print(f"\nreproduced: {same} of {n}")
