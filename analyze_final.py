# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Tests of addendum 5 (experiments/results_confirm/PREREGISTRATION.md), exactly as registered.
Output in experiments/results_final/: results.csv (all hypotheses), h2a.csv, per_stream.csv, per_type.csv
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from analyze_benchmark import holm, tolerance
from analyze_hypotheses import detection_metrics

OUT_DIR = os.path.join("experiments", "results_final")
MARGIN_H1C = 50
TOST = {("drift", "recall"): 0.05, ("drift", "mtd"): 50, ("drift", "fa"): 0.5, ("no drift", "fa_per_10k"): 1.0}


def load(name):
    """name 'runs' or 'pvalues': all shards (runs.0.jsonl, runs.1.jsonl, ...)"""
    import glob
    return pd.DataFrame([json.loads(l) for f in sorted(glob.glob(os.path.join(OUT_DIR, f"{name}.*.jsonl")))
                         for l in open(f) if l.strip()])


def per_stream(df):
    rows = []
    for r in df.itertuples():
        row = dict(group=r.group, dataset=r.dataset, variant=r.variant)
        if r.true_drifts:
            m = detection_metrics(r.detections, r.true_drifts, tolerance(r.dataset))
            row.update(recall=m["recall"], mtd=m["mtd"], fa=m["fa"])
        else:
            row.update(fa_per_10k=len(r.detections) / r.n_samples * 1e4)
        if getattr(r, "n_tests", 0) and r.n_tests > 0:
            row["perms_per_test"] = r.n_permutations_used / r.n_tests
        rows.append(row)
    return pd.DataFrame(rows).groupby(["group", "dataset", "variant"]).mean(numeric_only=True).reset_index()


def paired(ps, groups, metric, a, b):
    t = ps[ps.group.isin(groups)].pivot(index="dataset", columns="variant", values=metric)
    return (t[a] - t[b]).dropna(), t[a].mean(), t[b].mean()


def wil(d, alternative):
    return stats.wilcoxon(d, alternative=alternative).pvalue if (d != 0).any() else 1.0


def h4(df):
    hits = []
    for r in df[df.variant == "EADD"].itertuples():
        if not r.true_drifts or not r.shifted:
            continue
        true, k, seen = set(r.shifted), len(r.shifted), set()
        for at, ranking in zip(r.detections, r.rankings):
            d = [x for x in r.true_drifts if x <= at <= x + tolerance(r.dataset)]
            if not d or d[0] in seen:
                continue
            seen.add(d[0])
            hits.append(dict(dataset=r.dataset, precision=len(true & set(ranking[:k])) / k,
                             top1=ranking[0] in true, chance=k / r.n_features))
    return pd.DataFrame(hits).groupby("dataset").mean()


def main():
    df = load("runs")
    ps = per_stream(df)
    ps.round(4).to_csv(os.path.join(OUT_DIR, "per_stream.csv"), index=False)
    ps.groupby(["group", "variant"]).mean(numeric_only=True).round(3).to_csv(os.path.join(OUT_DIR, "per_type.csv"))
    rows = []
    for other in ("D3-LightGBM", "D3"):
        fam = []
        for test, grp, metric, alt, shift in [("H1a slow-drift recall higher", ["slow"], "recall", "greater", 0),
                                              ("H1b slow-drift MTD lower", ["slow"], "mtd", "less", 0),
                                              (f"H1c abrupt MTD not worse (margin {MARGIN_H1C})", ["abrupt"], "mtd", "less", MARGIN_H1C),
                                              ("H2 fewer false alarms on autocorrelated streams without drift", ["autocorrelated"], "fa_per_10k", "less", 0)]:
            d, ma, mb = paired(ps, grp, metric, "EADD", other)
            fam.append(dict(hypothesis=test, against=other, n=len(d), eadd=ma, other=mb, p=wil(d - shift, alt)))
        f = pd.DataFrame(fam)
        f["p_holm"] = holm(f.p)
        f["role"] = "primary" if other == "D3-LightGBM" else "secondary"
        rows.append(f)
    # H3a: equivalence (TOST) of ASPT and the full test; H3b: fewer permutations per test
    for (grp, metric), m in TOST.items():
        groups = ["slow", "abrupt"] if grp == "drift" else ["no drift"]
        d, ma, mb = paired(ps, groups, metric, "EADD", "EADD, full test")
        p = max(wil(d - m, "less"), wil(d + m, "greater"))
        rows.append(pd.DataFrame([dict(hypothesis=f"H3a equivalent {metric} on {grp} streams (margin {m})", against="EADD, full test",
                                       n=len(d), eadd=ma, other=mb, p=p, role="primary")]))
    d, ma, mb = paired(ps, ["slow", "abrupt", "no drift"], "perms_per_test", "EADD", "EADD, full test")
    rows.append(pd.DataFrame([dict(hypothesis="H3b fewer permutations per test", against="EADD, full test", n=len(d),
                                   eadd=ma, other=mb, p=wil(d, "less"), role="primary")]))
    h = h4(df)
    h.round(3).to_csv(os.path.join(OUT_DIR, "h4_per_stream.csv"))
    rows.append(pd.DataFrame([dict(hypothesis=f"H4 precision@k above chance (top-1 {h.top1.mean():.1%})", against="chance",
                                   n=len(h), eadd=h.precision.mean(), other=h.chance.mean(),
                                   p=wil(h.precision - h.chance, "greater"), role="primary")]))
    res = pd.concat(rows, ignore_index=True)
    res["supported"] = res.p_holm.fillna(res.p) < 0.05
    pv = load("pvalues")
    if len(pv):
        a = []
        for (s, c), g in pv.groupby(["source", "config"]):
            k, n = int((g.p <= 0.05).sum()), len(g)
            upper = stats.beta.ppf(0.95, k + 1, n - k)
            a.append(dict(source=s, config=c, n=n, reject_rate=k / n, upper95=upper, passes=upper <= 0.075))
        a = pd.DataFrame(a)
        a.round(4).to_csv(os.path.join(OUT_DIR, "h2a.csv"), index=False)
        print(a.round(4).to_string(index=False))
        res = pd.concat([res, pd.DataFrame([dict(hypothesis=f"test validity (descriptive): {int(a.passes.sum())} of {len(a)} conditions pass",
                                                 n=len(a), role="descriptive")])], ignore_index=True)
    res.round(4).to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
