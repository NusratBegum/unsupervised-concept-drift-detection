# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Tests of experiments/results_confirm/PREREGISTRATION.md, exactly as registered.

  results.csv     H1a, H1b, H1c, H2b (Holm-corrected) and secondary tests against D3
  h2a.csv         false-rejection rate and one-sided 95% Clopper-Pearson upper bound per condition
  per_type.csv    means per drift group and detector
  h3_h4.csv       H3a, H3b, H4 (addendum); h4_per_stream.csv
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from analyze_benchmark import holm, tolerance
from analyze_hypotheses import detection_metrics

OUT_DIR = os.path.join("experiments", "results_confirm")
MARGIN = 50


def load(name):
    return pd.DataFrame([json.loads(l) for l in open(os.path.join(OUT_DIR, name)) if l.strip()])


def per_stream(df):
    rows = []
    for r in df.itertuples():
        if r.true_drifts:
            m = detection_metrics(r.detections, r.true_drifts, tolerance(r.dataset))
            rows.append(dict(group=r.group, dataset=r.dataset, variant=r.variant, recall=m["recall"],
                             mtd=m["mtd"], fa=m["fa"]))
        else:
            rows.append(dict(group=r.group, dataset=r.dataset, variant=r.variant,
                             fa_per_10k=len(r.detections) / r.n_samples * 1e4))
    return pd.DataFrame(rows).groupby(["group", "dataset", "variant"]).mean(numeric_only=True).reset_index()


def wilcoxon(ps, group, metric, other, alternative, shift=0.0):
    t = ps[ps.group == group].pivot(index="dataset", columns="variant", values=metric)
    d = (t["EADD"] - t[other] - shift).dropna()
    p = stats.wilcoxon(d, alternative=alternative).pvalue if (d != 0).any() else 1.0
    return dict(n_streams=len(d), eadd=t["EADD"].mean(), other_mean=t[other].mean(), p=p)


def main():
    ps = per_stream(load("runs.jsonl"))
    ps.groupby(["group", "variant"]).mean(numeric_only=True).round(3).to_csv(os.path.join(OUT_DIR, "per_type.csv"))
    rows = []
    for other in ("D3-LightGBM", "D3"):
        fam = [
            dict(test="H1a slow-drift recall higher", **wilcoxon(ps, "slow", "recall", other, "greater")),
            dict(test="H1b slow-drift MTD lower", **wilcoxon(ps, "slow", "mtd", other, "less")),
            dict(test=f"H1c abrupt MTD not worse (margin {MARGIN})", **wilcoxon(ps, "abrupt", "mtd", other, "less", MARGIN)),
            dict(test="H2b fewer false alarms without drift", **wilcoxon(ps, "no drift", "fa_per_10k", other, "less")),
        ]
        f = pd.DataFrame(fam)
        f["p_holm"] = holm(f.p)
        f.insert(0, "against", other)
        f.insert(1, "role", "primary" if other == "D3-LightGBM" else "secondary")
        rows.append(f)
    res = pd.concat(rows)
    res["supported"] = res.p_holm < 0.05
    res.round(4).to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)
    print(res.round(4).to_string(index=False))

    if os.path.exists(os.path.join(OUT_DIR, "pvalues.jsonl")):
        pv = load("pvalues.jsonl")
        h = []
        for (s, c), g in pv.groupby(["source", "config"]):
            k, n = int((g.p <= 0.05).sum()), len(g)
            upper = stats.beta.ppf(0.95, k + 1, n - k)  # one-sided 95% Clopper-Pearson upper bound
            h.append(dict(source=s, config=c, n=n, reject_rate=k / n, upper95=upper, passes=upper <= 0.075))
        h = pd.DataFrame(h)
        h.round(4).to_csv(os.path.join(OUT_DIR, "h2a.csv"), index=False)
        print(h.round(4).to_string(index=False))




def h3_h4():
    """H3a, H3b and H4 as in the addendum of PREREGISTRATION.md."""
    df = load("runs.jsonl")
    e = df[df.variant.isin(["EADD", "EADD, full test"])]
    pairs = e.pivot_table(index=["dataset", "seed"], columns="variant", values="detections", aggfunc="first").dropna()
    same = sum(a == b for a, b in zip(pairs["EADD"], pairs["EADD, full test"]))
    e = e.assign(perms_per_test=e.n_permutations_used / e.n_tests.where(e.n_tests > 0))
    t = e.groupby(["dataset", "variant"]).perms_per_test.mean().unstack().dropna()
    d = t["EADD"] - t["EADD, full test"]
    rows = [dict(test="H3a identical detections (>= 95%)", value=same / len(pairs), n=len(pairs),
                 supported=same / len(pairs) >= 0.95),
            dict(test="H3b fewer permutations per test", value=f"{t['EADD'].mean():.1f} vs {t['EADD, full test'].mean():.1f}",
                 n=len(t), p=stats.wilcoxon(d, alternative="less").pvalue if (d != 0).any() else 1.0)]
    rows[1]["supported"] = rows[1]["p"] < 0.05
    if os.path.exists(os.path.join(OUT_DIR, "shap.jsonl")):
        hits = []
        for r in load("shap.jsonl").itertuples():
            true, k, seen = set(r.shifted), len(r.shifted), set()
            for det in r.detections:
                if det["drift"] is None or det["at"] - det["drift"] > tolerance(r.dataset) or det["drift"] in seen:
                    continue
                seen.add(det["drift"])
                hits.append(dict(dataset=r.dataset, precision=len(true & set(det["ranking"][:k])) / k,
                                 top1=det["ranking"][0] in true, chance=k / r.n_features))
        h = pd.DataFrame(hits).groupby("dataset").mean()
        dd = h.precision - h.chance
        rows.append(dict(test="H4 precision@k above chance", value=f"{h.precision.mean():.2f} vs {h.chance.mean():.2f} (top-1 {h.top1.mean():.0%})",
                         n=len(h), p=stats.wilcoxon(dd, alternative="greater").pvalue))
        rows[-1]["supported"] = rows[-1]["p"] < 0.05
        h.round(3).to_csv(os.path.join(OUT_DIR, "h4_per_stream.csv"))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "h3_h4.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
    if os.path.exists(os.path.join(OUT_DIR, "runs.jsonl")) and "EADD, full test" in load("runs.jsonl").variant.values:
        h3_h4()
    if os.path.exists(os.path.join(OUT_DIR, "runs.jsonl")) and "SPLL" in load("runs.jsonl").variant.values:
        all_groups()


def all_groups():
    """Addendum 2: every detector on the same streams; Friedman + Nemenyi over the 21 drift streams."""
    from analyze_benchmark import friedman, METRICS
    ps = per_stream(load("runs.jsonl"))
    drift = ps[ps.group != "no drift"]
    drift.groupby(["group", "variant"])[["recall", "mtd", "fa"]].mean().round(3).to_csv(
        os.path.join(OUT_DIR, "all_groups.csv"))
    fr = pd.concat([friedman(drift.assign(dataset=drift.dataset), m) for m in METRICS])
    fr.to_csv(os.path.join(OUT_DIR, "all_groups_friedman.csv"), index=False)
    nd = ps[ps.group == "no drift"].pivot(index="dataset", columns="variant", values="fa_per_10k")
    nd.round(2).to_csv(os.path.join(OUT_DIR, "all_groups_no_drift.csv"))
    print(drift.groupby("variant")[["recall", "mtd", "fa"]].mean().round(2).sort_values("mtd").to_string())
    print(fr.round(3).to_string(index=False))
