# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Tests of experiments/results_confirm/PREREGISTRATION.md, exactly as registered.

  results.csv     H1a, H1b, H1c, H2b (Holm-corrected) and secondary tests against D3
  h2a.csv         false-rejection rate and one-sided 95% Clopper-Pearson upper bound per condition
  per_type.csv    means per drift group and detector
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


if __name__ == "__main__":
    main()
