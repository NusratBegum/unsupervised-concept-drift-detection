# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Analysis of experiment_benchmark.py. The unit of analysis is a stream: runs
are first averaged over seeds, so seeds of a fixed real stream are not
counted as independent samples.

  per_stream.csv   recall, MTD, false alarms, delay per detector and stream
  summary.csv      means per detector and stream group
  friedman.csv     Friedman test over streams, mean ranks, Nemenyi critical difference
  matched.csv      EADD vs D3 in the same setup (windows, threshold, check interval, folds),
                   Wilcoxon over streams, Holm-corrected within each metric
  matched_two_sided.csv  the same, counting alarms up to 500 samples before a drift as hits
  drift_type.csv   means per drift type (abrupt, gradual, incremental, recurring)
  components.csv   EADD's parts one at a time in EADD's code, against both parts off
  pairwise.csv     EADD (fixed) vs each other detector: Wilcoxon signed-rank over
                   streams, Holm-corrected within each metric
  accuracy.csv     unannotated streams: prequential accuracy (Lukats et al.)
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from analyze_hypotheses import detection_metrics

OUT_DIR = os.path.join("experiments", "results_benchmark")
REF = "EADD (fixed)"
METRICS = {"mtd": "lower", "fa": "lower", "recall": "higher"}


def tolerance(dataset):
    # gradual and incremental changes last 1,000 samples
    # by prefix: "InjectAbrupt_InsectsIncrementalBalanced" is abrupt drift on an INSECTS base stream
    return 1500 if dataset.startswith(("SynthGradual", "SynthIncremental", "InjectIncremental", "InjectGradual")) else 500


def per_stream(df, before=0):
    """before > 0 also counts detections up to `before` samples before a drift point as hits
    (recall and false alarms only; MTD is then measured from the start of the window)."""
    # streams without drift points (SynthStable) have no recall or MTD; see experiment_validity part F
    a = df[(df.part == "annotated") & (df.true_drifts.str.len() > 0)].copy()
    m = a.apply(lambda r: pd.Series(detection_metrics(
        r.detections, [d - before for d in r.true_drifts], tolerance(r.dataset) + before)), axis=1)
    a = pd.concat([a, m], axis=1)
    a["delay"] = a.apply(lambda r: np.nan if r.n_matched == 0 else
                         (r.mtd * len(r.true_drifts) - tolerance(r.dataset) * (len(r.true_drifts) - r.n_matched))
                         / r.n_matched, axis=1)
    return a.groupby(["group", "dataset", "variant"])[["recall", "mtd", "fa", "delay"]].mean().reset_index()


def friedman(ps, metric):
    t = ps.pivot(index="dataset", columns="variant", values=metric).dropna(axis=1)
    sign = 1 if METRICS[metric] == "lower" else -1
    ranks = (sign * t).rank(axis=1).mean()
    n, k = t.shape
    stat, p = stats.friedmanchisquare(*[t[c] for c in t.columns])
    q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    return pd.DataFrame(dict(metric=metric, variant=ranks.index, mean_rank=ranks.values,
                             n_streams=n, friedman_p=p, nemenyi_cd=cd)).sort_values("mean_rank")


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (len(p) - i) * p[j])
        adj[j] = min(1.0, running)
    return adj


def pairwise(ps):
    rows = []
    for metric in METRICS:
        t = ps.pivot(index="dataset", columns="variant", values=metric)
        block = []
        for v in t.columns.drop(REF):
            d = (t[REF] - t[v]).dropna()
            p = stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0
            block.append(dict(metric=metric, other=v, n_streams=len(d), eadd_fixed_mean=t[REF].mean(),
                              other_mean=t[v].mean(), eadd_better_on=int(((d < 0) if METRICS[metric] == "lower"
                                                                         else (d > 0)).sum()),
                              ties=int((d == 0).sum()), p=p))
        b = pd.DataFrame(block)
        b["p_holm"] = holm(b.p)
        rows.append(b)
    return pd.concat(rows)


# EADD vs D3 with the same windows (500/200), threshold 0.7, check interval and CV folds
MATCHED = [
    ("every 50, 5-fold", "EADD (fixed, history null, stride 10)", "D3-LightGBM (500/200, every 50)"),
    ("every 50, 5-fold", "EADD (fixed, history null, stride 10)", "D3 (500/200, every 50)"),
    ("every 50, 5-fold", "EADD (fixed)", "D3-LightGBM (500/200, every 50)"),
    ("every 50, 5-fold", "EADD (fixed)", "D3 (500/200, every 50)"),
    ("every 200, 2-fold", "EADD (fixed, every 200)", "D3-LightGBM (500/200, every 200)"),
    ("every 200, 2-fold", "EADD (fixed, every 200)", "D3 (500/200, every 200)"),
]


def matched(ps, metrics=METRICS):
    """Wilcoxon over streams for each same-setup pair, Holm-corrected within each metric."""
    rows = []
    for metric in metrics:
        t = ps.pivot(index="dataset", columns="variant", values=metric)
        block = []
        for setup, a, b in MATCHED:
            if a not in t or b not in t:
                continue
            d = (t[a] - t[b]).dropna()
            better = (d < 0) if METRICS[metric] == "lower" else (d > 0)
            block.append(dict(metric=metric, setup=setup, eadd=a, d3=b, n_streams=len(d),
                              eadd_mean=t[a].mean(), d3_mean=t[b].mean(), eadd_better_on=int(better.sum()),
                              ties=int((d == 0).sum()), p=stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0))
        if block:
            b_ = pd.DataFrame(block)
            b_["p_holm"] = holm(b_.p)
            rows.append(b_)
    return pd.concat(rows)


DRIFT_TYPE = {"SynthAbrupt": "abrupt", "InsectsAbruptBalanced": "abrupt", "SynthGradual": "gradual",
              "InsectsGradualBalanced": "gradual", "SynthIncremental": "incremental",
              "InsectsIncrementalAbruptBalanced": "incremental", "SynthRecurring": "recurring",
              "InsectsIncrementalReoccurringBalanced": "recurring"}


def drift_type(dataset):
    if dataset.startswith("InjectAbrupt"):
        return "abrupt"
    if dataset.startswith("InjectIncremental"):
        return "incremental"
    return DRIFT_TYPE[dataset]


# one code path (EADD's), identical windows and check times; one part switched at a time
BASE = "EADD parts: sliding, threshold only"
COMPONENTS = [("+ ASPT", "EADD parts: sliding + ASPT"), ("+ reservoir", "EADD parts: reservoir, threshold only"),
              ("+ reservoir + ASPT", "EADD (fixed)"), ("+ reservoir + ASPT, history null", "EADD (fixed, history null)"),
              ("+ reservoir + ASPT, history null (stride 10)", "EADD (fixed, history null, stride 10)")]


def components(ps):
    """Each EADD part against EADD's code with both parts off; Wilcoxon over streams, Holm per metric."""
    rows = []
    for metric in METRICS:
        t = ps.pivot(index="dataset", columns="variant", values=metric)
        if BASE not in t:
            continue
        block = []
        for label, v in COMPONENTS:
            if v not in t:
                continue
            d = (t[v] - t[BASE]).dropna()
            better = (d < 0) if METRICS[metric] == "lower" else (d > 0)
            block.append(dict(metric=metric, step=label, variant=v, n_streams=len(d), mean=t[v].mean(),
                              base_mean=t[BASE].mean(), better_on=int(better.sum()), ties=int((d == 0).sum()),
                              p=stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0))
        b = pd.DataFrame(block)
        b["p_holm"] = holm(b.p)
        rows.append(b)
    return pd.concat(rows) if rows else pd.DataFrame()


def main():
    df = pd.DataFrame([json.loads(l) for l in open(os.path.join(OUT_DIR, "runs.jsonl")) if l.strip()])
    ps = per_stream(df)
    ps.to_csv(os.path.join(OUT_DIR, "per_stream.csv"), index=False)
    summary = ps.groupby(["group", "variant"])[["recall", "mtd", "fa", "delay"]].mean().reset_index()
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    print(summary.round(2).to_string(index=False))
    fr = pd.concat([friedman(ps, m) for m in METRICS])
    fr.to_csv(os.path.join(OUT_DIR, "friedman.csv"), index=False)
    print(fr.round(3).to_string(index=False))
    ps["drift_type"] = ps.dataset.map(drift_type)
    ps.groupby(["drift_type", "variant"])[["recall", "mtd", "fa"]].mean().reset_index().to_csv(
        os.path.join(OUT_DIR, "drift_type.csv"), index=False)
    comp = components(ps)
    comp.to_csv(os.path.join(OUT_DIR, "components.csv"), index=False)
    print(comp.round(4).to_string(index=False))
    mt = matched(ps)
    mt.to_csv(os.path.join(OUT_DIR, "matched.csv"), index=False)
    # sensitivity: alarms up to 500 samples before a drift point count as hits
    two = matched(per_stream(df, before=500), metrics=["recall", "fa"])
    two.to_csv(os.path.join(OUT_DIR, "matched_two_sided.csv"), index=False)
    print(two.round(4).to_string(index=False))
    print(mt.round(4).to_string(index=False))
    pw = pairwise(ps)
    pw.to_csv(os.path.join(OUT_DIR, "pairwise.csv"), index=False)
    print(pw.round(4).to_string(index=False))
    u = df[df.part == "unannotated"]
    if len(u):
        acc = u.pivot_table(index="dataset", columns="variant", values="acc_assisted", aggfunc="mean")
        acc.to_csv(os.path.join(OUT_DIR, "accuracy.csv"))
        print((acc * 100).round(1).to_string())


if __name__ == "__main__":
    main()
