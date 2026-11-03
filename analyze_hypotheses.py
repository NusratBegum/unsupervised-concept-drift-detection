# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Tables and Significance Tests for the Thesis Hypotheses
=======================================================
Reads experiments/results_hypotheses/runs.jsonl (experiment_hypotheses.py)
and writes one CSV per table plus the significance tests.

Detection metrics on the annotated streams (per stream and seed):
  a detection matches a drift point if it falls within TOLERANCE samples
  after it; each drift point is matched at most once.
  recall  = matched drift points / drift points
  MTD     = mean delay over all drift points, where a missed drift point
            counts as TOLERANCE (the latest delay that would still match)
  FA      = detections that match no drift point
  precision (pooled over runs) = matched detections / detections
Lukats et al. (2024) average the delay over detected drifts only; their MTD
and missed detection rate are also written for comparison.

Unannotated streams follow Lukats et al.: prequential accuracy of a Hoeffding
tree that is reset at every detection, and lift-per-drift against a tree
trained on the first 1,000 samples only.

Outputs (experiments/results_hypotheses/):
  per_run_annotated.csv, h1_h2_table.csv, h3_aspt_table.csv,
  h3_threshold_table.csv, e4_unannotated_table.csv, tests.csv
  latex/*.tex  table bodies for the thesis
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

TOLERANCE = 500
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "results_hypotheses")
ORDER = ["BNDM", "CSDDM", "IBDD", "OCDD", "SPLL", "UDetect", "D3-LightGBM", "EADD"]
STREAMS = ["InsectsAbruptBalanced", "InsectsGradualBalanced",
           "InsectsIncrementalAbruptBalanced", "InsectsIncrementalReoccurringBalanced"]


def load():
    with open(os.path.join(DIR, "runs.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)
    df["tau"] = df.get("tau", np.nan).fillna(0.7) if "tau" in df else 0.7
    df["test"] = df.get("test", np.nan).fillna("-") if "test" in df else "-"
    df["freq"] = df.get("freq", np.nan).fillna(50) if "freq" in df else 50
    # keep the last run of each job if a job was repeated
    return df.drop_duplicates("key", keep="last")


def detection_metrics(detections, drifts, tolerance=TOLERANCE):
    matched, delays = set(), []
    for d in sorted(detections):
        for td in drifts:
            if td not in matched and td <= d <= td + tolerance:
                matched.add(td)
                delays.append(d - td)
                break
    n = len(drifts)
    fa = len(detections) - len(matched)
    mtd = (sum(delays) + tolerance * (n - len(matched))) / n
    # Lukats et al.: first detection after a drift and before the next one
    lk_delays, was_drift, prev = [], False, 0
    for t in sorted(set(drifts) | set(detections)):
        if t in drifts:
            was_drift, prev = True, t
        if t in detections and was_drift:
            lk_delays.append(t - prev)
            was_drift = False
    return dict(recall=len(matched) / n, mtd=mtd, fa=fa, n_detections=len(detections),
                n_matched=len(matched),
                mtd_lukats=np.mean(lk_delays) if lk_delays else np.nan,
                mdr_lukats=1 - len(lk_delays) / n)


def annotated_runs(df):
    a = df[df.part == "annotated"].copy()
    m = a.apply(lambda r: pd.Series(detection_metrics(r.detections, r.true_drifts)), axis=1)
    a = pd.concat([a, m], axis=1)
    a["variant"] = a.apply(variant_name, axis=1)
    return a


def variant_name(r):
    if r.detector != "EADD":
        return r.detector
    name = {"aspt": "EADD", "full": "EADD-FullPerm", "none": "EADD-NoTest"}[r.test]
    if r.freq != 50:
        name += f"-every{int(r.freq)}"
    return name


def one_sided(x, y, alternative):
    """Mann-Whitney U and paired Wilcoxon signed-rank p-values for x vs y."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mw = mannwhitneyu(x, y, alternative=alternative).pvalue
    diff = x - y
    wx = wilcoxon(x, y, alternative=alternative).pvalue if np.any(diff != 0) else np.nan
    return mw, wx


def main():
    df = load()
    a = annotated_runs(df)
    a.drop(columns=["detections", "true_drifts", "key"]).to_csv(
        os.path.join(DIR, "per_run_annotated.csv"), index=False)
    tests = []

    # H1 and H2 table: default configurations (tau = 0.7, ASPT, check every 50)
    main_rows = a[(a.tau == 0.7) & a.variant.isin(ORDER)]
    t12 = (main_rows.groupby(["variant", "dataset"])[["recall", "mtd", "fa", "mtd_lukats", "mdr_lukats"]]
           .mean().unstack("dataset"))
    t12 = t12.reindex(ORDER)
    t12.to_csv(os.path.join(DIR, "h1_h2_table.csv"))

    def paired(v1, v2, metric, tau1=0.7, tau2=0.7):
        x = a[(a.variant == v1) & (a.tau == tau1)].set_index(["dataset", "seed"])[metric]
        y = a[(a.variant == v2) & (a.tau == tau2)].set_index(["dataset", "seed"])[metric]
        idx = x.index.intersection(y.index)
        return x.loc[idx].values, y.loc[idx].values

    for metric, alt, h in [("recall", "greater", "H1a"), ("mtd", "less", "H1b"), ("fa", "less", "H2")]:
        x, y = paired("EADD", "D3-LightGBM", metric)
        mw, wx = one_sided(x, y, alt)
        tests.append(dict(hypothesis=h, comparison="EADD vs D3-LightGBM", metric=metric,
                          alternative=alt, n=len(x), mean_x=x.mean(), mean_y=y.mean(),
                          p_mannwhitney=mw, p_wilcoxon=wx))
    x, y = paired("EADD-every200", "D3-LightGBM", "mtd")
    mw, wx = one_sided(x, y, "less")
    tests.append(dict(hypothesis="H1b check", comparison="EADD (check every 200) vs D3-LightGBM",
                      metric="mtd", alternative="less", n=len(x), mean_x=x.mean(), mean_y=y.mean(),
                      p_mannwhitney=mw, p_wilcoxon=wx))

    # H3: ASPT vs full permutation test vs no test at the default threshold
    def summarise(rows, by=("variant",)):
        g = rows.groupby(list(by))
        out = g[["recall", "mtd", "fa", "detector_time_sec"]].mean()
        tests_ = g["n_tests"].sum()
        out["tests"] = g["n_tests"].mean()
        out["perms_per_test"] = g["n_permutations_used"].sum() / tests_.replace(0, np.nan)
        out["early_stop_pct"] = 100 * g["n_early_stops"].sum() / tests_.replace(0, np.nan)
        out["rejected_pct"] = 100 * g["n_rejections"].sum() / tests_.replace(0, np.nan)
        return out

    h3_rows = a[(a.tau == 0.7) & a.variant.isin(["EADD", "EADD-FullPerm", "EADD-NoTest"])]
    h3 = summarise(h3_rows)
    h3_stream = summarise(h3_rows, by=("dataset", "variant"))
    for metric in ("recall", "mtd", "fa"):
        x, y = paired("EADD", "EADD-FullPerm", metric)
        mw, wx = one_sided(x, y, "two-sided")
        tests.append(dict(hypothesis="H3", comparison="EADD (ASPT) vs EADD (full test)", metric=metric,
                          alternative="two-sided", n=len(x), mean_x=x.mean(), mean_y=y.mean(),
                          p_mannwhitney=mw, p_wilcoxon=wx, identical_runs=int(np.sum(x == y))))
    # Runtime: the parallel runs share the machine, so H3b uses the sequential
    # timing runs of experiment_aspt_runtime.py when they exist.
    timing = os.path.join(DIR, "aspt_runtime.csv")
    if os.path.exists(timing):
        r = pd.read_csv(timing)
        name = {"aspt": "EADD", "full": "EADD-FullPerm", "none": "EADD-NoTest"}
        h3["detector_time_sec"] = r.groupby("test").detector_time_sec.mean().rename(index=name)
        seq_time = r.assign(variant=r.test.map(name)).groupby(["dataset", "variant"]).detector_time_sec.mean()
        h3_stream["detector_time_sec"] = seq_time.reindex(h3_stream.index)
        w = r.pivot_table(index=["dataset", "seed"], columns="test", values="detector_time_sec")
        x, y, source = w["aspt"].values, w["full"].values, "sequential"
        # the sequential runs must reproduce the detections of the parallel runs
        par = a[(a.tau == 0.7) & a.variant.isin(["EADD", "EADD-FullPerm", "EADD-NoTest"])]
        par = par.set_index(["dataset", "seed", "test"]).n_detections
        seq = r.set_index(["dataset", "seed", "test"]).detections.apply(lambda d: len(json.loads(d)))
        print(f"sequential runs with the same number of detections: {(seq == par.loc[seq.index]).sum()} of {len(seq)}")
    else:
        x, y = paired("EADD", "EADD-FullPerm", "detector_time_sec")
        source = "parallel"
    mw, wx = one_sided(x, y, "less")
    tests.append(dict(hypothesis="H3", comparison=f"EADD (ASPT) vs EADD (full test), {source} runtime",
                      metric="detector_time_sec", alternative="less", n=len(x), mean_x=np.mean(x),
                      mean_y=np.mean(y), p_mannwhitney=mw, p_wilcoxon=wx,
                      identical_runs=int(np.sum(np.asarray(x) < np.asarray(y)))))
    h3.to_csv(os.path.join(DIR, "h3_aspt_table.csv"))
    h3_stream.to_csv(os.path.join(DIR, "h3_aspt_stream_table.csv"))

    # Threshold sensitivity
    rows = []
    for tau in sorted(a.tau.unique()):
        s = summarise(a[(a.tau == tau) & a.variant.isin(["EADD", "EADD-NoTest"])])
        s["tau"] = tau
        rows.append(s.reset_index())
        for v in ("EADD", "EADD-NoTest"):
            x, y = paired(v, "D3-LightGBM", "fa", tau1=tau)
            if len(x):
                mw, wx = one_sided(x, y, "less")
                tests.append(dict(hypothesis="threshold", comparison=f"{v} tau={tau} vs D3-LightGBM",
                                  metric="fa", alternative="less", n=len(x), mean_x=x.mean(),
                                  mean_y=y.mean(), p_mannwhitney=mw, p_wilcoxon=wx))
        x, y = paired("EADD", "EADD-NoTest", "fa", tau1=tau, tau2=tau)
        if len(x):
            mw, wx = one_sided(x, y, "less")
            tests.append(dict(hypothesis="threshold", comparison=f"EADD vs EADD-NoTest tau={tau}",
                              metric="fa", alternative="less", n=len(x), mean_x=x.mean(),
                              mean_y=y.mean(), p_mannwhitney=mw, p_wilcoxon=wx))
    pd.concat(rows).to_csv(os.path.join(DIR, "h3_threshold_table.csv"), index=False)

    # Experiment 4: unannotated streams (Lukats et al. proxy metrics)
    u = df[df.part == "unannotated"].copy()
    if len(u):
        u["n_detections"] = u.detections.apply(len)
        u["lpd"] = np.where(u.n_detections > 0,
                            (u.acc_assisted - u.acc_base) / u.n_detections.replace(0, 1), 0.0)
        e4 = u.groupby(["detector", "dataset"])[["acc_assisted", "acc_base", "n_detections", "lpd"]].mean()
        e4.to_csv(os.path.join(DIR, "e4_unannotated_table.csv"))
        acc = e4["acc_assisted"].unstack("dataset")
        if {"EADD", "D3-LightGBM"} <= set(acc.index):
            cols = acc.columns[acc.loc[["EADD", "D3-LightGBM"]].notna().all()]
            x, y = acc.loc["EADD", cols].values, acc.loc["D3-LightGBM", cols].values
            mw, wx = one_sided(x, y, "two-sided")
            tests.append(dict(hypothesis="E4", comparison="EADD vs D3-LightGBM (accuracy per stream)",
                              metric="accuracy", alternative="two-sided", n=len(x), mean_x=x.mean(),
                              mean_y=y.mean(), p_mannwhitney=mw, p_wilcoxon=wx))

    t = pd.DataFrame(tests)
    t.to_csv(os.path.join(DIR, "tests.csv"), index=False)
    write_latex(a, t12, h3, h3_stream, pd.concat(rows), df, t)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(t12.round(2))
        print(h3.round(2))
        print(t.round(4).to_string(index=False))


SHORT = {"InsectsAbruptBalanced": "Abrupt", "InsectsGradualBalanced": "Gradual",
         "InsectsIncrementalAbruptBalanced": "Incr.-abrupt",
         "InsectsIncrementalReoccurringBalanced": "Incr.-reoc.",
         "Electricity": "Elec.", "InsectsIncrementalBalanced": "Incr.", "Luxembourg": "Lux.",
         "NOAAWeather": "NOAA", "OutdoorObjects": "Outdoor", "Ozone": "Ozone",
         "PokerHand": "Poker", "Powersupply": "Power", "RialtoBridgeTimelapse": "Rialto"}
NAMES = {"D3-LightGBM": "D3-LightGBM", "EADD": "EADD (ASPT)", "EADD-FullPerm": "EADD (full test)",
         "EADD-NoTest": "EADD (no test)", "NoDetection": "No detector"}


def fmt(v, digits):
    return "--" if pd.isna(v) else f"{v:.{digits}f}"


def bold_best(values, digits, best):
    """Format a column and bold its best value(s)."""
    target = np.nanmax(values) if best == "max" else np.nanmin(values)
    return [("\\textbf{%s}" % fmt(v, digits)) if not pd.isna(v) and np.isclose(v, target)
            else fmt(v, digits) for v in values]


def pval(p):
    return "--" if pd.isna(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


def write_latex(a, t12, h3, h3_stream, thr, df, tests):
    out = os.path.join(DIR, "latex")
    os.makedirs(out, exist_ok=True)
    streams = [s for s in STREAMS if s in t12["recall"].columns]

    # H1: recall and MTD per stream, plus the mean over the annotated streams
    cols, heads = [], []
    for s in streams:
        cols += [bold_best(t12[("recall", s)].values, 2, "max"), bold_best(t12[("mtd", s)].values, 0, "min")]
        heads.append(SHORT[s])
    mean_rec = t12["recall"][streams].mean(axis=1).values
    mean_mtd = t12["mtd"][streams].mean(axis=1).values
    cols += [bold_best(mean_rec, 2, "max"), bold_best(mean_mtd, 0, "min")]
    lines = []
    for i, det in enumerate(t12.index):
        lines.append(" & ".join([det] + [c[i] for c in cols]) + " \\\\")
    with open(os.path.join(out, "h1_rows.tex"), "w") as f:
        f.write("% streams: " + ", ".join(heads + ["Mean"]) + "\n" + "\n".join(lines) + "\n")

    # H2: false alarms per stream and the total over the annotated streams
    cols = [bold_best(t12[("fa", s)].values, 1, "min") for s in streams]
    cols.append(bold_best(t12["fa"][streams].sum(axis=1).values, 1, "min"))
    main_rows = a[(a.tau == 0.7) & a.variant.isin(ORDER)].groupby("variant")
    precision = (main_rows.n_matched.sum() / main_rows.n_detections.sum()).reindex(t12.index)
    cols.append(bold_best(precision.values, 2, "max"))
    lines = [" & ".join([det] + [c[i] for c in cols]) + " \\\\"
             for i, det in enumerate(t12.index)]
    with open(os.path.join(out, "h2_rows.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # H3: ASPT, full permutation test and no test at tau = 0.7, per stream and overall
    def h3_line(first, v, r):
        return " & ".join([first, NAMES[v], fmt(r.recall, 2), fmt(r.mtd, 0), fmt(r.fa, 1),
                           fmt(r.detector_time_sec, 1), fmt(r.tests, 1),
                           fmt(r.perms_per_test, 1), fmt(r.early_stop_pct, 1),
                           fmt(r.rejected_pct, 1)]) + " \\\\"

    variants = ["EADD-NoTest", "EADD-FullPerm", "EADD"]
    lines = []
    for s in streams + ["mean"]:
        table = h3 if s == "mean" else h3_stream.loc[s]
        present = [v for v in variants if v in table.index]
        for k, v in enumerate(present):
            first = "" if k else ("\\multirow{%d}{*}{%s}" % (len(present), "Mean" if s == "mean" else SHORT[s]))
            lines.append(h3_line(first, v, table.loc[v]))
        lines.append("\\hline")
    with open(os.path.join(out, "h3_rows.tex"), "w") as f:
        f.write("\n".join(lines[:-1]) + "\n")

    # Threshold sensitivity, with the test of FA against D3-LightGBM
    lines = []
    for _, r in thr.sort_values(["tau", "variant"], ascending=[True, False]).iterrows():
        comp = f"{r.variant} tau={r.tau} vs D3-LightGBM"
        p = tests.loc[tests.comparison == comp, "p_mannwhitney"]
        lines.append(" & ".join([f"{r.tau:.1f}", NAMES.get(r.variant, r.variant), fmt(r.recall, 2),
                                 fmt(r.mtd, 0), fmt(r.fa, 1), fmt(r.detector_time_sec, 1),
                                 fmt(r.perms_per_test, 1), fmt(r.early_stop_pct, 1),
                                 fmt(r.rejected_pct, 1), pval(p.iloc[0] if len(p) else np.nan)])
                     + " \\\\")
    with open(os.path.join(out, "h3_threshold_rows.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # Experiment 4: accuracy per unannotated stream, detections and lift-per-drift
    u = df[df.part == "unannotated"].copy()
    if len(u):
        u["n_detections"] = u.detections.apply(len)
        u["lpd"] = np.where(u.n_detections > 0,
                            (u.acc_assisted - u.acc_base) / u.n_detections.replace(0, 1), 0.0)
        acc = u.groupby(["detector", "dataset"]).acc_assisted.mean().unstack("dataset")
        order = [d for d in ["NoDetection"] + ORDER if d in acc.index]
        acc = acc.reindex(order)
        ucols = [c for c in SHORT if c in acc.columns]
        cols = [bold_best(100 * acc[c].values, 1, "max") for c in ucols]
        cols.append(bold_best(100 * acc[ucols].mean(axis=1).values, 1, "max"))
        lines = [" & ".join([{"NoDetection": "No detector"}.get(d, d)] + [c[i] for c in cols]) + " \\\\"
                 for i, d in enumerate(acc.index)]
        with open(os.path.join(out, "e4_acc_rows.tex"), "w") as f:
            f.write("% streams: " + ", ".join(SHORT[c] for c in ucols) + ", Mean\n"
                    + "\n".join(lines) + "\n")
        g = u[u.detector != "NoDetection"].groupby(["detector", "dataset"])[["n_detections", "lpd"]].mean()
        det = g.n_detections.unstack("dataset").reindex([d for d in ORDER if d in g.index.levels[0]])
        lpd = g.lpd.unstack("dataset").reindex(det.index)
        lines = []
        for d in det.index:
            cells = [fmt(det.loc[d, c], 0) for c in ucols]
            lines.append(" & ".join([{"NoDetection": "No detector"}.get(d, d)] + cells + [fmt(det.loc[d, ucols].sum(), 0),
                                     fmt(100 * lpd.loc[d, ucols].mean(), 2)]) + " \\\\")
        with open(os.path.join(out, "e4_det_rows.tex"), "w") as f:
            f.write("\n".join(lines) + "\n")

    # Summary of the significance tests
    lines = []
    for _, r in tests[tests.hypothesis.isin(["H1a", "H1b", "H2", "H1b check", "H3", "E4"])].iterrows():
        lines.append(" & ".join([r.hypothesis, r.metric.replace("_", " "), r.alternative,
                                 fmt(r.mean_x, 2), fmt(r.mean_y, 2), pval(r.p_mannwhitney),
                                 pval(r.p_wilcoxon)]) + " \\\\")
    with open(os.path.join(out, "tests_rows.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
