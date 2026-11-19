# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""Vector figures of the final registered test for Chapter 4 (experiments/results_final/figures/*.pdf).
  python make_final_figures.py"""
import json
import os

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_benchmark import tolerance
from analyze_hypotheses import detection_metrics

D = os.path.join("experiments", "results_final")
OUT = os.path.join(D, "figures")
# Okabe-Ito colour-blind-safe palette
C = {"EADD": "#009E73", "D3-LightGBM": "#D55E00", "D3": "#0072B2", "EADD, full test": "#999999",
     "LSDD-Online": "#CC79A7", "MMD-Online": "#E69F00", "CVM-Online (100k)": "#56B4E9"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 150})


def pattern(ds):
    # by prefix: "InjectAbrupt_InsectsIncrementalBalanced" is abrupt drift on an INSECTS base
    for k in ("Gradual", "Incremental", "Abrupt"):
        if ds.startswith(("Synth" + k, "Inject" + k)):
            return k.lower()
    assert False, ds


def per_pattern():
    ps = pd.read_csv(os.path.join(D, "per_stream.csv"))
    ps = ps[ps.group.isin(["slow", "abrupt"])].assign(pattern=lambda d: d.dataset.map(pattern))
    runs = pd.DataFrame([json.loads(l) for f in ("runs.0.jsonl", "runs.1.jsonl") for l in open(os.path.join(D, f))])
    rows = []
    for r in runs[runs.true_drifts.str.len() > 0].itertuples():
        m = detection_metrics(r.detections, r.true_drifts, tolerance(r.dataset))
        d = np.nan if m["n_matched"] == 0 else (m["mtd"] * len(r.true_drifts) - tolerance(r.dataset) * (len(r.true_drifts) - m["n_matched"])) / m["n_matched"]
        rows.append(dict(dataset=r.dataset, variant=r.variant, delay=d))
    dl = pd.DataFrame(rows).groupby(["dataset", "variant"]).delay.mean().reset_index()
    return ps.merge(dl, on=["dataset", "variant"], how="left")


def bars(ax, df, metric, variants, pats, ylabel, fmt):
    w = 0.8 / len(variants)
    for i, v in enumerate(variants):
        vals = [df[(df.variant == v) & (df.pattern == p)][metric].mean() for p in pats]
        x = np.arange(len(pats)) + (i - (len(variants) - 1) / 2) * w
        ax.bar(x, vals, w, color=C[v], label=v)
        for xx, vv in zip(x, vals):
            ax.text(xx, vv, fmt.format(vv), ha="center", va="bottom", fontsize=5.5, rotation=90)
    ax.set_xticks(range(len(pats)), [{"gradual": "Grad.", "incremental": "Incr.", "abrupt": "Abrupt"}[p] for p in pats])
    ax.margins(y=0.18)
    ax.set_ylabel(ylabel)


def main():
    os.makedirs(OUT, exist_ok=True)
    three = ["EADD", "D3-LightGBM", "D3"]
    pats = ["gradual", "incremental", "abrupt"]
    df = per_pattern()
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.8))
    bars(axs[0], df, "recall", three, pats, "Recall", "{:.2f}")
    axs[0].set_ylim(0, 1.15)
    bars(axs[1], df, "mtd", three, pats, "MTD (samples)", "{:.0f}")
    bars(axs[2], df, "delay", three, pats, "Delay when detected (samples)", "{:.0f}")
    axs[0].legend(frameon=False, fontsize=7, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.18))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_final_recall_mtd.pdf"))

    pt = pd.read_csv(os.path.join(D, "per_type.csv"))
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    groups = [("no drift", "No drift\n(independent, real, AR 0.5/0.9)"), ("autocorrelated", "No drift,\nAR 0.8-0.97")]
    w = 0.25
    for i, v in enumerate(three):
        vals = [pt[(pt.group == g) & (pt.variant == v)].fa_per_10k.iloc[0] for g, _ in groups]
        x = np.arange(2) + (i - 1) * w
        ax.bar(x, vals, w, color=C[v], label=v)
        for xx, vv in zip(x, vals):
            ax.text(xx, vv, f"{vv:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(2), [n for _, n in groups])
    ax.set_ylabel("False alarms per 10,000 samples")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_final_false_alarms.pdf"))

    ps = pd.read_csv(os.path.join(D, "per_stream.csv"))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    drift = ps[ps.group.isin(["slow", "abrupt"])].pivot(index="dataset", columns="variant", values="mtd")
    axs[0].scatter(drift["EADD, full test"], drift["EADD"], s=12, color=C["EADD"])
    lim = [0, drift[["EADD", "EADD, full test"]].max().max() * 1.05]
    axs[0].plot(lim, lim, color="#888", lw=0.8, ls="--")
    axs[0].set_xlabel("MTD, full test"); axs[0].set_ylabel("MTD, ASPT")
    fa = ps[ps.group.isin(["slow", "abrupt"])].pivot(index="dataset", columns="variant", values="fa")
    axs[1].scatter(fa["EADD, full test"], fa["EADD"], s=12, color=C["EADD"])
    lim = [0, max(1, fa[["EADD", "EADD, full test"]].max().max() * 1.1)]
    axs[1].plot(lim, lim, color="#888", lw=0.8, ls="--")
    axs[1].set_xlabel("False alarms, full test"); axs[1].set_ylabel("False alarms, ASPT")
    pp = ps[ps.group.isin(["slow", "abrupt", "no drift"])].groupby("variant").perms_per_test.mean()
    axs[2].bar(["ASPT", "Full test"], [pp["EADD"], pp["EADD, full test"]], color=[C["EADD"], C["EADD, full test"]])
    for i, v in enumerate([pp["EADD"], pp["EADD, full test"]]):
        axs[2].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    axs[2].set_ylabel("Permutations per test")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_aspt_equivalence.pdf"))

    h = pd.read_csv(os.path.join(D, "h4_per_stream.csv")).sort_values("precision")
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(h))
    ax.barh(y, h.precision, color=C["EADD"], label="precision@k (TreeSHAP)")
    ax.scatter(h.chance, y, color="#222", marker="|", s=60, zorder=3, label="chance (k/d)")
    ax.set_yticks(y, [d.replace("Inject", "").replace("_", " ").replace("BridgeTimelapse", "").replace("Balanced", "") for d in h.dataset], fontsize=7)
    ax.set_xlim(0, 1.0); ax.set_xlabel("Precision@k at the first detection of each drift point")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_h4_precision.pdf"))

    on = os.path.join(D, "online_per_type.csv")
    if os.path.exists(on):
        t = pd.read_csv(on)
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
        for ax, grp, title in ((axs[0], "slow", "Slow drift (14 streams)"), (axs[1], "abrupt", "Abrupt drift (7 streams)")):
            for v in t.variant.unique():
                r = t[(t.group == grp) & (t.variant == v)]
                if r.empty or v not in C:
                    continue
                ax.scatter(r.fa, r.mtd, s=40 * r.recall + 10, color=C[v], label=v)
                ax.annotate(f"{v}\nrecall {r.recall.iloc[0]:.2f}", (r.fa.iloc[0], r.mtd.iloc[0]), fontsize=6,
                            xytext=(4, 4), textcoords="offset points")
            ax.set_title(title, fontsize=9); ax.set_xlabel("False alarms per stream"); ax.set_ylabel("MTD (samples)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_newer_detectors.pdf"))

    a = pd.read_csv(os.path.join(D, "h2a.csv"))
    lab = {"iid": "Gaussian", "real_elec": "Electricity", "real_poker": "PokerHand", "ar0.3": "AR 0.3", "ar0.5": "AR 0.5"}
    a["name"] = a.source.map(lab) + np.where(a.config == "warm31", ", after alarm", ", steady")
    a = a.sort_values("name")
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    y = np.arange(len(a))
    ax.hlines(y, 0, a.upper95, color="#bbb", lw=3)
    ax.scatter(a.reject_rate, y, color=C["EADD"], zorder=3, s=16)
    ax.axvline(0.05, color="#222", ls=":", lw=0.8)
    ax.axvline(0.075, color="#D55E00", ls="--", lw=0.8)
    ax.set_yticks(y, a.name, fontsize=7)
    ax.set_xlabel("Share of no-drift tests with p <= 0.05 (dot) and 95% upper bound (bar)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_test_validity.pdf"))
    print(sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
