# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""Vector figures of the final registered test for Chapter 4 (experiments/results_final/figures/*.pdf).
  python make_final_figures.py
Style: the MU-IEEE-Paper design system (5.8 in wide, Arial 8-9 pt, 1 pt strokes, colour + marker/hatch per series)."""
import json
import os

import matplotlib
matplotlib.use("pdf")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

from analyze_benchmark import tolerance
from analyze_hypotheses import detection_metrics

D = os.path.join("experiments", "results_final")
OUT = os.path.join(D, "figures")
W = 5.8  # thesis text width in inches; figures are included at \linewidth, so sizes print as drawn
GREEN, BLUE, AMBER, GREY, INK = "#006C6C", "#0035AD", "#D55E00", "#595959", "#1A1A1A"
# series -> (colour, marker, bar hatch); filled marker and solid bar for EADD only, so every series also reads in grayscale
S = {"EADD": (GREEN, "o", None), "D3-LightGBM": (BLUE, "s", "////"), "D3": (AMBER, "^", "...."),
     "LSDD-Online": (GREY, "D", "xxx"), "MMD-Online": (GREY, "x", "\\\\\\\\"), "CVM-Online (100k)": (GREY, "+", "++")}
REF = dict(color=GREY, lw=0.8, ls=":")  # reference lines: equality, nominal level
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.titlesize": 9,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 1, "xtick.major.width": 1, "ytick.major.width": 1, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "lines.linewidth": 1.2, "hatch.linewidth": 0.6, "patch.linewidth": 1,
    "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.5, "legend.labelspacing": 0.3,
    "legend.borderaxespad": 0.2, "legend.borderpad": 0.2,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "pdf.fonttype": 42, "figure.dpi": 150})
SAMPLES = StrMethodFormatter("{x:,.0f}")


def bar_style(v):
    c, _, hatch = S[v]
    return dict(facecolor=c if hatch is None else "white", edgecolor=c, hatch=hatch, linewidth=1)


def marker_style(v):
    c, m, hatch = S[v]
    return dict(ls="none", marker=m, ms=5.5, color=c, mec=c, mew=1.2, mfc=c if hatch is None else "white")


def panel(ax, letter, text=""):
    ax.set_title(rf"$\bf{{({letter})}}$  {text}", loc="left", pad=4)


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


def strips(ax, df, metric, variants, pats, xlabel):
    """One row per detector within each drift pattern: a dot per stream (seed mean) and a vertical bar at the mean,
    so the spread across streams stays visible instead of hiding behind a single bar."""
    for j, p in enumerate(pats):
        for i, v in enumerate(variants):
            y = -(j * (len(variants) + 1) + i)
            vals = df[(df.variant == v) & (df.pattern == p)][metric].dropna()
            c, m, hatch = S[v]
            ax.plot(vals, np.full(len(vals), y), ls="none", marker=m, ms=4, mew=0.9, color=c, mec=c,
                    mfc=c if hatch is None else "white", alpha=0.75, zorder=3, label=v if j == 0 else None)
            ax.plot([vals.mean()] * 2, [y - 0.38, y + 0.38], color=INK, lw=1.6, zorder=4,
                    label="Mean" if (j, i) == (0, 0) else None)
        if j:
            ax.axhline(-(j * (len(variants) + 1)) + 1, color="#D9D9D9", lw=0.5)
    ax.set_yticks([-(j * (len(variants) + 1) + 1) for j in range(len(pats))], [p.capitalize() for p in pats])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)


def main():
    os.makedirs(OUT, exist_ok=True)
    three = ["EADD", "D3-LightGBM", "D3"]
    pats = ["gradual", "incremental", "abrupt"]
    df = per_pattern()
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.6), sharey=True, layout="constrained")
    fig.get_layout_engine().set(wspace=0.06)
    strips(axs[0], df, "recall", three, pats, "Recall")
    axs[0].set_xlim(-0.05, 1.05); axs[0].set_xticks([0, 0.5, 1])
    strips(axs[1], df, "mtd", three, pats, "MTD (Samples)")
    strips(axs[2], df, "delay", three, pats, "Delay When Detected (Samples)")
    for ax, p in zip(axs, "abc"):
        panel(ax, p)
    for ax in axs[1:]:
        ax.xaxis.set_major_formatter(SAMPLES)
    axs[1].set_xlim(-50, 1550)
    axs[2].set_xlim(-50, 1550)
    # legend above the panels: inside panel (b) it sat on the bars
    h, l = axs[0].get_legend_handles_labels()
    fig.legend([h[i] for i in (0, 2, 3, 1)], [l[i] for i in (0, 2, 3, 1)], loc="outside upper center", ncol=4)
    fig.savefig(os.path.join(OUT, "fig_final_recall_mtd.pdf"))

    pt = pd.read_csv(os.path.join(D, "per_type.csv"))
    fig, ax = plt.subplots(figsize=(W, 1.95), layout="constrained")
    # both groups are without drift; labels use the row names of Table 4.4
    groups = [("no drift", "Other Streams\nwithout Drift (8)"), ("autocorrelated", "Autocorrelated\nStreams (6)")]
    h = 0.24
    for i, v in enumerate(three):
        vals = [pt[(pt.group == g) & (pt.variant == v)].fa_per_10k.iloc[0] for g, _ in groups]
        y = -np.arange(2) - (i - 1) * (h + 0.04)
        b = ax.barh(y, vals, h, label=v, **bar_style(v))
        ax.bar_label(b, fmt="%.2f", padding=3)
    ax.set_yticks(-np.arange(2), [n for _, n in groups])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 22); ax.set_xticks(range(0, 21, 5))
    ax.set_xlabel("False Alarms per 10,000 Samples")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=3)
    fig.savefig(os.path.join(OUT, "fig_final_false_alarms.pdf"))

    ps = pd.read_csv(os.path.join(D, "per_stream.csv"))
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.05), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1, 0.62]))
    pts = dict(ls="none", marker="o", ms=4.5, color=GREEN, mec="white", mew=0.5)  # white ring separates coincident streams
    drift = ps[ps.group.isin(["slow", "abrupt"])].pivot(index="dataset", columns="variant", values="mtd")
    lim = drift[["EADD", "EADD, full test"]].max().max() * 1.05
    lim = [-0.04 * lim, lim]  # start below 0 so streams at 0 stay clear of the spines
    axs[0].plot(lim, lim, **REF)
    axs[0].plot(drift["EADD, full test"], drift["EADD"], **pts)
    axs[0].set_xlabel("MTD, Full Test"); axs[0].set_ylabel("MTD, ASPT")
    fa = ps[ps.group.isin(["slow", "abrupt"])].pivot(index="dataset", columns="variant", values="fa")
    lim2 = max(1, fa[["EADD", "EADD, full test"]].max().max() * 1.1)
    lim2 = [-0.04 * lim2, lim2]  # 9 streams sit at (0, 0)
    axs[1].plot(lim2, lim2, **REF)
    axs[1].plot(fa["EADD, full test"], fa["EADD"], **pts)
    axs[1].set_xlabel("False Alarms, Full Test"); axs[1].set_ylabel("False Alarms, ASPT")
    for ax, l in ((axs[0], lim), (axs[1], lim2)):
        ax.set_xlim(l); ax.set_ylim(l); ax.set_aspect("equal", adjustable="box")
        for a in (ax.xaxis, ax.yaxis):
            a.set_major_locator(plt.MaxNLocator(4, integer=True))
    for a in (axs[0].xaxis, axs[0].yaxis):
        a.set_major_formatter(SAMPLES)
    pp = ps[ps.group.isin(["slow", "abrupt", "no drift"])].groupby("variant").perms_per_test.mean()
    b = axs[2].bar(["ASPT", "Full Test"], [pp["EADD"], pp["EADD, full test"]], 0.55,
                   color=[GREEN, "white"], edgecolor=GREEN, linewidth=1)
    axs[2].bar_label(b, fmt="%.0f", padding=2)
    axs[2].tick_params(axis="x", length=0)
    axs[2].margins(y=0.15)
    axs[2].set_ylabel("Permutations per Test")
    for ax, p in zip(axs, "abc"):
        panel(ax, p)
    fig.savefig(os.path.join(OUT, "fig_aspt_equivalence.pdf"))

    h = pd.read_csv(os.path.join(D, "h4_per_stream.csv")).sort_values("precision")
    fig, ax = plt.subplots(figsize=(W, 3.45), layout="constrained")
    y = np.arange(len(h))
    ax.grid(axis="x", color="#D9D9D9", lw=0.5)
    bh = 0.38
    ax.barh(y, h.precision, bh, color=GREEN, edgecolor=GREEN, linewidth=1, label="TreeSHAP Precision@$k$")

    def name(d):
        pat, _, src = d.replace("Inject", "").partition("_")
        if not src:  # SynthAbrupt -> Synthetic (abrupt)
            return f"Synthetic ({d.replace('Synth', '')})"
        src = src.replace("InsectsIncrementalBalanced", "INSECTS Incremental").replace("RialtoBridgeTimelapse", "Rialto")
        return f"{src} ({pat})"
    ax.set_yticks(y, [name(d) for d in h.dataset])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(h) - 0.4)
    ax.set_xlim(0, 1.0); ax.set_xlabel("Precision@$k$")
    fig.canvas.draw()  # run the layout so the axes height in points is final
    row_pt = ax.bbox.height * 72 / fig.dpi / np.ptp(ax.get_ylim())  # points per bar row
    # chance tick 0.77 of a row (about 2x the bar) so about 2 pt of dark end shows above and below every bar and the tick,
    # not its thin halo, dominates; the legend draws the same tick
    ax.scatter(h.chance, y, color=INK, marker="|", s=(0.77 * row_pt) ** 2, linewidths=1.2, zorder=3, label="Chance, $k/d$",
               path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
    hs, ls = ax.get_legend_handles_labels()
    fig.legend(hs[::-1], ls[::-1], loc="outside upper center", ncol=2)  # bars first, as plotted
    fig.savefig(os.path.join(OUT, "fig_h4_precision.pdf"))

    on = os.path.join(D, "online_per_type.csv")
    if os.path.exists(on):
        t = pd.read_csv(on)
        fig, axs = plt.subplots(1, 2, figsize=(W, 3.0), layout="constrained")
        for ax, grp, p, title in ((axs[0], "slow", "a", "Slow Drift (14 Streams)"), (axs[1], "abrupt", "b", "Abrupt Drift (7 Streams)")):
            for v in S:  # legend in the order of the encoding table, EADD first
                r = t[(t.group == grp) & (t.variant == v)]
                if r.empty:
                    continue
                ax.plot(r.fa, r.mtd, label=v.replace(" (100k)", ""), **marker_style(v))  # recall: Table 4.8
            ax.set_xlim(-0.7, 12.4); ax.set_xticks(range(0, 13, 2))  # same scale in both panels
            ax.set_xlabel("False Alarms per Stream"); ax.set_ylabel("MTD (Samples)")
            ax.yaxis.set_major_formatter(SAMPLES)
            panel(ax, p, title)
        axs[0].set_ylim(0, 1180)
        axs[1].set_ylim(0, 145)
        # one legend below both panels: with six detectors no corner of the plots is free of data
        fig.legend(*axs[0].get_legend_handles_labels(), loc="outside upper center", ncol=6, columnspacing=1.2)
        fig.savefig(os.path.join(OUT, "fig_newer_detectors.pdf"))

    a = pd.read_csv(os.path.join(D, "h2a.csv"))
    lab = {"iid": "Gaussian, Independent", "real_elec": "Shuffled Electricity", "real_poker": "Shuffled PokerHand",
           "ar0.3": "AR(1), $\\rho$ = 0.3", "ar0.5": "AR(1), $\\rho$ = 0.5"}
    order = list(lab.values())  # row order of Table 4.6, top to bottom
    a["name"] = a.source.map(lab) + np.where(a.config == "warm31", ", after alarm", ", steady")
    a = a.assign(k=a.source.map(lab).map(order.index), s=(a.config == "warm31")).sort_values(["k", "s"], ascending=False)
    fig, ax = plt.subplots(figsize=(W, 2.95), layout="constrained")
    y = (len(order) - 1 - a.k) * 2.6 + ~a.s  # pairs of rows (steady above after alarm), a gap between sources
    ax.barh(y, a.upper95, 0.34, color="#E6F0F0", edgecolor=GREEN, linewidth=1, label="95% Upper Bound")
    ax.plot(a.reject_rate, y, ls="none", marker="o", ms=5, color=GREEN, mec="white", mew=0.5, zorder=3,
            label="Observed Share")
    for x, ls, txt in ((0.05, ":", "Nominal Level 0.05"), (0.075, "--", "Margin 0.075")):
        ax.axvline(x, color=GREY, lw=0.8, ls=ls, zorder=1, label=txt)
    ax.set_yticks(y, np.where(a.s, "After Alarm", "Steady"))
    for src, k in zip(order, range(len(order))):  # source name left of its two rows
        ax.annotate(src, (0, (len(order) - 1 - k) * 2.6 + 0.5), xycoords=("axes fraction", "data"),
                    xytext=(-50, 0), textcoords="offset points", ha="right", va="center")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, y.max() + 0.7)
    ax.set_xlim(0, 0.09)
    ax.set_xlabel("Share of Tests with $p$ ≤ 0.05")
    hs, ls_ = ax.get_legend_handles_labels()
    fig.legend([hs[i] for i in (0, 3, 1, 2)], [ls_[i] for i in (0, 3, 1, 2)], loc="outside upper center", ncol=4)
    fig.savefig(os.path.join(OUT, "fig_test_validity.pdf"))
    print(sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
