# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Figures for Chapter 4 of the Thesis
===================================
Reads experiments/results_hypotheses/runs.jsonl and the tables written by
analyze_hypotheses.py.

  detection_timeline.pdf   detections of every detector on the four annotated
                           INSECTS streams (seed 42), with the drift points and
                           their tolerance windows
  detection_offsets.pdf    position of EADD and D3-LightGBM detections relative
                           to the nearest drift point, all seeds
  threshold_sensitivity.pdf  recall and false alarms of EADD with and without
                           ASPT against the AUC threshold

Outputs:
  experiments/results_hypotheses/figures/*.pdf
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_hypotheses import DIR, ORDER, SHORT, STREAMS, TOLERANCE, load

OUT = os.path.join(DIR, "figures")
EADD_COLOR, D3_COLOR, OTHER_COLOR, INK, MUTED = "#2a78d6", "#eb6834", "#7a7974", "#0b0b0b", "#52514e"
LONG = {"InsectsAbruptBalanced": "INSECTS abrupt", "InsectsGradualBalanced": "INSECTS gradual",
        "InsectsIncrementalAbruptBalanced": "INSECTS incremental-abrupt",
        "InsectsIncrementalReoccurringBalanced": "INSECTS incremental-reoccurring"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def color(detector):
    return {"EADD": EADD_COLOR, "D3-LightGBM": D3_COLOR}.get(detector, OTHER_COLOR)


def default_runs(df):
    a = df[(df.part == "annotated") & (df.tau == 0.7) & (df.freq == 50)]
    return a[(a.detector != "EADD") | (a.test == "aspt")]


def timeline(df, seed=42):
    a = default_runs(df)
    a = a[a.seed == seed]
    detectors = [d for d in ORDER if d in set(a.detector)][::-1]  # EADD at the bottom
    fig, axes = plt.subplots(len(STREAMS), 1, figsize=(6.3, 8.2))
    for ax, stream in zip(axes, STREAMS):
        runs = a[a.dataset == stream].set_index("detector")
        drifts = runs.true_drifts.iloc[0]
        n = runs.n_samples.iloc[0]
        for td in drifts:
            ax.axvspan(td, td + TOLERANCE, color=EADD_COLOR, alpha=0.12, lw=0)
            ax.axvline(td, color=INK, lw=0.8, ls=(0, (3, 2)))
        for y, det in enumerate(detectors):
            if det in runs.index:
                ax.plot(runs.loc[det, "detections"], [y] * len(runs.loc[det, "detections"]),
                        ls="none", marker="|", ms=7, mew=1.2, color=color(det))
        ax.set_yticks(range(len(detectors)), detectors)
        for label in ax.get_yticklabels():
            if label.get_text() in ("EADD", "D3-LightGBM"):
                label.set_color(INK)
                label.set_fontweight("bold")
        ax.set_ylim(-0.7, len(detectors) - 0.3)
        ax.set_xlim(0, n)
        ax.set_title(f"{LONG[stream]} ({n:,} samples)", loc="left", color=INK)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e4e3df", lw=0.5)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("Sample index")
    fig.tight_layout(h_pad=1.2)
    fig.savefig(os.path.join(OUT, "detection_timeline.pdf"))
    plt.close(fig)


def offsets(df, reach=1500):
    a = default_runs(df)
    a = a[a.detector.isin(["EADD", "D3-LightGBM"])]
    fig, axes = plt.subplots(1, len(STREAMS), figsize=(6.3, 2.4), sharey=True)
    for ax, stream in zip(axes, STREAMS):
        ax.axvspan(0, TOLERANCE, color=EADD_COLOR, alpha=0.12, lw=0)
        ax.axvline(0, color=INK, lw=0.8, ls=(0, (3, 2)))
        for y, det in enumerate(["EADD", "D3-LightGBM"]):
            rows = a[(a.dataset == stream) & (a.detector == det)]
            off = []
            for _, r in rows.iterrows():
                for d in r.detections:
                    nearest = min(r.true_drifts, key=lambda td: abs(d - td))
                    if abs(d - nearest) <= reach:
                        off.append(d - nearest)
            jitter = np.random.default_rng(0).uniform(-0.25, 0.25, len(off))
            ax.plot(off, y + jitter, ls="none", marker="o", ms=3.5, mew=0, alpha=0.6, color=color(det))
        ax.set_title(SHORT[stream], color=INK)
        ax.set_xlim(-reach, reach)
        ax.set_xticks([-1000, 0, 1000])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e4e3df", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_yticks([0, 1], ["EADD", "D3-LightGBM"])
    axes[0].set_ylim(-0.6, 1.6)
    fig.supxlabel("Detection position relative to the drift point (samples)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "detection_offsets.pdf"))
    plt.close(fig)


def threshold():
    thr = pd.read_csv(os.path.join(DIR, "h3_threshold_table.csv"))
    d3 = pd.read_csv(os.path.join(DIR, "h1_h2_table.csv"), header=[0, 1], index_col=0)
    d3_recall = d3["recall"][STREAMS].loc["D3-LightGBM"].mean()
    d3_fa = d3["fa"][STREAMS].loc["D3-LightGBM"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5))
    series = [("EADD", "EADD (ASPT)", EADD_COLOR, "o", "-"),
              ("EADD-NoTest", "EADD (no test)", EADD_COLOR, "s", (0, (4, 2)))]
    for ax, metric, d3_value, label in [(axes[0], "recall", d3_recall, "Recall"),
                                        (axes[1], "fa", d3_fa, "False alarms per run")]:
        for variant, name, c, m, ls in series:
            r = thr[thr.variant == variant].sort_values("tau")
            ax.plot(r.tau, r[metric], color=c, marker=m, ms=5, lw=2, ls=ls,
                    mfc=c if variant == "EADD" else "white", mew=1.5, label=name)
        ax.axhline(d3_value, color=D3_COLOR, lw=2, label="D3-LightGBM ($\\tau = 0.7$)")
        ax.set_xlabel("AUC threshold $\\tau$")
        ax.set_ylabel(label)
        ax.set_xticks(sorted(thr.tau.unique()))
        ax.grid(axis="y", color="#e4e3df", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, None)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, "threshold_sensitivity.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df = load()
    timeline(df)
    offsets(df)
    threshold()
    print(f"Figures in {OUT}")
