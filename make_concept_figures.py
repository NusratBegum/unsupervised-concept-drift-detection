# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""Vector concept figures for thesis Chapter 2 (experiments/results_final/figures/).
  fig_c2st_concept.pdf: no drift vs drift as a classifier sees it
  fig_drift_patterns.pdf: the four drift patterns of the synthetic streams used in the thesis
  python make_concept_figures.py"""
import os

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from synthetic_streams import POINTS, SyntheticStream

OUT = os.path.join("experiments", "results_final", "figures")
REF, CUR = "#56B4E9", "#CC79A7"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


def concept():
    x = np.linspace(-4, 9, 600)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, mu, title, note in ((axs[0], 2.15, "No drift: the windows overlap", "classifier cannot tell\nthe windows apart\n(AUC $\\approx$ 0.5)"),
                                (axs[1], 5.5, "Drift: the current window has shifted", "classifier separates\nthe windows\n(AUC $\\gg$ 0.5)")):
        r, c = norm.pdf(x, 2.0, 1.0), norm.pdf(x, mu, 1.0 if mu < 3 else 1.2)
        ax.fill_between(x, r, color=REF, alpha=0.45, lw=0)
        ax.plot(x, r, color=REF, lw=1.6, label="reference window, $P_{ref}(X)$")
        ax.fill_between(x, c, color=CUR, alpha=0.35, lw=0)
        ax.plot(x, c, color=CUR, lw=1.6, label="current window, $P_{cur}(X)$")
        ax.text(5.0 if mu < 3 else -3.8, 0.44, note, fontsize=7.5, va="top",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#888", lw=0.6))
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-4, 9); ax.set_ylim(0, 0.46); ax.set_xlabel("feature value $x$")
    axs[1].annotate("", xy=(5.4, 0.2), xytext=(2.2, 0.2), arrowprops=dict(arrowstyle="->", lw=1.2))
    axs[1].text(3.8, 0.215, "shift", ha="center", fontsize=8)
    axs[0].set_ylabel("density")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, "fig_c2st_concept.pdf"))


def patterns():
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.2), sharex=True, sharey=True)
    for ax, kind in zip(axs.flat, ("Abrupt", "Gradual", "Incremental", "Recurring")):
        s = SyntheticStream(kind, 0)
        v = s.X[:, 0]
        t = np.arange(len(v))
        ax.scatter(t[::10], v[::10], s=1.5, color="#999", alpha=0.6, rasterized=False)
        m = np.convolve(v, np.ones(200) / 200, mode="same")
        ax.plot(t[100:-100], m[100:-100], color="#009E73", lw=1.4, label="moving mean (200 samples)")
        for p in POINTS:
            ax.axvline(p, color="#D55E00", ls="--", lw=0.8)
            if kind in ("Gradual", "Incremental"):
                ax.axvspan(p, p + 1000, color="#D55E00", alpha=0.08, lw=0)
        ax.set_title(kind, fontsize=9)
    for ax in axs[1]:
        ax.set_xlabel("sample")
    for ax in axs[:, 0]:
        ax.set_ylabel("drifting feature $x_0$")
    axs[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_drift_patterns.pdf"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    concept()
    patterns()
