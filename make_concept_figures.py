# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""Vector concept figures for thesis Chapter 2 (experiments/results_final/figures/).
  fig_c2st_concept.pdf: no drift vs drift as a classifier sees it
  fig_drift_patterns.pdf: the four drift patterns of the synthetic streams used in the thesis
Style: docs/design/MU-IEEE-Paper.md in the workspace (5.8 in wide, Arial 8-9 pt, 1 pt axes,
reference = blue dashed + hatch, current = green solid, highlights = amber dashed).
  python make_concept_figures.py"""
import os

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import ScaledTranslation
from scipy.stats import norm

from synthetic_streams import POINTS, SyntheticStream

OUT = os.path.join("experiments", "results_final", "figures")
BLUE, GREEN, AMBER, NEUTRAL, INK = "#0035AD", "#006C6C", "#D55E00", "#595959", "#1A1A1A"
GREEN_TINT, AMBER_TINT = "#E6F0F0", "#FBEADF"
BAND = "#DDDDDD"  # transition bands remain distinct from the white background in grayscale
DOT_ALPHA = 0.45  # Neutral at 45% prints as light grey, below the moving-mean line in grayscale
DASH = (0, (4, 2))
BAND_HATCH = "//"  # transition bands: sparse amber hatch over the amber tint; "////" outweighed the mean line in gs/Quartz renders
BAND_HATCH_LW = 0.35  # thinner than the 0.5 pt rc default so the band stays below the 1.2 pt mean line
LEGEND_COLSPACE = 1.8  # right-anchored legend: puts every handle between the drift lines (checked below)
WIDTH = 5.8  # thesis text width in inches; included at \textwidth, so 1:1
plt.rcParams.update({
    # Arial Unicode MS only supplies glyphs Arial lacks (the "≫" in panel (b))
    "font.family": ["Arial", "Arial Unicode MS"], "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 1.0,
    "xtick.major.width": 1.0, "ytick.major.width": 1.0, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False, "hatch.linewidth": 0.5,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.cal": "Arial", "mathtext.fallback": None,
    "pdf.fonttype": 42, "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03})


def panel_label(ax, letter, text, loc, xy, dy_pt=0):
    """Bold "(a)" plus plain text on one baseline. Plain text, not mathtext, so glyphs Arial lacks
    fall back to Arial Unicode MS (mathtext would draw a placeholder)."""
    row = HPacker(children=[TextArea(f"({letter})", textprops=dict(size=9, weight="bold", family="Arial")),
                            TextArea(text, textprops=dict(size=9))], align="baseline", pad=0, sep=2.5)
    trans = ax.transAxes + ScaledTranslation(0, dy_pt / 72, ax.figure.dpi_scale_trans)
    ax.add_artist(AnchoredOffsetbox(loc, child=row, pad=0, borderpad=0, frameon=False,
                                    bbox_to_anchor=xy, bbox_transform=trans))


def concept():
    x = np.linspace(-4, 9, 600)
    r = norm.pdf(x, 2.0, 1.0)
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.0), sharey=True, layout="constrained")
    fig.get_layout_engine().set(wspace=0.075)  # 0.30 in between the panel boxes
    # words, not "≈"/"≫": the advisor marked the symbol shorthand in prose (rulebook R11)
    for ax, mu, letter, name in ((axs[0], 2.15, "a", "No Drift, AUC near 0.5"),
                                 (axs[1], 5.5, "b", "Drift, AUC Well above 0.5")):
        c = norm.pdf(x, mu, 1.0 if mu < 3 else 1.2)
        # curves and fills stop where the density is below 1e-3, so no flat tail runs along the axis
        ax.fill_between(x, c, where=c > 1e-3, fc=GREEN_TINT, lw=0)
        ax.fill_between(x, r, where=r > 1e-3, fc="none", ec=BLUE, hatch="///", lw=0)
        ax.plot(np.where(c > 1e-3, x, np.nan), c, color=GREEN, lw=1.2, solid_capstyle="butt")
        ax.plot(np.where(r > 1e-3, x, np.nan), r, color=BLUE, lw=1.2, ls=DASH)
        panel_label(ax, letter, name, "lower left", (0, 1), dy_pt=4)
        ax.set_xlim(-4, 9); ax.set_ylim(0, 0.46)
        ax.set_xticks([-4, 0, 4, 8]); ax.set_yticks([0, 0.2, 0.4])
        ax.set_xlabel("Feature Value $x$")
    # the shift of the mean, drawn above both peaks so it crosses no fill
    axs[1].annotate("", xy=(5.5, 0.43), xytext=(2.0, 0.43),
                    arrowprops=dict(arrowstyle="-|>", color=NEUTRAL, lw=1.0, mutation_scale=8, shrinkA=0, shrinkB=0))
    axs[0].set_ylabel("Density")
    axs[0].legend(handles=[Patch(fc="white", ec=BLUE, hatch="///", ls="--", lw=1.0, label="Reference"),
                           Patch(fc=GREEN_TINT, ec=GREEN, lw=1.0, label="Current")],
                  loc="upper left", handlelength=1.6, handleheight=0.9, borderaxespad=0.2)
    fig.savefig(os.path.join(OUT, "fig_c2st_concept.pdf"))


def patterns():
    kinds = ("Abrupt", "Gradual", "Incremental", "Recurring")
    fig, axs = plt.subplots(4, 1, figsize=(WIDTH, 4.0), sharex=True, sharey=True, layout="constrained")
    for ax, letter, kind in zip(axs, "abcd", kinds):
        s = SyntheticStream(kind, 0)
        v = s.X[:, 0]
        t = np.arange(len(v))
        for p in POINTS:
            if kind in ("Gradual", "Incremental"):
                ax.axvspan(p, p + 1000, fc=BAND, lw=0, zorder=0)
            ax.axvline(p, color=INK, ls=DASH, lw=0.8, zorder=2)
        ax.scatter(t[::10], v[::10], s=1.5, color=NEUTRAL, alpha=DOT_ALPHA, lw=0, zorder=1)
        m = np.convolve(v, np.ones(200) / 200, mode="same")
        ax.plot(t[100:-100], m[100:-100], color=GREEN, lw=1.2, zorder=3)
        panel_label(ax, letter, kind, "upper left", (0.008, 0.94))
        ax.set_yticks([-3, 0, 3, 6])
        ax.tick_params(axis="x", bottom=ax is axs[-1])
    axs[0].set_xlim(0, 20_000); axs[0].set_ylim(-4.5, 6.8)  # every plotted point lies in [-4.1, 6.0]
    axs[-1].set_xticks(range(0, 20_001, 5_000))
    axs[-1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axs[-1].set_xlabel("Sample")
    fig.supylabel("Drifting Feature", fontsize=9)
    leg = axs[0].legend(
        handles=[Line2D([], [], ls="", marker="o", ms=3, mfc=NEUTRAL, mec="none", alpha=DOT_ALPHA,
                        label="Every 10th Sample"),
                 Line2D([], [], color=GREEN, lw=1.2, label="Moving Mean"),
                 Line2D([], [], color=INK, lw=0.8, ls=DASH, label="Drift Point"),
                 Patch(fc=BAND, lw=0, label="Transition")],
        loc="lower right", bbox_to_anchor=(1, 1.0), ncol=4, borderaxespad=0.7,
        handlelength=1.8, columnspacing=LEGEND_COLSPACE, handletextpad=0.5)
    fig.savefig(os.path.join(OUT, "fig_drift_patterns.pdf"))
    check_legend_clear(fig, axs[0], leg)


def check_legend_clear(fig, ax, leg, min_gap_pt=4):
    """Self-check: the legend floats above the axes and no handle sits within min_gap_pt of a drift line."""
    pt = fig.dpi / 72
    gap = (leg.get_window_extent().y0 - ax.get_window_extent().y1) / pt
    assert gap >= min_gap_pt, f"legend only {gap:.1f} pt above the axes"
    lines_px = [ax.transData.transform((p, 0))[0] for p in POINTS]
    for h in leg.legend_handles:
        e = h.get_window_extent()
        hit = [x for x in lines_px if e.x0 - min_gap_pt * pt <= x <= e.x1 + min_gap_pt * pt]
        assert not hit, f"legend handle {h.get_label()!r} sits over a drift line"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    concept()
    patterns()
