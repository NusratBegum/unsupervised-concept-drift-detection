# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Figures for experiment_comparison.py (run analyze_comparison.py first).

  python make_comparison_figures.py

Writes PDF and 300-dpi PNG files to experiments/results_comparison/figures/.
Colours: Okabe-Ito palette (safe for colour-blind readers).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from analyze_comparison import load  # noqa: E402
from analyze_hypotheses import fmt  # noqa: E402
from experiment_comparison import OUT_DIR  # noqa: E402

FIG = os.path.join(OUT_DIR, "figures")
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "figure.dpi": 100})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), **kw)
    plt.close(fig)


def bars(summary, variants, labels, title, name):
    kinds = ["real (INSECTS)", "synthetic"]
    metrics = [("recall", "Recall (higher is better)"), ("mtd", "MTD in samples (lower is better)"),
               ("fa", "False alarms per run (lower is better)")]
    fig, axes = plt.subplots(len(kinds), 3, figsize=(10, 2.3 * len(kinds)))
    for r, kind in enumerate(kinds):
        s = summary[summary.kind == kind].set_index("variant").reindex(variants)
        for c, (m, mlabel) in enumerate(metrics):
            ax = axes[r, c]
            ax.barh(range(len(variants)), s[m], color=[OI[i % len(OI)] for i in range(len(variants))])
            for i, v in enumerate(s[m]):
                ax.text(v, i, " " + fmt(v, 2 if m == "recall" else 0 if m == "mtd" else 1),
                        va="center", fontsize=7)
            ax.set_yticks(range(len(variants)), labels if c == 0 else [""] * len(variants))
            ax.invert_yaxis()
            ax.set_title(f"{mlabel}\n{kind}" if r == 0 else kind, fontsize=8)
            if m == "fa" and s[m].max() > 50:
                ax.set_xscale("symlog", linthresh=10)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save(fig, name)


def timeline(df, dataset, seed, variants, labels, name, title):
    sub = df[(df.dataset == dataset) & (df.seed == seed)].set_index("variant")
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(variants) + 0.9))
    for i, v in enumerate(variants):
        d = sub.loc[v, "detections"]
        ax.scatter(d, [i] * len(d), marker="|", s=120, color=OI[i % len(OI)])
    for t in sub.iloc[0]["true_drifts"]:
        ax.axvline(t, color="grey", ls="--", lw=0.8)
    ax.set_yticks(range(len(variants)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("sample (dashed lines: annotated drift points)")
    ax.set_title(title, fontsize=10)
    save(fig, name)


def windows_diagram():
    """How the two reference memories pick their samples, at one check."""
    import numpy as np
    rng = np.random.default_rng(0)
    t, n_ref, n_cur = 2000, 500, 200
    fig, axes = plt.subplots(2, 1, figsize=(10, 3.2), sharex=True)
    rows = [("Sliding reference (D3, D3-LightGBM)", np.arange(t - n_cur - n_ref, t - n_cur), 200),
            ("Reservoir reference (EADD)", np.sort(rng.choice(t - n_cur, n_ref, replace=False)), 50)]
    for ax, (title, ref, every) in zip(axes, rows):
        ax.scatter(ref, np.zeros_like(ref), marker="|", s=200, color=OI[0], label="reference window (500 samples)")
        cur = np.arange(t - n_cur, t)
        ax.scatter(cur, np.zeros_like(cur), marker="|", s=200, color=OI[1], label="current window (newest 200)")
        checks = np.arange(every, t + 1, every)
        ax.scatter(checks, np.full_like(checks, 0.6), marker="v", s=14, color=OI[7],
                   label=f"a check (here every {every} samples)")
        ax.set_ylim(-0.6, 1.0)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=9, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axes[1].set_xlabel("samples seen since the last detected drift (now = 2,000)")
    fig.suptitle("Which samples a detector compares at one check", fontsize=10)
    fig.tight_layout()
    save(fig, "fig0_windows")


if __name__ == "__main__":
    windows_diagram()
    df = load()
    summary = pd.read_csv(os.path.join(OUT_DIR, "summary.csv"))
    main = ["D3 (authors)", "D3-LightGBM (thesis)", "EADD (thesis)",
            "S200 D3 (logistic regression)", "S200 D3-LightGBM", "S200 EADD",
            "S50 D3 (logistic regression)", "S50 D3-LightGBM", "S50 EADD"]
    labels = ["A  D3, authors' settings", "A  D3-LightGBM, thesis", "A  EADD, thesis",
              "B200  D3 (logistic reg.)", "B200  D3-LightGBM", "B200  EADD",
              "B50  D3 (logistic reg.)", "B50  D3-LightGBM", "B50  EADD"]
    bars(summary, main, labels, "Each detector with its own settings (A) and in one identical setup (B200: check every "
         "200, 2-fold; B50: every 50, 5-fold)", "fig1_detectors")
    comp = [f"S200 {x}" for x in ("sliding, threshold only", "sliding + ASPT", "reservoir, threshold only", "EADD",
                                  "reservoir without overlap, threshold only", "EADD, no overlap")]
    clabels = ["sliding reference, AUC threshold", "sliding reference + ASPT", "reservoir, AUC threshold",
               "reservoir + ASPT (= EADD)", "reservoir w/o overlap, AUC threshold", "reservoir w/o overlap + ASPT"]
    bars(summary, comp, clabels, "EADD's components one at a time (setup B200, tau = 0.7)", "fig2_components")
    # ASPT where it decides
    rows = summary[(summary.group == "D") & (summary.kind == "real (INSECTS)")].copy()
    rows["tau"] = rows.variant.str.extract(r"tau ([0-9.]+)").astype(float)
    rows["test"] = rows.variant.str.extract(r", (\w+)$")
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.6))
    for ax, (m, lab) in zip(axes, (("recall", "Recall"), ("fa", "False alarms per run"), ("seconds", "Seconds per run"))):
        for k, test in enumerate(("none", "aspt", "full")):
            r = rows[rows.test == test].sort_values("tau")
            ax.bar([t + (k - 1) * 0.025 for t in r.tau], r[m], width=0.025, color=OI[k],
                   label={"none": "no test", "aspt": "ASPT", "full": "full test (199 permutations)"}[test])
        ax.set_xticks([0.5, 0.6])
        ax.set_xlabel("AUC threshold tau")
        ax.set_title(lab, fontsize=9)
    axes[0].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(1.7, -0.3), ncol=3)
    fig.suptitle("ASPT where the permutation test decides (EADD, setup B200, INSECTS streams)", fontsize=10)
    fig.tight_layout()
    save(fig, "fig3_aspt")
    tl = ["D3 (authors)", "S200 D3 (logistic regression)", "S200 D3-LightGBM", "S200 EADD", "S200 sliding + ASPT"]
    tlab = ["D3, authors' settings", "D3 (logistic reg.), B200", "D3-LightGBM, B200", "EADD, B200",
            "EADD with sliding reference, B200"]
    timeline(df, "InsectsAbruptBalanced", 42, tl, tlab, "fig4_timeline_insects_abrupt",
             "Detections on INSECTS abrupt (seed 42)")
    timeline(df, "SynthGradual", 42, tl, tlab, "fig5_timeline_synth_gradual",
             "Detections on the synthetic gradual stream (seed 42; each change lasts 1,000 samples)")
    print("figures in", FIG)
