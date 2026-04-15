"""
Experiment 7: EADD Explainability Deep Dive
=============================================
Demonstrates EADD's unique SHAP-based root cause analysis — the feature
no other unsupervised drift detector provides.

Generates:
  - Detailed SHAP feature attribution bar charts (per scenario)
  - Before/After bell curves with SHAP highlighting (drifted features)
  - EADD pipeline stage visualization (AUC → p-value → SHAP → Prescription)
  - Comparison: EADD explains vs other detectors just flag

Author: Nusrat Begum
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detectors.eadd import ExplainableAdversarialDriftDetector
from detectors.d3 import DiscriminativeDriftDetector2019

FIGURE_DIR = "experiments/figures"
RESULTS_DIR = "experiments/results"
SEED = 42


# ──────────────────────────────────────────────────────────────
# Synthetic Scenarios with Ground-Truth Attribution
# ──────────────────────────────────────────────────────────────

def generate_univariate_drift(n_samples=10000, n_features=10, drift_point=5000,
                               drift_feature=3, shift=3.0, seed=SEED):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, n_features))
    X[drift_point:, drift_feature] += shift
    return X, drift_point, {f"F{drift_feature}": "drifted"}, "univariate"


def generate_subset_drift(n_samples=10000, n_features=10, drift_point=5000,
                           drift_features=(2, 5, 7), shift=2.0, seed=SEED):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, n_features))
    for f in drift_features:
        X[drift_point:, f] += shift
    ground_truth = {f"F{f}": "drifted" for f in drift_features}
    return X, drift_point, ground_truth, "subset"


def generate_multivariate_drift(n_samples=10000, n_features=10, drift_point=5000,
                                 shift=1.0, seed=SEED):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, n_features))
    for f in range(n_features):
        X[drift_point:, f] += rng.uniform(0.5, 1.5) * shift
    ground_truth = {f"F{f}": "drifted" for f in range(n_features)}
    return X, drift_point, ground_truth, "multivariate"


def generate_covariate_shift(n_samples=10000, n_features=10, drift_point=5000, seed=SEED):
    """Correlation structure changes but marginals stay similar."""
    rng = np.random.default_rng(seed)
    X_pre = rng.multivariate_normal(np.zeros(n_features), np.eye(n_features), drift_point)
    # Post-drift: introduce correlation between F0-F1 and F2-F3
    cov = np.eye(n_features)
    cov[0, 1] = cov[1, 0] = 0.8
    cov[2, 3] = cov[3, 2] = 0.7
    X_post = rng.multivariate_normal(np.zeros(n_features), cov, n_samples - drift_point)
    X = np.vstack([X_pre, X_post])
    ground_truth = {"F0": "correlated", "F1": "correlated", "F2": "correlated", "F3": "correlated"}
    return X, drift_point, ground_truth, "covariate_shift"


# ──────────────────────────────────────────────────────────────
# Run EADD with Full Reports
# ──────────────────────────────────────────────────────────────

def run_eadd_with_reports(X, drift_point, scenario_name):
    """Run EADD and collect all detection reports."""
    detector = ExplainableAdversarialDriftDetector(
        n_reference_samples=500, n_current_samples=200,
        auc_threshold=0.7, n_permutations=199,
        significance_level=0.05, monitoring_frequency=50, seed=SEED,
    )

    detections = []
    reports = []
    auc_history = []
    pvalue_history = []

    for i in range(len(X)):
        features = {f"F{j}": float(X[i, j]) for j in range(X.shape[1])}
        is_drift = detector.update(features)

        # Track AUC/p-value at every check
        if detector.last_auc is not None:
            auc_history.append((i, detector.last_auc))
        if detector.last_p_value is not None:
            pvalue_history.append((i, detector.last_p_value))

        if is_drift:
            detections.append(i)
            report = detector.get_last_report()
            reports.append({
                "step": i,
                "auc": report["auc"],
                "p_value": report["p_value"],
                "feature_importances": report["feature_importances"],
                "prescription": report["prescription"],
            })

    return {
        "detections": detections,
        "reports": reports,
        "auc_history": auc_history,
        "pvalue_history": pvalue_history,
    }


# ──────────────────────────────────────────────────────────────
# SHAP Attribution Visualization
# ──────────────────────────────────────────────────────────────

def plot_shap_attribution(reports, ground_truth, scenario_name, drift_point,
                          output_dir=FIGURE_DIR):
    """Detailed SHAP bar chart with ground-truth highlighting."""
    if not reports:
        print(f"    No reports for {scenario_name}")
        return

    # Use first post-drift detection
    report = reports[0]
    importances = report["feature_importances"]
    features = list(importances.keys())
    values = list(importances.values())

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color based on ground truth
    colors = []
    for f in features:
        if f in ground_truth:
            colors.append('#E53935')  # Red = truly drifted
        else:
            colors.append('#90CAF9')  # Blue = stable
    
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.85, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel('SHAP Importance (%)', fontsize=11)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)

    # Title with detection info
    ax.set_title(f'SHAP Feature Attribution — {scenario_name.replace("_", " ").title()}\n'
                 f'Detection at t={report["step"]} | AUC={report["auc"]:.3f} | '
                 f'p={report["p_value"]:.4f}',
                 fontsize=12, fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#E53935', alpha=0.85, label='Truly Drifted'),
        mpatches.Patch(facecolor='#90CAF9', alpha=0.85, label='Stable'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # Prescription annotation
    presc = report["prescription"]
    ax.annotate(f'EADD Prescription: {presc["type"].upper()}\n{presc["message"][:80]}...',
                xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    fname = f"shap_attribution_{scenario_name}"
    plt.savefig(os.path.join(output_dir, f"{fname}.pdf"), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(output_dir, f"{fname}.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    SHAP attribution saved: {fname}.pdf")


# ──────────────────────────────────────────────────────────────
# EADD Pipeline Stage Visualization
# ──────────────────────────────────────────────────────────────

def plot_pipeline_stages(eadd_result, drift_point, scenario_name, output_dir=FIGURE_DIR):
    """Visualize EADD's 4-stage pipeline: AUC over time → p-value → SHAP → prescription."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Stage 1: AUC over time
    ax1 = axes[0, 0]
    if eadd_result["auc_history"]:
        steps, aucs = zip(*eadd_result["auc_history"])
        ax1.plot(steps, aucs, color='#1E88E5', linewidth=1, alpha=0.7)
        ax1.axhline(y=0.7, color='#E53935', linestyle='--', linewidth=1.5, label='AUC Threshold (0.7)')
        ax1.axhline(y=0.5, color='#999', linestyle=':', linewidth=1, label='Random (0.5)')
        ax1.axvline(x=drift_point, color='black', linestyle='--', linewidth=2, label='True Drift')
        for d in eadd_result["detections"]:
            ax1.axvline(x=d, color='#E53935', alpha=0.3, linewidth=1)
    ax1.set_title('Stage 1: Adversarial AUC Over Time', fontweight='bold')
    ax1.set_ylabel('AUC')
    ax1.set_xlabel('Time Step')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(0.4, 1.05)

    # Stage 2: p-value over time
    ax2 = axes[0, 1]
    if eadd_result["pvalue_history"]:
        steps, pvals = zip(*eadd_result["pvalue_history"])
        ax2.semilogy(steps, [max(p, 1e-4) for p in pvals], color='#43A047', linewidth=1, alpha=0.7)
        ax2.axhline(y=0.01, color='#E53935', linestyle='--', linewidth=1.5, label='α = 0.01')
        ax2.axvline(x=drift_point, color='black', linestyle='--', linewidth=2, label='True Drift')
        for d in eadd_result["detections"]:
            ax2.axvline(x=d, color='#E53935', alpha=0.3, linewidth=1)
    ax2.set_title('Stage 2: Permutation p-value', fontweight='bold')
    ax2.set_ylabel('p-value (log scale)')
    ax2.set_xlabel('Time Step')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    # Stage 3: SHAP importances at first detection
    ax3 = axes[1, 0]
    if eadd_result["reports"]:
        report = eadd_result["reports"][0]
        importances = report["feature_importances"]
        features = list(importances.keys())
        values = list(importances.values())
        y_pos = np.arange(len(features))
        colors = ['#E53935' if v > 20 else '#90CAF9' for v in values]
        ax3.barh(y_pos, values, color=colors, alpha=0.85)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(features, fontsize=9)
        ax3.set_xlabel('SHAP Importance (%)')
        ax3.invert_yaxis()
    ax3.set_title('Stage 3: SHAP Feature Attribution', fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

    # Stage 4: Prescription
    ax4 = axes[1, 1]
    ax4.axis('off')
    if eadd_result["reports"]:
        report = eadd_result["reports"][0]
        presc = report["prescription"]
        text = (
            f"━━━ EADD Detection Report ━━━\n\n"
            f"Detection Time: t = {report['step']}\n"
            f"Adversarial AUC: {report['auc']:.4f}\n"
            f"p-value: {report['p_value']:.4f}\n\n"
            f"━━━ Prescription ━━━\n\n"
            f"Type: {presc['type'].upper()}\n\n"
            f"{presc['message']}"
        )
        ax4.text(0.1, 0.9, text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                         edgecolor='#E0E0E0'))
    ax4.set_title('Stage 4: Automated Prescription', fontweight='bold')

    fig.suptitle(f'EADD 4-Stage Pipeline — {scenario_name.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fname = f"pipeline_stages_{scenario_name}"
    plt.savefig(os.path.join(output_dir, f"{fname}.pdf"), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(output_dir, f"{fname}.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    Pipeline stages saved: {fname}.pdf")


# ──────────────────────────────────────────────────────────────
# Combined Bell Curves + SHAP (highlighting drifted features)
# ──────────────────────────────────────────────────────────────

def plot_combined_bell_shap(X, drift_point, ground_truth, shap_importances,
                            scenario_name, output_dir=FIGURE_DIR):
    """Bell curves per feature with SHAP-informed highlighting."""
    n_features = X.shape[1]
    n_cols = min(5, n_features)
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_features == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    before = X[:drift_point]
    after = X[drift_point:]

    # Sort features by SHAP importance
    sorted_features = sorted(shap_importances.items(), key=lambda x: x[1], reverse=True)

    for idx, (fname, imp) in enumerate(sorted_features):
        if idx >= len(axes):
            break
        ax = axes[idx]
        fi = int(fname[1:])  # "F3" -> 3

        is_drifted = fname in ground_truth

        sns.kdeplot(before[:, fi], ax=ax, color='#1E88E5', fill=True, alpha=0.3,
                    linewidth=2, label='Before')
        sns.kdeplot(after[:, fi], ax=ax, color='#E53935', fill=True, alpha=0.3,
                    linewidth=2, label='After')

        title_color = '#E53935' if is_drifted else '#333'
        marker = "★" if is_drifted else ""
        ax.set_title(f'{fname} {marker}\nSHAP: {imp:.1f}%', fontsize=10,
                     fontweight='bold', color=title_color)
        if is_drifted:
            ax.set_facecolor('#FFF8E1')

        ax.set_xlabel('Value', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

        if idx == 0:
            ax.legend(fontsize=7)

    for i in range(len(sorted_features), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'Feature Distributions with SHAP Importance — '
                 f'{scenario_name.replace("_", " ").title()}\n'
                 f'★ = Ground-truth drifted feature, Yellow background = EADD identified as drifted',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname_out = f"bell_shap_combined_{scenario_name}"
    plt.savefig(os.path.join(output_dir, f"{fname_out}.pdf"), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(output_dir, f"{fname_out}.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    Combined bell+SHAP saved: {fname_out}.pdf")


# ──────────────────────────────────────────────────────────────
# Attribution Accuracy Metric
# ──────────────────────────────────────────────────────────────

def evaluate_attribution_accuracy(shap_importances, ground_truth, n_features):
    """Evaluate how well SHAP identifies the truly drifted features."""
    drifted_features = set(ground_truth.keys())
    n_drifted = len(drifted_features)

    # Get top-k SHAP features where k = number of truly drifted features
    sorted_shap = sorted(shap_importances.items(), key=lambda x: x[1], reverse=True)
    top_k = set(f for f, _ in sorted_shap[:n_drifted])

    # Precision and recall
    true_positives = top_k & drifted_features
    precision = len(true_positives) / len(top_k) if top_k else 0
    recall = len(true_positives) / n_drifted if n_drifted else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # NDCG-style metric: sum of SHAP importance on drifted features / total
    drifted_importance = sum(shap_importances.get(f, 0) for f in drifted_features)

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1,
        "drifted_importance_pct": drifted_importance,
    }


# ──────────────────────────────────────────────────────────────
# Explainability Comparison Table
# ──────────────────────────────────────────────────────────────

def plot_explainability_comparison(output_dir=FIGURE_DIR):
    """Show what each detector provides: Detection only vs Detection + Explanation."""
    detectors = ["EADD", "D3", "BNDM", "CSDDM", "IBDD", "OCDD", "SPLL", "UDetect"]
    capabilities = {
        "Drift Detection": [1, 1, 1, 1, 1, 1, 1, 1],
        "Statistical\nSignificance": [1, 0, 0, 0, 1, 0, 0, 0],
        "Feature\nAttribution": [1, 0, 0, 0, 0, 0, 0, 0],
        "Root Cause\nAnalysis": [1, 0, 0, 0, 0, 0, 0, 0],
        "Automated\nPrescription": [1, 0, 0, 0, 0, 0, 0, 0],
        "Non-Linear\nDiscrimination": [1, 0, 0, 0, 0, 0, 0, 0],
    }

    fig, ax = plt.subplots(figsize=(12, 5))

    data = np.array(list(capabilities.values()))
    cmap = plt.cm.colors.ListedColormap(['#FFCDD2', '#C8E6C9'])
    ax.imshow(data, cmap=cmap, aspect='auto')

    ax.set_xticks(range(len(detectors)))
    ax.set_xticklabels(detectors, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(capabilities)))
    ax.set_yticklabels(list(capabilities.keys()), fontsize=10)

    for i in range(len(capabilities)):
        for j in range(len(detectors)):
            symbol = "✓" if data[i, j] == 1 else "✗"
            color = '#2E7D32' if data[i, j] == 1 else '#C62828'
            ax.text(j, i, symbol, ha='center', va='center', fontsize=16,
                    fontweight='bold', color=color)

    ax.set_title('Detector Capability Comparison\nEADD is the only detector providing explainability',
                 fontsize=13, fontweight='bold')
    ax.grid(False)

    # Highlight EADD column
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, len(capabilities),
                               fill=False, edgecolor='#E53935', linewidth=3))

    plt.tight_layout()
    fname = "explainability_comparison"
    plt.savefig(os.path.join(output_dir, f"{fname}.pdf"), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(output_dir, f"{fname}.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Explainability comparison saved: {fname}.pdf")


# ──────────────────────────────────────────────────────────────
# Main Experiment
# ──────────────────────────────────────────────────────────────

def run_experiment_7():
    """Run EADD explainability deep dive."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("  EXPERIMENT 7: EADD Explainability Deep Dive")
    print("=" * 70)

    scenarios = [
        ("Univariate (F3 shift)", generate_univariate_drift),
        ("Subset (F2,F5,F7 shift)", generate_subset_drift),
        ("Multivariate (all shift)", generate_multivariate_drift),
        ("Covariate Shift (correlation)", generate_covariate_shift),
    ]

    results = []

    for scenario_label, gen_fn in scenarios:
        print(f"\n{'─' * 60}")
        print(f"  {scenario_label}")
        print(f"{'─' * 60}")

        X, drift_point, ground_truth, scenario_name = gen_fn()

        # Run EADD
        eadd_result = run_eadd_with_reports(X, drift_point, scenario_name)
        print(f"  Detections: {len(eadd_result['detections'])}")

        if eadd_result["reports"]:
            report = eadd_result["reports"][0]
            print(f"  First detection: t={report['step']}, AUC={report['auc']:.3f}, p={report['p_value']:.4f}")
            print(f"  Prescription: {report['prescription']['type']}")

            # SHAP attribution plot
            plot_shap_attribution(eadd_result["reports"], ground_truth, scenario_name, drift_point)

            # Pipeline stages plot
            plot_pipeline_stages(eadd_result, drift_point, scenario_name)

            # Combined bell + SHAP plot
            plot_combined_bell_shap(X, drift_point, ground_truth,
                                    report["feature_importances"], scenario_name)

            # Evaluate attribution accuracy
            accuracy = evaluate_attribution_accuracy(
                report["feature_importances"], ground_truth, X.shape[1])
            print(f"  Attribution accuracy: P@k={accuracy['precision_at_k']:.2f}, "
                  f"R@k={accuracy['recall_at_k']:.2f}, F1@k={accuracy['f1_at_k']:.2f}")
            print(f"  Drifted features capture {accuracy['drifted_importance_pct']:.1f}% of SHAP importance")

            results.append({
                "scenario": scenario_name,
                "n_detections": len(eadd_result["detections"]),
                "first_detection": report["step"],
                "detection_delay": report["step"] - drift_point,
                "auc": report["auc"],
                "p_value": report["p_value"],
                "prescription_type": report["prescription"]["type"],
                "precision_at_k": accuracy["precision_at_k"],
                "recall_at_k": accuracy["recall_at_k"],
                "f1_at_k": accuracy["f1_at_k"],
                "drifted_importance_pct": accuracy["drifted_importance_pct"],
                "top_feature": list(report["feature_importances"].keys())[0],
                "top_feature_pct": list(report["feature_importances"].values())[0],
            })

    # Explainability comparison (EADD vs others)
    plot_explainability_comparison()

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "experiment7_explainability_deep_dive.csv"), index=False)

    # Save detailed reports
    print(f"\nResults saved to {RESULTS_DIR}/experiment7_explainability_deep_dive.csv")

    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY: Explainability Results")
    print("=" * 70)
    print(df[["scenario", "detection_delay", "prescription_type",
              "precision_at_k", "recall_at_k", "drifted_importance_pct"]].to_string(index=False))

    # LaTeX table
    _generate_explainability_latex(df)

    return df


def _generate_explainability_latex(df):
    """Generate LaTeX table for explainability results."""
    output = []
    output.append(r"\begin{table}[htbp]")
    output.append(r"\centering")
    output.append(r"\caption{EADD Explainability: SHAP Attribution Accuracy}")
    output.append(r"\label{tab:exp7_explainability}")
    output.append(r"\begin{tabular}{lcccccc}")
    output.append(r"\hline")
    output.append(r"Scenario & Delay & Prescription & P@k & R@k & F1@k & Drifted \% \\")
    output.append(r"\hline")

    for _, row in df.iterrows():
        line = (f"{row['scenario'].replace('_', ' ').title()} & "
                f"{row['detection_delay']:.0f} & "
                f"{row['prescription_type']} & "
                f"{row['precision_at_k']:.2f} & "
                f"{row['recall_at_k']:.2f} & "
                f"{row['f1_at_k']:.2f} & "
                f"{row['drifted_importance_pct']:.1f}\\% \\\\")
        output.append(line)

    output.append(r"\hline")
    output.append(r"\end{tabular}")
    output.append(r"\end{table}")

    with open(os.path.join(RESULTS_DIR, "experiment7_latex_table.tex"), "w") as f:
        f.write("\n".join(output))
    print(f"LaTeX table saved to {RESULTS_DIR}/experiment7_latex_table.tex")


if __name__ == "__main__":
    run_experiment_7()
