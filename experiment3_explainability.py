"""
EADD Experiments - Experiment 3: Explainability Case Study
==========================================================
Demonstrates EADD's SHAP-based root cause analysis on synthetic data
with controlled single-feature drift vs multi-feature drift.

Author: Nusrat Begum
Thesis: Feature Drift Detection via Adversarial Validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detectors.eadd import ExplainableAdversarialDriftDetector

# ──────────────────────────────────────────────────────────────
# Scenario Generators
# ──────────────────────────────────────────────────────────────

def generate_univariate_drift(n_samples=10000, n_features=10, drift_point=5000,
                               drift_feature=3, shift_magnitude=3.0, seed=42):
    """Only feature F3 drifts; all others remain stable."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, n_features))
    # Shift only feature `drift_feature` after drift_point
    X[drift_point:, drift_feature] += shift_magnitude
    return X, drift_point, f"F{drift_feature}"


def generate_subset_drift(n_samples=10000, n_features=10, drift_point=5000,
                           drift_features=[2, 5, 7], shift_magnitude=2.0, seed=42):
    """Features F2, F5, F7 drift together (correlated subset)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, n_features))
    for f in drift_features:
        X[drift_point:, f] += shift_magnitude
    return X, drift_point, [f"F{f}" for f in drift_features]


def generate_multivariate_drift(n_samples=10000, n_features=10, drift_point=5000,
                                 shift_magnitude=1.0, seed=42):
    """All features drift moderately (multivariate concept shift)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, n_features))
    for f in range(n_features):
        X[drift_point:, f] += rng.uniform(0.5, 1.5) * shift_magnitude
    return X, drift_point, [f"F{f}" for f in range(n_features)]


# ──────────────────────────────────────────────────────────────
# Experiment Runner
# ──────────────────────────────────────────────────────────────

def run_explainability_scenario(X, drift_point, scenario_name, seed=42):
    """Run EADD on a scenario and collect SHAP reports."""
    detector = ExplainableAdversarialDriftDetector(
        n_reference_samples=500,
        n_current_samples=200,
        auc_threshold=0.65,
        n_permutations=50,
        significance_level=0.05,
        monitoring_frequency=50,
        seed=seed,
    )

    detections = []
    reports = []

    for i in range(len(X)):
        features = {f"F{j}": float(X[i, j]) for j in range(X.shape[1])}
        is_drift = detector.update(features)
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

    return detections, reports


def run_experiment_3(output_dir="experiments/results"):
    """Run Experiment 3: Explainability Case Study."""
    os.makedirs(output_dir, exist_ok=True)

    scenarios = {}

    # Scenario A: Univariate drift on F3
    print("\n" + "="*60)
    print("  Scenario A: Univariate Drift (F3 only)")
    print("="*60)
    X_uni, dp_uni, target_uni = generate_univariate_drift()
    dets_uni, reports_uni = run_explainability_scenario(X_uni, dp_uni, "Univariate")
    print(f"  Detections: {len(dets_uni)} at positions {dets_uni[:5]}...")
    if reports_uni:
        r = reports_uni[0]
        print(f"  First detection AUC: {r['auc']:.3f}, p-value: {r['p_value']:.4f}")
        print(f"  Top features: {dict(list(r['feature_importances'].items())[:3])}")
        print(f"  Prescription: {r['prescription']['type']}")
        print(f"  Message: {r['prescription']['message']}")
    scenarios["univariate"] = {"detections": dets_uni, "reports": reports_uni, "target": target_uni}

    # Scenario B: Subset drift on F2, F5, F7
    print("\n" + "="*60)
    print("  Scenario B: Subset Drift (F2, F5, F7)")
    print("="*60)
    X_sub, dp_sub, target_sub = generate_subset_drift()
    dets_sub, reports_sub = run_explainability_scenario(X_sub, dp_sub, "Subset")
    print(f"  Detections: {len(dets_sub)} at positions {dets_sub[:5]}...")
    if reports_sub:
        r = reports_sub[0]
        print(f"  First detection AUC: {r['auc']:.3f}, p-value: {r['p_value']:.4f}")
        print(f"  Top features: {dict(list(r['feature_importances'].items())[:5])}")
        print(f"  Prescription: {r['prescription']['type']}")
        print(f"  Message: {r['prescription']['message']}")
    scenarios["subset"] = {"detections": dets_sub, "reports": reports_sub, "target": target_sub}

    # Scenario C: Multivariate drift (all features)
    print("\n" + "="*60)
    print("  Scenario C: Multivariate Drift (all features)")
    print("="*60)
    X_multi, dp_multi, target_multi = generate_multivariate_drift()
    dets_multi, reports_multi = run_explainability_scenario(X_multi, dp_multi, "Multivariate")
    print(f"  Detections: {len(dets_multi)} at positions {dets_multi[:5]}...")
    if reports_multi:
        r = reports_multi[0]
        print(f"  First detection AUC: {r['auc']:.3f}, p-value: {r['p_value']:.4f}")
        print(f"  Top features: {dict(list(r['feature_importances'].items())[:5])}")
        print(f"  Prescription: {r['prescription']['type']}")
        print(f"  Message: {r['prescription']['message']}")
    scenarios["multivariate"] = {"detections": dets_multi, "reports": reports_multi, "target": target_multi}

    # Save results
    results_summary = []
    for sname, sdata in scenarios.items():
        if sdata["reports"]:
            first_report = sdata["reports"][0]
            top_feature = list(first_report["feature_importances"].keys())[0]
            top_importance = list(first_report["feature_importances"].values())[0]
            results_summary.append({
                "scenario": sname,
                "n_detections": len(sdata["detections"]),
                "first_detection": sdata["detections"][0] if sdata["detections"] else None,
                "auc": first_report["auc"],
                "p_value": first_report["p_value"],
                "top_feature": top_feature,
                "top_importance_pct": round(top_importance, 1),
                "prescription_type": first_report["prescription"]["type"],
                "target": str(sdata["target"]),
            })

    df = pd.DataFrame(results_summary)
    df.to_csv(os.path.join(output_dir, "experiment3_explainability.csv"), index=False)

    # Save detailed reports as JSON
    serializable = {}
    for sname, sdata in scenarios.items():
        serializable[sname] = {
            "detections": sdata["detections"],
            "reports": [
                {
                    "step": r["step"],
                    "auc": r["auc"],
                    "p_value": r["p_value"],
                    "feature_importances": r["feature_importances"],
                    "prescription_type": r["prescription"]["type"],
                    "prescription_message": r["prescription"]["message"],
                }
                for r in sdata["reports"][:5]  # Top 5 reports
            ]
        }
    with open(os.path.join(output_dir, "experiment3_detailed_reports.json"), "w") as f:
        json.dump(serializable, f, indent=2)

    # Plot
    _plot_experiment3(scenarios, output_dir)

    print(f"\nResults saved to {output_dir}/experiment3_explainability.csv")
    return scenarios


def _plot_experiment3(scenarios, output_dir):
    """Generate SHAP-style feature importance plots for each scenario."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    scenario_names = ["univariate", "subset", "multivariate"]
    titles = [
        "Scenario A: Univariate Drift\n(F3 shifted)",
        "Scenario B: Subset Drift\n(F2, F5, F7 shifted)",
        "Scenario C: Multivariate Drift\n(All features shifted)"
    ]

    for ax, sname, title in zip(axes, scenario_names, titles):
        sdata = scenarios[sname]
        if sdata["reports"]:
            importances = sdata["reports"][0]["feature_importances"]
            features = list(importances.keys())
            values = list(importances.values())

            # Color: highlight drifting features
            colors = []
            if sname == "univariate":
                target_features = {"F3"}
            elif sname == "subset":
                target_features = {"F2", "F5", "F7"}
            else:
                target_features = set(features)

            for f in features:
                if f in target_features:
                    colors.append('#E53935')  # Red for drifting
                else:
                    colors.append('#90CAF9')  # Light blue for stable

            y_pos = np.arange(len(features))
            ax.barh(y_pos, values, color=colors, alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=9)
            ax.set_xlabel('SHAP Importance (%)')
            ax.set_title(title, fontsize=11)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)

            # Add prescription annotation
            prescription = sdata["reports"][0]["prescription"]["type"]
            ax.annotate(f'Prescription: {prescription}',
                        xy=(0.95, 0.02), xycoords='axes fraction',
                        ha='right', va='bottom', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "experiment3_explainability.png"), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "experiment3_explainability.pdf"), bbox_inches='tight')
    plt.close()
    print(f"  Plots saved to {output_dir}/experiment3_explainability.{{png,pdf}}")


if __name__ == "__main__":
    run_experiment_3()
