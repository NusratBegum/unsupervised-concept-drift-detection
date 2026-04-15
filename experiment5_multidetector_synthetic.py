"""
Experiment 5: Multi-Detector Comparison on Synthetic Data
==========================================================
Compares EADD against 7 state-of-the-art unsupervised drift detectors
on controlled synthetic streams with known ground-truth drift points.

Generates:
  - Bell-shaped distribution plots (before/after drift per feature)
  - Time-series detection comparison plots (all detectors overlaid)
  - Metrics tables (MTD, MDR, false alarms)
  - CSV results

Detectors: EADD, D3, BNDM, CSDDM, IBDD, OCDD, SPLL, UDetect

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
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detectors.eadd import ExplainableAdversarialDriftDetector
from detectors.d3 import DiscriminativeDriftDetector2019
from detectors.bndm import BayesianNonparametricDetectionMethod
from detectors.csddm import ClusteredStatisticalTestDriftDetectionMethod
from detectors.ibdd import ImageBasedDriftDetector
from detectors.ocdd import OneClassDriftDetector
from detectors.spll import SemiParametricLogLikelihood
from detectors.udetect import UDetect

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

FIGURE_DIR = "experiments/figures"
RESULTS_DIR = "experiments/results"
SEED = 42
N_SAMPLES = 5000
N_FEATURES = 5
DRIFT_POINT = 2500
MAX_SAMPLES = 5000

# Detector configurations with reasonable defaults
DETECTORS = {
    "EADD": (ExplainableAdversarialDriftDetector, {
        "n_reference_samples": 500,
        "n_current_samples": 200,
        "auc_threshold": 0.7,
        "n_permutations": 199,
        "significance_level": 0.05,
        "monitoring_frequency": 50,
        "seed": SEED,
    }),
    "D3": (DiscriminativeDriftDetector2019, {
        "n_reference_samples": 300,
        "recent_samples_proportion": 0.4,
        "threshold": 0.7,
        "seed": SEED,
    }),
    "BNDM": (BayesianNonparametricDetectionMethod, {
        "n_samples": 250,
        "const": 1.0,
        "threshold": 0.5,
        "max_depth": 3,
        "seed": SEED,
    }),
    "CSDDM": (ClusteredStatisticalTestDriftDetectionMethod, {
        "n_samples": 250,
        "n_clusters": 2,
        "confidence": 0.01,
        "feature_proportion": 0.1,
        "seed": SEED,
    }),
    "IBDD": (ImageBasedDriftDetector, {
        "n_samples": 250,
        "n_permutations": 20,
        "update_interval": 100,
        "n_consecutive_deviations": 1,
        "seed": SEED,
    }),
    "OCDD": (OneClassDriftDetector, {
        "n_samples": 250,
        "threshold": 0.3,
        "seed": SEED,
    }),
    "SPLL": (SemiParametricLogLikelihood, {
        "n_samples": 250,
        "n_clusters": 2,
        "threshold": 0.05,
        "seed": SEED,
    }),
    "UDetect": (UDetect, {
        "n_windows": 50,
        "n_samples": 100,
        "disjoint_training_windows": True,
        "seed": SEED,
    }),
}

DETECTOR_COLORS = {
    "EADD": "#E53935",
    "D3": "#1E88E5",
    "BNDM": "#43A047",
    "CSDDM": "#FB8C00",
    "IBDD": "#8E24AA",
    "OCDD": "#00ACC1",
    "SPLL": "#6D4C41",
    "UDetect": "#546E7A",
}

DETECTOR_MARKERS = {
    "EADD": "D",
    "D3": "s",
    "BNDM": "^",
    "CSDDM": "v",
    "IBDD": "o",
    "OCDD": "P",
    "SPLL": "X",
    "UDetect": "*",
}


# ──────────────────────────────────────────────────────────────
# Synthetic Data Generators
# ──────────────────────────────────────────────────────────────

def generate_abrupt_drift(n_samples=N_SAMPLES, n_features=N_FEATURES,
                          drift_point=DRIFT_POINT, shift=2.0, seed=SEED):
    rng = np.random.default_rng(seed)
    X_pre = rng.normal(0, 1, (drift_point, n_features))
    X_post = rng.normal(shift, 1, (n_samples - drift_point, n_features))
    return np.vstack([X_pre, X_post]), [drift_point]


def generate_gradual_drift(n_samples=N_SAMPLES, n_features=N_FEATURES,
                           drift_start=2000, drift_end=3000, shift=2.0, seed=SEED):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        if i < drift_start:
            X[i] = rng.normal(0, 1, n_features)
        elif i < drift_end:
            p = (i - drift_start) / (drift_end - drift_start)
            if rng.random() < p:
                X[i] = rng.normal(shift, 1, n_features)
            else:
                X[i] = rng.normal(0, 1, n_features)
        else:
            X[i] = rng.normal(shift, 1, n_features)
    return X, [drift_start]


def generate_incremental_drift(n_samples=N_SAMPLES, n_features=N_FEATURES,
                                drift_start=1500, drift_end=3500, shift=2.0, seed=SEED):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        if i < drift_start:
            mean = 0
        elif i < drift_end:
            mean = shift * (i - drift_start) / (drift_end - drift_start)
        else:
            mean = shift
        X[i] = rng.normal(mean, 1, n_features)
    return X, [drift_start]


def generate_recurring_drift(n_samples=N_SAMPLES, n_features=N_FEATURES,
                              period=1250, shift=2.0, seed=SEED):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, n_features))
    drifts = []
    for i in range(n_samples):
        cycle = (i // period) % 2
        X[i] = rng.normal(shift if cycle == 1 else 0, 1, n_features)
        if i > 0 and i % period == 0:
            drifts.append(i)
    return X, drifts


# ──────────────────────────────────────────────────────────────
# Single-Feature Drift (for targeted bell curves)
# ──────────────────────────────────────────────────────────────

def generate_single_feature_drift(n_samples=N_SAMPLES, n_features=N_FEATURES,
                                   drift_point=DRIFT_POINT, drift_feature=2,
                                   shift=3.0, seed=SEED):
    """Only feature `drift_feature` drifts; others remain stable."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, n_features))
    X[drift_point:, drift_feature] += shift
    return X, [drift_point], drift_feature


def generate_variance_drift(n_samples=N_SAMPLES, n_features=N_FEATURES,
                             drift_point=DRIFT_POINT, drift_feature=1,
                             variance_mult=3.0, seed=SEED):
    """Only variance changes in one feature."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, n_features))
    X[drift_point:, drift_feature] *= variance_mult
    return X, [drift_point], drift_feature


# ──────────────────────────────────────────────────────────────
# Detector Runner
# ──────────────────────────────────────────────────────────────

def run_detector(X, detector_name, detector_class, detector_kwargs, max_samples=MAX_SAMPLES):
    """Run a detector on a numpy array stream, return detections and timing."""
    detector = detector_class(**detector_kwargs)
    detections = []
    start = time.time()
    n = min(len(X), max_samples)

    for i in range(n):
        features = {f"F{j}": float(X[i, j]) for j in range(X.shape[1])}
        try:
            is_drift = detector.update(features)
            if is_drift:
                detections.append(i)
        except Exception as e:
            if i == 0:
                print(f"    {detector_name} error at step {i}: {e}")
            continue

    elapsed = time.time() - start
    return detections, elapsed


def compute_metrics(drift_points, detections, n_samples, tolerance=1500):
    """Compute detection delay, missed drifts, and false alarms."""
    if not drift_points or not detections:
        return {
            "mtd": np.nan,
            "mdr": 1.0 if drift_points else 0.0,
            "false_alarms": len(detections) if not drift_points else np.nan,
            "n_detections": len(detections),
        }

    # Detection delay: for each true drift, find first detection within tolerance
    delays = []
    matched = set()
    for dp in drift_points:
        candidates = [d for d in detections if dp <= d <= dp + tolerance and d not in matched]
        if candidates:
            delays.append(candidates[0] - dp)
            matched.add(candidates[0])

    # False alarms: detections not matched to any drift point
    true_det_positions = set()
    for dp in drift_points:
        for d in detections:
            if dp <= d <= dp + tolerance:
                true_det_positions.add(d)
                break
    false_alarms = len([d for d in detections if d not in true_det_positions])

    mtd = np.mean(delays) if delays else np.nan
    mdr = 1.0 - len(delays) / len(drift_points)

    return {
        "mtd": mtd,
        "mdr": mdr,
        "false_alarms": false_alarms,
        "n_detections": len(detections),
    }


# ──────────────────────────────────────────────────────────────
# Bell-Shaped Distribution Visualization
# ──────────────────────────────────────────────────────────────

def plot_bell_curves(X, drift_point, drift_name, feature_names=None, output_dir=FIGURE_DIR):
    """Plot bell-shaped (KDE) distributions for each feature, before and after drift."""
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    # Layout: 2 rows if >3 features
    n_cols = min(3, n_features)
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    before = X[:drift_point]
    after = X[drift_point:]

    for i in range(n_features):
        ax = axes[i]

        # KDE plots
        sns.kdeplot(before[:, i], ax=ax, color='#1E88E5', fill=True, alpha=0.3,
                    linewidth=2, label='Before Drift')
        sns.kdeplot(after[:, i], ax=ax, color='#E53935', fill=True, alpha=0.3,
                    linewidth=2, label='After Drift')

        ax.set_title(feature_names[i], fontsize=12, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        # Add statistics annotation
        mu_b, std_b = before[:, i].mean(), before[:, i].std()
        mu_a, std_a = after[:, i].mean(), after[:, i].std()
        stats_text = f"Before: μ={mu_b:.2f}, σ={std_b:.2f}\nAfter:  μ={mu_a:.2f}, σ={std_a:.2f}"
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # Hide unused axes
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'Feature Distributions — {drift_name}\n'
                 f'(Before vs After Drift Point at t={drift_point})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fname = f"bell_curves_{drift_name.lower().replace(' ', '_')}"
    plt.savefig(os.path.join(output_dir, f"{fname}.pdf"), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(output_dir, f"{fname}.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Bell curves saved: {fname}.pdf")


def plot_single_feature_bell_curves(X, drift_point, drift_feature, drift_name, output_dir=FIGURE_DIR):
    """Detailed bell curves for single-feature drift showing drifted vs stable features."""
    n_features = X.shape[1]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    before = X[:drift_point]
    after = X[drift_point:]

    for i in range(min(n_features, 6)):
        ax = axes[i]
        is_drifted = (i == drift_feature)

        sns.kdeplot(before[:, i], ax=ax, color='#1E88E5', fill=True, alpha=0.3,
                    linewidth=2, label='Before Drift')
        sns.kdeplot(after[:, i], ax=ax, color='#E53935', fill=True, alpha=0.3,
                    linewidth=2, label='After Drift')

        title = f"Feature F{i}"
        if is_drifted:
            title += " ★ DRIFTED"
            ax.set_facecolor('#FFF3E0')
        ax.set_title(title, fontsize=11, fontweight='bold',
                     color='#E53935' if is_drifted else 'black')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        mu_b, std_b = before[:, i].mean(), before[:, i].std()
        mu_a, std_a = after[:, i].mean(), after[:, i].std()
        stats_text = f"Before: μ={mu_b:.2f}, σ={std_b:.2f}\nAfter:  μ={mu_a:.2f}, σ={std_a:.2f}"
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    for i in range(min(n_features, 6), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'Feature Distribution Analysis — {drift_name}\n'
                 f'(Only F{drift_feature} drifts; others stable)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = f"bell_curves_single_feature_{drift_name.lower().replace(' ', '_')}"
    plt.savefig(os.path.join(output_dir, f"{fname}.pdf"), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(output_dir, f"{fname}.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Single-feature bell curves saved: {fname}.pdf")


# ──────────────────────────────────────────────────────────────
# Time-Series Detection Comparison
# ──────────────────────────────────────────────────────────────

def plot_timeseries_comparison(X, drift_points, all_detections, drift_name,
                               feature_idx=0, output_dir=FIGURE_DIR):
    """Plot time-series of a feature with detection markers from all detectors."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [2, 1]})

    # Top: Feature time series with detection markers
    ax1 = axes[0]
    # Subsample for plotting if needed
    n = len(X)
    step = max(1, n // 2000)
    t = np.arange(0, n, step)
    ax1.plot(t, X[::step, feature_idx], color='#78909C', alpha=0.5, linewidth=0.5,
             label=f'Feature F{feature_idx}')

    # Add rolling mean
    window = 200
    if len(X) > window:
        rolling_mean = pd.Series(X[:, feature_idx]).rolling(window=window, center=True).mean()
        ax1.plot(rolling_mean.index, rolling_mean.values, color='#37474F',
                 linewidth=1.5, label=f'Rolling Mean (w={window})')

    # Ground truth drift lines
    for dp in drift_points:
        ax1.axvline(x=dp, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(x=drift_points[0], color='black', linestyle='--', linewidth=2,
                alpha=0.7, label='True Drift')

    ax1.set_title(f'Detection Timeline — {drift_name}', fontsize=13, fontweight='bold')
    ax1.set_ylabel(f'Feature F{feature_idx} Value')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.2)

    # Bottom: Detection events per detector (event plot)
    ax2 = axes[1]
    detector_names = list(all_detections.keys())
    y_positions = list(range(len(detector_names)))

    for idx, (det_name, dets) in enumerate(all_detections.items()):
        color = DETECTOR_COLORS.get(det_name, '#999999')
        if dets:
            ax2.eventplot([dets], lineoffsets=idx, linelengths=0.6,
                         colors=[color], linewidths=1.5)

    # Ground truth drift lines on event plot
    for dp in drift_points:
        ax2.axvline(x=dp, color='black', linestyle='--', linewidth=2, alpha=0.7)

    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(detector_names, fontsize=10)
    ax2.set_xlabel('Time Step', fontsize=11)
    ax2.set_ylabel('Detector')
    ax2.set_title('Detection Events by Detector', fontsize=11)
    ax2.set_xlim(0, len(X))
    ax2.grid(True, axis='x', alpha=0.2)

    # Color legend
    handles = [mpatches.Patch(color=DETECTOR_COLORS.get(d, '#999'), label=d)
               for d in detector_names]
    handles.append(mpatches.Patch(color='black', label='True Drift', linestyle='--'))
    ax2.legend(handles=handles, loc='upper right', fontsize=7, ncol=3)

    plt.tight_layout()
    fname = f"timeseries_{drift_name.lower().replace(' ', '_')}"
    plt.savefig(os.path.join(output_dir, f"{fname}.pdf"), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(output_dir, f"{fname}.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Time-series plot saved: {fname}.pdf")


# ──────────────────────────────────────────────────────────────
# Summary Heatmap
# ──────────────────────────────────────────────────────────────

def plot_metrics_heatmap(results_df, metric, title, output_dir=FIGURE_DIR):
    """Plot a heatmap of a given metric across detectors and drift types."""
    pivot = results_df.pivot_table(index='detector', columns='drift_type', values=metric)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = 'RdYlGn_r' if metric in ['mtd', 'mdr', 'false_alarms'] else 'RdYlGn'
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap, ax=ax,
                linewidths=0.5, cbar_kws={'label': metric.upper()})
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Drift Type')
    ax.set_ylabel('Detector')
    plt.tight_layout()
    fname = f"heatmap_{metric}"
    plt.savefig(os.path.join(output_dir, f"{fname}.pdf"), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(output_dir, f"{fname}.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Heatmap saved: {fname}.pdf")


# ──────────────────────────────────────────────────────────────
# Main Experiment
# ──────────────────────────────────────────────────────────────

def run_experiment_5():
    """Run comprehensive multi-detector synthetic experiment."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    drift_generators = {
        "Abrupt Drift":      (generate_abrupt_drift, {"shift": 2.0}, 1000),
        "Gradual Drift":     (generate_gradual_drift, {"shift": 2.0}, 2000),
        "Incremental Drift": (generate_incremental_drift, {"shift": 2.0}, 3000),
        "Recurring Drift":   (generate_recurring_drift, {"shift": 2.0}, 1000),
    }

    all_results = []

    # ── Part A: Multi-drift-type comparison ──
    print("\n" + "=" * 70)
    print("  EXPERIMENT 5A: Multi-Detector Comparison on Synthetic Drift Patterns")
    print("=" * 70)

    for drift_name, (gen_fn, gen_kwargs, tolerance) in drift_generators.items():
        print(f"\n{'─' * 60}")
        print(f"  {drift_name}")
        print(f"{'─' * 60}")

        X, drift_points = gen_fn(**gen_kwargs)

        # Generate bell-curve distributions
        plot_bell_curves(X, drift_points[0], drift_name)

        # Run all detectors
        all_detections = {}
        for det_name, (det_class, det_kwargs) in DETECTORS.items():
            print(f"  Running {det_name}...", end=" ", flush=True)
            try:
                dets, elapsed = run_detector(X, det_name, det_class, det_kwargs)
                print(f"{len(dets)} detections in {elapsed:.1f}s")
                all_detections[det_name] = dets

                metrics = compute_metrics(drift_points, dets, len(X), tolerance)
                all_results.append({
                    "drift_type": drift_name,
                    "detector": det_name,
                    "n_detections": metrics["n_detections"],
                    "mtd": metrics["mtd"],
                    "mdr": metrics["mdr"],
                    "false_alarms": metrics["false_alarms"],
                    "time_s": round(elapsed, 2),
                })
            except Exception as e:
                print(f"ERROR: {e}")
                all_detections[det_name] = []
                all_results.append({
                    "drift_type": drift_name,
                    "detector": det_name,
                    "n_detections": 0,
                    "mtd": np.nan,
                    "mdr": 1.0,
                    "false_alarms": 0,
                    "time_s": 0,
                })

        # Generate time-series comparison plot
        plot_timeseries_comparison(X, drift_points, all_detections, drift_name)

    # ── Part B: Single-Feature Drift (Bell Curve Focus) ──
    print("\n" + "=" * 70)
    print("  EXPERIMENT 5B: Single-Feature Drift Distribution Analysis")
    print("=" * 70)

    # Mean shift on F2 only
    X_mean, dp_mean, df_mean = generate_single_feature_drift(
        drift_feature=2, shift=3.0)
    plot_single_feature_bell_curves(X_mean, dp_mean[0], df_mean,
                                    "Mean Shift (F2)")

    # Variance change on F1 only
    X_var, dp_var, df_var = generate_variance_drift(
        drift_feature=1, variance_mult=3.0)
    plot_single_feature_bell_curves(X_var, dp_var[0], df_var,
                                    "Variance Change (F1)")

    # ── Part C: Save results and generate summary ──
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(RESULTS_DIR, "experiment5_multidetector_synthetic.csv"), index=False)
    print(f"\nResults saved to {RESULTS_DIR}/experiment5_multidetector_synthetic.csv")

    # Generate heatmaps
    plot_metrics_heatmap(df, "mtd", "Mean Time to Detection (MTD) — Lower is Better")
    plot_metrics_heatmap(df, "mdr", "Missed Detection Rate (MDR) — Lower is Better")
    plot_metrics_heatmap(df, "false_alarms", "False Alarms — Lower is Better")

    # Print summary table
    print("\n" + "=" * 70)
    print("  SUMMARY: Synthetic Experiment Results")
    print("=" * 70)
    summary = df.pivot_table(
        index='detector',
        columns='drift_type',
        values=['mtd', 'mdr', 'false_alarms'],
        aggfunc='mean'
    )
    print(summary.to_string())

    # LaTeX table
    _generate_latex_table(df)

    return df


def _generate_latex_table(df):
    """Generate a LaTeX-ready comparison table."""
    output = []
    output.append(r"\begin{table}[htbp]")
    output.append(r"\centering")
    output.append(r"\caption{Multi-Detector Comparison on Synthetic Data}")
    output.append(r"\label{tab:exp5_synthetic}")
    output.append(r"\small")
    output.append(r"\begin{tabular}{l|rrrr|rrrr}")
    output.append(r"\hline")
    output.append(r"& \multicolumn{4}{c|}{Mean Time to Detection (MTD) $\downarrow$} "
                  r"& \multicolumn{4}{c}{Missed Detection Rate (MDR) $\downarrow$} \\")
    output.append(r"Detector & Abrupt & Gradual & Increm. & Recur. "
                  r"& Abrupt & Gradual & Increm. & Recur. \\")
    output.append(r"\hline")

    drift_types = ["Abrupt Drift", "Gradual Drift", "Incremental Drift", "Recurring Drift"]
    for det_name in DETECTORS.keys():
        row = det_name
        for metric in ["mtd", "mdr"]:
            for dt in drift_types:
                val = df[(df['detector'] == det_name) & (df['drift_type'] == dt)][metric].values
                if len(val) > 0 and not np.isnan(val[0]):
                    if metric == "mtd":
                        row += f" & {val[0]:.0f}"
                    else:
                        row += f" & {val[0]:.2f}"
                else:
                    row += " & --"
        row += r" \\"
        output.append(row)

    output.append(r"\hline")
    output.append(r"\end{tabular}")
    output.append(r"\end{table}")

    latex = "\n".join(output)
    with open(os.path.join(RESULTS_DIR, "experiment5_latex_table.tex"), "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to {RESULTS_DIR}/experiment5_latex_table.tex")


if __name__ == "__main__":
    run_experiment_5()
