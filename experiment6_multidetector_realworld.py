"""
Experiment 6: Multi-Detector Comparison on Real-World Datasets
===============================================================
Compares EADD against 7 state-of-the-art unsupervised drift detectors
on real-world benchmark datasets from Lukats et al. (2024).

Generates:
  - Time-series detection comparison plots per dataset
  - Comparison heatmap (detectors × datasets × metrics)
  - Bell-curve distributions for real-world features at detected drift points
  - CSV + LaTeX tables

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detectors.eadd import ExplainableAdversarialDriftDetector
from detectors.d3 import DiscriminativeDriftDetector2019
from detectors.bndm import BayesianNonparametricDetectionMethod
from detectors.csddm import ClusteredStatisticalTestDriftDetectionMethod
from detectors.ibdd import ImageBasedDriftDetector
from detectors.ocdd import OneClassDriftDetector
from detectors.spll import SemiParametricLogLikelihood
from detectors.udetect import UDetect
from metrics.drift import calculate_drift_metrics

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

FIGURE_DIR = "experiments/figures"
RESULTS_DIR = "experiments/results"
SEED = 42
MAX_SAMPLES = 20000
DETECTOR_TIMEOUT = 120  # seconds per detector per dataset

DETECTORS = {
    "EADD": (ExplainableAdversarialDriftDetector, {
        "n_reference_samples": 500,
        "n_current_samples": 200,
        "auc_threshold": 0.7,
        "n_permutations": 50,
        "significance_level": 0.01,
        "monitoring_frequency": 100,
        "seed": SEED,
    }),
    "D3": (DiscriminativeDriftDetector2019, {
        "n_reference_samples": 500,
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
    "EADD": "#E53935", "D3": "#1E88E5", "BNDM": "#43A047",
    "CSDDM": "#FB8C00", "IBDD": "#8E24AA", "OCDD": "#00ACC1",
    "SPLL": "#6D4C41", "UDetect": "#546E7A",
}


# ──────────────────────────────────────────────────────────────
# Dataset Loading
# ──────────────────────────────────────────────────────────────

def load_datasets():
    """Load real-world benchmark datasets with ground-truth drifts."""
    from datasets import (
        Electricity,
        InsectsAbruptBalanced,
        InsectsGradualBalanced,
        InsectsIncrementalAbruptBalanced,
        InsectsIncrementalBalanced,
        InsectsIncrementalReoccurringBalanced,
        NOAAWeather,
        OutdoorObjects,
        Powersupply,
        SineClusters,
        WaveformDrift2,
    )

    return {
        "Electricity": Electricity(),
        "InsectsAbrupt": InsectsAbruptBalanced(),
        "InsectsGradual": InsectsGradualBalanced(),
        "InsectsIncrAbrupt": InsectsIncrementalAbruptBalanced(),
        "InsectsIncremental": InsectsIncrementalBalanced(),
        "InsectsReoccurring": InsectsIncrementalReoccurringBalanced(),
        "NOAAWeather": NOAAWeather(),
        "OutdoorObjects": OutdoorObjects(),
        "Powersupply": Powersupply(),
        "SineClusters": SineClusters(drift_frequency=5000, stream_length=50000, seed=531874),
        "WaveformDrift2": WaveformDrift2(drift_frequency=5000, stream_length=50000, seed=2401137),
    }


def stream_to_arrays(stream, max_samples=MAX_SAMPLES):
    """Convert a River-style stream to numpy arrays."""
    X_list, y_list = [], []
    for i, (x, y) in enumerate(stream):
        if i >= max_samples:
            break
        X_list.append(list(x.values()))
        y_list.append(y)
    feature_names = list(X_list[0]) if X_list else []
    # Get actual feature names from the stream
    return np.array(X_list, dtype=float), np.array(y_list)


def run_detector_on_array(X, det_name, det_class, det_kwargs, timeout=DETECTOR_TIMEOUT):
    """Run detector on numpy array, return detections and timing. Aborts if timeout exceeded."""
    detector = det_class(**det_kwargs)
    detections = []
    start = time.time()
    timed_out = False
    for i in range(len(X)):
        if time.time() - start > timeout:
            timed_out = True
            break
        features = {f"F{j}": float(X[i, j]) for j in range(X.shape[1])}
        try:
            if detector.update(features):
                detections.append(i)
        except Exception:
            continue
    elapsed = time.time() - start
    return detections, elapsed, detector, timed_out


# ──────────────────────────────────────────────────────────────
# Visualizations
# ──────────────────────────────────────────────────────────────

def plot_realworld_timeseries(X, known_drifts, all_detections, ds_name, output_dir=FIGURE_DIR):
    """Time-series + event plot for a real-world dataset."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [2, 1]})

    ax1 = axes[0]
    n = len(X)
    step = max(1, n // 2000)
    t = np.arange(0, n, step)

    # Plot first 3 features
    n_feat_plot = min(3, X.shape[1])
    feat_colors = ['#78909C', '#B0BEC5', '#CFD8DC']
    for fi in range(n_feat_plot):
        ax1.plot(t, X[::step, fi], color=feat_colors[fi], alpha=0.4, linewidth=0.5)

    # Rolling mean of first feature
    window = min(200, n // 10)
    if n > window:
        rm = pd.Series(X[:, 0]).rolling(window=window, center=True).mean()
        ax1.plot(rm.index, rm.values, color='#37474F', linewidth=1.5,
                 label=f'F0 rolling mean')

    # Ground truth
    for dp in known_drifts:
        if dp < n:
            ax1.axvline(x=dp, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    if known_drifts:
        ax1.axvline(x=known_drifts[0], color='black', linestyle='--', linewidth=1.5,
                     alpha=0.6, label='True Drift')

    ax1.set_title(f'Detection Timeline — {ds_name}', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Feature Values')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.2)

    # Event plot
    ax2 = axes[1]
    det_names = list(all_detections.keys())
    for idx, (dname, dets) in enumerate(all_detections.items()):
        color = DETECTOR_COLORS.get(dname, '#999')
        if dets:
            ax2.eventplot([dets], lineoffsets=idx, linelengths=0.6,
                         colors=[color], linewidths=1.5)

    for dp in known_drifts:
        if dp < n:
            ax2.axvline(x=dp, color='black', linestyle='--', linewidth=1.5, alpha=0.6)

    ax2.set_yticks(range(len(det_names)))
    ax2.set_yticklabels(det_names, fontsize=10)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Detector')
    ax2.set_xlim(0, n)
    ax2.grid(True, axis='x', alpha=0.2)

    handles = [mpatches.Patch(color=DETECTOR_COLORS.get(d, '#999'), label=d) for d in det_names]
    ax2.legend(handles=handles, loc='upper right', fontsize=7, ncol=4)

    plt.tight_layout()
    fname = f"timeseries_realworld_{ds_name.lower().replace(' ', '_')}"
    plt.savefig(os.path.join(output_dir, f"{fname}.pdf"), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(output_dir, f"{fname}.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    Time-series saved: {fname}.pdf")


def plot_realworld_bell_curves(X, known_drifts, ds_name, output_dir=FIGURE_DIR):
    """Bell curves for real-world dataset: split at first known drift."""
    if not known_drifts:
        return
    drift_point = known_drifts[0]
    if drift_point >= len(X) or drift_point < 100:
        return

    n_features = min(X.shape[1], 6)  # Plot max 6 features
    n_cols = min(3, n_features)
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_features == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    before = X[:drift_point]
    after = X[drift_point:min(drift_point * 2, len(X))]

    for i in range(n_features):
        ax = axes[i]
        try:
            sns.kdeplot(before[:, i], ax=ax, color='#1E88E5', fill=True, alpha=0.3,
                        linewidth=2, label='Before Drift')
            sns.kdeplot(after[:, i], ax=ax, color='#E53935', fill=True, alpha=0.3,
                        linewidth=2, label='After Drift')
        except Exception:
            ax.hist(before[:, i], bins=50, density=True, alpha=0.3, color='#1E88E5', label='Before')
            ax.hist(after[:, i], bins=50, density=True, alpha=0.3, color='#E53935', label='After')

        ax.set_title(f'Feature F{i}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'Feature Distributions — {ds_name}\n(Split at first drift: t={drift_point})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = f"bell_curves_realworld_{ds_name.lower().replace(' ', '_')}"
    plt.savefig(os.path.join(output_dir, f"{fname}.pdf"), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(output_dir, f"{fname}.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    Bell curves saved: {fname}.pdf")


def plot_comparison_heatmap(results_df, output_dir=FIGURE_DIR):
    """Overall comparison heatmap: detectors vs datasets for key metrics."""
    for metric, title, cmap in [
        ("mtd", "Mean Time to Detection (↓ better)", "RdYlGn_r"),
        ("mdr", "Missed Detection Rate (↓ better)", "RdYlGn_r"),
    ]:
        sub = results_df.dropna(subset=[metric])
        if sub.empty:
            continue

        pivot = sub.pivot_table(index='detector', columns='dataset', values=metric)
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(14, 7))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap, ax=ax,
                    linewidths=0.5, cbar_kws={'label': metric.upper()})
        ax.set_title(f'Real-World Benchmark: {title}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        fname = f"heatmap_realworld_{metric}"
        plt.savefig(os.path.join(output_dir, f"{fname}.pdf"), bbox_inches='tight', dpi=150)
        plt.savefig(os.path.join(output_dir, f"{fname}.png"), bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  Heatmap saved: {fname}.pdf")


def plot_bar_comparison(results_df, output_dir=FIGURE_DIR):
    """Grouped bar chart comparing detectors on key metrics."""
    # Only datasets with ground truth
    gt_df = results_df.dropna(subset=["mtd"])
    if gt_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # MTD comparison
    ax1 = axes[0]
    mtd_pivot = gt_df.pivot_table(index='detector', values='mtd', aggfunc='mean')
    mtd_pivot = mtd_pivot.sort_values('mtd')
    colors = [DETECTOR_COLORS.get(d, '#999') for d in mtd_pivot.index]
    mtd_pivot.plot(kind='barh', ax=ax1, color=colors, legend=False)
    ax1.set_title('Avg. Mean Time to Detection (MTD)\n↓ Lower is Better', fontweight='bold')
    ax1.set_xlabel('MTD (time steps)')

    # MDR comparison
    ax2 = axes[1]
    mdr_pivot = gt_df.pivot_table(index='detector', values='mdr', aggfunc='mean')
    mdr_pivot = mdr_pivot.sort_values('mdr')
    colors = [DETECTOR_COLORS.get(d, '#999') for d in mdr_pivot.index]
    mdr_pivot.plot(kind='barh', ax=ax2, color=colors, legend=False)
    ax2.set_title('Avg. Missed Detection Rate (MDR)\n↓ Lower is Better', fontweight='bold')
    ax2.set_xlabel('MDR')

    plt.tight_layout()
    fname = "bar_comparison_realworld"
    plt.savefig(os.path.join(output_dir, f"{fname}.pdf"), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(output_dir, f"{fname}.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Bar comparison saved: {fname}.pdf")


# ──────────────────────────────────────────────────────────────
# Main Experiment
# ──────────────────────────────────────────────────────────────

def run_experiment_6():
    """Run comprehensive multi-detector real-world benchmark."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("  EXPERIMENT 6: Multi-Detector Real-World Benchmark")
    print("=" * 70)

    datasets = load_datasets()
    all_results = []

    for ds_idx, (ds_name, stream) in enumerate(datasets.items(), 1):
        n_total = getattr(stream, 'n_samples', None) or getattr(stream, 'stream_length', MAX_SAMPLES)
        n_features = getattr(stream, 'n_features', None)
        known_drifts = getattr(stream, 'drifts', [])
        known_in_range = [d for d in known_drifts if d < MAX_SAMPLES]

        print(f"\n{'─' * 60}")
        print(f"  [{ds_idx}/{len(datasets)}] {ds_name}")
        print(f"  Features: {n_features}, Known drifts: {len(known_in_range)}")
        print(f"{'─' * 60}")

        # Convert stream to arrays for reuse
        print("  Loading data...", flush=True)
        X, y = stream_to_arrays(stream, MAX_SAMPLES)
        print(f"  Loaded {len(X)} × {X.shape[1]} features")

        # Bell curves for datasets with known drifts
        if known_in_range:
            plot_realworld_bell_curves(X, known_in_range, ds_name)

        # Run all detectors
        all_detections = {}
        for det_name, (det_class, det_kwargs) in DETECTORS.items():
            print(f"    {det_name}...", end=" ", flush=True)
            try:
                dets, elapsed, detector, timed_out = run_detector_on_array(X, det_name, det_class, det_kwargs)
                status = f"{len(dets)} dets, {elapsed:.1f}s"
                if timed_out:
                    status += " (TIMEOUT)"
                print(status)
                all_detections[det_name] = dets

                # Compute metrics if ground truth available
                if known_in_range and dets:
                    metrics = calculate_drift_metrics(known_in_range, dets)
                else:
                    metrics = {"mtr": np.nan, "mtfa": np.nan, "mtd": np.nan, "mdr": np.nan}

                # Store EADD-specific info
                eadd_report = None
                if det_name == "EADD" and hasattr(detector, 'last_shap_importances') and detector.last_shap_importances:
                    eadd_report = detector.get_last_report()

                all_results.append({
                    "dataset": ds_name,
                    "detector": det_name,
                    "n_features": X.shape[1],
                    "n_known_drifts": len(known_in_range),
                    "n_detections": len(dets),
                    "time_s": round(elapsed, 2),
                    "mtd": metrics.get("mtd", np.nan),
                    "mdr": metrics.get("mdr", np.nan),
                    "mtr": metrics.get("mtr", np.nan),
                    "mtfa": metrics.get("mtfa", np.nan),
                })
            except Exception as e:
                print(f"ERROR: {e}")
                all_detections[det_name] = []
                all_results.append({
                    "dataset": ds_name,
                    "detector": det_name,
                    "n_features": X.shape[1],
                    "n_known_drifts": len(known_in_range),
                    "n_detections": 0,
                    "time_s": 0,
                    "mtd": np.nan,
                    "mdr": 1.0,
                    "mtr": np.nan,
                    "mtfa": np.nan,
                })

        # Time-series visualization
        plot_realworld_timeseries(X, known_in_range, all_detections, ds_name)

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(RESULTS_DIR, "experiment6_multidetector_realworld.csv"), index=False)
    print(f"\nResults saved to {RESULTS_DIR}/experiment6_multidetector_realworld.csv")

    # Generate comparison visualizations
    plot_comparison_heatmap(df)
    plot_bar_comparison(df)

    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY: Real-World Benchmark Results")
    print("=" * 70)
    summary = df.pivot_table(index='detector', values=['mtd', 'mdr', 'n_detections', 'time_s'],
                              aggfunc='mean')
    print(summary.round(2).to_string())

    # LaTeX table
    _generate_realworld_latex(df)

    return df


def _generate_realworld_latex(df):
    """Generate LaTeX table for real-world benchmark."""
    output = []
    output.append(r"\begin{table}[htbp]")
    output.append(r"\centering")
    output.append(r"\caption{Multi-Detector Real-World Benchmark Comparison}")
    output.append(r"\label{tab:exp6_realworld}")
    output.append(r"\small")

    datasets = df['dataset'].unique()
    n_ds = len(datasets)
    col_spec = "l|" + "r" * n_ds + "|r"
    output.append(r"\begin{tabular}{" + col_spec + "}")
    output.append(r"\hline")

    # Header
    header = "Detector"
    for ds in datasets:
        short = ds[:8]
        header += f" & {short}"
    header += r" & Avg. \\"
    output.append(header)
    output.append(r"\hline")
    output.append(r"\multicolumn{" + str(n_ds + 2) + r"}{c}{\textit{Mean Time to Detection (MTD)} $\downarrow$} \\")
    output.append(r"\hline")

    for det in DETECTORS.keys():
        row = det
        vals = []
        for ds in datasets:
            v = df[(df['detector'] == det) & (df['dataset'] == ds)]['mtd'].values
            if len(v) > 0 and not np.isnan(v[0]):
                row += f" & {v[0]:.0f}"
                vals.append(v[0])
            else:
                row += " & --"
        avg = np.mean(vals) if vals else np.nan
        row += f" & {avg:.0f}" if not np.isnan(avg) else " & --"
        row += r" \\"
        output.append(row)

    output.append(r"\hline")
    output.append(r"\end{tabular}")
    output.append(r"\end{table}")

    latex = "\n".join(output)
    with open(os.path.join(RESULTS_DIR, "experiment6_latex_table.tex"), "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to {RESULTS_DIR}/experiment6_latex_table.tex")


if __name__ == "__main__":
    run_experiment_6()
