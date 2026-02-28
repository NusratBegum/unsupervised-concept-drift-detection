"""
EADD Experiments - Experiment 1: Sensitivity to Temporal Drift Patterns
========================================================================
Tests EADD detection across four temporal drift types:
  - Abrupt, Gradual, Incremental, Recurring

Synthetic data with 5 features, 10,000 samples, drift at t=5000.

Author: Nusrat Begum
Thesis: Feature Drift Detection via Adversarial Validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import json
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detectors.eadd import ExplainableAdversarialDriftDetector
from detectors.d3 import DiscriminativeDriftDetector2019

# ──────────────────────────────────────────────────────────────
# Synthetic Data Generators
# ──────────────────────────────────────────────────────────────

def generate_abrupt_drift(n_samples=10000, n_features=5, drift_point=5000, shift_magnitude=2.0, seed=42):
    """Abrupt drift: sudden mean shift at drift_point."""
    rng = np.random.default_rng(seed)
    X_pre = rng.normal(loc=0, scale=1, size=(drift_point, n_features))
    X_post = rng.normal(loc=shift_magnitude, scale=1, size=(n_samples - drift_point, n_features))
    X = np.vstack([X_pre, X_post])
    y = np.zeros(n_samples, dtype=int)
    y[drift_point:] = 1
    return X, y, [drift_point]


def generate_gradual_drift(n_samples=10000, n_features=5, drift_start=4000, drift_end=6000,
                           shift_magnitude=2.0, seed=42):
    """Gradual drift: transition zone where samples mix old/new distributions."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        if i < drift_start:
            X[i] = rng.normal(loc=0, scale=1, size=n_features)
        elif i < drift_end:
            prob_new = (i - drift_start) / (drift_end - drift_start)
            if rng.random() < prob_new:
                X[i] = rng.normal(loc=shift_magnitude, scale=1, size=n_features)
            else:
                X[i] = rng.normal(loc=0, scale=1, size=n_features)
        else:
            X[i] = rng.normal(loc=shift_magnitude, scale=1, size=n_features)
    y = np.zeros(n_samples, dtype=int)
    y[drift_start:] = 1
    return X, y, [drift_start]


def generate_incremental_drift(n_samples=10000, n_features=5, drift_start=3000, drift_end=7000,
                                shift_magnitude=2.0, seed=42):
    """Incremental drift: mean shifts linearly over a long period."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        if i < drift_start:
            mean = 0
        elif i < drift_end:
            mean = shift_magnitude * (i - drift_start) / (drift_end - drift_start)
        else:
            mean = shift_magnitude
        X[i] = rng.normal(loc=mean, scale=1, size=n_features)
    y = np.zeros(n_samples, dtype=int)
    y[drift_start:] = 1
    return X, y, [drift_start]


def generate_recurring_drift(n_samples=10000, n_features=5, period=2500,
                              shift_magnitude=2.0, seed=42):
    """Recurring drift: alternating mean shifts at regular intervals."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, n_features))
    drifts = []
    for i in range(n_samples):
        cycle = (i // period) % 2
        mean = shift_magnitude if cycle == 1 else 0
        X[i] = rng.normal(loc=mean, scale=1, size=n_features)
        if i > 0 and i % period == 0:
            drifts.append(i)
    y = np.zeros(n_samples, dtype=int)
    return X, y, drifts


# ──────────────────────────────────────────────────────────────
# Experiment Runner
# ──────────────────────────────────────────────────────────────

def run_eadd_on_stream(X, drift_points, n_ref=500, n_cur=200, n_perm=50,
                       alpha=0.01, freq=50, seed=42):
    """Run EADD on a feature stream and return detection results."""
    detector = ExplainableAdversarialDriftDetector(
        n_reference_samples=n_ref,
        n_current_samples=n_cur,
        auc_threshold=0.7,
        n_permutations=n_perm,
        significance_level=alpha,
        use_reservoir_sampling=True,
        monitoring_frequency=freq,
        seed=seed,
    )

    detections = []
    aucs = []
    p_values = []

    for i in range(len(X)):
        features = {f"F{j}": float(X[i, j]) for j in range(X.shape[1])}
        is_drift = detector.update(features)
        if is_drift:
            detections.append(i)
            aucs.append(detector.last_auc)
            p_values.append(detector.last_p_value)

    return detections, aucs, p_values


def run_d3_on_stream(X, n_ref=500, proportion=0.4, threshold=0.7, seed=42):
    """Run D3 baseline on a feature stream."""
    detector = DiscriminativeDriftDetector2019(
        n_reference_samples=n_ref,
        recent_samples_proportion=proportion,
        threshold=threshold,
        seed=seed,
    )

    detections = []
    for i in range(len(X)):
        features = {f"F{j}": float(X[i, j]) for j in range(X.shape[1])}
        is_drift = detector.update(features)
        if is_drift:
            detections.append(i)

    return detections


def compute_detection_delay(drift_points, detections, tolerance=1500):
    """Compute mean detection delay for true drift points."""
    delays = []
    for dp in drift_points:
        post_detections = [d for d in detections if dp <= d <= dp + tolerance]
        if post_detections:
            delays.append(post_detections[0] - dp)
        else:
            delays.append(np.nan)
    return delays


def compute_false_alarm_rate(drift_points, detections, tolerance=1500):
    """Count detections before the first drift onset (genuine false alarms).

    Post-onset detections during/after drift are not false alarms since
    the distribution has indeed changed.
    """
    first_onset = min(drift_points)
    return sum(1 for d in detections if d < first_onset)


# ──────────────────────────────────────────────────────────────
# Main Experiment
# ──────────────────────────────────────────────────────────────

def run_experiment_1(output_dir="experiments/results"):
    """Run Experiment 1: Temporal Drift Pattern Sensitivity."""
    os.makedirs(output_dir, exist_ok=True)

    # (generator_fn, tolerance) — tolerance covers full transition + buffer
    drift_types = {
        "Abrupt":      (generate_abrupt_drift,      1500),   # instantaneous
        "Gradual":     (generate_gradual_drift,      3500),   # transition 4000-6000 + 1500
        "Incremental": (generate_incremental_drift,  5500),   # transition 3000-7000 + 1500
        "Recurring":   (generate_recurring_drift,    1500),   # instantaneous per change
    }

    n_runs = 5
    results = []

    for drift_name, (generator, tol) in drift_types.items():
        print(f"\n{'='*60}")
        print(f"  Testing: {drift_name} Drift")
        print(f"{'='*60}")

        eadd_delays_all = []
        d3_delays_all = []
        eadd_false_alarms_all = []
        d3_false_alarms_all = []
        eadd_detections_all = []
        d3_detections_all = []

        for run in range(n_runs):
            seed = 42 + run * 100
            X, y, drift_points = generator(seed=seed)

            # EADD
            eadd_dets, aucs, pvals = run_eadd_on_stream(X, drift_points, seed=seed)
            eadd_delays = compute_detection_delay(drift_points, eadd_dets, tolerance=tol)
            eadd_fa = compute_false_alarm_rate(drift_points, eadd_dets)

            # D3
            d3_dets = run_d3_on_stream(X, seed=seed)
            d3_delays = compute_detection_delay(drift_points, d3_dets, tolerance=tol)
            d3_fa = compute_false_alarm_rate(drift_points, d3_dets)

            eadd_delays_all.append(eadd_delays)
            d3_delays_all.append(d3_delays)
            eadd_false_alarms_all.append(eadd_fa)
            d3_false_alarms_all.append(d3_fa)
            eadd_detections_all.append(len(eadd_dets))
            d3_detections_all.append(len(d3_dets))

            print(f"  Run {run+1}: EADD detections={len(eadd_dets)}, "
                  f"D3 detections={len(d3_dets)}")

        # Aggregate
        eadd_mean_delay = np.nanmean([d for delays in eadd_delays_all for d in delays])
        d3_mean_delay = np.nanmean([d for delays in d3_delays_all for d in delays])
        eadd_success_rate = np.mean([
            1 - np.mean([np.isnan(d) for d in delays]) for delays in eadd_delays_all
        ]) * 100
        d3_success_rate = np.mean([
            1 - np.mean([np.isnan(d) for d in delays]) for delays in d3_delays_all
        ]) * 100

        result = {
            "drift_type": drift_name,
            "eadd_mean_delay": eadd_mean_delay,
            "d3_mean_delay": d3_mean_delay,
            "eadd_success_rate": eadd_success_rate,
            "d3_success_rate": d3_success_rate,
            "eadd_mean_false_alarms": np.mean(eadd_false_alarms_all),
            "d3_mean_false_alarms": np.mean(d3_false_alarms_all),
            "eadd_mean_detections": np.mean(eadd_detections_all),
            "d3_mean_detections": np.mean(d3_detections_all),
        }
        results.append(result)

        print(f"\n  EADD: delay={eadd_mean_delay:.0f}, success={eadd_success_rate:.0f}%, "
              f"false_alarms={np.mean(eadd_false_alarms_all):.1f}")
        print(f"  D3:   delay={d3_mean_delay:.0f}, success={d3_success_rate:.0f}%, "
              f"false_alarms={np.mean(d3_false_alarms_all):.1f}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "experiment1_temporal_patterns.csv"), index=False)
    print(f"\nResults saved to {output_dir}/experiment1_temporal_patterns.csv")

    # Plot
    _plot_experiment1(results, output_dir)

    return results


def _plot_experiment1(results, output_dir):
    """Generate plots for Experiment 1."""
    drift_types = [r["drift_type"] for r in results]
    eadd_delays = [r["eadd_mean_delay"] for r in results]
    d3_delays = [r["d3_mean_delay"] for r in results]
    eadd_success = [r["eadd_success_rate"] for r in results]
    d3_success = [r["d3_success_rate"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Detection Delay
    x = np.arange(len(drift_types))
    width = 0.35
    axes[0].bar(x - width/2, eadd_delays, width, label='EADD', color='#2196F3', alpha=0.85)
    axes[0].bar(x + width/2, d3_delays, width, label='D3', color='#FF9800', alpha=0.85)
    axes[0].set_xlabel('Drift Type')
    axes[0].set_ylabel('Mean Detection Delay (samples)')
    axes[0].set_title('Detection Delay by Drift Type')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(drift_types)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Success Rate
    axes[1].bar(x - width/2, eadd_success, width, label='EADD', color='#2196F3', alpha=0.85)
    axes[1].bar(x + width/2, d3_success, width, label='D3', color='#FF9800', alpha=0.85)
    axes[1].set_xlabel('Drift Type')
    axes[1].set_ylabel('Detection Success Rate (%)')
    axes[1].set_title('Detection Success Rate by Drift Type')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(drift_types)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "experiment1_temporal_patterns.png"), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "experiment1_temporal_patterns.pdf"), bbox_inches='tight')
    plt.close()
    print(f"  Plots saved to {output_dir}/experiment1_temporal_patterns.{{png,pdf}}")


if __name__ == "__main__":
    run_experiment_1()
