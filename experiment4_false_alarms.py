"""
EADD Experiments - Experiment 4: Robustness to Stable Data (False Alarms)
=========================================================================
Tests EADD's false alarm rate on stable data with no drift.
Compares EADD's permutation test against D3's fixed threshold.

Author: Nusrat Begum
Thesis: Feature Drift Detection via Adversarial Validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detectors.eadd import ExplainableAdversarialDriftDetector
from detectors.d3 import DiscriminativeDriftDetector2019


# ──────────────────────────────────────────────────────────────
# Stable Stream Generators (No Drift)
# ──────────────────────────────────────────────────────────────

def generate_stable_gaussian(n_samples=10000, n_features=5, seed=42):
    """Purely stable Gaussian stream with no drift."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, n_features))
    return X


def generate_autocorrelated_stable(n_samples=10000, n_features=5, seed=42):
    """Stable AR(1) stream — autocorrelated but stationary."""
    rng = np.random.default_rng(seed)
    phi = 0.7  # AR(1) coefficient (stationary since |phi|<1)
    X = np.zeros((n_samples, n_features))
    X[0] = rng.normal(0, 1, size=n_features)
    for t in range(1, n_samples):
        X[t] = phi * X[t - 1] + rng.normal(0, 1, size=n_features)
    return X


def generate_heteroscedastic_stable(n_samples=10000, n_features=5, seed=42):
    """Stable mean but mildly time-varying variance — mean=0 throughout."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, n_features))
    for t in range(n_samples):
        # Mild variance oscillation (0.85 to 1.15) — not enough to constitute real drift
        sigma = 1.0 + 0.15 * np.sin(2 * np.pi * t / 2000)
        X[t] = rng.normal(0, sigma, size=n_features)
    return X


def generate_correlated_stable(n_samples=10000, n_features=5, seed=42):
    """Stable stream with correlated features."""
    rng = np.random.default_rng(seed)
    cov = np.eye(n_features)
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                cov[i, j] = 0.5 ** abs(i - j)
    X = rng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    return X


# ──────────────────────────────────────────────────────────────
# Experiment Runner
# ──────────────────────────────────────────────────────────────

def count_false_alarms(X, detector_class, detector_kwargs):
    """Run detector on stable data and count false alarms."""
    detector = detector_class(**detector_kwargs)
    false_alarms = []
    for i in range(len(X)):
        features = {f"F{j}": float(X[i, j]) for j in range(X.shape[1])}
        if detector.update(features):
            false_alarms.append(i)
    return false_alarms


def run_experiment_4(output_dir="experiments/results"):
    """Run Experiment 4: False Alarm Robustness."""
    os.makedirs(output_dir, exist_ok=True)

    stable_streams = {
        "Gaussian (i.i.d.)": generate_stable_gaussian,
        "Autocorrelated": generate_autocorrelated_stable,
        "Heteroscedastic": generate_heteroscedastic_stable,
        "Correlated": generate_correlated_stable,
    }

    n_runs = 5
    results = []

    for stream_name, generator in stable_streams.items():
        print(f"\n{'='*60}")
        print(f"  Testing stable stream: {stream_name}")
        print(f"{'='*60}")

        eadd_fa_counts = []
        d3_fa_counts_06 = []
        d3_fa_counts_07 = []
        d3_fa_counts_08 = []

        for run in range(n_runs):
            seed = 42 + run * 100
            X = generator(seed=seed)

            # EADD with thesis parameters (permutation test guards against false alarms)
            eadd_fa = count_false_alarms(X, ExplainableAdversarialDriftDetector, {
                "n_reference_samples": 500,
                "n_current_samples": 200,
                "auc_threshold": 0.7,
                "n_permutations": 50,
                "significance_level": 0.01,
                "monitoring_frequency": 50,
                "seed": seed,
            })
            eadd_fa_counts.append(len(eadd_fa))

            # D3 benchmark configs: smaller windows = more checks = more potential false alarms
            # D3 (τ=0.6, n_ref=100) — most sensitive configuration
            d3_fa_06 = count_false_alarms(X, DiscriminativeDriftDetector2019, {
                "n_reference_samples": 100,
                "recent_samples_proportion": 0.5,
                "threshold": 0.6,
                "seed": seed,
            })
            d3_fa_counts_06.append(len(d3_fa_06))

            # D3 (τ=0.7, n_ref=100)
            d3_fa_07 = count_false_alarms(X, DiscriminativeDriftDetector2019, {
                "n_reference_samples": 100,
                "recent_samples_proportion": 0.5,
                "threshold": 0.7,
                "seed": seed,
            })
            d3_fa_counts_07.append(len(d3_fa_07))

            # D3 (τ=0.8, n_ref=100)
            d3_fa_08 = count_false_alarms(X, DiscriminativeDriftDetector2019, {
                "n_reference_samples": 100,
                "recent_samples_proportion": 0.5,
                "threshold": 0.8,
                "seed": seed,
            })
            d3_fa_counts_08.append(len(d3_fa_08))

            print(f"  Run {run+1}: EADD={len(eadd_fa)}, D3(0.6)={len(d3_fa_06)}, "
                  f"D3(0.7)={len(d3_fa_07)}, D3(0.8)={len(d3_fa_08)}")

        result = {
            "stream_type": stream_name,
            "eadd_mean_fa": np.mean(eadd_fa_counts),
            "eadd_std_fa": np.std(eadd_fa_counts),
            "d3_06_mean_fa": np.mean(d3_fa_counts_06),
            "d3_06_std_fa": np.std(d3_fa_counts_06),
            "d3_07_mean_fa": np.mean(d3_fa_counts_07),
            "d3_07_std_fa": np.std(d3_fa_counts_07),
            "d3_08_mean_fa": np.mean(d3_fa_counts_08),
            "d3_08_std_fa": np.std(d3_fa_counts_08),
        }
        results.append(result)

        print(f"\n  EADD: {np.mean(eadd_fa_counts):.1f} ± {np.std(eadd_fa_counts):.1f} false alarms")
        print(f"  D3(0.6): {np.mean(d3_fa_counts_06):.1f} ± {np.std(d3_fa_counts_06):.1f} false alarms")
        print(f"  D3(0.7): {np.mean(d3_fa_counts_07):.1f} ± {np.std(d3_fa_counts_07):.1f} false alarms")
        print(f"  D3(0.8): {np.mean(d3_fa_counts_08):.1f} ± {np.std(d3_fa_counts_08):.1f} false alarms")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "experiment4_false_alarms.csv"), index=False)
    print(f"\nResults saved to {output_dir}/experiment4_false_alarms.csv")

    # Plot
    _plot_experiment4(results, output_dir)

    return results


def _plot_experiment4(results, output_dir):
    """Generate false alarm comparison plot."""
    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(12, 6))

    streams = df["stream_type"]
    x = np.arange(len(streams))
    width = 0.2

    bars1 = ax.bar(x - 1.5*width, df["eadd_mean_fa"], width, yerr=df["eadd_std_fa"],
                   label='EADD (p<0.01)', color='#2196F3', alpha=0.85, capsize=4)
    bars2 = ax.bar(x - 0.5*width, df["d3_06_mean_fa"], width, yerr=df["d3_06_std_fa"],
                   label='D3 (τ=0.6)', color='#FFB74D', alpha=0.85, capsize=4)
    bars3 = ax.bar(x + 0.5*width, df["d3_07_mean_fa"], width, yerr=df["d3_07_std_fa"],
                   label='D3 (τ=0.7)', color='#FF9800', alpha=0.85, capsize=4)
    bars4 = ax.bar(x + 1.5*width, df["d3_08_mean_fa"], width, yerr=df["d3_08_std_fa"],
                   label='D3 (τ=0.8)', color='#E65100', alpha=0.85, capsize=4)

    ax.set_xlabel('Stable Stream Type')
    ax.set_ylabel('Number of False Alarms')
    ax.set_title('False Alarm Comparison on Stable Data (No Drift)\nLower is Better')
    ax.set_xticks(x)
    ax.set_xticklabels(streams, fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.0f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "experiment4_false_alarms.png"), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "experiment4_false_alarms.pdf"), bbox_inches='tight')
    plt.close()
    print(f"  Plots saved to {output_dir}/experiment4_false_alarms.{{png,pdf}}")


if __name__ == "__main__":
    run_experiment_4()
