"""
EADD Experiments - Experiment 2: Real-World Benchmark
=====================================================
Benchmark EADD against D3 on real-world datasets from the
Lukats et al. (2025) benchmark suite.

Datasets: Electricity, Insects variants, NOAA Weather, Poker Hand,
Powersupply, Outdoor Objects, etc.

Author: Nusrat Begum
Thesis: Feature Drift Detection via Adversarial Validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detectors.eadd import ExplainableAdversarialDriftDetector
from detectors.d3 import DiscriminativeDriftDetector2019
from metrics.drift import calculate_drift_metrics

# ──────────────────────────────────────────────────────────────
# Dataset Loading
# ──────────────────────────────────────────────────────────────

def load_real_world_datasets():
    """Load all available real-world benchmark datasets."""
    from datasets import (
        Electricity,
        InsectsAbruptBalanced,
        InsectsGradualBalanced,
        InsectsIncrementalAbruptBalanced,
        InsectsIncrementalBalanced,
        InsectsIncrementalReoccurringBalanced,
        NOAAWeather,
        OutdoorObjects,
        PokerHand,
        Powersupply,
        RialtoBridgeTimelapse,
        SineClusters,
        WaveformDrift2,
    )

    datasets = {
        "Electricity": Electricity(),
        "InsectsAbrupt": InsectsAbruptBalanced(),
        "InsectsGradual": InsectsGradualBalanced(),
        "InsectsIncrAbrupt": InsectsIncrementalAbruptBalanced(),
        "InsectsIncremental": InsectsIncrementalBalanced(),
        "InsectsReoccurring": InsectsIncrementalReoccurringBalanced(),
        "NOAAWeather": NOAAWeather(),
        "OutdoorObjects": OutdoorObjects(),
        "PokerHand": PokerHand(),
        "Powersupply": Powersupply(),
        "RialtoBridge": RialtoBridgeTimelapse(),
        "SineClusters": SineClusters(drift_frequency=5000, stream_length=50000, seed=531874),
        "WaveformDrift2": WaveformDrift2(drift_frequency=5000, stream_length=50000, seed=2401137),
    }
    return datasets


def stream_to_arrays(stream, max_samples=50000):
    """Convert a River-style stream to numpy arrays."""
    X_list = []
    y_list = []
    for i, (x, y) in enumerate(stream):
        if i >= max_samples:
            break
        X_list.append(list(x.values()))
        y_list.append(y)
    return np.array(X_list, dtype=float), np.array(y_list)


# ──────────────────────────────────────────────────────────────
# Detection on Real-World Streams
# ──────────────────────────────────────────────────────────────

def run_detector_on_stream(stream, detector_class, detector_kwargs, max_samples=50000):
    """Run a detector on a stream and return detections + timing."""
    detector = detector_class(**detector_kwargs)
    detections = []
    start_time = time.time()

    for i, (x, y) in enumerate(stream):
        if i >= max_samples:
            break
        is_drift = detector.update(x)
        if is_drift:
            detections.append(i)

    elapsed = time.time() - start_time
    return detections, elapsed, detector


# ──────────────────────────────────────────────────────────────
# Main Experiment
# ──────────────────────────────────────────────────────────────

def run_experiment_2(output_dir="experiments/results"):
    """Run Experiment 2: Real-World Benchmark Comparison."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading datasets...")
    datasets = load_real_world_datasets()
    max_samples = 30000  # Limit for computational feasibility

    results = []

    for ds_name, stream in datasets.items():
        n_samples = getattr(stream, 'n_samples', None) or getattr(stream, 'stream_length', None)
        n_features = getattr(stream, 'n_features', None)
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"  Samples: {n_samples}, Features: {n_features}")
        known_drifts = getattr(stream, 'drifts', [])
        if known_drifts:
            print(f"  Known drifts: {known_drifts}")
        print(f"{'='*60}")

        # EADD
        print("  Running EADD...")
        try:
            eadd_dets, eadd_time, eadd_detector = run_detector_on_stream(
                stream,
                ExplainableAdversarialDriftDetector,
                {
                    "n_reference_samples": 500,
                    "n_current_samples": 200,
                    "auc_threshold": 0.7,
                    "n_permutations": 50,  # Reduced for speed
                    "significance_level": 0.01,
                    "monitoring_frequency": 100,
                    "seed": 42,
                },
                max_samples=max_samples,
            )
            print(f"    EADD: {len(eadd_dets)} drifts in {eadd_time:.1f}s")
        except Exception as e:
            print(f"    EADD Error: {e}")
            eadd_dets, eadd_time = [], 0.0

        # D3 baseline
        print("  Running D3...")
        try:
            d3_dets, d3_time, d3_detector = run_detector_on_stream(
                stream,
                DiscriminativeDriftDetector2019,
                {
                    "n_reference_samples": 500,
                    "recent_samples_proportion": 0.4,
                    "threshold": 0.7,
                    "seed": 42,
                },
                max_samples=max_samples,
            )
            print(f"    D3: {len(d3_dets)} drifts in {d3_time:.1f}s")
        except Exception as e:
            print(f"    D3 Error: {e}")
            d3_dets, d3_time = [], 0.0

        # Drift metrics (if ground truth available)
        eadd_metrics = {"mtr": np.nan, "mtfa": np.nan, "mtd": np.nan, "mdr": np.nan}
        d3_metrics = {"mtr": np.nan, "mtfa": np.nan, "mtd": np.nan, "mdr": np.nan}
        if known_drifts:
            known_in_range = [d for d in known_drifts if d < max_samples]
            if known_in_range and eadd_dets:
                eadd_metrics = calculate_drift_metrics(known_in_range, eadd_dets)
            if known_in_range and d3_dets:
                d3_metrics = calculate_drift_metrics(known_in_range, d3_dets)

        result = {
            "dataset": ds_name,
            "n_samples": min(n_samples, max_samples) if n_samples else max_samples,
            "n_features": n_features,
            "known_drifts": len(known_drifts) if known_drifts else 0,
            "eadd_n_detections": len(eadd_dets),
            "d3_n_detections": len(d3_dets),
            "eadd_time_s": round(eadd_time, 2),
            "d3_time_s": round(d3_time, 2),
            "eadd_mtd": eadd_metrics.get("mtd", np.nan),
            "d3_mtd": d3_metrics.get("mtd", np.nan),
            "eadd_mdr": eadd_metrics.get("mdr", np.nan),
            "d3_mdr": d3_metrics.get("mdr", np.nan),
            "eadd_mtr": eadd_metrics.get("mtr", np.nan),
            "d3_mtr": d3_metrics.get("mtr", np.nan),
        }
        results.append(result)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "experiment2_realworld_benchmark.csv"), index=False)
    print(f"\nResults saved to {output_dir}/experiment2_realworld_benchmark.csv")

    # Plot
    _plot_experiment2(results, output_dir)

    return results


def _plot_experiment2(results, output_dir):
    """Generate plots for Experiment 2."""
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Number of detections comparison
    datasets = df["dataset"]
    x = np.arange(len(datasets))
    width = 0.35
    axes[0].barh(x - width/2, df["eadd_n_detections"], width, label='EADD', color='#2196F3', alpha=0.85)
    axes[0].barh(x + width/2, df["d3_n_detections"], width, label='D3', color='#FF9800', alpha=0.85)
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(datasets, fontsize=8)
    axes[0].set_xlabel('Number of Detections')
    axes[0].set_title('Drift Detections per Dataset')
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)

    # Computation time comparison
    axes[1].barh(x - width/2, df["eadd_time_s"], width, label='EADD', color='#2196F3', alpha=0.85)
    axes[1].barh(x + width/2, df["d3_time_s"], width, label='D3', color='#FF9800', alpha=0.85)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(datasets, fontsize=8)
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_title('Computation Time per Dataset')
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)

    # MTD for datasets with ground truth
    df_gt = df[df["known_drifts"] > 0].copy()
    if len(df_gt) > 0:
        x2 = np.arange(len(df_gt))
        axes[2].barh(x2 - width/2, df_gt["eadd_mtd"].fillna(0), width, label='EADD', color='#2196F3', alpha=0.85)
        axes[2].barh(x2 + width/2, df_gt["d3_mtd"].fillna(0), width, label='D3', color='#FF9800', alpha=0.85)
        axes[2].set_yticks(x2)
        axes[2].set_yticklabels(df_gt["dataset"], fontsize=8)
        axes[2].set_xlabel('Mean Time to Detection')
        axes[2].set_title('MTD (Datasets with Ground Truth)')
        axes[2].legend()
        axes[2].grid(axis='x', alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No ground truth\navailable', ha='center', va='center', fontsize=12)
        axes[2].set_title('MTD (Datasets with Ground Truth)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "experiment2_realworld_benchmark.png"), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "experiment2_realworld_benchmark.pdf"), bbox_inches='tight')
    plt.close()
    print(f"  Plots saved to {output_dir}/experiment2_realworld_benchmark.{{png,pdf}}")


if __name__ == "__main__":
    run_experiment_2()
