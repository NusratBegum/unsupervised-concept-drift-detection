"""
Demo: Explainable Adversarial Drift Detection (EADD)
=====================================================
Run this script to see how EADD detects concept drift and explains
which features are responsible, with automated prescriptions.

Usage:
    python demo_eadd.py

Author: Nusrat Begum
"""

import warnings
warnings.filterwarnings("ignore")   # silence sklearn / lightgbm chatter

from datasets import (
    InsectsAbruptBalanced,
    InsectsGradualBalanced,
    InsectsIncrementalBalanced,
    Electricity,
    SineClusters,
)
from detectors import ExplainableAdversarialDriftDetector


def run_eadd_on_dataset(stream, name, max_samples=15_000):
    """Run EADD on a single dataset and print detailed detection reports."""

    detector = ExplainableAdversarialDriftDetector(
        n_reference_samples=500,
        n_current_samples=200,
        auc_threshold=0.7,
        n_permutations=50,        # fewer permutations for demo speed
        significance_level=0.05,
        use_reservoir_sampling=True,
        monitoring_frequency=50,
        seed=42,
    )

    detected_drifts = []
    samples_processed = 0

    n_samples = getattr(stream, 'n_samples', None) or getattr(stream, 'stream_length', None)
    n_features = getattr(stream, 'n_features', '?')

    print(f"\n{'=' * 70}")
    print(f"  Dataset: {name}")
    print(f"  Total samples: {n_samples:,} | Features: {n_features}")

    has_ground_truth = hasattr(stream, 'drifts') and stream.drifts
    if has_ground_truth:
        gt_in_range = [d for d in stream.drifts if d < max_samples]
        print(f"  Ground truth drifts (in range): {gt_in_range}")
    else:
        print(f"  Ground truth drifts: unknown")

    print(f"  Processing up to {max_samples:,} samples ...")
    print(f"{'=' * 70}")

    for i, (x, y) in enumerate(stream):
        if i >= max_samples:
            break

        is_drift = detector.update(x)
        samples_processed = i + 1

        if is_drift:
            detected_drifts.append(i)
            report = detector.get_last_report()

            print(f"\n  *** DRIFT #{len(detected_drifts)} detected at sample {i:,} ***")
            print(f"      AUC:     {report['auc']:.4f}")
            print(f"      p-value: {report['p_value']:.4f}")

            # Show top feature importances from SHAP
            if report['feature_importances']:
                print(f"      Top drifting features (SHAP %):")
                for rank, (feat, imp) in enumerate(report['feature_importances'].items()):
                    if rank >= 5:
                        break
                    bar = "█" * int(imp / 2)
                    print(f"        {rank+1}. {feat:>20s}  {imp:5.1f}%  {bar}")

            # Show prescription
            if report['prescription']:
                rx = report['prescription']
                print(f"      Prescription ({rx['type']}): {rx['message']}")

        # Progress indicator every 5000 samples
        if (i + 1) % 5000 == 0:
            print(f"  ... processed {i+1:,} samples, {len(detected_drifts)} drift(s) so far")

    # Summary
    print(f"\n  {'─' * 60}")
    print(f"  SUMMARY for {name}")
    print(f"  Samples processed: {samples_processed:,}")
    print(f"  Drifts detected:   {len(detected_drifts)}")
    if detected_drifts:
        print(f"  Detection points:  {detected_drifts}")

    if has_ground_truth:
        gt_in_range = [d for d in stream.drifts if d < max_samples]
        tolerance = 2000
        hits = sum(1 for d in gt_in_range if any(abs(det - d) < tolerance for det in detected_drifts))
        print(f"  Ground truth hit:  {hits}/{len(gt_in_range)} (tolerance={tolerance})")

    print()
    return detected_drifts


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   EADD — Explainable Adversarial Drift Detection  (Demo)           ║")
    print("║   Author: Nusrat Begum | Mahidol University Thesis 2026            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Pick datasets that showcase different drift types
    datasets = [
        # Synthetic with known abrupt drifts
        ("SineClusters (synthetic, abrupt)",
         SineClusters(drift_frequency=5000, stream_length=15_000, seed=42)),
        # Real-world: abrupt insect drift
        ("INSECTS Abrupt Balanced", InsectsAbruptBalanced()),
        # Real-world: gradual insect drift
        ("INSECTS Gradual Balanced", InsectsGradualBalanced()),
        # Real-world: electricity market
        ("Electricity (Elec2)", Electricity()),
    ]

    all_results = []
    for name, stream in datasets:
        try:
            drifts = run_eadd_on_dataset(stream, name, max_samples=15_000)
            all_results.append((name, len(drifts)))
        except FileNotFoundError:
            print(f"\n  ⚠ Skipping '{name}' — dataset file not downloaded yet.")
            print(f"    Run `python convert_datasets.py` first to download it.")
            all_results.append((name, "N/A (missing file)"))

    # Final table
    print("\n" + "=" * 70)
    print("  OVERALL RESULTS")
    print("=" * 70)
    print(f"  {'Dataset':<45} {'Drifts Detected':>15}")
    print(f"  {'─' * 45} {'─' * 15}")
    for name, count in all_results:
        print(f"  {name:<45} {count:>15}")
    print()


if __name__ == "__main__":
    main()
