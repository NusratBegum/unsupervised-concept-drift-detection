"""
Demo: How Unsupervised Concept Drift Detection Works
=====================================================
This script demonstrates drift detection on ALL available datasets.

Created by: Nusrat Begum (Learning Fork)
"""

from datasets import (
    InsectsAbruptBalanced,
    InsectsGradualBalanced,
    InsectsIncrementalBalanced,
    InsectsIncrementalAbruptBalanced,
    InsectsIncrementalReoccurringBalanced,
    Electricity,
    NOAAWeather,
    OutdoorObjects,
    PokerHand,
    Powersupply,
    RialtoBridgeTimelapse,
    Luxembourg,
    Ozone,
)
from detectors import DiscriminativeDriftDetector2019  # D3 detector


def run_detection_on_dataset(stream, max_samples=10_000):
    """Run drift detection on a single dataset and return results."""
    
    # Initialize detector for each dataset
    detector = DiscriminativeDriftDetector2019(
        n_reference_samples=200,
        recent_samples_proportion=0.5,
        threshold=0.7,
        seed=42
    )
    
    detected_drifts = []
    samples_processed = min(max_samples, stream.n_samples)
    
    for i, (x, y) in enumerate(stream):
        if i >= max_samples:
            break
        
        is_drift = detector.update(x)
        if is_drift:
            detected_drifts.append(i)
    
    return detected_drifts, samples_processed


def main():
    print("=" * 70)
    print("   UNSUPERVISED CONCEPT DRIFT DETECTION - FULL DEMO")
    print("   Testing D3 detector on ALL available datasets")
    print("=" * 70)
    print()
    
    # Define all datasets to test
    datasets = [
        # INSECTS datasets (with known ground truth drifts)
        ("INSECTS Abrupt Balanced", InsectsAbruptBalanced),
        ("INSECTS Gradual Balanced", InsectsGradualBalanced),
        ("INSECTS Incremental Balanced", InsectsIncrementalBalanced),
        ("INSECTS Incremental-Abrupt Balanced", InsectsIncrementalAbruptBalanced),
        ("INSECTS Incremental-Reoccurring Balanced", InsectsIncrementalReoccurringBalanced),
        # Other real-world datasets (no ground truth drifts)
        ("Electricity", Electricity),
        ("NOAA Weather", NOAAWeather),
        ("Outdoor Objects", OutdoorObjects),
        ("Poker Hand", PokerHand),
        ("Powersupply", Powersupply),
        ("Rialto Bridge Timelapse", RialtoBridgeTimelapse),
        ("Luxembourg", Luxembourg),
        ("Ozone", Ozone),
    ]
    
    results = []
    
    for name, DatasetClass in datasets:
        print(f"{'─' * 70}")
        print(f"Testing: {name}")
        print(f"{'─' * 70}")
        
        try:
            stream = DatasetClass()
            print(f"  Samples: {stream.n_samples:,} | Features: {stream.n_features}")
            
            # Check if dataset has ground truth drifts
            has_ground_truth = hasattr(stream, 'drifts') and stream.drifts
            if has_ground_truth:
                print(f"  Ground truth drifts: {stream.drifts}")
            else:
                print(f"  Ground truth drifts: None (unknown)")
            
            # Run detection (limit samples for speed)
            max_samples = min(20_000, stream.n_samples)
            detected, processed = run_detection_on_dataset(stream, max_samples)
            
            print(f"  Processed: {processed:,} samples")
            print(f"  Drifts detected: {len(detected)}")
            if detected:
                print(f"  Detection points: {detected[:10]}{'...' if len(detected) > 10 else ''}")
            
            # If we have ground truth, evaluate
            if has_ground_truth:
                actual_in_range = [d for d in stream.drifts if d < max_samples]
                hits = 0
                for actual in actual_in_range:
                    if any(abs(det - actual) < 2000 for det in detected):
                        hits += 1
                print(f"  Evaluation: {hits}/{len(actual_in_range)} ground truth drifts detected")
            
            results.append((name, "✅ Success", len(detected)))
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append((name, f"❌ Error", 0))
        
        print()
    
    # Summary
    print("=" * 70)
    print("   SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Dataset':<45} {'Status':<15} {'Drifts'}")
    print("-" * 70)
    for name, status, drifts in results:
        print(f"{name:<45} {status:<15} {drifts}")
    
    print()
    print("=" * 70)
    print("Demo complete! All datasets tested with D3 detector.")
    print("=" * 70)


if __name__ == "__main__":
    main()
