"""
Demo: How Unsupervised Concept Drift Detection Works
=====================================================
This script demonstrates drift detection step-by-step.
"""

from datasets import InsectsAbruptBalanced
from detectors import DiscriminativeDriftDetector2019  # D3 detector


def main():
    # ==========================================================
    # STEP 1: Load a dataset with known drift points
    # ==========================================================
    print("=" * 60)
    print("STEP 1: Loading the INSECTS Abrupt Balanced dataset")
    print("=" * 60)
    
    stream = InsectsAbruptBalanced()
    print(f"Dataset: {stream.__class__.__name__}")
    print(f"Number of samples: {stream.n_samples:,}")
    print(f"Number of features: {stream.n_features}")
    print(f"Known drift points: {stream.drifts}")
    print()
    
    # ==========================================================
    # STEP 2: Initialize a drift detector
    # ==========================================================
    print("=" * 60)
    print("STEP 2: Initializing the D3 (Discriminative Drift Detector)")
    print("=" * 60)
    
    detector = DiscriminativeDriftDetector2019(
        n_reference_samples=200,        # Size of reference window
        recent_samples_proportion=0.5,  # Size of recent window (relative)
        threshold=0.7,                  # AUC threshold for drift detection
        seed=42
    )
    
    print(f"Detector: D3 (Discriminative Drift Detector 2019)")
    print(f"Reference window size: {detector.n_reference_samples}")
    print(f"Total samples before detection starts: {detector.n_samples}")
    print(f"Detection threshold (AUC): {detector.threshold}")
    print()
    print("How D3 works:")
    print("  1. Collects 'reference' samples (old data)")
    print("  2. Collects 'recent' samples (new data)")  
    print("  3. Trains a classifier to distinguish old vs new")
    print("  4. If classifier succeeds (AUC > threshold) → DRIFT!")
    print()
    
    # ==========================================================
    # STEP 3: Process the stream and detect drifts
    # ==========================================================
    print("=" * 60)
    print("STEP 3: Processing the data stream...")
    print("=" * 60)
    
    detected_drifts = []
    
    # Process only first 40,000 samples for speed
    max_samples = 40_000
    
    for i, (x, y) in enumerate(stream):
        if i >= max_samples:
            break
            
        # Feed the sample to the detector
        is_drift = detector.update(x)
        
        # If drift detected, record it
        if is_drift:
            detected_drifts.append(i)
            print(f"  🔴 DRIFT DETECTED at sample {i:,}")
        
        # Progress indicator
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1:,} samples...")
    
    print()
    
    # ==========================================================
    # STEP 4: Compare with ground truth
    # ==========================================================
    print("=" * 60)
    print("STEP 4: Results - Comparing detected vs actual drifts")
    print("=" * 60)
    
    # Filter ground truth drifts within our processed range
    actual_drifts = [d for d in stream.drifts if d < max_samples]
    
    print(f"\nActual drift points (ground truth): {actual_drifts}")
    print(f"Detected drift points:              {detected_drifts}")
    print()
    
    # Simple evaluation
    print("Analysis:")
    print("-" * 40)
    
    for actual in actual_drifts:
        # Find closest detection
        if detected_drifts:
            closest = min(detected_drifts, key=lambda d: abs(d - actual))
            delay = closest - actual
            if abs(delay) < 2000:  # Within 2000 samples
                print(f"  ✅ Drift at {actual:,} detected at {closest:,} (delay: {delay:+,})")
            else:
                print(f"  ❌ Drift at {actual:,} NOT detected nearby")
        else:
            print(f"  ❌ Drift at {actual:,} NOT detected")
    
    # Check for false alarms
    false_alarms = []
    for detected in detected_drifts:
        is_near_actual = any(abs(detected - actual) < 2000 for actual in actual_drifts)
        if not is_near_actual:
            false_alarms.append(detected)
    
    if false_alarms:
        print(f"\n  ⚠️  False alarms (detections not near actual drifts): {false_alarms}")
    else:
        print(f"\n  ✅ No false alarms!")
    
    print()
    print("=" * 60)
    print("Demo complete! This shows how drift detectors work.")
    print("=" * 60)


if __name__ == "__main__":
    main()
