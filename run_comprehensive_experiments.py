"""
Master Experiment Runner — Runs All Comprehensive Experiments
==============================================================
Executes Experiments 5, 6, 7 in sequence and generates all figures.

Usage:
    python run_comprehensive_experiments.py          # Run all
    python run_comprehensive_experiments.py --exp 5  # Run only Exp 5
    python run_comprehensive_experiments.py --exp 6  # Run only Exp 6
    python run_comprehensive_experiments.py --exp 7  # Run only Exp 7

Author: Nusrat Begum
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive EADD experiments")
    parser.add_argument("--exp", type=int, choices=[5, 6, 7],
                        help="Run only a specific experiment (5, 6, or 7)")
    args = parser.parse_args()

    os.makedirs("experiments/figures", exist_ok=True)
    os.makedirs("experiments/results", exist_ok=True)

    experiments = {
        5: ("Experiment 5: Multi-Detector Synthetic Comparison",
            "experiment5_multidetector_synthetic", "run_experiment_5"),
        6: ("Experiment 6: Multi-Detector Real-World Benchmark",
            "experiment6_multidetector_realworld", "run_experiment_6"),
        7: ("Experiment 7: EADD Explainability Deep Dive",
            "experiment7_explainability_deep_dive", "run_experiment_7"),
    }

    to_run = [args.exp] if args.exp else [5, 6, 7]

    total_start = time.time()

    for exp_num in to_run:
        title, module_name, func_name = experiments[exp_num]

        print("\n" + "#" * 70)
        print(f"  {title}")
        print("#" * 70)

        start = time.time()
        module = __import__(module_name)
        run_fn = getattr(module, func_name)
        run_fn()
        elapsed = time.time() - start

        print(f"\n  {title} completed in {elapsed:.0f}s")

    total_elapsed = time.time() - total_start
    print("\n" + "#" * 70)
    print(f"  ALL EXPERIMENTS COMPLETE — Total time: {total_elapsed:.0f}s")
    print(f"  Figures: experiments/figures/")
    print(f"  Results: experiments/results/")
    print("#" * 70)

    # List generated files
    for d in ["experiments/figures", "experiments/results"]:
        if os.path.exists(d):
            files = sorted(os.listdir(d))
            if files:
                print(f"\n  {d}/")
                for f in files:
                    size = os.path.getsize(os.path.join(d, f))
                    print(f"    {f} ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
