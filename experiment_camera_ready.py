# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Camera-Ready Experiments for JCSSE 2026 (Round 3 Reviewer Comments)
====================================================================
Addresses reviewer comments:
  R2/R3: Runtime analysis EADD vs D3, varying d and N.
  R3:   Component ablation — LightGBM-only (no ASPT) vs LightGBM+ASPT vs D3.
  R2:   Performance under highly correlated features.

Outputs:
  experiments/results_cameraready/runtime_scaling.csv
  experiments/results_cameraready/ablation_synthetic.csv
  experiments/results_cameraready/ablation_falsealarms.csv
  experiments/results_cameraready/correlated_features.csv
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detectors.eadd import ExplainableAdversarialDriftDetector
from detectors.d3 import DiscriminativeDriftDetector2019
from experiment1_temporal_patterns import (
    generate_abrupt_drift,
    generate_gradual_drift,
    generate_incremental_drift,
    generate_recurring_drift,
)
from experiment4_false_alarms import (
    generate_stable_gaussian,
    generate_autocorrelated_stable,
    generate_heteroscedastic_stable,
    generate_correlated_stable,
)


# ──────────────────────────────────────────────────────────────
# EADD without ASPT (LightGBM-only ablation)
# ──────────────────────────────────────────────────────────────
class EADDNoASPT(ExplainableAdversarialDriftDetector):
    """EADD with ASPT disabled: drift confirmed when AUC > tau (no permutation test)."""

    def _aspt_test(self, ref_data, cur_data, actual_auc):
        # Bypass: emit p_value=0 so detection always passes the alpha gate.
        self.last_aspt_permutations = 0
        return 0.0


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def feed_stream(detector, X):
    detections = []
    t0 = time.perf_counter()
    for i in range(len(X)):
        feats = {f"F{j}": float(X[i, j]) for j in range(X.shape[1])}
        if detector.update(feats):
            detections.append(i)
    elapsed = time.perf_counter() - t0
    return detections, elapsed


def detected_within(detections, drift_point, tol_after=2000):
    return any(drift_point <= d <= drift_point + tol_after for d in detections)


def make_eadd(seed=42):
    return ExplainableAdversarialDriftDetector(
        n_reference_samples=500, n_current_samples=200, auc_threshold=0.7,
        n_permutations=199, significance_level=0.05, monitoring_frequency=50, seed=seed)


def make_eadd_noaspt(seed=42):
    return EADDNoASPT(
        n_reference_samples=500, n_current_samples=200, auc_threshold=0.7,
        n_permutations=199, significance_level=0.05, monitoring_frequency=50, seed=seed)


def make_d3(seed=42):
    return DiscriminativeDriftDetector2019(
        n_reference_samples=500, recent_samples_proportion=0.4, threshold=0.7, seed=seed)


# ──────────────────────────────────────────────────────────────
# 1. Runtime scaling: vary d (with fixed N), vary N (with fixed d)
# ──────────────────────────────────────────────────────────────
def runtime_scaling(out_dir):
    rows = []

    # Vary d (fixed n=8000 stable Gaussian, single drift at t=4000)
    for d in [5, 10, 20, 50, 100]:
        X_pre = np.random.default_rng(0).normal(0, 1, size=(4000, d))
        X_post = np.random.default_rng(1).normal(2.0, 1, size=(4000, d))
        X = np.vstack([X_pre, X_post])
        for name, ctor in [("EADD-Full", make_eadd),
                           ("EADD-NoASPT", make_eadd_noaspt),
                           ("D3", make_d3)]:
            dets, t = feed_stream(ctor(seed=42), X)
            rows.append(dict(experiment="vary_d", d=d, N=200, name=name,
                             time_s=round(t, 3), n_detections=len(dets)))
            print(f"  vary_d d={d:>3} {name:<12} {t:>7.2f}s  dets={len(dets)}", flush=True)

    # Vary N_cur (fixed d=10)
    for N in [100, 200, 500, 1000]:
        rng = np.random.default_rng(2)
        X_pre = rng.normal(0, 1, size=(4000, 10))
        X_post = rng.normal(2.0, 1, size=(4000, 10))
        X = np.vstack([X_pre, X_post])
        # Build EADD with custom N_cur (and matching N_ref=500 by default)
        for name, factory in [
            ("EADD-Full", lambda: ExplainableAdversarialDriftDetector(
                n_reference_samples=500, n_current_samples=N, auc_threshold=0.7,
                n_permutations=199, significance_level=0.05, monitoring_frequency=50, seed=42)),
            ("EADD-NoASPT", lambda: EADDNoASPT(
                n_reference_samples=500, n_current_samples=N, auc_threshold=0.7,
                n_permutations=199, significance_level=0.05, monitoring_frequency=50, seed=42)),
            ("D3", lambda: DiscriminativeDriftDetector2019(
                n_reference_samples=500, recent_samples_proportion=N/500.0,
                threshold=0.7, seed=42)),
        ]:
            dets, t = feed_stream(factory(), X)
            rows.append(dict(experiment="vary_N", d=10, N=N, name=name,
                             time_s=round(t, 3), n_detections=len(dets)))
            print(f"  vary_N N={N:>4} {name:<12} {t:>7.2f}s  dets={len(dets)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "runtime_scaling.csv"), index=False)
    return df


# ──────────────────────────────────────────────────────────────
# 2. Ablation on synthetic drift types (drift coverage)
# ──────────────────────────────────────────────────────────────
def ablation_synthetic(out_dir):
    drift_generators = {
        "Abrupt": lambda: generate_abrupt_drift(n_samples=10000, n_features=5, drift_point=5000),
        "Gradual": lambda: generate_gradual_drift(n_samples=10000, n_features=5),
        "Incremental": lambda: generate_incremental_drift(n_samples=10000, n_features=5),
        "Recurring": lambda: generate_recurring_drift(n_samples=10000, n_features=5),
    }
    rows = []
    for drift_name, gen in drift_generators.items():
        X, y, drift_points = gen()
        true_drift = drift_points[0]
        for det_name, ctor in [("EADD-Full", make_eadd),
                               ("EADD-NoASPT", make_eadd_noaspt),
                               ("D3", make_d3)]:
            dets, t = feed_stream(ctor(seed=42), X)
            ok = detected_within(dets, true_drift)
            rows.append(dict(drift_type=drift_name, detector=det_name,
                             detected=int(ok), n_detections=len(dets),
                             time_s=round(t, 3),
                             first_detection=(dets[0] if dets else None)))
            print(f"  {drift_name:<12} {det_name:<12} det={ok}  count={len(dets)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "ablation_synthetic.csv"), index=False)
    return df


# ──────────────────────────────────────────────────────────────
# 3. Ablation on stable streams (false alarms)
# ──────────────────────────────────────────────────────────────
def ablation_falsealarms(out_dir, n_runs=3):
    streams = {
        "Gaussian (i.i.d.)": generate_stable_gaussian,
        "Autocorrelated": generate_autocorrelated_stable,
        "Heteroscedastic": generate_heteroscedastic_stable,
        "Correlated": generate_correlated_stable,
    }
    rows = []
    for stream_name, gen in streams.items():
        for run in range(n_runs):
            X = gen(n_samples=10000, n_features=5, seed=100 + run)
            for det_name, ctor in [("EADD-Full", make_eadd),
                                   ("EADD-NoASPT", make_eadd_noaspt),
                                   ("D3", make_d3)]:
                dets, t = feed_stream(ctor(seed=100 + run), X)
                rows.append(dict(stream=stream_name, run=run, detector=det_name,
                                 false_alarms=len(dets), time_s=round(t, 3)))
                print(f"  {stream_name:<22} run={run} {det_name:<12} FA={len(dets)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "ablation_falsealarms.csv"), index=False)
    return df


# ──────────────────────────────────────────────────────────────
# 4. Correlated features test (R2)
# ──────────────────────────────────────────────────────────────
def correlated_features(out_dir):
    rows = []

    def gen_corr_drift(rho, n_samples=8000, n_features=10, drift_point=4000, seed=42):
        rng = np.random.default_rng(seed)
        cov = np.full((n_features, n_features), rho)
        np.fill_diagonal(cov, 1.0)
        L = np.linalg.cholesky(cov)
        # Pre-drift
        Z_pre = rng.normal(0, 1, size=(drift_point, n_features))
        X_pre = Z_pre @ L.T
        # Post-drift: shift mean of first 3 features by +2σ
        Z_post = rng.normal(0, 1, size=(n_samples - drift_point, n_features))
        X_post = Z_post @ L.T
        X_post[:, :3] += 2.0
        return np.vstack([X_pre, X_post]), drift_point

    for rho in [0.0, 0.5, 0.9]:
        X, dp = gen_corr_drift(rho)
        for det_name, ctor in [("EADD-Full", make_eadd),
                               ("EADD-NoASPT", make_eadd_noaspt),
                               ("D3", make_d3)]:
            dets, t = feed_stream(ctor(seed=42), X)
            ok = detected_within(dets, dp)
            rows.append(dict(rho=rho, detector=det_name,
                             detected=int(ok), n_detections=len(dets),
                             first_detection=(dets[0] if dets else None),
                             time_s=round(t, 3)))
            print(f"  rho={rho} {det_name:<12} det={ok}  count={len(dets)}  first={dets[0] if dets else None}", flush=True)

    # Also check stable correlated streams (no drift; measure FAs)
    for rho in [0.0, 0.5, 0.9]:
        rng = np.random.default_rng(7)
        cov = np.full((10, 10), rho)
        np.fill_diagonal(cov, 1.0)
        L = np.linalg.cholesky(cov)
        X = rng.normal(0, 1, size=(8000, 10)) @ L.T
        for det_name, ctor in [("EADD-Full", make_eadd),
                               ("EADD-NoASPT", make_eadd_noaspt),
                               ("D3", make_d3)]:
            dets, t = feed_stream(ctor(seed=42), X)
            rows.append(dict(rho=rho, detector=det_name + "(stable)",
                             detected=0, n_detections=len(dets),
                             first_detection=None, time_s=round(t, 3)))
            print(f"  STABLE rho={rho} {det_name:<12} FA={len(dets)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "correlated_features.csv"), index=False)
    return df


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "experiments", "results_cameraready")
    os.makedirs(out_dir, exist_ok=True)

    print("\n=== 1/4  Runtime scaling (vary d, vary N) ===", flush=True)
    df_rt = runtime_scaling(out_dir)

    print("\n=== 2/4  Ablation: synthetic drift coverage ===", flush=True)
    df_ab = ablation_synthetic(out_dir)

    print("\n=== 3/4  Ablation: false alarms on stable streams ===", flush=True)
    df_fa = ablation_falsealarms(out_dir)

    print("\n=== 4/4  Correlated features ===", flush=True)
    df_corr = correlated_features(out_dir)

    print(f"\nAll results saved to {out_dir}", flush=True)
