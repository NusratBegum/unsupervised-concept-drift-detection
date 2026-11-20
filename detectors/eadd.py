# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# See LICENSE-EADD for full terms. Noncommercial use only.
"""
Enhanced Adversarial Drift Detection (EADD)
============================================
A framework that extends the classifier two-sample test (C2ST)
[Lopez-Paz & Oquab, 2017] with feature attribution and adaptive testing.
Items 3 and 5 (interaction values, DSI) were used in the JCSSE 2026 paper;
the final thesis reports the TreeSHAP ranking but not the DSI.

  1. LightGBM [Ke et al., 2017] as the discriminative classifier for
     the C2ST, replacing logistic regression with a non-linear model.
  2. Adaptive Sequential Permutation Testing (ASPT) [Gandy, 2009] for
     efficient p-value estimation with early stopping.
  3. SHAP interaction values [Lundberg et al., 2020] for correlated
     drift diagnosis via feature-level and pairwise attribution.
  4. Prediction Entropy ($H_{\text{pred}}$) based on Shannon entropy
     [Shannon, 1948] of classifier predictions for early warning.
  5. Drift Severity Index (DSI) combining AUC with normalised SHAP
     entropy for automated severity prescriptions.
  6. Reservoir sampling [Vitter, 1985] for representative reference
     windows in streaming settings.

The term "Adversarial" in EADD refers to the C2ST paradigm where a
discriminative classifier is trained to distinguish reference from
current data---not adversarial attacks in the security sense.

References
----------
- Gandy (2009). Sequential Monte Carlo tests. JASA 104(488):1504-1511.
- Ke et al. (2017). LightGBM. NeurIPS.
- Lopez-Paz & Oquab (2017). Classifier two-sample tests. ICLR.
- Lundberg & Lee (2017). SHAP. NeurIPS.
- Lundberg, Erion & Lee (2020). SHAP interaction values. arXiv:1802.03888.
- Shannon (1948). A mathematical theory of communication. Bell Syst. Tech. J.
- Vitter (1985). Random sampling with a reservoir. ACM TOMS 11(1):37-57.

Author: Nusrat Begum
Thesis: Enhanced Adversarial Drift Detection for MLOps Feature Monitoring
Mahidol University, 2026
"""

from collections import deque
from typing import Optional, Dict, List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from .base import UnsupervisedDriftDetector


class ExplainableAdversarialDriftDetector(UnsupervisedDriftDetector):
    """
    Explainable Adversarial Drift Detection (EADD).

    Parameters
    ----------
    n_reference_samples : int
        Number of samples in the reference window.
    n_current_samples : int
        Number of samples in the current/detection window.
    auc_threshold : float
        AUC threshold above which drift is suspected.
    n_permutations : int
        Max permutations for the ASPT test.
    significance_level : float
        Alpha for drift confirmation.
    use_reservoir_sampling : bool
        Whether to use reservoir sampling for the reference window.
    monitoring_frequency : int
        How often (in samples) to run detection.
    seed : int or None
        Random seed.
    permutation_test : str
        "aspt" (sequential stopping, default), "full" (all n_permutations)
        or "none" (accept every alarm with AUC above auc_threshold).
    min_permutations : int
        Minimum number of permutations before ASPT may stop early.
    n_folds : int
        Cross-validation folds for the AUC (D3 uses 2).
    sliding_reference : bool
        Without reservoir sampling, keep the reference as the samples just
        before the current window, as D3 does, instead of fixing it.
    reservoir_after_current : bool
        Offer a sample to the reservoir only once it leaves the current
        window, so no sample is in both windows (False in the thesis runs).
    perm_n_folds : int or None
        CV folds of the permuted AUCs. The thesis used 2 while the observed
        AUC used n_folds, so the null did not match the statistic; None
        uses n_folds, which makes the permutation p-value valid.
    aspt_alpha_split : float or None
        None is the thesis stopping rule, which is not an exact level-alpha
        test (simulated Type I 0.059 at alpha 0.05). A fraction f spends
        f * alpha on the early reject (no exceedance in the first i draws,
        1/(i+1) <= f * alpha) and the rest on the final test, so by the
        union bound the Type I error is at most alpha.
    block_size : int
        1 treats samples as exchangeable. b > 1 groups each window into
        blocks of b samples that are consecutive in time; the CV keeps a
        block in one fold and the permutation test swaps whole blocks, so
        autocorrelated neighbours cannot leak between train and test.
        Not a valid null: a reference block spans a much longer time than a
        current-window block, so swapped blocks are not exchangeable
        (rejects 68% of no-drift tests at AR 0.9).
    null : str
        "permutation" (thesis): the null AUCs come from shuffled window
        labels, which assumes exchangeable samples. "history": the null AUCs
        are those of earlier windows of the same stream (contiguous, as the
        current window) against the reservoir, so the null includes the
        stream's normal variation, e.g. autocorrelation. Needs
        reservoir_after_current.
    history_size, history_stride : int
        Samples kept for the history null, and the step between the starts
        of its windows.
    null_match_size : bool
        History null only: the observed and every null AUC use a reference of
        n_reference_samples - n_current_samples samples (default False).
    min_null_draws : int, optional
        History null only: no test, and so no alarm, while fewer earlier
        windows exist (default None: test with any number).
    """

    def __init__(
        self,
        n_reference_samples: int = 500,
        n_current_samples: int = 200,
        auc_threshold: float = 0.7,
        n_permutations: int = 199,
        significance_level: float = 0.05,
        use_reservoir_sampling: bool = True,
        monitoring_frequency: int = 50,
        seed: Optional[int] = None,
        permutation_test: str = "aspt",
        min_permutations: int = 10,
        n_folds: int = 5,
        sliding_reference: bool = False,
        reservoir_after_current: bool = False,
        perm_n_folds: Optional[int] = 2,
        aspt_alpha_split: Optional[float] = None,
        block_size: int = 1,
        null: str = "permutation",
        history_size: int = 5000,
        history_stride: int = 50,
        min_null_draws: Optional[int] = None,
        null_match_size: bool = False,
    ):
        super().__init__(seed)
        self.n_reference_samples = n_reference_samples
        self.n_current_samples = n_current_samples
        self.auc_threshold = auc_threshold
        self.n_permutations = n_permutations
        self.significance_level = significance_level
        self.use_reservoir_sampling = use_reservoir_sampling
        self.monitoring_frequency = monitoring_frequency
        self.n_folds = n_folds
        assert not (sliding_reference and use_reservoir_sampling), "sliding_reference replaces reservoir sampling"
        self.sliding_reference = sliding_reference
        # False reproduces the thesis runs, where a sample can be in both windows at once
        self.reservoir_after_current = reservoir_after_current
        self.perm_n_folds = perm_n_folds or n_folds
        assert aspt_alpha_split is None or 0 < aspt_alpha_split < 1
        self.aspt_alpha_split = aspt_alpha_split
        self.block_size = block_size
        assert null in ("permutation", "history"), null
        assert null == "permutation" or reservoir_after_current, "history null needs reservoir_after_current"
        self.null = null
        self.history_stride = history_stride
        # history null: no test (and no alarm) until this many earlier windows exist;
        # right after an alarm the few null windows overlap heavily and the test is too liberal
        self.min_null_draws = min_null_draws
        # history null: observed and null AUCs both use a reference of
        # n_reference_samples - n_current_samples samples (a null window's samples
        # are removed from the reference, so null draws have fewer reference samples)
        self.null_match_size = null_match_size
        self.history = deque(maxlen=history_size)  # samples since the last reset, with arrival time
        self.n_updates = 0

        self.reference_window: List[np.ndarray] = []
        self.current_window: List[np.ndarray] = []
        self.samples_seen = 0
        self.steps_since_check = 0
        self.rng = np.random.default_rng(seed)
        # Separate generator for the permutation test, so that the reservoir
        # samples do not depend on how many permutations were drawn.
        self.perm_rng = np.random.default_rng(None if seed is None else seed + 1)
        if permutation_test not in ("aspt", "full", "none"):
            raise ValueError(f"unknown permutation_test: {permutation_test}")
        self.permutation_test = permutation_test
        self.min_permutations = min_permutations
        # Permutation test statistics over the whole stream
        self.n_tests = 0
        self.n_early_stops = 0
        self.n_rejections = 0
        self.n_permutations_used = 0

        # Last detection results
        self.last_auc = None
        self.last_p_value = None
        self.last_hpred = None
        self.last_dsi = None
        self.last_shap_importances = None
        self.last_interactions = None
        self.last_prescription = None
        self.last_feature_names = None
        self.last_aspt_permutations = None
        self.last_hpred_trend: List[float] = []

    def update(self, features: dict) -> bool:
        # the arrival time rides along as the last element, so it follows a
        # sample through the reservoir; _detect_drift strips it
        feature_vector = np.append(np.fromiter(features.values(), dtype=float), self.n_updates)
        self.n_updates += 1
        if self.last_feature_names is None:
            self.last_feature_names = list(features.keys())
        self.samples_seen += 1
        self.steps_since_check += 1

        self.history.append(feature_vector)
        if len(self.reference_window) < self.n_reference_samples:
            self.reference_window.append(feature_vector)
            return False

        if self.use_reservoir_sampling and not self.reservoir_after_current:
            self._offer_to_reservoir(feature_vector, self.samples_seen)

        self.current_window.append(feature_vector)
        if len(self.current_window) > self.n_current_samples:
            oldest = self.current_window.pop(0)
            if self.sliding_reference:
                self.reference_window.pop(0)
                self.reference_window.append(oldest)
            elif self.use_reservoir_sampling and self.reservoir_after_current:
                # the oldest sample arrived n_current_samples samples ago
                self._offer_to_reservoir(oldest, self.samples_seen - self.n_current_samples)

        if (self.steps_since_check >= self.monitoring_frequency and
                len(self.current_window) >= self.n_current_samples):
            self.steps_since_check = 0
            return self._detect_drift()

        return False

    def _offer_to_reservoir(self, x, index):
        # reservoir sampling (Algorithm R); index = position of x since the last reset
        if self.rng.random() < self.n_reference_samples / index:
            self.reference_window[self.rng.integers(0, self.n_reference_samples)] = x

    def _detect_drift(self) -> bool:
        """Run the full EADD detection pipeline with ASPT and H_pred."""
        ref_all = np.array(self.reference_window)
        cur_all = np.array(self.current_window)
        ref_data, cur_data = ref_all[:, :-1], cur_all[:, :-1]
        groups = self._time_blocks(ref_all[:, -1], cur_all[:, -1])

        # Step 1: C2ST discriminative classification + H_pred
        auc, predictions = self._adversarial_auc_with_preds(ref_data, cur_data, groups)
        self.last_auc = auc
        self.last_hpred = -np.mean(predictions * np.log(predictions + 1e-9) + (1 - predictions) * np.log(1 - predictions + 1e-9))
        self.last_hpred_trend.append(self.last_hpred)

        if auc < self.auc_threshold:
            self.last_p_value = 1.0
            return False

        # Step 2: Adaptive Sequential Permutation Test (ASPT)
        if self.permutation_test == "none":
            self.last_aspt_permutations = 0
            p_val = 0.0
            self.last_aspt_reject = True
        else:
            p_val = self._aspt_test(ref_data, cur_data, auc, groups)
            self.n_tests += 1
            self.n_permutations_used += self.last_aspt_permutations
        self.last_p_value = p_val

        if not self.last_aspt_reject:
            self.n_rejections += 1
            return False

        # Step 3: Interaction-Aware Attribution
        self.last_shap_importances = self._compute_shap(ref_data, cur_data)

        # Step 4: DSI-based Prescription
        self.last_prescription = self._generate_prescription(self.last_shap_importances)

        # Adapt
        self.reference_window = list(self.current_window)
        self.history = deque(self.current_window, maxlen=self.history.maxlen)
        self.current_window = []
        self.samples_seen = len(self.reference_window)
        self.steps_since_check = 0
        return True

    def _aspt_test(self, ref_data: np.ndarray, cur_data: np.ndarray, actual_auc: float,
                   groups: Optional[np.ndarray] = None) -> float:
        """Permutation p-value of the observed AUC; sets self.last_aspt_reject."""
        X = np.vstack([ref_data, cur_data])
        y = np.concatenate([np.zeros(len(ref_data)), np.ones(len(cur_data))])
        alpha = self.significance_level
        if self.aspt_alpha_split is None:  # thesis rule
            alpha_early, alpha_final = alpha, alpha
        else:
            alpha_early = self.aspt_alpha_split * alpha
            alpha_final = alpha - alpha_early
        count_ge = 0
        early = self.permutation_test == "aspt"
        if self.null == "history":
            draw, n_draws = self._history_null()
            if self.min_null_draws is not None and n_draws < self.min_null_draws:
                self.last_aspt_permutations = 0
                self.last_aspt_reject = False
                return 1.0
            if self.null_match_size:
                m = self.n_reference_samples - self.n_current_samples
                sub = ref_data[self.perm_rng.choice(len(ref_data), min(m, len(ref_data)), replace=False)]
                actual_auc = self._cv_auc(np.vstack([sub, cur_data]),
                                          np.concatenate([np.zeros(len(sub)), np.ones(len(cur_data))]),
                                          self.perm_n_folds, int(self.perm_rng.integers(0, 2**31)))
        else:
            n_draws = self.n_permutations
            draw = lambda: self._cv_auc(X, self._permute_labels(y, groups), self.perm_n_folds,
                                        int(self.perm_rng.integers(0, 2**31)), groups)
        self.last_aspt_permutations = 0
        for i in range(1, n_draws + 1):
            auc_perm = draw()
            if auc_perm >= actual_auc: count_ge += 1
            if not early or i < self.min_permutations or i == n_draws:
                continue
            # Early reject: no exceedance in the first i draws, which happens
            # with probability 1/(i+1) under H0.
            if count_ge == 0 and (1.0 / (i + 1)) <= alpha_early and (
                    self.aspt_alpha_split is not None or 1.0 / (i + 1) < alpha):
                self.last_aspt_permutations = i
                self.n_early_stops += 1
                self.last_aspt_reject = True
                return 1.0 / (i + 1)
            # Early accept (futility): it can only lower the Type I error.
            if count_ge / i > 2 * alpha:
                self.last_aspt_permutations = i
                self.n_early_stops += 1
                self.last_aspt_reject = False
                return (count_ge + 1) / (i + 1)
        self.last_aspt_permutations = n_draws
        p = (count_ge + 1) / (n_draws + 1)
        self.last_aspt_reject = p < alpha if self.aspt_alpha_split is None else p <= alpha_final
        return p

    def _history_null(self):
        """Draws of the AUC of an earlier window of the same stream against the reservoir."""
        n = self.n_current_samples
        past = np.array(self.history)[:-n]  # everything before the current window
        starts = np.arange(0, len(past) - n + 1, self.history_stride)
        starts = self.perm_rng.permutation(starts)[:self.n_permutations]
        ref = np.array(self.reference_window)
        it = iter(starts)

        def draw():
            w = past[next(it):][:n]
            r = ref[~np.isin(ref[:, -1], w[:, -1])]  # a window sample is not also a reference sample
            if self.null_match_size:
                m = self.n_reference_samples - self.n_current_samples
                r = r[self.perm_rng.choice(len(r), min(m, len(r)), replace=False)]
            X = np.vstack([r[:, :-1], w[:, :-1]])
            y = np.concatenate([np.zeros(len(r)), np.ones(len(w))])
            return self._cv_auc(X, y, self.perm_n_folds, int(self.perm_rng.integers(0, 2**31)))
        return draw, len(starts)

    def _time_blocks(self, t_ref: np.ndarray, t_cur: np.ndarray) -> Optional[np.ndarray]:
        """Block id per sample: consecutive (in time) runs of block_size within each window."""
        if self.block_size <= 1:
            return None
        def blocks(t, offset):
            rank = np.empty(len(t), dtype=int)
            rank[np.argsort(t, kind="stable")] = np.arange(len(t))
            return offset + rank // self.block_size
        g_ref = blocks(t_ref, 0)
        return np.concatenate([g_ref, blocks(t_cur, g_ref.max() + 1)])

    def _permute_labels(self, y: np.ndarray, groups: Optional[np.ndarray]) -> np.ndarray:
        if groups is None:
            return self.perm_rng.permutation(y)
        # swap whole blocks: as many blocks get label 1 as the current window has
        ids = np.unique(groups)
        n_cur_blocks = len(np.unique(groups[y == 1]))
        cur_ids = self.perm_rng.choice(ids, n_cur_blocks, replace=False)
        return np.isin(groups, cur_ids).astype(float)

    def _adversarial_auc_with_preds(self, ref_data: np.ndarray, cur_data: np.ndarray,
                                    groups: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        X = np.vstack([ref_data, cur_data])
        y = np.concatenate([np.zeros(len(ref_data)), np.ones(len(cur_data))])
        return self._cv_auc(X, y, self.n_folds, self.seed, groups, return_predictions=True)

    def _cv_auc(self, X, y, n_splits, random_state, groups=None, return_predictions=False):
        """Cross-validated AUC; the observed and the permuted AUCs both use this."""
        if groups is None:
            folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X, y)
        else:
            folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                         random_state=random_state).split(X, y, groups)
        predictions = np.zeros(len(y))
        for train_idx, test_idx in folds:
            clf = self._create_classifier()
            try:
                clf.fit(X[train_idx], y[train_idx])
                predictions[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
            except Exception:
                predictions[test_idx] = 0.5
        try:
            auc = roc_auc_score(y, predictions)
        except ValueError:
            auc = 0.5
        return (auc, predictions) if return_predictions else auc

    def _compute_shap(self, ref_data: np.ndarray, cur_data: np.ndarray) -> Dict[str, float]:
        X = np.vstack([ref_data, cur_data])
        y = np.concatenate([np.zeros(len(ref_data)), np.ones(len(cur_data))])
        clf = self._create_classifier()
        clf.fit(X, y)
        if HAS_SHAP:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list): shap_values = shap_values[1]
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            try:
                X_sample = X if len(X) < 100 else X[self.rng.choice(len(X), 100, replace=False)]
                interact_vals = explainer.shap_interaction_values(X_sample)
                if isinstance(interact_vals, list): interact_vals = interact_vals[1]
                self.last_interactions = np.mean(np.abs(interact_vals), axis=0)
            except Exception: self.last_interactions = None
        else:
            mean_abs_shap = clf.feature_importances_.astype(float)
            self.last_interactions = None
        total = mean_abs_shap.sum() or 1.0
        self.last_shap_vector = mean_abs_shap / total
        importances = {str(self.last_feature_names[i]): float(self.last_shap_vector[i] * 100) 
                       for i in range(len(mean_abs_shap))}
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    def _generate_prescription(self, importances: Dict[str, float]) -> Dict:
        if not hasattr(self, 'last_shap_vector'): return {"type": "unknown"}
        s = self.last_shap_vector
        h_norm = (-np.sum(s * np.log2(s + 1e-9))) / np.log2(len(s)) if len(s) > 1 else 0
        dsi = self.last_auc * (1 - h_norm)
        self.last_dsi = dsi
        top_feat = list(importances.keys())[0]
        if dsi > 0.6:
            return {"type": "critical", "dsi": dsi, "message": f"CRITICAL: Sensor failure on '{top_feat}' (DSI={dsi:.2f}). Audit pipeline."}
        elif dsi > 0.3:
            return {"type": "moderate", "dsi": dsi, "message": f"MODERATE: Subset drift (DSI={dsi:.2f}). Investigate shared source."}
        return {"type": "low", "dsi": dsi, "message": f"LOW: Diffuse shift (DSI={dsi:.2f}). Scheduled retraining."}

    def _create_classifier(self):
        model_seed = 42 if self.seed is None else int(self.seed)
        if HAS_LIGHTGBM:
            return lgb.LGBMClassifier(
                n_estimators=50,
                learning_rate=0.1,
                num_leaves=31,
                verbose=-1,
                random_state=model_seed,
                n_jobs=1,
            )
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=50, random_state=model_seed)

    def get_last_report(self) -> Dict:
        return {
            "auc": self.last_auc,
            "p_value": self.last_p_value,
            "hpred": self.last_hpred,
            "hpred_trend": list(self.last_hpred_trend),
            "dsi": self.last_dsi,
            "aspt_permutations_used": self.last_aspt_permutations,
            "aspt_permutations_budget": self.n_permutations,
            "feature_importances": self.last_shap_importances,
            "interactions": self.last_interactions,
            "prescription": self.last_prescription,
            "samples_seen": self.samples_seen,
        }
