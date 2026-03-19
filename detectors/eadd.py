"""
Explainable Adversarial Drift Detection (EADD)
================================================
A novel framework that extends the classifier two-sample test (C2ST)
[Lopez-Paz & Oquab, 2017] with explainability and adaptive testing:

  1. LightGBM [Ke et al., 2017] as the discriminative classifier for
     the C2ST, replacing logistic regression with a non-linear model.
  2. Adaptive Sequential Permutation Testing (ASPT) [Gandy, 2009] for
     efficient p-value estimation with early stopping.
  3. SHAP interaction values [Lundberg et al., 2020] for correlated
     drift diagnosis via feature-level and pairwise attribution.
  4. Prediction Uncertainty Index (PUI) based on Shannon entropy
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
Thesis: Feature Drift Detection via Discriminative Classification
Mahidol University, 2026
"""

from typing import Optional, Dict, List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold
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
    """

    def __init__(
        self,
        n_reference_samples: int = 500,
        n_current_samples: int = 200,
        auc_threshold: float = 0.7,
        n_permutations: int = 50,
        significance_level: float = 0.05,
        use_reservoir_sampling: bool = True,
        monitoring_frequency: int = 50,
        seed: Optional[int] = None,
    ):
        super().__init__(seed)
        self.n_reference_samples = n_reference_samples
        self.n_current_samples = n_current_samples
        self.auc_threshold = auc_threshold
        self.n_permutations = n_permutations
        self.significance_level = significance_level
        self.use_reservoir_sampling = use_reservoir_sampling
        self.monitoring_frequency = monitoring_frequency

        self.reference_window: List[np.ndarray] = []
        self.current_window: List[np.ndarray] = []
        self.samples_seen = 0
        self.steps_since_check = 0
        self.rng = np.random.default_rng(seed)

        # Last detection results
        self.last_auc = None
        self.last_p_value = None
        self.last_pui = None
        self.last_dsi = None
        self.last_shap_importances = None
        self.last_interactions = None
        self.last_prescription = None
        self.last_feature_names = None
        self.last_aspt_permutations = None
        self.last_pui_trend: List[float] = []

    def update(self, features: dict) -> bool:
        feature_vector = np.fromiter(features.values(), dtype=float)
        if self.last_feature_names is None:
            self.last_feature_names = list(features.keys())
        self.samples_seen += 1
        self.steps_since_check += 1

        if len(self.reference_window) < self.n_reference_samples:
            self.reference_window.append(feature_vector)
            return False

        if self.use_reservoir_sampling:
            prob = self.n_reference_samples / self.samples_seen
            if self.rng.random() < prob:
                idx = self.rng.integers(0, self.n_reference_samples)
                self.reference_window[idx] = feature_vector

        self.current_window.append(feature_vector)
        if len(self.current_window) > self.n_current_samples:
            self.current_window.pop(0)

        if (self.steps_since_check >= self.monitoring_frequency and
                len(self.current_window) >= self.n_current_samples):
            self.steps_since_check = 0
            return self._detect_drift()

        return False

    def _detect_drift(self) -> bool:
        """Run the full EADD detection pipeline with ASPT and PUI."""
        ref_data = np.array(self.reference_window)
        cur_data = np.array(self.current_window)

        # Step 1: C2ST discriminative classification + PUI
        auc, predictions = self._adversarial_auc_with_preds(ref_data, cur_data)
        self.last_auc = auc
        self.last_pui = -np.mean(predictions * np.log(predictions + 1e-9) + (1 - predictions) * np.log(1 - predictions + 1e-9))
        self.last_pui_trend.append(self.last_pui)

        if auc < self.auc_threshold:
            self.last_p_value = 1.0
            return False

        # Step 2: Adaptive Sequential Permutation Test (ASPT)
        p_val = self._aspt_test(ref_data, cur_data, auc)
        self.last_p_value = p_val

        if p_val >= self.significance_level:
            return False

        # Step 3: Interaction-Aware Attribution
        self.last_shap_importances = self._compute_shap(ref_data, cur_data)

        # Step 4: DSI-based Prescription
        self.last_prescription = self._generate_prescription(self.last_shap_importances)

        # Adapt
        self.reference_window = list(self.current_window)
        self.current_window = []
        self.samples_seen = len(self.reference_window)
        self.steps_since_check = 0
        return True

    def _aspt_test(self, ref_data: np.ndarray, cur_data: np.ndarray, actual_auc: float) -> float:
        X = np.vstack([ref_data, cur_data])
        y = np.concatenate([np.zeros(len(ref_data)), np.ones(len(cur_data))])
        count_ge = 0
        min_perms = 10
        alpha = self.significance_level
        for i in range(1, self.n_permutations + 1):
            y_perm = self.rng.permutation(y)
            auc_perm = self._quick_auc(X, y_perm)
            if auc_perm >= actual_auc: count_ge += 1
            # Conservative early-reject rule: if no exceedances were observed,
            # the best possible permutation p-value after i draws is 1/(i+1).
            if count_ge == 0 and i >= min_perms:
                if (1.0 / (i + 1)) < alpha:
                    self.last_aspt_permutations = i
                    return 1.0 / (i + 1)
            if count_ge / i > 2 * alpha and i >= min_perms:
                self.last_aspt_permutations = i
                return 1.0
        self.last_aspt_permutations = self.n_permutations
        return (count_ge + 1) / (self.n_permutations + 1)

    def _adversarial_auc_with_preds(self, ref_data: np.ndarray, cur_data: np.ndarray) -> Tuple[float, np.ndarray]:
        X = np.vstack([ref_data, cur_data])
        y = np.concatenate([np.zeros(len(ref_data)), np.ones(len(cur_data))])
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        predictions = np.zeros(len(y))
        for train_idx, test_idx in kfold.split(X, y):
            clf = self._create_classifier()
            clf.fit(X[train_idx], y[train_idx])
            predictions[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        try:
            return roc_auc_score(y, predictions), predictions
        except ValueError:
            return 0.5, predictions

    def _quick_auc(self, X: np.ndarray, y: np.ndarray) -> float:
        rs = int(self.rng.integers(0, 2**31))
        kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=rs)
        predictions = np.zeros(len(y))
        for train_idx, test_idx in kfold.split(X, y):
            clf = self._create_classifier()
            try:
                clf.fit(X[train_idx], y[train_idx])
                predictions[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
            except Exception: return 0.5
        try:
            return roc_auc_score(y, predictions)
        except ValueError: return 0.5

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
            "pui": self.last_pui,
            "pui_trend": list(self.last_pui_trend),
            "dsi": self.last_dsi,
            "aspt_permutations_used": self.last_aspt_permutations,
            "aspt_permutations_budget": self.n_permutations,
            "feature_importances": self.last_shap_importances,
            "interactions": self.last_interactions,
            "prescription": self.last_prescription,
            "samples_seen": self.samples_seen,
        }
