"""
Explainable Adversarial Drift Detection (EADD)
================================================
A novel framework that extends adversarial validation with:
  1. LightGBM as the adversarial classifier (non-linear)
  2. Permutation testing for statistical significance
  3. SHAP-based root cause analysis for feature attribution
  4. Automated drift prescriptions

Author: Nusrat Begum
Thesis: Feature Drift Detection via Adversarial Validation
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

    EADD extends the Discriminative Drift Detector (D3) with:
      - LightGBM for non-linear discrimination
      - Permutation testing for statistical significance (p-value)
      - SHAP feature attribution for root cause analysis
      - Automated drift prescriptions

    Parameters
    ----------
    n_reference_samples : int
        Number of samples in the reference window.
    n_current_samples : int
        Number of samples in the current/detection window.
    auc_threshold : float
        AUC threshold above which drift is suspected (before permutation test).
    n_permutations : int
        Number of permutations for the permutation test.
    significance_level : float
        Significance level (alpha) for the permutation test.
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
        significance_level: float = 0.01,
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

        # Last detection results (for querying after detection)
        self.last_auc = None
        self.last_p_value = None
        self.last_shap_importances = None
        self.last_prescription = None
        self.last_feature_names = None

    def update(self, features: dict) -> bool:
        """
        Update the detector with a new observation.

        Parameters
        ----------
        features : dict
            Feature dictionary {name: value}.

        Returns
        -------
        bool
            True if drift is detected, False otherwise.
        """
        feature_vector = np.fromiter(features.values(), dtype=float)
        if self.last_feature_names is None:
            self.last_feature_names = list(features.keys())
        self.samples_seen += 1
        self.steps_since_check += 1

        # Fill reference window
        if len(self.reference_window) < self.n_reference_samples:
            self.reference_window.append(feature_vector)
            return False

        # Update reference window via reservoir sampling
        if self.use_reservoir_sampling:
            prob = self.n_reference_samples / self.samples_seen
            if self.rng.random() < prob:
                idx = self.rng.integers(0, self.n_reference_samples)
                self.reference_window[idx] = feature_vector

        # Fill current window (sliding)
        self.current_window.append(feature_vector)
        if len(self.current_window) > self.n_current_samples:
            self.current_window.pop(0)

        # Check for drift at monitoring frequency
        if (self.steps_since_check >= self.monitoring_frequency and
                len(self.current_window) >= self.n_current_samples):
            self.steps_since_check = 0
            return self._detect_drift()

        return False

    def _detect_drift(self) -> bool:
        """Run the full EADD detection pipeline."""
        ref_data = np.array(self.reference_window)
        cur_data = np.array(self.current_window)

        # Step 2: Adversarial validation
        auc = self._adversarial_auc(ref_data, cur_data)
        self.last_auc = auc

        if auc < self.auc_threshold:
            self.last_p_value = 1.0
            self.last_shap_importances = None
            self.last_prescription = None
            return False

        # Step 3: Permutation test
        p_value = self._permutation_test(ref_data, cur_data, auc)
        self.last_p_value = p_value

        if p_value >= self.significance_level:
            self.last_shap_importances = None
            self.last_prescription = None
            return False

        # Step 4: SHAP feature attribution
        self.last_shap_importances = self._compute_shap(ref_data, cur_data)

        # Automated prescription
        self.last_prescription = self._generate_prescription(self.last_shap_importances)

        # Adapt to new distribution: seed reference with current data and reset
        self.reference_window = list(self.current_window)
        self.current_window = []
        self.samples_seen = len(self.reference_window)
        self.steps_since_check = 0

        return True

    def _adversarial_auc(self, ref_data: np.ndarray, cur_data: np.ndarray) -> float:
        """
        Train adversarial classifier and compute AUC via stratified cross-validation.
        """
        X = np.vstack([ref_data, cur_data])
        y = np.concatenate([np.zeros(len(ref_data)), np.ones(len(cur_data))])

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        predictions = np.zeros(len(y))

        for train_idx, test_idx in kfold.split(X, y):
            clf = self._create_classifier()
            clf.fit(X[train_idx], y[train_idx])
            predictions[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

        try:
            return roc_auc_score(y, predictions)
        except ValueError:
            return 0.5

    def _permutation_test(self, ref_data: np.ndarray, cur_data: np.ndarray,
                          actual_auc: float) -> float:
        """
        Permutation test: shuffle labels B times and compute null distribution of AUC.
        """
        X = np.vstack([ref_data, cur_data])
        y = np.concatenate([np.zeros(len(ref_data)), np.ones(len(cur_data))])

        count_ge = 0
        for _ in range(self.n_permutations):
            y_perm = self.rng.permutation(y)
            auc_perm = self._quick_auc(X, y_perm)
            if auc_perm >= actual_auc:
                count_ge += 1

        return count_ge / self.n_permutations

    def _quick_auc(self, X: np.ndarray, y: np.ndarray) -> float:
        """Quick AUC computation with a single train/test split for permutation efficiency."""
        # Use a fresh random state per call to avoid identical splits across permutations
        rs = int(self.rng.integers(0, 2**31))
        kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=rs)
        predictions = np.zeros(len(y))
        for train_idx, test_idx in kfold.split(X, y):
            clf = self._create_classifier()
            try:
                clf.fit(X[train_idx], y[train_idx])
                predictions[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
            except Exception:
                return 0.5
        try:
            return roc_auc_score(y, predictions)
        except ValueError:
            return 0.5

    def _compute_shap(self, ref_data: np.ndarray, cur_data: np.ndarray) -> Dict[str, float]:
        """
        Compute SHAP feature importances from the adversarial classifier.
        """
        X = np.vstack([ref_data, cur_data])
        y = np.concatenate([np.zeros(len(ref_data)), np.ones(len(cur_data))])

        clf = self._create_classifier()
        clf.fit(X, y)

        if HAS_SHAP:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # class 1 (current)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        else:
            # Fallback: use feature importances from LightGBM
            mean_abs_shap = clf.feature_importances_.astype(float)

        total = mean_abs_shap.sum()
        if total == 0:
            total = 1.0

        feature_names = self.last_feature_names or [f"F{i}" for i in range(len(mean_abs_shap))]
        importances = {}
        for i, name in enumerate(feature_names):
            importances[str(name)] = float(mean_abs_shap[i] / total * 100)

        # Sort by importance
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        return importances

    def _generate_prescription(self, importances: Dict[str, float]) -> Dict:
        """
        Generate automated drift prescription based on SHAP importance pattern.
        """
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        if not sorted_features:
            return {"type": "unknown", "message": "No feature importance data available."}

        top_feature_name, top_importance = sorted_features[0]

        # Scenario A: Univariate Drift
        if top_importance > 50:
            return {
                "type": "univariate",
                "dominant_feature": top_feature_name,
                "importance": top_importance,
                "message": (
                    f"Univariate drift detected on '{top_feature_name}' "
                    f"({top_importance:.1f}% contribution). "
                    f"Investigate data pipeline for this feature. "
                    f"Consider feature-specific preprocessing or exclude temporarily."
                )
            }

        # Scenario C: Feature Subset Drift
        cumulative = 0
        subset_features = []
        for fname, fimp in sorted_features:
            cumulative += fimp
            subset_features.append(fname)
            if cumulative > 70:
                break
        if 2 <= len(subset_features) <= 5:
            return {
                "type": "subset",
                "features": subset_features,
                "cumulative_importance": cumulative,
                "message": (
                    f"Correlated feature drift detected in {subset_features} "
                    f"({cumulative:.1f}% cumulative). "
                    f"Investigate common data source. Consider partial retraining."
                )
            }

        # Scenario B: Multivariate Concept Shift
        if top_importance <= 30:
            return {
                "type": "multivariate",
                "message": (
                    f"Multivariate concept shift detected. No single feature exceeds 30% importance. "
                    f"Full model retraining required. Expand training data to include recent distribution."
                )
            }

        # Default
        return {
            "type": "mixed",
            "message": (
                f"Mixed drift pattern detected. Top feature: '{top_feature_name}' "
                f"({top_importance:.1f}%). Review feature engineering and consider retraining."
            )
        }

    def _create_classifier(self):
        """Create a LightGBM classifier instance."""
        if HAS_LIGHTGBM:
            return lgb.LGBMClassifier(
                boosting_type='gbdt',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=-1,
                num_leaves=31,
                objective='binary',
                metric='auc',
                verbose=-1,
                random_state=self.seed,
                n_jobs=1,
            )
        else:
            # Fallback to sklearn GradientBoosting
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=self.seed,
            )

    def get_last_report(self) -> Dict:
        """
        Get a comprehensive report of the last detection.
        """
        return {
            "auc": self.last_auc,
            "p_value": self.last_p_value,
            "feature_importances": self.last_shap_importances,
            "prescription": self.last_prescription,
            "samples_seen": self.samples_seen,
        }
