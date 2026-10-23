from typing import Optional

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from .d3 import DiscriminativeDriftDetector2019


class DiscriminativeDriftDetectorLightGBM(DiscriminativeDriftDetector2019):
    """
    D3 with a LightGBM discriminator instead of logistic regression.

    Windowing, 2-fold cross-validation and the fixed AUC threshold are unchanged from D3. Only the classifier differs,
    and it uses the same LightGBM settings as EADD. This isolates the effect of the classifier choice when D3 and EADD
    are compared.

    Source: Gözüaçık, Ö.; Büyükçakır, A.; Bonab, H.; Can, F. (2019). Unsupervised concept drift detection with a
        discriminative classifier. Proceedings of the 28th ACM International Conference on Information and Knowledge
        Management. ACM.
    """

    def __init__(
        self,
        n_reference_samples: int = 100,
        recent_samples_proportion: float = 0.1,
        threshold: float = 0.7,
        seed: Optional[int] = None,
    ):
        """
        Init new D3-LightGBM instance.

        :param n_reference_samples: the number of data used to represent the current concept
        :param recent_samples_proportion: the proportion of data used to represent the new concept relative to the
            number of data used to represent current concept
        :param threshold: the threshold above which two concepts can be reliably discerned and a drift is signalled
        """
        super().__init__(n_reference_samples, recent_samples_proportion, threshold, seed)

    def _detect_drift(self) -> bool:
        """
        Detect if a drift occurred.

        :return: True if a drift occurred, else False
        """
        labels = self._get_labels()
        # same settings as EADD's discriminator (detectors/eadd.py)
        discriminator = lgb.LGBMClassifier(
            n_estimators=50,
            learning_rate=0.1,
            num_leaves=31,
            verbose=-1,
            random_state=42 if self.seed is None else int(self.seed),
            n_jobs=1,
        )
        predictions = self._predict(discriminator, np.array(self.data), labels)
        auc_score = roc_auc_score(labels, predictions)
        return auc_score >= self.threshold
