# Enhanced Adversarial Drift Detection (EADD)

> **Thesis**: Master's thesis, Faculty of ICT, Mahidol University, 2026 (LaTeX source: [Nusrat-Thesis-Mahidol-University](https://github.com/NusratBegum/Nusrat-Thesis-Mahidol-University))
> **Author**: Nusrat Begum
> **Built on**: [DFKI-NI/unsupervised-concept-drift-detection](https://github.com/DFKI-NI/unsupervised-concept-drift-detection), the benchmark of [Lukats et al. (2024)](https://link.springer.com/article/10.1007/s41060-024-00620-y)

---

## Overview

EADD is an integrated framework for unsupervised feature drift detection. It combines:

1. A **LightGBM classifier two-sample test** (C2ST) on a reference and a current window
2. A **reservoir-sampled reference window** that summarises all data since the last confirmed drift
3. **Prediction entropy monitoring** ($H_{pred}$) of the discriminator's outputs
4. An **adaptive sequential permutation test (ASPT)** that confirms a clear drift after 20 of 199 permutations
5. A **TreeSHAP ranking** of the features behind each confirmed drift (reported, not evaluated in the thesis)

The thesis compares EADD with seven unsupervised detectors and runs an ablation with D3 using a LightGBM classifier to find which design choice causes the differences.

---

## Key Results (thesis, Chapter 4)

**Experiment 1, synthetic drift (8 detectors):** EADD is the only detector that detects all four drift types (abrupt, gradual, incremental, recurring) with zero false alarms. D3 and SPLL miss gradual and incremental drift.

**Experiment 2, 13 real-world streams (3 seeds, matched windows):**

| Dataset | MTD EADD | MTD D3 | Recall EADD | Recall D3 |
|---|---|---|---|---|
| InsectsAbrupt | **124** | 166 | 0.60 | 0.73 |
| InsectsGradual | -- | -- | 0.00 | 0.00 |
| InsectsIncrAbrupt | **50** | 126 | 1.00 | 1.00 |
| InsectsIncrRecur | **84** | 256 | 0.83 | 1.00 |

**Experiment 3, stable streams:** EADD raises no false alarms; D3 with small windows (100/50) raises up to 87.4 on autocorrelated data.

**Experiment 4, ablation:**
- D3 with LightGBM behaves like D3 (misses gradual and incremental drift), so the classifier is not EADD's advantage.
- D3's false alarms disappear with EADD's window sizes (500/200).
- The long-memory reference window is what lets EADD detect slow drift.
- At the default threshold (AUC 0.7), ASPT confirmed every alarm it tested; it halves false alarms only when the threshold is lowered to 0.5.

**Hypotheses:** H1a supported (sign test p = 0.001); H1b not supported (D3 faster where both detect, Wilcoxon p = 0.020); H2 supported at D3 threshold 0.6 (exact Mann-Whitney p = 0.014), explained by window size.

---

## How EADD Works

```
Data stream --> Step 1: reservoir-sampled W_ref (500), sliding W_cur (200)
            --> Step 2: LightGBM C2ST, 5-fold CV AUC and H_pred (every 50 samples)
            --> AUC > 0.7 ? --> Step 3: ASPT (B_max = 199, B_min = 10, alpha = 0.05)
            --> p < alpha ? --> Step 4: TreeSHAP feature ranking, reset W_ref
```

---

## Project Structure

```
unsupervised-concept-drift-detection/
|-- detectors/
|   |-- eadd.py                          # EADD detector
|   |-- d3_lightgbm.py                   # D3 with a LightGBM classifier (ablation)
|   |-- d3.py                            # D3 baseline (Gozuacik et al., 2019)
|   +-- ...                              # Other benchmark detectors
|-- experiment5_multidetector_synthetic.py   # Thesis Exp 1: 8 detectors, synthetic drift
|-- experiment_realworld_ablation.py         # Thesis Exp 2 and 4: 13 real-world streams, 3 seeds
|-- experiment6_multidetector_realworld.py   # Thesis Exp 2: supplementary 8-detector run
|-- experiment4_false_alarms.py              # Thesis Exp 3: stable streams
|-- experiment_d3_lightgbm.py                # Thesis Exp 4: synthetic ablation
|-- experiment_aspt_threshold.py             # Thesis Exp 4: AUC threshold study
|-- experiment_camera_ready.py               # JCSSE camera-ready runs
|-- EADD_Thesis_Experiments.ipynb            # Notebook (earlier Exp 2 run)
|-- test/detectors/                          # Unit tests, incl. test_eadd.py, test_d3_lightgbm.py
|-- datasets/                                # Dataset loaders (CSV files not in git)
+-- experiments/                             # results_*/ CSVs and figures
```

`experiment1_temporal_patterns.py`, `experiment3_explainability.py` and `experiment7_explainability_deep_dive.py` belong to earlier versions of the work and are not used in the final thesis.

---

## Installation

```bash
git clone https://github.com/NusratBegum/unsupervised-concept-drift-detection.git
cd unsupervised-concept-drift-detection
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m pytest -q test/detectors/test_eadd.py test/detectors/test_d3_lightgbm.py
```

---

## Running the Thesis Experiments

| Command | Thesis | Output |
|---|---|---|
| `python experiment5_multidetector_synthetic.py` | Exp 1 | `experiments/results/experiment5_multidetector_synthetic.csv` |
| `python experiment_realworld_ablation.py` | Exp 2, Exp 4 (real-world) | `experiments/results_realworld_ablation/` |
| `python experiment6_multidetector_realworld.py` | Exp 2 (supplementary) | `experiments/results/experiment6_multidetector_realworld.csv` |
| `python experiment4_false_alarms.py` | Exp 3 | `experiments/results/experiment4_false_alarms.csv` |
| `python experiment_d3_lightgbm.py` | Exp 4 (synthetic) | `experiments/results_d3_lightgbm/` |
| `python experiment_aspt_threshold.py` | Exp 4 (threshold) | `experiments/results_aspt_threshold/` |

Real-world experiments need the dataset CSVs in `datasets/files/` (see Datasets).

---

## Benchmark Detectors

| Detector | Method | Reference |
|----------|--------|-----------|
| **D3** | Logistic regression adversarial validation | Gozuacik et al., 2019 |
| **BNDM** | Bayesian non-parametric Polya tree test | -- |
| **CSDDM** | PCA + K-Means + statistical tests | 2021 |
| **IBDD** | Image-based comparison | 2020 |
| **NN-DVI** | Nearest neighbor density | 2018 |
| **OCDD** | One-class SVM | 2021 |
| **SPLL** | Semi-parametric log-likelihood | 2013 |
| **UCDD** | Clustering + pseudo-labels | 2020 |
| **UDetect** | Shewhart control charts | -- |
| **EDFS** | Ensemble feature subspaces | -- |

---

## Datasets

### Real-World (USP Data Stream Repository)

| Dataset | Samples | Features | Ground Truth |
|---------|---------|----------|-------------|
| INSECTS Abrupt Balanced | 52,848 | 33 | Yes |
| INSECTS Gradual Balanced | 24,150 | 33 | Yes |
| INSECTS Incremental Balanced | 57,018 | 33 | No |
| INSECTS Incr-Abrupt | 79,986 | 33 | Yes |
| INSECTS Incr-Reoccurring | 79,986 | 33 | Yes |
| Electricity | 45,312 | 8 | No |
| NOAA Weather | 18,159 | 8 | No |
| Outdoor Objects | 4,000 | 21 | No |
| Ozone | 2,534 | 72 | No |
| Poker Hand | 829,201 | 10 | No |
| Powersupply | 29,928 | 2 | No |
| Rialto Bridge | 82,250 | 27 | No |
| Luxembourg | 1,900 | 31 | No |

Setup: Download from [USP DS Repository](https://sites.google.com/view/uspdsrepository), copy to `datasets/files/`, then `python add_headers.py`.

---

## Modifications from Original Repo

### New Files (Our Contributions)

| File | Purpose |
|------|---------|
| `detectors/eadd.py` | **EADD detector** -- core thesis contribution |
| `detectors/d3_lightgbm.py` | D3 with a LightGBM classifier (ablation) |
| `experiment_d3_lightgbm.py`, `experiment_realworld_ablation.py`, `experiment_aspt_threshold.py` | Thesis ablation and threshold experiments |
| `test/detectors/test_eadd.py`, `test/detectors/test_d3_lightgbm.py` | Unit tests for EADD/ASPT and D3-LightGBM |
| `experiment1_temporal_patterns.py` | Experiment 1: temporal drift patterns |
| `experiment2_realworld_benchmark.py` | Experiment 2: real-world benchmark |
| `experiment3_explainability.py` | Experiment 3: SHAP explainability |
| `experiment4_false_alarms.py` | Experiment 4: false alarm evaluation |
| `run_all_experiments.py` | Master runner + hypothesis tests + LaTeX |
| `EADD_Thesis_Experiments.ipynb` | Interactive Jupyter notebook |
| `demo_eadd.py` | EADD demonstration script |
| `demo.py` | D3 demo on all datasets |
| `add_headers.py` | CSV header script for USP datasets |

### Modified Files

| File | Change |
|------|--------|
| `detectors/__init__.py` | Added EADD import |
| `requirements.txt` | Added lightgbm, shap, seaborn; `==` to `>=` |
| `datasets/airlines.py` | CSV instead of ARFF |
| `datasets/chess.py` | CSV, 8 features (at1-at8) |
| `datasets/electricity.py` | CSV instead of ARFF |
| `datasets/intrusion_detection.py` | CSV instead of ARFF |
| `datasets/keystroke.py` | CSV instead of ARFF |

---

## References

- **EADD Framework**: This thesis -- Nusrat Begum, Mahidol University, 2026
- **Lukats et al. (2024)**: [Benchmark of fully unsupervised concept drift detectors](https://link.springer.com/article/10.1007/s41060-024-00620-y)
- **D3**: Gozuacik et al. (2019), Discriminative Drift Detector
- **LightGBM**: Ke et al. (2017), [LightGBM](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
- **SHAP**: Lundberg and Lee (2017), [SHAP](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)
- **Original Repository**: [DFKI-NI/unsupervised-concept-drift-detection](https://github.com/DFKI-NI/unsupervised-concept-drift-detection)
- **USP DS Repository**: [sites.google.com/view/uspdsrepository](https://sites.google.com/view/uspdsrepository)

---

## License

**This repository is dual-licensed. See [NOTICE](NOTICE) for the full file-by-file breakdown.**

| Scope | License | File |
|---|---|---|
| Upstream code (Lukats et al. / DFKI base detectors, evaluation harness, datasets) | [BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause) | [LICENSE](LICENSE) |
| EADD contributions (© 2026 Nusrat Begum): `detectors/eadd.py`, all `experiment*.py`, `demo*.py`, `EADD_Thesis_Experiments.ipynb` | [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) | [LICENSE-EADD](LICENSE-EADD) |

**PolyForm Noncommercial 1.0.0** allows personal, academic, and non-profit research use of the EADD-specific files. **Commercial use** (including internal commercial operations) requires written permission from the author: `nusrat.beg@student.mahidol.ac.th`.

Every EADD-licensed source file carries an SPDX header:
```
# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
```

**Citation** (required for academic use):

```bibtex
@inproceedings{begum2026eadd,
  author    = {Begum, Nusrat and Yamchote, Phaphontee and
               Amornbunchornvej, Chainarong and Noraset, Thanapon},
  title     = {Enhanced Adversarial Drift Detection for MLOps
               Feature Monitoring},
  booktitle = {Proc. 23rd International Joint Conference on Computer
               Science and Software Engineering (JCSSE)},
  year      = {2026},
  publisher = {IEEE},
}
```
