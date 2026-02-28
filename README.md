# Explainable Adversarial Drift Detection (EADD)

> **Thesis**: *Feature Drift Detection via Adversarial Validation*
> **Author**: Nusrat Begum — Mahidol University, ICT, 2026
> **Built on**: [DFKI-NI/unsupervised-concept-drift-detection](https://github.com/DFKI-NI/unsupervised-concept-drift-detection) benchmark by [Lukats et al. (2025)](https://link.springer.com/article/10.1007/s41060-024-00620-y)

---

## Table of Contents

1. [Overview](#overview)
2. [Key Results](#key-results)
3. [What is Feature Drift?](#what-is-feature-drift)
4. [How EADD Works](#how-eadd-works)
5. [Project Structure](#project-structure)
6. [Installation](#installation)
7. [Running the Experiments](#running-the-experiments)
8. [Experiment Details](#experiment-details)
9. [Hypothesis Testing](#hypothesis-testing)
10. [Benchmark Detectors](#benchmark-detectors)
11. [Datasets](#datasets)
12. [Modifications from Original Repo](#modifications-from-original-repo)
13. [References](#references)

---

## Overview

**EADD (Explainable Adversarial Drift Detection)** is a novel framework for unsupervised feature drift detection that extends adversarial validation with:

1. **LightGBM** as the adversarial classifier (non-linear, captures feature interactions)
2. **Permutation testing** for statistically principled threshold calibration (p < 0.01)
3. **SHAP-based root cause analysis** that identifies *which* features drifted
4. **Automated prescriptions** that recommend remediation actions

Unlike existing detectors (e.g., D3) that only signal *that* drift occurred, EADD transforms drift detection from a binary alarm into an **intelligent diagnostic tool** — answering both *"Is there drift?"* and *"What caused it?"*.

### The Problem EADD Solves

| Existing Detectors | EADD |
|---|---|
| "Drift detected" (binary alarm) | "Drift detected: **Feature Age** (45%), **Income** (25%)" |
| No guidance on response | "Prescription: Univariate drift on Age — investigate data pipeline" |
| Fixed thresholds causing false alarms | Permutation test provides statistical validation (p < 0.01) |
| Linear classifiers miss interactions | LightGBM captures non-linear multivariate shifts |

---

## Key Results

Results from our four experiments (synthetic + 13 real-world datasets):

### Experiment 1: Temporal Drift Pattern Sensitivity

| Drift Type | EADD Delay | D3 Delay | EADD Success | D3 Success |
|---|---|---|---|---|
| **Abrupt** | 129 | 122 | **100%** | 100% |
| **Gradual** | 1,309 | -- | **100%** | **0%** |
| **Incremental** | 1,349 | -- | **100%** | **0%** |
| **Recurring** | 146 | 221 | **100%** | 100% |

EADD detects **all 4 drift types** (100%), D3 only detects 2/4 (misses gradual and incremental).

### Experiment 3: Explainability (SHAP Attribution)

| Scenario | Target | EADD Top Feature | SHAP % | Prescription |
|---|---|---|---|---|
| Univariate | F3 | **F3** | 49.7% | Correct |
| Subset | F2, F5, F7 | **F5** (22.3%) | 75% combined | Correct |
| Multivariate | All features | Distributed | Max 14.1% | Correct |

SHAP correctly identifies the drifting features in **all 3 scenarios**.

### Experiment 4: False Alarm Robustness

| Stream Type | EADD | D3 (t=0.6) | D3 (t=0.7) | D3 (t=0.8) |
|---|---|---|---|---|
| Gaussian (i.i.d.) | **0** | 10.4 | 0 | 0 |
| Autocorrelated | **0** | 87.4 | 54.6 | 13.8 |
| Heteroscedastic | **0** | 10.4 | 0 | 0 |
| Correlated | **0** | 10.0 | 0 | 0 |

EADD produces **zero false alarms** across all stable stream types.

### Hypothesis Testing Summary

| Hypothesis | Test | Result | p-value |
|---|---|---|---|
| H1a: EADD detects more drift types | Binomial | 4/4 vs 2/4 | 0.69 |
| H2: EADD FA < D3(t=0.6) FA | Mann-Whitney U | **Supported** | **0.0101** |
| H3: SHAP correct attribution | Qualitative | **3/3 correct** | -- |

---

## What is Feature Drift?

In machine learning, **feature drift** (also called covariate shift or virtual concept drift) occurs when the distribution of input features P(X) changes over time, while the relationship between inputs and outputs P(y|X) may or may not change.

| Type | Definition | What Changes | Requires Labels? |
|------|-----------|--------------|------------------|
| **Feature Drift** | P(X) changes | Input feature distributions | No (unsupervised) |
| **Concept Drift** | P(y\|X) changes | Input-output relationship | Yes (supervised) |

### Why Does It Matter?

If left undetected, drift makes ML models unreliable. EADD enables:
- **Early warning** before performance degrades (no labels needed)
- **Root cause diagnosis** -- which features drifted and by how much
- **Targeted remediation** -- fix the pipeline, not blindly retrain

### Temporal Drift Patterns

| Pattern | Description | EADD Detection |
|---------|-------------|----------------|
| **Abrupt** | Sudden distribution change | Fast (129 samples) |
| **Gradual** | Old/new distributions intermixed during transition | Detected (1,309 samples) |
| **Incremental** | Slow continuous change over time | Detected (1,349 samples) |
| **Recurring** | Periodic distribution oscillation | Fast (146 samples) |

---

## How EADD Works

### Pipeline Overview

```
Data Stream --> Step 1: Windowing (Reservoir Sampling)
            --> Step 2: Adversarial Validation (LightGBM AUC)
            --> Step 3: Permutation Test (p < 0.01)
            --> Step 4: SHAP Feature Attribution + Prescription
```

### Step 1: Adaptive Reference Windowing
- **Reference window** (W_ref = 500 samples): maintained via **reservoir sampling** to capture global distribution
- **Current window** (W_cur = 200 samples): sliding window of most recent data
- After confirmed drift: reference resets to current data

### Step 2: Adversarial Validation
- Label reference as class 0, current as class 1
- Train **LightGBM** classifier (5-fold stratified CV)
- Compute **AUC-ROC** -- if AUC > 0.7, drift is suspected

### Step 3: Permutation Test
- Shuffle labels 50x, retrain each time, build null AUC distribution
- p-value = #(AUC_perm >= AUC_actual) / B
- Drift confirmed **only if p < 0.01** (99% confidence)

### Step 4: SHAP Feature Attribution (Novel)
- Apply TreeSHAP to the adversarial classifier
- Rank features by mean |SHAP| as importance percentages
- **Automated prescription** based on distribution pattern:
  - **Univariate**: single feature > 50% --> investigate that data source
  - **Subset**: 2-5 features > 70% combined --> check shared pipeline
  - **Multivariate**: no feature > 30% --> full model retraining needed

### Example Output

```
DRIFT DETECTED (AUC = 0.85, p < 0.01)
Drift Diagnosis (SHAP):
  1. Feature Age: 45.2% contribution
  2. Feature Income: 24.8% contribution
Prescription: "Correlated feature-subset drift. Investigate common data source."
```

Compare to D3: `DRIFT DETECTED (AUC = 0.82)` -- no explanation.

---

## Project Structure

```
unsupervised-concept-drift-detection/
|-- detectors/
|   |-- eadd.py                  # EADD detector (OUR CONTRIBUTION)
|   |-- d3.py                    # D3 baseline (Gozuacik et al., 2019)
|   |-- base.py                  # Abstract base class
|   +-- ...                      # Other benchmark detectors
|-- experiment1_temporal_patterns.py   # Exp 1: 4 drift types
|-- experiment2_realworld_benchmark.py # Exp 2: 13 real-world datasets
|-- experiment3_explainability.py      # Exp 3: SHAP attribution
|-- experiment4_false_alarms.py        # Exp 4: False alarm robustness
|-- run_all_experiments.py             # Master runner + hypothesis tests
|-- EADD_Thesis_Experiments.ipynb      # Interactive notebook
|-- datasets/                   # Dataset loaders + CSV files
|-- metrics/                    # Detection performance metrics
|-- optimization/               # Experiment infrastructure
|-- eval/                       # Result analysis and plotting
|-- experiments/                # Output: results/ and figures/
|-- results/                    # Lukats benchmark raw results
+-- requirements.txt
```

---

## Installation

```bash
git clone https://github.com/NusratBegum/unsupervised-concept-drift-detection.git
cd unsupervised-concept-drift-detection
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Verify:

```bash
python -c "from detectors.eadd import ExplainableAdversarialDriftDetector; print('OK')"
python -m unittest discover -s test -t .   # 105 tests should pass
```

---

## Running the Experiments

| Command | Description | Time |
|---------|-------------|------|
| `python _run_tests.py` | Quick smoke test (Exp 1, 3, 4) | ~25 min |
| `python run_all_experiments.py --all` | Full suite + analysis | ~60 min |
| `python experiment1_temporal_patterns.py` | Temporal patterns only | ~15 min |
| `python experiment3_explainability.py` | SHAP explainability only | ~3 min |
| `python experiment4_false_alarms.py` | False alarm test only | ~8 min |
| `python run_all_experiments.py --analyze` | Hypothesis tests + plots (from existing CSVs) | ~10 sec |

### Output Files

| File | Description |
|------|-------------|
| `experiments/results/experiment1_temporal_patterns.csv` | Delay, success rate, false alarms per drift type |
| `experiments/results/experiment3_explainability.csv` | SHAP attribution accuracy per scenario |
| `experiments/results/experiment4_false_alarms.csv` | False alarm counts per stream type |
| `experiments/results/hypothesis_tests.csv` | Statistical test results |
| `experiments/figures/summary_comparison.png` | 4-panel comparison figure |
| `experiments/figures/latex_tables.tex` | Publication-ready LaTeX tables |

---

## Experiment Details

### Experiment 1: Sensitivity to Temporal Drift Patterns

**Setup**: Synthetic streams of 10,000 samples x 5 features, drift at t=5,000. 5 runs per type. Config: n_ref=500, n_cur=200, n_perm=50, alpha=0.01, freq=50.

**Key Finding**: EADD detects all 4 types (100% success), D3 misses gradual and incremental entirely. Both have zero false alarms.

### Experiment 3: Explainability Case Study

**Setup**: 10,000 samples x 10 features with controlled drift at t=5,000. Three scenarios: univariate (F3 only), subset (F2, F5, F7), multivariate (all features).

**Key Finding**: SHAP correctly identifies drift sources in all 3 scenarios. Prescription system correctly categorizes each drift pattern.

### Experiment 4: False Alarm Robustness

**Setup**: 10,000 samples x 5 features, 4 stable stream types (Gaussian, autocorrelated, heteroscedastic, correlated), 5 runs each.

**Key Finding**: EADD: zero false alarms across all streams. D3(t=0.6): 87.4 false alarms on autocorrelated data. The permutation test correctly identifies temporal correlations as stationary noise.

---

## Hypothesis Testing

| Hypothesis | Test | Statistic | p-value | Decision |
|---|---|---|---|---|
| H1a: EADD detects more types | Binomial | 2 / 4 | 0.69 | Directional support |
| H1b: EADD delay <= D3 | Mann-Whitney U | -- | 0.67 | No significant diff |
| H2: EADD FA < D3(t=0.7) | Mann-Whitney U | 6.0 | 0.23 | Not significant |
| H2: EADD FA < D3(t=0.6) | Mann-Whitney U | 0.0 | **0.0101** | **Supported** |
| H3: SHAP attribution | Qualitative | 3/3 | -- | **All correct** |

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
| INSECTS Incremental Balanced | 57,018 | 33 | Yes |
| INSECTS Incr-Abrupt | 79,986 | 33 | Yes |
| INSECTS Incr-Reoccurring | 79,986 | 33 | Yes |
| Electricity | 45,312 | 8 | No |
| NOAA Weather | 18,159 | 8 | No |
| Outdoor Objects | 4,000 | 21 | No |
| Ozone | 2,534 | 72 | No |
| Poker Hand | 829,201 | 10 | No |
| Powersupply | 29,928 | 2 | No |
| Rialto Bridge | varies | 27 | No |
| Luxembourg | varies | 31 | No |

Setup: Download from [USP DS Repository](https://sites.google.com/view/uspdsrepository), copy to `datasets/files/`, then `python add_headers.py`.

---

## Modifications from Original Repo

### New Files (Our Contributions)

| File | Purpose |
|------|---------|
| `detectors/eadd.py` | **EADD detector** -- core thesis contribution |
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
- **Lukats et al. (2025)**: [Benchmark of fully unsupervised concept drift detectors](https://link.springer.com/article/10.1007/s41060-024-00620-y)
- **D3**: Gozuacik et al. (2019), Discriminative Drift Detector
- **LightGBM**: Ke et al. (2017), [LightGBM](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
- **SHAP**: Lundberg and Lee (2017), [SHAP](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)
- **Original Repository**: [DFKI-NI/unsupervised-concept-drift-detection](https://github.com/DFKI-NI/unsupervised-concept-drift-detection)
- **USP DS Repository**: [sites.google.com/view/uspdsrepository](https://sites.google.com/view/uspdsrepository)

---

## License

BSD 3-Clause License -- See [LICENSE](LICENSE) file.
