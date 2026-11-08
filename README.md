# Enhanced Adversarial Drift Detection (EADD)

> **Thesis**: Master's thesis, Faculty of ICT, Mahidol University, 2026 (LaTeX source: [Nusrat-Thesis-Mahidol-University](https://github.com/NusratBegum/Nusrat-Thesis-Mahidol-University))
> **Author**: Nusrat Begum
> **Built on**: [DFKI-NI/unsupervised-concept-drift-detection](https://github.com/DFKI-NI/unsupervised-concept-drift-detection), the benchmark of [Lukats et al. (2024)](https://link.springer.com/article/10.1007/s41060-024-00620-y)

---

## Overview

EADD is an integrated framework for unsupervised feature drift detection based on adversarial validation (the classifier two-sample test, C2ST). It combines:

1. A **LightGBM classifier two-sample test** on a reference and a current window
2. A **reservoir-sampled reference window** that summarises all data since the last confirmed drift
3. **Prediction entropy monitoring** ($H_{pred}$) of the discriminator's outputs
4. An **adaptive sequential permutation test (ASPT)** that stops the permutation test early (`permutation_test="aspt"`; `"full"` draws all 199 permutations, `"none"` skips the test)
5. A **TreeSHAP ranking** of the features behind each confirmed drift (reported, not evaluated in the thesis)

The thesis compares EADD with D3 using the same LightGBM classifier and windows (D3-LightGBM) and with six detectors of the Lukats et al. benchmark, on real-world streams only.

---

## Key Results (thesis, Chapter 4)

Four annotated INSECTS streams (10 drift points), 5 seeds, 20 runs per detector; a detection matches if it is within 500 samples after a drift point; a missed drift point counts as MTD 500. One-sided Mann-Whitney U tests.

| | Recall | MTD | False alarms per run |
|---|---|---|---|
| D3-LightGBM | 0.65 | 279 | 5.6 |
| EADD | 0.50 | 302 | 8.9 |

- **H1 (higher recall, lower MTD than D3-LightGBM):** not supported (p = 0.90, p = 0.78). 16 of EADD's 21 missed drift points had an EADD detection in the 500 samples before the annotated point.
- **H2 (fewer false alarms):** not supported; D3-LightGBM raised significantly fewer (p < 0.001).
- **H3 (ASPT keeps the decisions of a full permutation test and is faster):** supported; identical detections in 20 of 20 runs, runtime 112 s vs. 149 s per run, measured one run at a time (25% less, faster in 20 of 20 runs; one-sided paired Wilcoxon p < 0.001, Mann-Whitney p = 0.022).
- **Threshold sensitivity:** at tau <= 0.5 ASPT rejects most alarms and cuts false alarms by 31-81% against no test.
- **Unannotated streams (Lukats et al. protocol, Hoeffding tree reset at every detection):** EADD 65.4% vs. D3-LightGBM 65.7% accuracy (p = 0.65).

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
|   |-- d3_lightgbm.py                   # D3 with a LightGBM classifier (used in the thesis)
|   |-- d3.py                            # D3 baseline (Gozuacik et al., 2019)
|   +-- ...                              # Other benchmark detectors
|-- experiment_hypotheses.py                # Thesis Exp 1-4: all detectors, real-world streams
|-- analyze_hypotheses.py                  # Thesis tables and significance tests
|-- experiment_aspt_runtime.py             # Thesis Exp 3: ASPT runtime, one run at a time
|-- make_hypothesis_figures.py             # Thesis Ch 4 figures
|-- experiment_aspt_type1.py                 # Thesis Ch 3: Type I error of ASPT stopping rules
|-- experiment_*.py, EADD_Thesis_Experiments.ipynb  # Earlier versions of the work and the JCSSE paper
|-- test/detectors/                          # Unit tests, incl. test_eadd.py, test_d3_lightgbm.py
|-- datasets/                                # Dataset loaders (CSV files not in git)
+-- experiments/                             # results_*/ CSVs and figures
```

The other `experiment*.py` scripts (synthetic streams, D3 with logistic regression, the 20,000-sample multi-detector run) belong to earlier versions of the work and the JCSSE paper; they are not used in the final thesis.

---

## Installation

```bash
git clone https://github.com/NusratBegum/unsupervised-concept-drift-detection.git
cd unsupervised-concept-drift-detection
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m pytest -q test/detectors/test_eadd.py test/detectors/test_d3_lightgbm.py test/detectors/test_bndm.py
```

---

## Running the Thesis Experiments

| Command | Thesis | Output |
|---|---|---|
| `python experiment_hypotheses.py` | Exp 1-4 (resumable; about 25 CPU-hours) | `experiments/results_hypotheses/runs.jsonl` |
| `python experiment_aspt_runtime.py` | Exp 3 runtime (about 3 hours, run alone on the machine) | `experiments/results_hypotheses/aspt_runtime.csv` |
| `python analyze_hypotheses.py` | Tables 4.1-4.7, tests | `experiments/results_hypotheses/*.csv`, `latex/*.tex` |
| `python make_hypothesis_figures.py` | Figures 4.1-4.3 | `experiments/results_hypotheses/figures/*.pdf` |
| `python experiment_aspt_type1.py` | Ch 3 (ASPT Type I error) | `experiments/results_aspt_type1/` |

Real-world experiments need the dataset CSVs in `datasets/files/` (see Datasets).

---

## Benchmark Detectors

| Detector | Method | Reference |
|----------|--------|-----------|
| **D3** | Logistic regression adversarial validation | Gozuacik et al., 2019 |
| **BNDM** | Bayesian non-parametric Polya tree test, computed for all features at once | Xuan et al. (2020) |
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

File names the loaders expect: `INSECTS <type>_<balanced|imbalanced>.csv` becomes `INSECTS-<type>_<balanced|imbalanced>_norm.csv`; `Electricity.csv` becomes `elec.csv`, `Outdoor.csv` becomes `outdoor.csv`, `Ozone.csv` becomes `ozone.csv`, `PokerHand.csv` becomes `poker-lsn.csv`, `Powersupply.csv` becomes `powersupply.csv`, `Rialto.csv` becomes `rialto.csv`, `Luxembourg.csv` becomes `luxembourg.csv`; `NOAA.csv` keeps its name. `Luxembourg.csv` ships with its own header: delete its first two lines (`sed -i '' '1,2d' luxembourg.csv`) before `add_headers.py`. The thesis results used these 1,900 rows, so the first data row is left out to reproduce them.

---

## Modifications from Original Repo

### New Files (Our Contributions)

| File | Purpose |
|------|---------|
| `detectors/eadd.py` | **EADD detector** -- core thesis contribution |
| `detectors/d3_lightgbm.py` | D3 with a LightGBM classifier (the D3 used in the thesis) |
| `experiment_hypotheses.py`, `analyze_hypotheses.py`, `experiment_aspt_type1.py` | Thesis experiments, tables and tests |
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
