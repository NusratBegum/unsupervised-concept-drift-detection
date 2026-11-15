# Enhanced Adversarial Drift Detection (EADD)

> **Thesis**: Master's thesis, Faculty of ICT, Mahidol University, 2026 (LaTeX source: [Nusrat-Thesis-Mahidol-University](https://github.com/NusratBegum/Nusrat-Thesis-Mahidol-University))
> **Paper**: [JCSSE 2026, IEEE Xplore](https://ieeexplore.ieee.org/document/11596765) (source: [JCSEE-Submission](https://github.com/NusratBegum/JCSEE-Submission))
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
- **Unannotated streams (Lukats et al. protocol, Hoeffding tree reset at every detection):** EADD 65.4% vs. D3-LightGBM 65.7% accuracy (two-sided paired Wilcoxon over 9 streams, p = 0.65).

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
|-- experiment_root_cause.py, experiment_offsets.py  # Further analyses (described in Ch 4; results only in this repository)
|-- experiment_comparison.py, synthetic_streams.py  # EADD vs D3 vs D3-LightGBM (10 Nov 2026), INSECTS and synthetic streams; not in the thesis
|-- analyze_comparison.py, make_comparison_figures.py  # Tables, tests and figures for that comparison
|-- legacy/                                  # JCSSE paper scripts, October drafts, demos (see legacy/README.md)
|-- main.py, runner.py, config.py, eval.py, eval/, optimization/, metrics/, results/  # Upstream benchmark harness and its results
|-- test/detectors/                          # Unit tests, incl. test_eadd.py, test_d3_lightgbm.py
|-- datasets/                                # Dataset loaders (CSV files not in git)
+-- experiments/                             # results_*/ CSVs and figures
```

The scripts in `legacy/` (synthetic streams, D3 with logistic regression, the 20,000-sample multi-detector run) belong to the JCSSE paper and to earlier versions of the work; they are not used in the final thesis. Run them from the repository root, e.g. `PYTHONPATH=. python legacy/demo_eadd.py`.

---

## Installation

```bash
git clone https://github.com/NusratBegum/unsupervised-concept-drift-detection.git
cd unsupervised-concept-drift-detection
uv venv --python 3.13 && source .venv/bin/activate
uv pip install -r requirements-lock.txt     # exact versions used for the thesis results
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
| `python experiment_root_cause.py` | Ablation: one design choice at a time, D3-like EADD to EADD (described in Ch 4, results here only) | `experiments/results_hypotheses/root_cause_table.csv` |
| `python experiment_offsets.py` | Runs with a different stream start per seed (described in Ch 4, results here only) | `experiments/results_hypotheses/offsets_table.csv` |

Real-world experiments need the dataset CSVs in `datasets/files/` (see Datasets).

### EADD vs D3 comparison (10 Nov 2026)

A comparison requested at the 10 Nov 2026 meeting: D3 with its authors' settings, D3-LightGBM and EADD, then all three in the same setup, then EADD's components one at a time, on the four annotated INSECTS streams and five synthetic streams (`synthetic_streams.py`). It is not part of the thesis. Results and discussion: [experiments/results_comparison/REPORT.md](experiments/results_comparison/REPORT.md).

```bash
taskset -c 0-9 python experiment_comparison.py   # runs, resumable; writes experiments/results_comparison/runs.jsonl
python analyze_comparison.py                      # summary.csv, per_stream.csv, tests.csv, reproduced.txt
python make_comparison_figures.py                 # experiments/results_comparison/figures/
```

---

## Benchmark Detectors

| Detector | Method | Reference |
|----------|--------|-----------|
| **EADD** | LightGBM C2ST, reservoir reference window, ASPT | this thesis |
| **D3-LightGBM** | D3 with a LightGBM discriminator and EADD's LightGBM settings (thesis baseline; `detectors/d3_lightgbm.py`) | this repository |
| **D3** | Logistic regression adversarial validation | Gozuacik et al., 2019 |
| **BNDM** | Bayesian non-parametric Polya tree test, computed for all features at once | Xuan et al., 2020 |
| **CSDDM** | PCA + K-Means + statistical tests | Wan and Wang, 2021 |
| **IBDD** | Image-based comparison | Souza et al., 2021 |
| **NN-DVI** | Nearest neighbor density | Liu et al., 2018 |
| **OCDD** | One-class SVM | Gozuacik and Can, 2020 |
| **SPLL** | Semi-parametric log-likelihood | Kuncheva, 2013 |
| **UCDD** | Clustering + pseudo-labels | Shang et al., 2020 |
| **UDetect** | Shewhart control charts | Bashir et al., 2017 |
| **EDFS** | Ensemble feature subspaces | Korycki and Krawczyk, 2019 |

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

Setup: download from the [USP DS Repository](https://sites.google.com/view/uspdsrepository), copy to `datasets/files/`, rename the files as listed below, then run `python add_headers.py` (it looks for the renamed files and skips any it cannot find).

File names the loaders expect: `INSECTS <type>_<balanced|imbalanced>.csv` becomes `INSECTS-<type>_<balanced|imbalanced>_norm.csv`; `Electricity.csv` becomes `elec.csv`, `Outdoor.csv` becomes `outdoor.csv`, `Ozone.csv` becomes `ozone.csv`, `PokerHand.csv` becomes `poker-lsn.csv`, `Powersupply.csv` becomes `powersupply.csv`, `Rialto.csv` becomes `rialto.csv`, `Luxembourg.csv` becomes `luxembourg.csv`; `NOAA.csv` keeps its name. `Luxembourg.csv` ships with its own header: delete its first two lines (`sed -i '' '1,2d' luxembourg.csv`) before `add_headers.py`. The thesis results used these 1,900 rows, so the first data row is left out to reproduce them.

Airlines, Chess, IntrusionDetection, Keystroke and SensorStream are not used in the thesis; their dataset tests in `test/datasets/` fail unless `airlines.csv`, `chess.csv`, `kddcup99.csv`, `keystroke.csv` and `sensorstream.csv` are supplied.

---

## Modifications from Original Repo

### New Files (Our Contributions)

Compared with the last upstream commit (`bc66d2c`): `git diff --name-status bc66d2c HEAD`.

| File | Purpose |
|------|---------|
| `detectors/eadd.py` | **EADD detector** -- core thesis contribution |
| `detectors/d3_lightgbm.py` | D3 with a LightGBM classifier (the D3 used in the thesis) |
| `experiment_hypotheses.py`, `analyze_hypotheses.py`, `make_hypothesis_figures.py` | Thesis Exp 1-4: runs, tables, tests and figures |
| `experiment_aspt_runtime.py`, `experiment_aspt_type1.py` | Thesis ASPT runtime (Exp 3) and Type I error (Ch 3) |
| `experiment_root_cause.py`, `experiment_offsets.py` | Further analyses (Ch 4) |
| `experiment_comparison.py`, `synthetic_streams.py`, `analyze_comparison.py`, `make_comparison_figures.py` | EADD vs D3 comparison of 10 Nov 2026 (not in the thesis) |
| `test/detectors/test_eadd.py`, `test/detectors/test_d3_lightgbm.py`, `test/test_analysis_formatting.py` | Unit tests for EADD/ASPT, D3-LightGBM and table number formatting |
| `legacy/` (all files) | JCSSE paper scripts, October 2026 drafts and demos (see `legacy/README.md`) |
| `experiments/` (all files) | Results, tables and figures |
| `add_headers.py` | CSV header script for USP datasets |
| `requirements-lock.txt` | Exact package versions used for the thesis results |
| `LICENSE-EADD`, `NOTICE` | Licence of the EADD files and the licence breakdown |
| `CLAUDE.md` | Notes for working on this code with Claude Code |

### Modified Files

| File | Change |
|------|--------|
| `detectors/__init__.py` | Added EADD import |
| `detectors/bndm.py` | Polya tree test computed for all features at once (same decisions, checked in `test/detectors/test_bndm.py`) |
| `detectors/d3.py` | Options `step` (check interval) and `n_folds`; the defaults keep the upstream behaviour |
| `requirements.txt` | Added lightgbm, shap, seaborn; removed ydata-profiling; `==` to `>=` |
| `datasets/airlines.py` | CSV instead of ARFF |
| `datasets/chess.py` | CSV, 8 features (at1-at8), `n_samples` 533 to 503 |
| `datasets/electricity.py` | CSV instead of ARFF |
| `datasets/intrusion_detection.py` | CSV instead of ARFF |
| `datasets/keystroke.py` | CSV instead of ARFF |
| `datasets/luxembourg.py` | `n_samples` 1,901 to 1,900 (see Datasets) |
| `datasets/waveform_drift2.py` | `h_functions` built from a NumPy array, because river now stores `_H_FUNCTION` as a list |
| `test/datasets/test_chess.py` | Expects 8 features; sample count taken from the loader |
| `test/detectors/test_bndm.py` | Test that the vectorised BNDM test matches the per-feature test |
| `test/detectors/test_edfs.py` | Feature values are floats instead of Python sets, which current SciPy rejects |
| `README.md`, `.gitignore` | This description; ignore caches, macOS files and the raw dataset folder |

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
| EADD contributions (© 2026 Nusrat Begum): `detectors/eadd.py`, all `experiment*.py`, `analyze_*.py`, `make_*_figures.py`, `synthetic_streams.py`, and the files in `legacy/` that carry the PolyForm SPDX header, plus `legacy/EADD_Thesis_Experiments.ipynb` | [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) | [LICENSE-EADD](LICENSE-EADD) |

**PolyForm Noncommercial 1.0.0** allows personal, academic, and non-profit research use of the EADD-specific files. **Commercial use** (including internal commercial operations) requires written permission from the author: `nusrat.beg@student.mahidol.ac.th`.

Every EADD-licensed `.py` file carries an SPDX header:
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
