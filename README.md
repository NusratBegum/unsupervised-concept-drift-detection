# EADD: feature drift detection via adversarial validation

[![tests](https://github.com/NusratBegum/unsupervised-concept-drift-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/NusratBegum/unsupervised-concept-drift-detection/actions/workflows/tests.yml)

Code, raw runs and registered analysis plans for the M.Sc. thesis *Enhanced Adversarial Drift Detection for MLOps Feature Monitoring* (Nusrat Begum, Faculty of ICT, Mahidol University, 2026; LaTeX source: [Nusrat-Thesis-Mahidol-University](https://github.com/NusratBegum/Nusrat-Thesis-Mahidol-University)). An earlier version was published at [JCSSE 2026 (IEEE Xplore)](https://ieeexplore.ieee.org/document/11596765) (source: [JCSEE-Submission](https://github.com/NusratBegum/JCSEE-Submission)). The repository is a fork of [DFKI-NI/unsupervised-concept-drift-detection](https://github.com/DFKI-NI/unsupervised-concept-drift-detection), the benchmark of [Lukats et al. (2024)](https://link.springer.com/article/10.1007/s41060-024-00620-y).

## Summary

EADD (`ExplainableAdversarialDriftDetector`, `detectors/eadd.py`) detects feature drift (a change in P(X)) in a data stream without labels. Every 50 samples it runs a classifier two-sample test (C2ST): a LightGBM classifier tries to tell a reference window (500 samples) from the current window (the newest 200), scored by 5-fold cross-validated AUC. An alarm needs AUC >= 0.7 and a rejection by a permutation test run as an adaptive sequential permutation test (ASPT), which stops drawing null AUCs as soon as the decision is clear. The reference is a reservoir sample of all data since the last alarm, and the null AUCs come from earlier windows of the same stream (the "history null"), so the test keeps the stream's autocorrelation. After each alarm TreeSHAP ranks the features behind the drift.

EADD is compared with D3 (Gozuacik et al., 2019) and with D3-LightGBM (D3 with EADD's classifier, which isolates EADD's reference and test) in an identical setup, with six detectors of the Lukats et al. benchmark, and with the online detectors MMD-Online and LSDD-Online of Alibi Detect (Cobb et al., AISTATS 2022). Every confirmatory test was registered in [`experiments/results_confirm/PREREGISTRATION.md`](experiments/results_confirm/PREREGISTRATION.md) and committed before its runs.

### Key results (final registered test, thesis Chapter 4)

Seeds 3000-3009; 21 drift streams (7 abrupt, 7 gradual, 7 incremental: synthetic plus real streams with injected drift), 8 no-drift streams and 6 autocorrelated no-drift streams; EADD, D3-LightGBM and D3 with identical windows (500/200), threshold 0.7, 5-fold CV, a check every 50 samples and the same random start per seed. One-sided Wilcoxon over streams, Holm-corrected. Plan registered before the runs (addendum 5).

| Hypothesis | EADD | D3-LightGBM | p (Holm) | Result |
|---|---|---|---|---|
| H1a slow-drift recall | 1.00 | 0.53 | 0.015 | supported |
| H1b slow-drift MTD (samples) | 443 | 879 | 0.023 | supported |
| H1c abrupt MTD, at most 50 higher | 115 | 122 | 0.023 | supported |
| H2 false alarms per 10k, autocorrelated streams without drift | 5.6 | 19.5 | 0.023 | supported |
| H3 ASPT vs full test: recall / MTD / false alarms equivalent; permutations per test | 52 | 163 (full test) | < 0.01 | supported |
| H4 TreeSHAP precision@k vs chance | 0.75 | 0.31 (chance) | < 0.001 | supported |

Against the newer online detectors (seeds 3000-3002), EADD has the best mean recall rank and about five times fewer false alarms on slow-drift streams (1.5 vs 6.9-7.7 per stream); LSDD-Online detects faster. All numbers: [`experiments/results_final/results.csv`](experiments/results_final/results.csv), `per_type.csv`, `per_stream.csv`, `h2a.csv`, `h4_per_stream.csv`, `online_*.csv`.

### Registered tests and how to reproduce them

Each test of addenda 1-6 uses new seeds; addendum 7 reuses earlier seeds for three secondary checks. The git history shows that each plan precedes its runs.

| Test | Seeds | Script, then analysis | Results | Verdict |
|---|---|---|---|---|
| Confirmatory test of the revised H1/H2, plus H3, H4 | 1000-1009 | `experiment_confirm.py`, `analyze_confirm.py` | `experiments/results_confirm/` | H1, H4 supported; revised H2 not; H3 timing supported |
| New H2 (addendum 3) | 2000-2009 | `experiment_h2.py`, `analyze_confirm.py` | `h2_new*.csv` | fewer false alarms supported; test validity 8 of 10 |
| Robustness R1/R2 (addendum 4) | 1000-1009 | `experiment_robust.py`, `analyze_confirm.py` | `robust*.csv` | H1 holds with identical check schedules |
| **Final test of H1-H4 (addendum 5)** | **3000-3009** | `experiment_final.py`, `analyze_final.py` | `experiments/results_final/results.csv` | **H1, H2, H3, H4 supported** |
| Newer online detectors (addenda 6-6b) | 3000-3002 | `baselines_online.py`, `analyze_final.py` | `experiments/results_final/online_*.csv` | best recall rank, fewer false alarms; not significant with 3 seeds (the CVM-Online rerun with 100,000 draws is not yet in the repository) |
| Secondary checks (addendum 7): 7a TreeSHAP vs univariate rankings, 7b source of the H2 effect, 7c ASPT wall-clock time with the history null | 7a 3000-3009, 7b 3000-3004, 7c 42 and 142 | `experiment_addendum7.py` (7c: `--timing`), `analyze_addendum7.py` | `experiments/results_addendum7/` | 7a: TreeSHAP precision@k 0.75 below KS 0.97 and SMD 0.89 (H4 above chance unchanged); 7b: the history null, not the reservoir, gives the H2 effect (4.15 vs 19.5 false alarms per 10k, p = 0.016); 7c: ASPT 322 vs 569 s per run, faster in 8 of 8, p = 0.004 |

To rerun the final test from scratch (the scripts skip finished runs, so an interrupted run resumes with the same command):

```bash
uv venv --python 3.13 && source .venv/bin/activate && uv pip install -r requirements-lock.txt
# datasets: see "Datasets" below; synthetic and AR streams need no files
mv experiments/results_final experiments/results_final_published                          # keep the published results
OMP_NUM_THREADS=1 python experiment_final.py --shard 0/1    # 1,340 stream runs + 10,000 single tests, about 15 CPU-hours
python analyze_final.py                                      # writes experiments/results_final/results.csv
diff experiments/results_final/results.csv experiments/results_final_published/results.csv
```

Runs are deterministic for a given seed and library versions (`requirements-lock.txt`), so a rerun gives the same detections. On a machine with more cores the worker pool grows with `os.process_cpu_count()`; limit it with `taskset` if needed. Every other test, its runtime and its outputs: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Quick start

```bash
git clone https://github.com/NusratBegum/unsupervised-concept-drift-detection.git
cd unsupervised-concept-drift-detection
uv venv --python 3.13 && source .venv/bin/activate
uv pip install -r requirements-lock.txt     # exact versions used for the thesis results
python examples/quickstart.py               # EADD on a synthetic gradual-drift stream, about 30 s
```

[`examples/quickstart.py`](examples/quickstart.py) needs no dataset files. It prints each alarm with its AUC, p-value and top TreeSHAP features (`python examples/quickstart.py SynthAbrupt 3` for another stream and seed).

## Using the detector

Every detector takes one sample at a time as a `dict` of feature values and returns `True` at an alarm. The settings below are those of the final test (`experiment_final.EADD_KW`, which is `experiment_validity.FIXED` plus the history null). The constructor defaults keep the behaviour of the first thesis runs, so the corrected options must be passed explicitly.

```python
from detectors.eadd import ExplainableAdversarialDriftDetector

detector = ExplainableAdversarialDriftDetector(
    n_reference_samples=500, n_current_samples=200, auc_threshold=0.7,
    monitoring_frequency=50, n_permutations=199, significance_level=0.05,
    permutation_test="aspt", seed=0,
    reservoir_after_current=True,   # no sample in both windows
    perm_n_folds=None,              # null AUCs use the same 5-fold CV as the observed AUC
    aspt_alpha_split=1 / 3,         # exact level-alpha ASPT stopping rule
    null="history", history_stride=10)  # null AUCs from earlier windows of the same stream

for i, (x, _) in enumerate(stream):     # x: {"feature name": value, ...}; labels are not used
    if detector.update(x):
        print(i, detector.last_auc, detector.last_p_value, detector.last_shap_importances)
```

`last_shap_importances` maps each feature to its share (in %) of the mean absolute TreeSHAP value, sorted from the largest. `get_last_report()` returns the last test's AUC, p-value, permutations used and feature ranking. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes every option and how the pieces fit together.

## Repository map

| Purpose | Files |
|---|---|
| Detectors | `detectors/eadd.py` (EADD), `detectors/d3_lightgbm.py` (D3-LightGBM), `detectors/d3.py` (D3), the other upstream benchmark detectors in `detectors/` |
| Streams | `synthetic_streams.py` (synthetic, AR(1), shuffled real and injected-drift streams; run it for a self-check), `datasets/` (real-stream loaders) |
| First thesis runs (INSECTS, seeds 42-442) | `experiment_hypotheses.py` (also defines `make_detector` and `run_job`, used by every later script), `analyze_hypotheses.py`, `experiment_aspt_runtime.py`, `experiment_aspt_type1.py`, `experiment_root_cause.py`, `experiment_offsets.py` |
| 10 Nov comparison (EADD vs D3, identical setup) | `experiment_comparison.py`, `analyze_comparison.py` (report: `experiments/results_comparison/REPORT.md`) |
| Validity checks and benchmark (16-17 Nov) | `experiment_validity.py`, `analyze_validity.py`, `experiment_benchmark.py`, `analyze_benchmark.py`, `experiment_shap.py` |
| Confirmatory and final tests (17-18 Nov) | `experiment_confirm.py`, `experiment_h2.py`, `experiment_robust.py`, `analyze_confirm.py`, `experiment_final.py`, `analyze_final.py`; development checks (not registered tests): `dev_match.py`, `dev_warm.py` |
| Secondary checks of addendum 7 (19-20 Nov) | `experiment_addendum7.py`, `analyze_addendum7.py` |
| Newer detectors (19 Nov) | `baselines_online.py` (Alibi Detect MMD-, LSDD- and CVM-Online; own environment) |
| Figures | `make_final_figures.py`, `make_concept_figures.py` (thesis Chapters 2 and 4, MU-IEEE-Paper print style), `make_hypothesis_figures.py`, `make_comparison_figures.py`, `trace_example.py` (illustration only) |
| Checks | `check_reproduced.py` (first thesis runs reproduce with the current code), `test/` (unit tests) |
| Examples and docs | `examples/quickstart.py`, `docs/ARCHITECTURE.md`, `docs/REPRODUCIBILITY.md` |
| Results | `experiments/results_*/` (raw runs as JSON lines, tables, figures) |
| Provenance | `legacy/` (JCSSE paper scripts and October drafts, see `legacy/README.md`); `main.py`, `runner.py`, `config.py`, `eval.py`, `eval/`, `optimization/`, `metrics/`, `results/` (upstream benchmark harness) |
| Dataset setup | `add_headers.py`, `convert_datasets.py` |

## Detectors

| Detector | Method | Reference |
|----------|--------|-----------|
| **EADD** | LightGBM C2ST, reservoir reference, history null, ASPT, TreeSHAP | this thesis |
| **D3-LightGBM** | D3 with EADD's LightGBM classifier | this repository |
| **D3** | C2ST with logistic regression and a sliding reference | Gozuacik et al., 2019 |
| **MMD-Online**, **LSDD-Online** | Sequential kernel and least-squares density-difference tests, calibrated to an expected run time | Cobb et al., 2022 (Alibi Detect 0.13) |
| **BNDM** | Bayesian non-parametric Polya tree test | Xuan et al., 2020 |
| **CSDDM** | PCA + K-Means + statistical tests | Wan and Wang, 2021 |
| **IBDD** | Image-based comparison | Souza et al., 2021 |
| **NN-DVI** | Nearest neighbor density | Liu et al., 2018 |
| **OCDD** | One-class SVM | Gozuacik and Can, 2020 |
| **SPLL** | Semi-parametric log-likelihood | Kuncheva, 2013 |
| **UCDD** | Clustering + pseudo-labels | Shang et al., 2020 |
| **UDetect** | Shewhart control charts | Bashir et al., 2017 |
| **EDFS** | Ensemble feature subspaces | Korycki and Krawczyk, 2019 |

## Datasets

Synthetic, AR(1) and injected-drift streams are generated by `synthetic_streams.py` (injected and shuffled streams need the real CSV they are built from). Real streams come from the [USP Data Stream Repository](https://sites.google.com/view/uspdsrepository):

| Dataset | Samples | Features | Drift points known |
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

Setup: download from the USP DS Repository, copy to `datasets/files/` (git-ignored), rename the files as listed below, then run `python add_headers.py` (it looks for the renamed files and skips any it cannot find).

File names the loaders expect: `INSECTS <type>_<balanced|imbalanced>.csv` becomes `INSECTS-<type>_<balanced|imbalanced>_norm.csv`; `Electricity.csv` becomes `elec.csv`, `Outdoor.csv` becomes `outdoor.csv`, `Ozone.csv` becomes `ozone.csv`, `PokerHand.csv` becomes `poker-lsn.csv`, `Powersupply.csv` becomes `powersupply.csv`, `Rialto.csv` becomes `rialto.csv`, `Luxembourg.csv` becomes `luxembourg.csv`; `NOAA.csv` keeps its name. `Luxembourg.csv` ships with its own header: delete its first two lines (`sed -i '' '1,2d' luxembourg.csv`) before `add_headers.py`. The thesis results used these 1,900 rows, so the first data row is left out to reproduce them.

Airlines, Chess, IntrusionDetection, Keystroke and SensorStream are not used in the thesis; their dataset tests in `test/datasets/` fail unless `airlines.csv`, `chess.csv`, `kddcup99.csv`, `keystroke.csv` and `sensorstream.csv` are supplied.

## Testing

```bash
python -m pytest -q test --ignore=test/datasets    # about 30 s; needs no dataset files (run in CI)
python -m pytest -q test/datasets                  # loader checks; need the CSVs in datasets/files/
python synthetic_streams.py                        # self-check of the stream generators (needs NOAA.csv)
```

`test/detectors/test_eadd.py` covers EADD's reference, each validity option, ASPT and the history null; `test_d3_lightgbm.py` and `test_bndm.py` the modified baselines.

## Changes to the upstream code

Compared with the last upstream commit (`bc66d2c`; full list: `git diff --name-status bc66d2c HEAD`), all EADD files are new (see the repository map). Modified upstream files:

| File | Change |
|------|--------|
| `detectors/__init__.py` | Added EADD import |
| `detectors/bndm.py` | Polya tree test computed for all features at once (same decisions, checked in `test/detectors/test_bndm.py`) |
| `detectors/d3.py` | Options `step` (check interval) and `n_folds`; the defaults keep the upstream behaviour |
| `requirements.txt` | Added lightgbm, shap, seaborn; removed ydata-profiling; `==` to `>=` |
| `datasets/airlines.py`, `electricity.py`, `intrusion_detection.py`, `keystroke.py` | CSV instead of ARFF |
| `datasets/chess.py` | CSV, 8 features (at1-at8), `n_samples` 533 to 503 |
| `datasets/luxembourg.py` | `n_samples` 1,901 to 1,900 (see Datasets) |
| `datasets/waveform_drift2.py` | `h_functions` built from a NumPy array, because river now stores `_H_FUNCTION` as a list |
| `test/datasets/test_chess.py` | Expects 8 features; sample count taken from the loader |
| `test/detectors/test_bndm.py`, `test_edfs.py` | Test of the vectorised BNDM; EDFS feature values as floats (current SciPy rejects sets) |

## Citation

Please cite the thesis (metadata in [CITATION.cff](CITATION.cff); GitHub's "Cite this repository" button reads it) and the JCSSE paper:

```bibtex
@mastersthesis{begum2026thesis,
  author = {Begum, Nusrat},
  title  = {Enhanced Adversarial Drift Detection for {MLOps} Feature Monitoring},
  school = {Faculty of Information and Communication Technology, Mahidol University},
  year   = {2026},
  type   = {M.Sc. thesis},
}

@inproceedings{begum2026eadd,
  author    = {Begum, Nusrat and Yamchote, Phaphontee and
               Amornbunchornvej, Chainarong and Noraset, Thanapon},
  title     = {Enhanced Adversarial Drift Detection for MLOps
               Feature Monitoring},
  booktitle = {Proc. 23rd International Joint Conference on Computer
               Science and Software Engineering (JCSSE)},
  year      = {2026},
  pages     = {684--689},
  publisher = {IEEE},
  doi       = {10.1109/JCSSE68839.2026.11596765},
  url       = {https://ieeexplore.ieee.org/document/11596765},
}
```

Other references: Lukats et al. (2024), [benchmark of fully unsupervised concept drift detectors](https://link.springer.com/article/10.1007/s41060-024-00620-y); Ke et al. (2017), [LightGBM](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree); Lundberg and Lee (2017), [SHAP](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions); Cobb et al. (2022), sequential multivariate change detection (AISTATS). History of the repository: [CHANGELOG.md](CHANGELOG.md).

## License

This repository is dual-licensed; [NOTICE](NOTICE) has the file-by-file breakdown.

| Scope | License | File |
|---|---|---|
| Upstream code (Lukats et al. / DFKI base detectors, evaluation harness, datasets) | [BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause) | [LICENSE](LICENSE) |
| EADD contributions (© 2026 Nusrat Begum): `detectors/eadd.py`, the experiment, analysis and figure scripts, `synthetic_streams.py`, `examples/`, and every file that carries the PolyForm SPDX header | [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) | [LICENSE-EADD](LICENSE-EADD) |

PolyForm Noncommercial 1.0.0 allows personal, academic and non-profit research use of the EADD files. Commercial use (including internal commercial operations) requires written permission from the author: `nusrat.beg@student.mahidol.ac.th`. Every EADD-licensed `.py` file carries the header:

```
# SPDX-FileCopyrightText: 2026 Nusrat Begum <nusrat.beg@student.mahidol.ac.th>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
```
