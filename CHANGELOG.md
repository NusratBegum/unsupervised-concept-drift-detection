# Changelog

Main milestones, from the git history. Dates are commit dates; each registered plan was committed before its runs.

## 2026-11-19 to 2026-11-20: Addendum 7 and restyled thesis charts
- Addendum 7, three secondary checks registered before their runs (`experiment_addendum7.py`, `analyze_addendum7.py`, `experiments/results_addendum7/`): 7a TreeSHAP precision@k beside KS and standardised-mean-difference rankings (EADD reruns reproduce all 210 final-test runs); 7b the source of the H2 effect on the autocorrelated streams (the history null); 7c ASPT against the full test in wall-clock time with the history null (322 vs 569 s per run, faster in 8 of 8, p = 0.004).
- `make_final_figures.py` and `make_concept_figures.py` restyled to the MU-IEEE-Paper print style (Arial 8-9 pt, 1 pt strokes, colour plus marker or hatch per series); numbers unchanged.

## 2026-11-19: Newer detectors and thesis figures
- Comparison with MMD-Online and LSDD-Online (Alibi Detect, Cobb et al., AISTATS 2022) on the final-test streams, seeds 3000-3002 (addenda 6-6b; `baselines_online.py`); CVM-Online rerun with 100,000 bootstrap draws started (addendum 6b; its results are not yet in the repository).
- Vector figures of the final test (`make_final_figures.py`) and concept figures for Chapter 2 (`make_concept_figures.py`); fix of the drift-pattern grouping in the figures.
- README with the key results of the final test; quick-start example, architecture and reproducibility notes, citation metadata and CI.

## 2026-11-17 to 2026-11-18: Registered tests
- Confirmatory test of the revised H1/H2 on new seeds 1000-1009, then H3 and H4 on the same data (`experiment_confirm.py`, `analyze_confirm.py`).
- New H2 (addendum 3, seeds 2000-2009, `experiment_h2.py`) and robustness checks R1/R2 after an independent review (addendum 4, `experiment_robust.py`).
- Final test of H1-H4 (addendum 5, seeds 3000-3009, every detector checks every 50 samples from a random start per seed; `experiment_final.py`, `analyze_final.py`): H1, H2, H3 and H4 supported.
- Development checks of the first test after an alarm (`dev_match.py`, `dev_warm.py`; not used).

## 2026-11-16 to 2026-11-17: Validity fixes and benchmark
- Four fixes to EADD's test, each an option of `ExplainableAdversarialDriftDetector` whose default keeps the earlier behaviour: `reservoir_after_current`, `perm_n_folds=None`, `aspt_alpha_split=1/3`, and the history null (`null="history"`).
- Validity checks of the p-values and false alarms without drift (`experiment_validity.py`) and a benchmark of all detectors on INSECTS, synthetic, injected-drift and unannotated real streams (`experiment_benchmark.py`, `analyze_benchmark.py`), with part-by-part runs inside EADD's code and a TreeSHAP localisation check (`experiment_shap.py`).
- `check_reproduced.py`: the first thesis runs still reproduce with the current code.

## 2026-11-14 to 2026-11-15: Setup alignment after the 10 Nov meeting
- Packages managed with uv, versions pinned in `requirements-lock.txt`; worker pools respect the CPU limit.
- Paper-era scripts, October drafts and demos moved into `legacy/`.
- Comparison of EADD, D3 with its authors' settings and D3-LightGBM in an identical setup, with EADD's components one at a time, on INSECTS and five synthetic streams (`experiment_comparison.py`, `synthetic_streams.py`, report in `experiments/results_comparison/REPORT.md`); review fixes and the reruns they required.

## 2026-10-23 to 2026-11-08: Thesis experiments by hypothesis
- D3-LightGBM baseline and unit tests; thesis runs by hypothesis on INSECTS (`experiment_hypotheses.py`, `analyze_hypotheses.py`); faster BNDM; simulation of ASPT's Type I error (`experiment_aspt_type1.py`); sequential ASPT timing; root-cause ablation and start-offset runs.

## 2026-02 to 2026-05: JCSSE 2026 version
- EADD added to the Lukats et al. (2024) benchmark (fork of DFKI-NI/unsupervised-concept-drift-detection, last upstream commit `bc66d2c`); dataset loaders for the USP DS Repository CSVs.
- Experiments, camera-ready results and SPDX licence headers for the paper "Enhanced Adversarial Drift Detection for MLOps Feature Monitoring" (JCSSE 2026, IEEE). The scripts are kept in `legacy/`.
