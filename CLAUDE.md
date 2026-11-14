# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Experiment code for Nusrat Begum's M.Sc. thesis (Mahidol University) on EADD, an unsupervised feature-drift detector. It is a fork of the Lukats et al. (2024) benchmark (DFKI-NI/unsupervised-concept-drift-detection). The thesis text lives in the sibling repo `Nusrat-Thesis-Mahidol-University`; the published JCSSE paper in `JCSEE-Submission`.

## Commands

```bash
source .venv/bin/activate                 # Python 3.13; or: pip install -r requirements.txt
python -m pytest -q test/detectors/test_eadd.py test/detectors/test_d3_lightgbm.py test/detectors/test_bndm.py
python -m pytest -q test/detectors/test_eadd.py -k sliding_reference      # single test

python experiment_hypotheses.py           # all thesis runs; resumable; ~25 CPU-hours
python experiment_aspt_runtime.py         # H3 timing; ~3 h; run with nothing else on the machine
python analyze_hypotheses.py              # tables, tests, latex/*.tex rows (~1 min)
python make_hypothesis_figures.py         # Chapter 4 figures
python experiment_root_cause.py           # one-design-choice-at-a-time ablation (reuses runs.jsonl)
python experiment_offsets.py              # runs with a different stream start per seed
```

One run, for a quick check (D3 ~12 s, EADD ~2-3 min):

```bash
python -c "
from experiment_hypotheses import run_job
from analyze_hypotheses import detection_metrics
r = run_job(dict(part='annotated', dataset='InsectsAbruptBalanced', detector='D3-LightGBM', seed=42))
print(detection_metrics(r['detections'], r['true_drifts']))"
```

There is no linter or build step.

## Architecture

- `detectors/`: every detector subclasses `UnsupervisedDriftDetector` (`base.py`) and implements `update(features: dict) -> bool`, called once per sample. `eadd.py` is the thesis detector; `d3_lightgbm.py` is the baseline (D3 with EADD's LightGBM settings). The rest are upstream benchmark detectors.
- `datasets/`: one iterable class per stream yielding `(x, y)`; annotated streams expose `.drifts`. CSVs go in `datasets/files/` and are git-ignored (setup in README, "Datasets").
- Thesis pipeline: `experiment_hypotheses.py` defines `make_detector`, `run_job` and `all_jobs`, and appends one JSON line per run (detections only) to `experiments/results_hypotheses/runs.jsonl`. `analyze_hypotheses.py` recomputes every metric from that file, so metrics can change without rerunning detectors. Its `latex/*.tex` rows are copied into `tables/` of the thesis repo, and the figures into its `figures/`.
- Jobs are keyed by their JSON (`job_key`); a finished key is skipped on rerun. Adding a detector means adding it to `make_detector` and `all_jobs`.
- Metrics (`detection_metrics`): a detection matches a drift point if it falls 0-500 samples after it; a missed drift counts as MTD 500; unmatched detections are false alarms.
- `legacy/` holds the scripts that produced the JCSSE paper (`experiment1..7_*.py`, `experiment_camera_ready.py`, the notebook; outputs in `experiments/results/`, `results_cameraready/`, `figures/`), the October 2026 thesis drafts and two demos. Do not change their logic: they are the paper's provenance. Run them from the repo root with `PYTHONPATH=. python legacy/<script>.py`.
- `main.py`, `runner.py`, `config.py`, `optimization/`, `eval/`, `metrics/` are the upstream benchmark harness; the thesis pipeline uses only `metrics/` ideas, not this harness.

## Things to know before changing code

- Changing a detector's default behaviour invalidates stored runs. Add an option with the old behaviour as default (as `n_folds`, `sliding_reference`, `permutation_test` in `eadd.py`).
- EADD's permutation test uses its own RNG (`perm_rng`) so that test variants see the same reservoir samples.
- Known open issues in EADD (found 7-8 Nov 2026, not yet fixed): new samples enter the reservoir while still in the current window; the observed AUC uses 5-fold CV but permuted AUCs use 2-fold.
- Timing runs must be sequential; LightGBM is multithreaded and parallel jobs distort runtimes.
- On the advisor's server use at most 8 CPUs.
- New EADD files carry the SPDX header used in `detectors/eadd.py` (PolyForm Noncommercial); upstream files are BSD-3.
- Commit messages: plain imperative sentences describing what and why.
