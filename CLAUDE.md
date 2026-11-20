# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Experiment code for Nusrat Begum's M.Sc. thesis (Mahidol University) on EADD, an unsupervised feature-drift detector. It is a fork of the Lukats et al. (2024) benchmark (DFKI-NI/unsupervised-concept-drift-detection). The thesis text lives in the sibling repo `Nusrat-Thesis-Mahidol-University`; the published JCSSE paper in `JCSEE-Submission`.

## Commands

```bash
source .venv/bin/activate                 # Python 3.13, managed with uv: uv pip install -r requirements-lock.txt
python -m pytest -q test/detectors/test_eadd.py test/detectors/test_d3_lightgbm.py test/detectors/test_bndm.py
python -m pytest -q test/detectors/test_eadd.py -k sliding_reference      # single test
python -m pytest -q test --ignore=test/datasets   # every test that needs no dataset files (~30 s); what CI runs (.github/workflows/tests.yml)
python examples/quickstart.py             # EADD with the final-test settings on SynthGradual (~30 s, no data files)

python experiment_hypotheses.py           # all thesis runs; resumable; ~25 CPU-hours
python experiment_aspt_runtime.py         # H3 timing; ~3 h; run with nothing else on the machine
python analyze_hypotheses.py              # tables, tests, latex/*.tex rows (~1 min)
python make_hypothesis_figures.py         # Chapter 4 figures
python experiment_root_cause.py           # one-design-choice-at-a-time ablation (reuses runs.jsonl)
python experiment_offsets.py              # runs with a different stream start per seed

# 10 Nov 2026 comparison (authors' D3, identical setup, components, ASPT at tau 0.5/0.6); report: experiments/results_comparison/REPORT.md
taskset -c 0-9 python experiment_comparison.py   # 1,645 runs, resumable; --list, --only A B
python analyze_comparison.py              # summary.csv, per_stream.csv, tests.csv, aspt_tests.csv, reproduced.txt
python make_comparison_figures.py         # figures/fig0-fig5
python synthetic_streams.py               # self-check of the five synthetic streams

# 16 Nov 2026 validity checks and benchmark after the EADD fixes
taskset -c 0-9 python check_reproduced.py        # thesis EADD runs still reproduced with the current code
taskset -c 0-9 python experiment_validity.py     # P: p-values under no drift; F: false alarms on no-drift streams
taskset -c 0-9 python experiment_benchmark.py    # all detectors x INSECTS, synthetic, injected-drift real, unannotated
python analyze_validity.py; python analyze_benchmark.py   # per-stream stats, Friedman + Nemenyi CD, Holm-corrected Wilcoxon

# 18-20 Nov 2026 registered tests (plans: experiments/results_confirm/PREREGISTRATION.md); the thesis reports these
OMP_NUM_THREADS=1 python experiment_final.py --shard 0/1   # final test of H1-H4 (addendum 5), seeds 3000-3009, ~15 CPU-hours
python analyze_final.py                   # experiments/results_final/results.csv, per_type.csv, per_stream.csv, h2a.csv, h4_per_stream.csv, online_*.csv
python make_final_figures.py              # thesis Chapter 4 figures; then python make_concept_figures.py (Chapter 2); run them separately
python experiment_addendum7.py            # addendum 7a/7b; with --timing: 7c, sequential, nothing else running
python analyze_addendum7.py               # experiments/results_addendum7/
# baselines_online.py (addenda 6-6b) needs its own Python 3.12 environment: docs/REPRODUCIBILITY.md
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

Docs for readers: `docs/ARCHITECTURE.md` (detectors, streams, `run_job`, analysis, result files, every EADD option), `docs/REPRODUCIBILITY.md` (environment, seeds, how to rerun each registered test, runtimes, verifying `results.csv`), `CHANGELOG.md`, `CITATION.cff`. Keep them in step with the code when an option, script or result folder changes. `examples/` holds runnable examples (no dataset files needed).

## Architecture

- `detectors/`: every detector subclasses `UnsupervisedDriftDetector` (`base.py`) and implements `update(features: dict) -> bool`, called once per sample. `eadd.py` is the thesis detector; `d3_lightgbm.py` is the baseline (D3 with EADD's LightGBM settings). The rest are upstream benchmark detectors.
- `datasets/`: one iterable class per stream yielding `(x, y)`; annotated streams expose `.drifts`. CSVs go in `datasets/files/` and are git-ignored (setup in README, "Datasets").
- `synthetic_streams.py`: five synthetic streams (Stable, Abrupt, Gradual, Incremental, Recurring), 20,000 samples, drifts at 5,000/10,000/15,000; `run_job` reads any dataset named `Synth<Kind>`. The generator is seeded with (seed, kind), so kinds do not share noise.
- Thesis pipeline: `experiment_hypotheses.py` defines `make_detector`, `run_job` and `all_jobs`, and appends one JSON line per run (detections only) to `experiments/results_hypotheses/runs.jsonl`. `analyze_hypotheses.py` recomputes every metric from that file, so metrics can change without rerunning detectors. Its `latex/*.tex` rows are copied into `tables/` of the thesis repo, and the figures into its `figures/`.
- Jobs are keyed by their JSON (`job_key`); a finished key is skipped on rerun. Adding a detector means adding it to `make_detector` and `all_jobs`.
- Metrics (`detection_metrics`): a detection matches a drift point if it falls 0-500 samples after it; a missed drift counts as MTD 500; unmatched detections are false alarms. `analyze_comparison.py` uses 0-1,500 on SynthGradual and SynthIncremental (the change lasts 1,000 samples), so a miss there counts 1,500; its `delay` column excludes misses.
- D3 (`d3.py`, `d3_lightgbm.py`) drops the sample that triggers a check, so it checks every `step + 1` samples (201, not 200).
- `legacy/` holds the scripts that produced the JCSSE paper (`experiment1..7_*.py`, `experiment_camera_ready.py`, the notebook; outputs in `experiments/results/`, `results_cameraready/`, `figures/`), the October 2026 thesis drafts and two demos. Do not change their logic: they are the paper's provenance. Run them from the repo root with `PYTHONPATH=. python legacy/<script>.py`.
- `main.py`, `runner.py`, `config.py`, `optimization/`, `eval/`, `metrics/` are the upstream benchmark harness; the thesis pipeline uses only `metrics/` ideas, not this harness.

## Things to know before changing code

- Changing a detector's default behaviour invalidates stored runs. Add an option with the old behaviour as default (as `n_folds`, `sliding_reference`, `permutation_test` in `eadd.py`).
- EADD's permutation test uses its own RNG (`perm_rng`) so that test variants see the same reservoir samples.
- EADD options that fix validity problems (defaults keep the thesis behaviour; `experiment_validity.FIXED` turns them on):
  - `reservoir_after_current=True`: no sample is in both windows.
  - `perm_n_folds=None`: permuted AUCs use the same CV as the observed AUC (the thesis used 2-fold vs 5-fold).
  - `aspt_alpha_split=1/3`: ASPT spends alpha/3 on the early reject and the rest on the final test, an exact level-alpha test (the thesis rule has Type I 0.059 at alpha 0.05; `experiment_aspt_type1.py`).
  - `null="history"`, `history_stride=10`: null AUCs from earlier windows of the same stream (needs `reservoir_after_current`); with `FIXED` this is `experiment_final.EADD_KW`, the detector of the final test.
  - `block_size=b`: blocked CV and block permutation against autocorrelation (not a valid null; not used).
- Each stored window vector carries its arrival time as the last element; `_detect_drift` strips it.
- Stream names understood by `run_job`: `Synth<Kind>`, `AR<rho>` (stationary AR(1)), `Shuffled<Dataset>` (real rows in random order), `Inject<Abrupt|Gradual|Incremental>_<Dataset>` (shuffled real stream with drifts injected at 5k/10k/15k).
- Timing runs must be sequential; LightGBM is multithreaded and parallel jobs distort runtimes.
- Use uv for every Python package (`uv pip install`, `uv venv`); never plain pip.
- The advisor's server has no internet access and allows at most 10 cores (run under `taskset -c 0-9`); its setup is described in the workspace notes outside this repository. The scripts size their worker pool from `os.process_cpu_count()`, so they follow that limit.
- New EADD files carry the SPDX header used in `detectors/eadd.py` (PolyForm Noncommercial); upstream files are BSD-3.
- Commit messages: plain imperative sentences describing what and why.
