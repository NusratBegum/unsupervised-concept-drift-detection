# Architecture

How the detectors, streams, run function, analysis scripts and result files fit together. Every path is relative to the repository root.

## Data flow

```
stream (datasets/ or synthetic_streams.py)  --(x, y) per sample-->  detector.update(x) -> bool
        \                                                                  |
         job dict --> experiment_hypotheses.run_job(job) --> one JSON line per run (detections, drift points, counts)
                                                                           |
          experiment_*.py appends to experiments/results_<test>/runs*.jsonl (resumable, keyed by the job's JSON)
                                                                           |
          analyze_*.py recomputes every metric and test from the JSON lines --> *.csv (and latex/*.tex)
                                                                           |
          make_*_figures.py --> experiments/results_<test>/figures/*.pdf --> copied into the thesis repository
```

Detections are stored, metrics are not, so a change of a metric or test needs no detector rerun.

## Detectors (`detectors/`)

Every detector subclasses `UnsupervisedDriftDetector` (`detectors/base.py`) and implements `update(features: dict) -> bool`, called once per sample with the feature values only (labels never reach a detector).

- `eadd.py`: `ExplainableAdversarialDriftDetector`, the thesis detector (below).
- `d3_lightgbm.py`: D3 with EADD's LightGBM classifier; `d3.py`: D3 with logistic regression (Gozuacik et al., 2019). Both use a sliding reference (the samples just before the current window) and alarm when the AUC reaches the threshold, with no test. They drop the sample that triggers a check, so `step=49` gives a check every 50 samples.
- The other files are the upstream detectors of Lukats et al. (2024).
- The Alibi Detect online detectors are not in `detectors/`: `baselines_online.py` wraps them and runs in its own environment.

`experiment_hypotheses.make_detector(name, seed, tau, test, freq, **eadd_kwargs)` builds any detector by name with the thesis settings; `eadd_kwargs` override them.

## EADD step by step (`detectors/eadd.py`)

1. **Windows.** The first `n_reference_samples` (500) samples fill the reference. The current window holds the newest `n_current_samples` (200). The reference is then a reservoir sample (Algorithm R) of all samples since the last alarm. Each stored vector carries its arrival time as the last element, which `_detect_drift` strips before the classifier sees the data.
2. **C2ST.** Every `monitoring_frequency` (50) samples, a LightGBM classifier (50 trees, learning rate 0.1, 31 leaves, one thread) separates reference (label 0) from current (label 1); the statistic is the `n_folds`-fold (5) stratified cross-validated AUC. Below `auc_threshold` (0.7) there is no alarm.
3. **Permutation test.** Above the threshold, a p-value is computed from null AUCs: `permutation_test="aspt"` stops early, `"full"` uses every draw (up to `n_permutations`, 199), `"none"` accepts every threshold crossing. The test has its own random generator (`perm_rng`), so test variants see the same reservoir.
4. **Alarm.** If the test rejects, TreeSHAP ranks the features (`last_shap_importances`, % of mean absolute SHAP value, sorted), the current window becomes the new reference, and the history restarts; the next check comes once the current window has refilled.

`get_last_report()` returns the last AUC, p-value, permutations used and ranking. Run counters (`n_tests`, `n_early_stops`, `n_permutations_used`) are copied into each run's JSON line by `run_job`. The detector also records the prediction entropy of the classifier (`last_hpred`) and a drift severity index (`last_dsi`) from the JCSSE version; neither affects a decision, and the thesis does not evaluate them.

### Options and what each fix does

The constructor defaults reproduce the first thesis runs (`check_reproduced.py` checks this), so stored runs stay valid. The corrected detector is `experiment_validity.FIXED` plus the history null, i.e. `experiment_final.EADD_KW`:

| Option (final test) | Default | Problem fixed |
|---|---|---|
| `reservoir_after_current=True` | `False` | A sample could sit in the reservoir and the current window at once. Now a sample is offered to the reservoir only when it leaves the current window. |
| `perm_n_folds=None` | `2` | Null AUCs used 2-fold CV while the observed AUC used 5-fold, so the null did not match the statistic. `None` uses `n_folds`. |
| `aspt_alpha_split=1/3` | `None` | The original stopping rule had a Type I error of 0.059 at alpha 0.05 (`experiment_aspt_type1.py`). Now alpha/3 is spent on the early reject (no exceedance in the first i draws with 1/(i+1) <= alpha/3) and the rest on the final test (p <= 2 alpha/3), an exact level-alpha test (simulated 0.031). Early accept when more than 2 alpha of the draws reach the observed AUC. |
| `null="history"`, `history_stride=10` | `"permutation"`, `50` | Shuffling window labels assumes exchangeable samples; on autocorrelated streams the test rejected almost always (AR 0.9). The history null draws null AUCs from earlier 200-sample windows of the same stream (one start every `history_stride` samples, from the last `history_size` = 5,000), each against the reservoir without that window's samples. Requires `reservoir_after_current=True`. Right after an alarm few earlier windows exist, so the full test averages 163, not 199, draws. |

Options kept for the record and not used in the final test: `block_size` (blocked CV and block permutation; not a valid null, see the docstring), `min_null_draws` and `null_match_size` (development checks `dev_warm.py`, `dev_match.py`; they did not help), `sliding_reference` and `use_reservoir_sampling=False` (switch EADD's reference to D3's, used in the part-by-part runs).

## Streams

- `datasets/`: one iterable class per real stream yielding `(x, y)`; CSVs in `datasets/files/` (git-ignored). Annotated INSECTS streams expose `.drifts`.
- `synthetic_streams.synthetic_stream(name, seed)` understands:
  - `Synth<Stable|Abrupt|Gradual|Incremental|Recurring>`: 20,000 x 10 Gaussian features, features x0-x2 move by 1 SD at 5,000/10,000/15,000 (gradual and incremental changes last 1,000 samples);
  - `AR<rho>`: stationary AR(1), no drift;
  - `Shuffled<Dataset>`: real rows in random order (removes the stream's own drift);
  - `Inject<Abrupt|Gradual|Incremental>_<Dataset>`: a shuffled real stream with 30% of its features shifted by 1 SD at 5,000/10,000/15,000; `.shifted` lists the drifting features (ground truth for H4).
  The generator is seeded with (seed, kind), so each seed is a new stream and kinds do not share noise.

## Running jobs

- A job is a dict, e.g. `dict(part="annotated", dataset="SynthGradual", detector="EADD", seed=3000, tau=0.7, test="aspt", eadd_kwargs=EADD_KW, offset=17, record_shap=True)`. `run_job` builds the stream and detector, skips the first `offset` samples (a random start per seed moves the check grid relative to the drift points), feeds the stream, and returns the job plus `detections`, `true_drifts`, timings, test counters and, with `record_shap`, the feature ranking at each alarm. `part="unannotated"` also scores a Hoeffding tree reset at each alarm (Lukats et al., 2024).
- `experiment_validity.run(jobs, fn, out_file)` runs the jobs not yet in `out_file` on a process pool sized by `os.process_cpu_count()`, appends one JSON line per job, and starts a fresh pool if a worker dies (crashes go to `<out_file>.crashes.log`). The key of a job is its sorted JSON (`job_key`), so a rerun of the same command resumes. `experiment_final.py --shard i/n` splits the jobs over machines.
- `experiment_validity.p_job` runs one single test on a window pair without drift (the p-value checks of H2a and test validity).

## Analysis

- `analyze_hypotheses.detection_metrics` and `analyze_benchmark.tolerance` match detections to drift points: a detection matches if it falls within [d, d + T], T = 500 for abrupt and recurring drift and 1,500 for gradual and incremental drift; each drift point takes the first matching detection. Recall, MTD (a miss counts T), delay (matched only) and false alarms (unmatched detections, also per 10,000 samples) follow.
- The unit of analysis is a stream: runs are averaged over seeds first. Tests: one-sided Wilcoxon signed-rank over streams, Holm correction within the registered family (`analyze_benchmark.holm`), non-inferiority margin 50 for H1c, TOST for H3a, Friedman with a Nemenyi critical difference for several detectors (`analyze_benchmark.friedman`), Clopper-Pearson bounds for false-rejection rates.
- `analyze_confirm.py` and `analyze_final.py` implement the registered plans in `experiments/results_confirm/PREREGISTRATION.md` exactly; anything else is labelled exploratory or descriptive.

## Result files

| Folder | Produced by | Content |
|---|---|---|
| `experiments/results_hypotheses/` | `experiment_hypotheses.py`, `analyze_hypotheses.py` | First thesis runs (INSECTS, seeds 42-442), tables, `latex/*.tex` |
| `experiments/results_aspt_type1/` | `experiment_aspt_type1.py` | Type I error of the ASPT stopping rules |
| `experiments/results_comparison/` | `experiment_comparison.py`, `analyze_comparison.py` | 10 Nov comparison and `REPORT.md` |
| `experiments/results_validity/`, `results_benchmark/`, `results_shap/` | validity, benchmark and SHAP scripts | 16-17 Nov checks and benchmark |
| `experiments/results_confirm/` | `experiment_confirm.py`, `experiment_h2.py`, `experiment_robust.py`, `analyze_confirm.py` | `PREREGISTRATION.md`, confirmatory runs and tests |
| `experiments/results_final/` | `experiment_final.py`, `baselines_online.py`, `analyze_final.py`, `make_final_figures.py`, `make_concept_figures.py` | Final test: `runs.*.jsonl`, `pvalues.*.jsonl`, `online.jsonl`, `results.csv`, per-stream and per-type tables, figures |
| `experiments/results/`, `results_cameraready/`, `figures/` | `legacy/` scripts | JCSSE paper |
