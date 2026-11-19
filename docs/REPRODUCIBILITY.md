# Reproducibility

## Environment

- Python 3.13 with the exact package versions in `requirements-lock.txt` (LightGBM, SHAP, scikit-learn, river, NumPy, SciPy, pandas). Install with uv:

  ```bash
  uv venv --python 3.13 && source .venv/bin/activate
  uv pip install -r requirements-lock.txt
  ```

  On a machine without internet access, build a wheel folder elsewhere and install with `uv pip install --offline --no-index --find-links <wheels> -r requirements-lock.txt`.
- `baselines_online.py` (Alibi Detect 0.13, PyTorch CPU) needs its own environment (Python 3.12, numpy 1.26), because Alibi Detect does not install beside the pinned packages. It imports only `synthetic_streams.py` from this repository.
- Real streams need the CSVs in `datasets/files/` (README, "Datasets"). Synthetic and AR(1) streams need no files; shuffled and injected streams need the CSV of their base dataset.
- The runs of the thesis were made on the advisor's server limited to 10 cores (`taskset -c 0-9`) and on a MacBook. The scripts size their worker pool from `os.process_cpu_count()`, so `taskset` limits them. `OMP_NUM_THREADS=1` keeps LightGBM and NumPy from oversubscribing the cores.

## Seeds and determinism

- Each run is fixed by its seed: the stream generator is seeded with `(seed, kind)` (or `(seed, 13)` for the injected shift), EADD's reservoir with `seed`, its permutation test with `seed + 1`, LightGBM with `seed`, and the random start (`offset`, 0-49) of the final test with `(seed, 21)`. A rerun with the same library versions gives the same detections, whatever the number of workers.
- Each registered test used fresh seeds: first thesis runs 42-442; confirmatory test 1000-1009 (p-value reps 1000-1999); new H2 2000-2009; final test 3000-3009 (p-value reps 3000-3999); online detectors 3000-3002. Development checks used reps 0-499.
- `check_reproduced.py` reruns 16 of the first thesis EADD runs with the current code and default options and compares the detections with the stored ones (expected: 16 of 16).

## Registered plans

`experiments/results_confirm/PREREGISTRATION.md` holds the hypotheses, streams, seeds, tests and decision rules of every confirmatory test (the main plan and addenda 1-6b). Each plan was committed before its runs; `git log --follow experiments/results_confirm/PREREGISTRATION.md` and the commit dates of the result files show the order. The analysis scripts implement the plans exactly; any other number is labelled exploratory or descriptive.

## Rerunning each test

All experiment scripts skip jobs already in their output file, so an interrupted run resumes with the same command. To rerun from scratch, move the result folder aside first (as below for the final test). CPU-hours are approximate.

| Test | Commands | Size, time | Output |
|---|---|---|---|
| Final test of H1-H4 (addendum 5, thesis main result) | `OMP_NUM_THREADS=1 python experiment_final.py --shard 0/1`; `python analyze_final.py`; `python make_final_figures.py` | 1,340 stream runs + 10,000 single tests, about 15 CPU-hours | `experiments/results_final/` |
| H3 wall-clock timing (EADD with `experiment_validity.FIXED`, INSECTS) | `python experiment_aspt_runtime.py --fixed` (alone on the machine, sequential) | 20 run pairs, several hours | `experiments/results_hypotheses/aspt_runtime_fixed.csv` |
| Newer online detectors (addenda 6-6b) | `<py312 env>/bin/python baselines_online.py 10`; `python analyze_final.py` | 315 runs, about 20 min each with recalibration | `experiments/results_final/online.jsonl`, `online_*.csv` |
| Confirmatory test (seeds 1000-1009) | `python experiment_confirm.py`; `python analyze_confirm.py` | 870 runs + 8,000 single tests | `experiments/results_confirm/results.csv`, `h2a.csv`, `h3_h4.csv` |
| New H2 (addendum 3) | `python experiment_h2.py --part P F`; `python analyze_confirm.py` | 10,000 single tests + 180 runs | `experiments/results_confirm/h2_new*.csv` |
| Robustness R1/R2 (addendum 4) | `python experiment_robust.py`; `python analyze_confirm.py` | | `experiments/results_confirm/robust*.csv` |
| Validity checks and benchmark (16-17 Nov) | `python experiment_validity.py --part P F`; `python experiment_benchmark.py`; `python analyze_validity.py`; `python analyze_benchmark.py` | 1,584 + 396 benchmark runs | `experiments/results_validity/`, `results_benchmark/` |
| First thesis runs (INSECTS) | `python experiment_hypotheses.py`; `python analyze_hypotheses.py`; `python make_hypothesis_figures.py` | about 25 CPU-hours | `experiments/results_hypotheses/` |
| ASPT Type I error | `python experiment_aspt_type1.py` | minutes | `experiments/results_aspt_type1/` |

Prefix the long runs with `taskset -c 0-9` on a shared server. Timing runs (`experiment_aspt_runtime.py`) must run with nothing else on the machine: LightGBM is multithreaded and parallel jobs distort wall-clock times.

## Verifying `results.csv`

```bash
mv experiments/results_final experiments/results_final_published
OMP_NUM_THREADS=1 python experiment_final.py --shard 0/1
python analyze_final.py
diff experiments/results_final/results.csv experiments/results_final_published/results.csv   # expected: no output
```

Without rerunning the detectors, `python analyze_final.py` recomputes every table from the published `runs.*.jsonl`, `pvalues.*.jsonl` and `online.jsonl`; `git diff experiments/results_final/` should then be empty. `results.csv` has one row per registered test: the hypothesis, the comparison detector, the number of streams, both means, the p-value, the Holm-corrected p-value, its role (primary, secondary, descriptive) and whether it is supported. The other tables are `per_stream.csv`, `per_type.csv` (means per drift group and detector), `h2a.csv` (false-rejection rates with Clopper-Pearson bounds), `h4_per_stream.csv` and `online_*.csv`.
