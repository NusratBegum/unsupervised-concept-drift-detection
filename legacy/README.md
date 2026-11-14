# legacy/

Scripts that are not part of the final thesis pipeline. Kept as the record of how earlier numbers were produced; do not change their logic.

| Files | What they produced |
|---|---|
| `experiment1..7_*.py`, `experiment_camera_ready.py`, `run_all_experiments.py`, `run_comprehensive_experiments.py`, `_run_tests.py`, `EADD_Thesis_Experiments.ipynb` | The JCSSE 2026 paper (`experiments/results/`, `results_cameraready/`, `figures/`) |
| `experiment_d3_lightgbm.py`, `experiment_realworld_ablation.py`, `experiment_aspt_threshold.py` | Thesis drafts of October 2026, replaced by `experiment_hypotheses.py` |
| `demo.py`, `demo_eadd.py` | Console demos of D3 and EADD |

Run from the repository root:

```bash
PYTHONPATH=. python legacy/demo_eadd.py
```
