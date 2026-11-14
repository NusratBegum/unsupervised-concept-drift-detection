"""Full experiment runner: Experiments 1, 3, 4 + hypothesis testing."""
import warnings
warnings.filterwarnings('ignore')
import time

# ── Experiment 1 ──────────────────────────────────────────────
print('=' * 60)
print('EXPERIMENT 1: Temporal Drift Patterns')
print('=' * 60)
t0 = time.time()
from experiment1_temporal_patterns import run_experiment_1
exp1 = run_experiment_1()
print(f'  Total time: {time.time()-t0:.1f}s')

# ── Experiment 3 ──────────────────────────────────────────────
print('\n' + '=' * 60)
print('EXPERIMENT 3: Explainability')
print('=' * 60)
t0 = time.time()
from experiment3_explainability import run_experiment_3
exp3 = run_experiment_3()
print(f'  Total time: {time.time()-t0:.1f}s')

# ── Experiment 4 ──────────────────────────────────────────────
print('\n' + '=' * 60)
print('EXPERIMENT 4: False Alarms')
print('=' * 60)
t0 = time.time()
from experiment4_false_alarms import run_experiment_4
exp4 = run_experiment_4()
print(f'  Total time: {time.time()-t0:.1f}s')

# ── Hypothesis tests ─────────────────────────────────────────
print('\n' + '=' * 60)
print('HYPOTHESIS TESTING')
print('=' * 60)
try:
    from run_all_experiments import hypothesis_testing
    hyp = hypothesis_testing()
except Exception as e:
    print(f'  Hypothesis testing error: {e}')
    print('  (This is expected if experiment2 results are not available)')

print('\n' + '=' * 60)
print('ALL DONE!')
print('=' * 60)
