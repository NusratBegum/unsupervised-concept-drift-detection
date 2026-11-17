# Confirmatory test of the revised hypotheses (registered 17 Nov 2026, before any run)

The revised H1 and H2 come from the benchmark of 16-17 Nov 2026 (`experiments/results_benchmark/`). To test them on data they were not derived from, this experiment uses new random seeds (1000-1009; the benchmark used 42-442), a drift type not injected before (gradual drift on real data), new p-value repetitions (1000-1999; the validity check used 0-299), and two real bases not in the no-drift checks before (PokerHand, Rialto). Nothing below is changed after the runs start; any later analysis is reported as exploratory.

## Detectors (identical setup)

Windows 500/200, AUC threshold 0.7, a check every 50 samples, 5-fold CV, LightGBM settings as in EADD.

- **EADD**: reservoir reference without overlap, ASPT with alpha split 1/3, permuted and observed AUC with the same CV, history null with stride 10 (`experiment_validity.FIXED` + `null="history", history_stride=10`).
- **D3-LightGBM**: sliding reference, AUC threshold only.
- **D3**: as D3-LightGBM with logistic regression.

## Hypotheses and tests

Unit of analysis: a stream (mean over its 10 seeds). Primary family H1a, H1b, H1c, H2b: Holm correction at alpha = 0.05.

- **H1a (slow drift, recall).** On streams with gradual or incremental drift, EADD has a higher recall than D3-LightGBM. One-sided Wilcoxon signed-rank test over the 14 slow-drift streams.
- **H1b (slow drift, delay).** On the same streams, EADD has a lower MTD than D3-LightGBM (a miss counts 1,500). One-sided Wilcoxon over the 14 streams.
- **H1c (abrupt drift, not worse).** On abrupt streams, EADD's MTD is at most 50 samples higher than D3-LightGBM's (non-inferiority margin 50, one check interval). One-sided Wilcoxon of (MTD_EADD - MTD_D3LightGBM - 50) < 0 over the 7 abrupt streams.
- **H2a (valid test).** On data without drift, EADD's test rejects at alpha = 0.05 at a rate whose one-sided 95% Clopper-Pearson upper bound is at most 0.075, in each of: independent Gaussian, shuffled real (INSECTS), AR 0.5, AR 0.9; in the steady state (5,000-sample history) and right after a reset. 1,000 tests per condition. Each condition is reported pass or fail.
- **H2b (fewer false alarms).** On streams without drift, EADD raises fewer false alarms per 10,000 samples than D3-LightGBM. One-sided Wilcoxon over the 8 no-drift streams (pairs with no difference are dropped, as Wilcoxon does).

Secondary (reported, not in the family): the same tests against D3; per-type means; H3 and H4 from their own experiments.

## Data

- Slow drift (14): SynthGradual, SynthIncremental; InjectGradual and InjectIncremental on Electricity, NOAAWeather, PokerHand, Powersupply, RialtoBridgeTimelapse, InsectsIncrementalBalanced.
- Abrupt drift (7): SynthAbrupt; InjectAbrupt on the same six bases.
- No drift (8): SynthStable, AR0.5, AR0.9, ShuffledElectricity, ShuffledNOAAWeather, ShuffledPowersupply, ShuffledPokerHand, ShuffledRialtoBridgeTimelapse.

Matching: a detection counts if it falls 0-500 samples after a drift point (0-1,500 for gradual and incremental drift); unmatched detections are false alarms.

## Decision

H1 is supported if H1a and H1b are significant after Holm and H1c holds. H2 is supported if every H2a condition passes and H2b is significant after Holm. Otherwise the hypothesis is reported as not supported, with the numbers.
