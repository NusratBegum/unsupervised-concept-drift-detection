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

## Addendum: H3 and H4 on the same data (registered 17 Nov 2026, before these runs)

So that all four hypotheses are tested on the same streams, drift types and seeds, H3 and H4 are added to this experiment. EADD is the detector above; the streams and seeds are the 21 drift streams and 8 no-drift streams above, seeds 1000-1009.

- **H3a (same decisions).** EADD with ASPT and EADD with the full permutation test (199 draws, same null and alpha rule) give identical detections. Reported as the share of runs with identical detections over all 290 run pairs; supported if at least 95% are identical.
- **H3b (lower cost).** ASPT uses fewer permutations per test than the full test. One-sided Wilcoxon over the 29 streams of the mean number of permutations per test. Permutation counts do not depend on the machine, so these runs may share the server. Wall-clock time is measured separately one run at a time (`experiment_aspt_runtime.py --fixed`, INSECTS streams) and reported alongside.
- **H4 (localisation).** At the first detection of each drift point, EADD's TreeSHAP ranking is compared with the features that truly drift: precision@k (k = number of drifting features) and whether the top feature drifts. Supported if precision@k exceeds chance (k / d) by a one-sided Wilcoxon over the 21 drift streams (stream means). Only streams where the drifting features are known can be scored: the synthetic and injected streams (6 real bases, abrupt, gradual and incremental drift). INSECTS and the unannotated streams have no known drifting features and are not scored for H4.

H3a, H3b and H4 are each tested at alpha 0.05 (separate hypotheses, not in the H1/H2 family).

## Addendum 2: all detector groups on the same data (registered 17 Nov 2026, before these runs)

For a complete comparison on the same 29 streams and seeds 1000-1009, two groups are added: D3 with its authors' settings (logistic regression, windows 100/10, threshold 0.7, a check every 11 samples) and six fully unsupervised detectors from Lukats et al. (2024) with their benchmark settings (BNDM, CSDDM, IBDD, OCDD, SPLL, UDetect; 500-sample windows). They enter only secondary, descriptive comparisons: per-group means of recall, MTD and false alarms, and a Friedman test with Nemenyi critical difference over the 21 drift streams for MTD, recall and false alarms. The primary tests above are unchanged.

## Addendum 3: a new H2 (registered 17 Nov 2026, evening, before these runs)

H2 as registered above is not supported: the test is not valid at AR 0.9 (H2a), and H2b had too little power (only one no-drift stream had any false alarms). The new H2 below is formed after seeing those results, so it is tested only on data not used before: new p-value repetitions (2000-2999), two real bases not used in any p-value check (Electricity, PokerHand), AR 0.3 (not used before), and new autocorrelated streams and seeds (2000-2009). Detectors and settings are those above (`experiment_confirm.DETECTORS`). Code: `experiment_h2.py`.

- **H2-new a (valid test where its assumption holds).** On data without drift whose samples are independent or at most moderately autocorrelated (lag-1 autocorrelation up to 0.5), EADD's test rejects at alpha = 0.05 at a rate whose one-sided 95% Clopper-Pearson upper bound is at most 0.075, in each of: independent Gaussian, shuffled Electricity, shuffled PokerHand, AR 0.3, AR 0.5; in the steady state and right after a reset. 1,000 tests per condition (10 conditions). Supported if all 10 pass.
- **H2-new b (fewer false alarms under strong autocorrelation).** On stationary AR(1) streams without drift with rho 0.8, 0.85, 0.9, 0.93, 0.95, 0.97 (10 features, 20,000 samples), EADD raises fewer false alarms per 10,000 samples than D3-LightGBM. One-sided Wilcoxon signed-rank test over the 6 streams (mean of 10 seeds per stream); against D3 as secondary.

H2-new is supported if all 10 conditions of H2-new a pass and H2-new b (the only p-value of this addendum) is significant at alpha 0.05. The AR 0.9 failure of the original H2a is reported as a limitation in either case.

## Addendum 4: robustness checks after the independent review (registered 17 Nov 2026, evening, before these runs)

An independent code review found that the three detectors check on fixed schedules that are not aligned: EADD checks every 50 samples, D3 and D3-LightGBM every 51 (D3 skips the sample that triggers a check), and every seed starts at sample 0, so each detector meets each drift point at the same phase of its schedule in all 10 seeds. Part of each delay is then set by the schedule, not by the detector. A review of the data found that on PokerHand and Rialto some features that are constant in the first 5,000 rows are among the shifted features, so those drifts are trivially separable. The registered results above stand as registered; these checks are reported next to them as robustness checks.

- **R1 (schedule).** H1a, H1b, H1c repeated with D3 and D3-LightGBM at `step=49` (a check every 50 samples, as EADD) and each seed starting at a random offset in 0-49 (`numpy.random.default_rng([seed, 21]).integers(0, 50)`, the same offset for all three detectors), on the 21 drift streams and seeds 1000-1009. Same tests, same Holm family of three, same decision rule. Code: `experiment_robust.py`; output `robust_runs.jsonl`, `robust.csv`.
- **R2 (constant features).** H1a, H1b, H1c and H4 recomputed from the registered runs without the PokerHand and Rialto streams (15 drift streams left: 10 slow, 5 abrupt). Output `robust_no_constant.csv`.
