# Result status and claim boundaries

## Accepted evidence

The role-separated H1 warning bundle passed the fixed validation gates. In the security-disjoint and out-of-time panel, accepted H1 records 522 rows, 50 events, ROC-AUC 0.5664, average precision 0.1241, Brier score 0.0876, recall 0.0800 at the calibration-selected threshold, and top-decile lift 1.3789.

The full framework completed 14,616 warning prediction rows, 18 strategy-cost rows, 33 component-cost rows, 27,030 allocation-return rows, 363 warning-inference rows, and 220 allocation-inference rows. There were no model failures or evaluation skips.

## Failed evidence

The role-separated H5 bundle records validation ROC-AUC 0.5598, below the fixed floor of 0.58. Its final test remains unopened. The bundle is released because failed evidence is part of the audit contract.

Two input acquisitions stopped before scoring. One exposed an over-strict decimal round-trip identity check. One exposed incomplete CSI 300 outcome coverage. Neither produced labels, predictions, returns, or metrics. Their scientific failure categories are retained in `results/frozen/pre_scoring_failures.csv`; execution chronology is not part of the public package.

## Legacy evidence

The legacy H1 and H5 bundles are degraded. H1 fails the raw calibration-error gate. H5 fails the raw calibration-error and ROC-AUC gates. Their calibration and evaluation roles were not separated, so calibrated losses are not untouched validation evidence.

Six historical negative benchmark-excess rows inherit a superseded whole-test-window guard. They are retained in `results/frozen/legacy_allocation_impact.csv` with `current_evidence_admissible=false`. They do not establish the direction of the current allocation result.

## Null and non-estimable results

No primary warning comparison survives Holm correction. No parent component or Smart subcomponent comparison survives its specified correction. All market-specific allocation comparisons are non-estimable under the fixed information minimum. Two warning regime sensitivities survive Benjamini-Hochberg correction with mixed directions; they do not support broad generalization.

## Benchmark boundary

The final allocation files contain positive pooled benchmark-excess fields, but those values concatenate 10 portfolio paths. They are descriptive fields, not pre-specified endpoints, portfolio-annualized estimates, alpha, or formal strategy-benchmark tests. No strategy-benchmark superiority test was pre-specified.

## Provider and reproduction boundary

The completed result includes 395 allowed, non-fatal deviations: 384 insufficient-information slices, five temporal-MLP convergence warnings, four provider fallbacks, and two same-index CSI 300 completeness fallbacks. Provider data can change on a later retrieval.

All clean-environment inspection and reproduction was executed in the authoring workflow. No evaluator outside the author team independently reproduced the system. Matching hashes and materialized values does not remove this limitation.
