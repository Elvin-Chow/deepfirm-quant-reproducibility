# Reproduction guide

This guide covers the public, offline reproduction surface for version 1.0.0. It checks the released files and regenerates presentation artifacts from frozen outputs. It does not retrieve market data or repeat the frozen final evaluation.

## 1. Create the environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

The resolved lock was recorded with Python 3.13.9. The package metadata permits Python 3.11 or later, but a different Python or operating-system combination can resolve binary wheels differently.

## 2. Run the offline smoke

```bash
bash scripts/run_reproducibility_smoke.sh
```

The smoke performs these read-only checks:

1. verifies required public paths;
2. verifies every entry in `results/manifest.sha256`;
3. recomputes all four warning bundle hashes and their feature-schema identities;
4. checks the frozen CSV and compressed CSV schemas;
5. compares selected values with `results/expected_values.json`;
6. runs the targeted smoke tests.

The expected final line is:

```text
OFFLINE_SMOKE_STATUS=PASSED
```

The command sets a network-denial socket guard for its Python checks. It does not write a log into the repository.

## 3. Run the complete public test suite

```bash
python -m pytest -q
```

The suite checks artifact identities, probability semantics, deterministic exact-k ranking, H1/H5 purge and embargo, the fixed threshold and result values, allocation costs and constraints, the same-index benchmark gate, generator determinism, and the absence of provider data from the offline path.

## 4. Regenerate tables and figures

```bash
python scripts/generate_tables.py
python scripts/generate_figures.py
```

The scripts read only `results/frozen/` and write to `results/generated/` and `results/figures/`. They never fit or score a model. To check the tracked outputs without changing files, run:

```bash
python scripts/generate_tables.py --check
python scripts/generate_figures.py --check
```

Generated tables include:

- `results/generated/warning_model_summary.csv`;
- `results/generated/allocation_strategy_cost_summary.csv`;
- `results/generated/allocation_component_inference.csv`;
- `results/generated/coverage_summary.csv`.

The primary overview figure is written as PDF and 300 DPI PNG in `results/figures/`.

## 5. Inspect artifacts and results

Artifact roles are explicit:

| Path | Scientific status |
|---|---|
| `artifacts/crisis_warning/accepted/global_h1/` | Passed the frozen role-separated validation gates. |
| `artifacts/crisis_warning/failed/global_h5/` | Failed the ROC-AUC gate; final test unopened. |
| `artifacts/crisis_warning/legacy/global_h1/` | Degraded legacy diagnostic. |
| `artifacts/crisis_warning/legacy/global_h5/` | Degraded legacy diagnostic. |

The main frozen result files are:

| File | Contents |
|---|---|
| `warning_predictions.csv.gz` | 14,616 row-level H1 predictions across the accepted reference and 11 comparators. |
| `warning_metrics_overall.csv` | Overall warning metrics for each repeat. |
| `warning_inference.csv` | Dependence-aware warning intervals, tests, and correction fields. |
| `allocation_returns.csv.gz` | 27,030 gross and net allocation return rows. |
| `allocation_decisions.csv.gz` | Point-in-time decision inputs, cutoffs, weights, and guard states. |
| `cost_ledger.csv.gz` | Rebalance-level turnover and fee, spread, and slippage deductions. |
| `allocation_metrics.csv` | Strategy and component results at 5, 12, and 25 basis points. |
| `allocation_inference.csv` | Paired block intervals, tests, and non-estimable slices. |
| `protocol_deviations.csv` | Allowed provider, convergence, and insufficient-slice deviations. |
| `skips.csv` | Empty skip schema for the completed final evaluation. |
| `model_failures.csv` | Empty model-failure schema for the completed final evaluation. |

`results/frozen/pre_scoring_failures.csv` preserves two scientific failure boundaries without exposing execution chronology. One failure was caused by an over-strict decimal round-trip identity check. The other was caused by a CSI 300 route that ended before the required outcome date. Neither failure produced labels, predictions, returns, or performance metrics.

## 6. Interpret the checks correctly

A matching hash establishes file identity and mutation detection. A matching schema establishes structural compatibility. Matching expected values establishes agreement with selected materialized results. None of these checks proves that a model, threshold, allocation rule, or economic claim is scientifically valid.

The accepted H1 result still has low recall. H5 remains failed and unopened. Overall primary and component comparisons remain null after correction. Market-specific comparisons are often non-estimable. Historical negative benchmark-excess rows are excluded legacy records. Positive pooled benchmark-excess fields in the final panel are descriptive, and no strategy-benchmark test was pre-specified.

## 7. Live data and new experiments

This release has no command for live refresh, retraining, retuning, rescoring, or repetition of the frozen final evaluation. The provider code in `data_pipeline/` is included to make the method inspectable. If a researcher uses it for a later study, the new retrieval must use a separate output directory, comply with provider terms, and report its own dates, identities, failures, and protocol changes.
