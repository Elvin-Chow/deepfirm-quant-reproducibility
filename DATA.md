# Data availability and license boundary

This repository separates redistributable code and analytical outputs from provider-sourced market-price data.

## Included

The public package contains:

- frozen model files, calibration maps, feature schemas, and metadata;
- derived warning labels, probabilities, metrics, and inference results;
- derived allocation log returns, decision records, cost ledgers, metrics, and inference results;
- input snapshot identities, coverage dates, providers, row counts, and SHA-256 values;
- fixed portfolio, threshold, allocation, cost, component, and statistical configurations;
- generated tables and figures.

These files are provided for research inspection and reproduction of the archived calculations. They do not grant rights to reconstruct, redistribute, or commercially use provider datasets.

## Excluded

The repository does not contain raw or adjusted provider price tables, HTTP caches, SQLite caches, API responses, credentials, or tokens. Local `cache/` and `data/` directories are ignored. The canonical price and benchmark files used for the final evaluation are represented only by identity and coverage metadata in `results/frozen/input_snapshot_identities.csv`.

## Providers

The research pipeline accessed Yahoo Finance and AKShare. For the CSI 300 benchmark, the completed evaluation used the official China Securities Index series for the unchanged index code `000300` after the primary route ended before the required outcome date. The official series passed the fixed identity, overlap, and date-completeness checks recorded in the released input identity table.

Provider availability, corrections, corporate actions, symbol mappings, and trading calendars can change. Users must comply with each provider's terms and local law when retrieving data. This repository does not sublicense provider content.

## Published inspection versus live refresh

The offline smoke checks archived files and never contacts a provider. A live refresh is a new data retrieval, not an exact replay. It may produce different rows and must record the provider chain, retrieval time, requested and observed dates, missing-data handling, and canonical input identity.

The package intentionally omits a command that reacquires the final panel or repeats the frozen final evaluation. Repeating that experiment would violate the frozen evidence boundary described in the manuscript. The included provider source supports method inspection and future research, but later outputs must be labeled as a new study.

## Derived-output limits

Row-level predictions and portfolio returns can expose dates, public ticker-group labels, and model outputs. They do not contain provider price levels. Their inclusion supports the paper's table, figure, schema, and expected-value checks. No output is investment advice, and no benchmark-superiority test was pre-specified for the allocation results.
