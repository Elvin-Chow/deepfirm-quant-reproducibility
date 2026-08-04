# Frozen method contract

This document records the public configuration needed to interpret the source and archived results. Machine-readable values are in `configs/`.

## Warning roles and labels

The role-separated warning protocol assigns 55 percent to training, 15 percent to calibration, 20 percent to validation, and 10 percent to final test. Every boundary applies horizon-aware purge and embargo. H1 removes at least one decision row on each side of a boundary. H5 removes at least five.

For decision date `t` and horizon `h`, the target return uses `t+1` through `t+h`. The historical 5 percent threshold uses at most 252 prior horizon returns, requires at least 60, applies a one-period shift, uses linear quantile interpolation, and labels an event only when the future return is strictly below the threshold.

The feature contract contains 14 trailing features. It applies no feature-wise scaling or winsorization. Undefined skewness and kurtosis are filled with zero. Other remaining non-finite rows are excluded or rejected.

H1 passed the frozen validation gates. H5 failed the validation ROC-AUC floor of 0.58 and its final test remains unopened.

## Threshold

The primary threshold is 0.10074626865671642. It was selected on calibration only by minimizing `(FP + 20 * FN) / N`, subject to an aggregate flag-rate cap of 0.10. Scores equal to the threshold are flagged. For archived decimal CSV values, the public metric helper includes the immediately lower representable float so that a decimal round trip cannot exclude an equal score. Ties between candidate thresholds are resolved by lower objective, fewer false negatives, fewer flags, and then the higher threshold. The value 0.60 is a reporting-only historical reference.

## Warning comparison

The accepted H1 reference is compared with retrained XGBoost, logistic regression, random forest, gradient boosting, a historical tail rule, Historical VaR, Historical ES, Gaussian GARCH(1,1), SAV-CAViaR, linear quantile regression, and a causal five-step temporal MLP. Stochastic comparators use seeds 20260803 through 20260807. All fitted models use the same training roles, calibration rows, test rows, feature order, labels, and metric implementation.

Average precision is computed with `average_precision_score`. Top-decile lift uses exactly `ceil(0.10 * n)` rows, probability descending, and stable row identity ascending. Ties do not expand the bucket.

## Final panel

The final panel has two portfolios per market across the United States, Hong Kong, China, Japan, and Taiwan. All tickers are absent from the accepted-H1 training universe. Context begins 2025-01-02. Eligible H1 decisions run from 2026-05-18 through 2026-07-30, and outcomes run through 2026-07-31.

The CSI 300 fallback retains index code `000300`. The official series is eligible only when its code matches exactly, its outcome coverage reaches 2026-07-31, at least 250 return rows overlap the primary route, return correlation is at least 0.999, and maximum absolute return difference is at most 0.001.

## Allocation

The six allocation paths are equal weight, inverse volatility, mean variance, prior-only Black-Litterman, views-based Smart policy, and guarded Smart policy. Dynamic paths rebalance every 21 sessions.

The Black-Litterman path uses a Ledoit-Wolf covariance matrix, inverse-volatility equilibrium weights, risk aversion 2.5, and `tau = 1 / max(T, 60)`. A relative view uses the latest 63 complete returns, requires at least 40 rows, forms deterministic top and bottom legs, clips annualized expected return to [-0.20, 0.20], and maps coverage and score separation to confidence in [0.20, 0.80]. The optimizer is fully invested and long-only, with SLSQP tolerance `1e-9` and at most 1,000 iterations.

Execution uses a 40 percent effective position cap, 10 percent per-asset trade cap, and 25 percent one-way turnover cap. When uniform trade scaling cannot restore the position cap but the joint constraints are feasible, a closest-feasible projection uses a HiGHS feasibility and L1 solution followed by an SLSQP L2 projection. A genuinely infeasible trade fails closed.

Costs are reported at 5, 12, and 25 basis points one way. The 12 basis point primary case contains a 1 basis point fee, 10 basis point quoted spread, and 6 basis point slippage, with half of the quoted spread entering the one-way cost.

## Component analysis

The parent analysis changes one item at a time: model view, view uncertainty, Smart policy, or guard blending. Six additional rows neutralize the risk input block, regime input, anomaly input, weight bounds, turnover penalty, or concentration penalty. Every comparison holds the portfolio, benchmark, dates, cutoffs, prior, covariance, solver, constraints, model score, costs, and metric code fixed.

## Statistical inference

Warning intervals and tests preserve same-date cross-portfolio rows within 20-session circular moving blocks. Allocation variants use identical portfolio-date pairs and 21-session blocks. Both use 2,000 bootstrap replicates and 2,000 two-sided paired randomization replicates.

Primary warning and parent-component families use Holm correction at 0.05. Market, early and late, training-volatility, regime, and Smart subcomponent sensitivities use Benjamini-Hochberg control at 0.10. Slices below the fixed information minimum remain non-estimable and are not imputed.
