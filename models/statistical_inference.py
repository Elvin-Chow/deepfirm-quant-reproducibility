"""Dependence-aware paired inference primitives for the dependence-aware inference protocol.

The functions operate on row-level paired observations.  Warning inference
resamples unique-date blocks while keeping every same-date portfolio row
together.  Allocation inference resamples time blocks within each portfolio.
Neither path accepts portfolio-level metric summaries as bootstrap units.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


@dataclass(frozen=True)
class PairedInferenceResult:
    metric: str
    estimate: float
    ci_low: float
    ci_high: float
    p_value: float
    confidence_level: float
    bootstrap_replicates: int
    randomization_replicates: int
    block_length: int
    unique_dates: int
    portfolio_count: int
    block_count: int
    repeat_count: int
    status: str = "estimable"
    failure_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_statistics_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    """Validate protocol invariants that prevent iid or result-driven inference."""

    repeated = protocol["repeated_runs"]
    seeds = [int(seed) for seed in repeated["seeds"]]
    if len(seeds) != int(repeated["required_repeats"]) or len(set(seeds)) != len(seeds):
        raise ValueError("dependence-aware inference repeated-run seeds must be unique and match required_repeats")
    if int(repeated["minimum_paired_successes_for_inference"]) > len(seeds):
        raise ValueError("minimum paired successes cannot exceed the frozen seed count")

    warning = protocol["warning_inference"]
    if int(warning["primary_block_length_sessions"]) < int(
        warning["h5_minimum_block_length_sessions"]
    ):
        raise ValueError("warning block length must cover the H5 overlap horizon")
    if "iid" not in str(warning["invalid_method"]).lower():
        raise ValueError("the dependence-aware inference warning protocol must explicitly prohibit iid bootstrap")

    allocation = protocol["allocation_inference"]
    if int(allocation["primary_block_length_sessions"]) <= 1:
        raise ValueError("allocation inference requires multi-session time blocks")
    invalid_allocation = str(allocation["invalid_method"]).lower()
    if "nineteen" not in invalid_allocation or "prohibited" not in invalid_allocation:
        raise ValueError("the legacy 19-summary-row bootstrap must be prohibited")

    paired = protocol["paired_tests"]
    if paired["confidence_interval"] != "circular_moving_block_percentile_interval":
        raise ValueError("unsupported dependence-aware inference confidence-interval contract")
    if "paired_block_randomization" not in paired["p_value"]:
        raise ValueError("unsupported dependence-aware inference paired-test contract")

    corrections = protocol["multiple_comparisons"]
    if corrections["primary_method"] != "holm":
        raise ValueError("Holm must remain the primary family-wise correction")
    if corrections["exploratory_method"] != "benjamini_hochberg":
        raise ValueError("Benjamini-Hochberg must remain the exploratory FDR correction")

    scope = protocol["scope_boundaries"]
    if scope["development_outputs_are_performance_or_significance_evidence"] is not False:
        raise ValueError("development outputs cannot be dependence-aware inference performance evidence")
    if scope["synthetic_outputs_are_performance_or_significance_evidence"] is not False:
        raise ValueError("synthetic outputs cannot be dependence-aware inference performance evidence")
    if scope["locked_full_framework_test"] != "prohibited_in_development":
        raise ValueError("dependence-aware inference cannot authorize the locked final evaluation locked test")
    if scope["h5_final_test"] != "unopened":
        raise ValueError("dependence-aware inference cannot open the H5 final test")
    return protocol


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Return Holm family-wise adjusted p-values in original order."""

    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Holm correction requires a nonempty finite one-dimensional vector")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted_ranked = np.maximum.accumulate((values.size - np.arange(values.size)) * ranked)
    adjusted_ranked = np.minimum(adjusted_ranked, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def benjamini_hochberg_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Return Benjamini-Hochberg FDR adjusted p-values in original order."""

    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("BH correction requires a nonempty finite one-dimensional vector")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    factors = values.size / np.arange(1, values.size + 1)
    adjusted_ranked = np.minimum.accumulate((ranked * factors)[::-1])[::-1]
    adjusted_ranked = np.minimum(adjusted_ranked, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def chronological_period_labels(dates: Iterable[Any]) -> pd.Series:
    """Assign outcome-independent early/late labels over ordered unique dates."""

    converted = pd.to_datetime(pd.Series(list(dates), dtype="object"))
    if converted.isna().any():
        raise ValueError("period labels require finite dates")
    unique_dates = np.asarray(sorted(converted.unique()))
    if len(unique_dates) < 2:
        raise ValueError("period sensitivity requires at least two unique dates")
    split = len(unique_dates) // 2
    early = set(unique_dates[:split])
    labels = ["early" if date in early else "late" for date in converted.to_numpy()]
    return pd.Series(labels, index=converted.index, dtype="object")


def lagged_annualized_volatility(
    benchmark_log_returns: pd.Series,
    *,
    lookback_sessions: int = 63,
) -> pd.Series:
    """Compute point-in-time volatility using only returns available through t-1."""

    values = pd.Series(benchmark_log_returns, copy=True, dtype=float)
    if not values.index.is_monotonic_increasing or values.index.has_duplicates:
        raise ValueError("benchmark returns must have a unique increasing index")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("benchmark returns must be finite")
    if int(lookback_sessions) < 2:
        raise ValueError("volatility lookback must be at least two sessions")
    return values.shift(1).rolling(
        int(lookback_sessions), min_periods=int(lookback_sessions)
    ).std(ddof=1) * np.sqrt(252.0)


def training_volatility_tertiles(
    training_benchmark_log_returns: pd.Series,
    *,
    lookback_sessions: int = 63,
) -> tuple[float, float]:
    """Fit calm/stress cutoffs using training returns only."""

    volatility = lagged_annualized_volatility(
        training_benchmark_log_returns, lookback_sessions=lookback_sessions
    ).dropna()
    if len(volatility) < 10:
        raise ValueError("training-only regime cutoffs require at least ten finite volatility rows")
    low, high = np.quantile(volatility.to_numpy(dtype=float), [1.0 / 3.0, 2.0 / 3.0])
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError("invalid training-only volatility tertiles")
    return float(low), float(high)


def assign_volatility_regimes(
    benchmark_log_returns_with_history: pd.Series,
    *,
    low_cutoff: float,
    high_cutoff: float,
    lookback_sessions: int = 63,
) -> pd.Series:
    """Apply frozen training cutoffs to lagged point-in-time volatility."""

    if not np.isfinite([low_cutoff, high_cutoff]).all() or low_cutoff > high_cutoff:
        raise ValueError("regime cutoffs must be finite and ordered")
    volatility = lagged_annualized_volatility(
        benchmark_log_returns_with_history, lookback_sessions=lookback_sessions
    )
    labels = pd.Series("not_estimable", index=volatility.index, dtype="object")
    finite = volatility.notna()
    labels.loc[finite & (volatility <= float(low_cutoff))] = "calm"
    labels.loc[
        finite
        & (volatility > float(low_cutoff))
        & (volatility < float(high_cutoff))
    ] = "normal"
    labels.loc[finite & (volatility >= float(high_cutoff))] = "stress"
    return labels


def _circular_positions(length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if length < 2:
        raise ValueError("at least two ordered observations are required")
    if block_length < 2 or block_length > length:
        raise ValueError("block_length must be between 2 and the ordered-series length")
    positions: list[int] = []
    while len(positions) < length:
        start = int(rng.integers(0, length))
        positions.extend((start + offset) % length for offset in range(block_length))
    return np.asarray(positions[:length], dtype=int)


def _nonoverlapping_blocks(length: int, block_length: int) -> list[np.ndarray]:
    return [
        np.arange(start, min(start + block_length, length), dtype=int)
        for start in range(0, length, block_length)
    ]


def _warning_metric(
    metric: str,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> float:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(scores, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("warning labels and scores must be finite aligned vectors")
    if ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("warning scores must lie in [0, 1]")
    if metric == "roc_auc":
        if np.unique(y).size != 2:
            raise ValueError("ROC-AUC is not estimable for a single-class sample")
        return float(roc_auc_score(y, p))
    if metric == "average_precision":
        if np.unique(y).size != 2:
            raise ValueError("average precision is not estimable for a single-class sample")
        return float(average_precision_score(y, p))
    if metric == "brier_score":
        return float(brier_score_loss(y, p))
    if metric == "log_loss":
        return float(log_loss(y, np.clip(p, 1e-12, 1.0 - 1e-12), labels=[0, 1]))
    flagged = p >= float(threshold)
    if metric == "flag_rate_at_frozen_threshold":
        return float(flagged.mean())
    if metric == "recall_at_frozen_threshold":
        positives = int(y.sum())
        if positives == 0:
            raise ValueError("recall is not estimable without positive events")
        return float(np.logical_and(flagged, y == 1).sum() / positives)
    raise ValueError(f"unsupported warning metric: {metric}")


def paired_warning_block_inference(
    frame: pd.DataFrame,
    *,
    metric: str,
    date_col: str,
    portfolio_col: str,
    label_col: str,
    reference_score_col: str,
    challenger_score_col: str,
    threshold: float,
    block_length: int,
    bootstrap_replicates: int,
    randomization_replicates: int,
    confidence_level: float,
    seed: int,
    seed_col: str | None = None,
) -> PairedInferenceResult:
    """Paired warning inference over seed/date blocks preserving cross-portfolio rows."""

    required = {
        date_col,
        portfolio_col,
        label_col,
        reference_score_col,
        challenger_score_col,
    }
    if seed_col:
        required.add(seed_col)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"warning frame is missing columns: {', '.join(missing)}")
    data = frame[list(required)].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    internal_seed_col = seed_col or "__fixed_seed_identity__"
    if not seed_col:
        data[internal_seed_col] = 0
    pairing_keys = [internal_seed_col, portfolio_col, date_col]
    if data.duplicated(pairing_keys).any():
        raise ValueError("warning pairing keys must be unique")
    data = data.sort_values(
        [internal_seed_col, date_col, portfolio_col], kind="mergesort"
    ).reset_index(drop=True)
    seed_ids = np.asarray(sorted(data[internal_seed_col].unique()))
    seed_dates: dict[Any, np.ndarray] = {}
    seed_date_rows: dict[Any, dict[Any, np.ndarray]] = {}
    for seed_id in seed_ids:
        mask = data[internal_seed_col].to_numpy() == seed_id
        dates = np.asarray(sorted(data.loc[mask, date_col].unique()))
        if len(dates) < block_length:
            raise ValueError("every paired warning seed requires at least one full date block")
        seed_dates[seed_id] = dates
        seed_date_rows[seed_id] = {
            date: np.flatnonzero(mask & (data[date_col].to_numpy() == date)) for date in dates
        }
    labels = data[label_col].to_numpy(dtype=int)
    reference = data[reference_score_col].to_numpy(dtype=float)
    challenger = data[challenger_score_col].to_numpy(dtype=float)
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = lambda y, p: _warning_metric(
        metric, y, p, threshold
    )
    estimate = metric_fn(labels, challenger) - metric_fn(labels, reference)

    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(bootstrap_replicates), dtype=float)
    for replicate in range(int(bootstrap_replicates)):
        selected_seed_ids = rng.choice(seed_ids, size=len(seed_ids), replace=True)
        selected_row_parts = []
        for seed_id in selected_seed_ids:
            dates = seed_dates[seed_id]
            selected_dates = dates[_circular_positions(len(dates), block_length, rng)]
            selected_row_parts.extend(seed_date_rows[seed_id][date] for date in selected_dates)
        selected_rows = np.concatenate(selected_row_parts)
        bootstrap[replicate] = metric_fn(labels[selected_rows], challenger[selected_rows]) - metric_fn(
            labels[selected_rows], reference[selected_rows]
        )

    randomized = np.empty(int(randomization_replicates), dtype=float)
    blocks = {
        seed_id: _nonoverlapping_blocks(len(seed_dates[seed_id]), block_length)
        for seed_id in seed_ids
    }
    for replicate in range(int(randomization_replicates)):
        randomized_reference = reference.copy()
        randomized_challenger = challenger.copy()
        for seed_id in seed_ids:
            dates = seed_dates[seed_id]
            for block in blocks[seed_id]:
                if bool(rng.integers(0, 2)):
                    rows = np.concatenate(
                        [seed_date_rows[seed_id][dates[position]] for position in block]
                    )
                    randomized_reference[rows], randomized_challenger[rows] = (
                        randomized_challenger[rows].copy(),
                        randomized_reference[rows].copy(),
                    )
        randomized[replicate] = metric_fn(labels, randomized_challenger) - metric_fn(
            labels, randomized_reference
        )

    alpha = 1.0 - float(confidence_level)
    ci_low, ci_high = np.quantile(bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0])
    p_value = (1.0 + float(np.sum(np.abs(randomized) >= abs(estimate)))) / (
        float(randomization_replicates) + 1.0
    )
    return PairedInferenceResult(
        metric=metric,
        estimate=float(estimate),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        confidence_level=float(confidence_level),
        bootstrap_replicates=int(bootstrap_replicates),
        randomization_replicates=int(randomization_replicates),
        block_length=int(block_length),
        unique_dates=int(data[date_col].nunique()),
        portfolio_count=int(data[portfolio_col].nunique()),
        block_count=int(sum(len(seed_blocks) for seed_blocks in blocks.values())),
        repeat_count=int(len(seed_ids)),
    )


def _allocation_metric(
    metric: str,
    strategy: np.ndarray,
    benchmark: np.ndarray | None,
) -> float:
    values = np.asarray(strategy, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("allocation returns must be a finite vector")
    if metric in {"net_mean_log_return", "gross_mean_log_return"}:
        return float(values.mean())
    if metric in {"net_information_ratio", "gross_information_ratio"}:
        if benchmark is None:
            raise ValueError("information ratio requires aligned benchmark returns")
        active = values - np.asarray(benchmark, dtype=float)
        scale = float(np.std(active, ddof=1))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("information ratio is not estimable with zero active-return variance")
        return float(np.sqrt(252.0) * active.mean() / scale)
    raise ValueError(f"unsupported allocation metric: {metric}")


def paired_allocation_block_inference(
    frame: pd.DataFrame,
    *,
    metric: str,
    date_col: str,
    portfolio_col: str,
    reference_return_col: str,
    challenger_return_col: str,
    benchmark_return_col: str | None,
    block_length: int,
    bootstrap_replicates: int,
    randomization_replicates: int,
    confidence_level: float,
    seed: int,
) -> PairedInferenceResult:
    """Paired allocation inference using time blocks within each portfolio."""

    required = {date_col, portfolio_col, reference_return_col, challenger_return_col}
    if benchmark_return_col:
        required.add(benchmark_return_col)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"allocation frame is missing columns: {', '.join(missing)}")
    data = frame[list(required)].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    if data.duplicated([portfolio_col, date_col]).any():
        raise ValueError("allocation pairing keys must be unique")
    data = data.sort_values([portfolio_col, date_col], kind="mergesort").reset_index(drop=True)
    groups = [group.reset_index(drop=True) for _, group in data.groupby(portfolio_col, sort=True)]
    if not groups or any(len(group) < block_length for group in groups):
        raise ValueError("every allocation portfolio requires at least one full time block")

    def statistic(sampled_groups: list[pd.DataFrame]) -> float:
        sample = pd.concat(sampled_groups, ignore_index=True)
        benchmark = (
            sample[benchmark_return_col].to_numpy(dtype=float) if benchmark_return_col else None
        )
        challenger_value = _allocation_metric(
            metric, sample[challenger_return_col].to_numpy(dtype=float), benchmark
        )
        reference_value = _allocation_metric(
            metric, sample[reference_return_col].to_numpy(dtype=float), benchmark
        )
        return float(challenger_value - reference_value)

    estimate = statistic(groups)
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(bootstrap_replicates), dtype=float)
    for replicate in range(int(bootstrap_replicates)):
        sampled_groups = []
        for group in groups:
            positions = _circular_positions(len(group), block_length, rng)
            sampled_groups.append(group.iloc[positions].reset_index(drop=True))
        bootstrap[replicate] = statistic(sampled_groups)

    block_count = sum(len(_nonoverlapping_blocks(len(group), block_length)) for group in groups)
    randomized = np.empty(int(randomization_replicates), dtype=float)
    for replicate in range(int(randomization_replicates)):
        randomized_groups = []
        for group in groups:
            randomized_group = group.copy()
            for block in _nonoverlapping_blocks(len(group), block_length):
                if bool(rng.integers(0, 2)):
                    reference_values = randomized_group.loc[block, reference_return_col].copy()
                    randomized_group.loc[block, reference_return_col] = randomized_group.loc[
                        block, challenger_return_col
                    ].to_numpy()
                    randomized_group.loc[block, challenger_return_col] = reference_values.to_numpy()
            randomized_groups.append(randomized_group)
        randomized[replicate] = statistic(randomized_groups)

    alpha = 1.0 - float(confidence_level)
    ci_low, ci_high = np.quantile(bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0])
    p_value = (1.0 + float(np.sum(np.abs(randomized) >= abs(estimate)))) / (
        float(randomization_replicates) + 1.0
    )
    return PairedInferenceResult(
        metric=metric,
        estimate=float(estimate),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        confidence_level=float(confidence_level),
        bootstrap_replicates=int(bootstrap_replicates),
        randomization_replicates=int(randomization_replicates),
        block_length=int(block_length),
        unique_dates=int(data[date_col].nunique()),
        portfolio_count=int(data[portfolio_col].nunique()),
        block_count=int(block_count),
        repeat_count=1,
    )
