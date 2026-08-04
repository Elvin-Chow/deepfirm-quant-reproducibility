"""Shared warning metrics used by the public result checks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def validate_probability_vector(values: object, *, context: str = "probabilities") -> np.ndarray:
    probabilities = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{context} contains non-finite values")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError(f"{context} must lie within [0, 1]")
    return probabilities


def expected_calibration_error(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = validate_probability_vector(probabilities)
    if y.size == 0 or y.size != p.size:
        return math.nan
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    result = 0.0
    for index in range(int(n_bins)):
        left, right = edges[index], edges[index + 1]
        mask = (p >= left) & ((p <= right) if index == n_bins - 1 else (p < right))
        if mask.any():
            result += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(result)


def deterministic_exact_k_indices(
    probabilities: Sequence[float] | np.ndarray,
    row_ids: Sequence[object] | np.ndarray,
    *,
    fraction: float,
) -> np.ndarray:
    p = validate_probability_vector(probabilities)
    ids = np.asarray(row_ids, dtype=str).reshape(-1)
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must lie within (0, 1]")
    if len(ids) != len(p):
        raise ValueError("probabilities and row_ids must have the same length")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("row_ids must be unique")
    count = int(math.ceil(float(fraction) * p.size)) if p.size else 0
    return np.lexsort((ids, -p))[:count]


def top_decile_lift(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    row_ids: Sequence[object] | np.ndarray,
) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = validate_probability_vector(probabilities)
    if y.size == 0 or y.size != p.size or float(y.mean()) <= 0.0:
        return math.nan
    selected = deterministic_exact_k_indices(p, row_ids, fraction=0.10)
    return float(y[selected].mean() / y.mean())


def warning_metrics(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    row_ids: Sequence[object] | np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = validate_probability_vector(probabilities)
    if y.size != p.size:
        raise ValueError("labels and probabilities must have the same length")
    threshold_value = float(threshold)
    if not 0.0 <= threshold_value <= 1.0:
        raise ValueError("threshold must lie within [0, 1]")
    # Decimal CSV round trips can place a score that was exactly equal to the
    # frozen threshold one representable float below its JSON value. Including
    # one lower ULP preserves the declared include-equal-score tie rule.
    effective_threshold = np.nextafter(threshold_value, -np.inf)
    flags = p >= effective_threshold
    positives = int(y.sum())
    true_positives = int(y[flags].sum()) if flags.any() else 0
    return {
        "row_count": float(y.size),
        "positive_event_count": float(positives),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else math.nan,
        "average_precision": float(average_precision_score(y, p)) if positives else math.nan,
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-9, 1.0 - 1e-9), labels=[0, 1])),
        "expected_calibration_error": expected_calibration_error(y, p),
        "precision_at_threshold": float(true_positives / flags.sum()) if flags.any() else 0.0,
        "recall_at_threshold": float(true_positives / positives) if positives else math.nan,
        "flag_rate_at_threshold": float(flags.mean()),
        "top_decile_lift": top_decile_lift(y, p, row_ids),
    }
