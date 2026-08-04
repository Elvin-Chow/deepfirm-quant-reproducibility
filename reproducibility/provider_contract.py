"""Result-independent checks for the unchanged CSI 300 completeness route."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


CSI300_SYMBOL = "000300"
MINIMUM_OVERLAP_RETURNS = 250
MINIMUM_RETURN_CORRELATION = 0.999
MAXIMUM_RETURN_ABSOLUTE_DIFFERENCE = 0.001
REQUIRED_OUTCOME_END = pd.Timestamp("2026-07-31")


def parse_official_index_payload(payload: dict[str, Any], symbol: str = CSI300_SYMBOL) -> pd.DataFrame:
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        raise ValueError("official index response contains no rows")
    frame = pd.DataFrame(rows)
    required = {"tradeDate", "indexCode", "close"}
    if not required.issubset(frame.columns):
        raise ValueError("official index response is missing identity, date, or close fields")
    codes = frame["indexCode"].astype(str).str.strip().str.zfill(6)
    if set(codes) != {symbol}:
        raise ValueError("official index response changed the index identity")
    dates = pd.to_datetime(frame["tradeDate"].astype(str), format="%Y%m%d", errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    if dates.isna().any() or close.isna().any():
        raise ValueError("official index response contains invalid dates or closes")
    values = close.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("official index response contains non-positive or non-finite closes")
    normalized = pd.DataFrame({"Date": dates.dt.normalize(), "Close": values})
    normalized = normalized.sort_values("Date", kind="mergesort")
    normalized = normalized.drop_duplicates(subset=["Date"], keep="last")
    if len(normalized) < 2:
        raise ValueError("official index response has fewer than two valid rows")
    return normalized.reset_index(drop=True)


def log_returns_from_prices(frame: pd.DataFrame) -> pd.Series:
    prices = frame.set_index("Date")["Close"].astype(float).sort_index(kind="mergesort")
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    return np.log(prices / prices.shift(1)).dropna().rename("benchmark_log_return")


def validate_same_index_returns(
    primary: pd.Series,
    official: pd.Series,
    *,
    required_outcome_end: pd.Timestamp = REQUIRED_OUTCOME_END,
) -> dict[str, Any]:
    if official.empty or official.index.max() < required_outcome_end:
        raise ValueError("official CSI 300 series ends before the required outcome date")
    common = primary.index.intersection(official.index)
    if len(common) < MINIMUM_OVERLAP_RETURNS:
        raise ValueError("primary and official routes have insufficient return overlap")
    left = primary.loc[common].to_numpy(dtype=float)
    right = official.loc[common].to_numpy(dtype=float)
    correlation = float(np.corrcoef(left, right)[0, 1])
    maximum_difference = float(np.max(np.abs(left - right)))
    if not math.isfinite(correlation) or correlation < MINIMUM_RETURN_CORRELATION:
        raise ValueError("same-index return correlation is below the frozen gate")
    if not math.isfinite(maximum_difference) or maximum_difference > MAXIMUM_RETURN_ABSOLUTE_DIFFERENCE:
        raise ValueError("same-index return difference exceeds the frozen gate")
    return {
        "identity_gate": "indexCode_exact_000300",
        "official_observed_end": official.index.max().date().isoformat(),
        "overlap_rows": int(len(common)),
        "overlap_return_correlation": correlation,
        "overlap_return_max_abs_difference": maximum_difference,
        "overlap_gate": "PASS",
    }
