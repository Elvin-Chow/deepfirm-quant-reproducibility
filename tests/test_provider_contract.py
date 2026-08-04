from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reproducibility.provider_contract import (
    REQUIRED_OUTCOME_END,
    log_returns_from_prices,
    parse_official_index_payload,
    validate_same_index_returns,
)


def _payload(*, code: str = "000300", end: str = "2026-07-31") -> dict:
    dates = pd.bdate_range(end=pd.Timestamp(end), periods=320)
    closes = 4000.0 * np.exp(np.cumsum(0.0002 + 0.002 * np.sin(np.arange(len(dates)) / 17.0)))
    return {
        "data": [
            {"tradeDate": date.strftime("%Y%m%d"), "indexCode": code, "close": float(close)}
            for date, close in zip(dates, closes)
        ]
    }


def test_official_index_identity_overlap_and_outcome_date_gates_pass() -> None:
    frame = parse_official_index_payload(_payload())
    returns = log_returns_from_prices(frame)
    result = validate_same_index_returns(returns.copy(), returns)
    assert result["identity_gate"] == "indexCode_exact_000300"
    assert result["official_observed_end"] == REQUIRED_OUTCOME_END.date().isoformat()
    assert result["overlap_rows"] >= 250
    assert result["overlap_gate"] == "PASS"


def test_identity_and_temporal_coverage_fail_closed() -> None:
    with pytest.raises(ValueError, match="index identity"):
        parse_official_index_payload(_payload(code="000905"))
    incomplete = log_returns_from_prices(parse_official_index_payload(_payload(end="2026-07-17")))
    with pytest.raises(ValueError, match="required outcome date"):
        validate_same_index_returns(incomplete.copy(), incomplete)
