from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.temporal_validation import four_way_purged_embargo_split, purged_embargo_split


def _frame(*, grouped: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=100, freq="B")
    groups = ["portfolio_a", "portfolio_b"] if grouped else ["portfolio_a"]
    frames = [
        pd.DataFrame(
            {"tail_event": np.tile([0, 0, 1, 0, 0], 20), "domain_portfolio": group},
            index=dates,
        )
        for group in groups
    ]
    return pd.concat(frames).sort_index(kind="mergesort")


@pytest.mark.parametrize("horizon", [1, 5])
def test_horizon_aware_purge_and_embargo_remove_boundary_overlap(horizon: int) -> None:
    frame = _frame()
    split = purged_embargo_split(
        frame,
        horizon=horizon,
        evaluation_fraction=0.20,
        min_train_rows=20,
        min_evaluation_rows=10,
    )
    dates = pd.DatetimeIndex(frame.index.unique()).sort_values()
    last_train = dates.get_loc(split.train.index.max())
    first_evaluation = dates.get_loc(split.evaluation.index.min())
    assert first_evaluation - last_train - 1 >= 2 * horizon
    assert split.audit["purged_rows"] == horizon
    assert split.audit["embargoed_rows"] == horizon


def test_four_roles_remain_chronological_and_security_group_aware() -> None:
    split = four_way_purged_embargo_split(
        _frame(grouped=True),
        horizon=5,
        fractions={"train": 0.55, "calibration": 0.15, "validation": 0.20, "final_test": 0.10},
        group_columns=("domain_portfolio",),
        min_rows={"train": 40, "calibration": 10, "validation": 20, "final_test": 5},
    )
    roles = [split.train, split.calibration, split.validation, split.final_test]
    assert all(earlier.index.max() < later.index.min() for earlier, later in zip(roles, roles[1:]))
    for boundary in split.audit["boundaries"].values():
        assert boundary["purged_rows"] == 10
        assert boundary["embargoed_rows"] == 10


def test_four_way_fraction_contract_fails_closed() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        four_way_purged_embargo_split(
            _frame(),
            horizon=1,
            fractions={"train": 0.50, "calibration": 0.10, "validation": 0.20, "final_test": 0.10},
        )
