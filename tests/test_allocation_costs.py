from __future__ import annotations

import numpy as np
import yaml

from models.allocation_costs import cost_scenarios, execute_rebalance, validate_cost_protocol
from reproducibility.release import ROOT


def _protocol() -> dict:
    return validate_cost_protocol(yaml.safe_load((ROOT / "configs/allocation.yaml").read_text()))


def test_cost_scenarios_and_one_way_cost_definition_are_frozen() -> None:
    scenarios = cost_scenarios(_protocol())
    assert [row.scenario_id for row in scenarios] == ["low", "base", "high"]
    assert [row.one_way_cost_bps for row in scenarios] == [5.0, 12.0, 25.0]
    assert [row.primary for row in scenarios] == [False, True, False]
    assert scenarios[1].half_spread_bps == 5.0


def test_rebalance_enforces_joint_long_only_cap_trade_and_turnover_constraints() -> None:
    execution = execute_rebalance(
        rebalance_index=0,
        pretrade_weights=[0.41, 0.20, 0.20, 0.19],
        raw_target_weights=[0.40, 0.40, 0.10, 0.10],
        tradable_assets=[True, True, True, True],
        protocol=_protocol(),
    )
    assert execution.status.startswith("executed_feasible_cap_restoration_projection")
    assert np.isclose(execution.executed_weights.sum(), 1.0, atol=1e-10)
    assert execution.executed_weights.min() >= -1e-12
    assert execution.executed_weights.max() <= 0.40 + 1e-8
    assert np.abs(execution.executed_weights - execution.pretrade_weights).max() <= 0.10 + 1e-8
    assert execution.one_way_turnover <= 0.25 + 1e-8
    assert np.isclose(execution.one_way_turnover, 0.5 * execution.l1_turnover)


def test_untradable_input_holds_the_entire_rebalance() -> None:
    execution = execute_rebalance(
        rebalance_index=0,
        pretrade_weights=[0.4, 0.3, 0.2, 0.1],
        raw_target_weights=[0.25, 0.25, 0.25, 0.25],
        tradable_assets=[True, False, True, True],
        protocol=_protocol(),
    )
    assert execution.status == "held_untradable"
    assert execution.one_way_turnover == 0.0
    np.testing.assert_allclose(execution.executed_weights, execution.pretrade_weights)
