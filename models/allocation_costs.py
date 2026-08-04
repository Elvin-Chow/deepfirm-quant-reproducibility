"""Point-in-time investability and transaction-cost execution for allocation paths.

The cost/investability contract is an execution-layer child of the frozen allocation-view/component allocation
design.  It does not alter how a raw target is produced.  Instead, it converts
each raw target into an executable long-only target, measures turnover against
the drifted pre-trade portfolio, and applies explicit fee, half-spread, and
slippage deductions at the rebalance boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize


@dataclass(frozen=True)
class CostScenario:
    scenario_id: str
    fee_bps: float
    quoted_spread_bps: float
    slippage_bps: float
    primary: bool = False

    @property
    def half_spread_bps(self) -> float:
        return 0.5 * self.quoted_spread_bps

    @property
    def one_way_cost_bps(self) -> float:
        return self.fee_bps + self.half_spread_bps + self.slippage_bps


@dataclass(frozen=True)
class RebalanceExecution:
    rebalance_index: int
    pretrade_weights: np.ndarray
    raw_target_weights: np.ndarray
    constrained_target_weights: np.ndarray
    executed_weights: np.ndarray
    effective_position_cap: float
    execution_fraction: float
    l1_turnover: float
    one_way_turnover: float
    tradable: bool
    status: str


@dataclass
class CostAwarePath:
    gross_log_returns: np.ndarray
    net_log_returns_by_scenario: dict[str, np.ndarray]
    executions: list[RebalanceExecution]
    cost_ledger: list[dict[str, Any]]


def _unit_sum_long_only(weights: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(weights, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    if float(array.min()) < -1e-12:
        raise ValueError(f"{name} violates the long-only rule")
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= 1e-12:
        raise ValueError(f"{name} has no investable mass")
    return array / total


def _project_capped_simplex(weights: np.ndarray, cap: float) -> np.ndarray:
    """Euclidean projection onto {w >= 0, sum(w)=1, w_i <= cap}."""
    n_assets = int(weights.size)
    effective_cap = max(float(cap), 1.0 / float(n_assets))
    if effective_cap >= 1.0:
        return _unit_sum_long_only(weights, name="target_weights")
    lower = float(np.min(weights) - effective_cap)
    upper = float(np.max(weights))
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        projected = np.clip(weights - midpoint, 0.0, effective_cap)
        if float(projected.sum()) > 1.0:
            lower = midpoint
        else:
            upper = midpoint
    projected = np.clip(weights - upper, 0.0, effective_cap)
    residual = 1.0 - float(projected.sum())
    if abs(residual) > 1e-10:
        order = np.argsort(weights, kind="mergesort")[::-1]
        for index in order:
            room = effective_cap - float(projected[index])
            addition = min(max(residual, 0.0), max(room, 0.0))
            projected[index] += addition
            residual -= addition
            if residual <= 1e-12:
                break
    if abs(float(projected.sum()) - 1.0) > 1e-8:
        raise ValueError("position-cap projection failed to preserve full investment")
    if float(projected.max()) > effective_cap + 1e-8:
        raise ValueError("position-cap projection exceeded the effective cap")
    return projected


def _project_jointly_feasible_trade(
    pretrade: np.ndarray,
    constrained_target: np.ndarray,
    *,
    effective_cap: float,
    max_trade_weight_per_asset: float,
    max_one_way_turnover: float,
) -> tuple[np.ndarray, str]:
    """Find the closest trade that satisfies every execution constraint.

    HiGHS supplies a deterministic feasible point and L1 fallback. SLSQP then
    minimizes squared distance to the capped target. A genuinely infeasible
    trade raises instead of weakening a constraint.
    """

    n_assets = int(pretrade.size)
    variable_count = 3 * n_assets
    inequalities = np.zeros((4 * n_assets + 1, variable_count), dtype=float)
    limits = np.zeros(4 * n_assets + 1, dtype=float)
    for index in range(n_assets):
        trade_positive = 4 * index
        trade_negative = trade_positive + 1
        target_positive = trade_positive + 2
        target_negative = trade_positive + 3

        inequalities[trade_positive, index] = 1.0
        inequalities[trade_positive, n_assets + index] = -1.0
        limits[trade_positive] = pretrade[index]
        inequalities[trade_negative, index] = -1.0
        inequalities[trade_negative, n_assets + index] = -1.0
        limits[trade_negative] = -pretrade[index]

        inequalities[target_positive, index] = 1.0
        inequalities[target_positive, 2 * n_assets + index] = -1.0
        limits[target_positive] = constrained_target[index]
        inequalities[target_negative, index] = -1.0
        inequalities[target_negative, 2 * n_assets + index] = -1.0
        limits[target_negative] = -constrained_target[index]

    inequalities[-1, n_assets : 2 * n_assets] = 1.0
    limits[-1] = 2.0 * max_one_way_turnover
    equality = np.zeros((1, variable_count), dtype=float)
    equality[0, :n_assets] = 1.0
    objective = np.zeros(variable_count, dtype=float)
    objective[2 * n_assets :] = 1.0
    variable_bounds = [
        *([(0.0, effective_cap)] * n_assets),
        *([(0.0, max_trade_weight_per_asset)] * n_assets),
        *([(0.0, None)] * n_assets),
    ]
    feasible = linprog(
        objective,
        A_ub=inequalities,
        b_ub=limits,
        A_eq=equality,
        b_eq=np.asarray([1.0]),
        bounds=variable_bounds,
        method="highs",
    )
    if not feasible.success:
        raise ValueError(
            "trade, turnover, and position-cap constraints are jointly infeasible"
        )

    def projection_objective(values: np.ndarray) -> float:
        return float(np.square(values[:n_assets] - constrained_target).sum())

    lower = np.zeros(variable_count, dtype=float)
    upper = np.asarray(
        [effective_cap] * n_assets
        + [max_trade_weight_per_asset] * n_assets
        + [np.inf] * n_assets,
        dtype=float,
    )
    optimized = minimize(
        projection_objective,
        feasible.x,
        method="SLSQP",
        bounds=Bounds(lower, upper),
        constraints=[
            LinearConstraint(inequalities, -np.inf, limits),
            LinearConstraint(equality, 1.0, 1.0),
        ],
        options={"ftol": 1e-12, "maxiter": 500, "disp": False},
    )
    candidates: list[tuple[np.ndarray, str]] = []
    if optimized.success:
        candidates.append((optimized.x, "slsqp_l2"))
    candidates.append((feasible.x, "highs_l1_fallback"))

    for values, solver in candidates:
        try:
            executed = _unit_sum_long_only(
                values[:n_assets], name="projected_executed_weights"
            )
        except ValueError:
            continue
        trades = np.abs(executed - pretrade)
        if float(executed.max()) > effective_cap + 1e-8:
            continue
        if float(trades.max()) > max_trade_weight_per_asset + 1e-8:
            continue
        if 0.5 * float(trades.sum()) > max_one_way_turnover + 1e-8:
            continue
        return executed, solver
    raise ValueError("executable-trade projection failed post-solve validation")


def validate_cost_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    """Validate and return the frozen cost/investability execution contract."""
    if protocol.get("protocol_version") not in {
        "cost-investability-v1",
        "allocation-and-cost-v1",
    }:
        raise ValueError("unsupported cost/investability cost protocol version")
    turnover = protocol.get("turnover", {})
    if turnover.get("reported_measure") != "sum_absolute_weight_changes":
        raise ValueError("cost/investability reported turnover must preserve the component L1 field")
    if turnover.get("cost_base") != "one_way_turnover_equals_half_l1":
        raise ValueError("cost/investability realized costs must use one-way turnover")
    if turnover.get("pretrade_reference") != "drifted_end_of_previous_holding_weights":
        raise ValueError("cost/investability turnover must use drifted pre-trade weights")

    rebalance = protocol.get("rebalancing", {})
    if int(rebalance.get("sessions", 0)) != 21:
        raise ValueError("cost/investability rebalance interval must remain 21 sessions")
    if rebalance.get("terminal_segment") != "allow_shorter_final_holding_block":
        raise ValueError("unsupported cost/investability terminal-segment rule")

    constraints = protocol.get("investability", {})
    required_literals = {
        "position_rule": "long_only_fully_invested",
        "short_sales": "prohibited",
        "position_cap_rule": "max(configured_cap,1/n_assets)",
        "untradable_action": "hold_all_and_log",
        "liquidity_interpretation": "portfolio_weight_trade_cap_proxy_no_ADV_capacity_claim",
    }
    for key, expected in required_literals.items():
        if constraints.get(key) != expected:
            raise ValueError(f"unsupported cost/investability investability {key}")
    for key in ("configured_position_cap", "max_trade_weight_per_asset", "max_one_way_turnover"):
        value = float(constraints.get(key, 0.0))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"cost/investability investability {key} must lie in (0,1]")
    if int(constraints.get("minimum_history_observations", 0)) < 1:
        raise ValueError("minimum_history_observations must be positive")
    if float(constraints.get("maximum_missing_fraction", -1.0)) != 0.0:
        raise ValueError("cost/investability tradability requires complete aligned history")

    scenarios = protocol.get("cost_scenarios", [])
    if len(scenarios) != 3:
        raise ValueError("cost/investability freezes exactly low, base, and high cost scenarios")
    parsed = []
    for row in scenarios:
        scenario = CostScenario(
            scenario_id=str(row["scenario_id"]),
            fee_bps=float(row["fee_bps"]),
            quoted_spread_bps=float(row["quoted_spread_bps"]),
            slippage_bps=float(row["slippage_bps"]),
            primary=bool(row.get("primary", False)),
        )
        if min(scenario.fee_bps, scenario.quoted_spread_bps, scenario.slippage_bps) < 0.0:
            raise ValueError("cost components cannot be negative")
        if not np.isclose(
            float(row.get("one_way_cost_bps", scenario.one_way_cost_bps)),
            scenario.one_way_cost_bps,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("declared one-way cost does not equal fee + half-spread + slippage")
        parsed.append(scenario)
    if [row.scenario_id for row in parsed] != ["low", "base", "high"]:
        raise ValueError("cost/investability cost-scenario order must be low/base/high")
    if sum(row.primary for row in parsed) != 1 or not parsed[1].primary:
        raise ValueError("cost/investability base scenario must be the unique primary scenario")
    rates = [row.one_way_cost_bps for row in parsed]
    if not rates[0] < rates[1] < rates[2]:
        raise ValueError("cost/investability sensitivity costs must increase from low to high")
    return protocol


def cost_scenarios(protocol: dict[str, Any]) -> list[CostScenario]:
    validate_cost_protocol(protocol)
    return [
        CostScenario(
            scenario_id=str(row["scenario_id"]),
            fee_bps=float(row["fee_bps"]),
            quoted_spread_bps=float(row["quoted_spread_bps"]),
            slippage_bps=float(row["slippage_bps"]),
            primary=bool(row.get("primary", False)),
        )
        for row in protocol["cost_scenarios"]
    ]


def resolve_tradability(history: pd.DataFrame, protocol: dict[str, Any]) -> np.ndarray:
    """Resolve the point-in-time complete-history tradability gate."""
    validate_cost_protocol(protocol)
    constraints = protocol["investability"]
    lookback = history.tail(int(constraints["tradability_lookback_sessions"]))
    minimum = int(constraints["minimum_history_observations"])
    finite = np.isfinite(lookback.to_numpy(dtype=float))
    observations = finite.sum(axis=0)
    missing_fraction = 1.0 - finite.mean(axis=0)
    return (observations >= minimum) & (
        missing_fraction <= float(constraints["maximum_missing_fraction"])
    )


def execute_rebalance(
    *,
    rebalance_index: int,
    pretrade_weights: Sequence[float],
    raw_target_weights: Sequence[float],
    tradable_assets: Sequence[bool],
    protocol: dict[str, Any],
) -> RebalanceExecution:
    """Convert one raw target into an executable target without future data."""
    validate_cost_protocol(protocol)
    pretrade = _unit_sum_long_only(pretrade_weights, name="pretrade_weights")
    raw_target = _unit_sum_long_only(raw_target_weights, name="raw_target_weights")
    if pretrade.size != raw_target.size:
        raise ValueError("pre-trade and target vectors have different lengths")
    tradable = np.asarray(tradable_assets, dtype=bool)
    if tradable.shape != pretrade.shape:
        raise ValueError("tradability mask has the wrong shape")

    constraints = protocol["investability"]
    effective_cap = max(
        float(constraints["configured_position_cap"]),
        1.0 / float(pretrade.size),
    )
    constrained = _project_capped_simplex(raw_target, effective_cap)
    if not bool(tradable.all()):
        return RebalanceExecution(
            rebalance_index=rebalance_index,
            pretrade_weights=pretrade,
            raw_target_weights=raw_target,
            constrained_target_weights=constrained,
            executed_weights=pretrade.copy(),
            effective_position_cap=effective_cap,
            execution_fraction=0.0,
            l1_turnover=0.0,
            one_way_turnover=0.0,
            tradable=False,
            status="held_untradable",
        )

    desired_trade = constrained - pretrade
    desired_one_way = 0.5 * float(np.abs(desired_trade).sum())
    max_abs_trade = float(np.abs(desired_trade).max())
    fractions = [1.0]
    if desired_one_way > 1e-15:
        fractions.append(float(constraints["max_one_way_turnover"]) / desired_one_way)
    if max_abs_trade > 1e-15:
        fractions.append(float(constraints["max_trade_weight_per_asset"]) / max_abs_trade)
    fraction = float(np.clip(min(fractions), 0.0, 1.0))
    executed = pretrade + fraction * desired_trade
    executed = _unit_sum_long_only(executed, name="executed_weights")
    status = "executed" if fraction >= 1.0 - 1e-12 else "partially_executed_trade_cap"
    if float(executed.max()) > effective_cap + 1e-8:
        executed, solver = _project_jointly_feasible_trade(
            pretrade,
            constrained,
            effective_cap=effective_cap,
            max_trade_weight_per_asset=float(
                constraints["max_trade_weight_per_asset"]
            ),
            max_one_way_turnover=float(constraints["max_one_way_turnover"]),
        )
        projected_one_way = 0.5 * float(np.abs(executed - pretrade).sum())
        fraction = (
            float(np.clip(projected_one_way / desired_one_way, 0.0, 1.0))
            if desired_one_way > 1e-15
            else 1.0
        )
        status = "executed_feasible_cap_restoration_projection"
        if solver != "slsqp_l2":
            status += "_highs_l1_fallback"
    l1_turnover = float(np.abs(executed - pretrade).sum())
    one_way_turnover = 0.5 * l1_turnover
    return RebalanceExecution(
        rebalance_index=rebalance_index,
        pretrade_weights=pretrade,
        raw_target_weights=raw_target,
        constrained_target_weights=constrained,
        executed_weights=executed,
        effective_position_cap=effective_cap,
        execution_fraction=fraction,
        l1_turnover=l1_turnover,
        one_way_turnover=one_way_turnover,
        tradable=True,
        status=status,
    )


def drift_weights(weights: Sequence[float], asset_log_returns: pd.DataFrame) -> np.ndarray:
    """Drift holdings through a completed holding block without rebalancing."""
    current = _unit_sum_long_only(weights, name="holding_weights")
    for row in asset_log_returns.to_numpy(dtype=float):
        if not np.isfinite(row).all():
            raise ValueError("holding returns contain non-finite values")
        gross_assets = np.exp(row)
        current = current * gross_assets
        total = float(current.sum())
        if total <= 1e-12 or not np.isfinite(total):
            raise ValueError("holding-period wealth is invalid")
        current /= total
    return current


def simulate_cost_aware_path(
    *,
    asset_log_returns: pd.DataFrame,
    target_weights_by_rebalance: Sequence[Sequence[float]],
    initial_weights: Sequence[float],
    tradable_by_rebalance: Sequence[Sequence[bool]],
    protocol: dict[str, Any],
) -> CostAwarePath:
    """Simulate gross/net paths and a per-rebalance cost ledger."""
    validate_cost_protocol(protocol)
    if asset_log_returns.empty:
        raise ValueError("asset_log_returns cannot be empty")
    rebalance_sessions = int(protocol["rebalancing"]["sessions"])
    expected_rebalances = (len(asset_log_returns) + rebalance_sessions - 1) // rebalance_sessions
    if len(target_weights_by_rebalance) != expected_rebalances:
        raise ValueError("target-weight count does not match the rebalance schedule")
    if len(tradable_by_rebalance) != expected_rebalances:
        raise ValueError("tradability count does not match the rebalance schedule")

    scenarios = cost_scenarios(protocol)
    gross_log_returns = np.zeros(len(asset_log_returns), dtype=float)
    net_by_scenario = {
        scenario.scenario_id: np.zeros(len(asset_log_returns), dtype=float)
        for scenario in scenarios
    }
    executions: list[RebalanceExecution] = []
    ledger: list[dict[str, Any]] = []
    pretrade = _unit_sum_long_only(initial_weights, name="initial_weights")

    for rebalance_index in range(expected_rebalances):
        start = rebalance_index * rebalance_sessions
        stop = min(start + rebalance_sessions, len(asset_log_returns))
        holding = asset_log_returns.iloc[start:stop]
        execution = execute_rebalance(
            rebalance_index=rebalance_index,
            pretrade_weights=pretrade,
            raw_target_weights=target_weights_by_rebalance[rebalance_index],
            tradable_assets=tradable_by_rebalance[rebalance_index],
            protocol=protocol,
        )
        executions.append(execution)

        weights = execution.executed_weights.copy()
        for local_index, row in enumerate(holding.to_numpy(dtype=float)):
            simple_asset_returns = np.expm1(row)
            simple_portfolio_return = float(weights @ simple_asset_returns)
            if simple_portfolio_return <= -1.0:
                raise ValueError("portfolio return is outside the log-return domain")
            absolute_index = start + local_index
            gross_log_returns[absolute_index] = float(np.log1p(simple_portfolio_return))
            post_return = weights * (1.0 + simple_asset_returns)
            weights = post_return / float(post_return.sum())

        for scenario in scenarios:
            fee = execution.one_way_turnover * scenario.fee_bps / 10_000.0
            spread = execution.one_way_turnover * scenario.half_spread_bps / 10_000.0
            slippage = execution.one_way_turnover * scenario.slippage_bps / 10_000.0
            total_cost = fee + spread + slippage
            if total_cost >= 1.0:
                raise ValueError("transaction cost exhausts portfolio wealth")
            net_by_scenario[scenario.scenario_id][start:stop] = gross_log_returns[start:stop]
            net_by_scenario[scenario.scenario_id][start] += float(np.log1p(-total_cost))
            ledger.append(
                {
                    "rebalance_index": rebalance_index,
                    "rebalance_date": pd.Timestamp(holding.index[0]).date().isoformat(),
                    "scenario_id": scenario.scenario_id,
                    "primary_scenario": scenario.primary,
                    "fee_bps": scenario.fee_bps,
                    "quoted_spread_bps": scenario.quoted_spread_bps,
                    "half_spread_bps": scenario.half_spread_bps,
                    "slippage_bps": scenario.slippage_bps,
                    "one_way_cost_bps": scenario.one_way_cost_bps,
                    "fee_cost": fee,
                    "spread_cost": spread,
                    "slippage_cost": slippage,
                    "realized_cost": total_cost,
                    "l1_turnover": execution.l1_turnover,
                    "one_way_turnover": execution.one_way_turnover,
                    "pretrade_weights": execution.pretrade_weights.tolist(),
                    "raw_target_weights": execution.raw_target_weights.tolist(),
                    "constrained_target_weights": execution.constrained_target_weights.tolist(),
                    "executed_weights": execution.executed_weights.tolist(),
                    "effective_position_cap": execution.effective_position_cap,
                    "execution_fraction": execution.execution_fraction,
                    "tradable": execution.tradable,
                    "status": execution.status,
                }
            )
        pretrade = weights

    return CostAwarePath(
        gross_log_returns=gross_log_returns,
        net_log_returns_by_scenario=net_by_scenario,
        executions=executions,
        cost_ledger=ledger,
    )
