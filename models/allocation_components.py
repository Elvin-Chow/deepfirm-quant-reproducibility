"""Point-in-time allocation views and identifiable component interventions.

allocation-view keeps the allocation view model deliberately deterministic and auditable.
It is separate from the crisis-warning artifact: only trailing asset returns
available through the rebalance information cutoff enter the view.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from models.allocation_policy import AllocationPolicyResult
from models.portfolio_opt import ViewSpec


ComponentState = Literal["enabled", "removed", "dynamic", "fixed", "adaptive", "disabled"]


class AllocationViewResult(BaseModel):
    """Auditable relative view resolved at one rebalance cutoff."""

    version: str
    asof_date: str
    source: str
    lookback_observations: int = Field(ge=1)
    long_assets: list[str]
    short_assets: list[str]
    asset_scores: dict[str, float]
    expected_return: float
    confidence: float = Field(gt=0.0, le=1.0)
    confidence_mode: Literal["dynamic", "fixed"] = "dynamic"

    def as_view_spec(self) -> ViewSpec:
        return ViewSpec(
            assets=self.long_assets,
            relative_assets=self.short_assets,
            expected_return=self.expected_return,
            confidence=self.confidence,
        )


class AllocationViewEngine:
    """Resolve one cross-sectional relative view from trailing returns."""

    VERSION = "allocation-view-v1"
    PROTOCOL: dict[str, Any] = {
        "version": VERSION,
        "source": "trailing_cross_sectional_risk_adjusted_return",
        "timing": "each_rebalance_from_complete_returns_through_information_cutoff",
        "lookback_sessions": 63,
        "minimum_observations": 40,
        "annualization_days": 252,
        "leg_fraction": 0.34,
        "minimum_leg_assets": 1,
        "expected_return_clip": [-0.20, 0.20],
        "score_epsilon": 1e-12,
        "stable_tie_break": "ticker_ascending",
        "confidence": {
            "method": "coverage_times_score_separation",
            "floor": 0.20,
            "ceiling": 0.80,
            "formula": "floor+(ceiling-floor)*coverage*separation/(1+separation)",
        },
        "warning_artifact_dependency": "none",
        "failure_handling": "fail_closed_and_log_skip",
    }

    @classmethod
    def protocol(cls) -> dict[str, Any]:
        return deepcopy(cls.PROTOCOL)

    @classmethod
    def resolve_from_returns(
        cls,
        tickers: list[str],
        returns_df: pd.DataFrame,
        asof_date: str | pd.Timestamp,
        *,
        confidence_override: float | None = None,
    ) -> AllocationViewResult:
        protocol = cls.PROTOCOL
        if len(tickers) < 2:
            raise ValueError("allocation view requires at least two assets")
        if returns_df.empty:
            raise ValueError("allocation view requires non-empty trailing returns")
        if list(returns_df.columns) != list(tickers):
            raise ValueError("allocation view columns must exactly match ticker order")

        cutoff = pd.Timestamp(asof_date)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_localize(None)
        index = pd.DatetimeIndex(returns_df.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        if index.max() > cutoff:
            raise ValueError("allocation view input extends beyond information cutoff")

        lookback = int(protocol["lookback_sessions"])
        trailing = returns_df.tail(lookback).replace([np.inf, -np.inf], np.nan).dropna(how="any")
        minimum = int(protocol["minimum_observations"])
        if len(trailing) < minimum:
            raise ValueError(
                f"allocation view requires at least {minimum} complete observations; got {len(trailing)}"
            )

        means = trailing.mean(axis=0).to_numpy(dtype=float)
        stds = trailing.std(axis=0, ddof=1).to_numpy(dtype=float)
        epsilon = float(protocol["score_epsilon"])
        scores = means / np.maximum(stds, epsilon) * np.sqrt(
            float(protocol["annualization_days"])
        )
        if not np.isfinite(scores).all():
            raise ValueError("allocation view produced non-finite asset scores")

        ranked = sorted(range(len(tickers)), key=lambda idx: (-float(scores[idx]), tickers[idx]))
        leg_count = max(
            int(protocol["minimum_leg_assets"]),
            int(np.floor(len(tickers) * float(protocol["leg_fraction"]))),
        )
        leg_count = min(leg_count, len(tickers) // 2)
        if leg_count < 1:
            raise ValueError("allocation view could not form disjoint legs")
        long_indices = ranked[:leg_count]
        short_indices = ranked[-leg_count:]
        if set(long_indices) & set(short_indices):
            raise RuntimeError("allocation view legs overlap")

        annualization = float(protocol["annualization_days"])
        raw_expected_return = annualization * (
            float(means[long_indices].mean()) - float(means[short_indices].mean())
        )
        expected_low, expected_high = protocol["expected_return_clip"]
        expected_return = float(np.clip(raw_expected_return, expected_low, expected_high))

        long_score = float(scores[long_indices].mean())
        short_score = float(scores[short_indices].mean())
        separation = max(long_score - short_score, 0.0)
        coverage = min(float(len(trailing)) / float(lookback), 1.0)
        confidence_contract = protocol["confidence"]
        floor = float(confidence_contract["floor"])
        ceiling = float(confidence_contract["ceiling"])
        dynamic_confidence = floor + (ceiling - floor) * coverage * separation / (1.0 + separation)
        confidence_mode: Literal["dynamic", "fixed"] = "dynamic"
        confidence = float(np.clip(dynamic_confidence, floor, ceiling))
        if confidence_override is not None:
            confidence = float(confidence_override)
            if not floor <= confidence <= ceiling:
                raise ValueError("fixed view confidence must lie within the frozen bounds")
            confidence_mode = "fixed"

        return AllocationViewResult(
            version=cls.VERSION,
            asof_date=cutoff.date().isoformat(),
            source=str(protocol["source"]),
            lookback_observations=int(len(trailing)),
            long_assets=[tickers[idx] for idx in long_indices],
            short_assets=[tickers[idx] for idx in short_indices],
            asset_scores={ticker: float(score) for ticker, score in zip(tickers, scores)},
            expected_return=expected_return,
            confidence=confidence,
            confidence_mode=confidence_mode,
        )


BASELINE_COMPONENT_STATES = {
    "model_view": "enabled",
    "view_uncertainty": "dynamic",
    "smart_policy": "adaptive",
    "guard": "enabled",
}


def validate_component_ablation_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    """Require every non-reference variant to intervene on exactly one component."""

    if protocol.get("reference_variant") != "full_stack":
        raise ValueError("component ablation reference must be full_stack")
    variants = protocol.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("component ablation protocol requires variants")
    by_id = {str(row.get("variant_id")): row for row in variants}
    if set(by_id) != {
        "full_stack",
        "without_model_view",
        "fixed_view_uncertainty",
        "fixed_smart_policy",
        "guard_disabled",
    }:
        raise ValueError("component ablation variants do not match the frozen allocation-view design")
    reference_states = by_id["full_stack"].get("component_states")
    if reference_states != BASELINE_COMPONENT_STATES:
        raise ValueError("full_stack component states do not match the executable baseline")

    expected_change = {
        "without_model_view": "model_view",
        "fixed_view_uncertainty": "view_uncertainty",
        "fixed_smart_policy": "smart_policy",
        "guard_disabled": "guard",
    }
    for variant_id, component in expected_change.items():
        row = by_id[variant_id]
        states = row.get("component_states")
        if not isinstance(states, dict) or set(states) != set(reference_states):
            raise ValueError(f"{variant_id} has an invalid component-state schema")
        changed = [name for name in reference_states if states[name] != reference_states[name]]
        if changed != [component]:
            raise ValueError(f"{variant_id} must change only {component}; changed {changed}")
        if row.get("changed_component") != component:
            raise ValueError(f"{variant_id} changed_component does not match its intervention")
    return deepcopy(protocol)


COMPONENT_VARIANTS = {
    "full_stack": "none",
    "without_model_view": "model_view",
    "fixed_view_uncertainty": "view_uncertainty",
    "fixed_smart_policy": "smart_adaptive_rule",
    "guard_disabled": "guard_blending",
    "risk_inputs_neutralized": "risk_input_block",
    "regime_input_neutralized": "regime_input",
    "anomaly_input_neutralized": "anomaly_input",
    "fixed_weight_bounds": "weight_bounds",
    "fixed_turnover_penalty": "turnover_penalty",
    "fixed_concentration_penalty": "concentration_penalty",
}

COMPONENT_OVERRIDE_KEYS = {
    "model_view",
    "view_confidence",
    "smart_all_controls",
    "guard",
    "risk_input_block",
    "regime_input",
    "anomaly_input",
    "weight_bounds",
    "turnover_penalty",
    "concentration_penalty",
}


def validate_component_ablation_execution_protocol(
    protocol: dict[str, Any],
    *,
    scientific_parent_sha256: str,
) -> dict[str, Any]:
    """Validate the component child matrix without changing the allocation-view scientific parent."""

    parent = protocol.get("scientific_parent", {})
    if parent.get("sha256") != scientific_parent_sha256:
        raise ValueError("component scientific-parent hash does not match the frozen allocation-view protocol")
    if parent.get("relationship") != "execution_layer_child_no_parent_scientific_change":
        raise ValueError("component must remain an execution-layer child of allocation-view")
    if protocol.get("reference_variant") != "full_stack":
        raise ValueError("component component-ablation reference must be full_stack")

    variants = protocol.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("component component-ablation protocol requires variants")
    by_id = {str(row.get("variant_id")): row for row in variants}
    if set(by_id) != set(COMPONENT_VARIANTS):
        raise ValueError("component variants do not match the frozen execution matrix")

    for variant_id, expected_component in COMPONENT_VARIANTS.items():
        row = by_id[variant_id]
        if row.get("changed_component") != expected_component:
            raise ValueError(f"{variant_id} changed_component is not frozen")
        override = row.get("override")
        if not isinstance(override, dict):
            raise ValueError(f"{variant_id} override must be an object")
        if variant_id == "full_stack":
            if override:
                raise ValueError("full_stack cannot override a component")
            continue
        if len(override) != 1:
            raise ValueError(f"{variant_id} must override exactly one component")
        if next(iter(override)) not in COMPONENT_OVERRIDE_KEYS:
            raise ValueError(f"{variant_id} uses an unknown override")

    parent_ids = {
        "full_stack",
        "without_model_view",
        "fixed_view_uncertainty",
        "fixed_smart_policy",
        "guard_disabled",
    }
    if {row["variant_id"] for row in variants if row["analysis_tier"] in {"reference", "parent_primary"}} != parent_ids:
        raise ValueError("component must inherit all five allocation-view parent rows")
    if protocol.get("scope_boundaries", {}).get("locked_full_framework_test") != "prohibited_in_development":
        raise ValueError("component cannot authorize the locked locked final evaluation test")
    return deepcopy(protocol)


def resolve_smart_component_intervention(
    reference: AllocationPolicyResult,
    *,
    override: dict[str, Any],
    base_max_weight: float,
    base_min_weight: float,
    base_turnover_penalty: float,
    base_concentration_penalty: float,
    n_assets: int,
    n_observations: int,
) -> AllocationPolicyResult:
    """Apply one component Smart subcomponent intervention to a resolved Smart-policy policy.

    The signal models and their as-of checks still execute.  An input-removal row
    only changes the named score after logging the reference inputs; controls are
    then recomputed with the unchanged Smart-policy equations.
    """

    if len(override) > 1:
        raise ValueError("a component row may override at most one Smart component")
    key = next(iter(override), None)
    if key in {None, "guard", "model_view", "view_confidence"}:
        return reference.model_copy(
            update={
                "optimizer_turnover_penalty": float(reference.turnover_penalty)
                * 252.0
                / max(float(n_observations), 30.0),
                "optimizer_concentration_penalty": float(reference.concentration_penalty)
                * 252.0
                / max(float(n_observations), 30.0),
            }
        )
    if key not in {
        "smart_all_controls",
        "risk_input_block",
        "regime_input",
        "anomaly_input",
        "weight_bounds",
        "turnover_penalty",
        "concentration_penalty",
    }:
        raise ValueError(f"unsupported component Smart-component override: {key}")

    effective = {
        "volatility_score": float(reference.volatility_score),
        "drawdown_score": float(reference.drawdown_score),
        "correlation_score": float(reference.correlation_score),
        "ml_score": float(reference.ml_score),
        "regime_score": float(reference.regime_score),
        "anomaly_score": float(reference.anomaly_score),
    }
    if key == "risk_input_block":
        for name in ("volatility_score", "drawdown_score", "correlation_score", "ml_score"):
            effective[name] = 0.0
    elif key == "regime_input":
        effective["regime_score"] = 0.0
    elif key == "anomaly_input":
        effective["anomaly_score"] = 0.0

    weights = reference.stress_component_weights
    stress_score = float(
        sum(float(weights[name.removesuffix("_score")]) * value for name, value in effective.items())
    )
    stress_score = float(np.clip(stress_score, 0.0, 1.0))
    feasible_cap = 1.0 / max(int(n_assets), 1)
    if n_assets == 1:
        max_weight = 1.0
    elif n_assets == 2:
        max_weight = float(
            np.clip(0.78 - 0.20 * stress_score - 0.05 * reference.concentration_score, feasible_cap, 0.85)
        )
    else:
        max_weight = float(
            np.clip(0.55 - 0.25 * stress_score - 0.05 * reference.concentration_score, feasible_cap, 1.0)
        )
    min_weight = float(
        np.clip(0.01 + 0.035 * stress_score, 0.0, min(0.20, 0.5 / max(int(n_assets), 1)))
    )
    turnover_penalty = float(
        np.clip(0.003 + 0.030 * stress_score + 0.010 * effective["drawdown_score"], 0.0, 0.05)
    )
    concentration_penalty = float(
        np.clip(0.004 + 0.035 * stress_score + 0.010 * effective["correlation_score"], 0.0, 0.05)
    )

    if key in {"smart_all_controls", "weight_bounds"}:
        max_weight = float(base_max_weight)
        min_weight = float(base_min_weight)
    if key in {"smart_all_controls", "turnover_penalty"}:
        turnover_penalty = float(base_turnover_penalty)
    if key in {"smart_all_controls", "concentration_penalty"}:
        concentration_penalty = float(base_concentration_penalty)

    boost = 252.0 / max(float(n_observations), 30.0)
    return reference.model_copy(
        update={
            **effective,
            "stress_score": round(stress_score, 8),
            "max_weight": round(max_weight, 4),
            "min_weight": round(min_weight, 4),
            "turnover_penalty": round(turnover_penalty, 4),
            "concentration_penalty": round(concentration_penalty, 4),
            "optimizer_turnover_penalty": round(turnover_penalty, 4) * boost,
            "optimizer_concentration_penalty": round(concentration_penalty, 4) * boost,
        }
    )
