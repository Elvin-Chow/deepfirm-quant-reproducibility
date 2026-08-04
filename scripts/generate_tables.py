#!/usr/bin/env python3
"""Generate or check public tables from the frozen result files only."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FROZEN = ROOT / "results/frozen"
OUTPUT_DIR = ROOT / "results/generated"

MODEL_LABELS = {
    "accepted_v2_h1_frozen_artifact": "Accepted H1 artifact",
    "caviar_sav": "SAV-CAViaR",
    "garch_1_1": "GARCH(1,1)",
    "gradient_boosting": "Gradient boosting",
    "historical_es": "Historical ES",
    "historical_tail_threshold": "Historical tail threshold",
    "historical_var": "Historical VaR",
    "logistic_regression": "Logistic regression",
    "quantile_regression": "Quantile regression",
    "random_forest": "Random forest",
    "temporal_mlp": "Temporal MLP",
    "xgboost_retrained_comparator": "Retrained XGBoost",
}

STRATEGY_LABELS = {
    "equal_weight": "Equal weight",
    "inverse_volatility": "Inverse volatility",
    "mean_variance": "Mean variance",
    "prior_only_bl": "Prior-only BL",
    "smart_policy": "Smart policy",
    "smart_policy_oos_guard": "Smart policy + OOS guard",
}

COMPONENT_LABELS = {
    "without_model_view": "Without model view",
    "fixed_view_uncertainty": "Fixed view uncertainty",
    "fixed_smart_policy": "Fixed Smart policy",
    "guard_disabled": "Guard disabled",
    "risk_inputs_neutralized": "Risk inputs neutralized",
    "regime_input_neutralized": "Regime input neutralized",
    "anomaly_input_neutralized": "Anomaly input neutralized",
    "fixed_weight_bounds": "Fixed weight bounds",
    "fixed_turnover_penalty": "Fixed turnover penalty",
    "fixed_concentration_penalty": "Fixed concentration penalty",
}

COST_LABELS = {"low": "5 bp", "base": "12 bp", "high": "25 bp"}
COST_ORDER = {"low": 0, "base": 1, "high": 2}
NUMERIC_CHECK_ATOL = 1e-12


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read archived decimals with a platform-independent round-trip parser."""

    return pd.read_csv(path, float_precision="round_trip")


def derive_tables() -> dict[str, pd.DataFrame]:
    """Derive the four public tables without fitting, scoring, or networking."""

    warning_metrics = _read_csv(FROZEN / "warning_metrics_overall.csv")
    warning_inference = _read_csv(FROZEN / "warning_inference.csv")
    allocation_metrics = _read_csv(FROZEN / "allocation_metrics.csv")
    allocation_inference = _read_csv(FROZEN / "allocation_inference.csv")
    snapshots = _read_csv(FROZEN / "input_snapshot_identities.csv")
    deviations = _read_csv(FROZEN / "protocol_deviations.csv")
    warning_predictions = _read_csv(FROZEN / "warning_predictions.csv.gz")
    allocation_returns = _read_csv(FROZEN / "allocation_returns.csv.gz")
    failures = _read_csv(FROZEN / "model_failures.csv")
    skips = _read_csv(FROZEN / "skips.csv")

    warning_columns = [
        "roc_auc",
        "average_precision",
        "brier_score",
        "recall_at_cost_threshold",
        "flag_rate_at_cost_threshold",
        "recall_at_legacy_0_60",
        "top_decile_lift",
    ]
    warning = (
        warning_metrics.groupby("model", sort=True)
        .agg(
            repeats=("repeat_id", "count"),
            **{column: (column, "mean") for column in warning_columns},
        )
        .reset_index()
    )
    warning.insert(1, "model_label", warning["model"].map(MODEL_LABELS))
    _require(len(warning) == 12, "warning summary must contain 12 model identities")
    _require(warning["model_label"].notna().all(), "warning display label is missing")

    strategies = allocation_metrics[
        allocation_metrics["analysis_type"].eq("strategy")
    ].copy()
    strategies.insert(2, "strategy_label", strategies["analysis_id"].map(STRATEGY_LABELS))
    strategies.insert(4, "cost_label", strategies["cost_scenario"].map(COST_LABELS))
    strategies["cost_order"] = strategies["cost_scenario"].map(COST_ORDER)
    strategies = strategies.sort_values(["analysis_id", "cost_order"], kind="mergesort")
    strategies = strategies[
        [
            "analysis_id",
            "strategy_label",
            "cost_scenario",
            "cost_label",
            "portfolio_count",
            "row_count",
            "net_mean_log_return",
            "net_information_ratio",
            "mean_one_way_turnover",
            "realized_cost",
            "net_cumulative_return",
            "benchmark_cumulative_return",
            "net_benchmark_excess_return",
        ]
    ]
    _require(len(strategies) == 18, "strategy-cost summary must contain 18 rows")
    _require(strategies["strategy_label"].notna().all(), "strategy display label is missing")

    component = allocation_inference[
        allocation_inference["slice_type"].eq("overall")
    ].copy()
    component.insert(1, "variant_label", component["model_or_variant"].map(COMPONENT_LABELS))
    component = component[
        [
            "model_or_variant",
            "variant_label",
            "analysis_tier",
            "metric",
            "estimate",
            "ci_low",
            "ci_high",
            "p_value",
            "holm_adjusted_p_value",
            "bh_adjusted_p_value",
            "status",
        ]
    ]
    _require(len(component) == 20, "overall component inference must contain 20 rows")
    _require(component["variant_label"].notna().all(), "component display label is missing")

    deviation_count = int(len(deviations))
    fatal_count = int(deviations["fatal"].astype(bool).sum())
    coverage = pd.DataFrame(
        [
            ("Input snapshot identities", len(snapshots), "10 portfolios; hashes and coverage metadata only"),
            ("Warning prediction rows", len(warning_predictions), "accepted H1 plus 11 comparators"),
            ("Warning overall primary comparisons", len(warning_inference.query("slice_type == 'overall' and status == 'estimable'")), "11 comparators x 3 metrics"),
            ("Allocation strategy-cost rows", len(strategies), "6 strategies x 3 costs"),
            ("Allocation component-cost rows", len(allocation_metrics.query("analysis_type == 'component'")), "11 variants x 3 costs"),
            ("Allocation return rows", len(allocation_returns), "gross and net paths retained"),
            ("Allocation parent-primary comparisons", len(component.query("analysis_tier == 'parent_primary' and status == 'estimable'")), "4 variants x 2 metrics"),
            ("Model failures / evaluation skips", len(failures) + len(skips), f"{len(failures)} / {len(skips)}"),
            ("Allowed / fatal deviations", deviation_count, f"{deviation_count - fatal_count} / {fatal_count}"),
        ],
        columns=["item", "count", "scope"],
    )

    return {
        "warning_model_summary.csv": warning,
        "allocation_strategy_cost_summary.csv": strategies,
        "allocation_component_inference.csv": component,
        "coverage_summary.csv": coverage,
    }


def csv_bytes(frame: pd.DataFrame) -> bytes:
    handle = io.StringIO(newline="")
    frame.to_csv(handle, index=False, lineterminator="\n")
    return handle.getvalue().encode("utf-8")


def frames_semantically_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> bool:
    """Compare generated tables exactly except for sub-tolerance float parsing noise."""

    if list(actual.columns) != list(expected.columns):
        return False
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=NUMERIC_CHECK_ATOL,
            check_like=False,
        )
    except AssertionError:
        return False
    return True


def write_or_check(*, check: bool) -> None:
    tables = derive_tables()
    if not check:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stale = []
    for filename, frame in tables.items():
        path = OUTPUT_DIR / filename
        expected = csv_bytes(frame)
        if check:
            if not path.is_file():
                stale.append(path.relative_to(ROOT).as_posix())
                continue
            if path.read_bytes() == expected:
                continue
            actual_frame = _read_csv(path)
            if not frames_semantically_equal(actual_frame, frame):
                stale.append(path.relative_to(ROOT).as_posix())
        else:
            path.write_bytes(expected)
            print(path.relative_to(ROOT))
    if stale:
        raise RuntimeError("generated table is missing or stale: " + ", ".join(stale))
    if check:
        print("generated tables are current")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    write_or_check(check=args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
