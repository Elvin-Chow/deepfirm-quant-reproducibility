"""Offline finance-semantic mutation checks for the Mathematics transfer.

This runner reads the frozen H1 metadata and protocol, but never scores a model,
opens H5 final test, calls a provider, or invokes a W25-family runner.
"""

from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter_ns
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "snapshot"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import (  # noqa: E402
    artifact_training_tickers,
    enforce_security_disjoint_if_required,
    load_holdout_portfolios,
)
from models.allocation_costs import simulate_cost_aware_path  # noqa: E402
from models.allocation_policy import AllocationPolicyEngine  # noqa: E402
from models.crisis_warning_engine import CrisisWarningEngine  # noqa: E402
from models.temporal_validation import purged_embargo_split  # noqa: E402

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "INJECTION_PROTOCOL.json"
OUT_DIR = HERE / "replay"
ARTIFACT_ROOT = ROOT / "artifacts/crisis_warning/v2"
PORTFOLIOS = ROOT / "experiments/portfolios/zero_overlap_supplement_portfolios.yaml"
COST_PROTOCOL = ROOT / "experiments/w22_cost_investability_protocol.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _label_overlap(train_dates: pd.DatetimeIndex, eval_dates: pd.DatetimeIndex,
                   all_dates: pd.DatetimeIndex, horizon: int) -> int:
    pos = {pd.Timestamp(day): i for i, day in enumerate(all_dates)}
    def realized(dates: pd.DatetimeIndex) -> set[int]:
        return {j for day in dates for j in range(pos[pd.Timestamp(day)] + 1,
                min(pos[pd.Timestamp(day)] + horizon + 1, len(all_dates)))}
    return len(realized(train_dates) & realized(eval_dates))


def _ledger_errors(path: object) -> list[str]:
    errors: list[str] = []
    for row in path.cost_ledger:
        component_sum = float(row["fee_cost"] + row["spread_cost"] + row["slippage_cost"])
        if not np.isclose(float(row["realized_cost"]), component_sum, rtol=0, atol=1e-12):
            errors.append(f"{row['scenario_id']}:component_sum")
        if int(row["rebalance_index"]) == 0:
            scenario = str(row["scenario_id"])
            actual = float(path.net_log_returns_by_scenario[scenario][0] - path.gross_log_returns[0])
            expected = float(np.log1p(-float(row["realized_cost"])))
            if not np.isclose(actual, expected, rtol=0, atol=1e-12):
                errors.append(f"{scenario}:gross_net_identity")
    return errors


def _case_security() -> tuple[bool, bool, str]:
    portfolio = load_holdout_portfolios(PORTFOLIOS)[0]
    config = {"security_disjoint_required": True}
    normal = enforce_security_disjoint_if_required(config, [portfolio], ARTIFACT_ROOT, [1])
    normal_alarm = not normal or not normal[0]["security_disjoint"]
    training = sorted(artifact_training_tickers(ARTIFACT_ROOT, [1])[1][portfolio.market])
    mutated = replace(portfolio, tickers=[training[0], *portfolio.tickers[1:]])
    try:
        enforce_security_disjoint_if_required(config, [mutated], ARTIFACT_ROOT, [1])
        detected = False
        detail = "overlap accepted"
    except ValueError as exc:
        detected = "overlap" in str(exc).lower()
        detail = f"rejected: {type(exc).__name__}"
    return normal_alarm, detected, detail


def _case_future_asof() -> tuple[bool, bool, str]:
    cutoff = "2026-01-07"
    normal = SimpleNamespace(diagnostics=SimpleNamespace(asof_date=cutoff))
    normal_alarm = AllocationPolicyEngine._validate_signal_asof("ml", normal, cutoff) != cutoff
    future = SimpleNamespace(diagnostics=SimpleNamespace(asof_date="2026-01-08"))
    try:
        AllocationPolicyEngine._validate_signal_asof("ml", future, cutoff)
        detected = False
    except ValueError as exc:
        detected = "exceeds train_asof" in str(exc)
    return normal_alarm, detected, "future-dated diagnostic vs fixed cutoff"


def _case_feature_schema() -> tuple[bool, bool, str]:
    expected = list(CrisisWarningEngine.feature_columns)
    CrisisWarningEngine.validate_feature_schema(expected, expected)
    mutated = expected.copy()
    mutated[0], mutated[1] = mutated[1], mutated[0]
    try:
        CrisisWarningEngine.validate_feature_schema(mutated, expected)
        detected = False
    except ValueError as exc:
        detected = "feature schema" in str(exc)
    return False, detected, "two names swapped; same 14 dimensions"


def _case_split() -> tuple[bool, bool, str]:
    dates = pd.bdate_range("2025-01-02", periods=120)
    frame = pd.DataFrame({"tail_event": np.tile([0, 0, 1, 0, 0], 24)}, index=dates)
    guarded = purged_embargo_split(frame, horizon=5, evaluation_fraction=0.2,
                                   min_train_rows=50, min_evaluation_rows=10)
    clean_overlap = _label_overlap(guarded.train.index, guarded.evaluation.index, dates, 5)
    raw_boundary = pd.Timestamp(guarded.audit["raw_boundary"])
    naive_overlap = _label_overlap(frame.loc[frame.index < raw_boundary].index,
                                   frame.loc[frame.index >= raw_boundary].index, dates, 5)
    return clean_overlap > 0, naive_overlap > 0, f"normal_shared={clean_overlap}; naive_shared={naive_overlap}"


def _case_cost_ledger() -> tuple[bool, bool, str]:
    protocol = json.loads(COST_PROTOCOL.read_text(encoding="utf-8"))
    returns = pd.DataFrame(np.tile([0.001, 0.0005, -0.0002, 0.0003], (21, 1)),
                           index=pd.bdate_range("2026-01-02", periods=21),
                           columns=list("ABCD"))
    normal = simulate_cost_aware_path(
        asset_log_returns=returns,
        target_weights_by_rebalance=[[0.4, 0.3, 0.2, 0.1]],
        initial_weights=[0.25, 0.25, 0.25, 0.25],
        tradable_by_rebalance=[[True] * 4], protocol=protocol)
    clean_errors = _ledger_errors(normal)
    mutated = replace(normal, cost_ledger=[dict(row) for row in normal.cost_ledger])
    row = next(x for x in mutated.cost_ledger if x["scenario_id"] == "base")
    row["realized_cost"] = float(row["realized_cost"]) + 0.0001
    errors = _ledger_errors(mutated)
    return bool(clean_errors), bool(errors), f"normal_errors={clean_errors}; mutation_errors={errors}"


CASES = {
    "security_role_overlap": _case_security,
    "future_signal_asof": _case_future_asof,
    "feature_schema_mismatch": _case_feature_schema,
    "split_crossing_label": _case_split,
    "trade_cost_ledger_mismatch": _case_cost_ledger,
}


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for relative, expected in protocol["baseline_files_sha256"].items():
        actual = _sha256(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"baseline hash changed: {relative}: {actual}")
    if [case["id"] for case in protocol["cases"]] != list(CASES):
        raise RuntimeError("declared cases differ from runner cases")
    results: list[dict[str, object]] = []
    started = perf_counter_ns()
    for item in protocol["cases"]:
        case_id = item["id"]
        begin = perf_counter_ns()
        try:
            normal_alarm, detected, detail = CASES[case_id]()
            error = ""
        except Exception as exc:
            normal_alarm, detected = True, False
            detail = "case could not complete"
            error = f"{type(exc).__name__}: {exc}"
        results.append({"case_id": case_id, "detector_origin": item["detector_origin"],
                        "normal_false_alarm": normal_alarm, "injected_detected": detected,
                        "injected_miss": not detected, "runtime_ms": round((perf_counter_ns()-begin)/1e6, 3),
                        "detail": detail, "error": error,
                        "pass": not normal_alarm and detected and not error})
    summary = {
        "protocol_sha256": _sha256(PROTOCOL),
        "runner_sha256": _sha256(Path(__file__)),
        "baseline_files_sha256": protocol["baseline_files_sha256"],
        "cases": results,
        "normal_false_alarms": sum(bool(x["normal_false_alarm"]) for x in results),
        "injected_misses": sum(bool(x["injected_miss"]) for x in results),
        "detected_injections": sum(bool(x["injected_detected"]) for x in results),
        "total_runtime_ms": round((perf_counter_ns()-started)/1e6, 3),
        "status": "PASS" if all(x["pass"] for x in results) else "FAIL",
        "scope": protocol["scope"], "claim_limit": protocol["claim_limit"],
        "h5_final_test_accessed": False, "market_data_fetched": False,
    }
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "RESULTS.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT_DIR / "RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
