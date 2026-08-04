"""Read-only verification for the public artifact release."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from models.crisis_warning_artifact_hash import compute_artifact_hash, sha256_file


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = (
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "DATA.md",
    "requirements-lock.txt",
    "docs/reproduction.md",
    "configs/training.json",
    "configs/threshold.json",
    "configs/allocation.yaml",
    "configs/components.json",
    "configs/statistics.json",
    "configs/final_evaluation.yaml",
    "artifacts/crisis_warning/accepted/global_h1/training_metadata.json",
    "artifacts/crisis_warning/failed/global_h5/training_metadata.json",
    "artifacts/crisis_warning/legacy/global_h1/training_metadata.json",
    "artifacts/crisis_warning/legacy/global_h5/training_metadata.json",
    "results/expected_values.json",
    "results/manifest.sha256",
    "results/frozen/warning_predictions.csv.gz",
    "results/frozen/allocation_returns.csv.gz",
    "results/frozen/allocation_decisions.csv.gz",
    "results/frozen/cost_ledger.csv.gz",
    "results/generated/warning_model_summary.csv",
    "results/generated/allocation_strategy_cost_summary.csv",
    "results/generated/allocation_component_inference.csv",
    "results/generated/coverage_summary.csv",
    "results/figures/primary_results_overview.pdf",
    "results/figures/primary_results_overview.png",
    "scripts/run_reproducibility_smoke.sh",
)

BUNDLES = (
    "artifacts/crisis_warning/accepted/global_h1",
    "artifacts/crisis_warning/failed/global_h5",
    "artifacts/crisis_warning/legacy/global_h1",
    "artifacts/crisis_warning/legacy/global_h5",
)


def csv_row_count(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def csv_columns(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))


def verify_required_paths() -> None:
    missing = [relative for relative in REQUIRED_PATHS if not (ROOT / relative).is_file()]
    if missing:
        raise AssertionError("missing release paths: " + ", ".join(missing))


def calculated_manifest() -> str:
    rows = []
    for path in sorted((ROOT / "results").rglob("*")):
        if not path.is_file() or path.name == "manifest.sha256":
            continue
        relative = path.relative_to(ROOT).as_posix()
        rows.append(f"{sha256_file(path)}  {relative}")
    return "\n".join(rows) + "\n"


def verify_manifest() -> None:
    actual = (ROOT / "results/manifest.sha256").read_text(encoding="utf-8")
    expected = calculated_manifest()
    if actual != expected:
        raise AssertionError("results/manifest.sha256 does not match released result files")


def verify_configs() -> None:
    for relative in ("training.json", "threshold.json", "components.json", "statistics.json"):
        value = json.loads((ROOT / "configs" / relative).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("status") != "frozen":
            raise AssertionError(f"configs/{relative} is not a frozen object")
    for relative in ("allocation.yaml", "final_evaluation.yaml", "portfolios.yaml"):
        value = yaml.safe_load((ROOT / "configs" / relative).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise AssertionError(f"configs/{relative} is not an object")
    final = yaml.safe_load((ROOT / "configs/final_evaluation.yaml").read_text())
    if final["status"] != "completed_frozen_do_not_repeat":
        raise AssertionError("final evaluation repeat boundary changed")
    if final["execution_boundary"]["final_evaluation_repeat_entry_in_release"]:
        raise AssertionError("public config must not authorize a final-evaluation repeat")


def verify_artifacts() -> None:
    for relative in BUNDLES:
        directory = ROOT / relative
        metadata = json.loads((directory / "training_metadata.json").read_text())
        actual_hash, actual_files = compute_artifact_hash(directory)
        if actual_hash != metadata.get("artifact_hash"):
            raise AssertionError(f"artifact bundle hash mismatch: {relative}")
        expected_files = metadata.get("artifact_hash_files")
        if expected_files != actual_files:
            raise AssertionError(f"artifact file identity mismatch: {relative}")
        schema_hash = sha256_file(directory / "feature_schema.json")
        if schema_hash != metadata.get("feature_schema_hash"):
            raise AssertionError(f"feature schema hash mismatch: {relative}")


def _assert_close(actual: float, expected: float, tolerance: float) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
        raise AssertionError(f"expected {expected!r}, observed {actual!r}")


def verify_frozen_results() -> None:
    expected = json.loads((ROOT / "results/expected_values.json").read_text())
    tolerance = float(expected["numeric_absolute_tolerance"])
    for relative, count in expected["row_counts"].items():
        observed = csv_row_count(ROOT / relative)
        if observed != int(count):
            raise AssertionError(f"row count mismatch for {relative}: {observed} != {count}")

    warning = pd.read_csv(ROOT / "results/frozen/warning_metrics_overall.csv")
    accepted = warning[
        warning["model"].eq("accepted_v2_h1_frozen_artifact")
        & warning["repeat_id"].eq(1)
        & warning["slice_type"].eq("overall")
    ].iloc[0]
    for field, value in expected["accepted_h1_primary"].items():
        _assert_close(float(accepted[field]), float(value), tolerance)

    gradient = warning[warning["model"].eq("gradient_boosting")]
    if len(gradient) != int(expected["gradient_boosting_five_seed_mean"]["repeat_count"]):
        raise AssertionError("gradient boosting repeat count changed")
    for field in ("average_precision", "recall_at_cost_threshold"):
        _assert_close(
            float(gradient[field].mean()),
            float(expected["gradient_boosting_five_seed_mean"][field]),
            tolerance,
        )

    allocation = pd.read_csv(ROOT / "results/frozen/allocation_metrics.csv")
    guarded = allocation[
        allocation["analysis_type"].eq("strategy")
        & allocation["analysis_id"].eq("smart_policy_oos_guard")
        & allocation["cost_scenario"].eq("base")
    ].iloc[0]
    for field, value in expected["guarded_smart_base_cost"].items():
        _assert_close(float(guarded[field]), float(value), tolerance)

    outcomes = pd.read_csv(ROOT / "results/frozen/artifact_outcomes.csv")
    failed_h5 = outcomes[(outcomes["version"].eq("v2")) & outcomes["horizon"].eq(5)].iloc[0]
    _assert_close(float(failed_h5["validation_roc_auc"]), float(expected["failed_h5"]["validation_roc_auc"]), tolerance)
    if str(failed_h5["final_test_status"]) != expected["failed_h5"]["final_test_status"]:
        raise AssertionError("failed H5 final-test boundary changed")

    deviations = pd.read_csv(ROOT / "results/frozen/deviation_summary.csv")
    if int(deviations["rows"].sum()) != int(expected["deviations"]["total"]):
        raise AssertionError("deviation total changed")
    if int(deviations.loc[deviations["fatal"].astype(str).str.lower().eq("true"), "rows"].sum()) != 0:
        raise AssertionError("fatal deviations are present")

    warning_inference = pd.read_csv(ROOT / "results/frozen/warning_inference.csv")
    warning_overall = warning_inference[
        warning_inference["slice_type"].eq("overall")
        & warning_inference["holm_adjusted_p_value"].notna()
    ]
    if (warning_overall["holm_adjusted_p_value"] <= 0.05).any():
        raise AssertionError("an overall primary warning comparison became Holm-significant")

    allocation_inference = pd.read_csv(ROOT / "results/frozen/allocation_inference.csv")
    parent = allocation_inference[
        allocation_inference["analysis_tier"].eq("parent_primary")
        & allocation_inference["slice_type"].eq("overall")
        & allocation_inference["cost_scenario"].eq("base")
    ]
    if (parent["holm_adjusted_p_value"].dropna() <= 0.05).any():
        raise AssertionError("a parent component comparison became Holm-significant")


def verify_schemas() -> None:
    required = {
        "results/frozen/warning_predictions.csv.gz": {"row_id", "date", "tail_event", "model", "calibrated_probability"},
        "results/frozen/allocation_returns.csv.gz": {"date", "portfolio_name", "analysis_type", "analysis_id", "gross_log_return", "net_log_return"},
        "results/frozen/allocation_decisions.csv.gz": {"information_cutoff", "holding_start", "holding_end", "trigger_rule", "post_guard_weights"},
        "results/frozen/cost_ledger.csv.gz": {"rebalance_date", "one_way_turnover", "realized_cost", "executed_weights"},
    }
    for relative, columns in required.items():
        observed = set(csv_columns(ROOT / relative))
        missing = columns - observed
        if missing:
            raise AssertionError(f"schema mismatch for {relative}: missing {sorted(missing)}")


def verify_release() -> list[str]:
    checks = (
        ("required paths", verify_required_paths),
        ("configuration boundary", verify_configs),
        ("artifact hashes and schemas", verify_artifacts),
        ("frozen result schemas", verify_schemas),
        ("frozen expected values", verify_frozen_results),
        ("result manifest", verify_manifest),
    )
    completed = []
    for label, check in checks:
        check()
        completed.append(label)
    return completed
