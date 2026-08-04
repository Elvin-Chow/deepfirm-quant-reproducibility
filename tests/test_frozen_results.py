from __future__ import annotations

import json

import pandas as pd

from reproducibility.release import ROOT, csv_row_count, verify_frozen_results, verify_schemas


def test_expected_rows_schemas_and_material_values_match() -> None:
    verify_schemas()
    verify_frozen_results()
    expected = json.loads((ROOT / "results/expected_values.json").read_text())
    for relative, count in expected["row_counts"].items():
        assert csv_row_count(ROOT / relative) == count


def test_negative_and_null_result_boundaries_remain_visible() -> None:
    expected = json.loads((ROOT / "results/expected_values.json").read_text())
    warning_metrics = pd.read_csv(ROOT / "results/frozen/warning_metrics_overall.csv")
    legacy = pd.read_csv(ROOT / "results/frozen/legacy_allocation_impact.csv")
    warning_inference = pd.read_csv(ROOT / "results/frozen/warning_inference.csv")
    allocation_inference = pd.read_csv(ROOT / "results/frozen/allocation_inference.csv")

    accepted = warning_metrics[
        warning_metrics["model"].eq("accepted_v2_h1_frozen_artifact")
        & warning_metrics["repeat_id"].eq(1)
    ].iloc[0]
    assert accepted["recall_at_cost_threshold"] <= 0.08 + 1e-12
    assert accepted["recall_at_legacy_0_60"] == 0.0
    assert (~legacy["current_evidence_admissible"].astype(bool)).all()
    overall = warning_inference[
        warning_inference["slice_type"].eq("overall")
        & warning_inference["holm_adjusted_p_value"].notna()
    ]
    assert int((overall["holm_adjusted_p_value"] <= 0.05).sum()) == expected["claim_checks"]["overall_warning_holm_significant"]
    parent = allocation_inference[
        allocation_inference["analysis_tier"].eq("parent_primary")
        & allocation_inference["slice_type"].eq("overall")
        & allocation_inference["cost_scenario"].eq("base")
    ]
    assert int((parent["holm_adjusted_p_value"].dropna() <= 0.05).sum()) == expected["claim_checks"]["parent_component_holm_significant"]
