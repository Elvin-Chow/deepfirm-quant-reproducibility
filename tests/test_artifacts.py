from __future__ import annotations

import json
from pathlib import Path

from models.crisis_warning_artifact_hash import compute_artifact_hash, sha256_file
from reproducibility.release import BUNDLES, ROOT, verify_artifacts


def test_all_released_bundle_hashes_and_feature_schemas_match() -> None:
    verify_artifacts()
    for relative in BUNDLES:
        directory = ROOT / relative
        metadata = json.loads((directory / "training_metadata.json").read_text(encoding="utf-8"))
        bundle_hash, files = compute_artifact_hash(directory)
        assert bundle_hash == metadata["artifact_hash"]
        assert files == metadata["artifact_hash_files"]
        assert sha256_file(directory / "feature_schema.json") == metadata["feature_schema_hash"]


def test_accepted_failed_and_legacy_roles_are_explicit() -> None:
    accepted = json.loads(
        (ROOT / "artifacts/crisis_warning/accepted/global_h1/training_metadata.json").read_text()
    )
    failed = json.loads(
        (ROOT / "artifacts/crisis_warning/failed/global_h5/training_metadata.json").read_text()
    )
    legacy = [
        json.loads(path.read_text())
        for path in sorted((ROOT / "artifacts/crisis_warning/legacy").glob("*/training_metadata.json"))
    ]

    assert accepted["acceptance_gate"]["passed"] is True
    assert accepted["final_test_status"] == "reported_after_validation_acceptance"
    assert failed["acceptance_gate"]["passed"] is False
    assert failed["final_test_status"] == "not_opened_validation_gate_failed"
    assert failed["final_test_metrics"] is None
    assert all(item["validation_status"] == "degraded_validation" for item in legacy)
