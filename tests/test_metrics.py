from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from reproducibility.metrics import deterministic_exact_k_indices, warning_metrics
from reproducibility.release import ROOT


def test_exact_k_is_deterministic_under_probability_ties() -> None:
    probabilities = [0.9, 0.8, 0.8, 0.8, 0.1]
    row_ids = ["z", "c", "a", "b", "y"]
    selected = deterministic_exact_k_indices(probabilities, row_ids, fraction=0.40)
    assert selected.tolist() == [0, 2]


def test_accepted_probability_metrics_match_archived_summary() -> None:
    predictions = pd.read_csv(ROOT / "results/frozen/warning_predictions.csv.gz")
    rows = predictions[
        predictions["model"].eq("accepted_v2_h1_frozen_artifact")
        & predictions["repeat_id"].eq(1)
    ]
    summary = pd.read_csv(ROOT / "results/frozen/warning_metrics_overall.csv")
    expected = summary[
        summary["model"].eq("accepted_v2_h1_frozen_artifact")
        & summary["repeat_id"].eq(1)
    ].iloc[0]
    assert np.isclose(
        average_precision_score(rows["tail_event"], rows["calibrated_probability"]),
        expected["average_precision"],
        rtol=0.0,
        atol=1e-12,
    )
    observed = warning_metrics(
        rows["tail_event"],
        rows["calibrated_probability"],
        rows["row_id"],
        threshold=0.10074626865671642,
    )
    assert np.isclose(observed["recall_at_threshold"], expected["recall_at_cost_threshold"], atol=1e-12)
    assert np.isclose(observed["top_decile_lift"], expected["top_decile_lift"], atol=1e-12)
