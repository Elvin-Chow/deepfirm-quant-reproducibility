"""Point-in-time temporal split utilities for overlapping financial labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd


@dataclass(frozen=True)
class PurgedEmbargoSplit:
    """Chronological train/evaluation frames plus an auditable boundary record."""

    train: pd.DataFrame
    evaluation: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class FourWayPurgedEmbargoSplit:
    """Chronological train/calibration/validation/final-test partitions."""

    train: pd.DataFrame
    calibration: pd.DataFrame
    validation: pd.DataFrame
    final_test: pd.DataFrame
    audit: dict[str, Any]


def _date_label(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).date().isoformat()


def purged_embargo_split(
    frame: pd.DataFrame,
    *,
    horizon: int,
    evaluation_fraction: float,
    purge_observations: int | None = None,
    embargo_observations: int | None = None,
    group_columns: Sequence[str] = (),
    min_train_rows: int = 1,
    min_evaluation_rows: int = 1,
) -> PurgedEmbargoSplit:
    """Split by decision date and remove boundary-adjacent overlapping labels.

    A row dated ``t`` owns a forward label spanning ``t+1`` through ``t+h``.
    For every requested group, the last ``purge_observations`` decision rows
    before the raw boundary and the first ``embargo_observations`` rows on or
    after it are excluded.  Both controls default to at least the forecast
    horizon, which prevents an H5 label on the training side from sharing a
    realized return with the evaluation side.
    """

    if frame.empty:
        raise ValueError("temporal split frame is empty")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("temporal split requires a DatetimeIndex")
    if int(horizon) < 1:
        raise ValueError("horizon must be at least 1")
    if not 0.0 < float(evaluation_fraction) < 1.0:
        raise ValueError("evaluation_fraction must be between 0 and 1")

    missing_groups = [column for column in group_columns if column not in frame.columns]
    if missing_groups:
        raise ValueError("missing temporal split group columns: " + ", ".join(missing_groups))

    purge = max(int(horizon), int(purge_observations or horizon))
    embargo = max(int(horizon), int(embargo_observations or horizon))
    ordered = frame.sort_index(kind="mergesort").copy()
    decision_dates = pd.DatetimeIndex(ordered.index.unique()).sort_values()
    raw_boundary_position = int(len(decision_dates) * (1.0 - float(evaluation_fraction)))
    if raw_boundary_position <= 0 or raw_boundary_position >= len(decision_dates):
        raise ValueError("temporal split leaves an empty raw train or evaluation window")
    raw_boundary = pd.Timestamp(decision_dates[raw_boundary_position])

    train_mask = pd.Series(ordered.index < raw_boundary, index=ordered.index, dtype=bool)
    evaluation_mask = pd.Series(ordered.index >= raw_boundary, index=ordered.index, dtype=bool)
    group_audits: list[dict[str, Any]] = []

    if group_columns:
        grouper: str | list[str]
        grouper = group_columns[0] if len(group_columns) == 1 else list(group_columns)
        grouped = ordered.groupby(grouper, sort=True, dropna=False)
        groups = [(key, group.index) for key, group in grouped]
    else:
        groups = [("all", ordered.index)]

    for key, group_index in groups:
        group_dates = pd.DatetimeIndex(group_index.unique()).sort_values()
        before = group_dates[group_dates < raw_boundary]
        after = group_dates[group_dates >= raw_boundary]
        purge_dates = before[-purge:]
        embargo_dates = after[:embargo]

        key_values = key if isinstance(key, tuple) else (key,)
        if group_columns:
            group_label = {
                column: str(value)
                for column, value in zip(group_columns, key_values)
            }
            group_row_mask = pd.Series(True, index=ordered.index, dtype=bool)
            for column, value in zip(group_columns, key_values):
                group_row_mask &= ordered[column].eq(value).to_numpy()
        else:
            group_label = {"group": "all"}
            group_row_mask = pd.Series(True, index=ordered.index, dtype=bool)

        purge_row_mask = group_row_mask & ordered.index.isin(purge_dates)
        embargo_row_mask = group_row_mask & ordered.index.isin(embargo_dates)
        train_mask.loc[purge_row_mask.to_numpy()] = False
        evaluation_mask.loc[embargo_row_mask.to_numpy()] = False
        group_audits.append(
            {
                **group_label,
                "purged_rows": int(purge_row_mask.sum()),
                "embargoed_rows": int(embargo_row_mask.sum()),
                "purge_start": _date_label(purge_dates.min() if len(purge_dates) else None),
                "purge_end": _date_label(purge_dates.max() if len(purge_dates) else None),
                "embargo_start": _date_label(embargo_dates.min() if len(embargo_dates) else None),
                "embargo_end": _date_label(embargo_dates.max() if len(embargo_dates) else None),
            }
        )

    train = ordered.loc[train_mask.to_numpy()].copy()
    evaluation = ordered.loc[evaluation_mask.to_numpy()].copy()
    if len(train) < int(min_train_rows):
        raise ValueError(f"not enough training rows after purge: {len(train)}")
    if len(evaluation) < int(min_evaluation_rows):
        raise ValueError(f"not enough evaluation rows after embargo: {len(evaluation)}")
    if pd.Timestamp(train.index.max()) >= pd.Timestamp(evaluation.index.min()):
        raise ValueError("purged temporal split is not strictly chronological")

    raw_train_rows = int((ordered.index < raw_boundary).sum())
    raw_evaluation_rows = int((ordered.index >= raw_boundary).sum())
    audit = {
        "method": "chronological_purged_embargo",
        "label_horizon_observations": int(horizon),
        "evaluation_fraction": float(evaluation_fraction),
        "raw_boundary": _date_label(raw_boundary),
        "purge_observations_per_group": purge,
        "embargo_observations_per_group": embargo,
        "raw_train_rows": raw_train_rows,
        "raw_evaluation_rows": raw_evaluation_rows,
        "train_rows_after_purge": int(len(train)),
        "evaluation_rows_after_embargo": int(len(evaluation)),
        "purged_rows": int(raw_train_rows - len(train)),
        "embargoed_rows": int(raw_evaluation_rows - len(evaluation)),
        "train_end": _date_label(pd.Timestamp(train.index.max())),
        "evaluation_start": _date_label(pd.Timestamp(evaluation.index.min())),
        "group_columns": list(group_columns),
        "groups": group_audits,
    }
    return PurgedEmbargoSplit(train=train, evaluation=evaluation, audit=audit)


def four_way_purged_embargo_split(
    frame: pd.DataFrame,
    *,
    horizon: int,
    fractions: dict[str, float],
    purge_observations: int | None = None,
    embargo_observations: int | None = None,
    group_columns: Sequence[str] = (),
    min_rows: dict[str, int] | None = None,
) -> FourWayPurgedEmbargoSplit:
    """Create four untouched chronological roles with guarded boundaries.

    The splitter works from the latest role backwards.  It first removes the
    final-test suffix, then the validation suffix, and finally the calibration
    suffix.  Each boundary independently applies the same horizon-aware purge
    and embargo contract as :func:`purged_embargo_split`.
    """

    required_roles = ("train", "calibration", "validation", "final_test")
    missing_roles = [role for role in required_roles if role not in fractions]
    if missing_roles:
        raise ValueError("missing four-way split fractions: " + ", ".join(missing_roles))
    normalized_fractions = {role: float(fractions[role]) for role in required_roles}
    if any(value <= 0.0 for value in normalized_fractions.values()):
        raise ValueError("all four-way split fractions must be positive")
    if abs(sum(normalized_fractions.values()) - 1.0) > 1.0e-9:
        raise ValueError("four-way split fractions must sum to 1.0")

    minimums = {role: 1 for role in required_roles}
    if min_rows:
        for role, value in min_rows.items():
            if role not in minimums:
                raise ValueError(f"unknown four-way split role: {role}")
            minimums[role] = max(1, int(value))

    final_fraction = normalized_fractions["final_test"]
    final_boundary = purged_embargo_split(
        frame,
        horizon=horizon,
        evaluation_fraction=final_fraction,
        purge_observations=purge_observations,
        embargo_observations=embargo_observations,
        group_columns=group_columns,
        min_train_rows=1,
        min_evaluation_rows=minimums["final_test"],
    )

    development_fraction = 1.0 - final_fraction
    validation_relative_fraction = (
        normalized_fractions["validation"] / development_fraction
    )
    validation_boundary = purged_embargo_split(
        final_boundary.train,
        horizon=horizon,
        evaluation_fraction=validation_relative_fraction,
        purge_observations=purge_observations,
        embargo_observations=embargo_observations,
        group_columns=group_columns,
        min_train_rows=1,
        min_evaluation_rows=minimums["validation"],
    )

    train_calibration_fraction = (
        normalized_fractions["train"] + normalized_fractions["calibration"]
    )
    calibration_relative_fraction = (
        normalized_fractions["calibration"] / train_calibration_fraction
    )
    calibration_boundary = purged_embargo_split(
        validation_boundary.train,
        horizon=horizon,
        evaluation_fraction=calibration_relative_fraction,
        purge_observations=purge_observations,
        embargo_observations=embargo_observations,
        group_columns=group_columns,
        min_train_rows=minimums["train"],
        min_evaluation_rows=minimums["calibration"],
    )

    roles = {
        "train": calibration_boundary.train,
        "calibration": calibration_boundary.evaluation,
        "validation": validation_boundary.evaluation,
        "final_test": final_boundary.evaluation,
    }
    for role, role_frame in roles.items():
        if len(role_frame) < minimums[role]:
            raise ValueError(
                f"not enough {role} rows after purge/embargo: {len(role_frame)}"
            )
    ordered_roles = [roles[role] for role in required_roles]
    for earlier, later in zip(ordered_roles, ordered_roles[1:]):
        if pd.Timestamp(earlier.index.max()) >= pd.Timestamp(later.index.min()):
            raise ValueError("four-way purged split is not strictly chronological")

    role_audit = {
        role: {
            "requested_fraction": normalized_fractions[role],
            "rows": int(len(role_frame)),
            "positive_events": int(role_frame["tail_event"].sum())
            if "tail_event" in role_frame.columns
            else None,
            "start": _date_label(pd.Timestamp(role_frame.index.min())),
            "end": _date_label(pd.Timestamp(role_frame.index.max())),
        }
        for role, role_frame in roles.items()
    }
    audit = {
        "method": "nested_chronological_purged_embargo",
        "label_horizon_observations": int(horizon),
        "requested_fractions": normalized_fractions,
        "group_columns": list(group_columns),
        "roles": role_audit,
        "boundaries": {
            "train_to_calibration": calibration_boundary.audit,
            "calibration_to_validation": validation_boundary.audit,
            "validation_to_final_test": final_boundary.audit,
        },
    }
    return FourWayPurgedEmbargoSplit(
        train=roles["train"],
        calibration=roles["calibration"],
        validation=roles["validation"],
        final_test=roles["final_test"],
        audit=audit,
    )
