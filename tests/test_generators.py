from __future__ import annotations

import matplotlib.pyplot as plt

from scripts.generate_figures import build_figure, write_or_check as check_figures
from scripts.generate_tables import derive_tables, write_or_check as check_tables


def test_public_table_families_are_complete() -> None:
    tables = derive_tables()
    assert len(tables["warning_model_summary.csv"]) == 12
    assert len(tables["allocation_strategy_cost_summary.csv"]) == 18
    assert len(tables["allocation_component_inference.csv"]) == 20
    assert len(tables["coverage_summary.csv"]) == 9


def test_primary_figure_contains_the_three_frozen_panels() -> None:
    figure = build_figure()
    try:
        assert len(figure.axes) == 3
        assert len(figure.axes[0].get_yticklabels()) == 11
        assert len(figure.axes[1].get_yticklabels()) == 6
        assert len(figure.axes[2].get_yticklabels()) == 4
    finally:
        plt.close(figure)


def test_tracked_tables_and_figures_are_current() -> None:
    check_tables(check=True)
    check_figures(check=True)
