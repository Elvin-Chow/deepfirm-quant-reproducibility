#!/usr/bin/env python3
"""Generate or check the primary figure from frozen public results only."""

from __future__ import annotations

import argparse
import tempfile
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_tables import COMPONENT_LABELS, MODEL_LABELS, STRATEGY_LABELS


OUTPUT_DIR = ROOT / "results/figures"
PDF_PATH = OUTPUT_DIR / "primary_results_overview.pdf"
PNG_PATH = OUTPUT_DIR / "primary_results_overview.png"
FROZEN = ROOT / "results/frozen"

WARNING_ORDER = [
    "xgboost_retrained_comparator",
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "historical_tail_threshold",
    "historical_var",
    "historical_es",
    "garch_1_1",
    "caviar_sav",
    "quantile_regression",
    "temporal_mlp",
]
STRATEGY_ORDER = [
    "equal_weight",
    "inverse_volatility",
    "mean_variance",
    "prior_only_bl",
    "smart_policy",
    "smart_policy_oos_guard",
]
PARENT_ORDER = [
    "without_model_view",
    "fixed_view_uncertainty",
    "fixed_smart_policy",
    "guard_disabled",
]
SHORT_LABELS = {
    "Historical tail threshold": "Historical tail rule",
    "Smart policy + OOS guard": "Smart + OOS guard",
}

BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#7A7A7A"
LIGHT_GRAY = "#D7D7D7"
DARK = "#222222"


def _ordered(frame, key: str, order: list[str]):
    indexed = frame.set_index(key)
    missing = [item for item in order if item not in indexed.index]
    if missing:
        raise RuntimeError(f"missing frozen figure rows for {key}: {missing}")
    return indexed.loc[order].reset_index()


def _forest_plot(ax, frame, *, scale: float, color: str) -> None:
    estimate = frame["estimate"].to_numpy(dtype=float) * scale
    low = frame["ci_low"].to_numpy(dtype=float) * scale
    high = frame["ci_high"].to_numpy(dtype=float) * scale
    y = np.arange(len(frame), dtype=float)
    ax.errorbar(
        estimate,
        y,
        xerr=np.vstack((estimate - low, high - estimate)),
        fmt="o",
        markersize=4.4,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.2,
        ecolor=color,
        elinewidth=1.0,
        capsize=2.0,
        capthick=0.9,
        zorder=3,
    )
    ax.axvline(0.0, color=DARK, linewidth=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT_LABELS.get(label, label) for label in frame["display_label"]])
    ax.invert_yaxis()
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)


def build_figure():
    import pandas as pd

    warning = pd.read_csv(FROZEN / "warning_inference.csv")
    warning = warning[
        warning["slice_type"].eq("overall")
        & warning["metric"].eq("average_precision")
    ].copy()
    warning = _ordered(warning, "model_or_variant", WARNING_ORDER)
    warning["display_label"] = warning["model_or_variant"].map(MODEL_LABELS)

    strategies = pd.read_csv(FROZEN / "allocation_metrics.csv")
    strategies = strategies[
        strategies["analysis_type"].eq("strategy")
        & strategies["cost_scenario"].eq("base")
    ].copy()
    strategies = _ordered(strategies, "analysis_id", STRATEGY_ORDER)
    strategies["display_label"] = strategies["analysis_id"].map(STRATEGY_LABELS)

    components = pd.read_csv(FROZEN / "allocation_inference.csv")
    components = components[
        components["slice_type"].eq("overall")
        & components["analysis_tier"].eq("parent_primary")
        & components["metric"].eq("net_mean_log_return")
        & components["cost_scenario"].eq("base")
    ].copy()
    components = _ordered(components, "model_or_variant", PARENT_ORDER)
    components["display_label"] = components["model_or_variant"].map(COMPONENT_LABELS)

    if len(warning) != 11 or not warning["status"].eq("estimable").all():
        raise RuntimeError("warning figure family is incomplete")
    if not (warning["holm_adjusted_p_value"] > 0.05).all():
        raise RuntimeError("a primary warning comparison became Holm-significant")
    if len(strategies) != 6:
        raise RuntimeError("allocation figure family is incomplete")
    if len(components) != 4 or not components["status"].eq("estimable").all():
        raise RuntimeError("parent-component figure family is incomplete")
    if not (components["holm_adjusted_p_value"] > 0.05).all():
        raise RuntimeError("a parent-component comparison became Holm-significant")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.15, 3.35),
        gridspec_kw={"width_ratios": [1.34, 1.03, 1.13], "wspace": 0.72},
    )

    ax = axes[0]
    _forest_plot(ax, warning, scale=1.0, color=BLUE)
    ax.set_title(r"(a) Warning: $\Delta$AP vs accepted H1", loc="left", pad=5)
    ax.set_xlabel(r"Comparator minus accepted H1 ($\Delta$AP)")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))

    ax = axes[1]
    values = strategies["net_mean_log_return"].to_numpy(dtype=float) * 1_000.0
    y = np.arange(len(strategies), dtype=float)
    colors = [GRAY, GRAY, GRAY, GRAY, ORANGE, BLUE]
    markers = ["o", "o", "o", "o", "s", "D"]
    for yi, value, color, marker in zip(y, values, colors, markers):
        ax.hlines(yi, 0.0, value, color=LIGHT_GRAY, linewidth=0.8, zorder=1)
        ax.plot(value, yi, marker=marker, markersize=4.6, color=color, linestyle="none", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT_LABELS.get(label, label) for label in strategies["display_label"]])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.22)
    ax.xaxis.set_major_locator(MultipleLocator(0.4))
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_title("(b) Allocation at 12 bp", loc="left", pad=5)
    ax.set_xlabel(r"Net mean log return ($\times 10^{-3}$)")

    ax = axes[2]
    _forest_plot(ax, components, scale=10_000.0, color=ORANGE)
    ax.set_title("(c) Parent-component deltas", loc="left", pad=5)
    ax.set_xlabel(r"$\Delta$ net mean log return ($\times 10^{-4}$)")
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    for ax in axes:
        ax.tick_params(axis="both", which="major", pad=2)
    return figure


def render_outputs(pdf_path: Path, png_path: Path) -> None:
    figure = build_figure()
    fixed_date = datetime(2026, 8, 4, tzinfo=timezone.utc)
    try:
        figure.savefig(
            pdf_path,
            bbox_inches="tight",
            pad_inches=0.03,
            metadata={
                "Title": "Primary full-framework result overview",
                "Author": "DeepFirm Quant public figure generator",
                "Subject": "Presentation of frozen security-disjoint and out-of-time results",
                "Keywords": "warning, allocation, component analysis, frozen results",
                "CreationDate": fixed_date,
                "ModDate": fixed_date,
            },
        )
        figure.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
            metadata={"Title": "Primary full-framework result overview", "Software": "matplotlib"},
        )
    finally:
        plt.close(figure)


def write_or_check(*, check: bool) -> None:
    if check:
        with tempfile.TemporaryDirectory(prefix="deepfirm-figure-check-") as directory:
            root = Path(directory)
            pdf = root / PDF_PATH.name
            png = root / PNG_PATH.name
            render_outputs(pdf, png)
            stale = [
                path.relative_to(ROOT).as_posix()
                for path, candidate in ((PDF_PATH, pdf), (PNG_PATH, png))
                if not path.is_file() or path.read_bytes() != candidate.read_bytes()
            ]
        if stale:
            raise RuntimeError("generated figure is missing or stale: " + ", ".join(stale))
        print("generated figures are current")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render_outputs(PDF_PATH, PNG_PATH)
    print(PDF_PATH.relative_to(ROOT))
    print(PNG_PATH.relative_to(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    write_or_check(check=args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
