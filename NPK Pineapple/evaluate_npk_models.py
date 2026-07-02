"""
Generate research-paper evaluation graphs for pineapple NPK/fertigation models.

The script reads the artifacts produced by model/train.py:
  - model/model_metrics.csv
  - model/model_report.json
  - model/sample_predictions.csv
  - model/feature_importance.csv

It does not retrain the models. It visualizes the saved validation/test results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "model"
DEFAULT_OUTPUT = ROOT / "evaluation_results"
TARGET_LABELS = {
    "delta_N": "Nitrogen adjustment",
    "delta_P": "Phosphorus adjustment",
    "delta_K": "Potassium adjustment",
    "irrigation_ml": "Irrigation volume",
    "pH_adj": "pH adjustment",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create NPK model evaluation plots from saved CSV/JSON artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Folder containing model artifacts.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Folder where plots and reports are saved.")
    parser.add_argument("--top-features", type=int, default=10, help="Top feature importances to show per target.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved PNG figures.")
    return parser.parse_args()


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "savefig.bbox": "tight",
        }
    )


def savefig(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | float | int | None) -> float:
    if value in (None, ""):
        return math.nan
    return float(value)


def target_label(target: str, unit: str | None = None) -> str:
    label = TARGET_LABELS.get(target, target)
    return f"{label} ({unit})" if unit else label


def metric_rows_to_arrays(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    numeric_columns = ["train_mae", "test_mae", "test_rmse", "train_r2", "test_r2", "test_mean", "test_min", "test_max"]
    data: dict[str, np.ndarray] = {"targets": np.array([row["target"] for row in rows], dtype=object)}
    for column in numeric_columns:
        data[column] = np.array([to_float(row.get(column)) for row in rows], dtype=float)
    return data


def plot_mae_rmse(rows: list[dict[str, str]], plot_dir: Path, dpi: int) -> None:
    if not rows:
        return
    x = np.arange(len(rows))
    labels = [row["target"] for row in rows]
    train_mae = np.array([to_float(row["train_mae"]) for row in rows])
    test_mae = np.array([to_float(row["test_mae"]) for row in rows])
    test_rmse = np.array([to_float(row["test_rmse"]) for row in rows])

    plt.figure(figsize=(8.5, 4.8))
    width = 0.26
    plt.bar(x - width, train_mae, width, label="Train MAE", color="#2f80ed")
    plt.bar(x, test_mae, width, label="Test MAE", color="#27ae60")
    plt.bar(x + width, test_rmse, width, label="Test RMSE", color="#f2994a")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Error")
    plt.title("Error Metrics Per NPK/Fertigation Target")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    savefig(plot_dir / "mae_rmse_per_target.png", dpi)


def plot_r2(rows: list[dict[str, str]], plot_dir: Path, dpi: int) -> None:
    if not rows:
        return
    x = np.arange(len(rows))
    labels = [row["target"] for row in rows]
    train_r2 = np.array([to_float(row["train_r2"]) for row in rows])
    test_r2 = np.array([to_float(row["test_r2"]) for row in rows])

    plt.figure(figsize=(8.5, 4.8))
    width = 0.34
    plt.bar(x - width / 2, train_r2, width, label="Train R2", color="#2f80ed")
    plt.bar(x + width / 2, test_r2, width, label="Test R2", color="#9b51e0")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylim(min(0, np.nanmin(test_r2) - 0.05), 1.05)
    plt.ylabel("R2 score")
    plt.title("Explained Variance Per Target")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    savefig(plot_dir / "r2_per_target.png", dpi)


def plot_error_relative_to_range(rows: list[dict[str, str]], plot_dir: Path, dpi: int) -> None:
    if not rows:
        return
    labels = [row["target"] for row in rows]
    test_mae = np.array([to_float(row["test_mae"]) for row in rows])
    test_rmse = np.array([to_float(row["test_rmse"]) for row in rows])
    target_range = np.array([to_float(row["test_max"]) - to_float(row["test_min"]) for row in rows])
    mae_pct = np.divide(test_mae, target_range, out=np.zeros_like(test_mae), where=target_range != 0) * 100
    rmse_pct = np.divide(test_rmse, target_range, out=np.zeros_like(test_rmse), where=target_range != 0) * 100

    x = np.arange(len(rows))
    width = 0.34
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(x - width / 2, mae_pct, width, label="MAE / target range", color="#27ae60")
    plt.bar(x + width / 2, rmse_pct, width, label="RMSE / target range", color="#eb5757")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Percent of target range")
    plt.title("Normalized Error Relative to Test-Set Target Range")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    savefig(plot_dir / "normalized_error_percent_range.png", dpi)


def prediction_arrays(sample_rows: list[dict[str, str]], target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual = np.array([to_float(row.get(f"{target}_actual")) for row in sample_rows], dtype=float)
    predicted = np.array([to_float(row.get(f"{target}_predicted")) for row in sample_rows], dtype=float)
    error = np.array([to_float(row.get(f"{target}_error")) for row in sample_rows], dtype=float)
    valid = ~(np.isnan(actual) | np.isnan(predicted) | np.isnan(error))
    return actual[valid], predicted[valid], error[valid]


def plot_actual_vs_predicted(sample_rows: list[dict[str, str]], metric_rows: list[dict[str, str]], plot_dir: Path, dpi: int) -> None:
    if not sample_rows or not metric_rows:
        return
    for row in metric_rows:
        target = row["target"]
        unit = row.get("unit", "")
        actual, predicted, _ = prediction_arrays(sample_rows, target)
        if actual.size == 0:
            continue

        low = min(float(actual.min()), float(predicted.min()))
        high = max(float(actual.max()), float(predicted.max()))
        pad = (high - low) * 0.05 or 1.0

        plt.figure(figsize=(5.5, 5.2))
        plt.scatter(actual, predicted, s=28, alpha=0.75, color="#2f80ed", edgecolors="none")
        plt.plot([low - pad, high + pad], [low - pad, high + pad], color="#eb5757", linewidth=2, label="Ideal")
        plt.xlabel(f"Actual {unit}".strip())
        plt.ylabel(f"Predicted {unit}".strip())
        plt.title(f"Actual vs Predicted: {target_label(target)}")
        plt.grid(alpha=0.25)
        plt.legend()
        savefig(plot_dir / f"actual_vs_predicted_{target}.png", dpi)


def plot_residual_histograms(sample_rows: list[dict[str, str]], metric_rows: list[dict[str, str]], plot_dir: Path, dpi: int) -> None:
    if not sample_rows or not metric_rows:
        return
    for row in metric_rows:
        target = row["target"]
        _, _, error = prediction_arrays(sample_rows, target)
        if error.size == 0:
            continue

        plt.figure(figsize=(6.5, 4.5))
        plt.hist(error, bins=min(16, max(6, int(np.sqrt(error.size)))), color="#56cc9d", edgecolor="#222222", alpha=0.85)
        plt.axvline(0, color="#eb5757", linewidth=2, label="Zero error")
        plt.axvline(float(np.mean(error)), color="#2f80ed", linewidth=2, linestyle="--", label="Mean error")
        plt.xlabel("Prediction error")
        plt.ylabel("Count")
        plt.title(f"Residual Distribution: {target_label(target)}")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        savefig(plot_dir / f"residual_histogram_{target}.png", dpi)


def plot_prediction_series(sample_rows: list[dict[str, str]], metric_rows: list[dict[str, str]], plot_dir: Path, dpi: int) -> None:
    if not sample_rows or not metric_rows:
        return
    x = np.arange(len(sample_rows))
    for row in metric_rows:
        target = row["target"]
        unit = row.get("unit", "")
        actual, predicted, _ = prediction_arrays(sample_rows, target)
        if actual.size == 0:
            continue

        plt.figure(figsize=(8, 4.5))
        plt.plot(x[: actual.size], actual, marker="o", linewidth=1.8, label="Actual", color="#2f80ed")
        plt.plot(x[: predicted.size], predicted, marker="s", linewidth=1.8, label="Predicted", color="#f2994a")
        plt.xlabel("Sample index")
        plt.ylabel(unit)
        plt.title(f"Prediction Trace: {target_label(target)}")
        plt.grid(alpha=0.25)
        plt.legend()
        savefig(plot_dir / f"prediction_trace_{target}.png", dpi)


def plot_residual_boxplot(sample_rows: list[dict[str, str]], metric_rows: list[dict[str, str]], plot_dir: Path, dpi: int) -> None:
    errors = []
    labels = []
    for row in metric_rows:
        _, _, error = prediction_arrays(sample_rows, row["target"])
        if error.size:
            errors.append(error)
            labels.append(row["target"])
    if not errors:
        return

    plt.figure(figsize=(8, 4.8))
    plt.boxplot(errors, tick_labels=labels, patch_artist=True, medianprops={"color": "#222222", "linewidth": 1.5})
    plt.axhline(0, color="#eb5757", linewidth=1.5)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Prediction error")
    plt.title("Residual Spread Across Targets")
    plt.grid(axis="y", alpha=0.25)
    savefig(plot_dir / "residual_boxplot_all_targets.png", dpi)


def plot_feature_importance(rows: list[dict[str, str]], plot_dir: Path, dpi: int, top_n: int) -> None:
    if not rows:
        return
    targets = sorted({row["target"] for row in rows})
    for target in targets:
        subset = [row for row in rows if row["target"] == target]
        subset.sort(key=lambda item: int(item["importance"]), reverse=True)
        top = subset[:top_n]
        if not top:
            continue
        features = [row["feature"] for row in top][::-1]
        importance = [int(row["importance"]) for row in top][::-1]

        plt.figure(figsize=(7, 4.8))
        plt.barh(features, importance, color="#2f80ed")
        plt.xlabel("LightGBM split importance")
        plt.title(f"Top Feature Importance: {target_label(target)}")
        plt.grid(axis="x", alpha=0.25)
        savefig(plot_dir / f"feature_importance_{target}.png", dpi)


def write_metrics_csv(metric_rows: list[dict[str, str]], sample_rows: list[dict[str, str]], output_dir: Path) -> None:
    summary_path = output_dir / "npk_metrics_summary.csv"
    fields = [
        "target",
        "unit",
        "train_mae",
        "test_mae",
        "test_rmse",
        "train_r2",
        "test_r2",
        "sample_mean_error",
        "sample_median_abs_error",
        "sample_max_abs_error",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in metric_rows:
            _, _, error = prediction_arrays(sample_rows, row["target"])
            abs_error = np.abs(error) if error.size else np.array([math.nan])
            writer.writerow(
                {
                    "target": row["target"],
                    "unit": row.get("unit", ""),
                    "train_mae": row.get("train_mae", ""),
                    "test_mae": row.get("test_mae", ""),
                    "test_rmse": row.get("test_rmse", ""),
                    "train_r2": row.get("train_r2", ""),
                    "test_r2": row.get("test_r2", ""),
                    "sample_mean_error": f"{float(np.nanmean(error)):.6f}" if error.size else "",
                    "sample_median_abs_error": f"{float(np.nanmedian(abs_error)):.6f}" if error.size else "",
                    "sample_max_abs_error": f"{float(np.nanmax(abs_error)):.6f}" if error.size else "",
                }
            )


def markdown_table(rows: list[dict[str, str]]) -> str:
    headers = ["Target", "Unit", "Train MAE", "Test MAE", "Test RMSE", "Train R2", "Test R2"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("target", ""),
                    row.get("unit", ""),
                    f"{to_float(row.get('train_mae')):.4f}",
                    f"{to_float(row.get('test_mae')):.4f}",
                    f"{to_float(row.get('test_rmse')):.4f}",
                    f"{to_float(row.get('train_r2')):.4f}",
                    f"{to_float(row.get('test_r2')):.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_readme(output_dir: Path, model_dir: Path, metric_rows: list[dict[str, str]], report: dict[str, Any]) -> None:
    dataset = report.get("dataset", {}) if isinstance(report, dict) else {}
    readme = output_dir / "README.md"
    text = [
        "# NPK/Fertigation Model Evaluation Results",
        "",
        "This folder contains paper-ready plots and tables generated from the saved LightGBM model artifacts.",
        "",
        "## Source Artifacts",
        "",
        f"- Model artifact folder: `{model_dir}`",
        "- `model_metrics.csv`: saved train/test metrics",
        "- `model_report.json`: dataset and metric metadata",
        "- `sample_predictions.csv`: actual, predicted, and error examples",
        "- `feature_importance.csv`: LightGBM split importance values",
        "",
        "## Dataset Summary",
        "",
        f"- Total rows: {dataset.get('total_rows', 'unknown')}",
        f"- Train rows: {dataset.get('train_rows', 'unknown')}",
        f"- Test rows: {dataset.get('test_rows', 'unknown')}",
        f"- Split: {dataset.get('split', 'unknown')}",
        f"- Features: {dataset.get('features', 'unknown')}",
        "",
        "## Main Figures for the Research Paper",
        "",
        "- `plots/mae_rmse_per_target.png`",
        "- `plots/r2_per_target.png`",
        "- `plots/normalized_error_percent_range.png`",
        "- `plots/actual_vs_predicted_<target>.png`",
        "- `plots/residual_histogram_<target>.png`",
        "- `plots/residual_boxplot_all_targets.png`",
        "- `plots/feature_importance_<target>.png`",
        "",
        "Every custom plot is also saved as a PDF.",
        "",
        "## Metric Summary",
        "",
        markdown_table(metric_rows) if metric_rows else "No metric rows were found.",
        "",
        "## Suggested Paper Reporting",
        "",
        "Report MAE, RMSE, and R2 for each target. Use actual-vs-predicted plots to show calibration, residual histograms to show error bias, and feature-importance plots to explain which sensor features most influenced nutrient, irrigation, and pH adjustment predictions.",
        "",
    ]
    readme.write_text("\n".join(text), encoding="utf-8")


def print_summary(metric_rows: list[dict[str, str]], output_dir: Path) -> None:
    print("\nNPK/Fertigation model summary")
    print("=" * 86)
    print(f"{'Target':<16} {'Unit':<14} {'Test MAE':>10} {'Test RMSE':>11} {'Test R2':>9}")
    print("-" * 86)
    for row in metric_rows:
        print(
            f"{row.get('target',''):<16} {row.get('unit',''):<14} "
            f"{to_float(row.get('test_mae')):>10.4f} {to_float(row.get('test_rmse')):>11.4f} "
            f"{to_float(row.get('test_r2')):>9.4f}"
        )
    print(f"\nSaved NPK evaluation results to: {output_dir}")


def main() -> None:
    args = parse_args()
    configure_plot()
    model_dir = args.model_dir.resolve()
    output_dir = args.output.resolve()
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = read_csv_dicts(model_dir / "model_metrics.csv")
    sample_rows = read_csv_dicts(model_dir / "sample_predictions.csv")
    importance_rows = read_csv_dicts(model_dir / "feature_importance.csv")

    report_path = model_dir / "model_report.json"
    report: dict[str, Any] = {}
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)

    if not metric_rows:
        raise FileNotFoundError(f"No metric rows found at {model_dir / 'model_metrics.csv'}")

    plot_mae_rmse(metric_rows, plot_dir, args.dpi)
    plot_r2(metric_rows, plot_dir, args.dpi)
    plot_error_relative_to_range(metric_rows, plot_dir, args.dpi)
    plot_actual_vs_predicted(sample_rows, metric_rows, plot_dir, args.dpi)
    plot_residual_histograms(sample_rows, metric_rows, plot_dir, args.dpi)
    plot_prediction_series(sample_rows, metric_rows, plot_dir, args.dpi)
    plot_residual_boxplot(sample_rows, metric_rows, plot_dir, args.dpi)
    plot_feature_importance(importance_rows, plot_dir, args.dpi, args.top_features)

    write_metrics_csv(metric_rows, sample_rows, output_dir)
    write_readme(output_dir, model_dir, metric_rows, report)
    print_summary(metric_rows, output_dir)


if __name__ == "__main__":
    main()
