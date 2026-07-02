"""
Evaluate Ultralytics YOLO pineapple detection/ripeness models and create
publication-ready plots.

Default paths are chosen for this repository, but every important path can be
overridden from the command line.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO


DEFAULT_CLASSES = ["Unripe", "Semi-ripe", "Ripe"]
ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "farmbot_dataset" / "data.yaml"
DEFAULT_MODEL = ROOT / "farmbot_dataset" / "runs" / "detect" / "train2" / "weights" / "best.pt"
DEFAULT_RUN = ROOT / "farmbot_dataset" / "runs" / "detect" / "train2"
DEFAULT_OUTPUT = ROOT / "evaluation_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO validation and generate research-paper evaluation graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", type=Path, default=[DEFAULT_MODEL], help="One or more .pt model files.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Ultralytics dataset YAML.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for reports and plots.")
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help="Classes to emphasize in plots. Use: --class-names Unripe Semi-ripe Ripe",
    )
    parser.add_argument(
        "--training-runs",
        nargs="*",
        type=Path,
        default=[DEFAULT_RUN],
        help="Training run folders or results.csv files for loss/metric curves.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for validation curves.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS during validation.")
    parser.add_argument("--device", default=None, help="Device string, for example cpu, 0, or 0,1.")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--sample-predictions", type=int, default=12, help="Number of validation images to save predictions for.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved paper figures.")
    return parser.parse_args()


def slugify(value: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in value.strip()]
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "model"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_dataset_yaml(data_path: Path, output_dir: Path, split: str) -> tuple[Path, dict[str, Any], list[str], list[str]]:
    """Load YAML and repair common relative paths into an output copy if needed."""
    data_path = data_path.resolve()
    data = read_yaml(data_path)
    warnings: list[str] = []

    names_raw = data.get("names", [])
    if isinstance(names_raw, dict):
        class_names = [names_raw[k] for k in sorted(names_raw, key=lambda x: int(x))]
    else:
        class_names = list(names_raw)

    fixed = dict(data)
    changed = False
    yaml_dir = data_path.parent

    for key in ("train", "val", "test"):
        if key not in fixed:
            continue
        value = fixed[key]
        if not isinstance(value, str):
            continue

        candidate = (yaml_dir / value).resolve()
        if candidate.exists():
            continue

        parts = Path(value).parts
        if "images" in parts:
            split_name = parts[-2] if len(parts) >= 2 else key
            sibling = (yaml_dir / split_name / "images").resolve()
            if sibling.exists():
                fixed[key] = str(sibling)
                changed = True
                warnings.append(f"Patched missing {key} path from '{value}' to '{sibling}'.")

    if changed:
        output_dir.mkdir(parents=True, exist_ok=True)
        patched = output_dir / "data_eval_patched.yaml"
        with patched.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(fixed, handle, sort_keys=False)
        data_path = patched
        data = fixed

    split_path = data.get(split)
    if isinstance(split_path, str) and not Path(split_path).is_absolute():
        split_abs = (data_path.parent / split_path).resolve()
        if not split_abs.exists():
            warnings.append(f"The {split} image path does not exist: {split_abs}")

    return data_path, data, class_names, warnings


def normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def selected_class_indices(all_names: list[str], requested: list[str] | None) -> tuple[list[int], list[str]]:
    if not requested:
        return list(range(len(all_names))), all_names

    normalized = {normalize_name(name): i for i, name in enumerate(all_names)}
    aliases = {
        "semiripe": ["semiripe", "semiripen", "semi_ripe", "semi-ripe"],
        "semi_ripe": ["semiripe", "semi_ripe", "semi-ripe"],
    }

    indices: list[int] = []
    missing: list[str] = []
    for name in requested:
        key = normalize_name(name)
        match = normalized.get(key)
        if match is None:
            for alias in aliases.get(key, []):
                match = normalized.get(normalize_name(alias))
                if match is not None:
                    break
        if match is None:
            missing.append(name)
        elif match not in indices:
            indices.append(match)

    if missing:
        print(f"Warning: requested classes not found in dataset/model names: {', '.join(missing)}")
    if not indices:
        print("Warning: no requested classes matched; using all classes instead.")
        return list(range(len(all_names))), all_names

    return indices, [all_names[i] for i in indices]


def metric_array(metric_obj: Any, attr: str) -> np.ndarray:
    value = getattr(metric_obj, attr, [])
    return np.asarray(value, dtype=float)


def safe_class_result(metrics: Any, metric_row_index: int) -> tuple[float, float]:
    try:
        result = metrics.class_result(metric_row_index)
        return float(result[2]), float(result[3])
    except Exception:
        return math.nan, math.nan


def extract_detection_summary(metrics: Any, class_names: list[str]) -> pd.DataFrame:
    try:
        rows = metrics.summary(normalize=True, decimals=6)
        df = pd.DataFrame(rows)
        return df.rename(
            columns={
                "Class": "class",
                "Images": "images",
                "Instances": "instances",
                "Box-P": "precision",
                "Box-R": "recall",
                "Box-F1": "f1",
            }
        )
    except Exception:
        pass

    box = getattr(metrics, "box", metrics)
    ap_class_index = np.asarray(getattr(metrics, "ap_class_index", np.arange(len(class_names))), dtype=int)
    p = metric_array(box, "p")
    r = metric_array(box, "r")
    f1 = metric_array(box, "f1")

    rows = []
    for row_idx, class_idx in enumerate(ap_class_index):
        map50, map5095 = safe_class_result(metrics, row_idx)
        rows.append(
            {
                "class": class_names[class_idx] if class_idx < len(class_names) else str(class_idx),
                "images": np.nan,
                "instances": np.nan,
                "precision": p[row_idx] if row_idx < len(p) else np.nan,
                "recall": r[row_idx] if row_idx < len(r) else np.nan,
                "f1": f1[row_idx] if row_idx < len(f1) else np.nan,
                "mAP50": map50,
                "mAP50-95": map5095,
            }
        )
    return pd.DataFrame(rows)


def apply_class_filter(df: pd.DataFrame, wanted_names: list[str]) -> pd.DataFrame:
    wanted = {normalize_name(name) for name in wanted_names}
    if "class" not in df.columns:
        return df
    filtered = df[df["class"].map(lambda x: normalize_name(str(x)) in wanted)].copy()
    return filtered if not filtered.empty else df


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


def plot_map_bar(df: pd.DataFrame, plot_dir: Path, dpi: int) -> None:
    required = {"class", "mAP50", "mAP50-95"}
    if not required.issubset(df.columns) or df.empty:
        return
    x = np.arange(len(df))
    width = 0.36
    fig_width = max(7, min(18, 1.1 * len(df)))
    plt.figure(figsize=(fig_width, 4.8))
    plt.bar(x - width / 2, df["mAP50"], width, label="mAP@0.50", color="#2f80ed")
    plt.bar(x + width / 2, df["mAP50-95"], width, label="mAP@0.50:0.95", color="#27ae60")
    plt.xticks(x, df["class"], rotation=35, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Per-Class Mean Average Precision")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    savefig(plot_dir / "map_per_class_bar.png", dpi)


def plot_metric_bar(df: pd.DataFrame, plot_dir: Path, dpi: int) -> None:
    metrics = [m for m in ["precision", "recall", "f1"] if m in df.columns]
    if not metrics or df.empty:
        return
    x = np.arange(len(df))
    width = 0.8 / len(metrics)
    colors = ["#2f80ed", "#f2994a", "#9b51e0"]
    fig_width = max(7, min(18, 1.1 * len(df)))
    plt.figure(figsize=(fig_width, 4.8))
    for i, metric in enumerate(metrics):
        offset = (i - (len(metrics) - 1) / 2) * width
        plt.bar(x + offset, df[metric], width, label=metric.capitalize(), color=colors[i])
    plt.xticks(x, df["class"], rotation=35, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Per-Class Precision, Recall, and F1 Score")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    savefig(plot_dir / "precision_recall_f1_per_class.png", dpi)


def plot_pr_curves(metrics: Any, all_names: list[str], selected_indices: list[int], plot_dir: Path, dpi: int) -> None:
    try:
        curve = metrics.curves_results[0]
        recall_x = np.asarray(curve[0], dtype=float)
        precision_y = np.asarray(curve[1], dtype=float)
    except Exception:
        return

    if precision_y.ndim != 2 or len(recall_x) == 0:
        return

    ap_class_index = np.asarray(getattr(metrics, "ap_class_index", np.arange(precision_y.shape[0])), dtype=int)
    class_to_curve_row = {int(class_idx): row for row, class_idx in enumerate(ap_class_index)}

    plt.figure(figsize=(7, 5))
    plotted = False
    for class_idx in selected_indices:
        row = class_to_curve_row.get(int(class_idx))
        if row is None or row >= precision_y.shape[0]:
            continue
        name = all_names[class_idx] if class_idx < len(all_names) else str(class_idx)
        plt.plot(recall_x, precision_y[row], linewidth=2, label=name)
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.title("Precision-Recall Curve Per Class")
    plt.grid(alpha=0.25)
    plt.legend()
    savefig(plot_dir / "precision_recall_curve_per_class.png", dpi)


def plot_confidence_curves(metrics: Any, all_names: list[str], selected_indices: list[int], plot_dir: Path, dpi: int) -> None:
    curve_specs = [
        (1, "F1", "F1 Score", "f1_confidence_curve_per_class.png"),
        (2, "Precision", "Precision", "precision_confidence_curve_per_class.png"),
        (3, "Recall", "Recall", "recall_confidence_curve_per_class.png"),
    ]
    ap_class_index = np.asarray(getattr(metrics, "ap_class_index", []), dtype=int)
    class_to_curve_row = {int(class_idx): row for row, class_idx in enumerate(ap_class_index)}

    for curve_idx, title_metric, ylabel, filename in curve_specs:
        try:
            x = np.asarray(metrics.curves_results[curve_idx][0], dtype=float)
            y = np.asarray(metrics.curves_results[curve_idx][1], dtype=float)
        except Exception:
            continue
        if y.ndim != 2:
            continue

        plt.figure(figsize=(7, 5))
        plotted = False
        for class_idx in selected_indices:
            row = class_to_curve_row.get(int(class_idx))
            if row is None or row >= y.shape[0]:
                continue
            name = all_names[class_idx] if class_idx < len(all_names) else str(class_idx)
            plt.plot(x, y[row], linewidth=2, label=name)
            plotted = True

        if not plotted:
            plt.close()
            continue
        plt.xlabel("Confidence")
        plt.ylabel(ylabel)
        plt.xlim(0, 1)
        plt.ylim(0, 1.05)
        plt.title(f"{title_metric}-Confidence Curve Per Class")
        plt.grid(alpha=0.25)
        plt.legend()
        savefig(plot_dir / filename, dpi)


def plot_confusion_matrix(metrics: Any, class_names: list[str], selected_indices: list[int], plot_dir: Path, dpi: int) -> None:
    cm_obj = getattr(metrics, "confusion_matrix", None)
    matrix = getattr(cm_obj, "matrix", None)
    if matrix is None:
        return

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        return

    has_background = matrix.shape[0] == len(class_names) + 1 and matrix.shape[1] == len(class_names) + 1
    indices = list(selected_indices)
    labels = [class_names[i] for i in indices]
    if has_background:
        indices_with_bg = indices + [len(class_names)]
        labels = labels + ["Background"]
    else:
        indices_with_bg = indices

    if max(indices_with_bg, default=-1) >= matrix.shape[0]:
        return

    sub = matrix[np.ix_(indices_with_bg, indices_with_bg)]
    row_sums = sub.sum(axis=1, keepdims=True)
    norm = np.divide(sub, row_sums, out=np.zeros_like(sub), where=row_sums != 0)

    for values, title, filename, fmt in [
        (sub, "Confusion Matrix", "confusion_matrix_custom.png", ".0f"),
        (norm, "Normalized Confusion Matrix", "confusion_matrix_normalized_custom.png", ".2f"),
    ]:
        size = max(5.5, 0.65 * len(labels))
        plt.figure(figsize=(size, size))
        plt.imshow(values, cmap="Blues", vmin=0, vmax=1 if "Normalized" in title else None)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("True Class")
        plt.ylabel("Predicted Class")
        plt.title(title)
        threshold = values.max() / 2 if values.size else 0
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                color = "white" if values[i, j] > threshold else "#222222"
                plt.text(j, i, format(values[i, j], fmt), ha="center", va="center", color=color, fontsize=8)
        savefig(plot_dir / filename, dpi)


def plot_training_curves(training_paths: list[Path], output_dir: Path, dpi: int) -> list[Path]:
    generated: list[Path] = []
    for path in training_paths:
        csv_path = path if path.suffix.lower() == ".csv" else path / "results.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        if "epoch" not in df.columns:
            df.insert(0, "epoch", np.arange(1, len(df) + 1))
        run_name = slugify(csv_path.parent.name)
        plot_dir = output_dir / "training_curves" / run_name
        plot_dir.mkdir(parents=True, exist_ok=True)

        loss_cols = [c for c in df.columns if c.startswith("train/") or c.startswith("val/")]
        if loss_cols:
            plt.figure(figsize=(8, 5))
            for col in loss_cols:
                if "loss" in col.lower():
                    plt.plot(df["epoch"], df[col], label=col)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss Curves")
            plt.grid(alpha=0.25)
            plt.legend()
            path_out = plot_dir / "training_validation_loss_curves.png"
            savefig(path_out, dpi)
            generated.append(path_out)

        metric_cols = [c for c in df.columns if c.startswith("metrics/")]
        if metric_cols:
            plt.figure(figsize=(8, 5))
            for col in metric_cols:
                plt.plot(df["epoch"], df[col], label=col.replace("metrics/", ""))
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.ylim(0, 1.05)
            plt.title("Validation Metrics During Training")
            plt.grid(alpha=0.25)
            plt.legend()
            path_out = plot_dir / "validation_metrics_curves.png"
            savefig(path_out, dpi)
            generated.append(path_out)

        lr_cols = [c for c in df.columns if c.startswith("lr/")]
        if lr_cols:
            plt.figure(figsize=(8, 5))
            for col in lr_cols:
                plt.plot(df["epoch"], df[col], label=col)
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.grid(alpha=0.25)
            plt.legend()
            path_out = plot_dir / "learning_rate_schedule.png"
            savefig(path_out, dpi)
            generated.append(path_out)

    return generated


def copy_ultralytics_plots(source_dir: Path, plot_dir: Path) -> list[Path]:
    copied: list[Path] = []
    patterns = [
        "*PR_curve.png",
        "*P_curve.png",
        "*R_curve.png",
        "*F1_curve.png",
        "confusion_matrix*.png",
        "val_batch*_pred.jpg",
        "val_batch*_labels.jpg",
    ]
    for pattern in patterns:
        for source in source_dir.glob(pattern):
            target = plot_dir / f"ultralytics_{source.name}"
            shutil.copy2(source, target)
            copied.append(target)
    return copied


def find_validation_images(data: dict[str, Any], data_yaml: Path, split: str) -> list[Path]:
    value = data.get(split)
    if not isinstance(value, str):
        return []
    image_dir = Path(value)
    if not image_dir.is_absolute():
        image_dir = (data_yaml.parent / image_dir).resolve()
    if not image_dir.exists():
        return []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in extensions)


def save_sample_predictions(model: YOLO, images: list[Path], model_dir: Path, sample_count: int, imgsz: int, device: str | None) -> None:
    if sample_count <= 0 or not images:
        return
    selected = images[:sample_count]
    model.predict(
        source=[str(p) for p in selected],
        imgsz=imgsz,
        device=device,
        save=True,
        project=str(model_dir),
        name="sample_predictions",
        exist_ok=True,
        verbose=False,
    )


def write_model_report(
    model_name: str,
    model_path: Path,
    model_dir: Path,
    metrics: Any,
    summary_df: pd.DataFrame,
    selected_names: list[str],
    warnings: list[str],
) -> None:
    results_dict = {}
    try:
        results_dict = {k: float(v) for k, v in metrics.results_dict.items()}
    except Exception:
        pass

    report = model_dir / "evaluation_summary.md"
    lines = [
        f"# Evaluation Summary: {model_name}",
        "",
        f"- Model: `{model_path}`",
        f"- Emphasized classes: {', '.join(selected_names)}",
        "",
        "## Overall Metrics",
        "",
    ]
    if results_dict:
        for key, value in results_dict.items():
            lines.append(f"- {key}: {value:.5f}")
    else:
        lines.append("- Overall metrics were not exposed by this Ultralytics version.")

    if warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- {warning}" for warning in warnings]

    if summary_df.empty:
        table_text = "No per-class summary was available."
    else:
        try:
            table_text = summary_df.to_markdown(index=False, floatfmt=".5f")
        except ImportError:
            table_text = "```csv\n" + summary_df.to_csv(index=False) + "```"

    lines += [
        "",
        "## Per-Class Summary",
        "",
        table_text,
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")

    with (model_dir / "overall_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results_dict, handle, indent=2)


def print_summary_table(model_name: str, summary_df: pd.DataFrame) -> None:
    columns = [c for c in ["class", "precision", "recall", "f1", "mAP50", "mAP50-95"] if c in summary_df.columns]
    print("\n" + "=" * 90)
    print(f"Per-class summary for {model_name}")
    print("=" * 90)
    if columns:
        print(summary_df[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("No per-class precision/recall/F1 table was available.")


def evaluate_model(
    model_path: Path,
    data_yaml: Path,
    data: dict[str, Any],
    class_names: list[str],
    selected_indices: list[int],
    selected_names: list[str],
    output_dir: Path,
    args: argparse.Namespace,
    warnings: list[str],
) -> None:
    model_path = model_path.resolve()
    model_name = slugify(model_path.stem)
    if model_path.parent.name.lower() == "weights":
        model_name = slugify(f"{model_path.parent.parent.name}_{model_path.stem}")

    model_dir = output_dir / model_name
    plot_dir = model_dir / "plots"
    ultralytics_dir = model_dir / "ultralytics_val"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model: {model_path}")
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=True,
        save_json=True,
        project=str(ultralytics_dir),
        name="run",
        exist_ok=True,
        verbose=False,
    )

    run_dir = ultralytics_dir / "run"
    copy_ultralytics_plots(run_dir, plot_dir)

    summary_df = extract_detection_summary(metrics, class_names)
    paper_df = apply_class_filter(summary_df, selected_names)
    summary_df.to_csv(model_dir / "per_class_metrics_all.csv", index=False)
    paper_df.to_csv(model_dir / "per_class_metrics_selected.csv", index=False)

    plot_pr_curves(metrics, class_names, selected_indices, plot_dir, args.dpi)
    plot_confidence_curves(metrics, class_names, selected_indices, plot_dir, args.dpi)
    plot_confusion_matrix(metrics, class_names, selected_indices, plot_dir, args.dpi)
    plot_map_bar(paper_df, plot_dir, args.dpi)
    plot_metric_bar(paper_df, plot_dir, args.dpi)

    validation_images = find_validation_images(data, data_yaml, args.split)
    save_sample_predictions(model, validation_images, model_dir, args.sample_predictions, args.imgsz, args.device)

    write_model_report(model_name, model_path, model_dir, metrics, paper_df, selected_names, warnings)
    print_summary_table(model_name, paper_df)
    print(f"\nSaved plots and tables to: {model_dir}")


def write_readme(output_dir: Path, model_paths: list[Path], data_yaml: Path, selected_names: list[str], training_plots: list[Path]) -> None:
    readme = output_dir / "README.md"
    model_list = "\n".join(f"- `{path}`" for path in model_paths)
    training_list = "\n".join(f"- `{path.relative_to(output_dir) if path.is_relative_to(output_dir) else path}`" for path in training_plots)
    text = f"""# Pineapple Model Evaluation Results

This folder contains validation metrics, plots, CSV tables, and Markdown summaries generated with `evaluate_pineapple_models.py`.

## Evaluation Setup

- Dataset YAML: `{data_yaml}`
- Emphasized classes: {", ".join(selected_names)}
- Models:
{model_list}

## Main Figures for the Research Paper

- `*/plots/precision_recall_curve_per_class.png`
- `*/plots/confusion_matrix_normalized_custom.png`
- `*/plots/map_per_class_bar.png`
- `*/plots/precision_recall_f1_per_class.png`
- `training_curves/*/training_validation_loss_curves.png`
- `training_curves/*/validation_metrics_curves.png`

Every custom plot is also saved as a PDF for paper submission workflows.

## Metrics Tables

- `*/per_class_metrics_selected.csv`: precision, recall, F1, mAP@0.50, and mAP@0.50:0.95 for the selected paper classes.
- `*/per_class_metrics_all.csv`: all classes reported by Ultralytics.
- `*/evaluation_summary.md`: readable summary for each model.
- `*/overall_metrics.json`: raw overall metric dictionary from Ultralytics.

## Additional Graphs

The script keeps both custom figures and Ultralytics-generated figures:

- PR, F1-confidence, precision-confidence, and recall-confidence curves
- Normalized and raw confusion matrices
- Per-class mAP grouped bar chart
- Per-class precision/recall/F1 grouped bar chart
- Validation batch prediction examples
- Training/validation loss curves when `results.csv` is available
- Learning-rate schedule when `results.csv` contains learning-rate columns

## Training Curves Generated

{training_list if training_list else "- No `results.csv` training curves were found."}

## Recommended Paper Reporting

Report the selected-class CSV table together with the normalized confusion matrix and PR curves. For object detection, include both mAP@0.50 and mAP@0.50:0.95 because mAP@0.50 shows detection permissiveness while mAP@0.50:0.95 better reflects localization quality across stricter IoU thresholds.
"""
    readme.write_text(textwrap.dedent(text), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_plot()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_yaml, data, class_names, warnings = load_dataset_yaml(args.data, output_dir, args.split)
    selected_indices, selected_names = selected_class_indices(class_names, args.class_names or DEFAULT_CLASSES)

    if len(class_names) != 3 or [normalize_name(x) for x in selected_names] != [normalize_name(x) for x in DEFAULT_CLASSES]:
        warnings.append(
            "The dataset/model class names do not exactly match Unripe, Semi-ripe, Ripe. "
            "Use --class-names to select the correct ripeness classes if your YAML differs."
        )

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")

    training_plots = plot_training_curves(args.training_runs, output_dir, args.dpi)

    for model_path in args.models:
        if not model_path.exists():
            print(f"Skipping missing model: {model_path}")
            continue
        evaluate_model(
            model_path=model_path,
            data_yaml=data_yaml,
            data=data,
            class_names=class_names,
            selected_indices=selected_indices,
            selected_names=selected_names,
            output_dir=output_dir,
            args=args,
            warnings=warnings,
        )

    write_readme(output_dir, args.models, data_yaml, selected_names, training_plots)
    print(f"\nDone. Master README saved to: {output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
