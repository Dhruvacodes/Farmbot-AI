# Pineapple Model Evaluation Guide

This guide explains how to generate validation graphs and metric tables for a research paper using Ultralytics YOLO `.pt` models.

## What the Script Produces

`evaluate_pineapple_models.py` runs validation and saves:

- Precision-Recall curve per selected class
- Normalized confusion matrix
- Raw confusion matrix
- mAP@0.50 and mAP@0.50:0.95 per class bar chart
- Precision, recall, and F1 score per class bar chart
- F1-confidence, precision-confidence, and recall-confidence curves
- Training and validation loss curves from `results.csv`
- Validation metric curves from `results.csv`
- Learning-rate schedule from `results.csv`
- Ultralytics-generated plots such as `BoxPR_curve.png`, `BoxF1_curve.png`, and validation batch prediction images
- CSV tables and Markdown summaries for paper writing

## Install Dependencies

From `FRUIT_PINEAPPLE`:

```powershell
pip install -r requirements.txt
```

## Basic Command

From the `FRUIT_PINEAPPLE` folder:

```powershell
python .\evaluate_pineapple_models.py `
  --models .\farmbot_dataset\runs\detect\train2\weights\best.pt `
  --data .\farmbot_dataset\data.yaml `
  --class-names Unripe Semi-ripe Ripe `
  --training-runs .\farmbot_dataset\runs\detect\train2 `
  --output .\evaluation_results
```

The final results will be saved in:

```text
FRUIT_PINEAPPLE/evaluation_results/
```

## Evaluating Multiple Models

You can evaluate multiple `.pt` files in one run:

```powershell
python .\evaluate_pineapple_models.py `
  --models .\best.pt .\farmbot_dataset\runs\detect\train2\weights\best.pt `
  --data .\farmbot_dataset\data.yaml `
  --class-names Unripe Semi-ripe Ripe
```

Each model gets its own result folder.

## Important Dataset Note

Your current checked-in `farmbot_dataset/data.yaml` lists many plant/leaf disease classes, not only:

```text
Unripe, Semi-ripe, Ripe
```

For your ripeness paper, make sure `data.yaml` points to the validation dataset used for pineapple ripeness classification/detection and that the class names match your trained model. If your class spelling is different, pass the exact names:

```powershell
--class-names unripe semi_ripe ripe
```

The script prints a warning when the requested classes are not found.

## Best Figures to Use in a Paper

Use these first:

- `plots/precision_recall_curve_per_class.png`
- `plots/confusion_matrix_normalized_custom.png`
- `plots/map_per_class_bar.png`
- `plots/precision_recall_f1_per_class.png`
- `training_curves/<run>/training_validation_loss_curves.png`
- `training_curves/<run>/validation_metrics_curves.png`

PDF versions are also saved beside each PNG.

## Output Files

For each model:

```text
evaluation_results/<model_name>/
  evaluation_summary.md
  overall_metrics.json
  per_class_metrics_all.csv
  per_class_metrics_selected.csv
  plots/
  sample_predictions/
  ultralytics_val/
```

The most useful table for the paper is:

```text
per_class_metrics_selected.csv
```

It contains per-class precision, recall, F1 score, mAP@0.50, and mAP@0.50:0.95.

## Common Options

```powershell
python .\evaluate_pineapple_models.py --help
```

Useful flags:

- `--imgsz 640`: validation image size
- `--batch 16`: batch size
- `--device 0`: use GPU 0
- `--device cpu`: force CPU
- `--split val`: evaluate validation split
- `--split test`: evaluate test split
- `--sample-predictions 20`: save predictions for 20 validation images
- `--dpi 300`: paper-quality plot resolution

## Suggested Text for Methodology

The trained Ultralytics YOLO model was evaluated on the validation set using standard object detection metrics. Precision, recall, F1 score, mAP@0.50, and mAP@0.50:0.95 were computed per ripeness class. A normalized confusion matrix was used to analyze class-wise misclassification, while Precision-Recall curves were plotted for Unripe, Semi-ripe, and Ripe pineapple classes.
