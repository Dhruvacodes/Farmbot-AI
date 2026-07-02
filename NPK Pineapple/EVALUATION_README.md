# NPK/Fertigation Model Evaluation Guide

This guide explains how to generate research-paper graphs for the NPK pineapple fertigation models.

## What the Script Uses

`evaluate_npk_models.py` reads the saved artifacts from the training pipeline:

- `model/model_metrics.csv`
- `model/model_report.json`
- `model/sample_predictions.csv`
- `model/feature_importance.csv`

It does not retrain the LightGBM models, so it runs quickly and keeps the results reproducible.

## What It Produces

The script saves:

- MAE/RMSE comparison per target
- Train vs test R2 comparison per target
- Normalized error as a percentage of target range
- Actual vs predicted scatter plot per target
- Residual histogram per target
- Prediction trace per target
- Residual boxplot across all targets
- Feature importance plot per target
- CSV metric summary
- Detailed result README

PNG and PDF versions are saved for each figure.

## Run

From the `NPK Pineapple` folder:

```powershell
python .\evaluate_npk_models.py `
  --model-dir .\model `
  --output .\evaluation_results
```

The output will be saved to:

```text
NPK Pineapple/evaluation_results/
```

## Main Figures for the Paper

Use these first:

- `plots/mae_rmse_per_target.png`
- `plots/r2_per_target.png`
- `plots/normalized_error_percent_range.png`
- `plots/actual_vs_predicted_delta_N.png`
- `plots/actual_vs_predicted_delta_P.png`
- `plots/actual_vs_predicted_delta_K.png`
- `plots/actual_vs_predicted_irrigation_ml.png`
- `plots/actual_vs_predicted_pH_adj.png`
- `plots/residual_boxplot_all_targets.png`
- `plots/feature_importance_<target>.png`

## Metrics to Report

For each model target, report:

- MAE: mean absolute error in the target unit
- RMSE: root mean squared error in the target unit
- R2: explained variance score
- Normalized MAE/RMSE relative to the observed test-set target range

The target models are:

- `delta_N`: nitrogen adjustment, mg/kg
- `delta_P`: phosphorus adjustment, mg/kg
- `delta_K`: potassium adjustment, mg/kg
- `irrigation_ml`: irrigation volume, mL
- `pH_adj`: pH adjustment

## Suggested Methodology Text

The NPK/fertigation subsystem was evaluated as a multi-target regression problem, with separate LightGBM regressors trained for nitrogen, phosphorus, potassium, irrigation, and pH adjustment outputs. Model performance was assessed using mean absolute error, root mean squared error, and R2 on a held-out time-ordered test set. Actual-vs-predicted plots and residual distributions were generated to assess calibration, bias, and target-specific error behavior.
