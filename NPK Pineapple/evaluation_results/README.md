# NPK/Fertigation Model Evaluation Results

This folder contains paper-ready plots and tables generated from the saved LightGBM model artifacts.

## Source Artifacts

- Model artifact folder: `C:\Users\Asus\Desktop\FARMBOT AI\FRC\NPK Pineapple\model`
- `model_metrics.csv`: saved train/test metrics
- `model_report.json`: dataset and metric metadata
- `sample_predictions.csv`: actual, predicted, and error examples
- `feature_importance.csv`: LightGBM split importance values

## Dataset Summary

- Total rows: 60000
- Train rows: 48000
- Test rows: 12000
- Split: time_ordered_80_20
- Features: 20

## Main Figures for the Research Paper

- `plots/mae_rmse_per_target.png`
- `plots/r2_per_target.png`
- `plots/normalized_error_percent_range.png`
- `plots/actual_vs_predicted_<target>.png`
- `plots/residual_histogram_<target>.png`
- `plots/residual_boxplot_all_targets.png`
- `plots/feature_importance_<target>.png`

Every custom plot is also saved as a PDF.

## Metric Summary

| Target | Unit | Train MAE | Test MAE | Test RMSE | Train R2 | Test R2 |
| --- | --- | --- | --- | --- | --- | --- |
| delta_N | mg/kg | 2.9613 | 3.2514 | 6.4019 | 0.5546 | 0.4436 |
| delta_P | mg/kg | 1.4309 | 1.5913 | 3.1714 | 0.5657 | 0.4542 |
| delta_K | mg/kg | 3.0706 | 3.4126 | 6.4761 | 0.5531 | 0.4284 |
| irrigation_ml | mL | 1.5095 | 1.6354 | 3.7426 | 0.9981 | 0.9980 |
| pH_adj | pH adjustment | 0.0439 | 0.0480 | 0.0856 | 0.3699 | 0.2124 |

## Suggested Paper Reporting

Report MAE, RMSE, and R2 for each target. Use actual-vs-predicted plots to show calibration, residual histograms to show error bias, and feature-importance plots to explain which sensor features most influenced nutrient, irrigation, and pH adjustment predictions.
