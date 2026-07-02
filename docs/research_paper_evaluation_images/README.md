# Improved Paper-Ready Evaluation Figures

I removed the misleading 86-class fruit validation plots and rebuilt this folder around the better pineapple-system results documented in the project plus the saved NPK metrics.

## Important Honesty Note

The exact raw validation artifacts for the documented pineapple detection/classification runs are not present in this workspace. The previous bad fruit plots came from evaluating `farmbot_dataset/data.yaml`, which is an 86-class plant/leaf dataset and was trained for only 1 epoch. These new fruit figures use the documented pineapple metrics from `FRUIT_PINEAPPLE/README.md` instead of that wrong evaluation output.

| Category | Figure Title | File | Source |
| --- | --- | --- | --- |
| fruit | Pineapple Detection Model Performance | [fruit_detection_reported_performance.png](./fruit_detection_reported_performance.png) | `FRUIT_PINEAPPLE/README.md lines 117-142` |
| fruit | Fruit AI Pipeline: Reported Best Results | [fruit_pipeline_reported_best_results.png](./fruit_pipeline_reported_best_results.png) | `FRUIT_PINEAPPLE/README.md lines 117-142` |
| npk | NPK/Fertigation: Actual vs Predicted Control Outputs | [npk_actual_vs_predicted_key_outputs.png](./npk_actual_vs_predicted_key_outputs.png) | `NPK Pineapple\model\sample_predictions.csv` |
| npk | NPK/Fertigation: Error Relative to Operating Range | [npk_error_relative_to_operating_range.png](./npk_error_relative_to_operating_range.png) | `NPK Pineapple\model\model_metrics.csv` |
| npk | NPK/Fertigation Control: Normalized Prediction Accuracy | [npk_normalized_control_accuracy.png](./npk_normalized_control_accuracy.png) | `NPK Pineapple\model\model_metrics.csv` |
