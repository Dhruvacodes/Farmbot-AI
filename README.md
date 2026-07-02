# FARMBOT AI

Edge AI stack for an agricultural robot focused on pineapple crop monitoring, fruit vision, leaf disease detection, and NPK fertigation control. This is a cleaned GitHub-ready extract of the project: it keeps the code, deployable lightweight model artifacts, configs, tests, and useful documentation; it leaves out raw datasets, virtual environments, caches, zip archives, and oversized training checkpoints.

## What This Project Does

FARMBOT AI combines four working subsystems:

1. **PC dashboard**
   - FastAPI web dashboard for live robot telemetry.
   - Shows USB/depth camera streams, NPK sensor readings, model predictions, leaf disease results, fruit count, ripeness field, estimated weight, trends, and events.
   - Runs in simulation mode when no Jetson is connected.

2. **NPK pineapple fertigation system**
   - Closed-loop nutrient, pH, and irrigation controller for pineapple cultivation.
   - Uses engineered sensor features, LightGBM regressors exported to ONNX, safety constraints, simulated/real sensor paths, actuator wrappers, SQLite logging, and tests.
   - Supports simulated operation today and a Modbus RTU/Jetson GPIO path for hardware integration.

3. **Fruit pineapple vision pipeline**
   - Pineapple detection using exported YOLO ONNX/TorchScript artifacts.
   - Weight prediction using a lightweight sklearn/ONNX model.
   - Export and evaluation scripts for detection, ripeness classification, and weight prediction models.

4. **Leaf disease detection**
   - ONNX YOLO-style detector for plant disease classes:
     `fruit_rot`, `healthy`, `mealybug_wilt`, and `root_rot`.
   - Used by the dashboard and Jetson publisher helpers.

## Repository Layout

```text
.
|-- Dashboard/                         # PC dashboard backend and frontend
|-- FRUIT_PINEAPPLE/                   # Fruit detection, weight prediction, export scripts
|   |-- config/                        # Centralized paths and Jetson utilities
|   |-- models/                        # Small deployable ONNX/TorchScript/pkl artifacts
|   |-- notebooks/                     # Small Jetson training notebooks only
|   `-- pineappleWeightPrediction/     # Weight prediction README and CSV
|-- LEAF_DETECTION/                    # Leaf disease ONNX model and notes
|-- NPK Pineapple/                     # NPK/fertigation control system
|   |-- config/
|   |-- control/
|   |-- data/
|   |-- features/
|   |-- inference/
|   |-- model/                         # ONNX models, LightGBM text models, metrics
|   |-- sensors/
|   `-- tests/
|-- running_frc_models/                # Jetson deployment folder with copied ONNX files
`-- docs/                              # ONNX notes and corrected evaluation figures
```

## Included Model Artifacts

| Subsystem | Files | Purpose |
| --- | --- | --- |
| Fruit detection | `FRUIT_PINEAPPLE/models/detection_model.onnx`, `detection_model.torchscript` | Detect pineapple/fruit regions from camera frames |
| Fruit weight | `FRUIT_PINEAPPLE/models/weight_prediction.onnx`, `weight_prediction_model.pkl`, scaler/poly files | Estimate pineapple weight from size features |
| NPK fertigation | `NPK Pineapple/model/pineapple_model_*.onnx` | Predict N, P, K, irrigation, and pH adjustment actions |
| Leaf disease | `LEAF_DETECTION/leaf_detection.onnx` | Detect fruit rot, healthy leaf, mealybug wilt, root rot |
| Jetson deployment | `running_frc_models/*.onnx` | Copies of deployment ONNX files for direct Jetson use |

## Quick Start: PC Dashboard

From this repo root on Windows PowerShell:

```powershell
cd Dashboard
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python server.py --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
```

The dashboard starts in simulation mode. If a Jetson posts telemetry to `/api/telemetry`, the dashboard switches from simulated data to Jetson data.

## Quick Start: NPK/Fertigation Simulation

```powershell
cd "NPK Pineapple"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --mode sim
```

Useful commands:

```powershell
python main.py --train-only
python demo.py
pytest tests/test_system.py -v
```

The NPK system reads `config/system_config.yaml`. Set `system.data_mode` to `real` only after the Modbus sensor and Jetson GPIO wiring are confirmed.

## Quick Start: Jetson Deployment

Copy the deployment folder to the Jetson as `~/running_frc_models`:

```bash
scp -r running_frc_models jetson_user@JETSON_IP:~/
ssh jetson_user@JETSON_IP
chmod +x ~/running_frc_models/*.sh
```

Then on the Jetson:

```bash
~/running_frc_models/1_test_models.sh
~/running_frc_models/2_start_camera_server.sh
~/running_frc_models/3_start_inference.sh
```

Before running `3_start_inference.sh`, edit `PC_IP` inside that script to the dashboard machine's IP address. The current scripts assume the deployment path is `/home/rm/running_frc_models`; update that path if the Jetson username is different.

## Fruit Pipeline Notes

The fruit pipeline keeps lightweight deployable artifacts and export scripts. Raw training datasets and large `.pt` checkpoints were intentionally excluded.

Useful commands:

```powershell
cd FRUIT_PINEAPPLE
pip install -r requirements.txt
python export_weight_prediction_onnx.py
python export_detection_onnx.py
python export_classification_onnx.py
```

Ripeness classification ONNX is not present locally. The export script expects the trained YOLO classification checkpoint from the original Colab/Drive location or a local equivalent.

## What Was Intentionally Left Out

These were excluded so the repo stays uploadable and usable:

- `.git/` history folders
- `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`
- raw Roboflow/PlantVillage datasets
- training output folders such as `runs/`
- zip archives and duplicate extracted archives
- large checkpoints such as `best.pt` over GitHub's 100 MB file limit
- older root-level experiments that are not needed for the current FRC deployment path
- the bad fruit evaluation dump that validated against an 86-class plant/leaf dataset instead of a pineapple-only dataset

The corrected paper-ready figures are in `docs/research_paper_evaluation_images/`.

## Current Status

- Dashboard runs without hardware using simulated telemetry.
- NPK/fertigation subsystem has source code, ONNX models, metrics, simulator, safety layer, and tests.
- Fruit detection and weight prediction have deployable model artifacts.
- Leaf disease detection ONNX is included and wired into dashboard helpers.
- Jetson scripts are present with deployment model copies.
- True ripeness classification still needs the missing trained checkpoint exported to ONNX.
- Real sensor/actuator operation needs hardware calibration and field testing.

## License

No open-source license file is included. Add a license before making the repository public if that is intended.
