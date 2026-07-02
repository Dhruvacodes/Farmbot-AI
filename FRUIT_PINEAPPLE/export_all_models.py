"""
Master export script to generate all ONNX models.
Exports:
  1. Weight Prediction (sklearn -> ONNX)
  2. Ripeness Classification (YOLOv8 -> ONNX)
  3. NPK Fertigation (LightGBM -> ONNX)
  4. Detection (YOLOv8 -> ONNX, if not already converted)
"""

import os
import sys
from pathlib import Path


def export_all_models():
    """Run all model exports."""

    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "EXPORTING ALL MODELS TO ONNX" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝")

    results = {}

    # 1. Export Weight Prediction
    print("\n\n[1/4] Exporting Weight Prediction Model...")
    print("-" * 80)
    try:
        from export_weight_prediction_onnx import export_weight_prediction_to_onnx
        weight_path = export_weight_prediction_to_onnx()
        results['weight_prediction'] = {
            'status': 'SUCCESS',
            'path': weight_path
        }
    except Exception as e:
        print(f"✗ Failed: {e}")
        results['weight_prediction'] = {
            'status': 'FAILED',
            'error': str(e)
        }

    # 2. Export Ripeness Classification
    print("\n\n[2/4] Exporting Ripeness Classification Model...")
    print("-" * 80)
    try:
        from export_classification_onnx import export_classification_to_onnx
        classification_path = export_classification_to_onnx()
        results['ripeness_classification'] = {
            'status': 'SUCCESS',
            'path': classification_path
        }
    except Exception as e:
        print(f"✗ Failed: {e}")
        results['ripeness_classification'] = {
            'status': 'FAILED',
            'error': str(e)
        }

    # 3. Export NPK Fertigation Models
    print("\n\n[3/4] Exporting NPK Fertigation Models...")
    print("-" * 80)
    try:
        npk_dir = Path("../NPK Pineapple")
        if npk_dir.exists():
            os.chdir(npk_dir)
            from model.train import train_pipeline
            from model.export import export_to_onnx
            import yaml

            with open('config/system_config.yaml', 'r') as f:
                config = yaml.safe_load(f)

            print("Training NPK models...")
            models, feature_names = train_pipeline(config)

            print("Exporting NPK models to ONNX...")
            npk_path = export_to_onnx(models, feature_names,
                                      config['model']['onnx_path'], config)
            results['npk_fertigation'] = {
                'status': 'SUCCESS',
                'path': npk_path
            }
            os.chdir("../FRUIT_PINEAPPLE")
        else:
            raise FileNotFoundError("NPK Pineapple directory not found")
    except Exception as e:
        print(f"✗ Failed: {e}")
        results['npk_fertigation'] = {
            'status': 'FAILED',
            'error': str(e)
        }

    # 4. Check Detection Model
    print("\n\n[4/4] Checking Detection Model...")
    print("-" * 80)
    try:
        detection_path = "models/detection_model.onnx"
        if os.path.exists(detection_path):
            print(f"✓ Detection model already exists at: {detection_path}")
            results['detection'] = {
                'status': 'EXISTS',
                'path': detection_path
            }
        else:
            # Try to export YOLOv8 detection if .pt exists
            from ultralytics import YOLO
            pt_files = [
                "best.pt",
                "farmbot_dataset/runs/detect/train2/weights/best.pt"
            ]

            for pt_path in pt_files:
                if os.path.exists(pt_path):
                    print(f"Found detection model at: {pt_path}")
                    model = YOLO(pt_path)
                    export_path = model.export(format='onnx')

                    import shutil
                    os.makedirs("models", exist_ok=True)
                    shutil.copy(export_path, detection_path)
                    results['detection'] = {
                        'status': 'SUCCESS',
                        'path': detection_path
                    }
                    break
            else:
                print("✗ Detection model .pt not found")
                results['detection'] = {
                    'status': 'NOT_FOUND',
                    'message': 'Please convert YOLOv8 detection model manually'
                }
    except Exception as e:
        print(f"✗ Failed: {e}")
        results['detection'] = {
            'status': 'FAILED',
            'error': str(e)
        }

    # Summary
    print("\n\n" + "=" * 80)
    print("EXPORT SUMMARY")
    print("=" * 80)

    for model_name, result in results.items():
        status = result['status']
        symbol = {
            'SUCCESS': '✓',
            'EXISTS': '✓',
            'FAILED': '✗',
            'NOT_FOUND': '?'
        }.get(status, '?')

        print(f"\n{symbol} {model_name.upper()}: {status}")
        if 'path' in result:
            print(f"  Location: {result['path']}")
        if 'error' in result:
            print(f"  Error: {result['error']}")

    print("\n" + "=" * 80)
    print(f"Export process complete. Check 'models/' directory for ONNX files.")
    print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Run all exports
    results = export_all_models()

    # Exit with error if any failed
    failed = sum(1 for r in results.values() if r['status'] == 'FAILED')
    sys.exit(failed)
