#!/usr/bin/env python
"""
Train and export NPK fertigation models to ONNX format.
Generates 5 independent LightGBM models for nutrient/irrigation control.
"""

import sys
import os

# Add the NPK folder to path
sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("=" * 80)
    print("TRAINING & EXPORTING NPK MODELS TO ONNX")
    print("=" * 80)

    try:
        # Import training components
        from model.train import train_pipeline
        from model.export import export_to_onnx
        import yaml

        # Load config
        config_path = "config/system_config.yaml"
        print(f"\nLoading config from: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("[OK] Config loaded")

        # Train models
        print("\n" + "-" * 80)
        print("Training LightGBM models...")
        print("-" * 80)
        models, feature_names = train_pipeline(config)
        print(f"\n[OK] Trained {len(models)} models:")
        for i, target in enumerate(config['model']['targets']):
            print(f"  {i+1}. {target}")

        # Export to ONNX
        print("\n" + "-" * 80)
        print("Exporting models to ONNX...")
        print("-" * 80)
        onnx_path = config['model']['onnx_path']
        export_to_onnx(models, feature_names, onnx_path, config)

        print("\n" + "=" * 80)
        print("[OK] NPK EXPORT COMPLETE")
        print("=" * 80)
        print(f"\nModels saved to: {onnx_path}")
        print("Files generated:")
        for target in config['model']['targets']:
            model_file = onnx_path.replace('.onnx', f'_{target}.onnx')
            if os.path.exists(model_file):
                size_mb = os.path.getsize(model_file) / (1024 * 1024)
                print(f"  - {model_file} ({size_mb:.2f} MB)")

        return 0

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
