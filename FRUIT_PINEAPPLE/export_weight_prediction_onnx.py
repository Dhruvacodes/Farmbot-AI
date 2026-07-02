"""
Export weight prediction models to ONNX format.
Converts sklearn models to ONNX for deployment.
"""

import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import os


def export_weight_prediction_to_onnx(
    model_path: str = "models/weight_prediction_model.pkl",
    scaler_path: str = "models/weight_scaler.pkl",
    poly_path: str = "models/weight_poly_features.pkl",
    output_path: str = "models/weight_prediction.onnx"
) -> str:
    """
    Export weight prediction sklearn model to ONNX format.

    Args:
        model_path: Path to trained sklearn model
        scaler_path: Path to scaler
        poly_path: Path to PolynomialFeatures transformer
        output_path: Path to save ONNX model

    Returns:
        Path to exported ONNX model
    """

    print("=" * 80)
    print("EXPORTING WEIGHT PREDICTION MODEL TO ONNX")
    print("=" * 80)

    # Create output directory
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Load models
    print(f"\nLoading model from: {model_path}")
    model = joblib.load(model_path)
    print(f"Loading scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)

    # Check if poly features exist
    try:
        print(f"Loading poly features from: {poly_path}")
        poly = joblib.load(poly_path)
        print("[OK] Loaded PolynomialFeatures transformer")
    except:
        print("Note: PolynomialFeatures not found, will export base model")
        poly = None

    # Define input type: 2 features (Length, Width in mm)
    n_features_input = 2
    initial_type = [('float_input', FloatTensorType([1, n_features_input]))]

    print(f"\nInput shape: 1 x {n_features_input}")
    print(f"Output: 1 x 1 (predicted weight in kg)")

    # Convert to ONNX
    print("\nConverting sklearn model to ONNX...")
    try:
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        # Save ONNX model
        onnx.save_model(onnx_model, output_path)
        print(f"[OK] ONNX model saved to: {output_path}")

        # Verify the model
        print("\nVerifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("[OK] ONNX model is valid")

    except Exception as e:
        print(f"[FAIL] Error converting model: {e}")
        raise

    print("\n" + "=" * 80)
    print("EXPORT COMPLETE")
    print("=" * 80)

    return output_path


if __name__ == "__main__":
    export_weight_prediction_to_onnx()
