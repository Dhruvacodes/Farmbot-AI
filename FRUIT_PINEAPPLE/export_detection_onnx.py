"""
Export detection model to ONNX format.
Converts YOLOv8 detection model to ONNX.
"""

from ultralytics import YOLO
import os


def export_detection_to_onnx(
    model_path: str = "best.pt",
    output_path: str = "models/detection_model.onnx"
) -> str:
    """
    Export YOLOv8 detection model to ONNX format.

    Args:
        model_path: Path to trained YOLOv8 model (.pt file)
        output_path: Path to save ONNX model

    Returns:
        Path to exported ONNX model
    """

    print("=" * 80)
    print("EXPORTING DETECTION MODEL TO ONNX")
    print("=" * 80)

    # Create output directory
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"[FAIL] Model not found at: {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load model
    print(f"\nLoading model from: {model_path}")
    model = YOLO(model_path)
    print("[OK] Model loaded successfully")

    # Export to ONNX
    print("\nExporting to ONNX format...")
    print("  Input: Variable resolution images (typically 640x640)")
    print("  Output: Bounding box detections + confidences")

    try:
        print("\nRunning export (this may take a while)...")
        export_result = model.export(format='onnx', imgsz=640, opset=12, simplify=True)
        print(f"\n[OK] ONNX model exported successfully")

        # The export returns the path to the exported file
        # Copy it to our models directory if needed
        import shutil
        if export_result and str(export_result) != output_path:
            shutil.copy(export_result, output_path)
            print(f"  Output: {output_path}")
        else:
            output_path = export_result

    except Exception as e:
        print(f"\n[FAIL] Error exporting model: {e}")
        raise

    print("\n" + "=" * 80)
    print("EXPORT COMPLETE")
    print("=" * 80)

    return str(output_path)


if __name__ == "__main__":
    try:
        result = export_detection_to_onnx()
        print(f"\n[OK] Exported to: {result}")
    except Exception as e:
        print(f"\n[FAIL] Export failed: {e}")
        import sys
        sys.exit(1)
