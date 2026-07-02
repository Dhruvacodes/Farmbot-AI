"""
Export ripeness classification model to ONNX format.
Converts trained YOLOv8 classification model to ONNX.
"""

from ultralytics import YOLO
import os


def export_classification_to_onnx(
    model_path: str = "/content/drive/MyDrive/train2/weights/best.pt",
    output_dir: str = "models"
) -> str:
    """
    Export YOLOv8 classification model to ONNX format.

    Args:
        model_path: Path to trained YOLOv8 model (.pt file)
        output_dir: Directory to save ONNX model

    Returns:
        Path to exported ONNX model
    """

    print("=" * 80)
    print("EXPORTING RIPENESS CLASSIFICATION MODEL TO ONNX")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"\nLoading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"✗ Model not found at: {model_path}")
        print("Please ensure the model is trained and saved.")
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(model_path)
    print("✓ Model loaded successfully")

    # Export to ONNX
    print("\nExporting to ONNX format...")
    print("  Input: 224x224 RGB image (3 channels)")
    print("  Output: Class predictions (ripe, semiripe, unripe)")

    try:
        export_path = model.export(format='onnx', imgsz=224)
        print(f"\n✓ ONNX model exported successfully")
        print(f"  Output path: {export_path}")

        # Copy to models directory with a consistent name
        import shutil
        output_path = os.path.join(output_dir, "ripeness_classification.onnx")
        if isinstance(export_path, str):
            shutil.copy(export_path, output_path)
        print(f"  Copied to: {output_path}")

    except Exception as e:
        print(f"\n✗ Error exporting model: {e}")
        raise

    print("\n" + "=" * 80)
    print("EXPORT COMPLETE")
    print("=" * 80)

    return output_path


if __name__ == "__main__":
    import sys

    # Try different model paths if not found
    model_paths = [
        "/content/drive/MyDrive/train2/weights/best.pt",
        "pineappleWeightPrediction/classification/train2/weights/best.pt",
        "runs/classify/train2/weights/best.pt",
    ]

    export_path = None
    for path in model_paths:
        if os.path.exists(path):
            print(f"Found model at: {path}")
            export_path = export_classification_to_onnx(model_path=path)
            break

    if export_path is None:
        print("Could not find classification model. Please provide the correct path.")
        print(f"Searched paths: {model_paths}")
