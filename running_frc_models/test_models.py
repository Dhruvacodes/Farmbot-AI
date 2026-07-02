#!/usr/bin/env python3
"""Test all ONNX models for loading and inference compatibility."""

import onnxruntime as ort
import os
import numpy as np

model_dir = "/home/rm/running_frc_models"

models = [
    "detection_model.onnx",
    "pineapple_model_delta_P.onnx",
    "pineapple_model_delta_K.onnx",
    "pineapple_model_delta_N.onnx",
    "pineapple_model_irrigation_ml.onnx",
    "pineapple_model_pH_adj.onnx",
    "weight_prediction.onnx",
    "leaf_detection.onnx"
]

for model in models:
    model_path = os.path.join(model_dir, model)
    print(f"\n--- Testing {model} ---")

    try:
        session = ort.InferenceSession(model_path, providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
        print("✔ Loaded successfully")

        print("Inputs:")
        for inp in session.get_inputs():
            print(f"  {inp.name} - shape: {inp.shape}, type: {inp.type}")

        print("Outputs:")
        for out in session.get_outputs():
            print(f"  {out.name} - shape: {out.shape}, type: {out.type}")

    except Exception as e:
        print("✖ Failed to load:", e)
