"""
ONNX Export

Exports both the EfficientNet classifier and YOLOv8 detector to ONNX format
with dynamic batch axes and validation.
"""

import numpy as np
import onnx
import onnxruntime as ort
import torch

import config
from model import load_model


def export_classifier_to_onnx() -> str:
    """Export EfficientNet classifier to ONNX format."""
    print("📦 Exporting EfficientNet classifier to ONNX...")

    # Load model
    model = load_model()
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, config.CLASSIFIER_IMG_SIZE, config.CLASSIFIER_IMG_SIZE)

    # Export
    onnx_path = str(config.CLASSIFIER_ONNX)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,  # Use legacy exporter to embed weights inline (TensorRT compatible)
    )

    # Validate ONNX model
    print("   Validating ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Test with ONNX Runtime
    session = ort.InferenceSession(onnx_path)
    ort_input = {"input": dummy_input.numpy()}
    ort_output = session.run(None, ort_input)

    # Compare with PyTorch output
    with torch.no_grad():
        pt_output = model(dummy_input).numpy()

    max_diff = np.max(np.abs(pt_output - ort_output[0]))
    print(f"   Max difference: {max_diff:.6f}")
    assert max_diff < 1e-4, f"ONNX validation failed: max diff = {max_diff}"

    print(f"   ✅ Classifier exported to {onnx_path}")
    return onnx_path


def export_yolo_to_onnx() -> str | None:
    """Export YOLOv8 detector to ONNX format."""
    if not config.YOLO_CHECKPOINT.exists():
        print("⚠️  YOLOv8 checkpoint not found. Skipping YOLO ONNX export.")
        return None

    print("\n📦 Exporting YOLOv8 detector to ONNX...")

    from ultralytics import YOLO

    model = YOLO(str(config.YOLO_CHECKPOINT))
    onnx_path = str(config.YOLO_ONNX)

    model.export(
        format="onnx",
        imgsz=config.YOLO_IMG_SIZE,
        dynamic=True,
        simplify=True,
        opset=17,
    )

    # Ultralytics exports to same dir as weights, move if needed
    exported = config.YOLO_CHECKPOINT.with_suffix(".onnx")
    if exported.exists() and str(exported) != onnx_path:
        import shutil
        shutil.move(str(exported), onnx_path)

    print(f"   ✅ YOLOv8 exported to {onnx_path}")
    return onnx_path


def main() -> None:
    print("=" * 60)
    print("  ONNX Model Export")
    print("=" * 60)
    print()

    # Export classifier
    classifier_path = export_classifier_to_onnx()

    # Export detector
    yolo_path = export_yolo_to_onnx()

    print()
    print("🎉 Export complete!")
    print(f"   Classifier: {classifier_path}")
    if yolo_path:
        print(f"   Detector:   {yolo_path}")
    print()
    print("   Next step: Run 'python quantize_tensorrt.py' for TensorRT optimization.")


if __name__ == "__main__":
    main()
