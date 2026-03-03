"""
TensorRT Quantization

Converts ONNX models to TensorRT engines with FP16/INT8 quantization
for optimized inference. Requires NVIDIA GPU with TensorRT installed.
"""

import sys
import time
from pathlib import Path

import numpy as np

import config


def check_tensorrt_available() -> bool:
    """Check if TensorRT is available."""
    try:
        import tensorrt as trt  # noqa: F401
        return True
    except ImportError:
        return False


def build_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp16",
    max_batch_size: int = 32,
    workspace_gb: int = 2,
) -> str | None:
    """
    Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path: Path to ONNX model.
        engine_path: Output path for TensorRT engine.
        precision: 'fp32', 'fp16', or 'int8'.
        max_batch_size: Maximum batch size for dynamic shapes.
        workspace_gb: GPU workspace in GB.
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("❌ TensorRT not installed. Install with: pip install tensorrt")
        return None

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX model
    print(f"   Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"   Error: {parser.get_error(i)}")
            return None

    # Configure builder
    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if precision == "fp16" and builder.platform_has_fast_fp16:
        build_config.set_flag(trt.BuilderFlag.FP16)
        print("   ⚡ FP16 precision enabled")
    elif precision == "int8" and builder.platform_has_fast_int8:
        build_config.set_flag(trt.BuilderFlag.INT8)
        print("   ⚡ INT8 precision enabled")
    else:
        print(f"   Using FP32 precision (requested: {precision})")

    # Set optimization profiles for dynamic shapes
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape

    # Dynamic batch: min=1, opt=8, max=max_batch_size
    min_shape = (1,) + tuple(input_shape[1:])
    opt_shape = (min(8, max_batch_size),) + tuple(input_shape[1:])
    max_shape = (max_batch_size,) + tuple(input_shape[1:])

    # Replace -1 dims with concrete values
    min_shape = tuple(abs(s) if s > 0 else 1 for s in min_shape)
    opt_shape = tuple(abs(s) if s > 0 else 8 for s in opt_shape)
    max_shape = tuple(abs(s) if s > 0 else max_batch_size for s in max_shape)

    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    build_config.add_optimization_profile(profile)

    # Build engine
    print("   🔨 Building TensorRT engine (this may take several minutes)...")
    start = time.time()
    serialized_engine = builder.build_serialized_network(network, build_config)
    build_time = time.time() - start

    if serialized_engine is None:
        print("   ❌ Engine build failed")
        return None

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    engine_size_mb = Path(engine_path).stat().st_size / (1024 * 1024)
    print(f"   ✅ Engine saved to {engine_path}")
    print(f"      Size: {engine_size_mb:.1f} MB")
    print(f"      Build time: {build_time:.1f}s")

    return engine_path


def quantize_classifier() -> str | None:
    """Quantize EfficientNet classifier."""
    if not config.CLASSIFIER_ONNX.exists():
        print("❌ Classifier ONNX not found. Run 'python export_onnx.py' first.")
        return None

    print("📦 Quantizing EfficientNet classifier...")
    return build_engine(
        str(config.CLASSIFIER_ONNX),
        str(config.CLASSIFIER_TRT),
        precision="fp16",
    )


def quantize_yolo() -> str | None:
    """Quantize YOLOv8 detector."""
    if not config.YOLO_ONNX.exists():
        print("⚠️  YOLOv8 ONNX not found. Skipping.")
        return None

    print("\n📦 Quantizing YOLOv8 detector...")
    return build_engine(
        str(config.YOLO_ONNX),
        str(config.YOLO_TRT),
        precision="fp16",
    )


def main() -> None:
    print("=" * 60)
    print("  TensorRT Quantization")
    print("=" * 60)
    print()

    if not check_tensorrt_available():
        print("❌ TensorRT is not available on this system.")
        print()
        print("   To install TensorRT:")
        print("   1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        print("   2. Install TensorRT: pip install tensorrt")
        print("   3. On Linux: pip install pycuda")
        print()
        print("   Alternatively, use ONNX Runtime for inference (slightly slower).")
        sys.exit(1)

    classifier_engine = quantize_classifier()
    yolo_engine = quantize_yolo()

    print()
    print("🎉 Quantization complete!")
    if classifier_engine:
        print(f"   Classifier: {classifier_engine}")
    if yolo_engine:
        print(f"   Detector:   {yolo_engine}")
    print()
    print("   Next step: Run 'python benchmark.py' to measure inference speed.")


if __name__ == "__main__":
    main()
