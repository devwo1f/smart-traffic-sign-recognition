"""
Inference Benchmark

Benchmarks the full detection + classification pipeline latency
across PyTorch, ONNX Runtime, and TensorRT backends.
"""

import json
import time
from dataclasses import dataclass

import numpy as np
import torch

import config


@dataclass
class BenchmarkResult:
    """Benchmark result for a single backend."""

    backend: str
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    fps: float
    num_iters: int

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "avg_ms": round(self.avg_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "fps": round(self.fps, 1),
            "num_iters": self.num_iters,
        }


def benchmark_pytorch() -> BenchmarkResult | None:
    """Benchmark PyTorch inference."""
    if not config.CLASSIFIER_CHECKPOINT.exists():
        print("   ⚠️  No PyTorch checkpoint found, skipping.")
        return None

    from model import load_model

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model = load_model()
    model = model.to(device)
    model.eval()

    dummy = torch.randn(1, 3, config.CLASSIFIER_IMG_SIZE, config.CLASSIFIER_IMG_SIZE).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(config.BENCHMARK_WARMUP_ITERS):
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(config.BENCHMARK_TEST_ITERS):
            start = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    return _compute_stats("PyTorch", latencies)


def benchmark_onnx() -> BenchmarkResult | None:
    """Benchmark ONNX Runtime inference."""
    if not config.CLASSIFIER_ONNX.exists():
        print("   ⚠️  No ONNX model found, skipping.")
        return None

    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(config.CLASSIFIER_ONNX), providers=providers)

    dummy = np.random.randn(1, 3, config.CLASSIFIER_IMG_SIZE, config.CLASSIFIER_IMG_SIZE).astype(
        np.float32
    )
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(config.BENCHMARK_WARMUP_ITERS):
        session.run(None, {input_name: dummy})

    # Benchmark
    latencies = []
    for _ in range(config.BENCHMARK_TEST_ITERS):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - start) * 1000)

    return _compute_stats("ONNX Runtime", latencies)


def benchmark_tensorrt() -> BenchmarkResult | None:
    """Benchmark TensorRT inference."""
    if not config.CLASSIFIER_TRT.exists():
        print("   ⚠️  No TensorRT engine found, skipping.")
        return None

    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
    except ImportError:
        print("   ⚠️  TensorRT/PyCUDA not available, skipping.")
        return None

    # Load engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(str(config.CLASSIFIER_TRT), "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate buffers
    input_shape = (1, 3, config.CLASSIFIER_IMG_SIZE, config.CLASSIFIER_IMG_SIZE)
    input_size = int(np.prod(input_shape) * np.dtype(np.float32).itemsize)
    output_shape = (1, engine.get_tensor_shape(engine.get_tensor_name(1))[1])
    output_size = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)

    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)
    h_input = np.random.randn(*input_shape).astype(np.float32)
    h_output = np.zeros(output_shape, dtype=np.float32)

    stream = cuda.Stream()

    # Set tensor addresses
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    context.set_input_shape(input_name, input_shape)
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # Warmup
    for _ in range(config.BENCHMARK_WARMUP_ITERS):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    # Benchmark
    latencies = []
    for _ in range(config.BENCHMARK_TEST_ITERS):
        start = time.perf_counter()
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    d_input.free()
    d_output.free()

    return _compute_stats("TensorRT", latencies)


def _compute_stats(backend: str, latencies: list[float]) -> BenchmarkResult:
    """Compute statistics from latency measurements."""
    arr = np.array(latencies)
    return BenchmarkResult(
        backend=backend,
        avg_ms=float(np.mean(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        fps=1000.0 / float(np.mean(arr)),
        num_iters=len(latencies),
    )


def main() -> None:
    print("=" * 60)
    print("  Inference Benchmark")
    print("=" * 60)
    print(f"  Warmup: {config.BENCHMARK_WARMUP_ITERS} iterations")
    print(f"  Test:   {config.BENCHMARK_TEST_ITERS} iterations")
    print()

    results = []

    # PyTorch
    print("🔥 Benchmarking PyTorch...")
    pt_result = benchmark_pytorch()
    if pt_result:
        results.append(pt_result)
        print(f"   Avg: {pt_result.avg_ms:.2f}ms | FPS: {pt_result.fps:.1f}")

    # ONNX Runtime
    print("\n⚡ Benchmarking ONNX Runtime...")
    ort_result = benchmark_onnx()
    if ort_result:
        results.append(ort_result)
        print(f"   Avg: {ort_result.avg_ms:.2f}ms | FPS: {ort_result.fps:.1f}")

    # TensorRT
    print("\n🚀 Benchmarking TensorRT...")
    trt_result = benchmark_tensorrt()
    if trt_result:
        results.append(trt_result)
        print(f"   Avg: {trt_result.avg_ms:.2f}ms | FPS: {trt_result.fps:.1f}")

    # Summary
    print()
    print("=" * 60)
    print("  Results Summary")
    print("=" * 60)
    print(f"  {'Backend':<20s} {'Avg (ms)':>10s} {'P50 (ms)':>10s} {'P95 (ms)':>10s} {'FPS':>10s}")
    print("-" * 60)

    for r in results:
        print(f"  {r.backend:<20s} {r.avg_ms:>10.2f} {r.p50_ms:>10.2f} {r.p95_ms:>10.2f} {r.fps:>10.1f}")

    # Speedup calculation
    if pt_result and trt_result:
        speedup = pt_result.avg_ms / trt_result.avg_ms
        print(f"\n  🏎️  TensorRT speedup: {speedup:.1f}x faster than PyTorch")
        target_met = trt_result.fps >= 60
        print(f"  {'✅' if target_met else '❌'} 60 FPS target: {trt_result.fps:.1f} FPS")
    elif pt_result and ort_result:
        speedup = pt_result.avg_ms / ort_result.avg_ms
        print(f"\n  ⚡ ONNX Runtime speedup: {speedup:.1f}x faster than PyTorch")

    # Save results
    results_dict = {r.backend: r.to_dict() for r in results}
    with open(config.BENCHMARK_RESULTS_FILE, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n  💾 Results saved to {config.BENCHMARK_RESULTS_FILE}")


if __name__ == "__main__":
    main()
