"""
ML Pipeline Configuration
Central configuration for all hyperparameters, paths, and constants.
"""

import os
from pathlib import Path


# ============================================================================
# Paths
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CROPPED_DIR = DATA_DIR / "cropped"
YOLO_DIR = DATA_DIR / "yolo"
NEW_LABELS_DIR = DATA_DIR / "new_labels"
MODELS_DIR = BASE_DIR / "models"
RUNS_DIR = BASE_DIR / "runs"

# Create directories
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, CROPPED_DIR, YOLO_DIR, NEW_LABELS_DIR, MODELS_DIR, RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Dataset Configuration
# ============================================================================
NUM_WORKERS = min(2, os.cpu_count() or 1)  # Reduced to ease GPU memory pressure on Windows
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
MIN_SAMPLES_PER_CLASS = 20  # Minimum samples to include a class
RANDOM_SEED = 42

# Class label mapping file
CLASS_MAP_FILE = PROCESSED_DIR / "class_map.json"
SPLIT_MANIFEST_FILE = PROCESSED_DIR / "split_manifest.csv"


# ============================================================================
# EfficientNet Classifier Configuration
# ============================================================================
CLASSIFIER_IMG_SIZE = 224
CLASSIFIER_BATCH_SIZE = 64  # Max safe batch size for B3 on RTX 4060 8GB
CLASSIFIER_LR = 1e-3  # Scaled with batch size (linear scaling rule)
CLASSIFIER_WEIGHT_DECAY = 1e-4
CLASSIFIER_EPOCHS = 50
CLASSIFIER_PATIENCE = 10  # Early stopping patience
CLASSIFIER_BACKBONE = "efficientnet_b3"  # Higher accuracy, needs careful VRAM management
CLASSIFIER_FREEZE_EPOCHS = 5  # Freeze backbone for first N epochs
NUM_CLASSES = None  # Determined dynamically from dataset

# Classifier model paths
CLASSIFIER_CHECKPOINT = MODELS_DIR / "classifier_best.pth"
CLASSIFIER_ONNX = MODELS_DIR / "classifier.onnx"
CLASSIFIER_TRT = MODELS_DIR / "classifier.engine"


# ============================================================================
# YOLOv8 Detector Configuration
# ============================================================================
YOLO_IMG_SIZE = 640
YOLO_BATCH_SIZE = 16
YOLO_EPOCHS = 100
YOLO_MODEL = "yolov8n.pt"  # Options: yolov8n.pt, yolov8s.pt
YOLO_DATA_YAML = YOLO_DIR / "data.yaml"

# YOLO model paths
YOLO_CHECKPOINT = MODELS_DIR / "yolo_best.pt"
YOLO_ONNX = MODELS_DIR / "yolo_detector.onnx"
YOLO_TRT = MODELS_DIR / "yolo_detector.engine"


# ============================================================================
# Augmentation Parameters
# ============================================================================
AUGMENTATION = {
    "rotation_degrees": 15,
    "translate": (0.1, 0.1),
    "scale": (0.9, 1.1),
    "brightness": 0.3,
    "contrast": 0.3,
    "saturation": 0.3,
    "hue": 0.1,
    "gaussian_blur_kernel": 3,
    "random_erasing_prob": 0.1,
}

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================================
# Training Configuration
# ============================================================================
USE_AMP = True  # Mixed precision training (FP16 on CUDA)
DEVICE = "cuda"  # Will fallback to cpu if not available
GRADIENT_CLIP_VALUE = 1.0
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch = 64 * 2 = 128 without extra VRAM
USE_COMPILE = False  # torch.compile requires Triton (Linux-only)
CUDNN_BENCHMARK = True  # Re-enabled — small VRAM cost but significant speedup for fixed input sizes
PERSISTENT_WORKERS = False  # Disabled — can cause hangs on Windows
PREFETCH_FACTOR = 2  # Default prefetch


# ============================================================================
# Model Versioning
# ============================================================================
VERSIONS_FILE = MODELS_DIR / "versions.json"


# ============================================================================
# Benchmark Configuration
# ============================================================================
BENCHMARK_WARMUP_ITERS = 50
BENCHMARK_TEST_ITERS = 500
BENCHMARK_RESULTS_FILE = BASE_DIR / "benchmark_results.json"


# ============================================================================
# Sign Class Names (MTSD top classes, populated by preprocessing)
# ============================================================================
SIGN_CLASSES: list[str] = []  # Populated at runtime from class_map.json
