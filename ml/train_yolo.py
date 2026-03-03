"""
YOLOv8 Training Script

Fine-tunes a YOLOv8 model on the Mapillary Traffic Sign Dataset
for traffic sign detection in full images/video frames.
"""

from pathlib import Path

import config


def train_yolo() -> None:
    """Fine-tune YOLOv8 on MTSD for traffic sign detection."""
    # Import here to avoid loading ultralytics for non-YOLO tasks
    from ultralytics import YOLO

    print("=" * 60)
    print("  YOLOv8 Detector Training")
    print("=" * 60)
    print()

    # Verify data.yaml exists
    if not config.YOLO_DATA_YAML.exists():
        print("❌ YOLO data.yaml not found. Run 'python prepare_yolo_data.py' first.")
        return

    # Load pretrained YOLOv8
    print(f"📦 Loading pretrained model: {config.YOLO_MODEL}")
    model = YOLO(config.YOLO_MODEL)

    # Train
    print(f"\n🚀 Starting training...")
    print(f"   Data: {config.YOLO_DATA_YAML}")
    print(f"   Image size: {config.YOLO_IMG_SIZE}")
    print(f"   Batch size: {config.YOLO_BATCH_SIZE}")
    print(f"   Epochs: {config.YOLO_EPOCHS}")
    print()

    results = model.train(
        data=str(config.YOLO_DATA_YAML),
        epochs=config.YOLO_EPOCHS,
        imgsz=config.YOLO_IMG_SIZE,
        batch=config.YOLO_BATCH_SIZE,
        project=str(config.RUNS_DIR),
        name="yolo_training",
        device=0 if config.DEVICE == "cuda" else "cpu",
        workers=config.NUM_WORKERS,
        patience=20,
        save=True,
        verbose=True,
        # Augmentation
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,  # No vertical flip for traffic signs
        fliplr=0.3,
    )

    # Copy best weights
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        import shutil
        shutil.copy2(best_weights, config.YOLO_CHECKPOINT)
        print(f"\n💾 Best weights saved to {config.YOLO_CHECKPOINT}")

    # Validation metrics
    print("\n📊 Validation Results:")
    metrics = model.val()
    print(f"   mAP50:    {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")

    print()
    print("🎉 YOLOv8 training complete!")
    print("   Next step: Run 'python export_onnx.py' to export both models.")


if __name__ == "__main__":
    train_yolo()
