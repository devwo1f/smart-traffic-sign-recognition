"""
Continuous Retraining Workflow

Merges new labeled data, retrains models, re-exports to ONNX/TensorRT,
and versions the new model artifacts.
"""

import json
import sys
import time
from pathlib import Path

import config
from version_manager import VersionManager


def check_new_data() -> int:
    """Check for new labeled data in new_labels directory."""
    new_dir = config.NEW_LABELS_DIR
    if not new_dir.exists():
        return 0

    count = 0
    for class_dir in new_dir.iterdir():
        if class_dir.is_dir():
            count += sum(1 for f in class_dir.iterdir() if f.suffix in {".jpg", ".jpeg", ".png"})

    return count


def retrain(from_scratch: bool = False) -> dict:
    """
    Full retraining pipeline.

    Steps:
        1. Check for new labeled data
        2. Re-run preprocessing to merge new data
        3. Retrain classifier (from checkpoint or scratch)
        4. Retrain YOLO detector (optional)
        5. Evaluate
        6. Export ONNX
        7. Quantize TensorRT
        8. Version the new model

    Args:
        from_scratch: If True, train from scratch instead of fine-tuning.
    """
    print("=" * 60)
    print("  Continuous Retraining Pipeline")
    print("=" * 60)
    print()

    start_time = time.time()

    # Step 1: Check new data
    new_count = check_new_data()
    print(f"📂 New labeled images found: {new_count}")

    if new_count == 0:
        print("   No new data to merge. You can still retrain with existing data.")
        response = input("   Continue with retraining? (y/n): ").strip().lower()
        if response != "y":
            print("   Aborted.")
            return {}

    # Step 2: Preprocess (merges new data)
    print("\n🔧 Step 1/6: Preprocessing (merging new data)...")
    from preprocess import main as preprocess_main
    preprocess_main()

    # Step 3: Train classifier
    print("\n🏋️ Step 2/6: Training classifier...")
    from train import train

    resume = None if from_scratch else str(config.CLASSIFIER_CHECKPOINT)
    if not from_scratch and not config.CLASSIFIER_CHECKPOINT.exists():
        resume = None
        print("   No existing checkpoint, training from scratch.")

    train_results = train(resume_from=resume)

    # Step 4: Evaluate
    print("\n📊 Step 3/6: Evaluating...")
    from evaluate import evaluate
    eval_results = evaluate()

    # Step 5: Export ONNX
    print("\n📦 Step 4/6: Exporting to ONNX...")
    from export_onnx import export_classifier_to_onnx
    export_classifier_to_onnx()

    # Step 6: TensorRT (optional)
    print("\n⚡ Step 5/6: TensorRT quantization...")
    try:
        from quantize_tensorrt import quantize_classifier
        quantize_classifier()
    except Exception as e:
        print(f"   ⚠️  TensorRT quantization skipped: {e}")

    # Step 7: Version model
    print("\n🏷️ Step 6/6: Versioning model...")
    vm = VersionManager()
    version_info = vm.create_version(
        accuracy=eval_results.get("accuracy", 0),
        notes=f"Retrained with {new_count} new images",
    )

    total_time = time.time() - start_time

    # Clean up new_labels after successful retraining
    if new_count > 0:
        import shutil
        for class_dir in config.NEW_LABELS_DIR.iterdir():
            if class_dir.is_dir():
                shutil.rmtree(class_dir)
        print(f"\n🧹 Cleaned up {new_count} processed images from new_labels/")

    print()
    print("=" * 60)
    print("  Retraining Complete!")
    print("=" * 60)
    print(f"  Version:  {version_info['version']}")
    print(f"  Accuracy: {100 * eval_results.get('accuracy', 0):.2f}%")
    print(f"  Time:     {total_time / 60:.1f} minutes")

    return {
        "version": version_info["version"],
        "accuracy": eval_results.get("accuracy", 0),
        "new_images_merged": new_count,
        "time_seconds": total_time,
    }


if __name__ == "__main__":
    from_scratch = "--from-scratch" in sys.argv
    retrain(from_scratch=from_scratch)
