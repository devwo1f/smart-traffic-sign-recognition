"""
Resume YOLOv8 Training

Run this script to resume training from the last saved epoch 
after pausing or stopping the process.
"""

from pathlib import Path
from ultralytics import YOLO

# The directory where the current training is saving checkpoints
RUNS_DIR = Path("runs")
ACTIVE_RUN = RUNS_DIR / "yolo_training2"
LAST_WEIGHTS = ACTIVE_RUN / "weights" / "last.pt"

def resume_training():
    if not LAST_WEIGHTS.exists():
        print(f"❌ Cannot find checkpoint at: {LAST_WEIGHTS}")
        print("Training may not have completed its first epoch yet.")
        return

    print("=" * 60)
    print("  Resuming YOLOv8 Training")
    print("=" * 60)
    print(f"\n🔄 Resuming from: {LAST_WEIGHTS}")
    
    # Load the model from the last checkpoint
    model = YOLO(str(LAST_WEIGHTS))
    
    # Resume training
    model.train(resume=True)

if __name__ == "__main__":
    resume_training()
