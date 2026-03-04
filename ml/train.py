"""
EfficientNet Training Script

Full training loop with mixed precision, early stopping, learning rate scheduling,
and comprehensive metric logging.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import config
from dataset import get_data_loaders, TrafficSignDataset
from model import create_model


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Training", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if config.USE_AMP and device.type == "cuda":
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VALUE)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VALUE)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * correct / total:.1f}%")

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Validating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if config.USE_AMP and device.type == "cuda":
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train(resume_from: str | None = None) -> dict:
    """
    Full training pipeline.

    Args:
        resume_from: Path to checkpoint to resume from.

    Returns:
        Dictionary with training results.
    """
    print("=" * 60)
    print("  EfficientNet Classifier Training")
    print("=" * 60)
    print()

    # Device
    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )
    print(f"🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Enable cuDNN auto-tuner for fixed input sizes
        if config.CUDNN_BENCHMARK:
            torch.backends.cudnn.benchmark = True
            print("   cuDNN benchmark: enabled")

    # Data
    print("\n📂 Loading data...")
    train_loader, val_loader, _ = get_data_loaders()
    train_ds: TrafficSignDataset = train_loader.dataset  # type: ignore
    num_classes = train_ds.num_classes

    # Model
    print("\n🧠 Creating model...")
    model = create_model(num_classes)

    start_epoch = 0
    if resume_from:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"   Resuming from epoch {start_epoch}")

    model = model.to(device)

    # torch.compile for kernel fusion (PyTorch 2.x)
    if config.USE_COMPILE and hasattr(torch, "compile"):
        print("⚡ Compiling model with torch.compile...")
        model = torch.compile(model)

    # Freeze backbone initially
    if start_epoch < config.CLASSIFIER_FREEZE_EPOCHS:
        model.freeze_backbone()

    # Loss, optimizer, scheduler
    class_weights = train_ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.CLASSIFIER_LR,
        weight_decay=config.CLASSIFIER_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.CLASSIFIER_EPOCHS
    )
    scaler = GradScaler(enabled=config.USE_AMP and device.type == "cuda")

    # Training loop
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    print(f"\n🚀 Starting training for {config.CLASSIFIER_EPOCHS} epochs...")
    print(f"   Backbone freeze for first {config.CLASSIFIER_FREEZE_EPOCHS} epochs")
    print()

    start_time = time.time()

    for epoch in range(start_epoch, config.CLASSIFIER_EPOCHS):
        epoch_start = time.time()

        # Unfreeze backbone after freeze period
        if epoch == config.CLASSIFIER_FREEZE_EPOCHS:
            model.unfreeze_backbone()
            # Reset optimizer to include unfrozen params
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.CLASSIFIER_LR * 0.1,  # Lower LR for fine-tuning
                weight_decay=config.CLASSIFIER_WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.CLASSIFIER_EPOCHS - epoch
            )

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Log history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch + 1:3d}/{config.CLASSIFIER_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {100 * train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {100 * val_acc:.1f}% | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "num_classes": num_classes,
                "backbone": config.CLASSIFIER_BACKBONE,
                "class_map_file": str(config.CLASS_MAP_FILE),
            }
            torch.save(checkpoint, config.CLASSIFIER_CHECKPOINT)
            print(f"   💾 Best model saved! (val_acc: {100 * val_acc:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= config.CLASSIFIER_PATIENCE:
                print(f"\n⏹️  Early stopping at epoch {epoch + 1} (patience: {config.CLASSIFIER_PATIENCE})")
                break

    total_time = time.time() - start_time

    # Training summary
    results = {
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "total_epochs": epoch + 1,
        "total_time_seconds": total_time,
        "num_classes": num_classes,
        "backbone": config.CLASSIFIER_BACKBONE,
    }

    print()
    print("=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Best Val Accuracy: {100 * best_val_acc:.2f}%")
    print(f"  Best Val Loss:     {best_val_loss:.4f}")
    print(f"  Total Epochs:      {epoch + 1}")
    print(f"  Total Time:        {total_time / 60:.1f} minutes")
    print(f"  Checkpoint:        {config.CLASSIFIER_CHECKPOINT}")
    print()
    print("  Next step: Run 'python evaluate.py' for full test evaluation.")

    # Save history
    history_file = config.RUNS_DIR / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    return results


if __name__ == "__main__":
    train()
