"""
Model Evaluation

Comprehensive evaluation on the test set with metrics:
accuracy, precision, recall, F1 (per-class), confusion matrix.
"""

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

import config
from dataset import get_data_loaders, TrafficSignDataset
from model import load_model

matplotlib.use("Agg")  # Non-interactive backend


@torch.no_grad()
def evaluate() -> dict:
    """Run full evaluation on the test set."""
    print("=" * 60)
    print("  Model Evaluation")
    print("=" * 60)
    print()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # Load model
    print("🧠 Loading best model...")
    model = load_model()
    model = model.to(device)
    model.eval()

    # Load test data
    _, _, test_loader = get_data_loaders()
    test_ds: TrafficSignDataset = test_loader.dataset  # type: ignore

    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []

    print("\n📊 Running inference on test set...")
    for images, labels in tqdm(test_loader, desc="  Evaluating"):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Overall accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Overall Accuracy: {100 * accuracy:.2f}%")

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    # Weighted averages
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    print(f"   Weighted Precision: {100 * precision_w:.2f}%")
    print(f"   Weighted Recall:    {100 * recall_w:.2f}%")
    print(f"   Weighted F1:        {100 * f1_w:.2f}%")

    # Classification report
    class_names = [test_ds.get_class_name(i) for i in range(test_ds.num_classes)]
    report = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True
    )

    # Save classification report
    report_file = config.RUNS_DIR / "classification_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Classification report saved to {report_file}")

    # Print top/bottom performing classes
    print("\n📊 Top 5 classes by F1:")
    class_f1 = [(class_names[i], f1[i], int(support[i])) for i in range(len(f1)) if support[i] > 0]
    class_f1.sort(key=lambda x: x[1], reverse=True)
    for name, score, n in class_f1[:5]:
        print(f"   {name[:50]:50s} F1: {100 * score:.1f}% (n={n})")

    print("\n   Bottom 5 classes by F1:")
    for name, score, n in class_f1[-5:]:
        print(f"   {name[:50]:50s} F1: {100 * score:.1f}% (n={n})")

    # Confusion matrix
    print("\n📊 Generating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    _plot_confusion_matrix(cm, class_names)

    # Results summary
    results = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_weighted": float(f1_w),
        "num_test_samples": int(len(all_labels)),
        "num_classes": int(test_ds.num_classes),
    }

    results_file = config.RUNS_DIR / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Evaluation results saved to {results_file}")

    target_met = accuracy >= 0.95
    print(f"\n{'✅' if target_met else '❌'} Target accuracy (>95%): {100 * accuracy:.2f}%")

    return results


def _plot_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    """Generate and save confusion matrix plot."""
    n_classes = len(class_names)

    # For large number of classes, use a subsample or small plot
    fig_size = max(12, min(30, n_classes * 0.4))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    if n_classes > 30:
        # Normalize for better visibility with many classes
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
        sns.heatmap(cm_norm, cmap="Blues", ax=ax, vmin=0, vmax=1)
        ax.set_title("Confusion Matrix (Normalized)", fontsize=14)
    else:
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=class_names, yticklabels=class_names,
        )
        ax.set_title("Confusion Matrix", fontsize=14)

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.tight_layout()

    cm_path = config.RUNS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved to {cm_path}")


if __name__ == "__main__":
    evaluate()
