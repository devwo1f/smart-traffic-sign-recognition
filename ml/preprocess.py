"""
MTSD Preprocessing

Parses MTSD JSON annotations, crops individual traffic signs from full images,
filters to top classes, and creates train/val/test splits.
"""

import json
import random
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

import config


def load_annotations(anno_dir: Path) -> list[dict]:
    """Load all MTSD annotation JSON files."""
    annotations = []
    json_files = list(anno_dir.rglob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON annotation files found in {anno_dir}")

    print(f"📋 Loading {len(json_files)} annotation files...")

    for json_file in tqdm(json_files, desc="Loading annotations"):
        with open(json_file, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    annotations.append(data)
                elif isinstance(data, list):
                    annotations.extend(data)
            except json.JSONDecodeError:
                print(f"  ⚠️  Skipping invalid JSON: {json_file.name}")

    print(f"   Loaded {len(annotations)} annotations.")
    return annotations


def extract_sign_regions(annotations: list[dict], image_dirs: list[Path]) -> list[dict]:
    """
    Extract individual sign regions from annotations.

    MTSD annotation format (per image):
    {
        "key": "image_key",
        "objects": [
            {
                "label": "regulatory--speed-limit-50--g1",
                "bbox": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2},
                ...
            }
        ]
    }
    """
    # Build image lookup
    image_map: dict[str, Path] = {}
    for img_dir in image_dirs:
        for img_path in img_dir.rglob("*.jpg"):
            image_map[img_path.stem] = img_path

    print(f"🖼️  Found {len(image_map)} images in source directories.")

    sign_regions = []
    for anno in tqdm(annotations, desc="Extracting sign regions"):
        image_key = anno.get("key", anno.get("image_key", ""))
        objects = anno.get("objects", [])

        if not objects or image_key not in image_map:
            continue

        for i, obj in enumerate(objects):
            label = obj.get("label", "unknown")
            bbox = obj.get("bbox", {})

            if not bbox:
                continue

            sign_regions.append({
                "image_path": str(image_map[image_key]),
                "image_key": image_key,
                "sign_index": i,
                "label": label,
                "xmin": bbox.get("xmin", 0),
                "ymin": bbox.get("ymin", 0),
                "xmax": bbox.get("xmax", 0),
                "ymax": bbox.get("ymax", 0),
            })

    print(f"   Found {len(sign_regions)} sign regions.")
    return sign_regions


def filter_classes(sign_regions: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Filter to classes with enough samples and create class mapping."""
    # Count samples per class
    class_counts = Counter(r["label"] for r in sign_regions)

    # Filter classes with minimum samples
    valid_classes = {
        cls for cls, count in class_counts.items()
        if count >= config.MIN_SAMPLES_PER_CLASS
    }

    print(f"\n📊 Class statistics:")
    print(f"   Total unique classes: {len(class_counts)}")
    print(f"   Classes with >= {config.MIN_SAMPLES_PER_CLASS} samples: {len(valid_classes)}")

    # Create class mapping (sorted for reproducibility)
    sorted_classes = sorted(valid_classes)
    class_map = {cls: idx for idx, cls in enumerate(sorted_classes)}

    # Filter regions
    filtered = [r for r in sign_regions if r["label"] in valid_classes]
    print(f"   Regions after filtering: {len(filtered)}")

    # Print top 10 classes
    top_classes = class_counts.most_common(10)
    print("\n   Top 10 classes:")
    for cls, count in top_classes:
        if cls in valid_classes:
            print(f"     [{class_map[cls]:3d}] {cls}: {count}")

    return filtered, class_map


def crop_signs(sign_regions: list[dict], output_dir: Path) -> list[dict]:
    """Crop individual signs from full images and save."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cropped_records = []

    for region in tqdm(sign_regions, desc="Cropping signs"):
        try:
            img = Image.open(region["image_path"])
            w, h = img.size

            # Clamp bbox to image bounds with padding
            pad = 5
            xmin = max(0, int(region["xmin"]) - pad)
            ymin = max(0, int(region["ymin"]) - pad)
            xmax = min(w, int(region["xmax"]) + pad)
            ymax = min(h, int(region["ymax"]) + pad)

            # Skip tiny crops
            if (xmax - xmin) < 10 or (ymax - ymin) < 10:
                continue

            crop = img.crop((xmin, ymin, xmax, ymax))
            crop = crop.resize(
                (config.CLASSIFIER_IMG_SIZE, config.CLASSIFIER_IMG_SIZE),
                Image.LANCZOS,
            )

            # Save crop
            crop_name = f"{region['image_key']}_{region['sign_index']}.jpg"
            label_dir = output_dir / region["label"]
            label_dir.mkdir(parents=True, exist_ok=True)
            crop_path = label_dir / crop_name
            crop.save(crop_path, quality=95)

            cropped_records.append({
                "path": str(crop_path),
                "label": region["label"],
                "source_image": region["image_key"],
            })

        except Exception as e:
            print(f"  ⚠️  Error cropping {region['image_key']}: {e}")
            continue

    print(f"\n   ✅ Cropped {len(cropped_records)} sign images.")
    return cropped_records


def create_splits(
    cropped_records: list[dict], class_map: dict[str, int]
) -> pd.DataFrame:
    """Create stratified train/val/test splits."""
    random.seed(config.RANDOM_SEED)

    # Group by class for stratified split
    by_class: dict[str, list[dict]] = {}
    for record in cropped_records:
        by_class.setdefault(record["label"], []).append(record)

    splits = []
    for cls, records in by_class.items():
        random.shuffle(records)
        n = len(records)
        n_train = int(n * config.TRAIN_SPLIT)
        n_val = int(n * config.VAL_SPLIT)

        for i, record in enumerate(records):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"

            splits.append({
                "path": record["path"],
                "label": record["label"],
                "class_id": class_map[record["label"]],
                "split": split,
            })

    df = pd.DataFrame(splits)

    # Print split statistics
    print(f"\n📊 Split statistics:")
    for split in ["train", "val", "test"]:
        subset = df[df["split"] == split]
        print(f"   {split:5s}: {len(subset):6d} images, {subset['class_id'].nunique()} classes")

    return df


def merge_new_labels(
    manifest_df: pd.DataFrame, class_map: dict[str, int]
) -> pd.DataFrame:
    """Merge new labeled data from data/new_labels/ into train split."""
    new_labels_dir = config.NEW_LABELS_DIR
    if not new_labels_dir.exists():
        return manifest_df

    new_records = []
    for class_dir in new_labels_dir.iterdir():
        if not class_dir.is_dir():
            continue

        label = class_dir.name
        if label not in class_map:
            print(f"  ⚠️  Unknown class in new_labels: {label}, skipping")
            continue

        for img_file in class_dir.rglob("*.jpg"):
            # Copy to cropped dir
            dest = config.CROPPED_DIR / label / img_file.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_file, dest)

            new_records.append({
                "path": str(dest),
                "label": label,
                "class_id": class_map[label],
                "split": "train",  # New data goes to training
            })

    if new_records:
        print(f"\n➕ Merged {len(new_records)} new labeled images into training set.")
        new_df = pd.DataFrame(new_records)
        manifest_df = pd.concat([manifest_df, new_df], ignore_index=True)

    return manifest_df


def main() -> None:
    print("=" * 60)
    print("  MTSD Preprocessing Pipeline")
    print("=" * 60)
    print()

    # Find extracted data
    extract_dir = config.RAW_DIR / "extracted"
    if not extract_dir.exists():
        print("❌ No extracted data found. Run 'python download_dataset.py' first.")
        return

    # Find annotation files
    anno_candidates = [
        extract_dir / "annotations",
        extract_dir / "mtsd_fully_annotated",
        extract_dir,
    ]
    anno_dir = None
    for candidate in anno_candidates:
        jsons = list(candidate.rglob("*.json"))
        if jsons:
            anno_dir = candidate
            break

    if anno_dir is None:
        print("❌ No annotation JSON files found in extracted data.")
        return

    # Find image directories
    image_dirs = [d for d in extract_dir.rglob("images") if d.is_dir()]
    if not image_dirs:
        image_dirs = [extract_dir]

    # Process
    annotations = load_annotations(anno_dir)
    sign_regions = extract_sign_regions(annotations, image_dirs)
    filtered_regions, class_map = filter_classes(sign_regions)

    # Save class map
    with open(config.CLASS_MAP_FILE, "w") as f:
        json.dump(class_map, f, indent=2)
    print(f"\n💾 Class map saved to {config.CLASS_MAP_FILE}")

    # Crop signs
    cropped_records = crop_signs(filtered_regions, config.CROPPED_DIR)

    # Create splits
    manifest_df = create_splits(cropped_records, class_map)

    # Merge any new labels
    manifest_df = merge_new_labels(manifest_df, class_map)

    # Save manifest
    manifest_df.to_csv(config.SPLIT_MANIFEST_FILE, index=False)
    print(f"\n💾 Split manifest saved to {config.SPLIT_MANIFEST_FILE}")

    print(f"\n   Total classes: {len(class_map)}")
    print(f"   Total images:  {len(manifest_df)}")
    print("\n🎉 Preprocessing complete!")
    print("   Next step: Run 'python train.py' to train the classifier.")


if __name__ == "__main__":
    main()
