"""
Prepare YOLO Format Data

Converts MTSD bounding box annotations to YOLO format for YOLOv8 training.
YOLO format: <class_id> <x_center> <y_center> <width> <height> (all normalized 0-1)
"""

import json
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

import config


def load_class_map() -> dict[str, int]:
    """Load class mapping from preprocessing step."""
    if not config.CLASS_MAP_FILE.exists():
        raise FileNotFoundError(
            "Class map not found. Run 'python preprocess.py' first."
        )
    with open(config.CLASS_MAP_FILE, "r") as f:
        return json.load(f)


def convert_annotations_to_yolo(
    annotations: list[dict],
    class_map: dict[str, int],
    image_dir: Path,
    output_dir: Path,
    split: str,
) -> int:
    """Convert MTSD annotations to YOLO format for a given split."""
    images_out = output_dir / "images" / split
    labels_out = output_dir / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    converted = 0

    for anno in tqdm(annotations, desc=f"Converting {split}"):
        image_key = anno.get("key", anno.get("image_key", ""))
        objects = anno.get("objects", [])

        if not objects:
            continue

        # Find image file
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = image_dir / f"{image_key}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            # Search recursively
            matches = list(image_dir.rglob(f"{image_key}.*"))
            if matches:
                img_path = matches[0]
            else:
                continue

        # Get image dimensions
        try:
            img = Image.open(img_path)
            img_w, img_h = img.size
        except Exception:
            continue

        # Convert bounding boxes to YOLO format
        yolo_lines = []
        for obj in objects:
            label = obj.get("label", "")
            if label not in class_map:
                continue

            bbox = obj.get("bbox", {})
            xmin = float(bbox.get("xmin", 0))
            ymin = float(bbox.get("ymin", 0))
            xmax = float(bbox.get("xmax", 0))
            ymax = float(bbox.get("ymax", 0))

            # Convert to YOLO format (normalized center x, y, width, height)
            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            # Clamp to [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            if width <= 0 or height <= 0:
                continue

            class_id = class_map[label]
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if yolo_lines:
            # Copy image
            dest_img = images_out / img_path.name
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)

            # Write label file
            label_file = labels_out / f"{img_path.stem}.txt"
            with open(label_file, "w") as f:
                f.write("\n".join(yolo_lines))

            converted += 1

    return converted


def create_data_yaml(class_map: dict[str, int], output_dir: Path) -> None:
    """Create YOLO data.yaml configuration file."""
    # Invert class map: id -> name
    id_to_name = {v: k for k, v in class_map.items()}
    names = [id_to_name[i] for i in range(len(id_to_name))]

    data_config = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    print(f"\n💾 data.yaml saved to {yaml_path}")
    print(f"   Classes: {len(names)}")


def main() -> None:
    print("=" * 60)
    print("  Prepare YOLO Format Data")
    print("=" * 60)
    print()

    class_map = load_class_map()
    print(f"📋 Loaded {len(class_map)} classes from class map.")

    extract_dir = config.RAW_DIR / "extracted"
    if not extract_dir.exists():
        print("❌ Extracted data not found. Run download_dataset.py first.")
        return

    # Load all annotation JSONs
    all_annotations = []
    for json_file in extract_dir.rglob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and "objects" in data:
                    all_annotations.append(data)
                elif isinstance(data, list):
                    all_annotations.extend(
                        [d for d in data if isinstance(d, dict) and "objects" in d]
                    )
        except (json.JSONDecodeError, KeyError):
            continue

    print(f"   Loaded {len(all_annotations)} annotated images.")

    # Shuffle and split
    random.seed(config.RANDOM_SEED)
    random.shuffle(all_annotations)

    n = len(all_annotations)
    n_train = int(n * config.TRAIN_SPLIT)
    n_val = int(n * config.VAL_SPLIT)

    splits = {
        "train": all_annotations[:n_train],
        "val": all_annotations[n_train : n_train + n_val],
        "test": all_annotations[n_train + n_val :],
    }

    # Find image directories
    image_dirs = list(extract_dir.rglob("images"))
    image_dir = image_dirs[0] if image_dirs else extract_dir

    # Convert each split
    output_dir = config.YOLO_DIR
    total = 0
    for split_name, split_annos in splits.items():
        count = convert_annotations_to_yolo(
            split_annos, class_map, image_dir, output_dir, split_name
        )
        total += count
        print(f"   {split_name}: {count} images with labels")

    # Create data.yaml
    create_data_yaml(class_map, output_dir)

    print(f"\n🎉 YOLO data preparation complete! ({total} total images)")
    print("   Next step: Run 'python train_yolo.py' to train the detector.")


if __name__ == "__main__":
    main()
