"""
MTSD Dataset Download & Extraction

The Mapillary Traffic Sign Dataset requires manual download from:
https://www.mapillary.com/dataset/trafficsign

After downloading, place the fully annotated zip files in data/raw/ and run this script.
"""

import hashlib
import sys
import zipfile
from pathlib import Path

from tqdm import tqdm

import config


EXPECTED_FILES = [
    "mtsd_fully_annotated_annotation.zip",
    "mtsd_fully_annotated_images.train.0.zip",
    "mtsd_fully_annotated_images.train.1.zip",
    "mtsd_fully_annotated_images.train.2.zip",
    "mtsd_fully_annotated_images.val.zip",
    "mtsd_fully_annotated_images.test.zip",
]


def check_files_exist() -> bool:
    """Check if all required zip files are present in data/raw/."""
    missing = []
    for fname in EXPECTED_FILES:
        if not (config.RAW_DIR / fname).exists():
            missing.append(fname)

    if missing:
        print("❌ Missing files in data/raw/:")
        for f in missing:
            print(f"   - {f}")
        print()
        print("📥 Please download the MTSD fully annotated dataset from:")
        print("   https://www.mapillary.com/dataset/trafficsign")
        print()
        print(f"   Place all zip files in: {config.RAW_DIR.resolve()}")
        return False

    print("✅ All required zip files found.")
    return True


def verify_checksums() -> bool:
    """Verify MD5 checksums if checksum file is present."""
    checksum_file = config.RAW_DIR / "mtsd_fully_annotated_md5_sums.txt"
    if not checksum_file.exists():
        print("⚠️  No checksum file found, skipping verification.")
        return True

    print("🔍 Verifying checksums...")
    checksums = {}
    with open(checksum_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                checksums[parts[1].lstrip("*")] = parts[0]

    all_valid = True
    for fname, expected_hash in checksums.items():
        fpath = config.RAW_DIR / fname
        if not fpath.exists():
            continue

        md5 = hashlib.md5()
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)

        actual_hash = md5.hexdigest()
        if actual_hash == expected_hash:
            print(f"   ✅ {fname}")
        else:
            print(f"   ❌ {fname} (hash mismatch)")
            all_valid = False

    return all_valid


def extract_files() -> None:
    """Extract all zip files to data/raw/extracted/."""
    extract_dir = config.RAW_DIR / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    for fname in tqdm(EXPECTED_FILES, desc="Extracting"):
        fpath = config.RAW_DIR / fname
        if not fpath.exists():
            continue

        target_dir = extract_dir
        if fpath.stat().st_size > 0:
            print(f"📦 Extracting {fname}...")
            with zipfile.ZipFile(fpath, "r") as zf:
                zf.extractall(target_dir)
            print(f"   ✅ Extracted to {target_dir}")

    print()
    print("✅ All files extracted successfully!")
    print(f"   Location: {extract_dir.resolve()}")


def organize_dataset() -> None:
    """Organize extracted files into a clean structure."""
    extract_dir = config.RAW_DIR / "extracted"

    # Annotations
    anno_dir = extract_dir / "annotations"
    if not anno_dir.exists():
        # Try to find annotations in the extracted content
        for candidate in extract_dir.rglob("*.json"):
            if "annotation" in candidate.parent.name.lower():
                anno_dir = candidate.parent
                break

    if anno_dir.exists():
        print(f"📋 Annotations found at: {anno_dir}")
    else:
        print("⚠️  Annotations directory not found. Check extracted content.")

    # Count images
    image_count = sum(1 for _ in extract_dir.rglob("*.jpg"))
    print(f"🖼️  Total images found: {image_count}")


def main() -> None:
    print("=" * 60)
    print("  Mapillary Traffic Sign Dataset (MTSD) Setup")
    print("=" * 60)
    print()

    if not check_files_exist():
        sys.exit(1)

    if not verify_checksums():
        print("⚠️  Checksum verification failed. Files may be corrupted.")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != "y":
            sys.exit(1)

    extract_files()
    organize_dataset()

    print()
    print("🎉 Dataset setup complete!")
    print("   Next step: Run 'python preprocess.py' to prepare training data.")


if __name__ == "__main__":
    main()
