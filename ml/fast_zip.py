"""
Memory-safe zip utility for the YOLO dataset.

Uses ZIP_STORED (no compression) to minimize RAM usage and CPU load.
Files are written one at a time with no buffering to prevent RAM hogging.
"""
import gc
import os
import sys
import time
import zipfile
from pathlib import Path

DATA_DIR = Path("data/yolo")
OUT_PATH = Path("yolo_dataset.zip")


def zip_dataset():
    print(f"📦 Zipping {DATA_DIR} -> {OUT_PATH}")
    print(f"   Mode: ZIP_STORED (no compression, minimal RAM)")
    print()

    # Collect file paths only (not file contents) to minimize memory
    all_files = []
    total_size = 0
    for root, _dirs, files in os.walk(DATA_DIR):
        for f in files:
            fpath = os.path.join(root, f)
            fsize = os.path.getsize(fpath)
            all_files.append((fpath, fsize))
            total_size += fsize

    total_gb = total_size / (1024 ** 3)
    print(f"   {len(all_files)} files ({total_gb:.2f} GB)")
    print(f"   Starting zip...\n")

    start_time = time.time()
    written_bytes = 0

    # Open zip with ZIP_STORED = no compression = no extra RAM needed
    # allowZip64=True for files > 4GB total
    with zipfile.ZipFile(OUT_PATH, 'w', zipfile.ZIP_STORED, allowZip64=True) as zf:
        for i, (fpath, fsize) in enumerate(all_files):
            # Write one file at a time (zipfile handles streaming internally)
            zf.write(fpath, fpath)
            written_bytes += fsize

            # Progress every 2000 files
            if (i + 1) % 2000 == 0:
                pct = (i + 1) / len(all_files) * 100
                written_gb = written_bytes / (1024 ** 3)
                elapsed = time.time() - start_time
                rate = written_bytes / elapsed / (1024 ** 2) if elapsed > 0 else 0
                eta_s = (total_size - written_bytes) / (written_bytes / elapsed) if written_bytes > 0 else 0
                eta_m = eta_s / 60

                print(f"   {i+1:>6}/{len(all_files)} ({pct:5.1f}%) | "
                      f"{written_gb:.2f}/{total_gb:.2f} GB | "
                      f"{rate:.0f} MB/s | "
                      f"ETA: {eta_m:.1f} min")

                # Force garbage collection periodically to keep RAM low
                gc.collect()

    elapsed = time.time() - start_time
    zip_size_gb = OUT_PATH.stat().st_size / (1024 ** 3)
    print(f"\n✅ Done in {elapsed / 60:.1f} minutes!")
    print(f"   Output: {OUT_PATH} ({zip_size_gb:.2f} GB)")
    print(f"\n📤 Next: Upload this file to Google Drive at:")
    print(f"   My Drive/traffic-sign-training/yolo_dataset.zip")


if __name__ == "__main__":
    zip_dataset()
