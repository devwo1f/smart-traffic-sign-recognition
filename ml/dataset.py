"""
PyTorch Dataset with Data Augmentation

Provides a configurable Dataset class for loading cropped traffic sign images
with train-time augmentation and val/test-time transforms.
"""

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config


class TrafficSignDataset(Dataset):
    """
    PyTorch Dataset for traffic sign classification.

    Loads images from the split manifest CSV and applies appropriate transforms.
    """

    def __init__(self, split: str = "train", transform: transforms.Compose | None = None):
        """
        Args:
            split: One of 'train', 'val', 'test'
            transform: Custom transform. If None, uses default for split.
        """
        if not config.SPLIT_MANIFEST_FILE.exists():
            raise FileNotFoundError(
                "Split manifest not found. Run 'python preprocess.py' first."
            )

        self.manifest = pd.read_csv(config.SPLIT_MANIFEST_FILE)
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)

        if len(self.manifest) == 0:
            raise ValueError(f"No samples found for split '{split}'")

        # Load class map
        with open(config.CLASS_MAP_FILE, "r") as f:
            self.class_map: dict[str, int] = json.load(f)

        self.num_classes = len(self.class_map)
        self.transform = transform or self._default_transform(split)

        print(f"📂 Loaded {split} dataset: {len(self)} images, {self.num_classes} classes")

    def _default_transform(self, split: str) -> transforms.Compose:
        """Create default transforms for the given split."""
        if split == "train":
            return get_train_transforms()
        else:
            return get_eval_transforms()

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.manifest.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        label = int(row["class_id"])

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        id_to_name = {v: k for k, v in self.class_map.items()}
        return id_to_name.get(class_id, "unknown")

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights for balanced training."""
        class_counts = self.manifest["class_id"].value_counts().sort_index()
        total = len(self.manifest)
        weights = total / (len(class_counts) * class_counts.values)
        return torch.FloatTensor(weights)


def get_train_transforms() -> transforms.Compose:
    """Data augmentation pipeline for training."""
    aug = config.AUGMENTATION
    return transforms.Compose([
        transforms.Resize((config.CLASSIFIER_IMG_SIZE, config.CLASSIFIER_IMG_SIZE)),
        transforms.RandomRotation(degrees=aug["rotation_degrees"]),
        transforms.RandomAffine(
            degrees=0,
            translate=aug["translate"],
            scale=aug["scale"],
        ),
        transforms.ColorJitter(
            brightness=aug["brightness"],
            contrast=aug["contrast"],
            saturation=aug["saturation"],
            hue=aug["hue"],
        ),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.GaussianBlur(
            kernel_size=aug["gaussian_blur_kernel"],
            sigma=(0.1, 2.0),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        transforms.RandomErasing(p=aug["random_erasing_prob"]),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Standard transforms for validation and testing."""
    return transforms.Compose([
        transforms.Resize((config.CLASSIFIER_IMG_SIZE, config.CLASSIFIER_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_data_loaders(
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train, val, test DataLoaders."""
    bs = batch_size or config.CLASSIFIER_BATCH_SIZE
    nw = num_workers or config.NUM_WORKERS
    persistent = config.PERSISTENT_WORKERS and nw > 0
    prefetch = config.PREFETCH_FACTOR if nw > 0 else None

    train_ds = TrafficSignDataset("train")
    val_ds = TrafficSignDataset("val")
    test_ds = TrafficSignDataset("test")

    loader_kwargs = dict(
        num_workers=nw,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True, drop_last=True, **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=bs, shuffle=False, **loader_kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=bs, shuffle=False, **loader_kwargs,
    )

    return train_loader, val_loader, test_loader
