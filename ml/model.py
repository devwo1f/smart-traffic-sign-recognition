"""
EfficientNet Model Definition

Transfer learning model using EfficientNet backbone for traffic sign classification.
Supports multiple EfficientNet variants (B0-B3) with configurable fine-tuning.
"""

import json

import torch
import torch.nn as nn
from torchvision import models

import config


class TrafficSignNet(nn.Module):
    """
    EfficientNet-based traffic sign classifier.

    Architecture:
        - EfficientNet backbone (pretrained on ImageNet)
        - Custom classifier head with dropout
        - Supports freeze/unfreeze for staged fine-tuning
    """

    BACKBONE_MAP = {
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
        "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
        "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
        "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
    }

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        if backbone_name not in self.BACKBONE_MAP:
            raise ValueError(
                f"Unknown backbone: {backbone_name}. "
                f"Options: {list(self.BACKBONE_MAP.keys())}"
            )

        # Load pretrained backbone
        model_fn, weights = self.BACKBONE_MAP[backbone_name]
        if pretrained:
            self.backbone = model_fn(weights=weights)
        else:
            self.backbone = model_fn(weights=None)

        # Get the number of features from the backbone classifier
        in_features = self.backbone.classifier[1].in_features

        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze all backbone layers (features extractor)."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        print("🔒 Backbone frozen")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone layers for full fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True
        print("🔓 Backbone unfrozen")

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


def create_model(num_classes: int | None = None) -> TrafficSignNet:
    """Create a TrafficSignNet model using config settings."""
    if num_classes is None:
        if config.CLASS_MAP_FILE.exists():
            with open(config.CLASS_MAP_FILE, "r") as f:
                class_map = json.load(f)
            num_classes = len(class_map)
        else:
            raise ValueError(
                "num_classes not specified and class_map.json not found. "
                "Run preprocessing first."
            )

    model = TrafficSignNet(
        num_classes=num_classes,
        backbone_name=config.CLASSIFIER_BACKBONE,
        pretrained=True,
    )

    print(f"🧠 Created {config.CLASSIFIER_BACKBONE} model:")
    print(f"   Classes:     {num_classes}")
    print(f"   Total params: {model.get_total_params():,}")
    print(f"   Trainable:    {model.get_trainable_params():,}")

    return model


def load_model(checkpoint_path: str | None = None, num_classes: int | None = None) -> TrafficSignNet:
    """Load a model from checkpoint."""
    path = checkpoint_path or str(config.CLASSIFIER_CHECKPOINT)

    # Load checkpoint to inspect it
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    nc = num_classes or checkpoint.get("num_classes", None)
    if nc is None:
        raise ValueError("Cannot determine num_classes from checkpoint")

    model = TrafficSignNet(
        num_classes=nc,
        backbone_name=checkpoint.get("backbone", config.CLASSIFIER_BACKBONE),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✅ Loaded model from {path} (epoch {checkpoint.get('epoch', '?')})")

    return model
