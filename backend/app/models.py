"""
Database Models

SQLAlchemy ORM models for predictions and model versions.
"""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Index
from sqlalchemy.sql import func

from app.database import Base


class Prediction(Base):
    """Stores individual prediction results."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_filename = Column(String(255), nullable=False)
    predicted_class = Column(Integer, nullable=False)
    predicted_label = Column(String(255), nullable=False)
    confidence = Column(Float, nullable=False)
    model_version = Column(String(20), default="v1.0.0")
    latency_ms = Column(Float, nullable=True)
    source_type = Column(String(20), default="image")  # image, video, webcam
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Indexes for fast filtering
    __table_args__ = (
        Index("ix_predictions_created_at", "created_at"),
        Index("ix_predictions_predicted_label", "predicted_label"),
        Index("ix_predictions_confidence", "confidence"),
        Index("ix_predictions_source_type", "source_type"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "image_filename": self.image_filename,
            "predicted_class": self.predicted_class,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "latency_ms": self.latency_ms,
            "source_type": self.source_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ModelVersion(Base):
    """Tracks deployed model versions."""

    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(20), unique=True, nullable=False)
    accuracy = Column(Float, nullable=True)
    backbone = Column(String(50), default="efficientnet_b0")
    onnx_path = Column(String(500), nullable=True)
    tensorrt_path = Column(String(500), nullable=True)
    is_active = Column(Integer, default=0)  # 1 = currently deployed
    notes = Column(String(500), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "version": self.version,
            "accuracy": self.accuracy,
            "backbone": self.backbone,
            "is_active": bool(self.is_active),
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
