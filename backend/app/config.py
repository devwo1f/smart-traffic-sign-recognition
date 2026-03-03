"""
Application Configuration

Environment-based settings using pydantic-settings.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/traffic_signs"

    # Model paths
    MODEL_PATH: str = "../ml/models"
    ONNX_MODEL_PATH: str = "../ml/models/classifier.onnx"
    YOLO_MODEL_PATH: str = "../ml/models/yolo_detector.onnx"
    CLASS_MAP_PATH: str = "../ml/data/processed/class_map.json"

    # Inference
    CONFIDENCE_THRESHOLD: float = 0.5
    MAX_BATCH_SIZE: int = 32

    # Upload
    MAX_UPLOAD_SIZE_MB: int = 50
    UPLOAD_DIR: str = "uploads"

    # Retraining
    ML_DIR: str = "../ml"
    NEW_LABELS_DIR: str = "../ml/data/new_labels"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Ensure upload directory exists
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
