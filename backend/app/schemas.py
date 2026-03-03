"""
Pydantic Schemas

Request and response models for API endpoints.
"""

from datetime import datetime

from pydantic import BaseModel, Field


# ============================================================================
# Prediction Schemas
# ============================================================================

class DetectionResult(BaseModel):
    """Single detected sign in an image."""
    class_id: int
    label: str
    confidence: float = Field(ge=0, le=1)
    bbox: dict[str, float] | None = None  # xmin, ymin, xmax, ymax


class PredictionResponse(BaseModel):
    """Response for single image prediction."""
    filename: str
    detections: list[DetectionResult]
    inference_time_ms: float
    model_version: str = "v1.0.0"


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    results: list[PredictionResponse]
    total_images: int
    total_inference_time_ms: float


# ============================================================================
# Video Schemas
# ============================================================================

class VideoFrameResult(BaseModel):
    """Detections for a single video frame."""
    frame_number: int
    timestamp_ms: float
    detections: list[DetectionResult]


class VideoProcessingResponse(BaseModel):
    """Response for video processing."""
    filename: str
    total_frames: int
    processed_frames: int
    fps: float
    frame_results: list[VideoFrameResult]
    total_processing_time_ms: float


# ============================================================================
# History Schemas
# ============================================================================

class PredictionHistoryItem(BaseModel):
    """Single prediction history record."""
    id: int
    image_filename: str
    predicted_class: int
    predicted_label: str
    confidence: float
    model_version: str
    latency_ms: float | None
    source_type: str
    created_at: datetime


class PredictionHistoryResponse(BaseModel):
    """Paginated prediction history."""
    items: list[PredictionHistoryItem]
    total: int
    page: int
    page_size: int
    total_pages: int


# ============================================================================
# Retraining Schemas
# ============================================================================

class RetrainRequest(BaseModel):
    """Retraining request parameters."""
    from_scratch: bool = False
    notes: str = ""


class RetrainResponse(BaseModel):
    """Retraining job response."""
    job_id: str
    status: str  # queued, running, completed, failed
    message: str


class RetrainStatusResponse(BaseModel):
    """Retraining job status."""
    job_id: str
    status: str
    progress: float | None = None
    result: dict | None = None
