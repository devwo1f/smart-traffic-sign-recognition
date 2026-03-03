"""
Prediction Router

Endpoints for single and batch image prediction.
"""

import logging
import time

from fastapi import APIRouter, Depends, File, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Prediction
from app.schemas import DetectionResult, PredictionResponse, BatchPredictionResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("", response_model=PredictionResponse)
async def predict_single(
    request: Request,
    file: UploadFile = File(..., description="Traffic sign image"),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a single image for traffic sign detection and classification.

    Returns detected signs with class labels and confidence scores.
    """
    engine = request.app.state.inference_engine
    image_bytes = await file.read()

    detections, inference_time = engine.predict_image(image_bytes)

    # Log predictions to database
    for det in detections:
        prediction = Prediction(
            image_filename=file.filename or "unknown",
            predicted_class=det["class_id"],
            predicted_label=det["label"],
            confidence=det["confidence"],
            latency_ms=inference_time,
            source_type="image",
        )
        db.add(prediction)

    response = PredictionResponse(
        filename=file.filename or "unknown",
        detections=[DetectionResult(**d) for d in detections],
        inference_time_ms=round(inference_time, 2),
    )

    return response


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: Request,
    files: list[UploadFile] = File(..., description="Multiple traffic sign images"),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload multiple images for batch prediction.

    Returns predictions for each image.
    """
    engine = request.app.state.inference_engine
    results = []
    total_time = 0.0

    for file in files:
        image_bytes = await file.read()
        detections, inference_time = engine.predict_image(image_bytes)
        total_time += inference_time

        # Log to database
        for det in detections:
            prediction = Prediction(
                image_filename=file.filename or "unknown",
                predicted_class=det["class_id"],
                predicted_label=det["label"],
                confidence=det["confidence"],
                latency_ms=inference_time,
                source_type="image",
            )
            db.add(prediction)

        results.append(PredictionResponse(
            filename=file.filename or "unknown",
            detections=[DetectionResult(**d) for d in detections],
            inference_time_ms=round(inference_time, 2),
        ))

    return BatchPredictionResponse(
        results=results,
        total_images=len(files),
        total_inference_time_ms=round(total_time, 2),
    )
