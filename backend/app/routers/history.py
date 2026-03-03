"""
History Router

Paginated, filterable inspection log endpoint.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Prediction
from app.schemas import PredictionHistoryItem, PredictionHistoryResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("", response_model=PredictionHistoryResponse)
async def get_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    label: Optional[str] = Query(None, description="Filter by predicted label"),
    min_confidence: Optional[float] = Query(None, ge=0, le=1, description="Min confidence"),
    max_confidence: Optional[float] = Query(None, ge=0, le=1, description="Max confidence"),
    source_type: Optional[str] = Query(None, description="Filter by source: image, video, webcam"),
    date_from: Optional[datetime] = Query(None, description="Start date (ISO format)"),
    date_to: Optional[datetime] = Query(None, description="End date (ISO format)"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    db: AsyncSession = Depends(get_db),
):
    """
    Fetch paginated prediction history with filters.

    Supports filtering by label, confidence range, source type, and date range.
    Designed for sub-second response with 500+ daily entries via DB indexing.
    """
    # Build base query
    query = select(Prediction)
    count_query = select(func.count(Prediction.id))

    # Apply filters
    if label:
        query = query.where(Prediction.predicted_label.ilike(f"%{label}%"))
        count_query = count_query.where(Prediction.predicted_label.ilike(f"%{label}%"))

    if min_confidence is not None:
        query = query.where(Prediction.confidence >= min_confidence)
        count_query = count_query.where(Prediction.confidence >= min_confidence)

    if max_confidence is not None:
        query = query.where(Prediction.confidence <= max_confidence)
        count_query = count_query.where(Prediction.confidence <= max_confidence)

    if source_type:
        query = query.where(Prediction.source_type == source_type)
        count_query = count_query.where(Prediction.source_type == source_type)

    if date_from:
        query = query.where(Prediction.created_at >= date_from)
        count_query = count_query.where(Prediction.created_at >= date_from)

    if date_to:
        query = query.where(Prediction.created_at <= date_to)
        count_query = count_query.where(Prediction.created_at <= date_to)

    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Sort
    sort_column = getattr(Prediction, sort_by, Prediction.created_at)
    if sort_order == "desc":
        query = query.order_by(sort_column.desc())
    else:
        query = query.order_by(sort_column.asc())

    # Paginate
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)

    # Execute
    result = await db.execute(query)
    predictions = result.scalars().all()

    # Build response
    items = [
        PredictionHistoryItem(
            id=p.id,
            image_filename=p.image_filename,
            predicted_class=p.predicted_class,
            predicted_label=p.predicted_label,
            confidence=p.confidence,
            model_version=p.model_version or "v1.0.0",
            latency_ms=p.latency_ms,
            source_type=p.source_type or "image",
            created_at=p.created_at,
        )
        for p in predictions
    ]

    total_pages = (total + page_size - 1) // page_size

    return PredictionHistoryResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )
