"""
FastAPI Application Entry Point

Main application with CORS, lifespan events, and router mounting.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine, Base
from app.routers import predict, history, retrain, video

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    # Startup
    logger.info("🚀 Starting Traffic Sign Recognition API...")

    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database tables created")

    # Load inference engine
    from app.inference import InferenceEngine
    app.state.inference_engine = InferenceEngine(
        classifier_path=settings.ONNX_MODEL_PATH,
        detector_path=settings.YOLO_MODEL_PATH,
        class_map_path=settings.CLASS_MAP_PATH,
    )
    logger.info("✅ Inference engine loaded")

    yield

    # Shutdown
    logger.info("👋 Shutting down...")
    await engine.dispose()


app = FastAPI(
    title="Traffic Sign Recognition API",
    description=(
        "Production-grade traffic sign detection and classification API. "
        "Uses YOLOv8 for detection and EfficientNet for classification."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(video.router, tags=["Video"])
app.include_router(history.router, prefix="/history", tags=["History"])
app.include_router(retrain.router, prefix="/retrain", tags=["Retraining"])


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "service": "Traffic Sign Recognition API",
        "version": "1.0.0",
        "status": "healthy",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    has_engine = hasattr(app.state, "inference_engine")
    return {
        "status": "healthy" if has_engine else "degraded",
        "inference_engine": "loaded" if has_engine else "not loaded",
        "database": "connected",
    }
