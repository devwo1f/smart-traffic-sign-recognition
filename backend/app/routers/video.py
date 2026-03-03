"""
Video Router

Endpoints for video upload processing and WebSocket real-time streaming.
"""

import json
import logging
import os
import tempfile
import uuid

from fastapi import APIRouter, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.schemas import VideoProcessingResponse, VideoFrameResult, DetectionResult
from app.video_processor import VideoProcessor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict/video", response_model=VideoProcessingResponse)
async def process_video(
    request: Request,
    file: UploadFile = File(..., description="Video file for analysis"),
    frame_skip: int = 3,
    max_frames: int = 300,
):
    """
    Upload a video for frame-by-frame traffic sign detection.

    Args:
        file: Video file (MP4, AVI, MOV, etc.)
        frame_skip: Process every Nth frame (default: 3)
        max_frames: Maximum frames to process (default: 300)

    Returns:
        Detections per processed frame with metadata.
    """
    engine = request.app.state.inference_engine
    processor = VideoProcessor(engine)

    # Save uploaded video to temp file
    suffix = os.path.splitext(file.filename or ".mp4")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        results = processor.process_video_file(
            tmp_path,
            max_frames=max_frames,
            frame_skip=frame_skip,
        )

        # Convert to response schema
        frame_results = [
            VideoFrameResult(
                frame_number=fr["frame_number"],
                timestamp_ms=fr["timestamp_ms"],
                detections=[DetectionResult(**d) for d in fr["detections"]],
            )
            for fr in results["frame_results"]
        ]

        return VideoProcessingResponse(
            filename=file.filename or "unknown",
            total_frames=results["total_frames"],
            processed_frames=results["processed_frames"],
            fps=results["fps"],
            frame_results=frame_results,
            total_processing_time_ms=results["total_processing_time_ms"],
        )

    finally:
        os.unlink(tmp_path)


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time webcam stream processing.

    Protocol:
        Client sends: JPEG-encoded frame bytes
        Server responds: JSON with detections for each frame

    Example client usage:
        ws = new WebSocket('ws://localhost:8000/ws/stream')
        // Send camera frame as binary
        ws.send(frameBlob)
        // Receive JSON detections
        ws.onmessage = (event) => { detections = JSON.parse(event.data) }
    """
    await websocket.accept()
    logger.info("WebSocket client connected for streaming")

    engine = websocket.app.state.inference_engine
    processor = VideoProcessor(engine)

    try:
        while True:
            # Receive frame bytes
            frame_bytes = await websocket.receive_bytes()

            # Process frame
            result = processor.process_frame_bytes(frame_bytes)

            # Send back detections as JSON
            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except RuntimeError:
            pass
