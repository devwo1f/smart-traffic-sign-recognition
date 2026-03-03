"""
Video Processor

OpenCV-based video frame processing for traffic sign detection
in uploaded videos and real-time webcam streams.
"""

import logging
import time
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Processes video files and streams for traffic sign detection."""

    def __init__(self, inference_engine):
        """
        Args:
            inference_engine: InferenceEngine instance for predictions.
        """
        self.engine = inference_engine

    def process_video_file(
        self,
        video_path: str,
        max_frames: int = 0,
        frame_skip: int = 1,
    ) -> dict:
        """
        Process a video file and return detections per frame.

        Args:
            video_path: Path to video file.
            max_frames: Max frames to process (0 = all).
            frame_skip: Process every Nth frame.

        Returns:
            Dictionary with frame results and metadata.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Processing video: {total_frames} frames @ {fps:.1f} FPS ({width}x{height})")

        frame_results = []
        frame_count = 0
        processed_count = 0
        start_time = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames > 0 and processed_count >= max_frames:
                break

            if frame_count % frame_skip == 0:
                detections = self.engine.predict_frame(frame)
                timestamp_ms = (frame_count / fps) * 1000 if fps > 0 else 0

                frame_results.append({
                    "frame_number": frame_count,
                    "timestamp_ms": timestamp_ms,
                    "detections": detections,
                })
                processed_count += 1

            frame_count += 1

        cap.release()
        total_time = (time.perf_counter() - start_time) * 1000

        return {
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "fps": fps,
            "frame_results": frame_results,
            "total_processing_time_ms": total_time,
        }

    def process_frame_bytes(self, frame_bytes: bytes) -> dict:
        """
        Process a single frame from WebSocket stream.

        Args:
            frame_bytes: JPEG/PNG encoded frame bytes.

        Returns:
            Detection results for the frame.
        """
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "Invalid frame data", "detections": []}

        start_time = time.perf_counter()
        detections = self.engine.predict_frame(frame)
        inference_time = (time.perf_counter() - start_time) * 1000

        return {
            "detections": detections,
            "inference_time_ms": inference_time,
        }

    def annotate_frame(
        self, frame: np.ndarray, detections: list[dict]
    ) -> np.ndarray:
        """
        Draw detection bounding boxes and labels on a frame.

        Args:
            frame: BGR numpy array.
            detections: List of detection dicts with bbox.

        Returns:
            Annotated frame.
        """
        annotated = frame.copy()

        for det in detections:
            bbox = det.get("bbox")
            if not bbox:
                continue

            x1, y1 = int(bbox["xmin"]), int(bbox["ymin"])
            x2, y2 = int(bbox["xmax"]), int(bbox["ymax"])
            label = det.get("label", "unknown")
            conf = det.get("confidence", 0)

            # Draw box
            color = (0, 255, 0)  # Green
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            text = f"{label} {conf:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(
                annotated,
                (x1, y1 - th - baseline - 4),
                (x1 + tw + 4, y1),
                color,
                -1,
            )

            # Draw text
            cv2.putText(
                annotated, text,
                (x1 + 2, y1 - baseline - 2),
                font, font_scale, (0, 0, 0), thickness,
            )

        return annotated

    def generate_annotated_frames(
        self, video_path: str, frame_skip: int = 1
    ) -> Generator[bytes, None, None]:
        """
        Generator yielding annotated JPEG frames for streaming.

        Args:
            video_path: Path to video file.
            frame_skip: Process every Nth frame.

        Yields:
            JPEG-encoded annotated frames.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                detections = self.engine.predict_frame(frame)
                annotated = self.annotate_frame(frame, detections)
                _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield buffer.tobytes()

            frame_count += 1

        cap.release()
