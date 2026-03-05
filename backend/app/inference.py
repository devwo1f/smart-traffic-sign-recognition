"""
Inference Engine

Dual-model inference pipeline: YOLOv8 (detection) + EfficientNet (classification).
Uses ONNX Runtime for fast, portable inference.
"""

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class InferenceEngine:
    """
    Dual-model inference engine.

    Pipeline:
        1. YOLOv8 detects sign regions in the image
        2. EfficientNet classifies each detected region
        3. Returns list of (class_id, label, confidence, bbox)
    """

    def __init__(
        self,
        classifier_path: str,
        detector_path: str | None = None,
        class_map_path: str | None = None,
        confidence_threshold: float = 0.5,
    ):
        self.confidence_threshold = confidence_threshold
        self.classifier_session = None
        self.detector_session = None
        self.class_map: dict[str, int] = {}
        self.id_to_name: dict[int, str] = {}

        # Load class map
        if class_map_path and Path(class_map_path).exists():
            with open(class_map_path, "r") as f:
                self.class_map = json.load(f)
                self.id_to_name = {v: k for k, v in self.class_map.items()}
            logger.info(f"Loaded {len(self.class_map)} classes from {class_map_path}")

        # Load classifier
        if Path(classifier_path).exists():
            providers = self._get_providers()
            self.classifier_session = ort.InferenceSession(
                classifier_path, providers=providers
            )
            logger.info(f"Classifier loaded from {classifier_path}")
        else:
            logger.warning(f"Classifier not found at {classifier_path}")

        # Load detector
        if detector_path and Path(detector_path).exists():
            providers = self._get_providers()
            self.detector_session = ort.InferenceSession(
                detector_path, providers=providers
            )
            logger.info(f"Detector loaded from {detector_path}")
        else:
            logger.warning("Detector not loaded — will classify full images instead")

    def _get_providers(self) -> list[str]:
        """Get available ONNX Runtime providers."""
        available = ort.get_available_providers()
        preferred = ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        return [p for p in preferred if p in available]

    def predict_image(self, image_bytes: bytes) -> tuple[list[dict], float]:
        """
        Run full detection + classification pipeline on an image.

        Args:
            image_bytes: Raw image bytes.

        Returns:
            (detections, inference_time_ms)
            Each detection: {class_id, label, confidence, bbox}
        """
        start_time = time.perf_counter()

        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return [], 0.0
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections = []

        if self.detector_session and self.classifier_session:
            # Full pipeline: detect → classify
            detections = self._detect_and_classify(image_rgb)
        elif self.classifier_session:
            # Classify full image (no detector available)
            result = self._classify_image(image_rgb)
            if result:
                detections = [result]
        else:
            logger.error("No models loaded")

        inference_time = (time.perf_counter() - start_time) * 1000
        return detections, inference_time

    def predict_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Run prediction on a video frame (BGR numpy array).

        Args:
            frame: BGR numpy array from OpenCV.

        Returns:
            List of detections.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.detector_session and self.classifier_session:
            return self._detect_and_classify(frame_rgb)
        elif self.classifier_session:
            result = self._classify_image(frame_rgb)
            return [result] if result else []
        return []

    def _detect_and_classify(self, image_rgb: np.ndarray) -> list[dict]:
        """Detect signs with YOLO, then classify each crop."""
        h, w = image_rgb.shape[:2]
        detections = []

        # Run YOLO detection
        det_input = self._preprocess_yolo(image_rgb)
        det_outputs = self.detector_session.run(
            None, {self.detector_session.get_inputs()[0].name: det_input}
        )

        # Parse YOLO output (format depends on export)
        bboxes = self._parse_yolo_output(det_outputs, w, h)

        # Classify each detected region
        for bbox in bboxes:
            x1, y1, x2, y2, det_conf = bbox

            # Crop and classify
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))

            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            crop = image_rgb[y1:y2, x1:x2]
            cls_result = self._classify_crop(crop)

            if cls_result and cls_result["confidence"] >= self.confidence_threshold:
                cls_result["bbox"] = {
                    "xmin": float(x1), "ymin": float(y1),
                    "xmax": float(x2), "ymax": float(y2),
                }
                detections.append(cls_result)

        return detections

    def _classify_image(self, image_rgb: np.ndarray) -> dict | None:
        """Classify a full image without detection."""
        return self._classify_crop(image_rgb)

    def _classify_crop(self, crop_rgb: np.ndarray) -> dict | None:
        """Classify a single cropped sign image."""
        if self.classifier_session is None:
            return None

        # Preprocess
        input_tensor = self._preprocess_classifier(crop_rgb)

        # Inference
        input_name = self.classifier_session.get_inputs()[0].name
        outputs = self.classifier_session.run(None, {input_name: input_tensor})

        # Softmax
        logits = outputs[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        if confidence < self.confidence_threshold:
            return None

        label = self.id_to_name.get(class_id, f"class_{class_id}")

        return {
            "class_id": class_id,
            "label": label,
            "confidence": confidence,
        }

    def _preprocess_classifier(self, image_rgb: np.ndarray) -> np.ndarray:
        """Preprocess image for EfficientNet classifier."""
        img = cv2.resize(image_rgb, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        return np.expand_dims(img, axis=0)

    def _preprocess_yolo(self, image_rgb: np.ndarray) -> np.ndarray:
        """Preprocess image for YOLOv8 detector."""
        img = cv2.resize(image_rgb, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        return np.expand_dims(img, axis=0)

    def _parse_yolo_output(
        self, outputs: list[np.ndarray], orig_w: int, orig_h: int
    ) -> list[tuple[float, float, float, float, float]]:
        """
        Parse YOLOv8 ONNX output to bounding boxes.

        YOLOv8 outputs shape: [1, num_detections, 4+num_classes]
        Format: [x_center, y_center, width, height, class_scores...]
        """
        output = outputs[0]

        if len(output.shape) == 3:
            output = output[0]  # Remove batch dim

        # Standard YOLOv8 ONNX output shape: [1, 4 + num_classes, 8400]
        # We want to iterate over the 8400 detections.
        # So we transpose to [8400, 4 + num_classes]
        if output.shape[0] < output.shape[1]:
            output = output.T

        bboxes = []
        scale_x = orig_w / 640.0
        scale_y = orig_h / 640.0

        for det in output:
            # det is a 1D array of size (4 + num_classes)
            # The first 4 elements are the bounding box [x_center, y_center, width, height]
            x_center, y_center, w, h = det[0:4]
            # The rest are class scores
            class_scores = det[4:]

            max_score = float(np.max(class_scores))

            if max_score < self.confidence_threshold:
                continue

            # Convert center format to corner format, scale to original
            x1 = (x_center - w / 2) * scale_x
            y1 = (y_center - h / 2) * scale_y
            x2 = (x_center + w / 2) * scale_x
            y2 = (y_center + h / 2) * scale_y

            bboxes.append((x1, y1, x2, y2, max_score))

        # NMS (simple version)
        bboxes = self._nms(bboxes, iou_threshold=0.5)

        return bboxes

    def _nms(
        self, bboxes: list[tuple], iou_threshold: float = 0.5
    ) -> list[tuple]:
        """Simple Non-Maximum Suppression."""
        if not bboxes:
            return []

        bboxes.sort(key=lambda x: x[4], reverse=True)
        keep = []

        while bboxes:
            best = bboxes.pop(0)
            keep.append(best)
            bboxes = [
                b for b in bboxes
                if self._iou(best[:4], b[:4]) < iou_threshold
            ]

        return keep

    @staticmethod
    def _iou(box1: tuple, box2: tuple) -> float:
        """Compute IoU between two boxes (x1, y1, x2, y2)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0
