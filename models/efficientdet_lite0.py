import cv2
import numpy as np
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    ObjectDetector, ObjectDetectorOptions, RunningMode
)

from models.base import BaseDetector

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
MODEL_PATH = WEIGHTS_DIR / "efficientdet_lite0.tflite"


class EfficientDetLite0Detector(BaseDetector):
    """EfficientDet-Lite0 via MediaPipe Tasks — 320x320, INT8 quantized."""

    name = "efficientdet-lite0"

    def __init__(self, confidence: float = 0.5):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"EfficientDet-Lite0 weights not found.\n"
                f"Run: python download_weights.py efficientdet-lite0\n"
                f"Expected: {MODEL_PATH}"
            )
        self.confidence = confidence
        options = ObjectDetectorOptions(
            base_options=BaseOptions(
                model_asset_path=str(MODEL_PATH),
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=RunningMode.IMAGE,
            max_results=20,
            score_threshold=confidence,
            category_allowlist=["person"],
        )
        self.detector = ObjectDetector.create_from_options(options)

        # Warmup inference to avoid cold-start lag on first real frame
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.detect(dummy)

    def detect(self, frame: np.ndarray) -> list[dict]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        detections = []
        for det in result.detections:
            bbox = det.bounding_box
            x1 = max(0, bbox.origin_x)
            y1 = max(0, bbox.origin_y)
            x2 = min(w, bbox.origin_x + bbox.width)
            y2 = min(h, bbox.origin_y + bbox.height)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            conf = det.categories[0].score

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "confidence": float(conf)
            })

        return detections
